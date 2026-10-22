# xbf3_gemv/xbf3_gemv_v2.py -*- Python -*-
# SPDX-License-Identifier: Apache-2.0
"""XBF3 GEMV v2 — bandwidth version: two-level DMA (shim -> MemTile -> core).

mlir-aie 1.3.4 IRON. Per expert: OUT=2048 x IN=512 XBF3 (401,408B).
E experts concatenated => total wts = E * 401,408B streamed contiguously.

Pipeline:
  of_wts_l2: shim -> memtile, L2_ROWS=256 rows/object (50,176B), depth 2
  of_wts_l1: memtile -> core via forward(obj_type=16 rows) (3,136B), depth 2
  of_act:    shim -> core, fp32[512], resident (acquired once)
  of_out:    core -> shim, fp32[16]/object

--core scalar  : real unpack+MAC (v1 numerics, verify against out_ref)
--core discard : consume-only (measures pure pipe bandwidth; no verify)

Host ABI: [wts, act, out] -> arg slots 3, 4, 5.
"""

import argparse

import numpy as np
import aie.iron as iron
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    jit,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

# ---- shape constants ----
IN = 512
OUT = 2048
SB = 256
SB_BYTES = 98
ROW_BYTES = (IN // SB) * SB_BYTES        # 196
CHUNK_ROWS = 16
L2_ROWS = 256                            # rows per L2 (memtile) object
CHUNK_WTS = CHUNK_ROWS * ROW_BYTES       # 3,136
L2_WTS = L2_ROWS * ROW_BYTES             # 50,176
L1_PER_L2 = L2_ROWS // CHUNK_ROWS        # 16
NUM_CHUNKS = OUT // CHUNK_ROWS           # 128 per expert

act_ty = np.ndarray[(IN,), np.dtype[np.float32]]
l2_wts_ty = np.ndarray[(L2_WTS,), np.dtype[np.uint8]]
chunk_wts_ty = np.ndarray[(CHUNK_WTS,), np.dtype[np.uint8]]
chunk_out_ty = np.ndarray[(CHUNK_ROWS,), np.dtype[np.float32]]


@jit
def xbf3_gemv_v2(wts: In, act: In, out: Out, *, experts: CompileTime[int], core_mode: CompileTime[str]):
    E = experts
    wts_ty = np.ndarray[(E * OUT * ROW_BYTES,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(E * OUT,), np.dtype[np.float32]]

    c_name = "xbf3_gemv_chunk" if core_mode == "scalar" else "xbf3_discard_chunk"
    fn = ExternalFunction(
        c_name,
        source_file="xbf3_gemv.cc",
        arg_types=[chunk_wts_ty, act_ty, chunk_out_ty],
        compile_flags=["-O2"],
    )

    of_wts_l2 = ObjectFifo(l2_wts_ty, name="wts_l2", depth=2)
    of_wts_l1 = of_wts_l2.cons().forward(obj_type=chunk_wts_ty, name="wts_l1")
    of_act = ObjectFifo(act_ty, name="act")
    of_out = ObjectFifo(chunk_out_ty, name="out")

    total_chunks = NUM_CHUNKS * E

    def core_body(of_wts, of_act, of_out, fn):
        x = of_act.acquire(1)
        for _ in range_(total_chunks):
            w = of_wts.acquire(1)
            y = of_out.acquire(1)
            fn(w, x, y)
            of_wts.release(1)
            of_out.release(1)
        of_act.release(1)

    worker = Worker(core_body, fn_args=[of_wts_l1.cons(), of_act.cons(), of_out.prod(), fn])

    rt = Runtime()
    with rt.sequence(wts_ty, act_ty, out_ty) as (w_h, a_h, o_h):
        rt.start(worker)
        rt.fill(of_act.prod(), a_h)
        rt.fill(of_wts_l2.prod(), w_h)
        rt.drain(of_out.cons(), o_h, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="xbf3_gemv_v2")
    add_compile_args(p, default_dev="npu2", with_emit_mlir=True)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--core", choices=["scalar", "discard"], default="scalar")
    return p


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        xbf3_gemv_v2,
        opts,
        compile_kwargs={"experts": opts.experts, "core_mode": opts.core},
        run_and_verify=None,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
