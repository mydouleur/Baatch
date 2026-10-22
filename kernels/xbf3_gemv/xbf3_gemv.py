# xbf3_gemv/xbf3_gemv.py -*- Python -*-
# SPDX-License-Identifier: Apache-2.0
"""XBF3 GEMV — single-column single-core streaming design (M2'-1, correctness-first).

mlir-aie 1.3.4 IRON idiom. Matrix: (OUT x IN) = (2048 x 512), one expert
down_proj shape. Weights XBF3 packed row-major (2 superblocks/row, 98B/sb,
196B/row, 401,408B total), streamed DRAM->shim->core in 16-row chunks (3136B).
Activations x[512] fp32 resident on tile. Output y[2048] fp32 in 16-row chunks.

Host ABI (rt.sequence operand order): [wts, act, out] -> arg slots 3, 4, 5
(slot0=opcode, 1=instr, 2=ninstr).
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

# ---- shape constants (v1 fixed; parameterize later) ----
IN = 512          # reduction dim
OUT = 2048        # output rows
SB = 256          # superblock
SB_BYTES = 98     # 2B bf16 scale + 96B packed
ROW_BYTES = (IN // SB) * SB_BYTES          # 196
CHUNK_ROWS = 16
WTS_BYTES = OUT * ROW_BYTES                # 401,408
CHUNK_WTS = CHUNK_ROWS * ROW_BYTES         # 3,136
NUM_CHUNKS = OUT // CHUNK_ROWS             # 128

wts_ty = np.ndarray[(WTS_BYTES,), np.dtype[np.uint8]]
act_ty = np.ndarray[(IN,), np.dtype[np.float32]]
out_ty = np.ndarray[(OUT,), np.dtype[np.float32]]
chunk_wts_ty = np.ndarray[(CHUNK_WTS,), np.dtype[np.uint8]]
chunk_out_ty = np.ndarray[(CHUNK_ROWS,), np.dtype[np.float32]]


@jit
def xbf3_gemv(wts: In, act: In, out: Out):
    fn = ExternalFunction(
        "xbf3_gemv_chunk",
        source_file="xbf3_gemv.cc",
        arg_types=[chunk_wts_ty, act_ty, chunk_out_ty],
        compile_flags=["-O2"],
    )

    of_wts = ObjectFifo(chunk_wts_ty, name="wts")
    of_act = ObjectFifo(act_ty, name="act")
    of_out = ObjectFifo(chunk_out_ty, name="out")

    def core_body(of_wts, of_act, of_out, fn):
        x = of_act.acquire(1)
        for _ in range_(NUM_CHUNKS):
            w = of_wts.acquire(1)
            y = of_out.acquire(1)
            fn(w, x, y)
            of_wts.release(1)
            of_out.release(1)
        of_act.release(1)

    worker = Worker(core_body, fn_args=[of_wts.cons(), of_act.cons(), of_out.prod(), fn])

    rt = Runtime()
    with rt.sequence(wts_ty, act_ty, out_ty) as (w_h, a_h, o_h):
        rt.start(worker)
        rt.fill(of_act.prod(), a_h)
        rt.fill(of_wts.prod(), w_h)
        rt.drain(of_out.cons(), o_h, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="xbf3_gemv")
    add_compile_args(p, default_dev="npu2", with_emit_mlir=True)
    return p


def main():
    import logging, os
    if os.environ.get("XBF3_DEBUG"):
        logging.basicConfig(level=logging.DEBUG)
    opts = _make_argparser().parse_args()
    run_design_cli(
        xbf3_gemv,
        opts,
        compile_kwargs={},
        run_and_verify=None,  # 本机运行走 C++ host（test_host.cpp）
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
