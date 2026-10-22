# xbf3_gemv/xbf3_gemv_v5.py -*- Python -*-
# SPDX-License-Identifier: Apache-2.0
"""XBF3 GEMV v5 — 每列双 MM2S 通道 + 位平面计算（mlir-aie 1.3.4 IRON）。

- 每列 2 条独立 wts fifo（fifo A = [act 块][偶数专家]，fifo B = [奇数专家]），
  占满 shim 的 2 个 MM2S 通道（M0: 8 通道合计 = 60 GB/s）
- 核心计算：xbf3_bp.cc chunk4（mul/mac+srs 位平面提取 + fp32 MAC，M2'-3 验证）
- 输出顺序 = 专家自然序（a/b 交错消费）

Host ABI: [wtsA, wtsB, out] -> arg slots 3, 4, 5。
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
from aie.utils.hostruntime.argparse import add_compile_args
from aie.iron.device import NPU2
from aie.utils.hostruntime.cli import run_design_cli

IN = 512
OUT = 2048
ROW_BYTES = 196


@jit
def xbf3_gemv_v5(wtsA: In, wtsB: In, out: Out, *,
                 experts: CompileTime[int], core_mode: CompileTime[str],
                 ncols: CompileTime[int], chunk_rows: CompileTime[int],
                 l2_rows: CompileTime[int]):
    E, NC = experts, ncols
    rows_per_col = OUT // NC
    CHUNK_WTS = chunk_rows * ROW_BYTES
    L2_WTS = l2_rows * ROW_BYTES
    assert l2_rows % chunk_rows == 0 and OUT % NC == 0 and rows_per_col % chunk_rows == 0
    assert E % 2 == 0
    chunks_per_expert = rows_per_col // chunk_rows          # 4 (128 行块)
    l2_per_expert = L2_WTS // (OUT // NC * ROW_BYTES)       # 1
    assert (rows_per_col * ROW_BYTES) % L2_WTS == 0         # 专家 = 整数个 L2 块
    exp_per_fifo = E // 2

    exp_bytes = rows_per_col * ROW_BYTES                     # 每专家字节数
    slabA = L2_WTS + exp_per_fifo * exp_bytes               # act 块 + 偶数专家
    slabB = exp_per_fifo * exp_bytes
    wtsA_ty = np.ndarray[(NC * slabA,), np.dtype[np.uint8]]
    wtsB_ty = np.ndarray[(NC * slabB,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(NC * E * rows_per_col,), np.dtype[np.float32]]
    chunk_wts_ty = np.ndarray[(CHUNK_WTS,), np.dtype[np.uint8]]
    chunk_out_ty = np.ndarray[(chunk_rows,), np.dtype[np.float32]]
    l2_wts_ty = np.ndarray[(L2_WTS,), np.dtype[np.uint8]]

    c_name = "xbf3_gemv_chunk4" if core_mode == "scalar" else "xbf3_discard_chunk4"
    fn = ExternalFunction(
        c_name, source_file="xbf3_bp.cc",
        arg_types=[chunk_wts_ty, chunk_out_ty],
        compile_flags=["-O2", f"-DCHUNK_ROWS_C={chunk_rows}"],
    )
    ld = ExternalFunction(
        "xbf3_load_act4", source_file="xbf3_bp_load.cc",
        arg_types=[chunk_wts_ty, chunk_wts_ty],
        compile_flags=["-O2", f"-DCHUNK_ROWS_C={chunk_rows}"],
    )

    of_wa, of_wb, of_out = [], [], []
    workers = []
    l1_per_l2 = l2_rows // chunk_rows   # 4

    def make_body():
        def core_body(wa, wb, out, ld, fn):
            w0 = wa.acquire(1)
            ld(w0, w0)                      # 首 L1 块 = 激活相位平面
            wa.release(1)
            for _ in range_(l1_per_l2 - 1):  # act L2 块的填充 L1 块
                wp = wa.acquire(1)
                wa.release(1)
            for _ in range_(exp_per_fifo):
                for _ in range_(chunks_per_expert):
                    w = wa.acquire(1); y = out.acquire(1); fn(w, y)
                    wa.release(1); out.release(1)
                for _ in range_(chunks_per_expert):
                    w = wb.acquire(1); y = out.acquire(1); fn(w, y)
                    wb.release(1); out.release(1)
        return core_body

    for c in range(NC):
        fa = ObjectFifo(l2_wts_ty, name=f"wts_a_{c}", depth=2)
        la = fa.cons().forward(obj_type=chunk_wts_ty, name=f"wts_a1_{c}")
        fb = ObjectFifo(l2_wts_ty, name=f"wts_b_{c}", depth=2)
        lb = fb.cons().forward(obj_type=chunk_wts_ty, name=f"wts_b1_{c}")
        fo = ObjectFifo(chunk_out_ty, name=f"out_{c}")
        of_wa.append(fa); of_wb.append(fb); of_out.append(fo)
        workers.append(Worker(make_body(),
                              fn_args=[la.cons(), lb.cons(), fo.prod(), ld, fn]))

    rt = Runtime()
    from aie.helpers.taplib.tensortiler2d import TensorTiler2D
    tapsA = TensorTiler2D.simple_tiler((1, NC * slabA), (1, slabA))
    tapsB = TensorTiler2D.simple_tiler((1, NC * slabB), (1, slabB))
    tapsO = TensorTiler2D.simple_tiler((1, NC * E * rows_per_col), (1, E * rows_per_col))

    with rt.sequence(wtsA_ty, wtsB_ty, out_ty) as (wa_h, wb_h, o_h):
        rt.start(*workers)
        for c in range(NC):
            rt.fill(of_wa[c].prod(), wa_h, tapsA[c])
            rt.fill(of_wb[c].prod(), wb_h, tapsB[c])
        for c in range(NC):
            rt.drain(of_out[c].cons(), o_h, tapsO[c], wait=(c == NC - 1))

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="xbf3_gemv_v5")
    add_compile_args(p, default_dev="npu2", with_emit_mlir=True)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--core", choices=["scalar", "discard"], default="scalar")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--chunk-rows", type=int, default=64)
    p.add_argument("--l2-rows", type=int, default=256)
    return p


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        xbf3_gemv_v5,
        opts,
        compile_kwargs={"experts": opts.experts, "core_mode": opts.core,
                        "ncols": opts.ncols, "chunk_rows": opts.chunk_rows,
                        "l2_rows": opts.l2_rows},
        run_and_verify=None,
        device=lambda opts: NPU2(),
    )


if __name__ == "__main__":
    main()
