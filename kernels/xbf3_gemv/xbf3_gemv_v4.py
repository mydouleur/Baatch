# xbf3_gemv/xbf3_gemv_v3.py -*- Python -*-
# SPDX-License-Identifier: Apache-2.0
"""XBF3 GEMV v4 — 位平面(XBF3-BP) + 向量化计算（aie_api, fp32 MAC 车道）。

结构与 v3 相同（4 列 shim→memtile→core，act 在列流首块）；kernel 换为
mul/mac+srs 位平面提取 + v32 fp32 MAC（xbf3_bp.cc）。

NCOLS columns in parallel; each column: shim -> memtile -> core -> shim.
Rows are split across columns (col c computes rows [c*OUT/NCOLS, (c+1)*OUT/NCOLS)
of every expert). Weights are pre-swizzled host-side (gen_data.py --cols) so each
column's stream is one contiguous buffer region; outputs likewise arrive
column-major and are remapped by test_host.cpp.

Host ABI: [wts, act, out] -> arg slots 3, 4, 5.
  wts: NCOLS * E * (OUT/NCOLS) * 196B contiguous per column (concatenated)
  act: NCOLS copies of x[512] fp32 (one per column)
  out: NCOLS * E * (OUT/NCOLS) fp32, column-major
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
SB = 256
SB_BYTES = 98
ROW_BYTES = (IN // SB) * SB_BYTES   # 196


@jit
def xbf3_gemv_v3(wts: In, act: In, out: Out, *,
                 experts: CompileTime[int], core_mode: CompileTime[str],
                 ncols: CompileTime[int], chunk_rows: CompileTime[int],
                 l2_rows: CompileTime[int]):
    E, NC = experts, ncols
    rows_per_col = OUT // NC
    CHUNK_WTS = chunk_rows * ROW_BYTES
    L2_WTS = l2_rows * ROW_BYTES
    assert l2_rows % chunk_rows == 0 and OUT % NC == 0 and rows_per_col % chunk_rows == 0
    chunks_per_col = (rows_per_col // chunk_rows) * E

    slab_bytes = L2_WTS + E * rows_per_col * ROW_BYTES   # act 占一个完整 L2 块（对齐）+ 权重
    wts_ty = np.ndarray[(NC * slab_bytes,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(4,), np.dtype[np.float32]]  # 占位，不用
    out_ty = np.ndarray[(NC * E * rows_per_col,), np.dtype[np.float32]]
    chunk_wts_ty = np.ndarray[(CHUNK_WTS,), np.dtype[np.uint8]]
    chunk_out_ty = np.ndarray[(chunk_rows,), np.dtype[np.float32]]
    l2_wts_ty = np.ndarray[(L2_WTS,), np.dtype[np.uint8]]
    wts_col_ty = np.ndarray[(E * rows_per_col * ROW_BYTES,), np.dtype[np.uint8]]
    out_col_ty = np.ndarray[(E * rows_per_col,), np.dtype[np.float32]]

    c_name = "xbf3_gemv_chunk4" if core_mode == "scalar" else "xbf3_discard_chunk4"
    fn = ExternalFunction(
        c_name,
        source_file="xbf3_bp.cc",
        arg_types=[chunk_wts_ty, chunk_out_ty],
        compile_flags=["-O2", f"-DCHUNK_ROWS_C={chunk_rows}"],
    )

    of_wts, of_out = [], []
    workers = []

    def make_body():
        def core_body(of_wts, of_out, ld, fn):
            w0 = of_wts.acquire(1)
            ld(w0, w0)          # 首 L1 块 = 激活（拷进静态 xbuf）
            of_wts.release(1)
            for _ in range_(l2_rows // chunk_rows - 1):  # act L2 块的其余 L1 块是填充
                wp = of_wts.acquire(1)
                of_wts.release(1)
            for _ in range_(chunks_per_col):
                w = of_wts.acquire(1)
                y = of_out.acquire(1)
                fn(w, y)
                of_wts.release(1)
                of_out.release(1)
        return core_body

    ld = ExternalFunction(
        "xbf3_load_act4",
        source_file="xbf3_bp_load.cc",
        arg_types=[chunk_wts_ty, chunk_wts_ty],
        compile_flags=["-O2", f"-DCHUNK_ROWS_C={chunk_rows}"],
    )

    for c in range(NC):
        l2 = ObjectFifo(l2_wts_ty, name=f"wts_l2_{c}", depth=2)
        l1 = l2.cons().forward(obj_type=chunk_wts_ty, name=f"wts_l1_{c}")
        fo = ObjectFifo(chunk_out_ty, name=f"out_{c}")
        of_wts.append(l2)
        of_out.append(fo)
        workers.append(Worker(make_body(), fn_args=[l1.cons(), fo.prod(), ld, fn]))

    rt = Runtime()
    # sequence buffers: [wts, act, out]; per-column fill/drain over contiguous slices.
    # host buffer layout is already column-major-contiguous, so a plain (NC-col split)
    # tap suffices: each column takes one contiguous slab.
    from aie.helpers.taplib.tensortiler2d import TensorTiler2D

    _ = wts_col_ty  # unused
    out_taps = TensorTiler2D.simple_tiler((1, NC * E * rows_per_col), (1, E * rows_per_col))

    slab_taps = TensorTiler2D.simple_tiler((1, NC * slab_bytes), (1, slab_bytes))
    with rt.sequence(wts_ty, act_ty, out_ty) as (w_h, a_h, o_h):
        rt.start(*workers)
        for c in range(NC):
            rt.fill(of_wts[c].prod(), w_h, slab_taps[c])
        for c in range(NC):
            rt.drain(of_out[c].cons(), o_h, out_taps[c], wait=(c == NC - 1))

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="xbf3_gemv_v4")
    add_compile_args(p, default_dev="npu2", with_emit_mlir=True)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--core", choices=["scalar", "discard"], default="scalar")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--chunk-rows", type=int, default=128)
    p.add_argument("--l2-rows", type=int, default=512)
    return p


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        xbf3_gemv_v3,
        opts,
        compile_kwargs={"experts": opts.experts, "core_mode": opts.core,
                        "ncols": opts.ncols, "chunk_rows": opts.chunk_rows,
                        "l2_rows": opts.l2_rows},
        run_and_verify=None,
        device=lambda opts: NPU2(),  # 全 8 列设备（默认 NPU2Col1 放不下 4 列设计）
    )


if __name__ == "__main__":
    main()
