# dequant_dbg.py — 最小复现设计：1 列 1 核，反量化链分阶段 dump
# in: 4352B（128x64 大瓦片：4096B 权重 + 128B scale + 128B zp）
# out: st1 8192B (u8) / st2 8192B (i8) / st3 16384B (bf16)
# Host ABI: [in, st1, st2, st3] -> arg slots 3,4,5,6
import argparse
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import (CompileTime, ExternalFunction, In, ObjectFifo, Out,
                      Program, Runtime, Worker, jit)
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

in_ty = np.ndarray[(4352,), np.dtype[np.int8]]
st1_ty = np.ndarray[(8192,), np.dtype[np.uint8]]
st2_ty = np.ndarray[(8192,), np.dtype[np.int8]]
st3_ty = np.ndarray[(8192,), np.dtype[bfloat16]]


@jit
def dequant_dbg(inp: In, s1: Out, s2: Out, s3: Out):
    fn = ExternalFunction(
        "dequant_dbg", source_file="dequant_dbg.cc",
        arg_types=[in_ty, st1_ty, st2_ty, st3_ty],
        compile_flags=["-O2"],
    )
    of_in = ObjectFifo(in_ty, name="in")
    of1 = ObjectFifo(st1_ty, name="st1")
    of2 = ObjectFifo(st2_ty, name="st2")
    of3 = ObjectFifo(st3_ty, name="st3")

    def core_body(if_, o1, o2, o3, fn):
        ei = if_.acquire(1)
        a = o1.acquire(1); b = o2.acquire(1); c = o3.acquire(1)
        fn(ei, a, b, c)
        if_.release(1)
        o1.release(1); o2.release(1); o3.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of1.prod(), of2.prod(), of3.prod(), fn])

    rt = Runtime()
    with rt.sequence(in_ty, st1_ty, st2_ty, st3_ty) as (i_h, o1_h, o2_h, o3_h):
        rt.start(worker)
        rt.fill(of_in.prod(), i_h)
        rt.drain(of1.cons(), o1_h, wait=False)
        rt.drain(of2.cons(), o2_h, wait=False)
        rt.drain(of3.cons(), o3_h, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="dequant_dbg")
    add_compile_args(p, default_dev="npu2", with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(dequant_dbg, opts, compile_kwargs={}, run_and_verify=None,
                   device=device_from_args)


if __name__ == "__main__":
    main()
