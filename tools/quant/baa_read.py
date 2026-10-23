#!/usr/bin/env python3
"""读 .baa v1 容器里的张量（反解 XBF4/XBF2/F32/BF16）——引擎正确性隔离用。
用法: python baa_read.py <baa> <tensor_name> → 写 <name>.npy（fp32）
"""
import json, struct, sys
import numpy as np

def load(path):
    f = open(path, "rb")
    h = f.read(128)
    assert h[:4] == b"BAA1"
    tc, doff, moff, mlen, dstart = struct.unpack("<5Q", h[40:80])
    f.seek(doff); dirb = f.read(96 * tc)
    f.seek(doff + 96 * tc); pool = f.read()
    def pname(noff):
        end = pool.index(b"\0", noff)
        return pool[noff:end].decode()
    out = {}
    for i in range(tc):
        e = dirb[96 * i:96 * i + 96]
        noff, sch, dh, ndim, _ = struct.unpack("<IBBBB", e[:8])
        sh = struct.unpack("<4Q", e[8:40])
        do, ps, _ = struct.unpack("<3Q", e[40:64])
        out[pname(noff)] = (sch, sh[:ndim], dstart + do, ps)
    return f, out

def read_tensor(f, ent):
    sch, shape, off, ps = ent
    f.seek(off)
    raw = np.frombuffer(f.read(ps), dtype=np.uint8)
    if sch == 5: return np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(shape)
    if sch == 0:
        u = np.frombuffer(raw.tobytes(), dtype=np.uint16)
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    rows, K = int(np.prod(shape[:-1])), shape[-1]
    if sch == 3:
        pk = raw.reshape(rows, K // 256, 130)
        s = (pk[..., 0].astype(np.uint16) | (pk[..., 1].astype(np.uint16) << 8))
        sf = (s.astype(np.uint32) << 16).view(np.float32)
        nib = pk[..., 2:]
        q = (np.stack([nib & 0xF, nib >> 4], axis=-1).reshape(rows, K // 256, 256)).astype(np.int16) - 8
        return (q * sf[..., None]).reshape(shape)
    if sch == 4:
        pk = raw.reshape(rows, K // 256, 66)
        s = (pk[..., 0].astype(np.uint16) | (pk[..., 1].astype(np.uint16) << 8))
        sf = (s.astype(np.uint32) << 16).view(np.float32)
        b = pk[..., 2:]
        q = (np.stack([(b >> 0) & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3], axis=-1)
             .reshape(rows, K // 256, 256)).astype(np.int16) - 2
        return (q * sf[..., None]).reshape(shape)
    raise ValueError(sch)

if __name__ == "__main__":
    f, idx = load(sys.argv[1])
    w = read_tensor(f, idx[sys.argv[2]])
    np.save(sys.argv[2].replace("/", "_") + ".npy", w)
    print(sys.argv[2], w.shape, w.dtype, "rms=%.4f" % float(np.sqrt((w ** 2).mean())))
