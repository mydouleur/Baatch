"""XBF3 reference quantizer (numpy).

XBF3 layout (per DESIGN.md section 3):
  superblock = 256 consecutive weights along the input (reduction) dim,
  sharing one BF16 scale (0.0625 bpw).
  weights: 3-bit two's complement, levels [-4..3], 8 weights -> 3 bytes.
  total 3.0625 bpw.

Pipeline:
  optional Hadamard rotation on the input dim (orthogonal, fused into the
  preceding RMSNorm/matmul at inference, applied to W offline here),
  then per-superblock optimal scale search (minimize MSE via clip-coefficient
  grid search), then 3-bit packing.
"""

import numpy as np

BLOCK = 256          # superblock size
QMIN, QMAX = -4, 3   # 3-bit two's complement levels


# ---------------------------------------------------------------- Hadamard

def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def fwht(a):
    """In-place fast Walsh-Hadamard transform along the last axis.

    Unnormalized (sum of +/- terms). Divide by sqrt(n) for orthonormal.
    Last-axis length must be a power of two.
    """
    n = a.shape[-1]
    h = 1
    while h < n:
        # vectorized butterfly: reshape into (..., n/(2h), 2, h)
        b = a.reshape(a.shape[:-1] + (n // (2 * h), 2, h))
        u = b[..., 0, :].copy()
        v = b[..., 1, :]
        b[..., 0, :] = u + v
        b[..., 1, :] = u - v
        h *= 2
    return a


def rotation_factors(n):
    """Split n into block-diagonal Hadamard chunk sizes (all powers of two).

    For n = 2^k this is just [n]. Otherwise n is decomposed greedily into
    powers of two (its binary representation), giving a block-diagonal
    orthogonal rotation. All dims in the target model (512/1024/2048) are
    powers of two, so the greedy path is a fallback.
    """
    chunks = []
    rem = n
    while rem:
        p = 1 << (rem.bit_length() - 1)
        chunks.append(p)
        rem -= p
    return chunks


def apply_rotation(w, inverse=False):
    """Apply orthonormal block-diagonal Hadamard rotation along last axis.

    Orthonormal => inverse is the same operation up to ordering of +/-,
    but FWHT unnormalized satisfies H @ H = n*I, and with the 1/sqrt(n)
    scaling applied both times we get exact inverse for pow2 blocks.
    """
    out = np.empty_like(w, dtype=np.float64 if w.dtype == np.float64 else np.float32)
    col = 0
    for c in rotation_factors(w.shape[-1]):
        blk = np.array(w[..., col:col + c], dtype=out.dtype, copy=True)
        fwht(blk)
        blk /= np.sqrt(c)
        out[..., col:col + c] = blk
        col += c
    return out


# ---------------------------------------------------------------- quantize

def _find_scales(blocks, grid_lo=0.30, grid_hi=1.0, grid_n=64):
    """Per-block scale minimizing MSE via clip-coefficient grid search.

    blocks: (..., BLOCK) float array. Returns scales (...) and quantized
    ints (..., BLOCK). scale = clip * max|w| / 4, clip in [grid_lo, grid_hi].
    """
    absmax = np.abs(blocks).max(axis=-1)     # (...,)
    absmax = np.maximum(absmax, 1e-12)
    clips = np.linspace(grid_lo, grid_hi, grid_n)  # (G,)
    # candidate scales: (..., 1, G)
    cand = absmax[..., None] * clips / 4.0   # (..., G)
    b = blocks[..., None]                    # (..., BLOCK, 1)
    q = np.clip(np.round(b / cand[..., None, :]), QMIN, QMAX)  # (..., BLOCK, G)
    err = ((q * cand[..., None, :] - b) ** 2).sum(axis=-2)     # (..., G)
    best = np.argmin(err, axis=-1)           # (...,)
    scale = np.take_along_axis(cand, best[..., None], axis=-1)[..., 0]
    q_best = np.take_along_axis(q, best[..., None, None], axis=-1)[..., 0]
    return scale, q_best.astype(np.int8)


def quantize(w, rotate=False, block=BLOCK, row_chunk=512):
    """Quantize a 2D (out, in) matrix to XBF3.

    Rows are processed in chunks to bound the grid-search memory footprint.
    Returns dict with int8 codes (out, in), scales (out, in//block), and the
    rotation flag. Dequantize with `dequantize`.
    """
    assert w.ndim == 2 and w.shape[1] % block == 0
    out, inn = w.shape
    q = np.empty((out, inn), dtype=np.int8)
    scales = np.empty((out, inn // block), dtype=np.float32)
    for r0 in range(0, out, row_chunk):
        x = w[r0:r0 + row_chunk].astype(np.float32)
        if rotate:
            x = apply_rotation(x)
        s, qc = _find_scales(x.reshape(x.shape[0], inn // block, block))
        q[r0:r0 + row_chunk] = qc.reshape(x.shape[0], inn)
        scales[r0:r0 + row_chunk] = s
    return {"q": q, "scales": scales,
            "rotate": rotate, "block": block, "shape": w.shape}


def dequantize(qq):
    """Inverse of quantize: returns float32 reconstruction."""
    q, s = qq["q"], qq["scales"]
    out, inn = q.shape
    block = qq["block"]
    x = (q.reshape(out, inn // block, block).astype(np.float32)
         * s[..., None]).reshape(out, inn)
    if qq["rotate"]:
        x = apply_rotation(x, inverse=True)
    return x


# ---------------------------------------------------------------- packing

def pack3(q):
    """Pack int8 codes in [-4..3] along the last axis, 8 values -> 3 bytes.

    Bit layout: little-endian 24-bit group per 8 values, each value stored
    as its 3-bit two's complement pattern (v & 7), value i at bits [3i, 3i+3).
    Last-axis length must be a multiple of 8.
    """
    assert q.shape[-1] % 8 == 0
    u = (q.astype(np.int16) & 0x7).astype(np.uint32)
    u = u.reshape(u.shape[:-1] + (u.shape[-1] // 8, 8))
    shifts = (np.arange(8, dtype=np.uint32) * 3)
    word = (u << shifts).sum(axis=-1)  # (..., n/8) uint32 in [0, 2^24)
    b0 = (word & 0xFF).astype(np.uint8)
    b1 = ((word >> 8) & 0xFF).astype(np.uint8)
    b2 = ((word >> 16) & 0xFF).astype(np.uint8)
    return np.stack([b0, b1, b2], axis=-1)  # (..., n/8, 3)


def unpack3(packed, n_out):
    """Inverse of pack3; n_out = number of weights along last axis."""
    w = (packed[..., 0].astype(np.uint32)
         | (packed[..., 1].astype(np.uint32) << 8)
         | (packed[..., 2].astype(np.uint32) << 16))
    shifts = (np.arange(8, dtype=np.uint32) * 3)
    u = ((w[..., None] >> shifts) & 0x7).reshape(packed.shape[:-2] + (-1,))
    u = u[..., :n_out].astype(np.int8)
    # 3-bit two's complement: values >= 4 are negative
    return np.where(u >= 4, u - 8, u).astype(np.int8)


# ---------------------------------------------------------------- metrics

def snr_db(ref, est):
    """10 * log10(||ref||^2 / ||ref - est||^2)."""
    num = float((ref.astype(np.float64) ** 2).sum())
    den = float(((ref.astype(np.float64) - est.astype(np.float64)) ** 2).sum())
    return 10.0 * np.log10(num / max(den, 1e-300))


def per_block_stats(w, qq):
    """Per-superblock relative error: ||err_blk|| / ||w_blk|| (0 if block is 0)."""
    q, s, block = qq["q"], qq["scales"], qq["block"]
    out, inn = q.shape
    x = w.astype(np.float32)
    if qq["rotate"]:
        x = apply_rotation(x)
    xb = x.reshape(out, inn // block, block)
    xdq = (q.reshape(out, inn // block, block).astype(np.float32) * s[..., None])
    num = np.sqrt(((xb - xdq) ** 2).sum(axis=-1))
    den = np.sqrt((xb ** 2).sum(axis=-1))
    rel = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    return rel.ravel()


def self_test():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 512)).astype(np.float32) * 0.02
    # heavy-tail outlier to make scale search non-trivial
    w[3, 7] = 0.5
    for rot in (False, True):
        qq = quantize(w, rotate=rot)
        rec = dequantize(qq)
        # packing roundtrip
        packed = pack3(qq["q"])
        q2 = unpack3(packed, qq["q"].shape[-1])
        assert np.array_equal(q2, qq["q"]), "pack/unpack mismatch"
        print(f"rotate={rot}  SNR={snr_db(w, rec):.2f} dB  pack-OK")
    # rotation orthogonality / inverse check on pow2 and non-pow2 dims
    for n in (256, 512, 768):
        a = rng.standard_normal((8, n)).astype(np.float32)
        r = apply_rotation(a)
        back = apply_rotation(r, inverse=True)
        assert np.allclose(a, back, atol=1e-5), f"rotation inverse failed n={n}"
        assert abs(np.linalg.norm(r[0]) - np.linalg.norm(a[0])) < 1e-4
    print("rotation inverse OK (256, 512, 768)")


if __name__ == "__main__":
    self_test()


# ---------------------------------------------------------------- tile-friendly bit-plane variant (M2'-3)
# Same 3.0 bpw density, lane-friendly for AIE2P v32int8 extract:
# per superblock (256 values): 96B = plane0(32B) | plane1(32B) | plane2(32B)
# plane p, byte b, bit k  <->  value[8b+k] bit p  (value in [0,8), i.e. q & 7)

def pack3_bp(q):
    """Bit-plane pack int8 codes along last axis. (..., 256) -> (..., 96) uint8."""
    assert q.shape[-1] % 256 == 0
    u = (q.astype(np.int16) & 0x7).astype(np.uint8)
    shp0 = u.shape
    u = u.reshape(shp0[:-1] + (-1, 32, 8))          # (..., nsb, 32, 8)
    nsb = u.shape[-3]
    planes = []
    for p in range(3):
        bits = (u >> p) & 1
        planes.append(np.packbits(bits, axis=-1, bitorder="little").squeeze(-1))
    out = np.stack(planes, axis=-2)                 # (..., nsb, 3, 32)
    return out.reshape(shp0[:-1] + (nsb * 96,))


def unpack3_bp(packed, n_out):
    """Inverse of pack3_bp."""
    nsb = packed.shape[-1] // 96
    v = packed.reshape(packed.shape[:-1] + (nsb, 3, 32))  # plane-major within sb
    out = np.zeros(v.shape[:-3] + (nsb, 32, 8), dtype=np.uint8)
    for p in range(3):
        bits = np.unpackbits(v[..., p, :], axis=-1, bitorder="little")
        out |= bits.reshape(out.shape).astype(np.uint8) << p
    out = out.reshape(out.shape[:-3] + (nsb * 256,))[..., :n_out].astype(np.int8)
    return np.where(out >= 4, out - 8, out).astype(np.int8)


def _self_test_bp():
    rng = np.random.default_rng(1)
    q = rng.integers(-4, 4, size=(4, 512)).astype(np.int8)
    p = pack3_bp(q)
    assert p.shape == (4, 192), p.shape
    q2 = unpack3_bp(p, 512)
    assert np.array_equal(q, q2), "bp round-trip mismatch"
    print("pack3_bp round-trip OK")
