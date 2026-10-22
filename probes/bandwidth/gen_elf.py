#!/usr/bin/env python3
"""Synthesize XDNA ctrlcode ELF (txn buffer) for shim DMA loopback bandwidth probe.
Replicates df_bw_17f0_10.elf structure with configurable column count and buffer span.

Format reference: aie-rt txn binary (dmacompiler/host_runtime/txn.hpp) +
XRT elf_patcher (shim_dma_48 reloc: BD addr words[1],[2] = bo_addr + addend + 0x80000000).

Per column c (register offsets include col<<25):
  locks+SS:   WRITE 0x3300c=1, 0x33000=1, 0x3f008=0x80000000, 0x3f100=0x80000000
  SS:         WRITE 0x3f010=0x80000005, 0x3f114=0x80000000
  DMA cfg+SS: WRITE 0x1f000=0x4400, 0x1f004=0x50, 0x3f014=0x80000009, 0x3f124=0x80000000
  BDs:        BD1@0x1d020 ifm read  @ c*SPAN
              BD3@0x1d060 ofm write @ c*SPAN
              BD2@0x1d040 ifm read  @ c*SPAN + SPAN//2
              BD4@0x1d080 ofm write @ c*SPAN + SPAN//2
  start:      8 WRITEs to 0x1d2xx (queue start + TCT handshake)
  sync:       2 C_TCT ops
"""
import struct, sys, argparse

def w32(col, reg, val):
    # WRITE op, 24 bytes: OpHdr(4) pad(4) RegOff(8) Value(4) Size(4)
    ophdr = bytes([0x00, 0x00, reg & 0xFF, 0x00])
    return ophdr + b"\0\0\0\0" + struct.pack("<QII", (col << 25) | reg, val, 24)

def bdwrite(col, reg, words):
    # BLOCKWRITE op, 16B header + data
    ophdr = bytes([0x01, 0x00, reg & 0xFF, 0x00])
    hdr = ophdr + struct.pack("<BBxxII", 0, 0, (col << 25) | reg, 16 + 4*len(words))
    return hdr + struct.pack(f"<{len(words)}I", *words)

def ddr_patch(col, bd_reg, bufid, offset):
    # C_DDR_PATCH custom op, 48 bytes
    ophdr = bytes([0x81, 0x00, 0x00, 0x00])
    return ophdr + struct.pack("<IQIIIIIIII", 48, 0, 0, 0x1f2, (col << 25) | (bd_reg + 4), 0, bufid, 0, offset, 0)

def tct(col, val):
    ophdr = bytes([0x80, 0x00, 0x00, 0x00])
    return ophdr + struct.pack("<III", 16, 0x03000000 | (col << 16), val)

BD_WORDS = [0x02000000, 0, 0, 0, 0x80000000, 0x02000000, 0, 0x02000000]
BD1, BD2, BD3, BD4 = 0x1d020, 0x1d040, 0x1d060, 0x1d080

def gen_ctrltext(ncols, span):
    """span = per-column buffer span in bytes (each col reads span/2+span/2 via 2 BDs)."""
    body = b""
    reloc = []  # (ctrltext_offset, sym_idx, addend)
    for c in range(ncols):
        for reg, val in [(0x3300c,1),(0x33000,1),(0x3f008,0x80000000),(0x3f100,0x80000000)]:
            body += w32(c, reg, val)
    for c in range(ncols):
        for reg, val in [(0x3f010,0x80000005),(0x3f114,0x80000000)]:
            body += w32(c, reg, val)
    for c in range(ncols):
        for reg, val in [(0x1f000,0x4400),(0x1f004,0x50),(0x3f014,0x80000009),(0x3f124,0x80000000)]:
            body += w32(c, reg, val)
    for c in range(ncols):
        for bd, sym, bufid, off in [(BD1, 1, 0, c*span), (BD3, 2, 2, c*span),
                                    (BD2, 1, 0, c*span + span//2), (BD4, 2, 2, c*span + span//2)]:
            op_off = 16 + len(body)
            body += bdwrite(c, bd, BD_WORDS)
            reloc.append((op_off + 16, sym, off))
            body += ddr_patch(c, bd, bufid, off)
    for c in range(ncols):
        for reg, val in [(0x1d200,0x300),(0x1d204,0x80000003),(0x1d210,0x100),(0x1d214,1),
                         (0x1d208,0x400),(0x1d20c,0x80000004),(0x1d218,0x200),(0x1d21c,2)]:
            body += w32(c, reg, val)
    for c in range(ncols):
        body += tct(c, 0x00010100)
        body += tct(c, 0x01010100)

    numops = ncols*(4+2+4+8) + ncols*8 + ncols*2  # config writes + (bdwrite+patch) pairs + start writes + tct
    hdr = struct.pack("<BBBBBBxxII", 0, 1, 4, 6, ncols, 1, numops, 16 + len(body))
    return hdr + body, reloc

def build_elf(ctrl, reloc, out_path, sym_size=0x08000000):
    shstr = b"\0.shstrtab\0.ctrltext\0.dynstr\0.dynsym\0.rela.dyn\0.dynamic\0.note.xrt.UID\0"
    def name_off(n): return shstr.index(b"\0"+n+b"\0") + 1
    dynstr = b"\0" + b"3\0" + b"5\0"
    dynsym = b"\0"*16
    dynsym += struct.pack("<IIIBBH", 1, 0, sym_size, 0x11, 0, 2)  # sym "3" (ifm, arg3)
    dynsym += struct.pack("<IIIBBH", 3, 0, sym_size, 0x11, 0, 2)  # sym "5" (ofm, arg5)
    rela = b"".join(struct.pack("<III", off, (sym<<8)|5, add) for off, sym, add in reloc)
    dynamic = struct.pack("<II", 7, 5)
    note = bytes.fromhex("040000002000000004000000585254003438616461646535343035303838353931396162646363383164353062616433")

    # layout: ehdr(52)+phdr(3*32)=0x94, pad to 0xa0
    off_ctrl = 0xa0
    off_dyn  = off_ctrl + len(ctrl)
    off_shstr = off_dyn + 16
    off_dynstr = off_shstr + len(shstr)
    # align dynsym to 4
    off_dynsym = (off_dynstr + len(dynstr) + 3) & ~3
    off_rela = off_dynsym + len(dynsym)
    off_note = (off_rela + len(rela) + 15) & ~15
    off_shoff = (off_note + len(note) + 3) & ~3

    ehdr = struct.pack("<16sHHIIIIIHHHHHH",
        b"\x7fELF\x01\x01\x01E\x02" + b"\0"*7,
        2, 1, 1, 0, 0x34, off_shoff, 0, 0x34, 0x20, 3, 0x28, 8, 1)
    phdrs = struct.pack("<IIIIIIII", 6, 0x34, 0, 0, 0x60, 0x60, 4, 0)
    phdrs += struct.pack("<IIIIIIII", 1, off_ctrl, 0, 0, len(ctrl), len(ctrl), 5, 0x10)
    phdrs += struct.pack("<IIIIIIII", 2, off_dyn, 0, 0, 16, 16, 6, 8)

    def shdr(name, stype, flags, addr, off, size, link, info, align, entsize):
        return struct.pack("<IIIIIIIIII", name, stype, flags, addr, off, size, link, info, align, entsize)
    shdrs = b"\0"*40
    shdrs += shdr(name_off(b".shstrtab"), 3, 0, 0, off_shstr, len(shstr), 0, 0, 0, 0)
    shdrs += shdr(name_off(b".ctrltext"), 1, 0x6, 0, off_ctrl, len(ctrl), 0, 0, 0, 0)
    shdrs += shdr(name_off(b".dynstr"), 3, 0, 0, off_dynstr, len(dynstr), 0, 0, 0, 0)
    shdrs += shdr(name_off(b".dynsym"), 11, 0x2, 0, off_dynsym, len(dynsym), 3, 1, 0, 16)
    shdrs += shdr(name_off(b".rela.dyn"), 4, 0x2, 0, off_rela, len(rela), 4, 0, 0, 12)
    shdrs += shdr(name_off(b".dynamic"), 6, 0x2, 0, off_dyn, 16, 3, 0, 0, 8)
    shdrs += shdr(name_off(b".note.xrt.UID"), 7, 0, 0, off_note, len(note), 0, 0, 0, 0)

    buf = bytearray(off_shoff + len(shdrs))
    buf[0:len(ehdr)] = ehdr
    buf[0x34:0x34+len(phdrs)] = phdrs
    buf[off_ctrl:off_ctrl+len(ctrl)] = ctrl
    buf[off_dyn:off_dyn+16] = dynamic
    buf[off_shstr:off_shstr+len(shstr)] = shstr
    buf[off_dynstr:off_dynstr+len(dynstr)] = dynstr
    buf[off_dynsym:off_dynsym+len(dynsym)] = dynsym
    buf[off_rela:off_rela+len(rela)] = rela
    buf[off_note:off_note+len(note)] = note
    buf[off_shoff:off_shoff+len(shdrs)] = shdrs
    open(out_path, "wb").write(bytes(buf))
    print(f"wrote {out_path}: {len(buf)} bytes, ctrltext={len(ctrl)}B, {len(reloc)} relocs")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--span", type=lambda x: int(x,0), default=0x10000000, help="per-col span bytes")
    ap.add_argument("--out", default="df_bw_8col.elf")
    a = ap.parse_args()
    ctrl, reloc = gen_ctrltext(a.cols, a.span)
    build_elf(ctrl, reloc, a.out)
