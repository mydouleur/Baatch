#!/usr/bin/env python3
"""Decode an aie-rt transaction buffer (XDNA txn binary) from a ctrlcode ELF .ctrltext
or raw bin. Format from dmacompiler/host_runtime/txn.hpp (aie-rt XAie_Txn*).
"""
import struct, sys

OPN = {0:"WRITE",1:"BLOCKWRITE",2:"BLOCKSET",3:"MASKWRITE",4:"MASKPOLL",
       5:"SHIMDMA_BD",6:"SHIMDMA_DMABUF_BD",128:"C_TCT",129:"C_DDR_PATCH",130:"C_NEXT"}

def load_ctrltext(path):
    data = open(path,"rb").read()
    if data[:4] == b"\x7fELF":
        e_shoff = struct.unpack_from("<I", data, 32)[0]
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 46)
        secs = []
        for i in range(e_shnum):
            o = e_shoff + i*e_shentsize
            secs.append(struct.unpack_from("<IIIIIIIIII", data, o))
        shstr = secs[e_shstrndx]
        strtab = data[shstr[4]:shstr[4]+shstr[5]]
        for s in secs:
            nm = strtab[s[0]:strtab.index(b"\0", s[0])].decode()
            if nm == ".ctrltext":
                return data[s[4]:s[4]+s[5]]
        raise SystemExit("no .ctrltext")
    return data

def decode(ctrl, verbose=True):
    major,minor,devgen,nrows,ncols,nmemtile,numops,txnsize = struct.unpack_from("<BBBBBBxxII", ctrl, 0)
    print(f"hdr: major={major} minor={minor} devgen={devgen} nrows={nrows} ncols={ncols} "
          f"nmemtile={nmemtile} numops={numops} txnsize={txnsize} (buf {len(ctrl)})")
    off = 16
    ops = []
    for i in range(numops):
        op, col, row = struct.unpack_from("<BBB", ctrl, off)
        name = OPN.get(op, f"?{op}")
        if op == 0:  # WRITE: OpHdr pad + u64 regoff + val + size = 24
            regoff, val, size = struct.unpack_from("<QII", ctrl, off+8)
            ops.append((off,"WRITE",col,row,regoff,val,None,size))
            if verbose: print(f"{i:3} @{off:5x} WRITE     col={col} row={row} reg={regoff:#x} val={val:#x} sz={size}")
            off += size
        elif op == 3:  # MASKWRITE: + mask = 32
            regoff, val, mask, size = struct.unpack_from("<QIII", ctrl, off+8)
            ops.append((off,"MASKWRITE",col,row,regoff,val,mask,size))
            if verbose: print(f"{i:3} @{off:5x} MASKWRITE col={col} row={row} reg={regoff:#x} val={val:#x} mask={mask:#x} sz={size}")
            off += size
        elif op in (1,2,5,6):  # BLOCKWRITE/BLOCKSET/SHIMDMA_BD: 16B hdr + data
            c2, r2, regoff, size = struct.unpack_from("<BBxxII", ctrl, off+4)
            ndata = (size - 16)//4
            words = struct.unpack_from(f"<{max(ndata,0)}I", ctrl, off+16)
            ops.append((off,name,c2,r2,regoff,words,None,size))
            if verbose:
                print(f"{i:3} @{off:5x} {name:9} col={c2} row={r2} reg={regoff:#x} sz={size} data={[hex(w) for w in words]}")
            off += size
        else:
            size = struct.unpack_from("<I", ctrl, off+4)[0]
            print(f"{i:3} @{off:5x} {name} col={col} row={row} sz={size}")
            if size == 0: break
            off += size
    print(f"end offset {off:#x} (ctrltext {len(ctrl):#x})")
    return ops

if __name__ == "__main__":
    ctrl = load_ctrltext(sys.argv[1])
    decode(ctrl)
