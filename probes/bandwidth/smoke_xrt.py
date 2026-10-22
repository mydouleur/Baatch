import ctypes
xrt = ctypes.CDLL(r"C:\Windows\System32\xrt_coreutil.dll")
xrt.xrtDeviceOpen.restype = ctypes.c_void_p
xrt.xrtDeviceOpen.argtypes = [ctypes.c_uint]
h = xrt.xrtDeviceOpen(0)
print("device handle:", hex(h) if h else h)
if h:
    # query xclbin uuid
    buf = ctypes.create_string_buffer(64)
    xrt.xrtDeviceGetXclbinUUID.restype = ctypes.c_int
    xrt.xrtDeviceGetXclbinUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    r = xrt.xrtDeviceGetXclbinUUID(h, buf)
    print("xclbin uuid rc:", r, buf.raw[:16].hex())
    xrt.xrtDeviceClose.argtypes = [ctypes.c_void_p]
    xrt.xrtDeviceClose(h)
    print("closed")
