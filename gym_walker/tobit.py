import struct


def floatToBits(f):
     s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]

def bitsToFloat(b):
     s = struct.pack('>l', b)
    return struct.unpack('>f', s)[0]

