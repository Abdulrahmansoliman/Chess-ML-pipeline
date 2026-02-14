import sys
print("Version:", sys.version)
print("Platform:", sys.platform)
print("Executable:", sys.executable)
import struct
print("Architecture:", struct.calcsize("P") * 8, "bit")
sasadada