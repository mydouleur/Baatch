# Manual MSVC env setup (vcvars64.bat is broken on this machine)
# Usage: source vsenv.sh
MSVC="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207"
SDK="/c/Program Files (x86)/Windows Kits/10"
SDKV="10.0.26100.0"
export PATH="$MSVC/bin/Hostx64/x64:$SDK/bin/$SDKV/x64:$PATH"
export INCLUDE="$MSVC/include;$SDK\\Include\\$SDKV\\ucrt;$SDK\\Include\\$SDKV\\um;$SDK\\Include\\$SDKV\\shared;$SDK\\Include\\$SDKV\\winrt"
export LIB="$MSVC\\lib\\x64;$SDK\\Lib\\$SDKV\\ucrt\\x64;$SDK\\Lib\\$SDKV\\um\\x64"
# INCLUDE/LIB need windows-style paths for cl.exe
MSVC_W='C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207'
SDK_W='C:\Program Files (x86)\Windows Kits\10'
export INCLUDE="$MSVC_W\\include;$SDK_W\\Include\\$SDKV\\ucrt;$SDK_W\\Include\\$SDKV\\um;$SDK_W\\Include\\$SDKV\\shared;$SDK_W\\Include\\$SDKV\\winrt"
export LIB="$MSVC_W\\lib\\x64;$SDK_W\\Lib\\$SDKV\\ucrt\\x64;$SDK_W\\Lib\\$SDKV\\um\\x64"
