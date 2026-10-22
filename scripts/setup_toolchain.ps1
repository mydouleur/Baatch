# Baatch 开发环境检查（只读，不安装）
$ErrorActionPreference = 'Continue'
$ok = $true

# 1. NPU 驱动
$npu = Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match 'NPU' } | Select-Object -First 1
if ($npu) {
    $ver = (Get-PnpDeviceProperty -InstanceId $npu.InstanceId -KeyName 'DEVPKEY_Device_DriverVersion').Data
    Write-Host "[OK] NPU 驱动: $ver（要求 >= 32.0.203.280）"
} else { Write-Host "[FAIL] 未找到 NPU 设备"; $ok = $false }

# 2. XRT 运行时
if (Test-Path C:\Windows\System32\xrt_coreutil.dll) { Write-Host "[OK] XRT 运行时（xrt_coreutil.dll）" }
else { Write-Host "[FAIL] xrt_coreutil.dll 缺失（随 NPU 驱动安装）"; $ok = $false }

# 3. Ryzen AI SW
$rai = "C:\Program Files\RyzenAI\1.8.0"
if (Test-Path $rai) { Write-Host "[OK] Ryzen AI SW: $rai" }
else { Write-Host "[FAIL] Ryzen AI SW 未安装"; $ok = $false }

# 4. conda 环境
$py = "C:\ProgramData\miniforge3\envs\ryzen-ai-1.8.0\python.exe"
if (Test-Path $py) { Write-Host "[OK] conda env ryzen-ai-1.8.0" }
else { Write-Host "[WARN] conda env 缺失（仅 EP 基准需要，引擎本体不依赖）" }

# 5. 中文系统规避变量
if ([Environment]::GetEnvironmentVariable("PYTHONUTF8","Machine") -eq "1") { Write-Host "[OK] PYTHONUTF8=1（中文系统规避）" }
else { Write-Host "[WARN] PYTHONUTF8 未设（装 Ryzen AI 时会崩，见 issue amd/RyzenAI-SW#399）" }

if ($ok) { Write-Host "`n环境就绪" } else { Write-Host "`n有缺失项，见上方 FAIL" }
