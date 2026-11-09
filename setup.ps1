# Hchimao_auto_Studio — 环境重建脚本
# 用法（在项目根目录）：powershell -ExecutionPolicy Bypass -File setup.ps1
# 作用：新建 .venv → 按 requirements.txt 安装 → 跑自检
# 不影响已有文件：.venv 已存在时默认跳过（-Force 重建）；models\ 里的权重不会被改动

param(
    [switch]$Force,
    [string]$Python = "py"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venv  = Join-Path $root ".venv"
$pyExe = Join-Path $venv "Scripts\python.exe"
$req   = Join-Path $root "requirements.txt"

Write-Host "== 1/3 检查 Python" -ForegroundColor Cyan
& $Python -3.11 -V
if ($LASTEXITCODE -ne 0) { Write-Host "未找到 Python 3.11，请先安装或用 -Python 指定路径" -ForegroundColor Yellow }

Write-Host "== 2/3 准备虚拟环境" -ForegroundColor Cyan
if ((Test-Path $pyExe) -and (-not $Force)) {
    Write-Host "  已存在 $venv，跳过创建（用 -Force 重建）"
} else {
    if (Test-Path $venv) { Remove-Item $venv -Recurse -Force }
    & $Python -3.11 -m venv $venv
}

Write-Host "== 3/3 安装依赖" -ForegroundColor Cyan
& $pyExe -m pip install --upgrade pip --quiet
& $pyExe -m pip install -r $req

Write-Host "== 自检" -ForegroundColor Cyan
& $pyExe -c "import numpy, onnxruntime, cv2, PIL; print('numpy', numpy.__version__); print('onnxruntime', onnxruntime.__version__); print('opencv', cv2.__version__)"
if (Test-Path (Join-Path $root "app\selftest.py")) {
    & $pyExe (Join-Path $root "app\selftest.py") --quick
}

Write-Host "完成。启动：双击 启动打码工作台.bat" -ForegroundColor Green
