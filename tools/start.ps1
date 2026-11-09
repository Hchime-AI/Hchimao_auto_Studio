<#
  启动打码工作台（PowerShell 版）
  由根目录的「启动打码工作台.bat」调用；用法：
      启动打码工作台.bat            默认端口 8090
      启动打码工作台.bat 9000       指定端口
      启动打码工作台.bat 9000 --lan 同时开放局域网访问
#>
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Find-Python {
    $venv = Join-Path $root ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }
    foreach ($name in @("python", "py")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

function Test-PortBusy([int]$port) {
    $listening = $false
    try {
        $lines = & "$env:SystemRoot\System32\netstat.exe" -ano 2>$null | Select-String -SimpleMatch ":$port "
        foreach ($l in $lines) { if ($l.Line -match "LISTENING") { $listening = $true; break } }
    } catch { }
    if ($listening) { return $true }
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $port)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

$python = Find-Python
if (-not $python) {
    Write-Host "[错误] 未找到 Python，请安装 Python 3.10+ 并勾选 Add Python to PATH" -ForegroundColor Red
    Write-Host "按任意键退出…"; [void][Console]::ReadKey($true); exit 1
}

Write-Host "[0/3] 清理旧实例…"
$cleanup = Join-Path $root "tools\cleanup.ps1"
if (Test-Path $cleanup) { & $cleanup }
Start-Sleep -Milliseconds 800

$port = 8090
$lan = ""
$argsList = @($args)
if ($argsList.Count -ge 1 -and $argsList[0] -match '^\d+$') { $port = [int]$argsList[0] }
if ($argsList -contains "--lan") { $lan = "--lan" }

$tries = 0
while ((Test-PortBusy $port) -and $tries -lt 20) {
    Write-Host "[提示] 端口 $port 被占用，改用 $($port + 1)"
    $port++
    $tries++
}
if ($tries -ge 20) {
    Write-Host "[错误] 8090 起 20 个端口都被占用，请手动指定：启动打码工作台.bat 9000" -ForegroundColor Red
    Write-Host "按任意键退出…"; [void][Console]::ReadKey($true); exit 1
}

Write-Host "=============================================="
Write-Host "  打码工作台   http://localhost:$port"
Write-Host "  关闭本窗口或按 Ctrl+C 停止服务"
Write-Host "=============================================="

Write-Host "[1/3] 检查依赖（首次较慢）…"
& $python (Join-Path $root "app\check_deps.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "[错误] 依赖检查未通过，看上面的日志" -ForegroundColor Red
    Write-Host "按任意键退出…"; [void][Console]::ReadKey($true); exit 1
}

Write-Host "[2/3] 启动服务，4 秒后自动打开浏览器…"
Start-Job -ScriptBlock {
    param($u)
    Start-Sleep -Seconds 4
    Start-Process $u
} -ArgumentList "http://localhost:$port" | Out-Null

Write-Host "[3/3] 就绪 —— 本窗口是服务端日志，关掉即停服"
$webuiArgs = @((Join-Path $root "app\webui.py"), "$port")
if ($lan) { $webuiArgs += $lan }
& $python @webuiArgs

Write-Host ""
Write-Host "[服务已退出]"
Write-Host "按任意键关闭…"
[void][Console]::ReadKey($true)
