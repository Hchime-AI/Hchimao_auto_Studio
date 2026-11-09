<#
  以管理员权限清理所有打码工作台实例。
  由根目录的「清理所有实例(管理员).bat」调用；权限不够会自动弹 UAC 重新拉起自己。
#>
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "需要管理员权限，正在提权…"
    $self = $PSCommandPath
    try {
        Start-Process -FilePath "powershell.exe" `
            -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$self`"") `
            -Verb RunAs | Out-Null
    } catch {
        Write-Host "[错误] 提权被取消，未做任何清理。" -ForegroundColor Red
    }
    exit 0
}

Write-Host "正在结束打码工作台的所有实例…"
$cleanup = Join-Path $root "tools\cleanup.ps1"
if (Test-Path $cleanup) { & $cleanup }
else { Write-Host "[错误] 找不到 tools\cleanup.ps1" -ForegroundColor Red }

Write-Host ""
Write-Host "完成。若上面还有「需要管理员权限」，那个进程权限更高，请到任务管理器里结束它。"
Write-Host "按任意键关闭…"
[void][Console]::ReadKey($true)
