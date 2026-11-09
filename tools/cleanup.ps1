param([int[]]$Ports = @(8080,8081,8090,8091,8092))

function Get-Targets {
    $list = @()
    # 1) 命令行里带 webui.py 的 python 进程
    try {
        $list += Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                 Where-Object { $_.CommandLine -like '*webui.py*' } |
                 Select-Object -ExpandProperty ProcessId
    } catch { }
    # 2) 占着这些端口的 python 进程
    foreach ($p in $Ports) {
        try {
            foreach ($c in Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction Stop) {
                $pr = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
                if ($pr -and $pr.ProcessName -like 'python*') { $list += $c.OwningProcess }
            }
        } catch { }
    }
    return ($list | Where-Object { $_ } | Sort-Object -Unique)
}

$stuck = @()
for ($round = 1; $round -le 2; $round++) {
    $targets = Get-Targets
    if (-not $targets) { break }
    foreach ($procId in $targets) {
        & taskkill.exe /PID $procId /T /F *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host ("  [已结束] PID " + $procId)
        } else {
            if ($round -eq 2) {
                Write-Host ("  [跳过] PID " + $procId + " 需要管理员权限")
                $stuck += $procId
            }
        }
    }
    Start-Sleep -Milliseconds 800
}

$left = Get-Targets
if (-not $left -and $stuck.Count -eq 0) { Write-Host "  没有发现旧实例" }
if ($stuck.Count -gt 0) {
    Write-Host ("  这些进程没权限结束：" + (($stuck | Sort-Object -Unique) -join ", ") + "，请用管理员运行「清理所有实例(管理员).bat」")
}
if ($left) {
    $desc = @()
    foreach ($id in $left) { $desc += ("PID " + $id) }
    Write-Host ("  复查仍有实例存活：" + ($desc -join ", "))
} else {
    Write-Host "  复查：项目端口已全部释放"
}
