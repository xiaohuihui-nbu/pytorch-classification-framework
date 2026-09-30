param(
    [string]$Server = 'gpu-server',
    [string]$BindAddress = '192.168.0.42',
    [ValidateRange(1, 65535)][int]$Port = 5173
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$logDirectory = Join-Path $projectRoot 'logs/remote-web'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$sshCommand = (Get-Command ssh.exe).Source
$tunnelLog = Join-Path $logDirectory 'lan-tunnel.log'
$PID | Set-Content (Join-Path $logDirectory 'lan-tunnel-supervisor.pid')
$forwardArgs = @('-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
    '-L', "127.0.0.1:${Port}:127.0.0.1:8000")
if ($BindAddress -ne '127.0.0.1') {
    $forwardArgs += @('-L', "${BindAddress}:${Port}:127.0.0.1:8000")
}

# 前台运行本脚本可用 Ctrl+C 停止；后台运行时按日志中的 supervisor PID 停止进程树。
while ($true) {
    Add-Content -Path $tunnelLog -Value "$(Get-Date -Format o) connecting $Server"
    try {
        & $sshCommand @forwardArgs $Server 2>> $tunnelLog
    } catch {
        Add-Content -Path $tunnelLog -Value $_.Exception.Message
    }
    Start-Sleep -Seconds 15
}
