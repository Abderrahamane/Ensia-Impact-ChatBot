param(
    [int]$IntervalSeconds = 8,
    [switch]$Once
)

$ErrorActionPreference = "SilentlyContinue"

function Test-ProcessRunning {
    param([string]$Pattern)
    $procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $Pattern }
    return ($procs -and $procs.Count -gt 0)
}

function Test-WebHealth {
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -Method Get -TimeoutSec 2
        return ($resp.ok -eq $true)
    } catch {
        return $false
    }
}

function Get-HealthStatus {
    $webProc = Test-ProcessRunning "web\\app.py"
    $botProc = Test-ProcessRunning "bot\\telegram_bot.py"
    $webApi = Test-WebHealth

    if ($webProc -and $botProc -and $webApi) {
        return @{ State = "OK"; Message = "ENSIA stack healthy (web+bot+api)"; Icon = "Information" }
    }

    $parts = @()
    if (-not $webProc) { $parts += "web process down" }
    if (-not $botProc) { $parts += "bot process down" }
    if (-not $webApi) { $parts += "web api down" }
    return @{ State = "DOWN"; Message = ($parts -join "; "); Icon = "Error" }
}

if ($Once) {
    $s = Get-HealthStatus
    Write-Output ("state={0} msg={1}" -f $s.State, $s.Message)
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = [System.Drawing.SystemIcons]::Information
$notify.Visible = $true
$notify.Text = "ENSIA Health Monitor"

$menu = New-Object System.Windows.Forms.ContextMenuStrip
$itemStatus = $menu.Items.Add("Status")
$itemOpen = $menu.Items.Add("Open Web")
$itemExit = $menu.Items.Add("Exit")
$notify.ContextMenuStrip = $menu

$itemOpen.Add_Click({ Start-Process "http://127.0.0.1:8000" })
$itemExit.Add_Click({
    $notify.Visible = $false
    $notify.Dispose()
    [System.Windows.Forms.Application]::Exit()
})

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = [Math]::Max(3, $IntervalSeconds) * 1000
$lastState = ""

$timer.Add_Tick({
    $status = Get-HealthStatus
    $itemStatus.Text = "Status: $($status.State)"

    if ($status.State -eq "OK") {
        $notify.Icon = [System.Drawing.SystemIcons]::Information
    } else {
        $notify.Icon = [System.Drawing.SystemIcons]::Error
    }

    $notify.Text = ("ENSIA: " + $status.Message)

    if ($status.State -ne $lastState) {
        $notify.BalloonTipTitle = "ENSIA Stack Status"
        $notify.BalloonTipText = $status.Message
        $notify.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::$($status.Icon)
        $notify.ShowBalloonTip(2500)
        $script:lastState = $status.State
    }
})

$timer.Start()
[System.Windows.Forms.Application]::Run()

