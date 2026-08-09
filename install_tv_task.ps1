# Registers the VTLTradingView scheduled task.
#
# Why this exists: TradingView launched normally does NOT expose the Chrome
# DevTools port, so tv_sync and every mcp__tradingview__* tool have nothing to
# attach to. That is why the feed cross-check logged TV_UNAVAILABLE every run
# from 2026-08-05 to 2026-08-09 without anything appearing to be broken.
#
# Must be run from an ELEVATED PowerShell — Register-ScheduledTask returns
# "Zugriff verweigert" otherwise. Right-click PowerShell -> Run as administrator,
# then:
#
#     & "$HOME\Projects\vibe-trading-local\install_tv_task.ps1"

$ErrorActionPreference = "Stop"

$exe = "C:\Program Files\WindowsApps\TradingView.Desktop_3.3.0.7992_x64__n534cwy3pjxzj\TradingView.exe"
if (-not (Test-Path $exe)) {
    Write-Host "TradingView not found at:`n  $exe" -ForegroundColor Red
    Write-Host "Find the current package with:  Get-AppxPackage -Name *TradingView*"
    exit 1
}

$action  = New-ScheduledTaskAction -Execute $exe -Argument "--remote-debugging-port=9222"

$trigger = New-ScheduledTaskTrigger -AtLogOn
# One minute, so the shell and network are up before an Electron app starts.
$trigger.Delay = "PT1M"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable
# ExecutionTimeLimit zero = unlimited. The 3-day default would silently kill a
# chart left open over a long weekend, and the port would just stop answering —
# the same failure mode the VTLDashboard task already had to work around.

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive
# Interactive, not Service: TradingView is a GUI app and needs a desktop.

Register-ScheduledTask `
    -TaskName "VTLTradingView" `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Launch TradingView Desktop with CDP on port 9222 so tv_sync and the TradingView MCP tools can attach. Without the flag the feed cross-check records TV_UNAVAILABLE." `
    -Force | Out-Null

Write-Host "`nRegistered:" -ForegroundColor Green
Get-ScheduledTask -TaskName "VTLTradingView" | Select-Object TaskName, State | Format-Table -AutoSize
(Get-ScheduledTask -TaskName "VTLTradingView").Actions |
    Select-Object Execute, Arguments | Format-List

Write-Host "Verify after the next logon with:" -ForegroundColor Cyan
Write-Host '  curl.exe -s http://127.0.0.1:9222/json/version'
