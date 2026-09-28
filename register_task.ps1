$RootDir = $PSScriptRoot
$TaskName = "AutoKaggle_Daily"
$BatPath = Join-Path $RootDir "run_autokaggle.bat"

$Action = New-ScheduledTaskAction -Execute $BatPath -WorkingDirectory $RootDir

$Trigger = New-ScheduledTaskTrigger -Daily -At 8:00AM

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "AutoKaggle v1 daily run" `
    -Force

Write-Host "Registered task '$TaskName' pointing at $BatPath. Verify with: Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "This does not wake the laptop from sleep. Add -WakeToRun to New-ScheduledTaskSettingsSet above for that, plus BIOS/power wake-timer support."
