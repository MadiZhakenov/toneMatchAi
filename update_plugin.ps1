# Update ToneMatch AI VST3 Plugin in system folder
# MUST RUN AS ADMINISTRATOR

$ErrorActionPreference = "Stop"

$source = "$PSScriptRoot\plugin\build\ToneMatchAI_artefacts\Release\VST3\ToneMatch AI.vst3"
$dest = "C:\Program Files\Common Files\VST3\ToneMatch AI.vst3"

Write-Host "=" * 70
Write-Host "Updating ToneMatch AI VST3 Plugin"
Write-Host "=" * 70

if (-not (Test-Path $source)) {
    Write-Host "ERROR: Plugin not found at: $source"
    Write-Host "Please compile the plugin first:"
    Write-Host "  cd plugin\build"
    Write-Host "  cmake --build . --config Release --target ToneMatchAI_VST3"
    exit 1
}

Write-Host "Found plugin at: $source"
Write-Host ""

Write-Host "Updating system folder..."
try {
    if (Test-Path $dest) {
        Remove-Item -Recurse -Force $dest -ErrorAction Stop
    }
    Copy-Item -Recurse -Force $source $dest -ErrorAction Stop
    
    Write-Host ""
    Write-Host "=" * 70
    Write-Host "SUCCESS: Plugin updated in system folder!"
    Write-Host "Location: $dest"
    Write-Host "=" * 70
}
catch {
    Write-Host ""
    Write-Host "ERROR: Failed to update plugin"
    Write-Host "Error: $_"
    Write-Host ""
    Write-Host "Make sure you're running PowerShell as Administrator"
    exit 1
}

