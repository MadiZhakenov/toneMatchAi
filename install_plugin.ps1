# Install ToneMatch AI VST3 Plugin
# Run this script as Administrator to install to system folder

$ErrorActionPreference = "Stop"

$sourcePath = "$PSScriptRoot\plugin\build\ToneMatchAI_artefacts\Release\VST3\ToneMatch AI.vst3"
$userDestPath = "$env:USERPROFILE\Documents\VST3\ToneMatch AI.vst3"
$systemDestPath = "C:\Program Files\Common Files\VST3\ToneMatch AI.vst3"

Write-Host ("=" * 70)
Write-Host "ToneMatch AI VST3 Plugin Installation"
Write-Host ("=" * 70)

# Check if plugin exists
if (-not (Test-Path $sourcePath)) {
    Write-Host "ERROR: Plugin not found at: $sourcePath"
    Write-Host "Please compile the plugin first."
    exit 1
}

Write-Host "Found plugin at: $sourcePath"

# Install to user folder (always works)
Write-Host ""
Write-Host "Installing to user folder..."
if (Test-Path $userDestPath) {
    Remove-Item -Recurse -Force $userDestPath
}
Copy-Item -Recurse -Force $sourcePath $userDestPath
Write-Host "OK Installed to: $userDestPath"

# Deploy Python resources
Write-Host ""
Write-Host "Deploying Python resources..."
$resourcesPath = Join-Path $userDestPath "Contents\Resources"
New-Item -ItemType Directory -Path $resourcesPath -Force | Out-Null

Copy-Item "$PSScriptRoot\plugin\Scripts\run_match.py" $resourcesPath -Force
Write-Host "OK Copied run_match.py"

xcopy /E /I /Y "$PSScriptRoot\src" "$resourcesPath\src" 2>&1 | Out-Null
Write-Host "OK Copied src/ module"

# Try to install to system folder (requires admin)
Write-Host ""
Write-Host "Attempting to install to system folder..."
try {
    if (Test-Path $systemDestPath) {
        Remove-Item -Recurse -Force $systemDestPath
    }
    Copy-Item -Recurse -Force $sourcePath $systemDestPath -ErrorAction Stop
    
    # Deploy resources to system folder too
    $sysResourcesPath = Join-Path $systemDestPath "Contents\Resources"
    New-Item -ItemType Directory -Path $sysResourcesPath -Force | Out-Null
    Copy-Item "$PSScriptRoot\plugin\Scripts\run_match.py" $sysResourcesPath -Force
    xcopy /E /I /Y "$PSScriptRoot\src" "$sysResourcesPath\src" 2>&1 | Out-Null
    
    Write-Host "OK Installed to: $systemDestPath"
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "SUCCESS: Plugin installed to system folder!"
    Write-Host ("=" * 70)
}
catch {
    Write-Host "WARNING: Could not install to system folder (requires Administrator rights)"
    Write-Host "   Plugin is available at: $userDestPath"
    Write-Host ""
    Write-Host "To install to system folder manually:"
    Write-Host "   1. Run PowerShell as Administrator"
    Write-Host "   2. Copy-Item -Recurse -Force '$userDestPath' 'C:\Program Files\Common Files\VST3\'"
}

Write-Host ""
Write-Host "Plugin is ready to use in FL Studio!"
Write-Host "Location: $userDestPath"
if (Test-Path $systemDestPath) {
    Write-Host "Also available at: $systemDestPath"
}
