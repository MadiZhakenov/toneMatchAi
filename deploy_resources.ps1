# Deploy Python resources to plugin installation location
# Run this script after the plugin has been compiled and installed

param(
    [string]$PluginPath = "",
    [string]$ProjectRoot = $PSScriptRoot
)

Write-Host "=" * 70
Write-Host "ToneMatch AI - Resource Deployment Script"
Write-Host "=" * 70

# Determine plugin path
if ([string]::IsNullOrEmpty($PluginPath)) {
    # Default VST3 locations
    $vst3Paths = @(
        "C:\Program Files\Common Files\VST3\ToneMatchAI.vst3",
        "$env:USERPROFILE\Documents\VST3\ToneMatchAI.vst3",
        "$env:LOCALAPPDATA\Programs\Common\VST3\ToneMatchAI.vst3"
    )
    
    foreach ($path in $vst3Paths) {
        if (Test-Path $path) {
            $PluginPath = $path
            Write-Host "Found plugin at: $PluginPath"
            break
        }
    }
    
    if ([string]::IsNullOrEmpty($PluginPath)) {
        Write-Host "ERROR: Plugin not found in default locations."
        Write-Host "Please specify plugin path: .\deploy_resources.ps1 -PluginPath 'C:\path\to\ToneMatchAI.vst3'"
        exit 1
    }
}

# Determine Resources directory (VST3 bundle structure)
$ResourcesPath = Join-Path $PluginPath "Contents\Resources"
if (-not (Test-Path $ResourcesPath)) {
    # Try Windows VST3 structure (no Contents folder)
    $ResourcesPath = Join-Path $PluginPath "Resources"
    if (-not (Test-Path $ResourcesPath)) {
        # Create Resources directory
        $ResourcesPath = Join-Path $PluginPath "Resources"
        New-Item -ItemType Directory -Path $ResourcesPath -Force | Out-Null
        Write-Host "Created Resources directory: $ResourcesPath"
    }
}

Write-Host "Deploying to: $ResourcesPath"

# Copy Python script
$scriptSource = Join-Path $ProjectRoot "plugin\Scripts\run_match.py"
$scriptDest = Join-Path $ResourcesPath "run_match.py"

if (Test-Path $scriptSource) {
    Copy-Item $scriptSource $scriptDest -Force
    Write-Host "✓ Copied run_match.py"
} else {
    Write-Host "✗ ERROR: run_match.py not found at $scriptSource"
    exit 1
}

# Copy Python source module
$srcSource = Join-Path $ProjectRoot "src"
$srcDest = Join-Path $ResourcesPath "src"

if (Test-Path $srcSource) {
    if (Test-Path $srcDest) {
        Remove-Item $srcDest -Recurse -Force
    }
    Copy-Item $srcSource $srcDest -Recurse -Force
    Write-Host "✓ Copied src/ module"
} else {
    Write-Host "✗ ERROR: src/ directory not found at $srcSource"
    exit 1
}

# Copy assets (if needed at runtime)
$assetsSource = Join-Path $ProjectRoot "assets"
$assetsDest = Join-Path $ResourcesPath "assets"

if (Test-Path $assetsSource) {
    if (Test-Path $assetsDest) {
        Remove-Item $assetsDest -Recurse -Force
    }
    Copy-Item $assetsSource $assetsDest -Recurse -Force
    Write-Host "✓ Copied assets/ directory"
} else {
    Write-Host "⚠ WARNING: assets/ directory not found (may not be needed)"
}

# Verify deployment
Write-Host "`nVerifying deployment..."

$checks = @(
    @{Path = $scriptDest; Name = "run_match.py"},
    @{Path = $srcDest; Name = "src/ module"},
    @{Path = (Join-Path $srcDest "core\optimizer.py"); Name = "optimizer.py"}
)

$allOk = $true
foreach ($check in $checks) {
    if (Test-Path $check.Path) {
        Write-Host "✓ $($check.Name) - OK"
    } else {
        Write-Host "✗ $($check.Name) - MISSING"
        $allOk = $false
    }
}

if ($allOk) {
    Write-Host "`n" + "=" * 70
    Write-Host "✅ Resource deployment completed successfully!"
    Write-Host "=" * 70
} else {
    Write-Host "`n" + "=" * 70
    Write-Host "❌ Resource deployment completed with errors"
    Write-Host "=" * 70
    exit 1
}

