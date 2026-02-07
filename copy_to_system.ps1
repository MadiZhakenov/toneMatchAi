# Copy plugin to system VST3 folder
# MUST RUN AS ADMINISTRATOR

$source = "$env:USERPROFILE\Documents\VST3\ToneMatch AI.vst3"
$dest = "C:\Program Files\Common Files\VST3\ToneMatch AI.vst3"

if (-not (Test-Path $source)) {
    Write-Host "ERROR: Plugin not found at: $source"
    exit 1
}

Write-Host "Copying plugin to system folder..."
if (Test-Path $dest) {
    Remove-Item -Recurse -Force $dest
}
Copy-Item -Recurse -Force $source $dest

# Deploy Python resources
$resPath = Join-Path $dest "Contents\Resources"
New-Item -ItemType Directory -Path $resPath -Force | Out-Null
Copy-Item "E:\Users\Desktop\toneMatchAi\plugin\Scripts\run_match.py" $resPath -Force
xcopy /E /I /Y "E:\Users\Desktop\toneMatchAi\src" "$resPath\src" 2>&1 | Out-Null

Write-Host "SUCCESS: Plugin installed to: $dest"

