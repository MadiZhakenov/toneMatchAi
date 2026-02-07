# ToneMatch AI VST - Resource Deployment Guide

## Overview

After compiling the plugin, Python resources must be deployed to the plugin installation location so the C++ plugin can find and execute the Python optimizer.

## Deployment Requirements

The plugin needs access to:
1. **run_match.py** - Python bridge script
2. **src/** - Python optimizer module
3. **assets/** - NAM models and IR files (if needed at runtime)

## Automatic Deployment

Use the provided PowerShell script:

```powershell
.\deploy_resources.ps1
```

The script will:
- Auto-detect plugin location in common VST3 directories
- Copy all required resources
- Verify deployment

### Manual Plugin Path

If plugin is in a custom location:

```powershell
.\deploy_resources.ps1 -PluginPath "C:\Custom\Path\ToneMatchAI.vst3"
```

## Manual Deployment

### Windows VST3

1. **Locate Plugin**
   - Default: `C:\Program Files\Common Files\VST3\ToneMatchAI.vst3`
   - Or: `%USERPROFILE%\Documents\VST3\ToneMatchAI.vst3`

2. **Copy Resources**
   ```powershell
   # Navigate to plugin Resources folder
   cd "C:\Program Files\Common Files\VST3\ToneMatchAI.vst3\Contents\Resources"
   
   # Copy Python script
   Copy-Item "E:\Users\Desktop\toneMatchAi\plugin\Scripts\run_match.py" .
   
   # Copy Python source
   xcopy /E /I "E:\Users\Desktop\toneMatchAi\src" "src"
   
   # Copy assets (if needed)
   xcopy /E /I "E:\Users\Desktop\toneMatchAi\assets" "assets"
   ```

### macOS VST3

1. **Locate Plugin**
   - Default: `~/Library/Audio/Plug-Ins/VST3/ToneMatchAI.vst3`

2. **Copy Resources**
   ```bash
   # Navigate to plugin Resources folder
   cd ~/Library/Audio/Plug-Ins/VST3/ToneMatchAI.vst3/Contents/Resources
   
   # Copy Python script
   cp /path/to/toneMatchAi/plugin/Scripts/run_match.py .
   
   # Copy Python source
   cp -r /path/to/toneMatchAi/src .
   
   # Copy assets (if needed)
   cp -r /path/to/toneMatchAi/assets .
   ```

### Linux VST3

1. **Locate Plugin**
   - Default: `~/.vst3/ToneMatchAI.so` or `/usr/local/lib/vst3/ToneMatchAI.so`

2. **Copy Resources**
   ```bash
   # Create Resources directory if needed
   mkdir -p ~/.vst3/ToneMatchAI/Resources
   cd ~/.vst3/ToneMatchAI/Resources
   
   # Copy Python script
   cp /path/to/toneMatchAi/plugin/Scripts/run_match.py .
   
   # Copy Python source
   cp -r /path/to/toneMatchAi/src .
   
   # Copy assets (if needed)
   cp -r /path/to/toneMatchAi/assets .
   ```

## Path Resolution

The C++ plugin searches for `run_match.py` in this order:

1. **Custom path** (if `setScriptPath()` was called)
2. **Scripts/run_match.py** relative to plugin executable
3. **Error** if not found

See `PythonBridge.cpp:40-48` for implementation details.

## Verification

After deployment, verify:

1. **File Structure**
   ```
   ToneMatchAI.vst3/
   └── Contents/
       └── Resources/
           ├── run_match.py
           ├── src/
           │   └── core/
           │       └── optimizer.py
           └── assets/
               ├── nam_models/
               └── impulse_responses/
   ```

2. **Python Script Test**
   ```powershell
   cd "C:\Program Files\Common Files\VST3\ToneMatchAI.vst3\Contents\Resources"
   python run_match.py --help
   ```

3. **Module Import Test**
   ```powershell
   cd "C:\Program Files\Common Files\VST3\ToneMatchAI.vst3\Contents\Resources"
   python -c "import sys; sys.path.insert(0, '.'); from src.core.optimizer import ToneOptimizer; print('OK')"
   ```

## Troubleshooting

### "run_match.py not found"

**Symptoms:**
- Plugin error: "run_match.py not found at: ..."

**Solutions:**
1. Verify script is in `Resources/` or `Scripts/` folder
2. Check file permissions (script must be readable)
3. Verify path resolution in `PythonBridge.cpp:40-48`

### "ModuleNotFoundError: No module named 'src'"

**Symptoms:**
- Python script fails with import error

**Solutions:**
1. Verify `src/` directory is copied to Resources
2. Check `run_match.py:16-20` path resolution
3. Ensure project root is added to `sys.path`

### "FileNotFoundError: assets/nam_models/..."

**Symptoms:**
- Python optimizer can't find NAM models

**Solutions:**
1. Copy `assets/` directory to Resources
2. Verify paths in `run_match.py` are relative to script location
3. Check that optimizer searches correct paths

## Post-Deployment Testing

After deployment, run Test 1 (Python Bridge Integration) from `TESTING_CHECKLIST.md`:

1. Load plugin in DAW
2. Click "MATCH TONE!"
3. Verify Python process launches
4. Check that `tonematch_result.json` is created

## Notes

- **Absolute vs Relative Paths**: The Python script converts relative paths to absolute paths in JSON output (see `run_match.py:66-80`)
- **Python Version**: Requires Python 3.9+ (verify with `python --version`)
- **Dependencies**: All Python dependencies from `requirements.txt` must be installed in the Python environment used by the plugin

