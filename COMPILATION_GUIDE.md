# ToneMatch AI VST - Compilation Guide

## Current Status

⚠️ **CRITICAL ISSUE**: `plugin/CMakeLists.txt` is currently **EMPTY**

The plugin cannot be compiled until a proper CMakeLists.txt is configured with JUCE integration.

## Required CMakeLists.txt Configuration

The `plugin/CMakeLists.txt` needs to include:

1. **JUCE Framework Integration**
   ```cmake
   cmake_minimum_required(VERSION 3.20)
   project(ToneMatchAI VERSION 1.0.0)
   
   # Add JUCE
   add_subdirectory(JUCE)
   # OR set JUCE_DIR if JUCE is installed separately
   ```

2. **Plugin Target Definition**
   ```cmake
   juce_add_plugin(ToneMatchAI
       COMPANY_NAME "YourCompany"
       PLUGIN_MANUFACTURER_CODE YrCo
       PLUGIN_CODE TmAI
       FORMATS VST3 Standalone
       PRODUCT_NAME "ToneMatch AI"
       DESCRIPTION "AI-Powered Tone Matching VST Plugin"
       VERSION 1.0.0
   )
   ```

3. **Source Files**
   ```cmake
   target_sources(ToneMatchAI PRIVATE
       Source/PluginProcessor.cpp
       Source/PluginProcessor.h
       Source/PluginEditor.cpp
       Source/PluginEditor.h
       Source/Bridge/PythonBridge.cpp
       Source/Bridge/PythonBridge.h
       Source/DSP/NAMProcessor.cpp
       Source/DSP/NAMProcessor.h
       Source/DSP/DSPChain.cpp
       Source/DSP/DSPChain.h
       Source/Preset/PresetManager.cpp
       Source/Preset/PresetManager.h
       Source/UI/SlotComponent.cpp
       Source/UI/SlotComponent.h
       Source/UI/KnobStrip.cpp
       Source/UI/KnobStrip.h
   )
   ```

4. **NeuralAmpModelerCore Integration**
   ```cmake
   add_subdirectory(ThirdParty/NeuralAmpModelerCore)
   target_link_libraries(ToneMatchAI PRIVATE NAM)
   ```

5. **JUCE Modules**
   ```cmake
   target_link_libraries(ToneMatchAI PRIVATE
       juce::juce_audio_processors
       juce::juce_audio_utils
       juce::juce_dsp
       juce::juce_gui_basics
       juce::juce_gui_extra
   )
   ```

## Build Steps (Once CMakeLists.txt is Configured)

### Windows (Visual Studio 2022)

```powershell
cd plugin
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target ToneMatchAI_VST3
```

### macOS

```bash
cd plugin
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target ToneMatchAI_VST3
```

### Linux

```bash
cd plugin
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target ToneMatchAI_VST3
```

## Expected Output Locations

### Windows
- VST3: `plugin/build/ToneMatchAI_artefacts/Release/VST3/ToneMatchAI.vst3`
- Standalone: `plugin/build/ToneMatchAI_artefacts/Release/Standalone/ToneMatchAI.exe`

### macOS
- VST3: `plugin/build/ToneMatchAI_artefacts/Release/VST3/ToneMatchAI.vst3`
- AU: `plugin/build/ToneMatchAI_artefacts/Release/AU/ToneMatchAI.component`
- Standalone: `plugin/build/ToneMatchAI_artefacts/Release/Standalone/ToneMatchAI.app`

### Linux
- VST3: `plugin/build/ToneMatchAI_artefacts/Release/VST3/ToneMatchAI.so`

## Post-Build Deployment

After successful compilation, copy Python resources:

### Windows
```powershell
# Determine plugin location (usually in VST3 folder)
$pluginPath = "C:\Program Files\Common Files\VST3\ToneMatchAI.vst3\Contents\Resources"

# Copy Python script
Copy-Item plugin\Scripts\run_match.py $pluginPath\

# Copy Python source
xcopy /E /I src $pluginPath\src

# Copy assets (if needed at runtime)
xcopy /E /I assets $pluginPath\assets
```

### macOS
```bash
# Copy to VST3 bundle
cp plugin/Scripts/run_match.py ~/Library/Audio/Plug-Ins/VST3/ToneMatchAI.vst3/Contents/Resources/
cp -r src ~/Library/Audio/Plug-Ins/VST3/ToneMatchAI.vst3/Contents/Resources/
cp -r assets ~/Library/Audio/Plug-Ins/VST3/ToneMatchAI.vst3/Contents/Resources/
```

## Verification After Build

1. **Check Plugin Binary**
   - Verify `.vst3` file exists and has reasonable size (< 50MB)
   - Check that all dependencies are linked

2. **Verify Python Script Location**
   - Ensure `run_match.py` is accessible from plugin location
   - Test path resolution in `PythonBridge.cpp:40-48`

3. **Test Plugin Loading**
   - Load plugin in DAW (Reaper, Cubase, Logic)
   - Verify UI displays correctly
   - Check for any initialization errors

## Troubleshooting

### CMake Cannot Find JUCE
```powershell
# Set JUCE directory explicitly
cmake .. -DJUCE_DIR="C:/path/to/JUCE" -DCMAKE_BUILD_TYPE=Release
```

### NeuralAmpModelerCore Fails to Build
- Verify Eigen library is in `ThirdParty/NeuralAmpModelerCore/Dependencies/eigen`
- Check CMake version (requires 3.10+)

### Linker Errors
- Verify all source files are included in `target_sources()`
- Check that JUCE modules are properly linked
- Ensure NeuralAmpModelerCore is built before linking

## Next Steps

1. **Create/Configure CMakeLists.txt** - This is the critical blocker
2. **Run CMake Configuration** - Verify no errors
3. **Build Plugin** - Compile in Release mode
4. **Deploy Resources** - Copy Python files to plugin location
5. **Test in DAW** - Load and verify basic functionality

