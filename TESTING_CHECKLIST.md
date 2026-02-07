# ToneMatch AI VST v1.0 - Testing Checklist

This checklist corresponds to the 5 critical test scenarios from the final testing plan.

## Test 1: Python Bridge Integration ⭐ CRITICAL

### Pre-Test Setup
- [ ] Plugin compiled and installed in DAW
- [ ] Python 3.9+ installed and in PATH
- [ ] `run_match.py` copied to plugin Resources folder
- [ ] `src/` module accessible from plugin location
- [ ] Test audio files available (DI and reference)

### Test Steps
1. [ ] Load plugin in DAW (Reaper/Cubase/Logic)
2. [ ] Play guitar DI signal through plugin
3. [ ] Click **"MATCH TONE!"** button
4. [ ] Select reference audio file (.wav or .mp3)
5. [ ] Monitor console/logs for:
   - [ ] `DBG("PythonBridge: launching: ...")` appears
   - [ ] Python process visible in Task Manager/Activity Monitor
   - [ ] `tonematch_di.wav` created in temp directory
   - [ ] Python script executes (check output)
   - [ ] `tonematch_result.json` created in temp directory

### Expected Results
- [x] Python process launches successfully
- [x] `tonematch_result.json` appears in temp directory
- [x] JSON contains valid `rig` and `params` sections
- [x] No error messages in console

### Verification Points
- [ ] Check `PythonBridge.cpp:73` logs the command
- [ ] Check `PythonBridge.cpp:121-169` parses JSON correctly
- [ ] Verify JSON format matches `run_match.py:82-105` output structure

### Failure Modes to Check
- [ ] Python not found → Check PATH, verify `pythonPath` in `PythonBridge.h:88`
- [ ] Script not found → Verify `run_match.py` location matches `PythonBridge.cpp:40-48` search paths
- [ ] JSON parse error → Check JSON format matches `MatchResult` structure

---

## Test 2: AI Parameters Update

### Pre-Test Setup
- [ ] Test 1 completed successfully

### Test Steps
1. [ ] Complete Test 1 (Python bridge works)
2. [ ] Wait for matching to complete (3-5 minutes)
3. [ ] Observe UI and audio changes

### Expected Results
- [x] Sound changes immediately after matching completes
- [x] UI sliders (Delay/Reverb/Gain) automatically move to new values
- [x] Slot components show loaded model names:
  - [ ] FX Slot: Shows FX NAM name
  - [ ] AMP Slot: Shows AMP NAM name
  - [ ] IR Slot: Shows IR name
- [x] "LAST RIG FOUND" label updates
- [x] All parameters update via APVTS

### Verification Points
- [ ] Check `PluginProcessor.cpp:338-369` `onMatchComplete()` is called
- [ ] Verify `PluginProcessor.cpp:385-437` `applyNewRig()` loads models and updates parameters
- [ ] Confirm NAM models load via `PluginProcessor.cpp:388-408`
- [ ] Confirm IR loads via `PluginProcessor.cpp:410-420`

### Failure Modes to Check
- [ ] Models don't load → Check absolute paths in JSON, verify files exist
- [ ] Parameters don't update → Check APVTS parameter IDs match JSON keys
- [ ] UI doesn't refresh → Check `startTimerHz(4)` in `PluginEditor.cpp:32` for periodic updates

---

## Test 3: DSP Processing (NAM/IR)

### Pre-Test Setup
- [ ] Plugin loaded in DAW
- [ ] Audio playing through plugin

### Test Steps
1. [ ] Load plugin in DAW
2. [ ] Play guitar DI signal continuously
3. [ ] Manually load different NAM models:
   - [ ] Click FX Slot → Select different `.nam` file from `assets/nam_models/`
   - [ ] Click AMP Slot → Select different `.nam` file
   - [ ] Click IR Slot → Select different `.wav` file from `assets/impulse_responses/`
4. [ ] Test extreme cases:
   - [ ] Clean model → Hi-Gain model (should sound dramatically different)
   - [ ] Different IR cabinets (should change tone significantly)

### Expected Results
- [x] Zero or minimal latency (< 10ms)
- [x] No clicks, pops, or artifacts when switching models
- [x] Each model produces distinct, recognizable tone
- [x] Stable processing during 5+ minutes of continuous playback
- [x] No memory leaks (monitor Task Manager/Activity Monitor)

### Verification Points
- [ ] Check `PluginProcessor.cpp:135-273` `processBlock()` processes audio correctly
- [ ] Verify `NAMProcessor.cpp` loads models thread-safely (mutex protection at line 66, 99, 125)
- [ ] Confirm `PluginProcessor.cpp:219-223` NAM processing is lock-free during playback

### Failure Modes to Check
- [ ] Clicks/pops → Check model loading is thread-safe, verify no buffer underruns
- [ ] High latency → Check buffer size settings, optimize `processBlock()`
- [ ] Models sound identical → Verify NAM models are actually different, check model loading logic

---

## Test 4: UI Interaction

### Pre-Test Setup
- [ ] Plugin loaded in DAW with audio playing

### Test Steps
1. [ ] Load plugin in DAW with audio playing
2. [ ] Test each slider/knob:
   - [ ] **Delay Time** slider → Delay should change length
   - [ ] **Delay Mix** slider → Wet/dry balance should change
   - [ ] **Reverb Wet** slider → Reverb amount should change
   - [ ] **Reverb Room Size** slider → Room size should change
   - [ ] **Pre-EQ Gain** knob → Frequency response should change
   - [ ] **Pre-EQ Freq** knob → Peak frequency should shift
   - [ ] **Input Gain** knob → Input level should change
   - [ ] **Final EQ Gain** slider → Overall gain should change

### Expected Results
- [x] All controls respond **instantly** (no delay)
- [x] Text values update in real-time as sliders move
- [x] Audio changes immediately reflect parameter changes
- [x] No "sticking" or unresponsive controls
- [x] Parameter values persist when saving/loading presets

### Verification Points
- [ ] Check `PluginEditor.cpp` creates `SliderAttachment` for each control
- [ ] Verify `PluginProcessor.cpp:182-190` reads parameters from APVTS lock-free
- [ ] Confirm `PluginProcessor.cpp:194-272` applies parameters in real-time

### Failure Modes to Check
- [ ] Controls don't affect sound → Check `SliderAttachment` connections, verify APVTS parameter IDs
- [ ] Delayed response → Check audio thread isn't blocked, verify lock-free parameter access
- [ ] Values don't persist → Check preset save/load logic

---

## Test 5: Preset Management

### Pre-Test Setup
- [ ] Test 1-2 completed (AI matching works)

### Test Steps

#### Save Preset
1. [ ] Complete AI matching (Test 1-2) or manually configure rig
2. [ ] Click **"SAVE AS PRESET"** button
3. [ ] Save to `Documents/ToneMatchAI/Presets/MyPreset.json`
4. [ ] Open JSON file and verify contents:
   - [ ] `rig.fx_nam_path` - absolute path to FX NAM
   - [ ] `rig.amp_nam_path` - absolute path to AMP NAM
   - [ ] `rig.ir_path` - absolute path to IR
   - [ ] `params.*` - all 8 parameters present

#### Load Preset
1. [ ] Change all parameters and models to different values
2. [ ] Click **"LOAD PRESET"** (or use DAW preset menu)
3. [ ] Select saved preset file

### Expected Results
- [x] All three models (FX/AMP/IR) load automatically from saved paths
- [x] All 8 parameters restore to saved values
- [x] UI sliders reflect restored values
- [x] Sound matches saved state exactly
- [x] Preset works after DAW restart (paths remain valid)

### Verification Points
- [ ] Check `PresetManager.cpp:16-23` `savePreset()` serializes complete state
- [ ] Verify `PresetManager.cpp:43-85` `stateToVar()` includes all paths
- [ ] Confirm `PresetManager.cpp:87-126` `varToState()` restores via `applyNewRig()`
- [ ] Check `PresetManager.cpp:129-139` default directory creation

### Failure Modes to Check
- [ ] Models don't load from preset → Check paths are absolute, verify files exist at saved locations
- [ ] Parameters don't restore → Check JSON format matches expected structure
- [ ] Preset fails after restart → Verify paths are absolute or relative to fixed base directory

---

## Final Verification Checklist

### Build Quality
- [ ] Plugin compiles in **Release** mode (not Debug)
- [ ] No critical warnings (warnings from dependencies OK)
- [ ] Binary size is reasonable (< 50MB for VST3 bundle)
- [ ] All dependencies are statically linked or bundled

### File Deployment
- [ ] `run_match.py` accessible from plugin location
- [ ] Python `src/` module accessible (if needed)
- [ ] `assets/nam_models/` accessible (if plugin searches at runtime)
- [ ] `assets/impulse_responses/` accessible

### Integration Points
- [ ] Python bridge path resolution works (`PythonBridge.cpp:40-48`)
- [ ] JSON parsing handles all required fields (`PythonBridge.cpp:121-169`)
- [ ] Model loading handles missing files gracefully
- [ ] Preset system uses absolute paths for portability

### Performance
- [ ] Latency < 10ms at 512 sample buffer
- [ ] CPU usage < 20% on modern CPU (single instance)
- [ ] Memory usage stable (no leaks over 30 minutes)
- [ ] No audio dropouts during model switching

---

## Test Execution Log

**Date:** _______________
**Tester:** _______________
**DAW Used:** _______________
**OS:** _______________

### Results Summary
- Test 1 (Python Bridge): [ ] PASS [ ] FAIL
- Test 2 (AI Parameters): [ ] PASS [ ] FAIL
- Test 3 (DSP Processing): [ ] PASS [ ] FAIL
- Test 4 (UI Interaction): [ ] PASS [ ] FAIL
- Test 5 (Preset Management): [ ] PASS [ ] FAIL

### Issues Found
1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

### Notes
_________________________________________________
_________________________________________________

