#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin CLI wrapper for the ToneMatch AI optimizer.

Called by the JUCE plugin via ChildProcess:
    python run_match.py --di path/to/di.wav --ref path/to/ref.wav --out preset.json

Writes a JSON file with the matched rig + parameters.
"""

import argparse
import json
import os
import sys

# Fix encoding for Windows console to prevent 'charmap' codec errors
if sys.platform == 'win32':
    import io
    import codecs
    # Set UTF-8 encoding for stdout and stderr
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Add project root to sys.path so we can import src.*
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.io import load_audio_file
from src.core.optimizer import ToneOptimizer


def find_assets_paths():
    """Determine paths to assets folder (NAM models and IR files).
    
    Automatically detects if running from VST3 installation or dev environment.
    Checks paths in order:
    1. Assets relative to script location (Resources/assets/)
    2. Common VST3 shared locations (C:\Program Files\Common Files\VST3\assets\)
    3. Dev environment (PROJECT_ROOT/assets/)
    
    Returns:
        tuple: (nam_folder, ir_folder) - absolute paths to NAM and IR folders
    """
    def normalize_path(path):
        """Normalize path for Windows compatibility."""
        return os.path.normpath(os.path.abspath(path))
    
    def check_assets_valid(nam_folder, ir_folder):
        """Check if asset folders exist and contain files.
        
        Returns:
            tuple: (is_valid, nam_count, ir_count) - validation result and file counts
        """
        nam_folder = normalize_path(nam_folder)
        ir_folder = normalize_path(ir_folder)
        
        if not os.path.exists(nam_folder) or not os.path.isdir(nam_folder):
            return False, 0, 0
        if not os.path.exists(ir_folder) or not os.path.isdir(ir_folder):
            return False, 0, 0
        
        # Check that folders contain files
        try:
            nam_files = [f for f in os.listdir(nam_folder) 
                        if f.lower().endswith('.nam') and os.path.isfile(os.path.join(nam_folder, f))]
            ir_files = [f for f in os.listdir(ir_folder) 
                       if f.lower().endswith('.wav') and os.path.isfile(os.path.join(ir_folder, f))]
            
            nam_count = len(nam_files)
            ir_count = len(ir_files)
            
            # Require at least one file in each folder
            if nam_count > 0 and ir_count > 0:
                return True, nam_count, ir_count
        except (OSError, PermissionError) as e:
            print(f"[Bridge] Warning: Error checking assets: {e}", file=sys.stderr)
        
        return False, 0, 0
    
    # Strategy 1: Check assets relative to script location (most reliable for VST3)
    # Path structure: .../ToneMatch AI.vst3/Contents/Resources/run_match.py
    # Assets should be at: .../ToneMatch AI.vst3/Contents/Resources/assets/
    script_assets_root = normalize_path(os.path.join(SCRIPT_DIR, "assets"))
    script_nam_folder = normalize_path(os.path.join(script_assets_root, "nam_models"))
    script_ir_folder = normalize_path(os.path.join(script_assets_root, "impulse_responses"))
    
    is_valid, nam_count, ir_count = check_assets_valid(script_nam_folder, script_ir_folder)
    if is_valid:
        print(f"[Bridge] Detected assets relative to script: {script_assets_root} ({nam_count} NAM, {ir_count} IR)", file=sys.stderr)
        return script_nam_folder, script_ir_folder
    else:
        print(f"[Bridge] Assets not found at script location: {script_assets_root}", file=sys.stderr)
    
    # Strategy 2: Check common VST3 shared locations
    common_vst3_paths = [
        r"C:\Program Files\Common Files\VST3",
        r"C:\Program Files (x86)\Common Files\VST3",
    ]
    
    for vst3_path in common_vst3_paths:
        vst3_path = normalize_path(vst3_path)
        if not os.path.exists(vst3_path):
            print(f"[Bridge] VST3 path does not exist: {vst3_path}", file=sys.stderr)
            continue
        
        assets_root = normalize_path(os.path.join(vst3_path, "assets"))
        nam_folder = normalize_path(os.path.join(assets_root, "nam_models"))
        ir_folder = normalize_path(os.path.join(assets_root, "impulse_responses"))
        
        is_valid, nam_count, ir_count = check_assets_valid(nam_folder, ir_folder)
        if is_valid:
            print(f"[Bridge] Detected VST3 shared assets at: {assets_root} ({nam_count} NAM, {ir_count} IR)", file=sys.stderr)
            return nam_folder, ir_folder
        else:
            print(f"[Bridge] Assets not found at VST3 location: {assets_root}", file=sys.stderr)
    
    # Strategy 3: Check if script is in VST3 installation and look for assets in parent VST3 folder
    script_path_lower = SCRIPT_DIR.lower()
    if "vst3" in script_path_lower:
        current = normalize_path(SCRIPT_DIR)
        # Go up until we find a folder named "VST3" or reach a reasonable depth
        for _ in range(5):  # Max 5 levels up
            parent = normalize_path(os.path.dirname(current))
            current_basename = os.path.basename(current).upper()
            parent_basename = os.path.basename(parent).upper()
            
            if parent_basename == "VST3" or current_basename == "VST3":
                # Found VST3 folder
                vst3_root = current if current_basename == "VST3" else parent
                assets_root = normalize_path(os.path.join(vst3_root, "assets"))
                nam_folder = normalize_path(os.path.join(assets_root, "nam_models"))
                ir_folder = normalize_path(os.path.join(assets_root, "impulse_responses"))
                
                is_valid, nam_count, ir_count = check_assets_valid(nam_folder, ir_folder)
                if is_valid:
                    print(f"[Bridge] Detected VST3 installation assets at: {assets_root} ({nam_count} NAM, {ir_count} IR)", file=sys.stderr)
                    return nam_folder, ir_folder
                else:
                    print(f"[Bridge] Assets not found at VST3 installation: {assets_root}", file=sys.stderr)
            
            if parent == current:  # Reached filesystem root
                break
            current = parent
    
    # Strategy 4: Dev environment fallback - use PROJECT_ROOT/assets/
    dev_assets_root = normalize_path(os.path.join(PROJECT_ROOT, "assets"))
    dev_nam_folder = normalize_path(os.path.join(dev_assets_root, "nam_models"))
    dev_ir_folder = normalize_path(os.path.join(dev_assets_root, "impulse_responses"))
    
    is_valid, nam_count, ir_count = check_assets_valid(dev_nam_folder, dev_ir_folder)
    if is_valid:
        print(f"[Bridge] Using dev environment paths: {dev_assets_root} ({nam_count} NAM, {ir_count} IR)", file=sys.stderr)
        return dev_nam_folder, dev_ir_folder
    else:
        print(f"[Bridge] Warning: Dev environment assets not found at: {dev_assets_root}", file=sys.stderr)
        # Return paths anyway - validation will happen later in main()
        return dev_nam_folder, dev_ir_folder


def main():
    parser = argparse.ArgumentParser(description="ToneMatch AI — CLI optimizer bridge")
    parser.add_argument("--di",  required=True, help="Path to DI (dry input) audio file")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--out", required=True, help="Path to write the result JSON")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Test duration in seconds (default: 5.0)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Max optimizer iterations (default: 50)")
    parser.add_argument("--force_high_gain", action="store_true",
                        help="Force high-gain mode: skip clean amps, only search high-gain models")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.di):
        print(f"[ERROR] DI file not found: {args.di}", file=sys.stderr)
        print(f"[ERROR] DI file not found: {args.di}", file=sys.stdout)  # Also to stdout for visibility
        sys.exit(1)
    if not os.path.exists(args.ref):
        print(f"[ERROR] Reference file not found: {args.ref}", file=sys.stderr)
        print(f"[ERROR] Reference file not found: {args.ref}", file=sys.stdout)  # Also to stdout for visibility
        sys.exit(1)

    # Load audio
    print(f"[Bridge] Loading DI: {args.di}")
    di_track = load_audio_file(args.di)

    print(f"[Bridge] Loading Reference: {args.ref}")
    ref_track = load_audio_file(args.ref)

    # Determine asset paths (NAM models and IR files)
    print(f"[Bridge] Searching for assets...", file=sys.stderr)
    print(f"[Bridge] Script location: {SCRIPT_DIR}", file=sys.stderr)
    print(f"[Bridge] Project root: {PROJECT_ROOT}", file=sys.stderr)
    
    nam_folder, ir_folder = find_assets_paths()
    
    # Ensure all paths are absolute and normalized (critical for plugin environment)
    nam_folder = os.path.normpath(os.path.abspath(nam_folder))
    ir_folder = os.path.normpath(os.path.abspath(ir_folder))
    
    # Log absolute paths for debugging
    print(f"[Bridge] Using absolute NAM folder: {nam_folder}", file=sys.stderr)
    print(f"[Bridge] Using absolute IR folder: {ir_folder}", file=sys.stderr)
    
    # Validate that asset folders exist and contain files
    validation_errors = []
    
    if not os.path.exists(nam_folder):
        validation_errors.append(f"NAM models folder does not exist: {nam_folder}")
    elif not os.path.isdir(nam_folder):
        validation_errors.append(f"NAM models path is not a directory: {nam_folder}")
    else:
        # Check for .nam files
        try:
            nam_files = [f for f in os.listdir(nam_folder) 
                        if f.lower().endswith('.nam') and os.path.isfile(os.path.join(nam_folder, f))]
            nam_count = len(nam_files)
            if nam_count == 0:
                validation_errors.append(f"NAM models folder is empty (no .nam files found): {nam_folder}")
        except (OSError, PermissionError) as e:
            validation_errors.append(f"Cannot read NAM models folder: {nam_folder} (Error: {e})")
    
    if not os.path.exists(ir_folder):
        validation_errors.append(f"IR files folder does not exist: {ir_folder}")
    elif not os.path.isdir(ir_folder):
        validation_errors.append(f"IR files path is not a directory: {ir_folder}")
    else:
        # Check for .wav files
        try:
            ir_files = [f for f in os.listdir(ir_folder) 
                       if f.lower().endswith('.wav') and os.path.isfile(os.path.join(ir_folder, f))]
            ir_count = len(ir_files)
            if ir_count == 0:
                validation_errors.append(f"IR files folder is empty (no .wav files found): {ir_folder}")
        except (OSError, PermissionError) as e:
            validation_errors.append(f"Cannot read IR files folder: {ir_folder} (Error: {e})")
    
    # If there are validation errors, report them and exit
    if validation_errors:
        error_header = "[ERROR] Asset validation failed:"
        print(error_header, file=sys.stderr)
        print(error_header, file=sys.stdout)
        
        for error in validation_errors:
            print(f"[ERROR] {error}", file=sys.stderr)
            print(f"[ERROR] {error}", file=sys.stdout)
        
        print(f"[ERROR] Script location: {SCRIPT_DIR}", file=sys.stderr)
        print(f"[ERROR] Project root: {PROJECT_ROOT}", file=sys.stderr)
        print(f"[ERROR] NAM folder: {nam_folder}", file=sys.stderr)
        print(f"[ERROR] IR folder: {ir_folder}", file=sys.stderr)
        print(f"[ERROR] Please ensure assets are installed correctly.", file=sys.stderr)
        print(f"[ERROR] Expected structure:", file=sys.stderr)
        print(f"[ERROR]   {nam_folder}/", file=sys.stderr)
        print(f"[ERROR]     *.nam files", file=sys.stderr)
        print(f"[ERROR]   {ir_folder}/", file=sys.stderr)
        print(f"[ERROR]     *.wav files", file=sys.stderr)
        sys.exit(1)
    
    # Count available models for debugging (already validated above)
    try:
        nam_count = len([f for f in os.listdir(nam_folder) 
                        if f.lower().endswith('.nam') and os.path.isfile(os.path.join(nam_folder, f))])
        ir_count = len([f for f in os.listdir(ir_folder) 
                       if f.lower().endswith('.wav') and os.path.isfile(os.path.join(ir_folder, f))])
    except (OSError, PermissionError) as e:
        print(f"[ERROR] Failed to count files in asset folders: {e}", file=sys.stderr)
        nam_count = 0
        ir_count = 0
    
    print(f"[Bridge] Using NAM models: {nam_folder} ({nam_count} models found)", file=sys.stderr)
    print(f"[Bridge] Using IR files: {ir_folder} ({ir_count} IRs found)", file=sys.stderr)

    # Run optimizer with correct asset paths
    print(f"[Bridge] Running Universal Optimization (duration={args.duration}s, "
          f"iterations={args.iterations})...")
    print(f"[Bridge] This may take 1-3 minutes...", flush=True)
    
    # #region agent log
    import json
    log_path = r"e:\Users\Desktop\toneMatchAi\.cursor\debug.log"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "location": "run_match.py:OPTIMIZER_START",
                "message": "Starting optimizer",
                "data": {
                    "di_samples": len(di_track.audio),
                    "ref_samples": len(ref_track.audio),
                    "sample_rate": di_track.sr,
                    "nam_folder": nam_folder,
                    "ir_folder": ir_folder
                },
                "timestamp": int(__import__("time").time() * 1000),
                "hypothesisId": "ALL"
            }) + "\n")
    except: pass
    # #endregion
    
    try:
        optimizer = ToneOptimizer(
            test_duration_sec=args.duration,
            max_iterations=args.iterations,
            nam_folder=nam_folder,
            ir_folder=ir_folder
        )
        result = optimizer.optimize_universal(di_track, ref_track, force_high_gain=args.force_high_gain)
    except Exception as e:
        error_msg = f"[ERROR] Optimization failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        print(error_msg, file=sys.stdout)
        import traceback
        full_traceback = traceback.format_exc()
        print(full_traceback, file=sys.stderr)
        print(full_traceback, file=sys.stdout)
        
        # Write error JSON so bridge can parse it
        try:
            error_output = {
                "rig": {
                    "fx_nam": "",
                    "amp_nam": "",
                    "ir": "",
                    "fx_nam_path": "",
                    "amp_nam_path": "",
                    "ir_path": ""
                },
                "params": {
                    "input_gain_db": 0.0,
                    "overdrive_db": 0.0,
                    "pre_eq_gain_db": 0.0,
                    "pre_eq_freq_hz": 800.0,
                    "reverb_wet": 0.0,
                    "reverb_room_size": 0.5,
                    "delay_time_ms": 100.0,
                    "delay_mix": 0.0,
                    "final_eq_gain_db": 0.0
                },
                "loss": 999.0,
                "error": error_msg
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(error_output, f, indent=2)
        except Exception:
            pass  # If we can't write error JSON, at least exit with code 1
        
        sys.exit(1)

    # Build output JSON matching the contract
    discovered_rig = result["discovered_rig"]
    best_params = result.get("best_parameters", {})

    # Convert relative paths to absolute paths
    def make_absolute(path):
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        # Try relative to project root
        abs_path = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(abs_path):
            return os.path.abspath(abs_path)
        # If not found, return original (might be valid relative path)
        return os.path.abspath(path) if os.path.exists(path) else path

    fx_nam_path = make_absolute(discovered_rig.get("fx_nam_path", ""))
    amp_nam_path = make_absolute(discovered_rig.get("amp_nam_path", ""))
    ir_path = make_absolute(discovered_rig.get("ir_path", ""))

    # Get input_gain_db from best_params or fallback to grid_search_results
    input_gain_db = float(best_params.get("input_gain_db",
                        result.get("grid_search_results", {})
                              .get("best_rig", {})
                              .get("input_gain_db", 0.0)))
    
    # input_gain_db is used as overdrive_db in the VST plugin
    # Add explicit overdrive_db field for clarity (duplicates input_gain_db)
    overdrive_db = input_gain_db

    output = {
        "rig": {
            "fx_nam":       discovered_rig.get("fx_nam_name", ""),
            "amp_nam":      discovered_rig.get("amp_nam_name", ""),
            "ir":           discovered_rig.get("ir_name", ""),
            "fx_nam_path":  fx_nam_path,
            "amp_nam_path": amp_nam_path,
            "ir_path":      ir_path
        },
        "params": {
            "input_gain_db":    input_gain_db,
            "overdrive_db":     overdrive_db,  # Explicit field for overdrive (duplicates input_gain_db)
            "pre_eq_gain_db":   float(best_params.get("pre_eq_gain_db", 0.0)),
            "pre_eq_freq_hz":   float(best_params.get("pre_eq_freq_hz", 800.0)),
            "reverb_wet":       float(best_params.get("reverb_wet", 0.0)),
            "reverb_room_size": float(best_params.get("reverb_room_size", 0.5)),
            "delay_time_ms":    float(best_params.get("delay_time_ms", 100.0)),
            "delay_mix":        float(best_params.get("delay_mix", 0.0)),
            "final_eq_gain_db": float(best_params.get("final_eq_gain_db", 0.0))
        },
        "loss": float(result.get("final_loss", 0.0))
    }

    # Write JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[Bridge] Result written to: {args.out}")
    print(f"[Bridge] Loss = {output['loss']:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    main()


