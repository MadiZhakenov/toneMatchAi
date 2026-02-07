#!/usr/bin/env python3
"""
ToneMatch AI - Standalone Tone Matcher
Packaged version for distribution with VST3 plugin.

Usage:
    tone_matcher.exe --di path/to/di.wav --ref path/to/ref.wav --out result.json
"""

import argparse
import json
import os
import sys
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Get the directory where this script/exe is located
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Assets are in ../assets relative to Scripts folder
ASSETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "assets"))
NAM_MODELS_DIR = os.path.join(ASSETS_DIR, "nam_models")
IR_DIR = os.path.join(ASSETS_DIR, "impulse_responses")

# Add parent directory to path for imports when running as script
if not getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def load_audio(filepath, target_sr=44100):
    """Load audio file and return numpy array + sample rate."""
    import numpy as np
    
    try:
        import librosa
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return audio, sr
    except ImportError:
        pass
    
    try:
        import soundfile as sf
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            # Simple resampling
            import scipy.signal as signal
            audio = signal.resample(audio, int(len(audio) * target_sr / sr))
            sr = target_sr
        return audio, sr
    except ImportError:
        pass
    
    raise ImportError("Neither librosa nor soundfile available for audio loading")


def find_nam_files(nam_dir):
    """Find all .nam files in directory."""
    if not os.path.exists(nam_dir):
        return []
    return [os.path.join(nam_dir, f) for f in os.listdir(nam_dir) if f.endswith('.nam')]


def find_ir_files(ir_dir):
    """Find all .wav files in IR directory."""
    if not os.path.exists(ir_dir):
        return []
    return [os.path.join(ir_dir, f) for f in os.listdir(ir_dir) if f.endswith('.wav')]


def classify_nam_model(filepath):
    """Classify NAM model as FX (pedal) or AMP based on filename."""
    name = os.path.basename(filepath).lower()
    
    fx_keywords = ['ds1', 'ts9', 'od808', 'klone', 'plumes', 'rat', 'muff', 'fuzz', 
                   'drive', 'boost', 'pedal', 'stomp', 'dirt', 'overdrive', 'distortion']
    
    for kw in fx_keywords:
        if kw in name:
            return 'fx'
    return 'amp'


def simple_tone_match(di_audio, ref_audio, sr, nam_dir, ir_dir, max_models=10):
    """
    Simplified tone matching without heavy dependencies.
    Returns best rig configuration.
    """
    import numpy as np
    
    # Find available models
    nam_files = find_nam_files(nam_dir)
    ir_files = find_ir_files(ir_dir)
    
    if not nam_files:
        print(f"[Warning] No NAM models found in {nam_dir}")
    if not ir_files:
        print(f"[Warning] No IR files found in {ir_dir}")
    
    # Classify models
    fx_models = [f for f in nam_files if classify_nam_model(f) == 'fx'][:max_models]
    amp_models = [f for f in nam_files if classify_nam_model(f) == 'amp'][:max_models]
    
    # If no classification worked, use first models as amps
    if not amp_models and nam_files:
        amp_models = nam_files[:max_models]
    
    print(f"[ToneMatcher] Found {len(fx_models)} FX models, {len(amp_models)} AMP models, {len(ir_files)} IRs")
    
    # Select best models (simplified - just pick first available)
    best_fx = fx_models[0] if fx_models else None
    best_amp = amp_models[0] if amp_models else None
    best_ir = ir_files[0] if ir_files else None
    
    # Default parameters
    result = {
        "rig": {
            "fx_nam": os.path.basename(best_fx) if best_fx else "",
            "amp_nam": os.path.basename(best_amp) if best_amp else "",
            "ir": os.path.basename(best_ir) if best_ir else "",
            "fx_nam_path": best_fx or "",
            "amp_nam_path": best_amp or "",
            "ir_path": best_ir or ""
        },
        "params": {
            "input_gain_db": 0.0,
            "pre_eq_gain_db": 0.0,
            "pre_eq_freq_hz": 800.0,
            "reverb_wet": 0.1,
            "reverb_room_size": 0.5,
            "delay_time_ms": 100.0,
            "delay_mix": 0.0,
            "final_eq_gain_db": 0.0
        },
        "loss": 0.5
    }
    
    return result


def full_tone_match(di_audio, ref_audio, sr, nam_dir, ir_dir):
    """
    Full tone matching using the optimizer (if available).
    """
    try:
        from src.core.io import AudioTrack
        from src.core.optimizer import ToneOptimizer
        
        di_track = AudioTrack(audio=di_audio, sr=sr, name="di")
        ref_track = AudioTrack(audio=ref_audio, sr=sr, name="ref")
        
        optimizer = ToneOptimizer(
            test_duration_sec=3.0,
            max_iterations=20
        )
        
        result = optimizer.optimize_universal(di_track, ref_track)
        
        discovered_rig = result["discovered_rig"]
        best_params = result.get("best_parameters", {})
        
        return {
            "rig": {
                "fx_nam": discovered_rig.get("fx_nam_name", ""),
                "amp_nam": discovered_rig.get("amp_nam_name", ""),
                "ir": discovered_rig.get("ir_name", ""),
                "fx_nam_path": discovered_rig.get("fx_nam_path", ""),
                "amp_nam_path": discovered_rig.get("amp_nam_path", ""),
                "ir_path": discovered_rig.get("ir_path", "")
            },
            "params": {
                "input_gain_db": float(best_params.get("input_gain_db", 
                    result.get("grid_search_results", {}).get("best_rig", {}).get("input_gain_db", 0.0))),
                "pre_eq_gain_db": float(best_params.get("pre_eq_gain_db", 0.0)),
                "pre_eq_freq_hz": float(best_params.get("pre_eq_freq_hz", 800.0)),
                "reverb_wet": float(best_params.get("reverb_wet", 0.0)),
                "reverb_room_size": float(best_params.get("reverb_room_size", 0.5)),
                "delay_time_ms": float(best_params.get("delay_time_ms", 100.0)),
                "delay_mix": float(best_params.get("delay_mix", 0.0)),
                "final_eq_gain_db": float(best_params.get("final_eq_gain_db", 0.0))
            },
            "loss": float(result.get("final_loss", 0.0))
        }
        
    except ImportError as e:
        print(f"[ToneMatcher] Full optimizer not available: {e}")
        print("[ToneMatcher] Falling back to simple matching...")
        return None


def main():
    parser = argparse.ArgumentParser(description="ToneMatch AI - Standalone Tone Matcher")
    parser.add_argument("--di", required=True, help="Path to DI (dry input) audio file")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--out", required=True, help="Path to write the result JSON")
    parser.add_argument("--nam-dir", default=NAM_MODELS_DIR, help="Path to NAM models directory")
    parser.add_argument("--ir-dir", default=IR_DIR, help="Path to IR files directory")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.di):
        print(f"[ERROR] DI file not found: {args.di}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.ref):
        print(f"[ERROR] Reference file not found: {args.ref}", file=sys.stderr)
        sys.exit(1)

    print(f"[ToneMatcher] Loading DI: {args.di}")
    print(f"[ToneMatcher] Loading Reference: {args.ref}")
    print(f"[ToneMatcher] NAM models dir: {args.nam_dir}")
    print(f"[ToneMatcher] IR dir: {args.ir_dir}")

    # Load audio
    di_audio, sr = load_audio(args.di)
    ref_audio, _ = load_audio(args.ref, target_sr=sr)
    
    print(f"[ToneMatcher] Audio loaded. SR={sr}, DI length={len(di_audio)}, Ref length={len(ref_audio)}")

    # Try full optimizer first, fall back to simple
    result = full_tone_match(di_audio, ref_audio, sr, args.nam_dir, args.ir_dir)
    
    if result is None:
        result = simple_tone_match(di_audio, ref_audio, sr, args.nam_dir, args.ir_dir)

    # Write JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[ToneMatcher] Result written to: {args.out}")
    print(f"[ToneMatcher] Loss = {result['loss']:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    main()

