#!/usr/bin/env python3
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

# Add project root to sys.path so we can import src.*
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.io import load_audio_file
from src.core.optimizer import ToneOptimizer


def main():
    parser = argparse.ArgumentParser(description="ToneMatch AI — CLI optimizer bridge")
    parser.add_argument("--di",  required=True, help="Path to DI (dry input) audio file")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--out", required=True, help="Path to write the result JSON")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Test duration in seconds (default: 5.0)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Max optimizer iterations (default: 50)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.di):
        print(f"[ERROR] DI file not found: {args.di}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.ref):
        print(f"[ERROR] Reference file not found: {args.ref}", file=sys.stderr)
        sys.exit(1)

    # Load audio
    print(f"[Bridge] Loading DI: {args.di}")
    di_track = load_audio_file(args.di)

    print(f"[Bridge] Loading Reference: {args.ref}")
    ref_track = load_audio_file(args.ref)

    # Run optimizer
    print(f"[Bridge] Running Universal Optimization (duration={args.duration}s, "
          f"iterations={args.iterations})...")
    optimizer = ToneOptimizer(
        test_duration_sec=args.duration,
        max_iterations=args.iterations
    )
    result = optimizer.optimize_universal(di_track, ref_track)

    # Build output JSON matching the contract
    discovered_rig = result["discovered_rig"]
    best_params = result.get("best_parameters", {})

    output = {
        "rig": {
            "fx_nam":       discovered_rig.get("fx_nam_name", ""),
            "amp_nam":      discovered_rig.get("amp_nam_name", ""),
            "ir":           discovered_rig.get("ir_name", ""),
            "fx_nam_path":  discovered_rig.get("fx_nam_path", ""),
            "amp_nam_path": discovered_rig.get("amp_nam_path", ""),
            "ir_path":      discovered_rig.get("ir_path", "")
        },
        "params": {
            "input_gain_db":    float(best_params.get("input_gain_db",
                                    result.get("grid_search_results", {})
                                          .get("best_rig", {})
                                          .get("input_gain_db", 0.0))),
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


