"""
Universal Tone Matching: Fast Grid Search + Deep Post-FX Optimization.
Finds the best equipment (FX_NAM, AMP_NAM, IR) and optimizes all Post-FX parameters.
"""

import os
import sys

# CRITICAL: Limit thread usage to prevent system lag
# Must be set BEFORE importing numpy/torch/librosa
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Limit PyTorch threads (must be after numpy import but before torch import)
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass  # torch not available, skip

from src.core.io import load_audio_file, save_audio_file, AudioTrack
from src.core.optimizer import ToneOptimizer


def main():
    """Main universal tone matching pipeline."""
    print("=" * 70)
    print("Universal Tone Matching: Fast Grid Search + Deep Post-FX Optimization")
    print("=" * 70)
    
    try:
        # Step 1: Load audio files
        print("\n[Step 1] Loading audio files...")
        
        di_path = None
        ref_path = None
        
        # Try to load audio files (supports both .wav and .mp3)
        # Priority: numbered files > base files (numbered are newer)
        if os.path.exists("my_guitar1.mp3") and os.path.exists("reference1.mp3"):
            di_path = "my_guitar1.mp3"
            ref_path = "reference1.mp3"
        elif os.path.exists("my_guitar.mp3") and os.path.exists("reference.mp3"):
            di_path = "my_guitar.mp3"
            ref_path = "reference.mp3"
        elif os.path.exists("my_guitar.wav") and os.path.exists("reference.wav"):
            di_path = "my_guitar.wav"
            ref_path = "reference.wav"
        elif os.path.exists("assets/demo_di.wav") and os.path.exists("assets/demo_ref.wav"):
            di_path = "assets/demo_di.wav"
            ref_path = "assets/demo_ref.wav"
        else:
            print("   [ERROR] Audio files not found!")
            print("   Please ensure one of the following pairs exists:")
            print("   - 'my_guitar1.mp3' and 'reference1.mp3'")
            print("   - 'my_guitar.mp3' and 'reference.mp3'")
            print("   - 'my_guitar.wav' and 'reference.wav'")
            return
        
        print(f"   Loading: {di_path} and {ref_path}")
        di_track = load_audio_file(di_path)
        ref_track = load_audio_file(ref_path)
        
        print(f"   DI: {len(di_track.audio)} samples, {len(di_track.audio)/di_track.sr:.2f} sec")
        print(f"   Ref: {len(ref_track.audio)} samples, {len(ref_track.audio)/ref_track.sr:.2f} sec")
        
        # Step 2: Universal optimization with Smart Sommelier
        print("\n[Step 2] Running Universal Optimization with 🍷 Smart Sommelier...")
        print("   Этапы 1-2: Интеллектуальный поиск оборудования из 261 NAM модели")
        print("              (анализ референса + двухуровневый Grid Search)")
        print("   Этап 3: Deep Post-FX Optimization - тонкая настройка")
        print("   This may take 3-5 minutes...")
        
        optimizer = ToneOptimizer(
            test_duration_sec=5.0,
            max_iterations=50
        )
        
        result = optimizer.optimize_universal(di_track, ref_track)
        
        # Step 3: Display results
        print("\n" + "=" * 70)
        print("[UNIVERSAL OPTIMIZATION RESULTS]")
        print("=" * 70)
        
        discovered_rig = result['discovered_rig']
        print(f"\n  Обнаруженный риг:")
        print(f"    FX (Pedal): {discovered_rig['fx_nam_name']}")
        print(f"    AMP: {discovered_rig['amp_nam_name']}")
        print(f"    IR Cabinet: {discovered_rig['ir_name']}")
        
        print(f"\n  Final Loss: {result['final_loss']:.6f}")
        
        # Display error components if available (from "Sighted" Optimizer)
        if 'post_fx_results' in result:
            post_fx_results = result['post_fx_results']
            if 'initial_error_components' in post_fx_results and 'final_error_components' in post_fx_results:
                initial = post_fx_results['initial_error_components']
                final = post_fx_results['final_error_components']
                
                print(f"\n  Error Components Comparison (Adam Optimizer):")
                print(f"    Harmonic Loss:      {initial['harmonic_loss']:.4f} -> {final['harmonic_loss']:.4f}")
                print(f"    Envelope Loss:      {initial['envelope_loss']:.4f} -> {final['envelope_loss']:.4f}")
                print(f"    Spectral Shape Loss: {initial['spectral_shape_loss']:.4f} -> {final['spectral_shape_loss']:.4f}")
                print(f"    Brightness Loss:    {initial['brightness_loss']:.4f} -> {final['brightness_loss']:.4f}")
        
        # Display optimized parameters
        best_params = result.get('best_parameters', {})
        if best_params:
            print(f"\n  Optimized Parameters:")
            if 'pre_eq_gain_db' in best_params:
                print(f"    Pre-EQ Gain: {best_params['pre_eq_gain_db']:.2f} dB @ {best_params['pre_eq_freq_hz']:.0f} Hz")
            print(f"    Reverb Wet Level: {best_params.get('reverb_wet', 0.0):.4f}")
            print(f"    Reverb Room Size: {best_params.get('reverb_room_size', 0.0):.4f}")
            print(f"    Delay Time: {best_params.get('delay_time_ms', 0.0):.2f} ms")
            print(f"    Delay Mix: {best_params.get('delay_mix', 0.0):.4f}")
            print(f"    Final EQ Gain: {best_params.get('final_eq_gain_db', 0.0):.2f} dB")
        
        # Step 4: Save result
        print("\n[Step 3] Saving final audio...")
        output_file = "universal_match_result.wav"
        save_audio_file(result['final_track'], output_file)
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"   [OK] Saved as '{output_file}' ({file_size / 1024:.2f} KB)")
        else:
            print(f"   [ERROR] Output file was not created!")
            return
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Universal tone matching complete!")
        print("=" * 70)
        print(f"\nОбнаруженный риг:")
        print(f"  - FX NAM (Pedal): {discovered_rig['fx_nam_name']}")
        print(f"  - AMP NAM (Amp): {discovered_rig['amp_nam_name']}")
        print(f"  - IR Cabinet: {discovered_rig['ir_name']}")
        print(f"\nListen to '{output_file}' to hear the final result.")
        
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease ensure:")
        print("  1. Audio files exist (my_guitar.wav and reference.wav)")
        print("  2. NAM models exist in assets/nam_models/")
        print("  3. IR files exist in assets/impulse_responses/")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

