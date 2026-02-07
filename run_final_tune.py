"""
Run final Post-FX optimization with fixed NAM/IR chain.
Uses fixed equipment: DS1 -> 5150 BlockLetter -> BlendOfAll IR.
Optimizes only Post-FX parameters: Reverb, Delay, Final EQ Gain.
"""

import os
import sys
import numpy as np

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from src.core.io import load_audio_file, save_audio_file, AudioTrack
from src.core.optimizer import ToneOptimizer
from src.core.processor import ToneProcessor
from src.core.analysis import analyze_track, calculate_audio_distance


def main():
    """Main final Post-FX optimization pipeline."""
    print("=" * 70)
    print("Running final Post-FX optimization on the DS1 -> 5150 -> BlendOfAll Rig.")
    print("=" * 70)
    
    try:
        # Step 1: Load audio files
        print("\n[Step 1] Loading audio files...")
        
        di_path = None
        ref_path = None
        
        # Try to load demo files first
        if os.path.exists("assets/demo_di.wav") and os.path.exists("assets/demo_ref.wav"):
            di_path = "assets/demo_di.wav"
            ref_path = "assets/demo_ref.wav"
        elif os.path.exists("my_guitar.wav") and os.path.exists("reference.wav"):
            di_path = "my_guitar.wav"
            ref_path = "reference.wav"
        else:
            print("   [ERROR] Audio files not found!")
            print("   Please ensure 'assets/demo_di.wav' and 'assets/demo_ref.wav' exist,")
            print("   or 'my_guitar.wav' and 'reference.wav' exist.")
            return
        
        print(f"   Loading: {di_path} and {ref_path}")
        di_track = load_audio_file(di_path)
        ref_track = load_audio_file(ref_path)
        
        print(f"   DI: {len(di_track.audio)} samples, {len(di_track.audio)/di_track.sr:.2f} sec")
        print(f"   Ref: {len(ref_track.audio)} samples, {len(ref_track.audio)/ref_track.sr:.2f} sec")
        
        # Step 2: Analyze tracks
        print("\n[Step 2] Analyzing tracks...")
        di_features = analyze_track(di_track)
        ref_features = analyze_track(ref_track)
        
        print(f"   DI - Spectral Centroid: {di_features.spectral_centroid:.2f} Hz")
        print(f"   DI - RMS Energy: {di_features.rms_energy:.6f}")
        print(f"   Ref - Spectral Centroid: {ref_features.spectral_centroid:.2f} Hz")
        print(f"   Ref - RMS Energy: {ref_features.rms_energy:.6f}")
        
        # Step 3: Create optimizer and run Post-FX optimization
        print("\n[Step 3] Running Post-FX optimization...")
        print("   Fixed Equipment Chain: DS1 (FX) -> 5150 BlockLetter (AMP) -> BlendOfAll (IR)")
        print("   Using 'Sighted' Optimizer: minimizing 4-component error vector")
        print("   Optimizing 7 Post-FX parameters:")
        print("     - Pre-EQ Gain [-12.0 ... +12.0 dB]")
        print("     - Pre-EQ Frequency [400 ... 3000 Hz]")
        print("     - Reverb Wet Level [0.0 ... 0.7]")
        print("     - Reverb Room Size [0.0 ... 1.0]")
        print("     - Delay Time [50 ... 500 ms]")
        print("     - Delay Mix [0.0 ... 0.5]")
        print("     - Final EQ Gain [-3.0 ... +3.0 dB]")
        print("   This may take a few minutes...")
        
        # Hardcoded paths to NAM/IR files (same as process_final_tune)
        FX_NAM_PATH = "assets/nam_models/Keith B DS1_g6_t5.nam"
        AMP_NAM_PATH = "assets/nam_models/Helga B 5150 BlockLetter - NoBoost.nam"
        IR_PATH = "assets/impulse_responses/BlendOfAll.wav"
        
        # Verify files exist
        if not os.path.exists(FX_NAM_PATH):
            print(f"   [WARNING] FX NAM file not found: {FX_NAM_PATH}")
            FX_NAM_PATH = None
        if not os.path.exists(AMP_NAM_PATH):
            print(f"   [WARNING] AMP NAM file not found: {AMP_NAM_PATH}")
            AMP_NAM_PATH = None
        if not os.path.exists(IR_PATH):
            raise FileNotFoundError(f"IR file not found: {IR_PATH}")
        
        optimizer = ToneOptimizer(
            test_duration_sec=5.0,
            max_iterations=50
        )
        
        # Use optimize_post_fx_for_rig with fixed rig configuration (enables "Sighted" Optimizer)
        opt_result = optimizer.optimize_post_fx_for_rig(
            di_track=di_track,
            ref_track=ref_track,
            fx_nam_path=FX_NAM_PATH,
            amp_nam_path=AMP_NAM_PATH,
            ir_path=IR_PATH,
            input_gain_db=0.0,
            use_predictor=False  # Use scipy.optimize (Sighted Optimizer) instead of neural predictor
        )
        
        # Step 4: Display optimization results
        print("\n" + "=" * 70)
        print("[POST-FX OPTIMIZATION RESULTS]")
        print("=" * 70)
        
        print(f"\n  Success: {opt_result['success']}")
        print(f"  Final Loss: {opt_result['final_loss']:.6f}")
        print(f"  Total Iterations: {opt_result['iterations']}")
        print(f"  Message: {opt_result['message']}")
        
        # Display error components if available (from "Sighted" Optimizer)
        if 'initial_error_components' in opt_result and 'final_error_components' in opt_result:
            initial = opt_result['initial_error_components']
            final = opt_result['final_error_components']
            
            print(f"\n  Error Components Comparison (Sighted Optimizer):")
            print(f"    Harmonic Loss:      {initial['harmonic_loss']:.4f} -> {final['harmonic_loss']:.4f}")
            print(f"    Envelope Loss:      {initial['envelope_loss']:.4f} -> {final['envelope_loss']:.4f}")
            print(f"    Spectral Shape Loss: {initial['spectral_shape_loss']:.4f} -> {final['spectral_shape_loss']:.4f}")
            print(f"    Brightness Loss:    {initial['brightness_loss']:.4f} -> {final['brightness_loss']:.4f}")
        
        best_params = opt_result['best_parameters']
        print(f"\n  Best Post-FX Parameters:")
        if 'pre_eq_gain_db' in best_params:
            print(f"    Pre-EQ Gain: {best_params['pre_eq_gain_db']:.2f} dB @ {best_params['pre_eq_freq_hz']:.0f} Hz")
        print(f"    Reverb Wet Level: {best_params['reverb_wet']:.4f}")
        print(f"    Reverb Room Size: {best_params['reverb_room_size']:.4f}")
        print(f"    Delay Time: {best_params['delay_time_ms']:.2f} ms")
        print(f"    Delay Mix: {best_params['delay_mix']:.4f}")
        print(f"    Final EQ Gain: {best_params['final_eq_gain_db']:.2f} dB")
        
        # Step 5: Apply optimized parameters to full track
        print("\n[Step 4] Applying optimized Post-FX parameters to full track...")
        print("   Fixed Chain: DS1 -> 5150 BlockLetter -> BlendOfAll")
        print("   Applying optimized Post-FX: Delay -> Reverb -> Final Match EQ -> Normalization...")
        
        processor = ToneProcessor()
        gain_params = {
            'input_gain_db': 0.0  # Fixed input gain
        }
        
        # Extract Post-FX parameters (may include Pre-EQ if using optimize_post_fx_for_rig)
        post_fx_params = {
            'pre_eq_gain_db': best_params.get('pre_eq_gain_db', 0.0),
            'pre_eq_freq_hz': best_params.get('pre_eq_freq_hz', 800.0),
            'reverb_wet': best_params['reverb_wet'],
            'reverb_room_size': best_params['reverb_room_size'],
            'delay_time_ms': best_params['delay_time_ms'],
            'delay_mix': best_params['delay_mix'],
            'final_eq_gain_db': best_params['final_eq_gain_db']
        }
        
        final_track = processor.process_final_tune(
            di_track,
            gain_params=gain_params,
            post_fx_params=post_fx_params,
            ref_track=ref_track
        )
        
        print(f"   [OK] Processing complete")
        print(f"        Output: {len(final_track.audio)} samples")
        
        # Step 6: Analyze final result
        print("\n[Step 5] Analyzing final result...")
        final_features = analyze_track(final_track)
        
        # Calculate final distance on full track
        test_samples = min(len(final_track.audio), len(ref_track.audio))
        final_distance = calculate_audio_distance(
            final_track.audio[:test_samples],
            ref_track.audio[:test_samples],
            sr=di_track.sr
        )
        
        print(f"   Final - Spectral Centroid: {final_features.spectral_centroid:.2f} Hz")
        print(f"   Final - RMS Energy: {final_features.rms_energy:.6f}")
        print(f"   Final Distance (full track): {final_distance:.6f}")
        
        # Step 7: Comparison
        print("\n[Step 6] Results Comparison:")
        print(f"   DI (before):")
        print(f"     - Spectral Centroid: {di_features.spectral_centroid:.2f} Hz")
        print(f"     - RMS Energy: {di_features.rms_energy:.6f}")
        print(f"   Final (after):")
        print(f"     - Spectral Centroid: {final_features.spectral_centroid:.2f} Hz")
        print(f"     - RMS Energy: {final_features.rms_energy:.6f}")
        print(f"   Target (reference):")
        print(f"     - Spectral Centroid: {ref_features.spectral_centroid:.2f} Hz")
        print(f"     - RMS Energy: {ref_features.rms_energy:.6f}")
        
        centroid_improvement = final_features.spectral_centroid - di_features.spectral_centroid
        centroid_distance = ref_features.spectral_centroid - final_features.spectral_centroid
        
        print(f"\n   Centroid improvement: {centroid_improvement:+.2f} Hz")
        print(f"   Distance to target: {centroid_distance:+.2f} Hz")
        
        # Step 8: Save final audio
        print("\n[Step 7] Saving final audio...")
        output_file = "final_tuned_result.wav"
        save_audio_file(final_track, output_file)
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"   [OK] Saved as '{output_file}' ({file_size / 1024:.2f} KB)")
        else:
            print(f"   [ERROR] Output file was not created!")
            return
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Final Post-FX optimization complete!")
        print("=" * 70)
        print(f"\nFixed Equipment Chain:")
        print(f"  - FX NAM (Pedal): DS1 (Keith B DS1_g6_t5.nam)")
        print(f"  - AMP NAM (Amp): 5150 BlockLetter (Helga B 5150 BlockLetter - NoBoost.nam)")
        print(f"  - IR Cabinet: BlendOfAll.wav")
        print(f"\nOptimized Post-FX Parameters:")
        if 'pre_eq_gain_db' in best_params:
            print(f"  - Pre-EQ Gain: {best_params['pre_eq_gain_db']:.2f} dB @ {best_params['pre_eq_freq_hz']:.0f} Hz")
        print(f"  - Reverb Wet Level: {best_params['reverb_wet']:.4f}")
        print(f"  - Reverb Room Size: {best_params['reverb_room_size']:.4f}")
        print(f"  - Delay Time: {best_params['delay_time_ms']:.2f} ms")
        print(f"  - Delay Mix: {best_params['delay_mix']:.4f}")
        print(f"  - Final EQ Gain: {best_params['final_eq_gain_db']:.2f} dB")
        print(f"  - Final Loss: {opt_result['final_loss']:.6f}")
        
        # Display error components in final summary if available
        if 'initial_error_components' in opt_result and 'final_error_components' in opt_result:
            initial = opt_result['initial_error_components']
            final = opt_result['final_error_components']
            print(f"\nError Components (Sighted Optimizer):")
            print(f"  - Harmonic Loss:      {initial['harmonic_loss']:.4f} -> {final['harmonic_loss']:.4f}")
            print(f"  - Envelope Loss:      {initial['envelope_loss']:.4f} -> {final['envelope_loss']:.4f}")
            print(f"  - Spectral Shape Loss: {initial['spectral_shape_loss']:.4f} -> {final['spectral_shape_loss']:.4f}")
            print(f"  - Brightness Loss:    {initial['brightness_loss']:.4f} -> {final['brightness_loss']:.4f}")
        print(f"\nListen to '{output_file}' to hear the final result.")
        print("\nProcessing chain applied:")
        print("  1. Input Gain (fixed: 0.0 dB)")
        print("  2. FX NAM: DS1 (hardcoded)")
        print("  3. AMP NAM: 5150 BlockLetter (hardcoded)")
        print("  4. IR: BlendOfAll (hardcoded)")
        print("  5. Delay (optimized)")
        print("  6. Reverb (optimized)")
        print("  7. Final Match EQ with Gain adjustment (optimized)")
        print("  8. Normalization (-1.0 dB peak)")
        
        # Save final report to text file
        report_file = "final_tune_result.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Final Post-FX Optimization Results\n")
            f.write("=" * 70 + "\n\n")
            f.write("Fixed Equipment Chain:\n")
            f.write("  - FX NAM (Pedal): DS1 (Keith B DS1_g6_t5.nam)\n")
            f.write("  - AMP NAM (Amp): 5150 BlockLetter (Helga B 5150 BlockLetter - NoBoost.nam)\n")
            f.write("  - IR Cabinet: BlendOfAll.wav\n\n")
            f.write("Optimized Post-FX Parameters:\n")
            if 'pre_eq_gain_db' in best_params:
                f.write(f"  - Pre-EQ Gain: {best_params['pre_eq_gain_db']:.2f} dB @ {best_params['pre_eq_freq_hz']:.0f} Hz\n")
            f.write(f"  - Reverb Wet Level: {best_params['reverb_wet']:.4f}\n")
            f.write(f"  - Reverb Room Size: {best_params['reverb_room_size']:.4f}\n")
            f.write(f"  - Delay Time: {best_params['delay_time_ms']:.2f} ms\n")
            f.write(f"  - Delay Mix: {best_params['delay_mix']:.4f}\n")
            f.write(f"  - Final EQ Gain: {best_params['final_eq_gain_db']:.2f} dB\n")
            f.write(f"  - Final Loss: {opt_result['final_loss']:.6f}\n")
            f.write(f"  - Total Iterations: {opt_result['iterations']}\n")
            
            # Add error components to report if available
            if 'initial_error_components' in opt_result and 'final_error_components' in opt_result:
                initial = opt_result['initial_error_components']
                final = opt_result['final_error_components']
                f.write("\nError Components (Sighted Optimizer):\n")
                f.write(f"  - Harmonic Loss:      {initial['harmonic_loss']:.4f} -> {final['harmonic_loss']:.4f}\n")
                f.write(f"  - Envelope Loss:      {initial['envelope_loss']:.4f} -> {final['envelope_loss']:.4f}\n")
                f.write(f"  - Spectral Shape Loss: {initial['spectral_shape_loss']:.4f} -> {final['spectral_shape_loss']:.4f}\n")
                f.write(f"  - Brightness Loss:    {initial['brightness_loss']:.4f} -> {final['brightness_loss']:.4f}\n")
            
            f.write("\n")
            f.write("Processing Chain:\n")
            f.write("  1. Input Gain (fixed: 0.0 dB)\n")
            f.write("  2. FX NAM: DS1 (hardcoded)\n")
            f.write("  3. AMP NAM: 5150 BlockLetter (hardcoded)\n")
            f.write("  4. IR: BlendOfAll (hardcoded)\n")
            f.write("  5. Delay (optimized)\n")
            f.write("  6. Reverb (optimized)\n")
            f.write("  7. Final Match EQ with Gain adjustment (optimized)\n")
            f.write("  8. Normalization (-1.0 dB peak)\n")
        
        print(f"\n   [OK] Final report saved to '{report_file}'")
        
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease ensure:")
        print("  1. IR file 'assets/impulse_responses/BlendOfAll.wav' exists")
        print("  2. NAM files exist:")
        print("     - 'assets/nam_models/Keith B DS1_g6_t5.nam' (FX)")
        print("     - 'assets/nam_models/Helga B 5150 BlockLetter - NoBoost.nam' (AMP)")
        print("  3. Files have correct sample rate (44.1 kHz or 48 kHz)")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

