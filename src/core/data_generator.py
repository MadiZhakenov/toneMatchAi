"""
Dataset generator for Post-FX Parameter Predictor training.

Generates training dataset by:
1. Taking DI fragments (real + synthetic)
2. Processing through "Golden Rig" (NAM/IR chain)
3. Optimizing Post-FX parameters using scipy.optimize.minimize with G-Loss
4. Saving (Reference Mel-Spectrogram, Optimal Post-FX Params, G-Loss) pairs
"""

import os
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import librosa
from scipy import optimize

from src.core.io import AudioTrack, load_audio_file
from src.core.processor import ToneProcessor
from src.core.analysis import calculate_g_loss
from src.core.optimizer import ToneOptimizer


def generate_synthetic_di_fragments(num_fragments: int, duration_sec: float = 2.0, sr: int = 44100) -> List[np.ndarray]:
    """Generate synthetic DI fragments (noise, sine waves, combinations).
    
    Args:
        num_fragments: Number of fragments to generate
        duration_sec: Duration of each fragment in seconds
        sr: Sample rate
    
    Returns:
        List of audio arrays
    """
    fragments = []
    samples = int(duration_sec * sr)
    
    for i in range(num_fragments):
        # Mix of different synthetic signals
        t = np.linspace(0, duration_sec, samples)
        
        if i % 3 == 0:
            # Pink noise (filtered white noise)
            noise = np.random.randn(samples).astype(np.float32)
            # Simple high-pass filter approximation
            noise = np.convolve(noise, [0.5, 0.3, 0.2], mode='same')
            fragment = noise * 0.3
        elif i % 3 == 1:
            # Sine wave with varying frequency
            freq = 80 + (i * 50) % 400  # 80-480 Hz
            fragment = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.5
        else:
            # Combination: sine + noise
            freq = 100 + (i * 30) % 300
            sine = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.3
            noise = np.random.randn(samples).astype(np.float32) * 0.2
            fragment = (sine + noise).astype(np.float32)
        
        # Normalize
        max_val = np.max(np.abs(fragment))
        if max_val > 0:
            fragment = fragment / max_val * 0.8
        
        fragments.append(fragment)
    
    return fragments


def extract_di_fragments(di_audio: np.ndarray, num_fragments: int, fragment_duration_sec: float = 2.0, sr: int = 44100) -> List[np.ndarray]:
    """Extract random fragments from real DI audio.
    
    Args:
        di_audio: Full DI audio array
        num_fragments: Number of fragments to extract
        fragment_duration_sec: Duration of each fragment
        sr: Sample rate
    
    Returns:
        List of audio fragments
    """
    fragments = []
    fragment_samples = int(fragment_duration_sec * sr)
    max_start = len(di_audio) - fragment_samples
    
    if max_start <= 0:
        # Audio too short, pad or repeat
        if len(di_audio) > 0:
            fragment = np.tile(di_audio, (fragment_samples // len(di_audio) + 1))[:fragment_samples]
        else:
            fragment = np.zeros(fragment_samples, dtype=np.float32)
        return [fragment] * num_fragments
    
    # Extract random fragments
    for _ in range(num_fragments):
        start_idx = np.random.randint(0, max_start)
        fragment = di_audio[start_idx:start_idx + fragment_samples].copy()
        fragments.append(fragment)
    
    return fragments


def generate_training_dataset(
    num_samples: int = 2000,
    golden_rig_config: Optional[Dict[str, str]] = None,
    di_audio_path: Optional[str] = None,
    output_path: str = "data/postfx_dataset.npz",
    test_duration_sec: float = 2.0,
    max_optimization_iterations: int = 30
) -> str:
    """Generate training dataset for Post-FX Parameter Predictor.
    
    For each DI fragment:
    1. Process through "Golden Rig" (NAM/IR) → rough matched audio
    2. Use scipy.optimize.minimize to find optimal Post-FX parameters (using G-Loss)
    3. Save: (Reference Mel-Spectrogram, Optimal Post-FX Params, G-Loss)
    
    Args:
        num_samples: Number of training samples to generate
        golden_rig_config: Dict with keys 'fx_nam_path', 'amp_nam_path', 'ir_path'
                          If None, uses default: DS1 -> 5150 BlockLetter -> BlendOfAll
        di_audio_path: Path to real DI audio file (my_guitar.wav). If None, uses only synthetic.
        output_path: Path to save dataset (.npz file)
        test_duration_sec: Duration of audio segments for optimization (default: 2.0 sec)
        max_optimization_iterations: Max iterations for scipy.optimize (default: 30, faster than 50)
    
    Returns:
        Path to saved dataset file
    
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If generation fails
    """
    print("=" * 70)
    print("Post-FX Dataset Generator")
    print("=" * 70)
    
    # Default golden rig configuration
    if golden_rig_config is None:
        golden_rig_config = {
            'fx_nam_path': 'assets/nam_models/Keith B DS1_g6_t5.nam',
            'amp_nam_path': 'assets/nam_models/Helga B 5150 BlockLetter - NoBoost.nam',
            'ir_path': 'assets/impulse_responses/BlendOfAll.wav'
        }
    
    # Validate golden rig files
    for key, path in golden_rig_config.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Golden rig file not found: {path}")
    
    print(f"\nGolden Rig Configuration:")
    print(f"  FX NAM: {os.path.basename(golden_rig_config['fx_nam_path'])}")
    print(f"  AMP NAM: {os.path.basename(golden_rig_config['amp_nam_path'])}")
    print(f"  IR: {os.path.basename(golden_rig_config['ir_path'])}")
    
    # Prepare DI fragments (mix of real and synthetic)
    print(f"\n[1/4] Preparing DI fragments...")
    di_fragments = []
    
    # Real DI fragments (50% of dataset)
    if di_audio_path and os.path.exists(di_audio_path):
        print(f"  Loading real DI from: {di_audio_path}")
        di_track = load_audio_file(di_audio_path)
        num_real = num_samples // 2
        real_fragments = extract_di_fragments(
            di_track.audio,
            num_real,
            fragment_duration_sec=test_duration_sec,
            sr=di_track.sr
        )
        di_fragments.extend(real_fragments)
        print(f"  [OK] Extracted {num_real} real DI fragments")
    else:
        print(f"  [SKIP] Real DI file not found, using only synthetic")
        num_real = 0
    
    # Synthetic DI fragments (remaining 50% or 100% if no real DI)
    num_synthetic = num_samples - len(di_fragments)
    if num_synthetic > 0:
        print(f"  Generating {num_synthetic} synthetic DI fragments...")
        synthetic_fragments = generate_synthetic_di_fragments(
            num_synthetic,
            duration_sec=test_duration_sec,
            sr=44100
        )
        di_fragments.extend(synthetic_fragments)
        print(f"  [OK] Generated {num_synthetic} synthetic fragments")
    
    print(f"  Total DI fragments: {len(di_fragments)}")
    
    # Initialize processor and optimizer
    processor = ToneProcessor()
    optimizer = ToneOptimizer(
        test_duration_sec=test_duration_sec,
        max_iterations=max_optimization_iterations
    )
    
    # Storage for dataset
    mel_spectrograms = []
    postfx_params = []
    g_losses = []
    
    print(f"\n[2/4] Processing fragments through Golden Rig and optimizing Post-FX...")
    # More realistic estimate: ~3-5 seconds per fragment (optimization + processing)
    estimated_minutes = len(di_fragments) * 4 / 60
    print(f"  This will take approximately {estimated_minutes:.1f} minutes ({estimated_minutes/60:.1f} hours)...")
    print(f"  Processing {len(di_fragments)} fragments with {max_optimization_iterations} iterations each")
    
    start_time = time.time()
    
    for i, di_fragment in enumerate(di_fragments):
        fragment_start_time = time.time()
        
        # Show progress immediately
        print(f"\r  [{i+1}/{len(di_fragments)}] Processing fragment {i+1}...", end='', flush=True)
        
        try:
            # Create AudioTrack from fragment
            di_track = AudioTrack(audio=di_fragment, sr=44100, name=f"di_fragment_{i}")
            
            # Step 1: Process through Golden Rig (without Post-FX) to get "rough matched" audio
            gain_params = {'input_gain_db': 0.0}
            post_fx_params_default = {
                'pre_eq_gain_db': 0.0,
                'pre_eq_freq_hz': 800.0,
                'reverb_wet': 0.0,
                'reverb_room_size': 0.5,
                'delay_time_ms': 100.0,
                'delay_mix': 0.0,
                'final_eq_gain_db': 0.0
            }
            
            rough_track = processor.process_with_custom_rig_and_post_fx(
                di_track,
                fx_nam_path=golden_rig_config['fx_nam_path'],
                amp_nam_path=golden_rig_config['amp_nam_path'],
                ir_path=golden_rig_config['ir_path'],
                gain_params=gain_params,
                post_fx_params=post_fx_params_default,
                ref_track=None  # No reference for rough processing
            )
            
            # Use rough matched audio as "reference" for optimization
            # (In real scenario, this would be the actual reference, but for dataset generation
            # we use the rough matched as target to learn the Post-FX mapping)
            ref_track = AudioTrack(
                audio=rough_track.audio,
                sr=rough_track.sr,
                name=f"ref_fragment_{i}"
            )
            
            # Step 2: Optimize Post-FX parameters using scipy.optimize.minimize with G-Loss
            # This is the key step - we find optimal parameters that minimize G-Loss
            
            # Parameter bounds (7 parameters: Pre-EQ + Post-FX)
            bounds = [
                (-12.0, 12.0),     # Pre-EQ Gain (dB)
                (400.0, 3000.0),   # Pre-EQ Frequency (Hz)
                (0.0, 0.7),        # Reverb Wet Level
                (0.0, 1.0),        # Reverb Room Size
                (50.0, 500.0),     # Delay Time (ms)
                (0.0, 0.5),        # Delay Feedback/Mix
                (-3.0, 3.0)        # Final Match EQ Gain (dB)
            ]
            
            # Initial guess (Pre-EQ: neutral gain=0, freq=800Hz)
            x0 = np.array([0.0, 800.0, 0.2, 0.5, 100.0, 0.2, 0.0])
            
            opt_iteration = [0]  # Counter for optimization iterations
            max_calls = max_optimization_iterations * 2  # Powell can call more than maxiter
            
            def objective_function(x: np.ndarray) -> float:
                """Objective: minimize G-Loss."""
                opt_iteration[0] += 1
                
                # Hard limit to prevent infinite loops
                if opt_iteration[0] > max_calls:
                    return float('inf')
                
                # Show progress during optimization (every 3 iterations for small max, every 5 for larger)
                show_interval = 3 if max_optimization_iterations <= 10 else 5
                if opt_iteration[0] % show_interval == 1 or opt_iteration[0] <= 3:
                    elapsed_opt = time.time() - fragment_start_time
                    print(f"\r  [{i+1}/{len(di_fragments)}] Optimizing... (call {opt_iteration[0]}, target: ~{max_optimization_iterations}, {elapsed_opt:.1f}s)", 
                          end='', flush=True)
                
                try:
                    post_fx_params = {
                        'pre_eq_gain_db': float(x[0]),
                        'pre_eq_freq_hz': float(x[1]),
                        'reverb_wet': float(x[2]),
                        'reverb_room_size': float(x[3]),
                        'delay_time_ms': float(x[4]),
                        'delay_mix': float(x[5]),
                        'final_eq_gain_db': float(x[6])
                    }
                    
                    # Process with Post-FX
                    processed_track = processor.process_with_custom_rig_and_post_fx(
                        di_track,
                        fx_nam_path=golden_rig_config['fx_nam_path'],
                        amp_nam_path=golden_rig_config['amp_nam_path'],
                        ir_path=golden_rig_config['ir_path'],
                        gain_params=gain_params,
                        post_fx_params=post_fx_params,
                        ref_track=ref_track
                    )
                    
                    # Calculate G-Loss
                    g_loss = calculate_g_loss(
                        processed_track.audio,
                        ref_track.audio,
                        sr=di_track.sr
                    )
                    
                    return g_loss
                    
                except Exception as e:
                    return float('inf')
            
            # Optimize with callback to show final status
            print(f"\r  [{i+1}/{len(di_fragments)}] Starting optimization...", end='', flush=True)
            
            result = optimize.minimize(
                objective_function,
                x0,
                method='Powell',  # Powell method works well for bounded optimization
                bounds=bounds,
                options={
                    'maxiter': max_optimization_iterations,
                    'maxfev': max_calls,  # Maximum function evaluations
                    'disp': False
                }
            )
            
            # Show completion
            opt_time = time.time() - fragment_start_time
            print(f"\r  [{i+1}/{len(di_fragments)}] Optimization done ({opt_time:.1f}s, {opt_iteration[0]} calls) → Extracting features...", 
                  end='', flush=True)
            
            optimal_params = result.x
            optimal_g_loss = float(result.fun)
            
            # Step 3: Extract Reference Mel-Spectrogram
            # Use the reference audio to create mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=ref_track.audio,
                sr=ref_track.sr,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1] range for neural network input
            mel_spec_normalized = (mel_spec_db + 80) / 80.0  # Typical range: -80 to 0 dB
            mel_spec_normalized = np.clip(mel_spec_normalized, 0.0, 1.0)
            
            # Store in dataset
            mel_spectrograms.append(mel_spec_normalized)
            postfx_params.append(optimal_params)
            g_losses.append(optimal_g_loss)
            
            # Show progress after each fragment
            fragment_time = time.time() - fragment_start_time
            elapsed = time.time() - start_time
            progress_pct = 100 * (i + 1) / len(di_fragments)
            
            if i > 0:
                remaining = (elapsed / (i + 1)) * (len(di_fragments) - i - 1)
                avg_time_per_sample = elapsed / (i + 1)
            else:
                remaining = estimated_minutes * 60
                avg_time_per_sample = fragment_time
            
            # Progress bar (simple text version)
            bar_length = 40
            filled = int(bar_length * progress_pct / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r  [{i+1}/{len(di_fragments)}] [{bar}] {progress_pct:.1f}% | "
                  f"Time: {fragment_time:.1f}s | Elapsed: {elapsed/60:.1f}m | "
                  f"Remaining: {remaining/60:.1f}m | G-Loss: {optimal_g_loss:.4f}", end='', flush=True)
            
        except Exception as e:
            print(f"\n  [ERROR] Failed to process fragment {i+1}: {e}")
            continue
    
    print(f"\n[3/4] Dataset generation complete!")
    print(f"  Successfully generated {len(mel_spectrograms)} samples")
    
    # Convert to numpy arrays
    # Pad mel-spectrograms to same length (use max length or fixed size)
    max_time_frames = max(mel.shape[1] for mel in mel_spectrograms) if mel_spectrograms else 128
    # Use fixed size for consistency (approximately 2 seconds at 512 hop_length)
    fixed_time_frames = int(test_duration_sec * 44100 / 512) + 1
    
    mel_array = np.zeros((len(mel_spectrograms), 128, fixed_time_frames), dtype=np.float32)
    for i, mel in enumerate(mel_spectrograms):
        if mel.shape[1] <= fixed_time_frames:
            mel_array[i, :, :mel.shape[1]] = mel
        else:
            mel_array[i, :, :] = mel[:, :fixed_time_frames]
    
    params_array = np.array(postfx_params, dtype=np.float32)
    g_loss_array = np.array(g_losses, dtype=np.float32)
    
    # Save dataset
    print(f"\n[4/4] Saving dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    np.savez_compressed(
        output_path,
        mel_spectrograms=mel_array,
        postfx_params=params_array,
        g_losses=g_loss_array,
        golden_rig_config=golden_rig_config,
        num_samples=len(mel_spectrograms),
        test_duration_sec=test_duration_sec,
        version='2.0'  # Updated to include Pre-EQ parameters (7 params total)
    )
    
    print(f"  [OK] Dataset saved!")
    print(f"  Shape: Mel={mel_array.shape}, Params={params_array.shape}, G-Loss={g_loss_array.shape}")
    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    return output_path

