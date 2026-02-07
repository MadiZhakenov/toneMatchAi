"""
Audio analysis module for extracting tonal features from audio tracks.
Implements Day 3: Core Analysis.
"""

from dataclasses import dataclass

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from src.core.io import AudioTrack


@dataclass
class ToneFeatures:
    """Represents tonal features extracted from an audio track.
    
    Attributes:
        spectrum: Averaged magnitude spectrum (frequency bins)
        frequencies: Frequency axis for mapping (Hz)
        rms_energy: Average RMS energy
        dynamic_range: Difference between peak and mean RMS
        spectral_centroid: Spectral centroid (brightness) in Hz
    """
    spectrum: np.ndarray
    frequencies: np.ndarray
    rms_energy: float
    dynamic_range: float
    spectral_centroid: float


def analyze_track(track: AudioTrack) -> ToneFeatures:
    """Analyze an audio track and extract tonal features.
    
    Args:
        track: AudioTrack object to analyze
    
    Returns:
        ToneFeatures object with extracted features
    
    Raises:
        ValueError: If audio is empty or invalid
    """
    if len(track.audio) == 0:
        raise ValueError("Cannot analyze empty audio track")
    
    # STFT parameters
    n_fft = 2048
    hop_length = 512
    
    # Compute Short-Time Fourier Transform
    stft = librosa.stft(track.audio, n_fft=n_fft, hop_length=hop_length)
    
    # Compute magnitude spectrum
    magnitude = np.abs(stft)
    
    # Average over time (axis=1 is time axis)
    spectrum = np.mean(magnitude, axis=1)
    
    # Apply spectral smoothing in dB domain for more natural frequency response
    # This reduces noise and creates smoother filter curves
    epsilon = 1e-10
    spectrum_db = 20 * np.log10(spectrum + epsilon)
    spectrum_db_smooth = ndimage.gaussian_filter1d(spectrum_db, sigma=3.0)
    spectrum = 10 ** (spectrum_db_smooth / 20)
    
    # Get frequency axis
    frequencies = librosa.fft_frequencies(sr=track.sr, n_fft=n_fft)
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=track.audio, frame_length=n_fft, hop_length=hop_length)[0]
    rms_energy = float(np.mean(rms))
    
    # Compute Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=track.audio, sr=track.sr)[0]
    spectral_centroid = float(np.mean(centroid))
    
    # Compute Dynamic Range (difference between peak and mean RMS)
    peak_rms = float(np.max(rms))
    mean_rms = float(np.mean(rms))
    dynamic_range = peak_rms - mean_rms
    
    return ToneFeatures(
        spectrum=spectrum,
        frequencies=frequencies,
        rms_energy=rms_energy,
        dynamic_range=dynamic_range,
        spectral_centroid=spectral_centroid
    )


def plot_comparison(ref_features: ToneFeatures, di_features: ToneFeatures) -> None:
    """Plot spectral comparison between reference and DI tracks.
    
    Creates a visualization with two spectrum curves on the same plot
    for debugging and visual inspection.
    
    Args:
        ref_features: ToneFeatures from reference track
        di_features: ToneFeatures from DI track
    """
    plt.figure(figsize=(12, 6))
    
    # Plot both spectra
    plt.semilogx(ref_features.frequencies, ref_features.spectrum, 
                 label='Reference', linewidth=2, alpha=0.8)
    plt.semilogx(di_features.frequencies, di_features.spectrum, 
                 label='DI', linewidth=2, alpha=0.8)
    
    # Labels and formatting
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.title('Spectral Comparison: Reference vs DI', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable frequency range (20 Hz to Nyquist)
    plt.xlim(20, ref_features.frequencies[-1])
    
    plt.tight_layout()


def calculate_drive_intensity(audio: np.ndarray, sr: int) -> float:
    """Calculate drive intensity based on Crest Factor.
    
    Crest Factor = Peak / RMS (in dB)
    - High Crest Factor (>10 dB) = Clean sound -> Low drive (0.0-0.2)
    - Low Crest Factor (<6 dB) = Compressed/High-gain -> High drive (0.8-1.0)
    
    Uses non-linear mapping to favor higher drive values.
    
    Args:
        audio: Audio samples array
        sr: Sample rate (not used, but kept for API consistency)
    
    Returns:
        Drive intensity value from 0.0 to 1.0
    """
    if len(audio) == 0:
        return 0.0
    
    # Calculate Peak and RMS
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0 or peak == 0:
        return 0.0
    
    # Crest Factor in dB
    crest_factor_db = 20 * np.log10(peak / rms)
    
    # Non-linear mapping: favor higher drive values
    # Crest Factor > 10 dB -> Drive 0.0-0.2 (clean)
    # Crest Factor < 6 dB -> Drive 0.8-1.0 (high-gain)
    # Use exponential curve for non-linear mapping
    
    if crest_factor_db > 10.0:
        # Clean sound: map to 0.0 - 0.2
        # Linear mapping from 10 dB to 20 dB -> 0.2 to 0.0
        intensity = max(0.0, 0.2 * (1.0 - (crest_factor_db - 10.0) / 10.0))
    elif crest_factor_db < 6.0:
        # High-gain/compressed: map to 0.8 - 1.0
        # Exponential curve: lower crest factor = higher drive
        # Map from 6 dB to 0 dB -> 0.8 to 1.0
        normalized = (6.0 - crest_factor_db) / 6.0  # 0 to 1
        intensity = 0.8 + 0.2 * (normalized ** 0.7)  # Non-linear, favors high values
    else:
        # Middle range: 6 dB to 10 dB -> 0.2 to 0.8
        # Linear interpolation
        normalized = (10.0 - crest_factor_db) / 4.0  # 0 to 1
        intensity = 0.2 + 0.6 * normalized
    
    # Clamp to [0.0, 1.0]
    return float(np.clip(intensity, 0.0, 1.0))


def calculate_distance(audio_a: np.ndarray, audio_b: np.ndarray, sr: int = 44100) -> float:
    """Calculate similarity distance between two audio signals.
    
    Uses MFCC (Mel-frequency cepstral coefficients) for timbre comparison
    and Spectral Flatness for distortion/noise comparison.
    
    Args:
        audio_a: First audio signal
        audio_b: Second audio signal
        sr: Sample rate (default: 44100)
    
    Returns:
        Combined distance metric (lower = more similar)
    """
    if len(audio_a) == 0 or len(audio_b) == 0:
        return float('inf')
    
    # Ensure both signals have the same length (take minimum)
    min_len = min(len(audio_a), len(audio_b))
    audio_a = audio_a[:min_len]
    audio_b = audio_b[:min_len]
    
    # 1. MFCC Distance (Timbre fingerprint)
    # Extract MFCC features (13 coefficients is standard)
    n_mfcc = 13
    mfcc_a = librosa.feature.mfcc(y=audio_a, sr=sr, n_mfcc=n_mfcc)
    mfcc_b = librosa.feature.mfcc(y=audio_b, sr=sr, n_mfcc=n_mfcc)
    
    # Average over time to get single vector per signal
    mfcc_a_mean = np.mean(mfcc_a, axis=1)
    mfcc_b_mean = np.mean(mfcc_b, axis=1)
    
    # Euclidean distance between MFCC vectors
    mfcc_distance = np.sqrt(np.sum((mfcc_a_mean - mfcc_b_mean) ** 2))
    
    # 2. Spectral Flatness Distance (Distortion/noise comparison)
    # Spectral Flatness = geometric_mean / arithmetic_mean of spectrum
    # Distorted signals have higher flatness (more noise-like)
    
    def spectral_flatness(audio: np.ndarray) -> float:
        """Calculate spectral flatness of audio signal."""
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Average over time
        spectrum = np.mean(magnitude, axis=1)
        
        # Avoid zeros
        spectrum = spectrum + 1e-10
        
        # Geometric mean
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        
        # Arithmetic mean
        arithmetic_mean = np.mean(spectrum)
        
        # Spectral flatness
        if arithmetic_mean == 0:
            return 0.0
        return geometric_mean / arithmetic_mean
    
    flatness_a = spectral_flatness(audio_a)
    flatness_b = spectral_flatness(audio_b)
    
    # Distance in flatness (absolute difference)
    flatness_distance = abs(flatness_a - flatness_b)
    
    # 3. Combined distance with weights
    # MFCC is more important for timbre matching (weight 0.7)
    # Flatness is important for distortion matching (weight 0.3)
    weight_mfcc = 0.7
    weight_flatness = 0.3
    
    # Normalize distances (MFCC typically 0-50, Flatness 0-1)
    # Normalize MFCC to similar scale as flatness
    mfcc_normalized = mfcc_distance / 50.0  # Rough normalization
    
    combined_distance = (weight_mfcc * mfcc_normalized) + (weight_flatness * flatness_distance)
    
    return float(combined_distance)


def calculate_loss(y_pred: np.ndarray, y_true: np.ndarray, sr: int = 44100) -> float:
    """Calculate multi-scale spectral loss between predicted and true audio.
    
    Uses a combination of:
    1. Log-Magnitude Spectrogram Loss (L1 distance)
    2. Spectral Convergence (peak alignment)
    3. Envelope Loss (RMS envelope difference)
    
    Args:
        y_pred: Predicted/processed audio signal
        y_true: True/reference audio signal
        sr: Sample rate (default: 44100)
    
    Returns:
        Combined loss value (lower = better match)
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return float('inf')
    
    # Ensure both signals have the same length
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    # STFT parameters
    n_fft = 2048
    hop_length = 512
    
    # 1. Log-Magnitude Spectrogram Loss (L1 distance)
    # Compute STFT for both signals
    stft_pred = librosa.stft(y_pred, n_fft=n_fft, hop_length=hop_length)
    stft_true = librosa.stft(y_true, n_fft=n_fft, hop_length=hop_length)
    
    # Magnitude spectrograms
    mag_pred = np.abs(stft_pred)
    mag_true = np.abs(stft_true)
    
    # Log-magnitude (add small epsilon to avoid log(0))
    epsilon = 1e-10
    log_mag_pred = np.log(mag_pred + epsilon)
    log_mag_true = np.log(mag_true + epsilon)
    
    # L1 distance in log-magnitude domain
    log_mag_loss = np.mean(np.abs(log_mag_pred - log_mag_true))
    
    # 2. Spectral Convergence
    # Measures how well spectral peaks align
    # Formula: ||X_true - X_pred||_F / ||X_true||_F
    numerator = np.linalg.norm(mag_true - mag_pred, ord='fro')
    denominator = np.linalg.norm(mag_true, ord='fro')
    
    if denominator == 0:
        spectral_convergence = 1.0
    else:
        spectral_convergence = numerator / denominator
    
    # 3. Envelope Loss (RMS envelope difference)
    # Compute RMS energy over time windows
    frame_length = n_fft
    rms_pred = librosa.feature.rms(y=y_pred, frame_length=frame_length, hop_length=hop_length)[0]
    rms_true = librosa.feature.rms(y=y_true, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Ensure same length
    min_rms_len = min(len(rms_pred), len(rms_true))
    rms_pred = rms_pred[:min_rms_len]
    rms_true = rms_true[:min_rms_len]
    
    # L1 distance between RMS envelopes
    envelope_loss = np.mean(np.abs(rms_pred - rms_true))
    
    # 4. Combined loss with weights
    # Log-magnitude is most important for spectral matching (weight 0.5)
    # Spectral convergence ensures peak alignment (weight 0.3)
    # Envelope loss ensures dynamic matching (weight 0.2)
    weight_log_mag = 0.5
    weight_spectral_conv = 0.3
    weight_envelope = 0.2
    
    # Normalize components to similar scales
    # Log-mag loss is typically 0-5, spectral conv 0-1, envelope 0-0.5
    log_mag_normalized = log_mag_loss / 5.0
    envelope_normalized = envelope_loss / 0.5
    
    combined_loss = (
        weight_log_mag * log_mag_normalized +
        weight_spectral_conv * spectral_convergence +
        weight_envelope * envelope_normalized
    )
    
    return float(combined_loss)


def calculate_audio_distance(y_pred: np.ndarray, y_true: np.ndarray, sr: int = 44100) -> float:
    """Calculate audio distance using Mel-spectrogram and Mean Absolute Error.
    
    Converts both signals to Mel-spectrograms, converts to logarithmic scale,
    and computes Mean Absolute Error (MAE) between them.
    Lower value = better match.
    
    Args:
        y_pred: Predicted/processed audio signal
        y_true: True/reference audio signal
        sr: Sample rate (default: 44100)
    
    Returns:
        MAE distance value (lower = better match)
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return float('inf')
    
    # Ensure both signals have the same length
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    # Mel-spectrogram parameters
    n_mels = 128
    hop_length = 512
    n_fft = 2048
    
    # Compute Mel-spectrograms for both signals
    mel_pred = librosa.feature.melspectrogram(
        y=y_pred,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft
    )
    mel_true = librosa.feature.melspectrogram(
        y=y_true,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    # Convert to logarithmic scale (dB)
    mel_pred_db = librosa.power_to_db(mel_pred, ref=np.max)
    mel_true_db = librosa.power_to_db(mel_true, ref=np.max)
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(mel_pred_db - mel_true_db))
    
    return float(mae)


def calculate_spectral_flux(audio: np.ndarray, sr: int = 44100) -> float:
    """Calculate Spectral Flux - measures temporal changes in spectrum.
    
    Spectral Flux quantifies how much the spectrum changes over time.
    Higher flux indicates more dynamic spectral content.
    
    Args:
        audio: Audio signal array
        sr: Sample rate (default: 44100)
    
    Returns:
        Normalized spectral flux value (0.0 to ~1.0)
    """
    if len(audio) == 0:
        return 0.0
    
    # STFT parameters
    n_fft = 2048
    hop_length = 512
    
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Normalize magnitude spectrogram
    # Convert to dB and normalize
    epsilon = 1e-10
    magnitude_db = librosa.power_to_db(magnitude ** 2 + epsilon, ref=np.max)
    
    # Calculate spectral flux: difference between consecutive frames
    # Flux = sum of squared differences between consecutive frames
    flux = np.sum(np.diff(magnitude_db, axis=1) ** 2, axis=0)
    
    # Normalize: take mean and scale to reasonable range
    # Typical flux values are in range 0-1000, normalize to 0-1
    mean_flux = np.mean(flux)
    normalized_flux = mean_flux / 1000.0  # Rough normalization
    
    return float(np.clip(normalized_flux, 0.0, 1.0))


def calculate_harmonic_ratio(audio: np.ndarray, sr: int = 44100) -> float:
    """Calculate Odd-to-Even Harmonic Ratio - key metric for "warmth" of sound.
    
    This ratio measures the balance between odd and even harmonics.
    Higher ratio (more odd harmonics) = warmer, richer sound.
    Lower ratio (more even harmonics) = brighter, harsher sound.
    
    Args:
        audio: Audio signal array
        sr: Sample rate (default: 44100)
    
    Returns:
        Harmonic ratio: sum(odd_harmonics) / sum(even_harmonics)
        Returns 1.0 if no harmonics found or division by zero
    """
    if len(audio) == 0:
        return 1.0
    
    # Find fundamental frequency using autocorrelation
    # Use a reasonable range: 80 Hz to 400 Hz (typical guitar range)
    min_period = int(sr / 400.0)  # Maximum frequency
    max_period = int(sr / 80.0)   # Minimum frequency
    
    if max_period >= len(audio) // 2:
        max_period = len(audio) // 2 - 1
    
    if min_period >= max_period:
        # Fallback: use default fundamental
        fundamental_hz = 220.0  # A3 note
    else:
        # Autocorrelation to find fundamental
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        
        # Find peak in the range
        search_range = autocorr[min_period:max_period]
        if len(search_range) > 0:
            peak_idx = np.argmax(search_range) + min_period
            fundamental_hz = sr / peak_idx if peak_idx > 0 else 220.0
        else:
            fundamental_hz = 220.0
    
    # Ensure fundamental is in reasonable range
    fundamental_hz = np.clip(fundamental_hz, 80.0, 400.0)
    
    # Compute FFT to get frequency spectrum
    n_fft = 8192  # Higher resolution for better harmonic detection
    fft = np.fft.rfft(audio[:min(len(audio), sr)])  # Use 1 second max
    magnitude = np.abs(fft)
    frequencies = np.fft.rfftfreq(len(audio[:min(len(audio), sr)]), 1.0 / sr)
    
    # Find harmonic peaks (up to 10th harmonic)
    num_harmonics = 10
    odd_harmonic_energy = 0.0
    even_harmonic_energy = 0.0
    
    for h in range(1, num_harmonics + 1):
        harmonic_freq = fundamental_hz * h
        
        # Find closest frequency bin
        freq_idx = np.argmin(np.abs(frequencies - harmonic_freq))
        
        if freq_idx < len(magnitude):
            harmonic_energy = magnitude[freq_idx] ** 2
            
            if h % 2 == 1:  # Odd harmonic
                odd_harmonic_energy += harmonic_energy
            else:  # Even harmonic
                even_harmonic_energy += harmonic_energy
    
    # Calculate ratio
    if even_harmonic_energy > 1e-10:
        ratio = odd_harmonic_energy / even_harmonic_energy
    else:
        # Avoid division by zero
        ratio = 1.0 if odd_harmonic_energy > 1e-10 else 1.0
    
    # Clamp to reasonable range (typically 0.5 to 3.0)
    return float(np.clip(ratio, 0.1, 10.0))


def extract_harmonics(audio: np.ndarray, sr: int = 44100, num_harmonics: int = 10) -> dict:
    """Extract fundamental frequency (f0) and harmonic amplitudes from audio.
    
    Uses librosa.pyin for f0 estimation and STFT for extracting harmonic amplitudes.
    This function is used to assess "warmth" of the sound through harmonic content.
    
    Optimized version with vectorized operations and efficient parameter settings.
    
    Args:
        audio: Audio signal array
        sr: Sample rate (default: 44100)
        num_harmonics: Number of harmonics to extract (default: 10)
    
    Returns:
        Dictionary with keys:
            - 'f0': Fundamental frequency (Hz), averaged over time
            - 'harmonics': Array of harmonic amplitudes (magnitude), shape (num_harmonics,)
            - 'f0_confidence': Average confidence of f0 estimation (0-1)
    """
    if len(audio) == 0:
        return {
            'f0': 220.0,  # Default A3
            'harmonics': np.zeros(num_harmonics, dtype=np.float32),
            'f0_confidence': 0.0
        }
    
    # STFT parameters
    n_fft = 2048
    hop_length = 512
    
    # 1. Estimate fundamental frequency using pyin
    # Using standard parameters compatible with all librosa versions
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz (lowest guitar note)
        fmax=librosa.note_to_hz('C6'),  # ~1047 Hz (high guitar note)
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length
    )
    
    # Average f0 over voiced frames only (vectorized)
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        avg_f0 = float(np.mean(voiced_f0))
        avg_confidence = float(np.mean(voiced_probs[voiced_flag]))
    else:
        # Fallback: use autocorrelation method (optimized)
        min_period = int(sr / 400.0)
        max_period = int(sr / 80.0)
        if max_period >= len(audio) // 2:
            max_period = len(audio) // 2 - 1
        if min_period < max_period:
            # Use shorter segment for autocorrelation (first 1 second max)
            autocorr_len = min(len(audio), sr)
            autocorr = np.correlate(audio[:autocorr_len], audio[:autocorr_len], mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            search_range = autocorr[min_period:max_period]
            if len(search_range) > 0:
                peak_idx = np.argmax(search_range) + min_period
                avg_f0 = float(sr / peak_idx) if peak_idx > 0 else 220.0
            else:
                avg_f0 = 220.0
        else:
            avg_f0 = 220.0
        avg_confidence = 0.5  # Medium confidence for fallback
    
    # Clamp f0 to reasonable range
    avg_f0 = np.clip(avg_f0, 80.0, 400.0)
    
    # 2. Extract harmonic amplitudes using STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Average magnitude over time (vectorized)
    avg_magnitude = np.mean(magnitude, axis=1)
    
    # Get frequency bins (cached computation)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Extract harmonic amplitudes (fully vectorized)
    harmonic_freqs = avg_f0 * np.arange(1, num_harmonics + 1, dtype=np.float32)
    
    # Vectorized search for closest frequency bins
    # Use searchsorted for faster lookup (frequencies are sorted)
    freq_indices = np.searchsorted(frequencies, harmonic_freqs, side='left')
    
    # Handle edge cases: clamp indices to valid range
    freq_indices = np.clip(freq_indices, 0, len(avg_magnitude) - 1)
    
    # Vectorized check: compare with left neighbor if exists
    # Create mask for indices that can check left neighbor
    can_check_left = (freq_indices > 0) & (freq_indices < len(frequencies) - 1)
    
    # Calculate distances to current and left bins (vectorized)
    if np.any(can_check_left):
        dist_to_current = np.abs(frequencies[freq_indices] - harmonic_freqs)
        dist_to_left = np.abs(frequencies[freq_indices - 1] - harmonic_freqs)
        
        # Where left is closer, use left index
        use_left = can_check_left & (dist_to_left < dist_to_current)
        freq_indices[use_left] = freq_indices[use_left] - 1
    
    # Extract amplitudes (vectorized)
    harmonic_amplitudes = avg_magnitude[freq_indices].astype(np.float32)
    
    return {
        'f0': float(avg_f0),
        'harmonics': harmonic_amplitudes,
        'f0_confidence': float(avg_confidence)
    }


def calculate_harmonic_warmth(ref_harmonics: dict, gen_harmonics: dict) -> float:
    """Calculate Harmonic Warmth Loss based on Even/Odd harmonic ratio.
    
    Tube amplifiers produce more even harmonics, resulting in warmer sound.
    This function compares the Even/Odd ratio between reference and generated audio.
    
    Optimized with vectorized operations.
    
    Args:
        ref_harmonics: Harmonic data from reference audio (from extract_harmonics)
        gen_harmonics: Harmonic data from generated audio (from extract_harmonics)
    
    Returns:
        Harmonic Warmth Loss (L1 distance between Even/Odd ratios)
    """
    # Extract harmonic amplitudes
    ref_harm = ref_harmonics['harmonics']
    gen_harm = gen_harmonics['harmonics']
    
    # Ensure same length
    min_len = min(len(ref_harm), len(gen_harm))
    ref_harm = ref_harm[:min_len]
    gen_harm = gen_harm[:min_len]
    
    # Calculate Even/Odd ratio (vectorized)
    # Even harmonics: 2nd, 4th, 6th, 8th, 10th (indices 1, 3, 5, 7, 9)
    # Odd harmonics: 1st, 3rd, 5th, 7th, 9th (indices 0, 2, 4, 6, 8)
    
    # Sum of even and odd harmonic amplitudes (vectorized)
    ref_even = np.sum(ref_harm[1::2])  # Start at index 1, step 2
    ref_odd = np.sum(ref_harm[::2])    # Start at index 0, step 2
    
    gen_even = np.sum(gen_harm[1::2])
    gen_odd = np.sum(gen_harm[::2])
    
    # Calculate Even/Odd ratio (avoid division by zero)
    epsilon = 1e-10
    ref_ratio = ref_even / ref_odd if ref_odd > epsilon else 1.0
    gen_ratio = gen_even / gen_odd if gen_odd > epsilon else 1.0
    
    # L1 distance between ratios
    ratio_diff = abs(float(ref_ratio) - float(gen_ratio))
    
    # Normalize to 0-1 range (ratios typically 0.1-2.0, so max diff ~2.0)
    normalized_loss = ratio_diff / 2.0
    
    return float(np.clip(normalized_loss, 0.0, 1.0))


def calculate_envelope_loss(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 44100) -> float:
    """Calculate Envelope Dynamics Loss by comparing RMS envelopes.
    
    Compares the attack (first 50ms) and sustain portions of the RMS envelope
    between reference and generated audio. This captures dynamic characteristics
    like how quickly the sound attacks and how it sustains.
    
    Args:
        ref_audio: Reference audio signal
        gen_audio: Generated audio signal
        sr: Sample rate (default: 44100)
    
    Returns:
        Envelope Dynamics Loss (MSE between attack and sustain envelopes)
    """
    if len(ref_audio) == 0 or len(gen_audio) == 0:
        return 1.0  # Maximum loss for empty audio
    
    # Ensure both signals have the same length
    min_len = min(len(ref_audio), len(gen_audio))
    ref_audio = ref_audio[:min_len]
    gen_audio = gen_audio[:min_len]
    
    # RMS parameters: short window for detailed envelope (5-10ms)
    # Using 7.5ms as compromise (approximately 331 samples at 44.1kHz)
    frame_length = int(0.0075 * sr)  # 7.5ms
    if frame_length < 64:
        frame_length = 64  # Minimum frame length
    if frame_length % 2 == 0:
        frame_length += 1  # Make odd for better windowing
    
    hop_length = frame_length // 4  # 25% overlap
    
    # Compute RMS envelopes (optimized: compute both at once if possible)
    ref_rms = librosa.feature.rms(
        y=ref_audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    gen_rms = librosa.feature.rms(
        y=gen_audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Ensure same length (vectorized)
    min_rms_len = min(len(ref_rms), len(gen_rms))
    if min_rms_len == 0:
        return 1.0
    
    ref_rms = ref_rms[:min_rms_len]
    gen_rms = gen_rms[:min_rms_len]
    
    # Calculate attack duration (first 50ms)
    attack_duration_ms = 50.0
    attack_samples = int(attack_duration_ms * sr / (hop_length * 1000.0))
    attack_samples = min(attack_samples, min_rms_len // 2)  # At most half the signal
    attack_samples = max(attack_samples, 1)  # At least 1 sample
    
    # Split into attack and sustain (vectorized slicing)
    ref_attack = ref_rms[:attack_samples]
    ref_sustain = ref_rms[attack_samples:]
    
    gen_attack = gen_rms[:attack_samples]
    gen_sustain = gen_rms[attack_samples:]
    
    # Normalize envelopes to [0, 1] for fair comparison (vectorized)
    # Use max of both signals for normalization
    max_both = max(np.max(ref_rms), np.max(gen_rms)) if len(ref_rms) > 0 and len(gen_rms) > 0 else 1.0
    
    if max_both > 1e-10:
        # Vectorized normalization
        ref_attack_norm = ref_attack / max_both
        ref_sustain_norm = ref_sustain / max_both
        gen_attack_norm = gen_attack / max_both
        gen_sustain_norm = gen_sustain / max_both
        
        # Calculate MSE for attack and sustain (vectorized)
        attack_mse = np.mean((ref_attack_norm - gen_attack_norm) ** 2) if len(ref_attack_norm) > 0 else 0.0
        sustain_mse = np.mean((ref_sustain_norm - gen_sustain_norm) ** 2) if len(ref_sustain_norm) > 0 else 0.0
    else:
        # All zeros, return zero loss
        return 0.0
    
    # Combined envelope loss (weighted: attack is more important for dynamics)
    envelope_loss = 0.6 * attack_mse + 0.4 * sustain_mse
    
    # Normalize to 0-1 range (MSE typically 0-1, but can be higher)
    normalized_loss = min(envelope_loss, 1.0)
    
    return float(normalized_loss)


def calculate_g_loss_v1(y_pred: np.ndarray, y_true: np.ndarray, sr: int = 44100) -> float:
    """Calculate G-Loss V1 (Legacy) - comprehensive loss function for tone matching.
    
    Original version combining:
    1. Mel-Spectrogram Loss (50%): Overall spectral similarity
    2. Spectral Flux Loss (30%): Temporal spectral dynamics
    3. Harmonic Ratio Loss (20%): Harmonic content balance (warmth)
    
    Formula: G-Loss = 0.5 * Mel_Loss + 0.3 * Flux_Loss + 0.2 * Harmonic_Loss
    
    Args:
        y_pred: Predicted/processed audio signal
        y_true: True/reference audio signal
        sr: Sample rate (default: 44100)
    
    Returns:
        Combined G-Loss value (lower = better match)
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return float('inf')
    
    # Ensure both signals have the same length
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    # 1. Mel-Spectrogram Loss (weight: 0.5)
    mel_loss = calculate_audio_distance(y_pred, y_true, sr=sr)
    
    # Normalize mel loss (typically 0-20, normalize to 0-1 scale)
    mel_loss_normalized = mel_loss / 20.0
    
    # 2. Spectral Flux Loss (weight: 0.3)
    flux_pred = calculate_spectral_flux(y_pred, sr=sr)
    flux_true = calculate_spectral_flux(y_true, sr=sr)
    flux_loss = abs(flux_pred - flux_true)
    
    # Flux loss is already in 0-1 range
    
    # 3. Harmonic Ratio Loss (weight: 0.2)
    ratio_pred = calculate_harmonic_ratio(y_pred, sr=sr)
    ratio_true = calculate_harmonic_ratio(y_true, sr=sr)
    
    # Normalize ratio difference (ratios typically 0.5-3.0, so max diff ~2.5)
    ratio_diff = abs(ratio_pred - ratio_true)
    ratio_loss_normalized = ratio_diff / 2.5
    
    # Combined G-Loss
    g_loss = (
        0.5 * mel_loss_normalized +
        0.3 * flux_loss +
        0.2 * ratio_loss_normalized
    )
    
    return float(g_loss)


def calculate_g_loss(y_pred: np.ndarray, y_true: np.ndarray, sr: int = 44100) -> float:
    """Calculate G-Loss (Perceptual-Physical Loss Fusion) - advanced loss function for tone matching.
    
    Sprint 1: Perceptual Loss Revolution
    Combines three key perceptual and physical metrics:
    1. L_mel (50%): Mel-spectrogram Distance - overall spectral similarity
    2. L_harm (30%): Harmonic Warmth Loss - Even/Odd harmonic ratio comparison
    3. L_dyn (20%): Envelope Dynamics Loss - attack and sustain envelope comparison
    
    Formula: G-Loss = 0.5 * L_mel + 0.3 * L_harm + 0.2 * L_dyn
    
    This loss function provides a more comprehensive assessment of tone matching,
    considering not only spectral similarity but also harmonic warmth (tube amp character)
    and dynamic response (attack and sustain characteristics).
    
    Args:
        y_pred: Predicted/processed audio signal
        y_true: True/reference audio signal
        sr: Sample rate (default: 44100)
    
    Returns:
        Combined G-Loss value (lower = better match)
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return float('inf')
    
    # Ensure both signals have the same length
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    # 1. L_mel: Mel-spectrogram Distance (weight: 0.5)
    mel_loss = calculate_audio_distance(y_pred, y_true, sr=sr)
    
    # Normalize mel loss (typically 0-20, normalize to 0-1 scale)
    mel_loss_normalized = mel_loss / 20.0
    
    # 2. L_harm: Harmonic Warmth Loss (weight: 0.3)
    # Extract harmonics for both signals
    ref_harmonics = extract_harmonics(y_true, sr=sr, num_harmonics=10)
    gen_harmonics = extract_harmonics(y_pred, sr=sr, num_harmonics=10)
    
    # Calculate harmonic warmth loss (Even/Odd ratio difference)
    harmonic_warmth_loss = calculate_harmonic_warmth(ref_harmonics, gen_harmonics)
    
    # 3. L_dyn: Envelope Dynamics Loss (weight: 0.2)
    envelope_dynamics_loss = calculate_envelope_loss(y_true, y_pred, sr=sr)
    
    # Combined G-Loss
    g_loss = (
        0.5 * mel_loss_normalized +
        0.3 * harmonic_warmth_loss +
        0.2 * envelope_dynamics_loss
    )
    
    return float(g_loss)


def calculate_spectral_shape_loss(y_pred: np.ndarray, y_true: np.ndarray, sr: int = 44100) -> float:
    """Calculate Spectral Shape Loss using MSE on smoothed Mel-Spectrogram.
    
    This loss measures the difference in overall timbre/EQ by comparing
    smoothed Mel-spectrograms. Smoothing reduces noise and focuses on
    the general spectral shape rather than fine details.
    
    Args:
        y_pred: Predicted/processed audio signal
        y_true: True/reference audio signal
        sr: Sample rate (default: 44100)
    
    Returns:
        Spectral Shape Loss (MSE between smoothed Mel-spectrograms, normalized to 0-1 range)
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return 1.0  # Maximum loss for empty audio
    
    # Ensure both signals have the same length
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    # Mel-spectrogram parameters
    n_mels = 128
    hop_length = 512
    n_fft = 2048
    
    # Compute Mel-spectrograms for both signals
    mel_pred = librosa.feature.melspectrogram(
        y=y_pred,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft
    )
    mel_true = librosa.feature.melspectrogram(
        y=y_true,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    # Convert to logarithmic scale (dB)
    epsilon = 1e-10
    mel_pred_db = librosa.power_to_db(mel_pred + epsilon, ref=np.max)
    mel_true_db = librosa.power_to_db(mel_true + epsilon, ref=np.max)
    
    # Apply Gaussian smoothing to focus on overall spectral shape
    # Smooth along frequency axis (axis=0) with sigma=2.0
    mel_pred_smooth = ndimage.gaussian_filter1d(mel_pred_db, sigma=2.0, axis=0)
    mel_true_smooth = ndimage.gaussian_filter1d(mel_true_db, sigma=2.0, axis=0)
    
    # Average over time to get single spectral shape vector
    mel_pred_avg = np.mean(mel_pred_smooth, axis=1)
    mel_true_avg = np.mean(mel_true_smooth, axis=1)
    
    # Calculate MSE between smoothed spectral shapes
    mse = np.mean((mel_pred_avg - mel_true_avg) ** 2)
    
    # Normalize to 0-1 range (MSE in dB domain typically 0-100, normalize by 100)
    normalized_loss = mse / 100.0
    
    return float(np.clip(normalized_loss, 0.0, 1.0))


def calculate_brightness_loss(y_pred: np.ndarray, y_true: np.ndarray, sr: int = 44100) -> float:
    """Calculate Brightness Loss using Spectral Centroid difference.
    
    Spectral Centroid measures the "brightness" of sound - higher values
    indicate more high-frequency content. This loss compares the brightness
    between predicted and reference audio.
    
    Args:
        y_pred: Predicted/processed audio signal
        y_true: True/reference audio signal
        sr: Sample rate (default: 44100)
    
    Returns:
        Brightness Loss (normalized difference in Spectral Centroid, 0-1 range)
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return 1.0  # Maximum loss for empty audio
    
    # Ensure both signals have the same length
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    # Compute Spectral Centroid for both signals
    centroid_pred = librosa.feature.spectral_centroid(y=y_pred, sr=sr)[0]
    centroid_true = librosa.feature.spectral_centroid(y=y_true, sr=sr)[0]
    
    # Average over time
    avg_centroid_pred = float(np.mean(centroid_pred))
    avg_centroid_true = float(np.mean(centroid_true))
    
    # Calculate absolute difference
    centroid_diff = abs(avg_centroid_pred - avg_centroid_true)
    
    # Normalize to 0-1 range
    # Spectral Centroid typically ranges from ~500 Hz (dark) to ~5000 Hz (bright)
    # Normalize by maximum expected difference (4500 Hz)
    max_expected_diff = 4500.0
    normalized_loss = centroid_diff / max_expected_diff
    
    return float(np.clip(normalized_loss, 0.0, 1.0))