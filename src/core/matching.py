"""
Match EQ filter generation module.
Implements Day 4: The Match EQ Filter Generation.
"""

import numpy as np
from scipy import ndimage
from scipy import signal

from src.core.analysis import ToneFeatures


def create_match_filter(
    source_features: ToneFeatures,
    target_features: ToneFeatures,
    num_taps: int = 4096,
    epsilon: float = 1e-10,
    smoothing_sigma: float = 5.0
) -> np.ndarray:
    """Create a FIR filter that matches the source spectrum to the target spectrum.
    
    This function computes a Match EQ filter that transforms the frequency response
    of the source signal to match the target signal's frequency response.
    
    Args:
        source_features: ToneFeatures from the source (DI) track
        target_features: ToneFeatures from the target (Reference) track
        num_taps: Number of filter taps (coefficients) for the FIR filter (default: 1025)
                  Must be odd to create a Type I FIR filter (allows non-zero gain at Nyquist)
        epsilon: Small constant to avoid division by zero (default: 1e-10)
        smoothing_sigma: Standard deviation for Gaussian smoothing of gain curve (default: 3.0)
    
    Returns:
        numpy array of FIR filter coefficients (impulse response)
    
    Raises:
        ValueError: If spectra have incompatible shapes or frequencies don't match
    """
    # Ensure num_taps is odd (Type I FIR filter allows non-zero gain at Nyquist)
    if num_taps % 2 == 0:
        num_taps += 1
    # Validate that spectra have the same shape
    if source_features.spectrum.shape != target_features.spectrum.shape:
        raise ValueError(
            f"Spectra shapes don't match: source {source_features.spectrum.shape} "
            f"vs target {target_features.spectrum.shape}"
        )
    
    if source_features.frequencies.shape != target_features.frequencies.shape:
        raise ValueError(
            f"Frequency axes don't match: source {source_features.frequencies.shape} "
            f"vs target {target_features.frequencies.shape}"
        )
    
    # Extract magnitude spectra
    source_spectrum = source_features.spectrum
    target_spectrum = target_features.spectrum
    frequencies = source_features.frequencies
    
    # Compute gain curve: target / source
    # Add epsilon to avoid division by zero
    gain = target_spectrum / (source_spectrum + epsilon)
    
    # Smooth the gain curve to prevent "comb filter" artifacts
    # Gaussian filter smooths out sharp peaks and valleys
    gain_smoothed = ndimage.gaussian_filter1d(gain, sigma=smoothing_sigma)
    
    # Clip gain to reasonable range: +/- 12 dB to prevent ringing artifacts
    # Extreme gain values cause phase issues and unnatural sound
    gain_min = 10 ** (-12 / 20)  # ~0.25 (linear)
    gain_max = 10 ** (12 / 20)   # ~4.0 (linear)
    gain_smoothed = np.clip(gain_smoothed, gain_min, gain_max)
    
    # Normalize frequencies to 0-0.5 range (required by firwin2)
    # Nyquist frequency = sample_rate / 2
    # We can infer sample_rate from the frequencies array:
    # The last frequency bin is at Nyquist = sample_rate / 2
    nyquist_freq = frequencies[-1]
    sample_rate = int(nyquist_freq * 2)
    
    # Ensure first frequency is 0 and last is exactly Nyquist (sample_rate/2)
    # firwin2 requires frequencies to start at 0 and end at fs/2
    frequencies_for_filter = frequencies.copy()
    
    # Ensure first frequency is exactly 0
    if frequencies_for_filter[0] != 0.0:
        frequencies_for_filter[0] = 0.0
    
    # Ensure last frequency is exactly Nyquist (sample_rate/2)
    frequencies_for_filter[-1] = sample_rate / 2.0
    
    # Create FIR filter using firwin2
    # firwin2 creates a filter with the specified frequency response
    # It interpolates the gain curve to create filter coefficients
    # Use fs parameter to pass sample rate, frequencies should be in Hz
    try:
        filter_coeffs = signal.firwin2(
            num_taps,
            frequencies_for_filter,
            gain_smoothed,
            fs=sample_rate
        )
    except Exception as e:
        raise ValueError(
            f"Failed to create FIR filter: {e}. "
            f"Check that frequencies are in [0, 0.5] range and gain values are valid."
        ) from e
    
    return filter_coeffs

