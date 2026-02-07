"""
Audio I/O module for loading, normalizing, and saving audio files.
Implements Day 2: Audio Loading & Normalization.
"""

import os
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioTrack:
    """Represents an audio track in memory.
    
    Attributes:
        audio: Audio samples as numpy array (float32)
        sr: Sample rate in Hz (default: 44100)
        name: Name of the audio file (basename)
    """
    audio: np.ndarray
    sr: int = 44100
    name: str = ""


def load_audio_file(path: str, max_duration_sec: float = 60.0) -> AudioTrack:
    """Load an audio file, normalize it, and return as AudioTrack.
    
    Args:
        path: Path to the audio file
        max_duration_sec: Maximum duration in seconds (default: 60.0)
                         If file is longer, the center portion will be taken.
    
    Returns:
        AudioTrack object with normalized audio
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If audio is empty or invalid
        Exception: For other librosa/soundfile errors
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Load audio with librosa: convert to mono, resample to 44100 Hz
    audio, sr = librosa.load(path, sr=44100, mono=True)
    
    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {path}")
    
    # Calculate duration and trim if necessary (take center portion)
    duration = len(audio) / sr
    if duration > max_duration_sec:
        max_samples = int(max_duration_sec * sr)
        start_sample = int((len(audio) - max_samples) // 2)
        audio = audio[start_sample:start_sample + max_samples]
    
    # Peak Normalization to -1.0 dB
    peak_linear = np.max(np.abs(audio))
    if peak_linear > 0:
        target_linear = 10 ** (-1.0 / 20.0)  # ≈ 0.89125
        audio = audio * (target_linear / peak_linear)
    
    # Ensure float32 dtype
    audio = audio.astype(np.float32)
    
    # Extract basename for the track name
    name = os.path.basename(path)
    
    return AudioTrack(audio=audio, sr=sr, name=name)


def save_audio_file(audio: AudioTrack, output_path: str) -> None:
    """Save an AudioTrack to a WAV file.
    
    Args:
        audio: AudioTrack object to save
        output_path: Path where to save the file
    
    Raises:
        ValueError: If audio is empty or invalid
        Exception: For soundfile write errors
    """
    if len(audio.audio) == 0:
        raise ValueError("Cannot save empty audio track")
    
    # Try to save with PCM_24 subtype, fallback to default if not supported
    try:
        sf.write(output_path, audio.audio, audio.sr, subtype='PCM_24')
    except (ValueError, RuntimeError):
        # Fallback to default format if PCM_24 is not available
        sf.write(output_path, audio.audio, audio.sr)

