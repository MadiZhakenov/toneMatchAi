"""
Parameter optimization module using scipy.optimize.
Finds best pedalboard parameters by minimizing audio distance function.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy import optimize
from pedalboard import Compressor, Distortion, Reverb
from scipy import signal as scipy_signal

# Try to import Convolution
try:
    from pedalboard import Convolution
    CONVOLUTION_AVAILABLE = True
except ImportError:
    CONVOLUTION_AVAILABLE = False
    print("Warning: pedalboard.Convolution not available. IR optimization will not work.")

from src.core.analysis import (
    calculate_audio_distance, 
    calculate_g_loss, 
    extract_harmonics, 
    calculate_harmonic_warmth,
    calculate_envelope_loss,
    calculate_spectral_shape_loss,
    calculate_brightness_loss
)
from src.core.io import AudioTrack
from src.core.nam_processor import NAMProcessor
from src.core.processor import ToneProcessor

# Try to import DifferentiableWaveshaper
try:
    from src.core.ddsp_processor import DifferentiableWaveshaperProcessor, DifferentiablePostFX
    WAVESHAPER_AVAILABLE = True
    DIFFERENTIABLE_POSTFX_AVAILABLE = True
except ImportError:
    WAVESHAPER_AVAILABLE = False
    DifferentiableWaveshaperProcessor = None
    DIFFERENTIABLE_POSTFX_AVAILABLE = False
    DifferentiablePostFX = None

# Import SOTA composite loss function
try:
    from src.core.loss import CompositeAudioLoss
    COMPOSITE_LOSS_AVAILABLE = True
except ImportError:
    COMPOSITE_LOSS_AVAILABLE = False
    print("Warning: CompositeAudioLoss not available. Falling back to calculate_g_loss.")

# Try to import PyTorch for differentiable optimization
try:
    import torch
    import torch.nn.functional as F
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None
    torchaudio = None
    print("Warning: PyTorch not available. Adam optimization will not work.")


def compute_g_loss_tensor(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    sr: int = 44100,
    device: Optional[str] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute G-Loss components using PyTorch tensors (differentiable).
    
    This is a simplified differentiable version of G-Loss that computes:
    1. Harmonic Loss (simplified using STFT magnitude)
    2. Envelope Loss (using RMS)
    3. Spectral Shape Loss (using Mel-spectrogram)
    4. Brightness Loss (using Spectral Centroid)
    
    Args:
        y_pred: Predicted audio tensor (samples,) or (batch, samples)
        y_true: Reference audio tensor (samples,) or (batch, samples)
        sr: Sample rate (default: 44100)
        device: Device to use ('cpu' or 'cuda'). If None, auto-detect.
    
    Returns:
        Tuple of (total_loss_tensor, components_dict) where:
        - total_loss_tensor: Scalar tensor for backward pass
        - components_dict: Dictionary with individual loss components (detached, for logging)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for compute_g_loss_tensor")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Ensure tensors are on correct device and have same shape
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)
    
    # Ensure same length
    min_len = min(y_pred.shape[-1], y_true.shape[-1])
    if y_pred.dim() == 1:
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
    else:
        y_pred = y_pred[..., :min_len]
        y_true = y_true[..., :min_len]
    
    # Ensure both are 1D for processing
    if y_pred.dim() > 1:
        y_pred = y_pred.flatten()
    if y_true.dim() > 1:
        y_true = y_true.flatten()
    
    # 1. Harmonic Loss (simplified): Compare STFT magnitude
    n_fft = 2048
    hop_length = 512
    
    # Compute STFT
    stft_pred = torch.stft(
        y_pred,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        normalized=False
    )
    stft_true = torch.stft(
        y_true,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        normalized=False
    )
    
    # Get magnitude
    mag_pred = torch.abs(stft_pred)
    mag_true = torch.abs(stft_true)
    
    # Average over time
    mag_pred_avg = torch.mean(mag_pred, dim=1)
    mag_true_avg = torch.mean(mag_true, dim=1)
    
    # Harmonic loss: L1 distance between magnitude spectra
    harmonic_loss = F.l1_loss(mag_pred_avg, mag_true_avg)
    
    # 2. Envelope Loss: Compare RMS envelopes
    frame_length = int(0.0075 * sr)  # 7.5ms
    frame_length = max(64, frame_length)
    if frame_length % 2 == 0:
        frame_length += 1
    
    hop_length_env = max(1, frame_length // 4)
    
    # Compute RMS using simple convolution approach (more stable)
    # Create a simple averaging kernel
    kernel = torch.ones(1, 1, frame_length, device=device, dtype=y_pred.dtype) / frame_length
    
    # Reshape for conv1d: (batch, channels, samples)
    y_pred_conv = y_pred.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
    y_true_conv = y_true.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
    
    # Apply convolution with padding to maintain length
    # Use 'same' padding by padding manually
    pad_size = frame_length // 2
    y_pred_padded = F.pad(y_pred_conv, (pad_size, pad_size), mode='constant', value=0.0)
    y_true_padded = F.pad(y_true_conv, (pad_size, pad_size), mode='constant', value=0.0)
    
    # Compute RMS energy (square, then average, then sqrt)
    rms_squared_pred = F.conv1d(y_pred_padded ** 2, kernel, stride=hop_length_env)
    rms_squared_true = F.conv1d(y_true_padded ** 2, kernel, stride=hop_length_env)
    
    # Take square root to get RMS
    rms_pred = torch.sqrt(rms_squared_pred.squeeze(0).squeeze(0))
    rms_true = torch.sqrt(rms_squared_true.squeeze(0).squeeze(0))
    
    # Normalize
    max_rms = torch.max(torch.stack([torch.max(rms_pred), torch.max(rms_true)]))
    if max_rms > 1e-10:
        rms_pred_norm = rms_pred / max_rms
        rms_true_norm = rms_true / max_rms
    else:
        rms_pred_norm = rms_pred
        rms_true_norm = rms_true
    
    # Envelope loss: L1 distance
    envelope_loss = F.l1_loss(rms_pred_norm, rms_true_norm)
    
    # 3. Spectral Shape Loss: Compare Mel-spectrogram
    # Use simplified version: compare log-magnitude STFT
    epsilon = 1e-10
    log_mag_pred = torch.log(mag_pred_avg + epsilon)
    log_mag_true = torch.log(mag_true_avg + epsilon)
    
    # Normalize
    log_mag_pred_norm = (log_mag_pred - torch.mean(log_mag_pred)) / (torch.std(log_mag_pred) + epsilon)
    log_mag_true_norm = (log_mag_true - torch.mean(log_mag_true)) / (torch.std(log_mag_true) + epsilon)
    
    # Spectral shape loss: MSE
    spectral_shape_loss = F.mse_loss(log_mag_pred_norm, log_mag_true_norm)
    
    # 4. Brightness Loss: Compare Spectral Centroid
    # Spectral centroid = sum(freq * magnitude) / sum(magnitude)
    frequencies = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=device)
    
    # mag_pred_avg and mag_true_avg are already 1D tensors (after mean over time)
    # So we can directly multiply and sum
    centroid_pred = torch.sum(frequencies * mag_pred_avg) / (torch.sum(mag_pred_avg) + epsilon)
    centroid_true = torch.sum(frequencies * mag_true_avg) / (torch.sum(mag_true_avg) + epsilon)
    
    # Brightness loss: normalized difference
    centroid_diff = torch.abs(centroid_pred - centroid_true)
    brightness_loss = centroid_diff / (sr / 2.0)  # Normalize by Nyquist
    
    # Combine losses (matching weights from optimizer's objective_function)
    # Original weights: harmonic=1.0, envelope=1.0, spectral=1.0, brightness=1.0
    total_loss = harmonic_loss + envelope_loss + spectral_shape_loss + brightness_loss
    
    # Create components dict for logging (detached)
    components = {
        'harmonic_loss': float(harmonic_loss.detach().cpu().item()),
        'envelope_loss': float(envelope_loss.detach().cpu().item()),
        'spectral_shape_loss': float(spectral_shape_loss.detach().cpu().item()),
        'brightness_loss': float(brightness_loss.detach().cpu().item()),
        'total_loss': float(total_loss.detach().cpu().item())
    }
    
    return total_loss, components


class ToneOptimizer:
    """Optimizer for finding best audio effect parameters using mathematical optimization.
    
    Uses scipy.optimize to minimize the distance between processed DI track
    and reference track using Mel-spectrogram-based distance metric.
    """
    
    def __init__(self, test_duration_sec: float = 5.0, max_iterations: int = 50, ir_folder: str = "assets/impulse_responses", nam_folder: str = "assets/nam_models", max_models_to_test: Optional[int] = None, selected_models_keywords: Optional[List[str]] = None, fast_grid_search: bool = False):
        """Initialize the optimizer.
        
        Args:
            test_duration_sec: Duration of audio to use for optimization (default: 5.0 sec)
            max_iterations: Maximum number of optimization iterations (default: 50)
            ir_folder: Path to folder containing IR files (default: "assets/impulse_responses")
            nam_folder: Path to folder containing NAM model files (default: "assets/nam_models")
            max_models_to_test: Maximum number of NAM models to test (None = test all, deprecated if selected_models_keywords is set)
            selected_models_keywords: List of keywords to filter NAM models (e.g., ["Fender", "VOX", "5150"])
            fast_grid_search: If True, uses fast Grid Search mode (1.5 sec test duration, 5 iterations per combination)
        """
        self.test_duration_sec = test_duration_sec
        self.max_iterations = max_iterations
        # Ensure all paths are absolute - critical for plugin environment where working directory may vary
        self.ir_folder = os.path.normpath(os.path.abspath(ir_folder))
        self.nam_folder = os.path.normpath(os.path.abspath(nam_folder))
        self.max_models_to_test = max_models_to_test
        self.selected_models_keywords = selected_models_keywords or []
        self.nam_processor = NAMProcessor()
        self.fast_grid_search = fast_grid_search
        
        # Initialize SOTA composite loss function
        if COMPOSITE_LOSS_AVAILABLE:
            self.composite_loss = CompositeAudioLoss()
        else:
            self.composite_loss = None
    
    def _find_ir_files(self) -> List[str]:
        """Find all .wav IR files in the IR folder.
        
        Returns:
            List of full paths to IR files
            
        Raises:
            ValueError: If IR folder doesn't exist or contains no .wav files
        """
        if not os.path.exists(self.ir_folder):
            raise ValueError(
                f"IR folder not found: {self.ir_folder}\n"
                f"Please create the folder and add .wav IR files to it."
            )
        
        ir_files = []
        for filename in os.listdir(self.ir_folder):
            if filename.lower().endswith('.wav'):
                ir_path = os.path.join(self.ir_folder, filename)
                # Ensure absolute path
                ir_path = os.path.normpath(os.path.abspath(ir_path))
                ir_files.append(ir_path)
        
        if len(ir_files) == 0:
            raise ValueError(
                f"No .wav IR files found in {self.ir_folder}\n"
                f"Please add at least one .wav IR file to the folder."
            )
        
        return sorted(ir_files)
    
    def _find_nam_files(self) -> List[str]:
        """Find .nam model files in the NAM folder, filtered by keywords if specified.
        
        Returns:
            List of full paths to NAM files (excluding WaveNet models)
            
        Note:
            If folder doesn't exist or contains no .nam files, returns empty list
            (allows graceful fallback to mock mode)
            If selected_models_keywords is set, only returns files matching those keywords
            WaveNet models are excluded because NAM Core doesn't support them yet
        """
        if not os.path.exists(self.nam_folder):
            return []
        
        nam_files = []
        
        # Define representative model keywords
        keywords = self.selected_models_keywords if self.selected_models_keywords else []
        
        for filename in os.listdir(self.nam_folder):
            if filename.lower().endswith('.nam'):
                nam_path = os.path.join(self.nam_folder, filename)
                # Ensure absolute path
                nam_path = os.path.normpath(os.path.abspath(nam_path))
                
                # Filter out WaveNet models - check architecture in JSON
                try:
                    import json
                    with open(nam_path, 'r', encoding='utf-8') as f:
                        model_data = json.load(f)
                        architecture = model_data.get('architecture', '').lower()
                        if architecture == 'wavenet':
                            continue  # Skip WaveNet models
                except (json.JSONDecodeError, KeyError, IOError):
                    # If we can't read the file, include it (might be valid)
                    pass
                
                # If keywords are specified, filter by them
                if keywords:
                    filename_upper = filename.upper()
                    # Check if any keyword is in the filename (case-insensitive)
                    if any(keyword.upper() in filename_upper for keyword in keywords):
                        nam_files.append(nam_path)
                else:
                    # No keywords specified, include all files (except WaveNet, already filtered)
                    nam_files.append(nam_path)
        
        # Sort for consistency
        return sorted(nam_files)
    
    def _split_nam_models(self, nam_files: List[str], max_models_per_category: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Split NAM models into FX (pedals) and AMP (amplifiers) categories.
        
        Args:
            nam_files: List of NAM file paths
            max_models_per_category: Maximum number of models per category (None = no limit)
            
        Returns:
            Tuple of (fx_models, amp_models) lists
        """
        # FX model keywords (pedals/boosters) - prioritized order
        fx_keywords = ['DS1', 'DirtyTree', 'PlumesClone', 'Klone', 'TS9', 'Tube Screamer', 
                      'OD808', 'M77', 'HM2', 'BOSS', 'Maxon', 'MXR', 'Precision-Drive',
                      'Soldano', 'SparkleDrive', 'ThroneTorcher', 'FX56B', '805']
        
        # AMP model keywords (amplifiers) - prioritized order
        amp_keywords = ['5150', '6505', '6534', 'Mesa', 'Fender', 'VOX', 'Friedman', 
                       'Suhr', 'Engl', 'JSX', 'XXX', 'JCM', 'TwinVerb', 'AC15', 
                       'MarkIV', 'Bugera', 'Laney', 'Ceriatone', 'Splawn', 'Magnatone',
                       'Jet City', 'Sovtek', 'MIG50', 'Driftwood', 'Nightmare']
        
        fx_models = []
        amp_models = []
        
        for nam_path in nam_files:
            filename = os.path.basename(nam_path).upper()
            
            # Check if it's an FX model
            is_fx = any(keyword.upper() in filename for keyword in fx_keywords)
            # Check if it's an AMP model
            is_amp = any(keyword.upper() in filename for keyword in amp_keywords)
            
            # Prioritize FX if both match (e.g., "Soldano" can be both, but if it's a pedal, it's FX)
            if is_fx and not is_amp:
                fx_models.append(nam_path)
            elif is_amp:
                amp_models.append(nam_path)
            # If neither matches, default to AMP (most models are amps)
            else:
                amp_models.append(nam_path)
        
        # Limit models per category if specified
        if max_models_per_category is not None:
            # Prioritize models with more common keywords (better quality models)
            # For FX: prioritize DS1, PlumesClone, Klone
            fx_priority_keywords = ['DS1', 'PlumesClone', 'Klone', 'DirtyTree']
            fx_models_prioritized = []
            fx_models_other = []
            
            for fx_path in fx_models:
                fx_name = os.path.basename(fx_path).upper()
                if any(kw.upper() in fx_name for kw in fx_priority_keywords):
                    fx_models_prioritized.append(fx_path)
                else:
                    fx_models_other.append(fx_path)
            
            fx_models = fx_models_prioritized + fx_models_other
            fx_models = fx_models[:max_models_per_category]
            
            # For AMP: prioritize 5150, 6505, Fender, VOX, Friedman, Mesa
            amp_priority_keywords = ['5150', '6505', 'Fender', 'VOX', 'Friedman', 'Mesa']
            amp_models_prioritized = []
            amp_models_other = []
            
            for amp_path in amp_models:
                amp_name = os.path.basename(amp_path).upper()
                if any(kw.upper() in amp_name for kw in amp_priority_keywords):
                    amp_models_prioritized.append(amp_path)
                else:
                    amp_models_other.append(amp_path)
            
            amp_models = amp_models_prioritized + amp_models_other
            amp_models = amp_models[:max_models_per_category]
        
        return sorted(fx_models), sorted(amp_models)
    
    def _analyze_reference_audio(self, ref_audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze reference audio to determine its characteristics.
        
        "Умный Сомелье" - Step 1: Анализ референса для интеллектуального подбора оборудования.
        
        Args:
            ref_audio: Reference audio signal
            sr: Sample rate
            
        Returns:
            Dictionary with reference characteristics:
            - 'spectral_centroid': Average spectral centroid (Hz)
            - 'brightness': Normalized brightness (0-1)
            - 'aggression_level': 'clean', 'crunch', 'high_gain'
            - 'recommended_amp_style': List of recommended amp keywords
            - 'recommended_fx_style': List of recommended FX keywords
        """
        import librosa
        
        # Calculate spectral centroid (brightness indicator)
        spectral_centroid = librosa.feature.spectral_centroid(y=ref_audio, sr=sr)[0]
        avg_centroid = float(np.mean(spectral_centroid))
        
        # Calculate spectral rolloff (high frequency content)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=ref_audio, sr=sr, roll_percent=0.85)[0]
        avg_rolloff = float(np.mean(spectral_rolloff))
        
        # Calculate RMS energy (loudness/dynamics)
        rms = librosa.feature.rms(y=ref_audio)[0]
        avg_rms = float(np.mean(rms))
        
        # Calculate spectral flatness (noise-like vs tonal)
        spectral_flatness = librosa.feature.spectral_flatness(y=ref_audio)[0]
        avg_flatness = float(np.mean(spectral_flatness))
        
        # Normalize brightness (0-1 scale based on typical guitar centroid range)
        # Clean guitar: ~1000-2000 Hz, Distorted: ~2000-4000 Hz
        brightness = min(1.0, max(0.0, (avg_centroid - 1000) / 3000))
        
        # Determine aggression level
        if avg_centroid > 2500 or avg_flatness > 0.1:
            aggression_level = 'high_gain'
        elif avg_centroid > 1800 or avg_rms > 0.15:
            aggression_level = 'crunch'
        else:
            aggression_level = 'clean'
        
        # Determine recommended amp styles based on aggression
        if aggression_level == 'high_gain':
            recommended_amps = ['5150', '6505', 'Mesa', 'Friedman', 'Engl', 'JSX', 'XXX', 
                               'Splawn', 'Driftwood', 'Nightmare', 'MIG50']
            recommended_fx = ['DS1', 'HM2', 'ThroneTorcher', 'Precision-Drive', 'M77']
        elif aggression_level == 'crunch':
            recommended_amps = ['Marshall', 'JCM', 'Friedman', 'Suhr', 'Ceriatone', 
                               'Laney', 'Jet City', '5150', 'Bugera']
            recommended_fx = ['Klone', 'PlumesClone', 'TS9', 'OD808', 'DirtyTree', 'DS1']
        else:  # clean
            recommended_amps = ['Fender', 'VOX', 'AC15', 'TwinVerb', 'Magnatone', 
                               'Suhr', 'Mesa']  # Mesa can be clean too
            recommended_fx = ['Klone', 'SparkleDrive', 'Boost', '805']  # Light pedals
        
        result = {
            'spectral_centroid': avg_centroid,
            'spectral_rolloff': avg_rolloff,
            'avg_rms': avg_rms,
            'spectral_flatness': avg_flatness,
            'brightness': brightness,
            'aggression_level': aggression_level,
            'recommended_amp_keywords': recommended_amps,
            'recommended_fx_keywords': recommended_fx
        }
        
        return result
    
    def _classify_and_select_models(
        self, 
        nam_files: List[str], 
        ref_analysis: Dict[str, Any],
        max_amps: int = 25,
        max_fx: int = 10,
        force_high_gain: bool = False
    ) -> Tuple[List[str], List[str]]:
        """Classify all NAM models and select relevant ones based on reference analysis.
        
        "Умный Сомелье" - Step 1: Интеллектуальная фильтрация моделей.
        
        Args:
            nam_files: List of all NAM file paths
            ref_analysis: Reference audio analysis from _analyze_reference_audio()
            max_amps: Maximum number of amp models to select (default: 25)
            max_fx: Maximum number of FX models to select (default: 10)
            force_high_gain: If True, exclude clean amps and only search high-gain models
            
        Returns:
            Tuple of (selected_fx_models, selected_amp_models)
        """
        # Full classification keywords
        fx_keywords = [
            'DS1', 'DirtyTree', 'PlumesClone', 'Klone', 'TS9', 'Tube Screamer',
            'OD808', 'M77', 'HM2', 'BOSS', 'Maxon', 'MXR', 'Precision-Drive',
            'SparkleDrive', 'ThroneTorcher', 'FX56B', '805', 'Boost', 'Drive',
            'Overdrive', 'Pedal', 'Klon'
        ]
        
        # Clean amp keywords to exclude when force_high_gain=True
        clean_amp_keywords = [
            'CLEAN', 'TWIN', 'JAZZ', 'FENDER', 'VOX', 'AC30', 'BLUES', 'CHAMP',
            'DELUXE', 'SUPER', 'BASSMAN', 'TWEED', 'BLACKFACE', 'SILVERFACE'
        ]
        
        # High-gain amp keywords to include when force_high_gain=True
        high_gain_keywords = [
            '5150', '6505', 'MESA', 'MARSHALL', 'DRIVE', 'HIGH GAIN', 'METAL',
            'DISTORTION', 'CRUNCH', 'FRIEDMAN', 'DIEZEL', 'ENGL', 'PEAVEY',
            'BOOGIE', 'RECTIFIER', 'MARK', 'JCM', 'JVM', 'SLO', 'SOLDANO'
        ]
        
        # Classify all models
        fx_models = []
        amp_models = []
        
        for nam_path in nam_files:
            filename = os.path.basename(nam_path).upper()
            is_fx = any(kw.upper() in filename for kw in fx_keywords)
            
            if is_fx:
                fx_models.append(nam_path)
            else:
                # If force_high_gain is enabled, filter out clean amps
                if force_high_gain:
                    # Check if it's a clean amp (exclude)
                    is_clean = any(kw in filename for kw in clean_amp_keywords)
                    # Check if it's a high-gain amp (include)
                    is_high_gain = any(kw in filename for kw in high_gain_keywords)
                    
                    # Only include if it's explicitly high-gain OR if it doesn't match clean keywords
                    # This way we include unknown amps but exclude known clean amps
                    if is_clean and not is_high_gain:
                        continue  # Skip this clean amp
                
                amp_models.append(nam_path)
        
        # Get recommended keywords from reference analysis
        recommended_amp_kw = ref_analysis.get('recommended_amp_keywords', [])
        recommended_fx_kw = ref_analysis.get('recommended_fx_keywords', [])
        aggression = ref_analysis.get('aggression_level', 'crunch')
        
        # Score and sort amp models by relevance
        def score_amp(path):
            name = os.path.basename(path).upper()
            score = 0
            # Primary match: recommended keywords get high score
            for kw in recommended_amp_kw:
                if kw.upper() in name:
                    score += 10
            # Universal good amps always get a bonus
            universal_good = ['5150', '6505', 'FRIEDMAN', 'MESA']
            for kw in universal_good:
                if kw in name:
                    score += 3
            return score
        
        def score_fx(path):
            name = os.path.basename(path).upper()
            score = 0
            for kw in recommended_fx_kw:
                if kw.upper() in name:
                    score += 10
            # Universal good pedals
            universal_good = ['DS1', 'KLONE', 'PLUMES']
            for kw in universal_good:
                if kw in name:
                    score += 3
            return score
        
        # Sort by score (descending)
        amp_models_scored = sorted(amp_models, key=score_amp, reverse=True)
        fx_models_scored = sorted(fx_models, key=score_fx, reverse=True)
        
        # Select top models, but ensure diversity
        selected_amps = []
        seen_amp_prefixes = set()
        
        # First pass: take high-scoring unique models
        for amp_path in amp_models_scored:
            if len(selected_amps) >= max_amps:
                break
            # Get first 10 chars as prefix for diversity
            prefix = os.path.basename(amp_path)[:10].upper()
            if prefix not in seen_amp_prefixes or score_amp(amp_path) > 5:
                selected_amps.append(amp_path)
                seen_amp_prefixes.add(prefix)
        
        # Fill remaining slots if needed
        for amp_path in amp_models_scored:
            if len(selected_amps) >= max_amps:
                break
            if amp_path not in selected_amps:
                selected_amps.append(amp_path)
        
        # Select FX models
        selected_fx = []
        seen_fx_prefixes = set()
        
        for fx_path in fx_models_scored:
            if len(selected_fx) >= max_fx:
                break
            prefix = os.path.basename(fx_path)[:8].upper()
            if prefix not in seen_fx_prefixes or score_fx(fx_path) > 5:
                selected_fx.append(fx_path)
                seen_fx_prefixes.add(prefix)
        
        # Fill remaining
        for fx_path in fx_models_scored:
            if len(selected_fx) >= max_fx:
                break
            if fx_path not in selected_fx:
                selected_fx.append(fx_path)
        
        return selected_fx, selected_amps
    
    def find_best_rig_smart(
        self, 
        di_track: AudioTrack, 
        ref_track: AudioTrack, 
        top_n: int = 3,
        force_high_gain: bool = False
    ) -> Dict[str, Any]:
        """Find the best rig using Smart Sommelier two-level search.
        
        "Умный Сомелье" - Интеллектуальный двухуровневый поиск:
        1. Уровень 1: Найти TOP-2 усилителя БЕЗ педали
        2. Уровень 2: Протестировать TOP-2 усилителя с КАЖДОЙ педалью
        
        Это сокращает комбинации с (25*10=250) до (25 + 2*10 = 45). Ускорение в 5 раз!
        
        Args:
            di_track: Input DI track
            ref_track: Reference track to match
            top_n: Number of top combinations to return
            force_high_gain: If True, force high-gain mode (skip clean amps)
            
        Returns:
            Dictionary with best rig configurations
        """
        if not CONVOLUTION_AVAILABLE:
            raise ValueError("pedalboard.Convolution is not available.")
        
        if di_track.sr != ref_track.sr:
            raise ValueError(f"Sample rates must match: DI={di_track.sr}, Ref={ref_track.sr}")
        
        print("\n" + "="*70)
        print("[SMART SOMMELIER] Intelligent Two-Level Rig Search")
        print("="*70)
        
        # ========== STEP 1: Analyze Reference ==========
        if force_high_gain:
            print("\n[Step 1/4] FORCE HIGH GAIN mode: Skipping spectral analysis")
            # Create a fake ref_analysis with high-gain settings
            ref_analysis = {
                'aggression_level': 'high-gain',
                'recommended_amp_keywords': ['5150', '6505', 'MESA', 'MARSHALL', 'DRIVE', 'HIGH GAIN', 'METAL', 'DISTORTION', 'CRUNCH'],
                'recommended_fx_keywords': [],
                'spectral_centroid': 3000.0,  # High value to indicate high-gain
                'brightness': 0.8
            }
            print(f"  Aggression Level: {ref_analysis['aggression_level'].upper()} (FORCED)")
            print(f"  Recommended Amps: {', '.join(ref_analysis['recommended_amp_keywords'][:5])}")
        else:
            print("\n[Step 1/4] Analyzing reference audio...")
            
            # Use first 3 seconds for analysis
            analysis_samples = int(3.0 * ref_track.sr)
            ref_audio_for_analysis = ref_track.audio[:min(analysis_samples, len(ref_track.audio))]
            
            ref_analysis = self._analyze_reference_audio(ref_audio_for_analysis, ref_track.sr)
            
            print(f"  Spectral Centroid: {ref_analysis['spectral_centroid']:.0f} Hz")
            print(f"  Brightness: {ref_analysis['brightness']:.2f}")
            print(f"  Aggression Level: {ref_analysis['aggression_level'].upper()}")
            print(f"  Recommended Amps: {', '.join(ref_analysis['recommended_amp_keywords'][:5])}")
            print(f"  Recommended FX: {', '.join(ref_analysis['recommended_fx_keywords'][:5])}")
        
        # ========== STEP 2: Classify and Select Models ==========
        print("\n[Step 2/4] Classifying and selecting relevant models...")
        
        # Get ALL NAM files (no keyword filtering)
        all_nam_files = []
        if os.path.exists(self.nam_folder):
            for filename in os.listdir(self.nam_folder):
                if filename.lower().endswith('.nam'):
                    nam_path = os.path.join(self.nam_folder, filename)
                    # Ensure absolute path
                    nam_path = os.path.normpath(os.path.abspath(nam_path))
                    all_nam_files.append(nam_path)
        
        print(f"  Total NAM models in library: {len(all_nam_files)}")
        
        fx_models, amp_models = self._classify_and_select_models(
            all_nam_files, 
            ref_analysis,
            max_amps=25,  # Test up to 25 amps
            max_fx=10,    # Test up to 10 FX pedals
            force_high_gain=force_high_gain
        )
        
        # Add "no FX" option
        fx_models_with_none = [None] + fx_models
        
        ir_files = self._find_ir_files()
        
        print(f"  Selected AMP models: {len(amp_models)}")
        print(f"  Selected FX models: {len(fx_models)} (+ no pedal option)")
        print(f"  IR files: {len(ir_files)}")
        
        # Print selected models
        print("\n  Selected AMP models:")
        for amp in amp_models[:10]:  # Show first 10
            print(f"    - {os.path.basename(amp)}")
        if len(amp_models) > 10:
            print(f"    ... and {len(amp_models) - 10} more")
        
        print("\n  Selected FX models:")
        for fx in fx_models[:5]:  # Show first 5
            print(f"    - {os.path.basename(fx)}")
        if len(fx_models) > 5:
            print(f"    ... and {len(fx_models) - 5} more")
        
        # ========== STEP 3: Level 1 - Find Best AMPs (without FX) ==========
        print("\n[Step 3/4] Level 1: Finding best AMPs (without FX pedal)...")
        
        # Prepare test audio
        test_duration = 1.0  # 1 second for fast testing
        test_samples = int(test_duration * di_track.sr)
        di_test_audio = di_track.audio[:min(test_samples, len(di_track.audio))].copy()
        ref_test_audio = ref_track.audio[:min(test_samples, len(ref_track.audio))].copy()
        
        # Test all AMP models without FX
        amp_results = []
        total_amps = len(amp_models)
        
        print(f"  Testing {total_amps} AMP models (no FX)...")
        
        for idx, amp_path in enumerate(amp_models, 1):
            amp_name = os.path.basename(amp_path)
            
            # Find best gain for this amp
            best_loss = float('inf')
            best_gain = 0.0
            
            # Quick gain sweep: -6, 0, 6, 12 dB
            for gain_db in [-6.0, 0.0, 6.0, 12.0]:
                try:
                    # Process: Gain -> AMP -> IR
                    audio = di_test_audio.copy().astype(np.float32)
                    audio = audio * (10 ** (gain_db / 20.0))
                    
                    # AMP processing
                    audio = self.nam_processor.process_audio(audio, amp_path, sample_rate=di_track.sr)
                    
                    # IR processing (use first IR)
                    convolution = Convolution(impulse_response_filename=ir_files[0])
                    audio = convolution(audio, sample_rate=di_track.sr)
                    
                    # Normalize
                    max_val = np.max(np.abs(audio))
                    if max_val > 0.95:
                        audio = audio / max_val * 0.95
                    
                    # Calculate loss
                    loss = calculate_audio_distance(audio, ref_test_audio, sr=di_track.sr)
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_gain = gain_db
                        
                except Exception as e:
                    continue
            
            amp_results.append({
                'amp_path': amp_path,
                'amp_name': amp_name,
                'loss': best_loss,
                'gain': best_gain
            })
            
            # Progress update
            if idx % 5 == 0 or idx == total_amps:
                print(f"    [{idx}/{total_amps}] Tested {amp_name}: Loss={best_loss:.4f}")
        
        # Sort and get TOP-2 amps
        amp_results_sorted = sorted(amp_results, key=lambda x: x['loss'])
        top_2_amps = amp_results_sorted[:2]
        
        print(f"\n  TOP-2 AMP models:")
        for idx, amp in enumerate(top_2_amps, 1):
            print(f"    {idx}. {amp['amp_name']}: Loss={amp['loss']:.4f}, Gain={amp['gain']:.0f}dB")
        
        # ========== STEP 4: Level 2 - Test TOP-2 AMPs with all FX ==========
        print("\n[Step 4/4] Level 2: Testing TOP-2 AMPs with FX pedals...")
        
        all_results = []
        total_combinations = len(top_2_amps) * len(fx_models_with_none) * len(ir_files)
        current = 0
        
        print(f"  Testing {total_combinations} combinations (2 amps x {len(fx_models_with_none)} FX x {len(ir_files)} IR)...")
        
        for amp_info in top_2_amps:
            amp_path = amp_info['amp_path']
            amp_name = amp_info['amp_name']
            
            for fx_path in fx_models_with_none:
                fx_name = os.path.basename(fx_path) if fx_path else "(no pedal)"
                
                for ir_path in ir_files:
                    current += 1
                    ir_name = os.path.basename(ir_path)
                    
                    # Find best gain
                    best_loss = float('inf')
                    best_gain = 0.0
                    
                    for gain_db in [-6.0, 0.0, 6.0, 12.0]:
                        try:
                            audio = di_test_audio.copy().astype(np.float32)
                            audio = audio * (10 ** (gain_db / 20.0))
                            
                            # FX processing (if any)
                            if fx_path:
                                audio = self.nam_processor.process_audio(audio, fx_path, sample_rate=di_track.sr)
                            
                            # AMP processing
                            audio = self.nam_processor.process_audio(audio, amp_path, sample_rate=di_track.sr)
                            
                            # IR processing
                            convolution = Convolution(impulse_response_filename=ir_path)
                            audio = convolution(audio, sample_rate=di_track.sr)
                            
                            # Normalize
                            max_val = np.max(np.abs(audio))
                            if max_val > 0.95:
                                audio = audio / max_val * 0.95
                            
                            loss = calculate_audio_distance(audio, ref_test_audio, sr=di_track.sr)
                            
                            if loss < best_loss:
                                best_loss = loss
                                best_gain = gain_db
                                
                        except Exception as e:
                            continue
                    
                    all_results.append({
                        'fx_nam_path': fx_path,
                        'amp_nam_path': amp_path,
                        'ir_path': ir_path,
                        'fx_nam_name': fx_name,
                        'amp_nam_name': amp_name,
                        'ir_name': ir_name,
                        'input_gain_db': best_gain,
                        'loss': best_loss,
                        'g_loss': best_loss  # For compatibility
                    })
            
            print(f"    Completed testing with {amp_name}")
        
        # Sort by loss
        all_results_sorted = sorted(all_results, key=lambda x: x['loss'])
        top_rigs = all_results_sorted[:top_n]
        
        # Calculate total combinations tested
        total_tested = total_amps + total_combinations
        total_possible = len(amp_models) * len(fx_models_with_none) * len(ir_files)
        speedup = total_possible / total_tested if total_tested > 0 else 1
        
        print(f"\n{'='*70}")
        print(f"[SMART SOMMELIER] Search Complete!")
        print(f"{'='*70}")
        print(f"  Reference: {ref_analysis['aggression_level'].upper()} style")
        print(f"  Combinations tested: {total_tested} (vs {total_possible} brute force)")
        print(f"  Speedup: {speedup:.1f}x faster")
        print(f"\n  TOP {top_n} Combinations:")
        
        for idx, rig in enumerate(top_rigs, 1):
            print(f"    {idx}. [{rig['fx_nam_name']} -> {rig['amp_nam_name']} -> {rig['ir_name']}]")
            print(f"       Loss: {rig['loss']:.4f}, Gain: {rig['input_gain_db']:.0f}dB")
        
        return {
            'top_rigs': top_rigs,
            'best_rig': top_rigs[0] if top_rigs else None,
            'all_results': all_results_sorted,
            'ref_analysis': ref_analysis,
            'total_tested': total_tested,
            'total_possible': total_possible,
            'speedup': speedup
        }
    
    def find_best_rig(self, di_track: AudioTrack, ref_track: AudioTrack, top_n: int = 3) -> Dict[str, Any]:
        """Find the best (FX_NAM, AMP_NAM, IR) triple using fast Grid Search.
        
        Tests all combinations of FX NAM models (pedals), AMP NAM models (amplifiers), and IR files.
        Uses quick optimization (max 3 seconds per combination) to find approximate optimal input gain.
        Returns TOP-N best combinations for further fine-tuning.
        
        Args:
            di_track: Input DI track to process
            ref_track: Reference track to match
            top_n: Number of top combinations to return (default: 3)
        
        Returns:
            Dictionary with best rig configurations:
            - 'top_rigs': List of top N rig configurations, each with:
                - 'fx_nam_path': str (path to FX NAM model, or None)
                - 'amp_nam_path': str (path to AMP NAM model, or None)
                - 'ir_path': str (path to IR file)
                - 'input_gain_db': float (optimal input gain)
                - 'loss': float (loss value)
            - 'best_rig': Best rig configuration (same structure as top_rigs[0])
            - 'all_results': List of all tested combinations with their losses
        """
        if not CONVOLUTION_AVAILABLE:
            raise ValueError("pedalboard.Convolution is not available. Cannot perform IR optimization.")
        
        # Validate inputs
        if len(di_track.audio) == 0 or len(ref_track.audio) == 0:
            raise ValueError("Cannot optimize with empty audio tracks")
        
        if di_track.sr != ref_track.sr:
            raise ValueError(f"Sample rates must match: DI={di_track.sr}, Ref={ref_track.sr}")
        
        # Find and split NAM files into FX and AMP
        nam_files = self._find_nam_files()
        # Limit models per category for speed (5 models × 5 models = 25 combinations max)
        max_models = 5 if self.fast_grid_search else None
        fx_models, amp_models = self._split_nam_models(nam_files, max_models_per_category=max_models)
        ir_files = self._find_ir_files()
        
        # If no AMP models found, use None to indicate mock mode
        if len(amp_models) == 0:
            print(f"\nWarning: No AMP NAM models found in {self.nam_folder}")
            if self.selected_models_keywords:
                print(f"Warning: No models found matching keywords: {', '.join(self.selected_models_keywords)}")
            print("Will use mock AMP processing (distortion fallback)")
            amp_models = [None]  # Use None to indicate mock mode
        
        # If no FX models found, allow None (FX is optional)
        if len(fx_models) == 0:
            print(f"\nInfo: No FX NAM models (pedals) found, will test without FX boost")
            fx_models = [None]  # None means no FX pedal
        
        # Display model count information
        if self.selected_models_keywords:
            print(f"\nUsing representative NAM models filtered by keywords: {', '.join(self.selected_models_keywords)}")
        print(f"Found {len(fx_models)} FX model(s) (pedals), {len(amp_models)} AMP model(s), and {len(ir_files)} IR file(s)")
        
        print("\nFX Models (Pedals):")
        for fx_file in fx_models:
            if fx_file:
                print(f"  FX: {os.path.basename(fx_file)}")
            else:
                print(f"  FX: (none - no pedal)")
        
        print("\nAMP Models (Amplifiers):")
        for amp_file in amp_models:
            if amp_file:
                print(f"  AMP: {os.path.basename(amp_file)}")
            else:
                print(f"  AMP: (mock mode)")
        
        print("\nIR Files:")
        for ir_file in ir_files:
            print(f"  IR: {os.path.basename(ir_file)}")
        
        # Create test segments - use very short duration for fast Grid Search (max 3 seconds per combination)
        fast_test_duration = 1.0 if self.fast_grid_search else self.test_duration_sec  # Reduced from 1.5 to 1.0 for speed
        test_samples = int(fast_test_duration * di_track.sr)
        di_test_audio = di_track.audio[:min(test_samples, len(di_track.audio))].copy()
        ref_test_audio = ref_track.audio[:min(test_samples, len(ref_track.audio))].copy()
        
        # Parameter bounds for input gain optimization
        bounds = [(-6.0, 12.0)]  # Input Gain (dB) only
        x0 = np.array([0.0])  # Initial guess: 0 dB
        
        # Grid Search: Test each (FX_NAM, AMP_NAM, IR) triple
        all_results = []
        total_combinations = len(fx_models) * len(amp_models) * len(ir_files)
        current_combination = 0
        
        # Limit iterations for fast Grid Search (max 3 seconds per combination)
        # Reduced from 5 to 3 iterations for even faster search
        fast_max_iterations = 3 if self.fast_grid_search else self.max_iterations
        
        print(f"\nStarting Fast Grid Search over {total_combinations} (FX_NAM, AMP_NAM, IR) triple(s)...")
        print(f"Testing on first {fast_test_duration} seconds of audio (fast mode: {fast_max_iterations} iterations per combination)")
        
        for fx_nam_path in fx_models:
            fx_nam_name = os.path.basename(fx_nam_path) if fx_nam_path else "(none)"
            
            for amp_nam_path in amp_models:
                amp_nam_name = os.path.basename(amp_nam_path) if amp_nam_path else "(mock)"
                
                for ir_path in ir_files:
                    current_combination += 1
                    ir_name = os.path.basename(ir_path)
                    triple_name = f"{fx_nam_name} -> {amp_nam_name} -> {ir_name}"
                    
                    print(f"\n[{current_combination}/{total_combinations}] Testing [{triple_name}]...")
                    
                    iteration_count = [0]
                    
                    def objective_function(x: np.ndarray) -> float:
                        """Objective function: process audio through FX_NAM -> AMP_NAM -> IR and calculate distance.
                        
                        Args:
                            x: Parameter vector [input_gain]
                        
                        Returns:
                            Distance value (lower = better)
                        """
                        iteration_count[0] += 1
                        
                        try:
                            input_gain_db = float(x[0])
                            
                            # Process audio: Input Gain -> FX_NAM -> AMP_NAM -> IR
                            audio_processed = di_test_audio.copy().astype(np.float32)
                            
                            # Step 1: Input Gain
                            input_gain_linear = 10 ** (input_gain_db / 20.0)
                            audio_processed = audio_processed * input_gain_linear
                            
                            # Step 2: FX NAM processing (pedal/booster) - optional
                            if fx_nam_path:
                                try:
                                    audio_processed = self.nam_processor.process_audio(
                                        audio_processed,
                                        fx_nam_path,
                                        sample_rate=di_track.sr
                                    )
                                except Exception as e:
                                    # If FX fails, continue without it
                                    pass
                            
                            # Step 3: AMP NAM processing
                            if amp_nam_path:
                                try:
                                    audio_processed = self.nam_processor.process_audio(
                                        audio_processed,
                                        amp_nam_path,
                                        sample_rate=di_track.sr
                                    )
                                except Exception as e:
                                    # Fallback to mock
                                    self.nam_processor.set_mock_mode(True)
                                    audio_processed = self.nam_processor.process_audio(
                                        audio_processed,
                                        "mock",
                                        sample_rate=di_track.sr
                                    )
                                    self.nam_processor.set_mock_mode(False)
                            else:
                                # Mock mode
                                self.nam_processor.set_mock_mode(True)
                                audio_processed = self.nam_processor.process_audio(
                                    audio_processed,
                                    "mock",
                                    sample_rate=di_track.sr
                                )
                                self.nam_processor.set_mock_mode(False)
                            
                            # Step 4: IR Convolution
                            convolution = Convolution(impulse_response_filename=ir_path)
                            audio_processed = convolution(audio_processed, sample_rate=di_track.sr)
                            
                            # Step 5: Normalize to prevent clipping
                            max_val = np.max(np.abs(audio_processed))
                            if max_val > 0.95:
                                audio_processed = audio_processed / max_val * 0.95
                            
                            # Calculate distance using Mel-spectrogram
                            distance = calculate_audio_distance(audio_processed, ref_test_audio, sr=di_track.sr)
                            
                            # Print progress (every 3 iterations or first iteration)
                            if iteration_count[0] % 3 == 0 or iteration_count[0] == 1:
                                print(f"    Iteration {iteration_count[0]}: Loss {distance:.4f}, Gain: {input_gain_db:.1f}dB")
                            
                            return distance
                            
                        except Exception as e:
                            print(f"    Warning: Error in objective function: {e}")
                            return float('inf')
                    
                    # Run quick optimization for this triple (fast Grid Search)
                    print(f"  Quick evaluation for [{triple_name}]...")
                    result = optimize.minimize(
                        objective_function,
                        x0,
                        method='Powell',
                        bounds=bounds,
                        options={
                            'maxiter': fast_max_iterations,
                            'disp': False
                        }
                    )
                    
                    loss = float(result.fun)
                    best_gain = float(result.x[0])
                    
                    print(f"  [OK] [{triple_name}]: Loss = {loss:.6f}, Gain = {best_gain:.2f}dB, Iterations = {iteration_count[0]}")
                    
                    # Track result
                    all_results.append({
                        'fx_nam_path': fx_nam_path,
                        'amp_nam_path': amp_nam_path,
                        'ir_path': ir_path,
                        'fx_nam_name': fx_nam_name,
                        'amp_nam_name': amp_nam_name,
                        'ir_name': ir_name,
                        'input_gain_db': best_gain,
                        'loss': loss,
                        'iterations': iteration_count[0]
                    })
        
        # Step 1: Fast pre-selection using Mel-Loss (quick filtering)
        print(f"\n{'='*70}")
        print(f"Step 1: Fast Pre-Selection (Mel-Loss)")
        print(f"{'='*70}")
        all_results_sorted = sorted(all_results, key=lambda x: x['loss'])
        
        # Select TOP-10 candidates for deep evaluation (instead of just top_n)
        top_10_candidates = all_results_sorted[:min(10, len(all_results_sorted))]
        
        print(f"Selected TOP-10 candidates based on Mel-Loss:")
        for idx, rig in enumerate(top_10_candidates, 1):
            print(f"  {idx}. [{rig['fx_nam_name']} -> {rig['amp_nam_name']} -> {rig['ir_name']}]: "
                  f"Mel-Loss = {rig['loss']:.6f}, Gain = {rig['input_gain_db']:.2f}dB")
        
        # Step 2: Deep evaluation with G-Loss (Smart Selection)
        print(f"\n{'='*70}")
        print(f"Step 2: Deep Evaluation (G-Loss Re-ranking)")
        print(f"{'='*70}")
        print(f"Evaluating {len(top_10_candidates)} candidates with full G-Loss (Harmonic + Dynamics)...")
        
        processor = ToneProcessor()
        
        for idx, rig in enumerate(top_10_candidates, 1):
            try:
                # Process audio with this rig configuration
                gain_params = {'input_gain_db': rig['input_gain_db']}
                post_fx_params_default = {
                    'pre_eq_gain_db': 0.0,
                    'pre_eq_freq_hz': 800.0,
                    'reverb_wet': 0.0,
                    'reverb_room_size': 0.5,
                    'delay_time_ms': 100.0,
                    'delay_mix': 0.0,
                    'final_eq_gain_db': 0.0
                }
                
                di_test_track = AudioTrack(audio=di_test_audio, sr=di_track.sr, name="di_test")
                ref_test_track = AudioTrack(audio=ref_test_audio, sr=ref_track.sr, name="ref_test")
                
                processed_track = processor.process_with_custom_rig_and_post_fx(
                    di_test_track,
                    fx_nam_path=rig['fx_nam_path'],
                    amp_nam_path=rig['amp_nam_path'],
                    ir_path=rig['ir_path'],
                    gain_params=gain_params,
                    post_fx_params=post_fx_params_default,
                    ref_track=None  # No reference for processing, we'll compare after
                )
                
                # Calculate full G-Loss using SOTA Composite Loss (includes MR-STFT + Harmonic + Envelope)
                if self.composite_loss is not None:
                    g_loss, loss_components = self.composite_loss.compute(
                        processed_track.audio,
                        ref_test_audio,
                        sr=di_track.sr
                    )
                else:
                    # Fallback to old G-Loss if Composite Loss not available
                    g_loss = calculate_g_loss(
                        processed_track.audio,
                        ref_test_audio,
                        sr=di_track.sr
                    )
                
                # Store G-Loss in rig data
                rig['g_loss'] = g_loss
                rig['mel_loss'] = rig['loss']  # Keep original Mel-Loss for reference
                
                print(f"  [{idx}/10] [{rig['fx_nam_name']} -> {rig['amp_nam_name']} -> {rig['ir_name']}]: "
                      f"G-Loss = {g_loss:.6f} (Mel-Loss = {rig['mel_loss']:.6f})")
                
            except Exception as e:
                print(f"  [{idx}/10] [ERROR] Failed to evaluate: {e}")
                rig['g_loss'] = float('inf')  # Mark as invalid
                rig['mel_loss'] = rig['loss']
        
        # Step 3: Re-rank by G-Loss and select best
        print(f"\n{'='*70}")
        print(f"Step 3: Final Selection (G-Loss Ranking)")
        print(f"{'='*70}")
        
        # Sort by G-Loss (lower is better)
        top_10_candidates_sorted = sorted(top_10_candidates, key=lambda x: x.get('g_loss', float('inf')))
        
        # Select top_n based on G-Loss
        top_rigs = top_10_candidates_sorted[:top_n]
        best_rig = top_10_candidates_sorted[0] if top_10_candidates_sorted else None
        
        if best_rig:
            best_fx_nam_name = best_rig['fx_nam_name']
            best_amp_nam_name = best_rig['amp_nam_name']
            best_ir_name = best_rig['ir_name']
            best_g_loss = best_rig.get('g_loss', best_rig['loss'])
            best_mel_loss = best_rig.get('mel_loss', best_rig['loss'])
            best_gain = best_rig['input_gain_db']
            
            print(f"Best Rig (selected by G-Loss): [{best_fx_nam_name} -> {best_amp_nam_name} -> {best_ir_name}]")
            print(f"  G-Loss: {best_g_loss:.6f} (includes Harmonic Warmth + Dynamics)")
            print(f"  Mel-Loss: {best_mel_loss:.6f} (spectral similarity only)")
            print(f"  Gain: {best_gain:.2f}dB")
            
            # ── DIAGNOSTIC: Log input_gain_db value ─────────────────────────────
            print(f"  [DIAGNOSTIC] input_gain_db = {best_gain:.2f} dB (will be used as overdrive_db in VST)")
            if best_gain < 10.0:
                print(f"    ⚠️  WARNING: Very low input_gain_db! May result in insufficient distortion.")
            elif best_gain < 15.0:
                print(f"    ⚠️  INFO: Moderate input_gain_db. May be sufficient for some tones.")
            # #endregion
            print(f"\nTop {top_n} Combinations (ranked by G-Loss):")
            for idx, rig in enumerate(top_rigs, 1):
                g_loss_val = rig.get('g_loss', rig['loss'])
                mel_loss_val = rig.get('mel_loss', rig['loss'])
                print(f"  {idx}. [{rig['fx_nam_name']} -> {rig['amp_nam_name']} -> {rig['ir_name']}]: "
                      f"G-Loss = {g_loss_val:.6f} (Mel = {mel_loss_val:.6f}), Gain = {rig['input_gain_db']:.2f}dB")
            print(f"{'='*70}\n")
        
        # Prepare result
        # Use G-Loss if available, otherwise fall back to Mel-Loss
        final_loss = best_rig.get('g_loss', best_rig.get('loss', float('inf'))) if best_rig else float('inf')
        
        result = {
            'top_rigs': top_rigs,
            'best_rig': best_rig,
            'all_results': all_results,
            # Backward compatibility fields
            'fx_nam_path': best_rig['fx_nam_path'] if best_rig else None,
            'amp_nam_path': best_rig['amp_nam_path'] if best_rig else None,
            'nam_path': best_rig['amp_nam_path'] if best_rig else None,
            'ir_path': best_rig['ir_path'] if best_rig else None,
            'input_gain_db': best_rig['input_gain_db'] if best_rig else 0.0,
            'final_loss': final_loss,
            'g_loss': best_rig.get('g_loss') if best_rig else None,
            'mel_loss': best_rig.get('mel_loss', best_rig.get('loss')) if best_rig else None
        }
        
        return result
    
    def optimize(self, di_track: AudioTrack, ref_track: AudioTrack) -> Dict[str, Any]:
        """Optimize effect parameters to match reference track.
        
        Uses Grid Search over IR files, then optimizes parameters for each IR.
        Optimizes the following parameters:
        - Input Gain (dB) [-6.0 ... +12.0]
        - Distortion Drive (dB) [0.0 ... 30.0]
        - Reverb Wet Level [0.0 ... 0.5]
        - IR file selection (via Grid Search)
        
        Args:
            di_track: Input DI track to process
            ref_track: Reference track to match
        
        Returns:
            Dictionary with optimization results including:
            - 'success': bool
            - 'message': str
            - 'final_loss': float
            - 'iterations': int
            - 'best_parameters': dict with parameter values including 'ir_path'
        """
        if not CONVOLUTION_AVAILABLE:
            raise ValueError("pedalboard.Convolution is not available. Cannot perform IR optimization.")
        
        # Validate inputs
        if len(di_track.audio) == 0 or len(ref_track.audio) == 0:
            raise ValueError("Cannot optimize with empty audio tracks")
        
        if di_track.sr != ref_track.sr:
            raise ValueError(f"Sample rates must match: DI={di_track.sr}, Ref={ref_track.sr}")
        
        # Find IR files
        ir_files = self._find_ir_files()
        print(f"\nFound {len(ir_files)} IR file(s) in {self.ir_folder}")
        for ir_file in ir_files:
            print(f"  - {os.path.basename(ir_file)}")
        
        # Create test segments (first N seconds for speed)
        test_samples = int(self.test_duration_sec * di_track.sr)
        di_test_audio = di_track.audio[:min(test_samples, len(di_track.audio))].copy()
        ref_test_audio = ref_track.audio[:min(test_samples, len(ref_track.audio))].copy()
        
        # Parameter bounds (without lowpass_cutoff)
        # [input_gain, distortion_drive, reverb_wet]
        bounds = [
            (-6.0, 12.0),      # Input Gain (dB)
            (0.0, 30.0),       # Distortion Drive (dB)
            (0.0, 0.5)         # Reverb Wet Level
        ]
        
        # Initial guess (middle values)
        x0 = np.array([
            0.0,    # Input Gain: 0 dB
            10.0,   # Distortion Drive: 10 dB
            0.1     # Reverb Wet: 0.1
        ])
        
        # Grid Search: Test each IR file
        best_result: Tuple[float, np.ndarray, str] = (float('inf'), None, None)
        all_results = []
        
        print(f"\nStarting Grid Search over {len(ir_files)} IR file(s)...")
        print(f"Testing on first {self.test_duration_sec} seconds of audio")
        
        for ir_idx, ir_path in enumerate(ir_files, 1):
            ir_name = os.path.basename(ir_path)
            print(f"\n[{ir_idx}/{len(ir_files)}] Testing IR: {ir_name}")
            
            iteration_count = [0]  # Use list to allow modification in nested function
            
            def objective_function(x: np.ndarray) -> float:
                """Objective function: process audio and calculate distance.
                
                Args:
                    x: Parameter vector [input_gain, distortion_drive, reverb_wet]
                
                Returns:
                    Distance value (lower = better)
                """
                iteration_count[0] += 1
                
                try:
                    # Extract parameters
                    input_gain_db = float(x[0])
                    distortion_drive_db = float(x[1])
                    reverb_wet = float(x[2])
                    
                    # Process audio with these parameters
                    audio_processed = di_test_audio.copy().astype(np.float32)
                    
                    # Step 1: Input Gain
                    input_gain_linear = 10 ** (input_gain_db / 20.0)
                    audio_processed = audio_processed * input_gain_linear
                    
                    # Step 2: Compressor (fixed parameters)
                    compressor = Compressor(ratio=4.0, threshold_db=-20.0, attack_ms=5.0)
                    audio_processed = compressor(audio_processed, sample_rate=di_track.sr)
                    
                    # Step 3: Distortion
                    distortion = Distortion(drive_db=distortion_drive_db)
                    audio_processed = distortion(audio_processed, sample_rate=di_track.sr)
                    
                    # Step 4: IR Convolution (replaces LowPass Filter)
                    convolution = Convolution(impulse_response_filename=ir_path)
                    audio_processed = convolution(audio_processed, sample_rate=di_track.sr)
                    
                    # Step 5: Reverb
                    reverb = Reverb(room_size=0.5, wet_level=reverb_wet)
                    audio_processed = reverb(audio_processed, sample_rate=di_track.sr)
                    
                    # Step 6: Normalize to prevent clipping
                    max_val = np.max(np.abs(audio_processed))
                    if max_val > 0.95:
                        audio_processed = audio_processed / max_val * 0.95
                    
                    # Calculate distance using Mel-spectrogram
                    distance = calculate_audio_distance(audio_processed, ref_test_audio, sr=di_track.sr)
                    
                    # Print progress (every 5 iterations or first iteration)
                    if iteration_count[0] % 5 == 0 or iteration_count[0] == 1:
                        print(f"  Iteration {iteration_count[0]}: Loss {distance:.4f}, "
                              f"Params = [Gain:{input_gain_db:.1f}dB, "
                              f"Drive:{distortion_drive_db:.1f}dB, "
                              f"Rev:{reverb_wet:.2f}]")
                    
                    return distance
                    
                except Exception as e:
                    print(f"  Warning: Error in objective function: {e}")
                    return float('inf')
            
            # Run optimization for this IR
            print(f"  Starting optimization (max {self.max_iterations} iterations)...")
            result = optimize.minimize(
                objective_function,
                x0,
                method='Powell',
                bounds=bounds,
                options={
                    'maxiter': self.max_iterations,
                    'disp': False
                }
            )
            
            loss = float(result.fun)
            best_params = result.x.copy()
            
            print(f"  [OK] IR {ir_name}: Loss = {loss:.6f}, Iterations = {iteration_count[0]}")
            
            # Track best result
            all_results.append((loss, best_params, ir_path, iteration_count[0], result.success))
            
            if loss < best_result[0]:
                best_result = (loss, best_params, ir_path)
        
        # Extract best result
        best_loss, best_params, best_ir_path = best_result
        best_ir_name = os.path.basename(best_ir_path)
        
        # Find total iterations (sum of all IR optimizations)
        total_iterations = sum(r[3] for r in all_results)
        
        # Find best result details for success status
        best_result_details = min(all_results, key=lambda x: x[0])
        
        print(f"\n{'='*70}")
        print(f"Grid Search Complete!")
        print(f"{'='*70}")
        print(f"Best IR: {best_ir_name}")
        print(f"Best Loss: {best_loss:.6f}")
        print(f"Total Iterations: {total_iterations}")
        print(f"{'='*70}\n")
        
        # Prepare optimization info
        opt_info = {
            'success': best_result_details[4],  # success from best result
            'message': f"Grid Search completed. Best IR: {best_ir_name}",
            'final_loss': best_loss,
            'iterations': total_iterations,
            'best_parameters': {
                'input_gain_db': float(best_params[0]),
                'distortion_drive_db': float(best_params[1]),
                'reverb_wet': float(best_params[2]),
                'ir_path': best_ir_path
            }
        }
        
        return opt_info
    
    def apply_parameters(self, di_track: AudioTrack, parameters: Dict[str, float]) -> AudioTrack:
        """Apply optimized parameters to full audio track.
        
        Args:
            di_track: Full DI track to process
            parameters: Dictionary with parameter values:
                - 'input_gain_db': float
                - 'distortion_drive_db': float
                - 'reverb_wet': float
                - 'ir_path': str (path to IR file)
        
        Returns:
            Processed AudioTrack
        """
        if not CONVOLUTION_AVAILABLE:
            raise ValueError("pedalboard.Convolution is not available. Cannot apply IR parameters.")
        
        # Extract parameters
        input_gain_db = parameters['input_gain_db']
        distortion_drive_db = parameters['distortion_drive_db']
        reverb_wet = parameters['reverb_wet']
        ir_path = parameters.get('ir_path')
        
        if ir_path is None:
            raise ValueError("'ir_path' is required in parameters dictionary")
        
        if not os.path.exists(ir_path):
            raise ValueError(f"IR file not found: {ir_path}")
        
        # Process audio
        audio_processed = di_track.audio.copy().astype(np.float32)
        
        # Step 1: Input Gain
        input_gain_linear = 10 ** (input_gain_db / 20.0)
        audio_processed = audio_processed * input_gain_linear
        
        # Step 2: Compressor
        compressor = Compressor(ratio=4.0, threshold_db=-20.0, attack_ms=5.0)
        audio_processed = compressor(audio_processed, sample_rate=di_track.sr)
        
        # Step 3: Distortion
        distortion = Distortion(drive_db=distortion_drive_db)
        audio_processed = distortion(audio_processed, sample_rate=di_track.sr)
        
        # Step 4: IR Convolution (replaces LowPass Filter)
        convolution = Convolution(impulse_response_filename=ir_path)
        audio_processed = convolution(audio_processed, sample_rate=di_track.sr)
        
        # Step 5: Reverb
        reverb = Reverb(room_size=0.5, wet_level=reverb_wet)
        audio_processed = reverb(audio_processed, sample_rate=di_track.sr)
        
        # Step 6: Normalize to -1.0 dB Peak
        max_val = np.max(np.abs(audio_processed))
        if max_val > 0:
            target_linear = 10 ** (-1.0 / 20.0)
            audio_processed = audio_processed / max_val * target_linear
        
        audio_processed = audio_processed.astype(np.float32)
        
        return AudioTrack(
            audio=audio_processed,
            sr=di_track.sr,
            name=f"optimized_{di_track.name}"
        )
    
    def _apply_lowpass(self, audio: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
        """Apply low-pass filter using scipy.signal.
        
        Args:
            audio: Input audio array
            cutoff_hz: Cutoff frequency in Hz
            sr: Sample rate
            order: Filter order (default: 4)
        
        Returns:
            Filtered audio array
        """
        nyquist = sr / 2.0
        normalized_cutoff = cutoff_hz / nyquist
        
        if normalized_cutoff >= 1.0:
            return audio  # No filtering needed
        
        b, a = scipy_signal.butter(order, normalized_cutoff, btype='low')
        filtered = scipy_signal.lfilter(b, a, audio)
        
        return filtered
    
    def optimize_post_fx(self, di_track: AudioTrack, ref_track: AudioTrack) -> Dict[str, Any]:
        """Optimize only Post-FX parameters with fixed NAM/IR chain.
        
        Uses fixed equipment chain: DS1 -> 5150 BlockLetter -> BlendOfAll IR.
        Optimizes only Post-FX parameters:
        - Reverb Wet Level [0.0 ... 0.7]
        - Reverb Room Size [0.0 ... 1.0]
        - Delay Time (ms) [50 ... 500]
        - Delay Feedback/Mix [0.0 ... 0.5]
        - Final Match EQ Gain ± 3dB
        
        Args:
            di_track: Input DI track to process
            ref_track: Reference track to match
        
        Returns:
            Dictionary with optimization results including:
            - 'success': bool
            - 'message': str
            - 'final_loss': float
            - 'iterations': int
            - 'best_parameters': dict with Post-FX parameter values
        """
        # Validate inputs
        if len(di_track.audio) == 0 or len(ref_track.audio) == 0:
            raise ValueError("Cannot optimize with empty audio tracks")
        
        if di_track.sr != ref_track.sr:
            raise ValueError(f"Sample rates must match: DI={di_track.sr}, Ref={ref_track.sr}")
        
        # Create processor instance
        processor = ToneProcessor()
        
        # Create test segments (first N seconds for speed)
        test_samples = int(self.test_duration_sec * di_track.sr)
        di_test_audio = di_track.audio[:min(test_samples, len(di_track.audio))].copy()
        ref_test_audio = ref_track.audio[:min(test_samples, len(ref_track.audio))].copy()
        
        di_test_track = AudioTrack(audio=di_test_audio, sr=di_track.sr, name="di_test")
        ref_test_track = AudioTrack(audio=ref_test_audio, sr=ref_track.sr, name="ref_test")
        
        # Parameter bounds for Post-FX optimization (7 parameters: Pre-EQ + Post-FX)
        # [pre_eq_gain_db, pre_eq_freq_hz, reverb_wet, reverb_room_size, delay_time_ms, delay_mix, final_eq_gain_db]
        bounds = [
            (-12.0, 12.0),     # Pre-EQ Gain (dB)
            (400.0, 3000.0),   # Pre-EQ Frequency (Hz)
            (0.0, 0.7),        # Reverb Wet Level
            (0.0, 1.0),        # Reverb Room Size
            (50.0, 500.0),     # Delay Time (ms)
            (0.0, 0.5),        # Delay Feedback/Mix
            (-3.0, 3.0)        # Final Match EQ Gain (dB)
        ]
        
        # Initial guess (neutral Pre-EQ, middle values for Post-FX)
        x0 = np.array([
            0.0,    # Pre-EQ Gain: 0 dB (neutral)
            800.0,  # Pre-EQ Frequency: 800 Hz
            0.2,    # Reverb Wet: 0.2
            0.5,    # Reverb Room Size: 0.5
            100.0,  # Delay Time: 100 ms
            0.2,    # Delay Mix: 0.2
            0.0     # Final EQ Gain: 0 dB
        ])
        
        iteration_count = [0]  # Use list to allow modification in nested function
        
        def objective_function(x: np.ndarray) -> float:
            """Objective function: process audio with Post-FX parameters and calculate distance.
            
            Args:
                x: Parameter vector [pre_eq_gain_db, pre_eq_freq_hz, reverb_wet, reverb_room_size, delay_time_ms, delay_mix, final_eq_gain_db]
            
            Returns:
                Distance value (lower = better)
            """
            iteration_count[0] += 1
            
            try:
                # Extract parameters (7 parameters: Pre-EQ + Post-FX)
                pre_eq_gain_db = float(x[0])
                pre_eq_freq_hz = float(x[1])
                reverb_wet = float(x[2])
                reverb_room_size = float(x[3])
                delay_time_ms = float(x[4])
                delay_mix = float(x[5])
                final_eq_gain_db = float(x[6])
                
                # Prepare parameters
                gain_params = {
                    'input_gain_db': 0.0  # Fixed input gain (can be optimized separately if needed)
                }
                
                post_fx_params = {
                    'pre_eq_gain_db': pre_eq_gain_db,
                    'pre_eq_freq_hz': pre_eq_freq_hz,
                    'reverb_wet': reverb_wet,
                    'reverb_room_size': reverb_room_size,
                    'delay_time_ms': delay_time_ms,
                    'delay_mix': delay_mix,
                    'final_eq_gain_db': final_eq_gain_db
                }
                
                # Process audio with fixed NAM/IR chain and Post-FX parameters
                processed_track = processor.process_final_tune(
                    di_test_track,
                    gain_params=gain_params,
                    post_fx_params=post_fx_params,
                    ref_track=ref_test_track
                )
                
                # Calculate distance using Mel-spectrogram
                distance = calculate_audio_distance(
                    processed_track.audio,
                    ref_test_audio,
                    sr=di_track.sr
                )
                
                # Print progress (every 5 iterations or first iteration)
                if iteration_count[0] % 5 == 0 or iteration_count[0] == 1:
                    print(f"  Iteration {iteration_count[0]}: Loss {distance:.4f}, "
                          f"Params = [RevWet:{reverb_wet:.2f}, "
                          f"RevRoom:{reverb_room_size:.2f}, "
                          f"Delay:{delay_time_ms:.0f}ms, "
                          f"DelayMix:{delay_mix:.2f}, "
                          f"EQGain:{final_eq_gain_db:.1f}dB]")
                
                return distance
                
            except Exception as e:
                print(f"  Warning: Error in objective function: {e}")
                import traceback
                traceback.print_exc()
                return float('inf')
        
        # Run optimization
        print(f"\nStarting Post-FX optimization (fixed NAM/IR chain: DS1 -> 5150 -> BlendOfAll)...")
        print(f"Testing on first {self.test_duration_sec} seconds of audio")
        print(f"Optimizing 7 parameters: Pre-EQ (Gain, Freq), Reverb (Wet, Room), Delay (Time, Mix), Final EQ Gain")
        print(f"Starting optimization (max {self.max_iterations} iterations)...")
        
        result = optimize.minimize(
            objective_function,
            x0,
            method='Powell',
            bounds=bounds,
            options={
                'maxiter': self.max_iterations,
                'disp': False
            }
        )
        
        loss = float(result.fun)
        best_params = result.x.copy()
        
        print(f"\n{'='*70}")
        print(f"Post-FX Optimization Complete!")
        print(f"{'='*70}")
        print(f"Best Loss: {loss:.6f}")
        print(f"Total Iterations: {iteration_count[0]}")
        print(f"Best Parameters:")
        print(f"  Pre-EQ Gain: {best_params[0]:.2f} dB @ {best_params[1]:.0f} Hz")
        print(f"  Reverb Wet Level: {best_params[2]:.4f}")
        print(f"  Reverb Room Size: {best_params[3]:.4f}")
        print(f"  Delay Time: {best_params[4]:.2f} ms")
        print(f"  Delay Mix: {best_params[5]:.4f}")
        print(f"  Final EQ Gain: {best_params[6]:.2f} dB")
        print(f"{'='*70}\n")
        
        # Prepare optimization info
        opt_info = {
            'success': result.success,
            'message': "Post-FX optimization completed with fixed NAM/IR chain",
            'final_loss': loss,
            'iterations': iteration_count[0],
            'best_parameters': {
                'pre_eq_gain_db': float(best_params[0]),
                'pre_eq_freq_hz': float(best_params[1]),
                'reverb_wet': float(best_params[2]),
                'reverb_room_size': float(best_params[3]),
                'delay_time_ms': float(best_params[4]),
                'delay_mix': float(best_params[5]),
                'final_eq_gain_db': float(best_params[6])
            }
        }
        
        return opt_info
    
    def optimize_post_fx_for_rig(
        self,
        di_track: AudioTrack,
        ref_track: AudioTrack,
        fx_nam_path: Optional[str],
        amp_nam_path: Optional[str],
        ir_path: str,
        input_gain_db: float = 0.0,
        use_predictor: bool = False,
        predictor_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Optimize Post-FX parameters for a specific rig configuration.
        
        Uses custom NAM/IR chain and optimizes only Post-FX parameters:
        - Reverb Wet Level [0.0 ... 0.7]
        - Reverb Room Size [0.0 ... 1.0]
        - Delay Time (ms) [50 ... 500]
        - Delay Feedback/Mix [0.0 ... 0.5]
        - Final Match EQ Gain ± 3dB
        
        Args:
            di_track: Input DI track to process
            ref_track: Reference track to match
            fx_nam_path: Path to FX NAM model file (optional)
            amp_nam_path: Path to AMP NAM model file (optional)
            ir_path: Path to IR cabinet file
            input_gain_db: Input gain in dB (default: 0.0)
            use_predictor: If True, use neural network predictor instead of scipy.optimize (default: False)
            predictor_model_path: Path to trained predictor model (.pth). If None and use_predictor=True,
                                uses default path: "models/postfx_predictor.pth"
        
        Returns:
            Dictionary with optimization results including:
            - 'success': bool
            - 'message': str
            - 'final_loss': float
            - 'iterations': int (0 if using predictor)
            - 'best_parameters': dict with Post-FX parameter values
        """
        # Validate inputs
        if len(di_track.audio) == 0 or len(ref_track.audio) == 0:
            raise ValueError("Cannot optimize with empty audio tracks")
        
        if di_track.sr != ref_track.sr:
            raise ValueError(f"Sample rates must match: DI={di_track.sr}, Ref={ref_track.sr}")
        
        # Use neural network predictor if requested
        if use_predictor:
            return self._optimize_post_fx_with_predictor(
                di_track, ref_track, fx_nam_path, amp_nam_path, ir_path,
                input_gain_db, predictor_model_path
            )
        
        # Create processor instance
        processor = ToneProcessor()
        
        # Create test segments (first N seconds for speed)
        test_samples = int(self.test_duration_sec * di_track.sr)
        di_test_audio = di_track.audio[:min(test_samples, len(di_track.audio))].copy()
        ref_test_audio = ref_track.audio[:min(test_samples, len(ref_track.audio))].copy()
        
        di_test_track = AudioTrack(audio=di_test_audio, sr=di_track.sr, name="di_test")
        ref_test_track = AudioTrack(audio=ref_test_audio, sr=ref_track.sr, name="ref_test")
        
        # Parameter bounds for Post-FX optimization (7 parameters: Pre-EQ + Post-FX)
        # [pre_eq_gain_db, pre_eq_freq_hz, reverb_wet, reverb_room_size, delay_time_ms, delay_mix, final_eq_gain_db]
        bounds = [
            (-12.0, 12.0),     # Pre-EQ Gain (dB)
            (400.0, 3000.0),   # Pre-EQ Frequency (Hz)
            (0.0, 0.7),        # Reverb Wet Level
            (0.0, 1.0),        # Reverb Room Size
            (50.0, 500.0),     # Delay Time (ms)
            (0.0, 0.5),        # Delay Feedback/Mix
            (-3.0, 3.0)        # Final Match EQ Gain (dB)
        ]
        
        # Initial guess (neutral Pre-EQ, middle values for Post-FX)
        x0 = np.array([
            0.0,    # Pre-EQ Gain: 0 dB (neutral)
            800.0,  # Pre-EQ Frequency: 800 Hz
            0.2,    # Reverb Wet: 0.2
            0.5,    # Reverb Room Size: 0.5
            100.0,  # Delay Time: 100 ms
            0.2,    # Delay Mix: 0.2
            0.0     # Final EQ Gain: 0 dB
        ])
        
        iteration_count = [0]  # Use list to allow modification in nested function
        
        # Calculate initial error components for baseline
        print("\nCalculating initial error components...")
        gain_params_initial = {'input_gain_db': input_gain_db}
        post_fx_params_initial = {
            'pre_eq_gain_db': x0[0],
            'pre_eq_freq_hz': x0[1],
            'reverb_wet': x0[2],
            'reverb_room_size': x0[3],
            'delay_time_ms': x0[4],
            'delay_mix': x0[5],
            'final_eq_gain_db': x0[6]
        }
        
        initial_track = processor.process_with_custom_rig_and_post_fx(
            di_test_track,
            fx_nam_path=fx_nam_path,
            amp_nam_path=amp_nam_path,
            ir_path=ir_path,
            gain_params=gain_params_initial,
            post_fx_params=post_fx_params_initial,
            ref_track=ref_test_track
        )
        
        # Calculate initial error components
        ref_harmonics_initial = extract_harmonics(ref_test_audio, sr=di_track.sr, num_harmonics=10)
        initial_harmonics = extract_harmonics(initial_track.audio, sr=di_track.sr, num_harmonics=10)
        initial_harmonic_loss = calculate_harmonic_warmth(ref_harmonics_initial, initial_harmonics)
        initial_envelope_loss = calculate_envelope_loss(ref_test_audio, initial_track.audio, sr=di_track.sr)
        initial_spectral_shape_loss = calculate_spectral_shape_loss(initial_track.audio, ref_test_audio, sr=di_track.sr)
        initial_brightness_loss = calculate_brightness_loss(initial_track.audio, ref_test_audio, sr=di_track.sr)
        
        initial_error_components = {
            'harmonic_loss': float(initial_harmonic_loss),
            'envelope_loss': float(initial_envelope_loss),
            'spectral_shape_loss': float(initial_spectral_shape_loss),
            'brightness_loss': float(initial_brightness_loss)
        }
        
        print(f"Initial Error Components:")
        print(f"  Harmonic Loss: {initial_harmonic_loss:.4f}")
        print(f"  Envelope Loss: {initial_envelope_loss:.4f}")
        print(f"  Spectral Shape Loss: {initial_spectral_shape_loss:.4f}")
        print(f"  Brightness Loss: {initial_brightness_loss:.4f}")
        
        def objective_function(x: np.ndarray) -> np.ndarray:
            """Objective function: process audio with Post-FX parameters and calculate error vector.
            
            Returns a vector of 4 error components instead of a single scalar:
            - error_vector[0]: Harmonic Loss (warmth)
            - error_vector[1]: Envelope Loss (dynamics)
            - error_vector[2]: Spectral Shape Loss (timbre/EQ)
            - error_vector[3]: Brightness Loss (spectral centroid)
            
            Args:
                x: Parameter vector [pre_eq_gain_db, pre_eq_freq_hz, reverb_wet, reverb_room_size, delay_time_ms, delay_mix, final_eq_gain_db]
            
            Returns:
                Error vector with 4 components (lower = better for each)
            """
            iteration_count[0] += 1
            
            try:
                # Extract parameters (7 parameters: Pre-EQ + Post-FX)
                pre_eq_gain_db = float(x[0])
                pre_eq_freq_hz = float(x[1])
                reverb_wet = float(x[2])
                reverb_room_size = float(x[3])
                delay_time_ms = float(x[4])
                delay_mix = float(x[5])
                final_eq_gain_db = float(x[6])
                
                # Prepare parameters
                gain_params = {
                    'input_gain_db': input_gain_db
                }
                
                post_fx_params = {
                    'pre_eq_gain_db': pre_eq_gain_db,
                    'pre_eq_freq_hz': pre_eq_freq_hz,
                    'reverb_wet': reverb_wet,
                    'reverb_room_size': reverb_room_size,
                    'delay_time_ms': delay_time_ms,
                    'delay_mix': delay_mix,
                    'final_eq_gain_db': final_eq_gain_db
                }
                
                # Process audio with custom NAM/IR chain and Post-FX parameters
                processed_track = processor.process_with_custom_rig_and_post_fx(
                    di_test_track,
                    fx_nam_path=fx_nam_path,
                    amp_nam_path=amp_nam_path,
                    ir_path=ir_path,
                    gain_params=gain_params,
                    post_fx_params=post_fx_params,
                    ref_track=ref_test_track
                )
                
                # Calculate error vector components
                # 1. Harmonic Loss (warmth)
                ref_harmonics = extract_harmonics(ref_test_audio, sr=di_track.sr, num_harmonics=10)
                processed_harmonics = extract_harmonics(processed_track.audio, sr=di_track.sr, num_harmonics=10)
                harmonic_loss = calculate_harmonic_warmth(ref_harmonics, processed_harmonics)
                
                # 2. Envelope Loss (dynamics)
                envelope_loss = calculate_envelope_loss(ref_test_audio, processed_track.audio, sr=di_track.sr)
                
                # 3. Spectral Shape Loss (timbre/EQ)
                spectral_shape_loss = calculate_spectral_shape_loss(processed_track.audio, ref_test_audio, sr=di_track.sr)
                
                # 4. Brightness Loss (spectral centroid)
                brightness_loss = calculate_brightness_loss(processed_track.audio, ref_test_audio, sr=di_track.sr)
                
                # Combine into error vector
                error_vector = np.array([
                    harmonic_loss,
                    envelope_loss,
                    spectral_shape_loss,
                    brightness_loss
                ])
                
                # Print progress (every 5 iterations or first iteration)
                if iteration_count[0] % 5 == 0 or iteration_count[0] == 1:
                    total_error = np.sum(error_vector)
                    print(f"  Iteration {iteration_count[0]}: Total Error {total_error:.4f}, "
                          f"Components = [Harm:{harmonic_loss:.3f}, Env:{envelope_loss:.3f}, "
                          f"Spec:{spectral_shape_loss:.3f}, Bright:{brightness_loss:.3f}], "
                          f"Params = [PreEQ:{pre_eq_gain_db:.1f}dB@{pre_eq_freq_hz:.0f}Hz, "
                          f"RevWet:{reverb_wet:.2f}, Delay:{delay_time_ms:.0f}ms]")
                
                return error_vector
                
            except Exception as e:
                print(f"  Warning: Error in objective function: {e}")
                import traceback
                traceback.print_exc()
                # Return large error vector on failure
                return np.array([1.0, 1.0, 1.0, 1.0])
        
        # Run optimization
        fx_name = os.path.basename(fx_nam_path) if fx_nam_path else "(none)"
        amp_name = os.path.basename(amp_nam_path) if amp_nam_path else "(mock)"
        ir_name = os.path.basename(ir_path)
        
        print(f"\nStarting Post-FX optimization for rig: [{fx_name} -> {amp_name} -> {ir_name}]...")
        print(f"Testing on first {self.test_duration_sec} seconds of audio")
        print(f"Optimizing 7 parameters: Pre-EQ (Gain, Freq), Reverb (Wet, Room), Delay (Time, Mix), Final EQ Gain")
        print(f"Using Differentiable DSP with Adam Optimizer: gradient-based optimization")
        print(f"Starting optimization (max {self.max_iterations} iterations)...")
        
        # Check if PyTorch and DifferentiablePostFX are available
        if not TORCH_AVAILABLE or not DIFFERENTIABLE_POSTFX_AVAILABLE:
            print("Warning: PyTorch or DifferentiablePostFX not available. Falling back to scipy.optimize.")
            # Fallback to scipy.optimize
            lower_bounds = np.array([b[0] for b in bounds])
            upper_bounds = np.array([b[1] for b in bounds])
            result = optimize.least_squares(
                objective_function,
                x0,
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                max_nfev=self.max_iterations * 10,
                verbose=0
            )
            best_params = result.x.copy()
            final_cost = float(result.cost)
            optimization_success = result.success
            n_iterations = result.nfev
        else:
            # Use PyTorch Adam optimization
            # Step 1: Get base audio (processed through NAM/IR rig without Post-FX)
            print("\n[Step 1] Processing base audio through NAM/IR rig (non-differentiable)...")
            gain_params_base = {'input_gain_db': input_gain_db}
            post_fx_params_base = {
                'pre_eq_gain_db': 0.0,  # No Pre-EQ
                'pre_eq_freq_hz': 800.0,
                'reverb_wet': 0.0,  # No Reverb
                'reverb_room_size': 0.5,
                'delay_time_ms': 100.0,
                'delay_mix': 0.0,  # No Delay
                'final_eq_gain_db': 0.0  # No Final EQ
            }
            
            base_track = processor.process_with_custom_rig_and_post_fx(
                di_test_track,
                fx_nam_path=fx_nam_path,
                amp_nam_path=amp_nam_path,
                ir_path=ir_path,
                gain_params=gain_params_base,
                post_fx_params=post_fx_params_base,
                ref_track=None  # No reference for base processing
            )
            
            # Step 2: Initialize DifferentiablePostFX model
            print("[Step 2] Initializing DifferentiablePostFX model...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            postfx_model = DifferentiablePostFX(sample_rate=di_track.sr, device=device)
            
            # Initialize parameters with x0 values
            with torch.no_grad():
                postfx_model.pre_eq_gain_db.data = torch.tensor(x0[0], device=device)
                postfx_model.pre_eq_freq_hz.data = torch.tensor(x0[1], device=device)
                postfx_model.reverb_wet.data = torch.tensor(x0[2], device=device)
                postfx_model.reverb_room_size.data = torch.tensor(x0[3], device=device)
                postfx_model.delay_time_ms.data = torch.tensor(x0[4], device=device)
                postfx_model.delay_mix.data = torch.tensor(x0[5], device=device)
                postfx_model.final_eq_gain_db.data = torch.tensor(x0[6], device=device)
            
            # Step 3: Initialize Adam optimizer
            optimizer = torch.optim.Adam(postfx_model.parameters(), lr=0.01)
            
            # Step 4: Convert audio to tensors
            base_audio = base_track.audio.astype(np.float32)
            ref_audio = ref_test_audio.astype(np.float32)
            
            # Ensure same length
            min_len = min(len(base_audio), len(ref_audio))
            base_audio = base_audio[:min_len]
            ref_audio = ref_audio[:min_len]
            
            base_tensor = torch.from_numpy(base_audio).to(device)
            ref_tensor = torch.from_numpy(ref_audio).to(device)
            
            # Step 5: Training loop
            print("[Step 3] Starting Adam optimization loop...")
            best_loss = float('inf')
            best_params_dict = None
            training_losses = []
            
            num_iterations = min(self.max_iterations, 300)  # Cap at 300 iterations
            
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                # Forward pass: apply differentiable Post-FX
                processed_tensor = postfx_model(base_tensor)
                
                # Compute G-Loss
                loss, loss_components = compute_g_loss_tensor(
                    processed_tensor,
                    ref_tensor,
                    sr=di_track.sr,
                    device=device
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(postfx_model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Clamp parameters to valid ranges
                postfx_model._clamp_parameters()
                
                # Track best result
                current_loss = loss_components['total_loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params_dict = postfx_model.get_parameters_dict()
                
                training_losses.append(loss_components)
                
                # Log progress every 10 iterations
                if (iteration + 1) % 10 == 0 or iteration == 0:
                    print(f"  Iteration {iteration+1}/{num_iterations}: Total Loss = {current_loss:.6f}, "
                          f"Components = [Harm:{loss_components['harmonic_loss']:.4f}, "
                          f"Env:{loss_components['envelope_loss']:.4f}, "
                          f"Spec:{loss_components['spectral_shape_loss']:.4f}, "
                          f"Bright:{loss_components['brightness_loss']:.4f}], "
                          f"Params = [PreEQ:{postfx_model.pre_eq_gain_db.item():.1f}dB@{postfx_model.pre_eq_freq_hz.item():.0f}Hz, "
                          f"RevWet:{postfx_model.reverb_wet.item():.2f}, Delay:{postfx_model.delay_time_ms.item():.0f}ms]")
            
            # Step 6: Load best parameters
            if best_params_dict is not None:
                with torch.no_grad():
                    postfx_model.pre_eq_gain_db.data = torch.tensor(best_params_dict['pre_eq_gain_db'], device=device)
                    postfx_model.pre_eq_freq_hz.data = torch.tensor(best_params_dict['pre_eq_freq_hz'], device=device)
                    postfx_model.reverb_wet.data = torch.tensor(best_params_dict['reverb_wet'], device=device)
                    postfx_model.reverb_room_size.data = torch.tensor(best_params_dict['reverb_room_size'], device=device)
                    postfx_model.delay_time_ms.data = torch.tensor(best_params_dict['delay_time_ms'], device=device)
                    postfx_model.delay_mix.data = torch.tensor(best_params_dict['delay_mix'], device=device)
                    postfx_model.final_eq_gain_db.data = torch.tensor(best_params_dict['final_eq_gain_db'], device=device)
            
            # Extract final parameters
            final_params_dict = postfx_model.get_parameters_dict()
            best_params = np.array([
                final_params_dict['pre_eq_gain_db'],
                final_params_dict['pre_eq_freq_hz'],
                final_params_dict['reverb_wet'],
                final_params_dict['reverb_room_size'],
                final_params_dict['delay_time_ms'],
                final_params_dict['delay_mix'],
                final_params_dict['final_eq_gain_db']
            ])
            
            final_cost = best_loss
            optimization_success = True
            n_iterations = num_iterations
        
        # Calculate final error components using numpy functions for comparison
        print("\nCalculating final error components...")
        gain_params_final = {'input_gain_db': input_gain_db}
        post_fx_params_final = {
            'pre_eq_gain_db': float(best_params[0]),
            'pre_eq_freq_hz': float(best_params[1]),
            'reverb_wet': float(best_params[2]),
            'reverb_room_size': float(best_params[3]),
            'delay_time_ms': float(best_params[4]),
            'delay_mix': float(best_params[5]),
            'final_eq_gain_db': float(best_params[6])
        }
        
        final_track = processor.process_with_custom_rig_and_post_fx(
            di_test_track,
            fx_nam_path=fx_nam_path,
            amp_nam_path=amp_nam_path,
            ir_path=ir_path,
            gain_params=gain_params_final,
            post_fx_params=post_fx_params_final,
            ref_track=ref_test_track
        )
        
        # Calculate final error components using numpy functions
        ref_harmonics_final = extract_harmonics(ref_test_audio, sr=di_track.sr, num_harmonics=10)
        final_harmonics = extract_harmonics(final_track.audio, sr=di_track.sr, num_harmonics=10)
        final_harmonic_loss = calculate_harmonic_warmth(ref_harmonics_final, final_harmonics)
        final_envelope_loss = calculate_envelope_loss(ref_test_audio, final_track.audio, sr=di_track.sr)
        final_spectral_shape_loss = calculate_spectral_shape_loss(final_track.audio, ref_test_audio, sr=di_track.sr)
        final_brightness_loss = calculate_brightness_loss(final_track.audio, ref_test_audio, sr=di_track.sr)
        
        final_error_components = {
            'harmonic_loss': float(final_harmonic_loss),
            'envelope_loss': float(final_envelope_loss),
            'spectral_shape_loss': float(final_spectral_shape_loss),
            'brightness_loss': float(final_brightness_loss)
        }
        
        print(f"\n{'='*70}")
        print(f"Post-FX Optimization Complete!")
        print(f"{'='*70}")
        print(f"Rig: [{fx_name} -> {amp_name} -> {ir_name}]")
        print(f"Total Iterations: {n_iterations}")
        print(f"Final Loss: {final_cost:.6f}")
        print(f"\nError Components Comparison:")
        print(f"  Harmonic Loss:      {initial_harmonic_loss:.4f} -> {final_harmonic_loss:.4f}")
        print(f"  Envelope Loss:      {initial_envelope_loss:.4f} -> {final_envelope_loss:.4f}")
        print(f"  Spectral Shape Loss: {initial_spectral_shape_loss:.4f} -> {final_spectral_shape_loss:.4f}")
        print(f"  Brightness Loss:    {initial_brightness_loss:.4f} -> {final_brightness_loss:.4f}")
        print(f"\nBest Parameters:")
        print(f"  Pre-EQ Gain: {best_params[0]:.2f} dB @ {best_params[1]:.0f} Hz")
        print(f"  Reverb Wet Level: {best_params[2]:.4f}")
        print(f"  Reverb Room Size: {best_params[3]:.4f}")
        print(f"  Delay Time: {best_params[4]:.2f} ms")
        print(f"  Delay Mix: {best_params[5]:.4f}")
        print(f"  Final EQ Gain: {best_params[6]:.2f} dB")
        print(f"{'='*70}\n")
        
        # Prepare optimization info
        opt_info = {
            'success': optimization_success,
            'message': f"Post-FX optimization completed for rig: [{fx_name} -> {amp_name} -> {ir_name}]",
            'final_loss': final_cost,
            'final_cost': final_cost,
            'iterations': n_iterations,
            'initial_error_components': initial_error_components,
            'final_error_components': final_error_components,
            'best_parameters': {
                'pre_eq_gain_db': float(best_params[0]),
                'pre_eq_freq_hz': float(best_params[1]),
                'reverb_wet': float(best_params[2]),
                'reverb_room_size': float(best_params[3]),
                'delay_time_ms': float(best_params[4]),
                'delay_mix': float(best_params[5]),
                'final_eq_gain_db': float(best_params[6])
            },
            'rig_config': {
                'fx_nam_path': fx_nam_path,
                'amp_nam_path': amp_nam_path,
                'ir_path': ir_path,
                'input_gain_db': input_gain_db
            }
        }
        
        return opt_info
    
    def _optimize_post_fx_with_predictor(
        self,
        di_track: AudioTrack,
        ref_track: AudioTrack,
        fx_nam_path: Optional[str],
        amp_nam_path: Optional[str],
        ir_path: str,
        input_gain_db: float,
        predictor_model_path: Optional[str]
    ) -> Dict[str, Any]:
        """Optimize Post-FX parameters using neural network predictor (fast method).
        
        Args:
            di_track: Input DI track to process
            ref_track: Reference track to match
            fx_nam_path: Path to FX NAM model file (optional)
            amp_nam_path: Path to AMP NAM model file (optional)
            ir_path: Path to IR cabinet file
            input_gain_db: Input gain in dB
            predictor_model_path: Path to trained predictor model (.pth)
        
        Returns:
            Dictionary with optimization results
        """
        import time
        import librosa
        
        try:
            from src.core.ddsp_processor import PostFXPredictorTrainer
        except ImportError:
            raise RuntimeError("PostFXPredictorTrainer not available. PyTorch may not be installed.")
        
        start_time = time.time()
        
        # Default model path
        if predictor_model_path is None:
            predictor_model_path = "models/postfx_predictor.pth"
        
        if not os.path.exists(predictor_model_path):
            raise FileNotFoundError(
                f"Predictor model not found: {predictor_model_path}\n"
                f"Please train the model first using run_predictor_test.py"
            )
        
        # Load predictor
        predictor_trainer = PostFXPredictorTrainer()
        predictor_trainer.load_model(predictor_model_path)
        
        # Create test segments
        test_samples = int(self.test_duration_sec * di_track.sr)
        ref_test_audio = ref_track.audio[:min(test_samples, len(ref_track.audio))].copy()
        ref_test_track = AudioTrack(audio=ref_test_audio, sr=ref_track.sr, name="ref_test")
        
        # Compute Mel-Spectrogram from reference
        mel_spec = librosa.feature.melspectrogram(
            y=ref_test_audio,
            sr=ref_track.sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range (same as training)
        mel_spec_normalized = (mel_spec_db + 80) / 80.0
        mel_spec_normalized = np.clip(mel_spec_normalized, 0.0, 1.0)
        
        # Predict parameters (now includes Pre-EQ)
        predicted_params = predictor_trainer.predict(mel_spec_normalized)
        
        # Extract parameters (7 parameters: Pre-EQ + Post-FX)
        pre_eq_gain_db = float(predicted_params[0])
        pre_eq_freq_hz = float(predicted_params[1])
        reverb_wet = float(predicted_params[2])
        reverb_room_size = float(predicted_params[3])
        delay_time_ms = float(predicted_params[4])
        delay_mix = float(predicted_params[5])
        final_eq_gain_db = float(predicted_params[6])
        
        # Process audio with predicted parameters to calculate final G-Loss
        processor = ToneProcessor()
        gain_params = {'input_gain_db': input_gain_db}
        post_fx_params = {
            'pre_eq_gain_db': pre_eq_gain_db,
            'pre_eq_freq_hz': pre_eq_freq_hz,
            'reverb_wet': reverb_wet,
            'reverb_room_size': reverb_room_size,
            'delay_time_ms': delay_time_ms,
            'delay_mix': delay_mix,
            'final_eq_gain_db': final_eq_gain_db
        }
        
        di_test_audio = di_track.audio[:min(test_samples, len(di_track.audio))].copy()
        di_test_track = AudioTrack(audio=di_test_audio, sr=di_track.sr, name="di_test")
        
        processed_track = processor.process_with_custom_rig_and_post_fx(
            di_test_track,
            fx_nam_path=fx_nam_path,
            amp_nam_path=amp_nam_path,
            ir_path=ir_path,
            gain_params=gain_params,
            post_fx_params=post_fx_params,
            ref_track=ref_test_track
        )
        
        # Calculate final G-Loss using SOTA Composite Loss
        if self.composite_loss is not None:
            final_g_loss, loss_components = self.composite_loss.compute(
                processed_track.audio,
                ref_test_audio,
                sr=di_track.sr
            )
        else:
            # Fallback to old G-Loss if Composite Loss not available
            final_g_loss = calculate_g_loss(
                processed_track.audio,
                ref_test_audio,
                sr=di_track.sr
            )
        
        elapsed_time = time.time() - start_time
        
        fx_name = os.path.basename(fx_nam_path) if fx_nam_path else "(none)"
        amp_name = os.path.basename(amp_nam_path) if amp_nam_path else "(mock)"
        ir_name = os.path.basename(ir_path)
        
        print(f"\n{'='*70}")
        print(f"Post-FX Optimization (Neural Predictor)")
        print(f"{'='*70}")
        print(f"Rig: [{fx_name} -> {amp_name} -> {ir_name}]")
        print(f"Prediction Time: {elapsed_time:.3f} seconds")
        print(f"Final G-Loss: {final_g_loss:.6f}")
        print(f"Predicted Parameters:")
        print(f"  Pre-EQ Gain: {pre_eq_gain_db:.2f} dB @ {pre_eq_freq_hz:.0f} Hz")
        print(f"  Reverb Wet Level: {reverb_wet:.4f}")
        print(f"  Reverb Room Size: {reverb_room_size:.4f}")
        print(f"  Delay Time: {delay_time_ms:.2f} ms")
        print(f"  Delay Mix: {delay_mix:.4f}")
        print(f"  Final EQ Gain: {final_eq_gain_db:.2f} dB")
        print(f"{'='*70}\n")
        
        return {
            'success': True,
            'message': f"Post-FX optimization completed using neural predictor for rig: [{fx_name} -> {amp_name} -> {ir_name}]",
            'final_loss': final_g_loss,
            'iterations': 0,  # No iterations for predictor
            'prediction_time_sec': elapsed_time,
            'best_parameters': {
                'pre_eq_gain_db': pre_eq_gain_db,
                'pre_eq_freq_hz': pre_eq_freq_hz,
                'reverb_wet': reverb_wet,
                'reverb_room_size': reverb_room_size,
                'delay_time_ms': delay_time_ms,
                'delay_mix': delay_mix,
                'final_eq_gain_db': final_eq_gain_db
            },
            'rig_config': {
                'fx_nam_path': fx_nam_path,
                'amp_nam_path': amp_nam_path,
                'ir_path': ir_path,
                'input_gain_db': input_gain_db
            }
        }
    
    def optimize_universal(
        self, 
        di_track: AudioTrack, 
        ref_track: AudioTrack,
        use_full_search: bool = False,
        force_high_gain: bool = False
    ) -> Dict[str, Any]:
        """Universal optimization: Smart Sommelier + Deep Post-FX optimization.
        
        🍷 "Умный Сомелье" - Интеллектуальная архитектура поиска:
        1. Этап 1: Анализ референса и интеллектуальная фильтрация моделей
           - Определяет "стиль" референса (clean/crunch/high-gain)
           - Выбирает релевантные модели из ВСЕЙ библиотеки (261 моделей)
        2. Этап 2: Двухуровневый Grid Search
           - Уровень 1: Найти TOP-2 усилителя БЕЗ педали
           - Уровень 2: Протестировать TOP-2 с КАЖДОЙ педалью
           - Ускорение до 5x по сравнению с брутфорсом
        3. Этап 3: Deep Post-FX Optimization
           - Тонкая настройка Pre-EQ, Reverb, Delay
           - Использует scipy.optimize.least_squares
        
        Args:
            di_track: Input DI track to process
            ref_track: Reference track to match
            use_full_search: Ignored (Smart Sommelier always uses intelligent search)
            force_high_gain: If True, skip spectral analysis and force high-gain mode (exclude clean amps)
        
        Returns:
            Dictionary with complete optimization results including:
            - 'grid_search_results': Results from Smart Sommelier search
            - 'post_fx_results': Results from Post-FX optimization
            - 'final_track': Processed AudioTrack
            - 'discovered_rig': Dictionary with discovered equipment (FX, AMP, IR names and paths)
            - 'final_loss': Final optimization loss
            - 'best_parameters': Optimized Post-FX parameters
        """
        print("=" * 70)
        print("[Universal Optimization] Smart Sommelier + Deep Post-FX Tuning")
        print("=" * 70)
        
        # Этап 1-2: Smart Sommelier - интеллектуальный поиск оборудования
        # Анализирует референс, выбирает релевантные модели, двухуровневый поиск
        grid_results = self.find_best_rig_smart(di_track, ref_track, top_n=3, force_high_gain=force_high_gain)
        
        if not grid_results.get('best_rig'):
            raise ValueError("Smart Sommelier failed: No valid rig configurations found")
        
        top_rigs = grid_results.get('top_rigs', [])
        best_rig = grid_results['best_rig']
        ref_analysis = grid_results.get('ref_analysis', {})
        
        # Show Smart Sommelier summary
        print(f"\n{'='*70}")
        print(f"[Steps 1-2 Complete] Smart Sommelier found the best rig!")
        print(f"{'='*70}")
        print(f"  Reference classified as: {ref_analysis.get('aggression_level', 'unknown').upper()}")
        print(f"  Combinations tested: {grid_results.get('total_tested', 0)}")
        print(f"  Speedup: {grid_results.get('speedup', 1):.1f}x")
        print(f"\n  Best combination:")
        print(f"    [{best_rig['fx_nam_name']} -> {best_rig['amp_nam_name']} -> {best_rig['ir_name']}]")
        print(f"    Loss: {best_rig.get('g_loss', best_rig.get('loss', 0.0)):.4f}, Gain: {best_rig['input_gain_db']:.0f}dB")
        
        # ── DIAGNOSTIC LOGGING FOR OVERDRIVE ──────────────────────────────────
        input_gain_db = best_rig['input_gain_db']
        aggression_level = ref_analysis.get('aggression_level', 'unknown').lower()
        
        print(f"\n  [DIAGNOSTIC] Overdrive Parameter Analysis:")
        print(f"    - input_gain_db (used as overdrive_db in VST): {input_gain_db:.2f} dB")
        print(f"    - Reference aggression level: {aggression_level.upper()}")
        
        # Check if overdrive is suspiciously low for high-gain tones
        if aggression_level in ['high-gain', 'metal', 'aggressive']:
            if input_gain_db < 15.0:
                print(f"    ⚠️  WARNING: input_gain_db ({input_gain_db:.2f} dB) is low for {aggression_level} tone!")
                print(f"       Expected range: 15-30 dB for metal/high-gain tones")
                print(f"       This might result in insufficient distortion in the VST plugin")
            elif input_gain_db < 20.0:
                print(f"    ⚠️  INFO: input_gain_db ({input_gain_db:.2f} dB) is moderate for {aggression_level} tone")
                print(f"       May be sufficient, but higher values (20-30 dB) typically produce better distortion")
            else:
                print(f"    ✓  input_gain_db ({input_gain_db:.2f} dB) is in good range for {aggression_level} tone")
        elif aggression_level in ['clean', 'crunch']:
            if input_gain_db > 20.0:
                print(f"    ⚠️  INFO: input_gain_db ({input_gain_db:.2f} dB) is high for {aggression_level} tone")
                print(f"       This is expected if reference has some distortion")
            else:
                print(f"    ✓  input_gain_db ({input_gain_db:.2f} dB) is appropriate for {aggression_level} tone")
        else:
            print(f"    - input_gain_db ({input_gain_db:.2f} dB) - no specific recommendation")
        
        # Log all top rigs' input_gain_db values for comparison
        if top_rigs:
            print(f"\n  [DIAGNOSTIC] Top {len(top_rigs)} rigs input_gain_db values:")
            for idx, rig in enumerate(top_rigs[:5], 1):  # Show top 5
                print(f"    {idx}. {rig['input_gain_db']:.2f} dB - [{rig['fx_nam_name']} -> {rig['amp_nam_name']}]")
        # #endregion
        
        # Step 3: Deep Post-FX Optimization
        print(f"\n{'='*70}")
        print("[Step 3: Final Tuning] Optimizing Post-FX (Pre-EQ, Reverb, Delay)...")
        print("Using 'Sighted' Optimizer: minimizing 4-component error vector")
        print("This may take 2-3 minutes...")
        
        # Use full test duration and iterations for Post-FX optimization
        post_fx_optimizer = ToneOptimizer(
            test_duration_sec=self.test_duration_sec,
            max_iterations=self.max_iterations,
            fast_grid_search=False
        )
        
        # Всегда использовать "зрячий" оптимизатор (scipy.optimize.least_squares)
        post_fx_results = post_fx_optimizer.optimize_post_fx_for_rig(
            di_track=di_track,
            ref_track=ref_track,
            fx_nam_path=best_rig['fx_nam_path'],
            amp_nam_path=best_rig['amp_nam_path'],
            ir_path=best_rig['ir_path'],
            input_gain_db=best_rig['input_gain_db'],
            use_predictor=False  # Всегда использовать "зрячий" оптимизатор
        )
        
        print(f"\n[Step 3 Complete] Post-FX optimization finished")
        print(f"Final Loss: {post_fx_results['final_loss']:.6f}")
        
        # Apply optimized parameters to full track
        print("\n[Applying] Applying optimized parameters to full track...")
        processor = ToneProcessor()
        
        # ── DIAGNOSTIC: Log final input_gain_db before processing ──────────────
        final_input_gain_db = best_rig['input_gain_db']
        print(f"\n[DIAGNOSTIC] Final parameters before processing:")
        print(f"  input_gain_db (overdrive_db): {final_input_gain_db:.2f} dB")
        print(f"  This value will be written to JSON and used as overdrive_db in VST plugin")
        if final_input_gain_db < 10.0:
            print(f"  ⚠️  WARNING: input_gain_db is very low ({final_input_gain_db:.2f} dB)")
            print(f"     For metal/high-gain tones, expected range is 15-30 dB")
        # #endregion
        
        gain_params = {
            'input_gain_db': final_input_gain_db
        }
        
        post_fx_params = post_fx_results['best_parameters']
        
        final_track = processor.process_with_custom_rig_and_post_fx(
            di_track=di_track,
            fx_nam_path=best_rig['fx_nam_path'],
            amp_nam_path=best_rig['amp_nam_path'],
            ir_path=best_rig['ir_path'],
            gain_params=gain_params,
            post_fx_params=post_fx_params,
            ref_track=ref_track
        )
        
        print(f"[OK] Processing complete")
        
        # Prepare final result
        result = {
            'grid_search_results': grid_results,
            'post_fx_results': post_fx_results,
            'final_track': final_track,
            'discovered_rig': {
                'fx_nam_name': best_rig['fx_nam_name'],
                'amp_nam_name': best_rig['amp_nam_name'],
                'ir_name': best_rig['ir_name'],
                'fx_nam_path': best_rig['fx_nam_path'],
                'amp_nam_path': best_rig['amp_nam_path'],
                'ir_path': best_rig['ir_path']
            },
            'final_loss': post_fx_results['final_loss'],
            'best_parameters': post_fx_params,
            'post_fx_params': post_fx_params  # Alias for UI convenience
        }
        
        return result
    
    def optimize_waveshaper(
        self,
        di_track: AudioTrack,
        ref_track: AudioTrack,
        golden_rig_config: Optional[Dict[str, str]] = None,
        iterations: int = 150,
        learning_rate: float = 0.001,
        output_weights_path: str = "waveshaper_harmonic_corrected.pth"
    ) -> Dict[str, Any]:
        """Optimize DifferentiableWaveshaper to minimize Harmonic Loss.
        
        Uses PyTorch Adam optimizer to train a waveshaper that corrects harmonic content
        after the Golden Rig processing chain. The waveshaper learns to adjust Even/Odd
        harmonic ratio to match reference audio.
        
        Args:
            di_track: Input DI track to process
            ref_track: Reference track to match
            golden_rig_config: Dictionary with Golden Rig configuration:
                - 'fx_nam_path': str (path to FX NAM model, optional)
                - 'amp_nam_path': str (path to AMP NAM model, optional)
                - 'ir_path': str (path to IR file)
                - 'input_gain_db': float (input gain in dB, default: 0.0)
                If None, uses default Golden Rig: DS1 -> 5150 BlockLetter -> BlendOfAll
            iterations: Number of optimization iterations (default: 150)
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            output_weights_path: Path to save optimized weights (default: "waveshaper_harmonic_corrected.pth")
        
        Returns:
            Dictionary with optimization results:
            - 'success': bool
            - 'message': str
            - 'harmonic_loss_before': float
            - 'harmonic_loss_after': float
            - 'improvement_percent': float
            - 'iterations': int
            - 'weights_path': str
        """
        if not WAVESHAPER_AVAILABLE:
            raise RuntimeError(
                "DifferentiableWaveshaperProcessor is not available. "
                "PyTorch may not be installed. Install with: pip install torch>=2.0.0"
            )
        
        # Validate inputs
        if len(di_track.audio) == 0 or len(ref_track.audio) == 0:
            raise ValueError("Cannot optimize with empty audio tracks")
        
        if di_track.sr != ref_track.sr:
            raise ValueError(f"Sample rates must match: DI={di_track.sr}, Ref={ref_track.sr}")
        
        # Use default Golden Rig if not provided
        if golden_rig_config is None:
            golden_rig_config = {
                'fx_nam_path': "assets/nam_models/Keith B DS1_g6_t5.nam",
                'amp_nam_path': "assets/nam_models/Helga B 5150 BlockLetter - NoBoost.nam",
                'ir_path': "assets/impulse_responses/BlendOfAll.wav",
                'input_gain_db': 0.0
            }
        
        # Verify Golden Rig files exist
        fx_nam_path = golden_rig_config.get('fx_nam_path')
        amp_nam_path = golden_rig_config.get('amp_nam_path')
        ir_path = golden_rig_config.get('ir_path')
        input_gain_db = golden_rig_config.get('input_gain_db', 0.0)
        
        if ir_path and not os.path.exists(ir_path):
            raise FileNotFoundError(f"IR file not found: {ir_path}")
        if fx_nam_path and not os.path.exists(fx_nam_path):
            print(f"Warning: FX NAM file not found: {fx_nam_path}, will skip FX")
            fx_nam_path = None
        if amp_nam_path and not os.path.exists(amp_nam_path):
            print(f"Warning: AMP NAM file not found: {amp_nam_path}, will use mock mode")
            amp_nam_path = None
        
        # Create processor
        processor = ToneProcessor()
        
        # Create test segments (first N seconds for speed)
        test_samples = int(self.test_duration_sec * di_track.sr)
        di_test_audio = di_track.audio[:min(test_samples, len(di_track.audio))].copy()
        ref_test_audio = ref_track.audio[:min(test_samples, len(ref_track.audio))].copy()
        
        di_test_track = AudioTrack(audio=di_test_audio, sr=di_track.sr, name="di_test")
        ref_test_track = AudioTrack(audio=ref_test_audio, sr=ref_track.sr, name="ref_test")
        
        # Step 1: Process through Golden Rig WITHOUT waveshaper to get baseline
        print("\n" + "="*70)
        print("Step 1: Processing through Golden Rig (baseline, no waveshaper)")
        print("="*70)
        
        gain_params = {'input_gain_db': input_gain_db}
        post_fx_params_default = {
            'pre_eq_gain_db': 0.0,
            'pre_eq_freq_hz': 800.0,
            'reverb_wet': 0.0,
            'reverb_room_size': 0.5,
            'delay_time_ms': 100.0,
            'delay_mix': 0.0,
            'final_eq_gain_db': 0.0
        }
        
        baseline_track = processor.process_with_custom_rig_and_post_fx(
            di_test_track,
            fx_nam_path=fx_nam_path,
            amp_nam_path=amp_nam_path,
            ir_path=ir_path,
            gain_params=gain_params,
            post_fx_params=post_fx_params_default,
            ref_track=None  # No reference for processing
        )
        
        # Calculate Harmonic Loss BEFORE optimization
        ref_harmonics = extract_harmonics(ref_test_audio, sr=di_track.sr, num_harmonics=10)
        baseline_harmonics = extract_harmonics(baseline_track.audio, sr=di_track.sr, num_harmonics=10)
        harmonic_loss_before = calculate_harmonic_warmth(ref_harmonics, baseline_harmonics)
        
        print(f"Baseline Harmonic Loss: {harmonic_loss_before:.6f}")
        
        # Step 2: Initialize Waveshaper and Optimizer
        print("\n" + "="*70)
        print(f"Step 2: Optimizing DifferentiableWaveshaper ({iterations} iterations)")
        print("="*70)
        
        waveshaper = DifferentiableWaveshaperProcessor(learning_rate=learning_rate)
        
        # Training loop
        best_harmonic_loss = harmonic_loss_before
        best_weights = None
        
        print(f"Starting optimization (target: minimize Harmonic Loss)...")
        print(f"Learning rate: {learning_rate}, Device: {waveshaper.device}")
        
        # Pre-process Golden Rig output once (it's not differentiable, so we do it outside the loop)
        # Use process_final_tune to match the script's method (ensures consistency)
        # Note: process_final_tune uses hardcoded paths, so we need to verify they match
        rough_track = processor.process_final_tune(
            di_test_track,
            gain_params=gain_params,
            post_fx_params=post_fx_params_default,
            ref_track=None,  # No reference for baseline
            waveshaper=None  # No waveshaper for baseline
        )
        
        # Convert to tensors once
        import torch
        rough_audio = rough_track.audio.astype(np.float32)
        rough_tensor = torch.from_numpy(rough_audio).to(waveshaper.device)
        ref_tensor = torch.from_numpy(ref_test_audio.astype(np.float32)).to(waveshaper.device)
        
        # Ensure same length
        min_len = min(len(rough_tensor), len(ref_tensor))
        rough_tensor = rough_tensor[:min_len]
        ref_tensor = ref_tensor[:min_len]
        
        for iteration in range(iterations):
            # Forward pass through waveshaper
            waveshaper.model.train()
            waveshaper.optimizer.zero_grad()
            
            processed_tensor = waveshaper.model(rough_tensor)
            
            # Use L1 loss as primary objective
            # Harmonic correction will emerge naturally as the model learns to match the reference
            loss = torch.nn.functional.l1_loss(processed_tensor, ref_tensor)
            
            # Add MSE component for smoother optimization
            mse_loss = torch.nn.functional.mse_loss(processed_tensor, ref_tensor)
            
            # Combined loss: 70% L1, 30% MSE
            combined_loss = 0.7 * loss + 0.3 * mse_loss
            
            # Backward pass
            combined_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(waveshaper.model.parameters(), max_norm=1.0)
            
            # Update weights
            waveshaper.optimizer.step()
            
            # Calculate Harmonic Loss for monitoring (detached, not part of gradient)
            if (iteration + 1) % 10 == 0 or iteration == 0:
                with torch.no_grad():
                    processed_audio = processed_tensor.detach().cpu().numpy()
                    processed_harmonics = extract_harmonics(processed_audio, sr=di_track.sr, num_harmonics=10)
                    harmonic_loss = calculate_harmonic_warmth(ref_harmonics, processed_harmonics)
                    
                    # Track best result
                    if harmonic_loss < best_harmonic_loss:
                        best_harmonic_loss = harmonic_loss
                        # Save best weights
                        best_weights = {k: v.cpu().clone() for k, v in waveshaper.model.state_dict().items()}
                    
                    print(f"  Iteration {iteration+1}/{iterations}: "
                          f"Harmonic Loss = {harmonic_loss:.6f}, "
                          f"Loss = {float(combined_loss.item()):.6f}")
        
        # Step 3: Load best weights and final evaluation
        print("\n" + "="*70)
        print("Step 3: Final Evaluation")
        print("="*70)
        
        if best_weights is not None:
            waveshaper.model.load_state_dict(best_weights)
        
        waveshaper.model.eval()
        
        # Process final result using same method as baseline
        final_rough_track = processor.process_final_tune(
            di_test_track,
            gain_params=gain_params,
            post_fx_params=post_fx_params_default,
            ref_track=None,  # No reference for final processing
            waveshaper=None  # We'll apply waveshaper manually
        )
        
        final_processed_audio = waveshaper.process_audio(final_rough_track.audio, sample_rate=di_track.sr)
        
        # Calculate final Harmonic Loss
        final_harmonics = extract_harmonics(final_processed_audio, sr=di_track.sr, num_harmonics=10)
        harmonic_loss_after = calculate_harmonic_warmth(ref_harmonics, final_harmonics)
        
        # Calculate improvement
        improvement_percent = ((harmonic_loss_before - harmonic_loss_after) / harmonic_loss_before) * 100.0
        
        print(f"Harmonic Loss before: {harmonic_loss_before:.6f}")
        print(f"Harmonic Loss after:  {harmonic_loss_after:.6f}")
        print(f"Improvement: {improvement_percent:.2f}%")
        
        # Save weights
        waveshaper.save_weights(output_weights_path)
        print(f"\nOptimized weights saved to: {output_weights_path}")
        
        return {
            'success': True,
            'message': f"Waveshaper optimization completed. Harmonic Loss: {harmonic_loss_before:.6f} -> {harmonic_loss_after:.6f}",
            'harmonic_loss_before': float(harmonic_loss_before),
            'harmonic_loss_after': float(harmonic_loss_after),
            'improvement_percent': float(improvement_percent),
            'iterations': iterations,
            'weights_path': output_weights_path
        }
