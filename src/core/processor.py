"""
Audio processing module for applying effects chain using pedalboard.
Implements Day 5: Full Processing Chain & Saturation.
"""

import os
import tempfile
from typing import Optional, Dict

import numpy as np
from pedalboard import Pedalboard, Compressor, Distortion, Reverb
from scipy import signal as scipy_signal
import soundfile as sf

from src.core.analysis import ToneFeatures, calculate_drive_intensity, calculate_distance
from src.core.io import AudioTrack
from src.core.nam_processor import NAMProcessor

# Try to import optional pedalboard plugins
try:
    from pedalboard import NoiseGate
    NOISEGATE_AVAILABLE = True
except ImportError:
    NOISEGATE_AVAILABLE = False

try:
    from pedalboard import Convolution
    CONVOLUTION_AVAILABLE = True
except ImportError:
    CONVOLUTION_AVAILABLE = False

try:
    from pedalboard import Delay
    DELAY_AVAILABLE = True
except ImportError:
    DELAY_AVAILABLE = False

# Try to import DifferentiableWaveshaper
try:
    from src.core.ddsp_processor import DifferentiableWaveshaperProcessor
    WAVESHAPER_AVAILABLE = True
except ImportError:
    WAVESHAPER_AVAILABLE = False
    DifferentiableWaveshaperProcessor = None


class ToneProcessor:
    """Processes audio through a chain of effects to match reference tone.
    
    The processing chain: Pre-Gain (+12dB) -> Compressor (4:1) -> Distortion (adaptive)
    -> Match EQ -> Delay (optional) -> Reverb (adaptive) -> Post-Gain (Normalize to -1dB)
    
    Distortion and Reverb parameters are adjusted based on drive_intensity (0.0-1.0)
    calculated from reference track's Crest Factor.
    """
    
    def __init__(self):
        """Initialize the tone processor."""
        self._temp_ir_file: Optional[str] = None
        self.nam_processor = NAMProcessor()
    
    def process_audio(
        self,
        di_track: AudioTrack,
        ref_features: ToneFeatures,
        match_ir: np.ndarray,
        drive_intensity: Optional[float] = None,
        mode: str = "aggressive"
    ) -> AudioTrack:
        """Process DI track through effects chain to match reference tone.
        
        Args:
            di_track: Input DI track to process
            ref_features: ToneFeatures from reference track (used for parameter selection)
            match_ir: Impulse response array for Match EQ (from create_match_filter)
            drive_intensity: Drive intensity (0.0-1.0) calculated from reference.
                            If None, will use default based on mode.
            mode: Processing mode: "conservative", "aggressive", or "lead"
        
        Returns:
            AudioTrack with processed audio
        
        Raises:
            ValueError: If input track is empty or invalid
        """
        if len(di_track.audio) == 0:
            raise ValueError("Cannot process empty audio track")
        
        if len(match_ir) == 0:
            raise ValueError("Match IR cannot be empty")
        
        # Determine drive_intensity based on mode if not provided
        if drive_intensity is None:
            if mode == "conservative":
                drive_intensity = 0.3
            elif mode == "aggressive":
                drive_intensity = 0.7
            elif mode == "lead":
                drive_intensity = 0.9
            else:
                drive_intensity = 0.5
        
        # Start with audio copy
        audio_processed = di_track.audio.copy().astype(np.float32)
        
        # Step 1: Pre-Gain (+12 dB) - Boost signal to drive distortion properly
        pre_gain_db = 12.0
        audio_processed = audio_processed * self._db_to_linear(pre_gain_db)
        
        # Step 2: Compressor (Fixed: 4:1 ratio, fast attack)
        compressor = Compressor(
            ratio=4.0,
            threshold_db=-20.0,
            attack_ms=5.0
        )
        audio_processed = compressor(audio_processed, sample_rate=di_track.sr)
        
        # Step 3: Distortion (Adaptive based on drive_intensity)
        if drive_intensity > 0.6:
            # High-gain: aggressive distortion
            drive_db = 25.0 + (drive_intensity * 15.0)  # 25-40 dB
            distortion = Distortion(drive_db=drive_db)
            audio_processed = distortion(audio_processed, sample_rate=di_track.sr)
            
            # Optional: Second distortion stage for extreme high-gain
            if drive_intensity > 0.85 and mode == "lead":
                distortion2 = Distortion(drive_db=15.0)
                audio_processed = distortion2(audio_processed, sample_rate=di_track.sr)
        elif drive_intensity < 0.4:
            # Clean/light overdrive
            drive_db = 5.0 + (drive_intensity * 5.0)  # 5-10 dB
            distortion = Distortion(drive_db=drive_db)
            audio_processed = distortion(audio_processed, sample_rate=di_track.sr)
        else:
            # Medium drive
            drive_db = 10.0 + ((drive_intensity - 0.4) * 25.0)  # 10-25 dB
            distortion = Distortion(drive_db=drive_db)
            audio_processed = distortion(audio_processed, sample_rate=di_track.sr)
        
        # Step 4: Match EQ (Convolution) - Apply after distortion
        try:
            audio_processed = self._apply_match_eq(
                audio_processed,
                match_ir,
                di_track.sr
            )
        finally:
            # Clean up temporary IR file if it was created
            self._cleanup_temp_ir_file()
        
        # Step 5: Delay (for high-gain lead tones)
        if drive_intensity > 0.7 and DELAY_AVAILABLE:
            delay = Delay(delay_seconds=0.1, mix=0.2)
            audio_processed = delay(audio_processed, sample_rate=di_track.sr)
        
        # Step 6: Reverb (Adaptive based on drive_intensity and mode)
        if mode == "lead":
            room_size = 0.7
            wet_level = 0.4
        elif mode == "aggressive":
            room_size = 0.6
            wet_level = 0.3
        else:  # conservative
            room_size = 0.3
            wet_level = 0.1
        
        # Adjust based on drive_intensity
        if drive_intensity > 0.6:
            wet_level = max(wet_level, 0.3)
            room_size = max(room_size, 0.6)
        
        reverb = Reverb(room_size=room_size, wet_level=wet_level)
        audio_processed = reverb(audio_processed, sample_rate=di_track.sr)
        
        # Step 7: Post-Gain - Normalize to -1.0 dB Peak
        audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
        
        # Ensure output is float32
        audio_processed = audio_processed.astype(np.float32)
        
        return AudioTrack(
            audio=audio_processed,
            sr=di_track.sr,
            name=f"processed_{di_track.name}"
        )
    
    def process_with_full_rig(
        self,
        di_track: AudioTrack,
        nam_path: Optional[str],
        ir_path: str,
        gain_params: Dict[str, float],
        ref_track: Optional[AudioTrack] = None,
        fx_nam_path: Optional[str] = None
    ) -> AudioTrack:
        """Process audio through full Pro chain: Input Gain -> FX NAM -> AMP NAM -> IR -> Final Match EQ -> Normalization.
        
        This is the complete professional processing chain with separate FX (pedal) and AMP models.
        FX NAM provides organic, nonlinear boost that the amplifier needs, replacing hardcoded Pre-EQ.
        
        Args:
            di_track: Input DI track to process
            nam_path: Path to AMP NAM model file (None for mock mode) - backward compatibility
            ir_path: Path to IR cabinet file
            gain_params: Dictionary with gain parameters:
                - 'input_gain_db': float (input gain in dB)
            ref_track: Reference track for final Match EQ (optional, if None, Match EQ is skipped)
            fx_nam_path: Path to FX NAM model file (pedal/booster, optional)
        
        Returns:
            Processed AudioTrack
        
        Raises:
            ValueError: If input track is empty or invalid
            FileNotFoundError: If IR file doesn't exist
        """
        if len(di_track.audio) == 0:
            raise ValueError("Cannot process empty audio track")
        
        if not os.path.exists(ir_path):
            raise FileNotFoundError(f"IR file not found: {ir_path}")
        
        if not CONVOLUTION_AVAILABLE:
            raise ValueError("pedalboard.Convolution is not available. Cannot apply IR.")
        
        # Start with audio copy
        audio_processed = di_track.audio.copy().astype(np.float32)
        
        # Step 1: Input Gain
        input_gain_db = gain_params.get('input_gain_db', 0.0)
        input_gain_linear = self._db_to_linear(input_gain_db)
        audio_processed = audio_processed * input_gain_linear
        
        # Step 2: FX NAM (Pedal/Booster) - Optional, replaces hardcoded Pre-EQ
        # This provides organic, nonlinear boost that the amplifier needs
        if fx_nam_path:
            try:
                audio_processed = self.nam_processor.process_audio(
                    audio_processed,
                    fx_nam_path,
                    sample_rate=di_track.sr
                )
            except Exception as e:
                print(f"Warning: FX NAM processing failed ({e}), continuing without FX")
        
        # Step 3: AMP NAM (Amplifier) - Main amp modeling
        amp_nam_path = nam_path  # Use nam_path for backward compatibility
        if amp_nam_path:
            try:
                audio_processed = self.nam_processor.process_audio(
                    audio_processed,
                    amp_nam_path,
                    sample_rate=di_track.sr
                )
            except Exception as e:
                print(f"Warning: AMP NAM processing failed ({e}), using mock fallback")
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
        
        # Step 4: IR - Cabinet simulation via Convolution
        convolution = Convolution(impulse_response_filename=ir_path)
        audio_processed = convolution(audio_processed, sample_rate=di_track.sr)
        
        # Step 5: Final Match EQ - Polish the last 10% of tone differences
        if ref_track is not None:
            try:
                from src.core.analysis import analyze_track
                from src.core.matching import create_match_filter
                from scipy import signal as scipy_signal
                
                # Create temporary AudioTrack for processed audio to analyze
                processed_track = AudioTrack(
                    audio=audio_processed,
                    sr=di_track.sr,
                    name="processed_temp"
                )
                
                # Analyze both processed and reference
                processed_features = analyze_track(processed_track)
                ref_features = analyze_track(ref_track)
                
                # Create Match EQ filter
                match_filter = create_match_filter(
                    source_features=processed_features,
                    target_features=ref_features,
                    num_taps=4096
                )
                
                # Apply Match EQ filter
                audio_processed = scipy_signal.lfilter(match_filter, 1.0, audio_processed)
            except Exception as e:
                print(f"Warning: Final Match EQ failed ({e}), continuing without it")
        
        # Step 6: Normalization - Normalize to -1.0 dB peak
        audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
        
        # Ensure output is float32
        audio_processed = audio_processed.astype(np.float32)
        
        return AudioTrack(
            audio=audio_processed,
            sr=di_track.sr,
            name=f"full_rig_{di_track.name}"
        )
    
    def process_final_tune(
        self,
        di_track: AudioTrack,
        gain_params: Dict[str, float],
        post_fx_params: Dict[str, float],
        ref_track: Optional[AudioTrack] = None,
        ddsp_weights_path: Optional[str] = None,
        waveshaper: Optional['DifferentiableWaveshaperProcessor'] = None
    ) -> AudioTrack:
        """Process audio through fixed NAM chain with Post-FX optimization.
        
        Fixed chain: Pre-EQ (Peak Filter) -> Input Gain -> DS1 (FX NAM) -> 5150 BlockLetter (AMP NAM) -> BlendOfAll (IR) 
        -> [DifferentiableWaveshaper] -> Delay -> Reverb -> Final Match EQ (with gain adjustment) -> Normalization.
        
        This method uses hardcoded paths to NAM/IR files for speed and focuses on optimizing
        Post-FX parameters including Pre-EQ (Reverb, Delay, Final EQ Gain, Pre-EQ).
        
        Args:
            di_track: Input DI track to process
            gain_params: Dictionary with gain parameters:
                - 'input_gain_db': float (input gain in dB)
            post_fx_params: Dictionary with Post-FX parameters (includes Pre-EQ for convenience):
                - 'pre_eq_gain_db': float [-12.0 ... +12.0] (Pre-EQ peak gain in dB, default: 0.0)
                - 'pre_eq_freq_hz': float [400.0 ... 3000.0] (Pre-EQ peak frequency in Hz, default: 800.0)
                - 'reverb_wet': float [0.0 ... 0.7] (reverb wet level)
                - 'reverb_room_size': float [0.0 ... 1.0] (reverb room size)
                - 'delay_time_ms': float [50 ... 500] (delay time in milliseconds)
                - 'delay_mix': float [0.0 ... 0.5] (delay feedback/mix)
                - 'final_eq_gain_db': float [-3.0 ... +3.0] (final EQ gain adjustment in dB)
            ref_track: Reference track for final Match EQ (optional, if None, Match EQ is skipped)
            ddsp_weights_path: Path to DDSP weights file (optional, replaces Match EQ if provided)
            waveshaper: DifferentiableWaveshaperProcessor instance (optional, for harmonic correction)
        
        Returns:
            Processed AudioTrack
        
        Raises:
            ValueError: If input track is empty or invalid
            FileNotFoundError: If hardcoded NAM/IR files don't exist
        """
        if len(di_track.audio) == 0:
            raise ValueError("Cannot process empty audio track")
        
        if not CONVOLUTION_AVAILABLE:
            raise ValueError("pedalboard.Convolution is not available. Cannot apply IR.")
        
        # Hardcoded paths to NAM/IR files
        FX_NAM_PATH = "assets/nam_models/Keith B DS1_g6_t5.nam"
        AMP_NAM_PATH = "assets/nam_models/Helga B 5150 BlockLetter - NoBoost.nam"
        IR_PATH = "assets/impulse_responses/BlendOfAll.wav"
        
        # Verify files exist
        if not os.path.exists(FX_NAM_PATH):
            raise FileNotFoundError(f"FX NAM file not found: {FX_NAM_PATH}")
        if not os.path.exists(AMP_NAM_PATH):
            raise FileNotFoundError(f"AMP NAM file not found: {AMP_NAM_PATH}")
        if not os.path.exists(IR_PATH):
            raise FileNotFoundError(f"IR file not found: {IR_PATH}")
        
        # Extract Pre-EQ parameters (applied at the beginning)
        pre_eq_gain_db = post_fx_params.get('pre_eq_gain_db', 0.0)
        pre_eq_freq_hz = post_fx_params.get('pre_eq_freq_hz', 800.0)
        
        # Extract Post-FX parameters
        reverb_wet = post_fx_params.get('reverb_wet', 0.2)
        reverb_room_size = post_fx_params.get('reverb_room_size', 0.5)
        delay_time_ms = post_fx_params.get('delay_time_ms', 100.0)
        delay_mix = post_fx_params.get('delay_mix', 0.2)
        final_eq_gain_db = post_fx_params.get('final_eq_gain_db', 0.0)
        
        # Start with audio copy
        audio_processed = di_track.audio.copy().astype(np.float32)
        
        # Step 0: Pre-EQ (Peak Filter) - Applied at the very beginning for tone shaping
        if abs(pre_eq_gain_db) > 0.01:  # Only apply if gain is significant
            # Use Q=2.0 for reasonable bandwidth (similar to typical guitar EQ)
            audio_processed = self._apply_peak_filter(
                audio_processed,
                center_hz=pre_eq_freq_hz,
                gain_db=pre_eq_gain_db,
                q=2.0,
                sr=di_track.sr
            )
        
        # Step 1: Input Gain
        input_gain_db = gain_params.get('input_gain_db', 0.0)
        input_gain_linear = self._db_to_linear(input_gain_db)
        audio_processed = audio_processed * input_gain_linear
        
        # Step 2: FX NAM (DS1) - Hardcoded
        try:
            audio_processed = self.nam_processor.process_audio(
                audio_processed,
                FX_NAM_PATH,
                sample_rate=di_track.sr
            )
        except Exception as e:
            print(f"Warning: FX NAM processing failed ({e}), continuing without FX")
        
        # Step 3: AMP NAM (5150 BlockLetter) - Hardcoded
        try:
            audio_processed = self.nam_processor.process_audio(
                audio_processed,
                AMP_NAM_PATH,
                sample_rate=di_track.sr
            )
        except Exception as e:
            print(f"Warning: AMP NAM processing failed ({e}), using mock fallback")
            # Fallback to mock
            self.nam_processor.set_mock_mode(True)
            audio_processed = self.nam_processor.process_audio(
                audio_processed,
                "mock",
                sample_rate=di_track.sr
            )
            self.nam_processor.set_mock_mode(False)
        
        # Step 4: IR (BlendOfAll) - Hardcoded
        convolution = Convolution(impulse_response_filename=IR_PATH)
        audio_processed = convolution(audio_processed, sample_rate=di_track.sr)
        
        # Step 4.5: Differentiable Waveshaper (if provided, for harmonic correction)
        if waveshaper is not None:
            try:
                audio_processed = waveshaper.process_audio(audio_processed, sample_rate=di_track.sr)
            except Exception as e:
                print(f"Warning: Waveshaper processing failed ({e}), continuing without it")
        
        # Step 4.6: DDSP Block (if weights provided, replaces Match EQ)
        if ddsp_weights_path is not None:
            try:
                from src.core.ddsp_processor import DDSPProcessor
                ddsp = DDSPProcessor()
                ddsp.load_weights(ddsp_weights_path)
                audio_processed = ddsp.process_audio(audio_processed, sample_rate=di_track.sr)
            except Exception as e:
                print(f"Warning: DDSP processing failed ({e}), falling back to Match EQ")
                ddsp_weights_path = None  # Fall back to Match EQ
        
        # Step 5: Delay (Post-FX parameter)
        if DELAY_AVAILABLE:
            delay_time_seconds = delay_time_ms / 1000.0  # Convert ms to seconds
            delay = Delay(delay_seconds=delay_time_seconds, mix=delay_mix)
            audio_processed = delay(audio_processed, sample_rate=di_track.sr)
        
        # Step 6: Reverb (Post-FX parameter)
        reverb = Reverb(room_size=reverb_room_size, wet_level=reverb_wet)
        audio_processed = reverb(audio_processed, sample_rate=di_track.sr)
        
        # Step 7: Final Match EQ with gain adjustment (Post-FX parameter)
        # Skip if DDSP was used (already applied after IR)
        if ref_track is not None and ddsp_weights_path is None:
            try:
                from src.core.analysis import analyze_track
                from src.core.matching import create_match_filter
                from scipy import signal as scipy_signal
                
                # Create temporary AudioTrack for processed audio to analyze
                processed_track = AudioTrack(
                    audio=audio_processed,
                    sr=di_track.sr,
                    name="processed_temp"
                )
                
                # Analyze both processed and reference
                processed_features = analyze_track(processed_track)
                ref_features = analyze_track(ref_track)
                
                # Create Match EQ filter
                match_filter = create_match_filter(
                    source_features=processed_features,
                    target_features=ref_features,
                    num_taps=4096
                )
                
                # Apply Match EQ filter
                audio_processed = scipy_signal.lfilter(match_filter, 1.0, audio_processed)
                
                # Apply final EQ gain adjustment (global gain shift)
                final_eq_gain_linear = self._db_to_linear(final_eq_gain_db)
                audio_processed = audio_processed * final_eq_gain_linear
            except Exception as e:
                print(f"Warning: Final Match EQ failed ({e}), continuing without it")
        
        # Step 8: Normalization - Normalize to -1.0 dB peak
        audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
        
        # Ensure output is float32
        audio_processed = audio_processed.astype(np.float32)
        
        return AudioTrack(
            audio=audio_processed,
            sr=di_track.sr,
            name=f"final_tuned_{di_track.name}"
        )
    
    def process_with_ddsp(
        self,
        di_track: AudioTrack,
        gain_params: Dict[str, float],
        post_fx_params: Dict[str, float],
        ddsp_weights_path: str,
        ref_track: Optional[AudioTrack] = None
    ) -> AudioTrack:
        """Process audio through fixed NAM chain with DDSP block instead of Match EQ.
        
        Fixed chain: Input Gain -> DS1 (FX NAM) -> 5150 BlockLetter (AMP NAM) -> BlendOfAll (IR) 
        -> DDSP Block -> Delay -> Reverb -> Normalization.
        
        This method uses DDSP (Differentiable DSP) neural network block for fine-tuning
        instead of traditional Match EQ. The DDSP block must be pre-trained.
        
        Args:
            di_track: Input DI track to process
            gain_params: Dictionary with gain parameters:
                - 'input_gain_db': float (input gain in dB)
            post_fx_params: Dictionary with Post-FX parameters:
                - 'reverb_wet': float [0.0 ... 0.7] (reverb wet level)
                - 'reverb_room_size': float [0.0 ... 1.0] (reverb room size)
                - 'delay_time_ms': float [50 ... 500] (delay time in milliseconds)
                - 'delay_mix': float [0.0 ... 0.5] (delay feedback/mix)
            ddsp_weights_path: Path to trained DDSP weights file (.pth)
            ref_track: Reference track (optional, not used but kept for API consistency)
        
        Returns:
            Processed AudioTrack
        
        Raises:
            ValueError: If input track is empty or invalid
            FileNotFoundError: If hardcoded NAM/IR files or DDSP weights don't exist
        """
        # Use process_final_tune with ddsp_weights_path parameter
        return self.process_final_tune(
            di_track=di_track,
            gain_params=gain_params,
            post_fx_params=post_fx_params,
            ref_track=ref_track,
            ddsp_weights_path=ddsp_weights_path
        )
    
    def process_with_custom_rig_and_post_fx(
        self,
        di_track: AudioTrack,
        fx_nam_path: Optional[str],
        amp_nam_path: Optional[str],
        ir_path: str,
        gain_params: Dict[str, float],
        post_fx_params: Dict[str, float],
        ref_track: Optional[AudioTrack] = None
    ) -> AudioTrack:
        """Process audio through custom NAM/IR chain with Post-FX optimization.
        
        Custom chain: Pre-EQ (Peak Filter) -> Input Gain -> FX NAM -> AMP NAM -> IR -> Delay -> Reverb -> Final Match EQ (with gain adjustment) -> Normalization.
        
        This method accepts custom paths to NAM/IR files and applies Post-FX parameters.
        Pre-EQ is applied at the very beginning to allow neural network control over tone shaping.
        
        Args:
            di_track: Input DI track to process
            fx_nam_path: Path to FX NAM model file (pedal/booster, optional)
            amp_nam_path: Path to AMP NAM model file (amplifier, optional)
            ir_path: Path to IR cabinet file
            gain_params: Dictionary with gain parameters:
                - 'input_gain_db': float (input gain in dB)
            post_fx_params: Dictionary with Post-FX parameters (includes Pre-EQ for convenience):
                - 'pre_eq_gain_db': float [-12.0 ... +12.0] (Pre-EQ peak gain in dB, default: 0.0)
                - 'pre_eq_freq_hz': float [400.0 ... 3000.0] (Pre-EQ peak frequency in Hz, default: 800.0)
                - 'reverb_wet': float [0.0 ... 0.7] (reverb wet level)
                - 'reverb_room_size': float [0.0 ... 1.0] (reverb room size)
                - 'delay_time_ms': float [50 ... 500] (delay time in milliseconds)
                - 'delay_mix': float [0.0 ... 0.5] (delay feedback/mix)
                - 'final_eq_gain_db': float [-3.0 ... +3.0] (final EQ gain adjustment in dB)
            ref_track: Reference track for final Match EQ (optional, if None, Match EQ is skipped)
        
        Returns:
            Processed AudioTrack
        
        Raises:
            ValueError: If input track is empty or invalid
            FileNotFoundError: If NAM/IR files don't exist
        """
        if len(di_track.audio) == 0:
            raise ValueError("Cannot process empty audio track")
        
        if not CONVOLUTION_AVAILABLE:
            raise ValueError("pedalboard.Convolution is not available. Cannot apply IR.")
        
        if not os.path.exists(ir_path):
            raise FileNotFoundError(f"IR file not found: {ir_path}")
        
        # Extract Pre-EQ parameters (applied at the beginning)
        pre_eq_gain_db = post_fx_params.get('pre_eq_gain_db', 0.0)
        pre_eq_freq_hz = post_fx_params.get('pre_eq_freq_hz', 800.0)
        
        # Extract Post-FX parameters
        reverb_wet = post_fx_params.get('reverb_wet', 0.2)
        reverb_room_size = post_fx_params.get('reverb_room_size', 0.5)
        delay_time_ms = post_fx_params.get('delay_time_ms', 100.0)
        delay_mix = post_fx_params.get('delay_mix', 0.2)
        final_eq_gain_db = post_fx_params.get('final_eq_gain_db', 0.0)
        
        # Start with audio copy
        audio_processed = di_track.audio.copy().astype(np.float32)
        
        # Step 0: Pre-EQ (Peak Filter) - Applied at the very beginning for tone shaping
        if abs(pre_eq_gain_db) > 0.01:  # Only apply if gain is significant
            # Use Q=2.0 for reasonable bandwidth (similar to typical guitar EQ)
            audio_processed = self._apply_peak_filter(
                audio_processed,
                center_hz=pre_eq_freq_hz,
                gain_db=pre_eq_gain_db,
                q=2.0,
                sr=di_track.sr
            )
        
        # Step 1: Input Gain
        input_gain_db = gain_params.get('input_gain_db', 0.0)
        input_gain_linear = self._db_to_linear(input_gain_db)
        audio_processed = audio_processed * input_gain_linear
        
        # Step 2: FX NAM (Pedal/Booster) - Optional
        if fx_nam_path:
            try:
                audio_processed = self.nam_processor.process_audio(
                    audio_processed,
                    fx_nam_path,
                    sample_rate=di_track.sr
                )
            except Exception as e:
                print(f"Warning: FX NAM processing failed ({e}), continuing without FX")
        
        # Step 3: AMP NAM (Amplifier) - Optional
        if amp_nam_path:
            try:
                audio_processed = self.nam_processor.process_audio(
                    audio_processed,
                    amp_nam_path,
                    sample_rate=di_track.sr
                )
            except Exception as e:
                print(f"Warning: AMP NAM processing failed ({e}), using mock fallback")
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
        
        # Step 4: IR - Cabinet simulation via Convolution
        convolution = Convolution(impulse_response_filename=ir_path)
        audio_processed = convolution(audio_processed, sample_rate=di_track.sr)
        
        # Step 5: Delay (Post-FX parameter)
        if DELAY_AVAILABLE:
            delay_time_seconds = delay_time_ms / 1000.0  # Convert ms to seconds
            delay = Delay(delay_seconds=delay_time_seconds, mix=delay_mix)
            audio_processed = delay(audio_processed, sample_rate=di_track.sr)
        
        # Step 6: Reverb (Post-FX parameter)
        reverb = Reverb(room_size=reverb_room_size, wet_level=reverb_wet)
        audio_processed = reverb(audio_processed, sample_rate=di_track.sr)
        
        # Step 7: Final Match EQ with gain adjustment (Post-FX parameter) - AGGRESSIVE MODE
        if ref_track is not None:
            try:
                from src.core.analysis import analyze_track
                from src.core.matching import create_match_filter
                from scipy import signal as scipy_signal
                
                # AGGRESSIVE ITERATIVE MATCH EQ - Apply multiple times for perfect matching
                max_iterations = 3  # Apply up to 3 times for perfect match
                for iteration in range(max_iterations):
                    # Create temporary AudioTrack for processed audio to analyze
                    processed_track = AudioTrack(
                        audio=audio_processed,
                        sr=di_track.sr,
                        name="processed_temp"
                    )
                    
                    # Analyze both processed and reference
                    processed_features = analyze_track(processed_track)
                    ref_features = analyze_track(ref_track)
                    
                    # Calculate current difference to check if we need more iterations
                    epsilon = 1e-10
                    processed_spectrum_db = 20 * np.log10(processed_features.spectrum + epsilon)
                    ref_spectrum_db = 20 * np.log10(ref_features.spectrum + epsilon)
                    
                    # Normalize for comparison
                    processed_peak_db = np.max(processed_spectrum_db)
                    ref_peak_db = np.max(ref_spectrum_db)
                    processed_normalized = processed_spectrum_db - processed_peak_db
                    ref_normalized = ref_spectrum_db - ref_peak_db
                    
                    # Interpolate to common frequency axis if needed
                    if len(processed_features.frequencies) == len(ref_features.frequencies):
                        diff = np.abs(processed_normalized - ref_normalized)
                    else:
                        from scipy.interpolate import interp1d
                        freq_axis = processed_features.frequencies
                        interp_func = interp1d(
                            ref_features.frequencies,
                            ref_normalized,
                            kind='linear',
                            bounds_error=False,
                            fill_value='extrapolate'
                        )
                        ref_interp = interp_func(freq_axis)
                        diff = np.abs(processed_normalized - ref_interp)
                    
                    # Check if difference is small enough (mean < 0.5 dB)
                    mean_diff = np.mean(diff)
                    if mean_diff < 0.5 and iteration > 0:
                        break  # Good enough, stop iterating
                    
                    # Create Match EQ filter with maximum taps for precision
                    match_filter = create_match_filter(
                        source_features=processed_features,
                        target_features=ref_features,
                        num_taps=8192  # Increased from 4096 for better precision
                    )
                    
                    # Apply Match EQ filter
                    audio_processed = scipy_signal.lfilter(match_filter, 1.0, audio_processed)
                
                # MAGNITUDE NORMALIZATION - Match RMS energy to reference
                try:
                    import librosa
                    ref_rms = np.sqrt(np.mean(ref_track.audio ** 2))
                    current_rms = np.sqrt(np.mean(audio_processed ** 2))
                    
                    if ref_rms > 1e-10 and current_rms > 1e-10:
                        # Match RMS energy
                        rms_gain = ref_rms / current_rms
                        audio_processed = audio_processed * rms_gain
                        
                        # Also consider peak levels
                        ref_peak = np.max(np.abs(ref_track.audio))
                        current_peak = np.max(np.abs(audio_processed))
                        
                        if ref_peak > 1e-10 and current_peak > 1e-10:
                            # Weighted: 80% RMS, 20% Peak
                            peak_gain = ref_peak / current_peak
                            combined_gain = 0.8 * rms_gain + 0.2 * peak_gain
                            audio_processed = audio_processed * (combined_gain / rms_gain)
                except Exception as e:
                    pass  # Silent fail, continue with EQ-matched result
                
                # Apply final EQ gain adjustment (global gain shift)
                final_eq_gain_linear = self._db_to_linear(final_eq_gain_db)
                audio_processed = audio_processed * final_eq_gain_linear
            except Exception as e:
                print(f"Warning: Final Match EQ failed ({e}), continuing without it")
        
        # Step 8: Normalization - Match to reference level if available, otherwise -1.0 dB peak
        if ref_track is not None:
            try:
                # Match to reference peak level for perfect magnitude matching
                ref_peak = np.max(np.abs(ref_track.audio))
                current_peak = np.max(np.abs(audio_processed))
                if ref_peak > 1e-10 and current_peak > 1e-10:
                    peak_gain = ref_peak / current_peak
                    audio_processed = audio_processed * peak_gain
                else:
                    # Fallback to -1.0 dB
                    audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
            except Exception:
                # Fallback to -1.0 dB
                audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
        else:
            audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
        
        # Ensure output is float32
        audio_processed = audio_processed.astype(np.float32)
        
        return AudioTrack(
            audio=audio_processed,
            sr=di_track.sr,
            name=f"custom_rig_{di_track.name}"
        )
    
    def _db_to_linear(self, db: float) -> float:
        """Convert dB to linear gain.
        
        Args:
            db: Gain in decibels
        
        Returns:
            Linear gain multiplier
        """
        return 10 ** (db / 20.0)
    
    def _normalize_to_db(self, audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        """Normalize audio to target peak level in dB.
        
        Args:
            audio: Input audio array
            target_db: Target peak level in dB (default: -1.0 dB)
        
        Returns:
            Normalized audio array
        """
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio
        
        target_linear = 10 ** (target_db / 20.0)
        normalized = audio * (target_linear / max_val)
        
        return normalized
    
    def _calculate_compressor_params(self, ref_features: ToneFeatures) -> dict:
        """Calculate compressor parameters based on reference dynamic range.
        
        Args:
            ref_features: ToneFeatures from reference track
        
        Returns:
            Dictionary with 'ratio' and 'threshold_db' keys
        """
        # Narrow dynamic range (compressed sound) -> stronger compression
        # Wide dynamic range -> lighter compression
        if ref_features.dynamic_range < 0.1:
            # Narrow range: strong compression
            return {
                'ratio': 7.0,
                'threshold_db': -20.0
            }
        else:
            # Wide range: light compression
            return {
                'ratio': 2.5,
                'threshold_db': -30.0
            }
    
    def _calculate_distortion_params(self, ref_features: ToneFeatures) -> dict:
        """Calculate distortion parameters based on reference spectral characteristics.
        
        Args:
            ref_features: ToneFeatures from reference track
        
        Returns:
            Dictionary with 'drive_db' key
        """
        # High RMS and high spectral centroid -> loud and dense -> more distortion
        # Clean signal -> minimal or no distortion
        rms_threshold = 0.3
        centroid_threshold = 3000.0
        
        if ref_features.rms_energy > rms_threshold and ref_features.spectral_centroid > centroid_threshold:
            # Loud and dense: strong distortion
            return {'drive_db': 18.0}
        else:
            # Clean signal: minimal distortion
            return {'drive_db': 3.0}
    
    def _apply_match_eq(
        self,
        audio: np.ndarray,
        match_ir: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Apply Match EQ using Convolution.
        
        Tries pedalboard.Convolution first, falls back to scipy.signal.fftconvolve.
        
        Args:
            audio: Input audio array
            match_ir: Impulse response for convolution
            sample_rate: Sample rate in Hz
        
        Returns:
            Processed audio array
        """
        # Try pedalboard.Convolution first
        if CONVOLUTION_AVAILABLE:
            try:
                # Save IR to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    # Ensure IR is mono and properly shaped
                    ir_audio = match_ir.copy()
                    if len(ir_audio.shape) == 1:
                        # Already mono
                        pass
                    elif len(ir_audio.shape) == 2:
                        # Convert stereo to mono (average channels)
                        ir_audio = np.mean(ir_audio, axis=0)
                    else:
                        raise ValueError(f"Unexpected IR shape: {ir_audio.shape}")
                    
                    # Normalize IR to prevent clipping
                    max_ir = np.max(np.abs(ir_audio))
                    if max_ir > 0:
                        ir_audio = ir_audio / max_ir * 0.9
                    
                    # Save IR to temporary file
                    sf.write(tmp_path, ir_audio.astype(np.float32), sample_rate)
                    
                    # Create Convolution plugin and apply
                    # Note: pedalboard.Convolution uses 'impulse_response_filename', not 'impulse_response_path'
                    convolution = Convolution(impulse_response_filename=tmp_path)
                    processed = convolution(audio, sample_rate=sample_rate)
                    
                    return processed
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
            except Exception as e:
                # Fallback to scipy if pedalboard.Convolution fails
                print(f"Warning: pedalboard.Convolution failed ({e}), using scipy fallback")
        
        # Fallback: Use scipy.signal.fftconvolve
        # Ensure IR is properly shaped
        ir_audio = match_ir.copy()
        if len(ir_audio.shape) > 1:
            ir_audio = np.mean(ir_audio, axis=0)
        
        # Normalize IR
        max_ir = np.max(np.abs(ir_audio))
        if max_ir > 0:
            ir_audio = ir_audio / max_ir
        
        # Apply convolution using fftconvolve (faster for long signals)
        processed = scipy_signal.fftconvolve(audio, ir_audio, mode='same')
        
        return processed
    
    def _create_convolution_plugin(
        self,
        match_ir: np.ndarray,
        sample_rate: int
    ) -> Optional[Convolution]:
        """Create a Convolution plugin with temporary IR file.
        
        Args:
            match_ir: Impulse response array
            sample_rate: Sample rate in Hz
        
        Returns:
            Convolution plugin instance, or None if creation fails
        """
        if not CONVOLUTION_AVAILABLE:
            return None
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Ensure IR is mono and properly shaped
            ir_audio = match_ir.copy()
            if len(ir_audio.shape) == 1:
                pass  # Already mono
            elif len(ir_audio.shape) == 2:
                ir_audio = np.mean(ir_audio, axis=0)
            else:
                raise ValueError(f"Unexpected IR shape: {ir_audio.shape}")
            
            # Normalize IR to prevent clipping
            max_ir = np.max(np.abs(ir_audio))
            if max_ir > 0:
                ir_audio = ir_audio / max_ir * 0.9
            
            # Save IR to temporary file
            sf.write(tmp_path, ir_audio.astype(np.float32), sample_rate)
            
            # Store path for cleanup
            self._temp_ir_file = tmp_path
            
            # Create and return Convolution plugin
            # Note: pedalboard.Convolution uses 'impulse_response_filename', not 'impulse_response_path'
            return Convolution(impulse_response_filename=tmp_path)
            
        except Exception as e:
            # Clean up on failure
            if self._temp_ir_file and os.path.exists(self._temp_ir_file):
                os.unlink(self._temp_ir_file)
                self._temp_ir_file = None
            print(f"Warning: Failed to create Convolution plugin ({e}), will use scipy fallback")
            return None
    
    def _cleanup_temp_ir_file(self) -> None:
        """Clean up temporary IR file if it exists."""
        if self._temp_ir_file and os.path.exists(self._temp_ir_file):
            try:
                os.unlink(self._temp_ir_file)
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self._temp_ir_file = None
    
    def optimize_tone(
        self,
        di_track: AudioTrack,
        ref_features: ToneFeatures,
        match_ir: np.ndarray,
        ref_audio_track: AudioTrack,
        test_duration_sec: float = 10.0
    ) -> AudioTrack:
        """Optimize tone by testing multiple profiles and selecting the best match.
        
        Tests 3 candidate profiles (Clean, Rock, Metal) and selects the one
        with minimum distance to reference using MFCC + Spectral Flatness metric.
        
        Args:
            di_track: Input DI track to process
            ref_features: ToneFeatures from reference track
            match_ir: Impulse response array for Match EQ
            ref_audio_track: Reference audio track for comparison
            test_duration_sec: Duration of audio to use for testing (default: 10.0 sec)
                               Use shorter duration for faster optimization
        
        Returns:
            AudioTrack with optimized processing applied to full track
        """
        if len(di_track.audio) == 0:
            raise ValueError("Cannot optimize empty audio track")
        
        # Create test segments (first N seconds for speed)
        test_samples = int(test_duration_sec * di_track.sr)
        di_test_audio = di_track.audio[:min(test_samples, len(di_track.audio))]
        ref_test_audio = ref_audio_track.audio[:min(test_samples, len(ref_audio_track.audio))]
        
        di_test_track = AudioTrack(audio=di_test_audio, sr=di_track.sr, name="di_test")
        ref_test_track = AudioTrack(audio=ref_test_audio, sr=ref_audio_track.sr, name="ref_test")
        
        # Define 3 candidate profiles
        profiles = [
            ("Clean", self._process_clean_profile),
            ("Rock", self._process_rock_profile),
            ("Metal", self._process_metal_profile)
        ]
        
        best_profile = None
        best_distance = float('inf')
        best_result = None
        
        # Test each profile
        for profile_name, profile_func in profiles:
            try:
                # Process test segment with this profile
                processed_test = profile_func(di_test_track, ref_features, match_ir)
                
                # Calculate distance to reference
                distance = calculate_distance(
                    processed_test.audio,
                    ref_test_track.audio,
                    sr=di_track.sr
                )
                
                # Track best result
                if distance < best_distance:
                    best_distance = distance
                    best_profile = profile_name
                    best_result = processed_test
                    
            except Exception as e:
                print(f"Warning: Profile '{profile_name}' failed: {e}")
                continue
        
        if best_result is None:
            raise ValueError("All profiles failed during optimization")
        
        # Apply best profile to full track
        if best_profile == "Clean":
            return self._process_clean_profile(di_track, ref_features, match_ir)
        elif best_profile == "Rock":
            return self._process_rock_profile(di_track, ref_features, match_ir)
        else:  # Metal
            return self._process_metal_profile(di_track, ref_features, match_ir)
    
    def _process_clean_profile(
        self,
        di_track: AudioTrack,
        ref_features: ToneFeatures,
        match_ir: np.ndarray
    ) -> AudioTrack:
        """Process with Clean profile: Low compression, no distortion, small reverb."""
        audio_processed = di_track.audio.copy().astype(np.float32)
        
        # Pre-Gain
        audio_processed = audio_processed * self._db_to_linear(8.0)  # Lower gain
        
        # Compressor (low)
        compressor = Compressor(ratio=2.0, threshold_db=-30.0, attack_ms=10.0)
        audio_processed = compressor(audio_processed, sample_rate=di_track.sr)
        
        # Distortion (0 dB - no distortion)
        # Skip distortion for clean profile
        
        # Match EQ
        try:
            audio_processed = self._apply_match_eq(audio_processed, match_ir, di_track.sr)
        finally:
            self._cleanup_temp_ir_file()
        
        # Reverb (small)
        reverb = Reverb(room_size=0.2, wet_level=0.08)
        audio_processed = reverb(audio_processed, sample_rate=di_track.sr)
        
        # Post-Gain
        audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
        audio_processed = audio_processed.astype(np.float32)
        
        return AudioTrack(audio=audio_processed, sr=di_track.sr, name=f"clean_{di_track.name}")
    
    def _process_rock_profile(
        self,
        di_track: AudioTrack,
        ref_features: ToneFeatures,
        match_ir: np.ndarray
    ) -> AudioTrack:
        """Process with Rock profile: Medium compression, moderate distortion, medium reverb."""
        audio_processed = di_track.audio.copy().astype(np.float32)
        
        # Pre-Gain
        audio_processed = audio_processed * self._db_to_linear(12.0)
        
        # Compressor (mid)
        compressor = Compressor(ratio=4.0, threshold_db=-20.0, attack_ms=5.0)
        audio_processed = compressor(audio_processed, sample_rate=di_track.sr)
        
        # Distortion (15 dB)
        distortion = Distortion(drive_db=15.0)
        audio_processed = distortion(audio_processed, sample_rate=di_track.sr)
        
        # Match EQ
        try:
            audio_processed = self._apply_match_eq(audio_processed, match_ir, di_track.sr)
        finally:
            self._cleanup_temp_ir_file()
        
        # Reverb (medium)
        reverb = Reverb(room_size=0.5, wet_level=0.2)
        audio_processed = reverb(audio_processed, sample_rate=di_track.sr)
        
        # Post-Gain
        audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
        audio_processed = audio_processed.astype(np.float32)
        
        return AudioTrack(audio=audio_processed, sr=di_track.sr, name=f"rock_{di_track.name}")
    
    def _process_metal_profile(
        self,
        di_track: AudioTrack,
        ref_features: ToneFeatures,
        match_ir: np.ndarray
    ) -> AudioTrack:
        """Process with Metal profile: High compression, heavy distortion, large reverb."""
        audio_processed = di_track.audio.copy().astype(np.float32)
        
        # Pre-Gain
        audio_processed = audio_processed * self._db_to_linear(15.0)  # Higher gain
        
        # Noise Gate (if available)
        if NOISEGATE_AVAILABLE:
            gate = NoiseGate(threshold_db=-40.0)
            audio_processed = gate(audio_processed, sample_rate=di_track.sr)
        
        # Compressor (high)
        compressor = Compressor(ratio=8.0, threshold_db=-15.0, attack_ms=3.0)
        audio_processed = compressor(audio_processed, sample_rate=di_track.sr)
        
        # Distortion (30 dB - heavy)
        distortion = Distortion(drive_db=30.0)
        audio_processed = distortion(audio_processed, sample_rate=di_track.sr)
        
        # Match EQ
        try:
            audio_processed = self._apply_match_eq(audio_processed, match_ir, di_track.sr)
        finally:
            self._cleanup_temp_ir_file()
        
        # Delay (if available)
        if DELAY_AVAILABLE:
            delay = Delay(delay_seconds=0.1, mix=0.15)
            audio_processed = delay(audio_processed, sample_rate=di_track.sr)
        
        # Reverb (large)
        reverb = Reverb(room_size=0.7, wet_level=0.35)
        audio_processed = reverb(audio_processed, sample_rate=di_track.sr)
        
        # Post-Gain
        audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
        audio_processed = audio_processed.astype(np.float32)
        
        return AudioTrack(audio=audio_processed, sr=di_track.sr, name=f"metal_{di_track.name}")
    
    def _apply_high_pass_filter(self, audio: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
        """Apply high-pass filter using scipy.signal.
        
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
        
        b, a = scipy_signal.butter(order, normalized_cutoff, btype='high')
        filtered = scipy_signal.lfilter(b, a, audio)
        
        return filtered
    
    def _apply_low_pass_filter(self, audio: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
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
    
    def _apply_peak_filter(self, audio: np.ndarray, center_hz: float, gain_db: float, q: float, sr: int) -> np.ndarray:
        """Apply peak/parametric EQ filter using scipy.signal.
        
        Args:
            audio: Input audio array
            center_hz: Center frequency in Hz
            gain_db: Gain in dB (positive = boost, negative = cut)
            q: Quality factor (bandwidth), higher = narrower
            sr: Sample rate
        
        Returns:
            Filtered audio array
        """
        # Convert gain from dB to linear
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Design peak filter using iirpeak
        # iirpeak creates a resonant filter at center frequency
        w0 = 2 * np.pi * center_hz / sr  # Normalized frequency
        
        # Create peak filter
        # Using biquad filter design for peak EQ
        # This is a simplified version - for more accuracy, use sosfilt
        b, a = scipy_signal.iirpeak(w0, Q=q)
        
        # Apply gain by scaling the filter coefficients
        # For a peak filter, we need to adjust the gain
        # Simple approach: apply filter and then scale the result
        filtered = scipy_signal.lfilter(b, a, audio)
        
        # Mix original and filtered signal based on gain
        if gain_db > 0:
            # Boost: mix more of the filtered signal
            mix = (gain_linear - 1.0) / gain_linear
            result = audio + (filtered - audio) * mix
        else:
            # Cut: reduce the filtered component
            mix = 1.0 - gain_linear
            result = audio - (filtered - audio) * mix
        
        return result
    
    def _process_smooth_lead_profile(
        self,
        di_track: AudioTrack,
        ref_features: ToneFeatures,
        match_ir: np.ndarray
    ) -> AudioTrack:
        """Process with Smooth Lead profile: Soft, singing, sustained tone.
        
        Chain: Compression -> Pre-Shaping -> Soft Clipping -> Cab Simulation -> Match EQ -> Reverb
        
        Args:
            di_track: Input DI track
            ref_features: Reference features (for Match EQ)
            match_ir: Match EQ impulse response
        
        Returns:
            Processed AudioTrack
        """
        audio_processed = di_track.audio.copy().astype(np.float32)
        
        # Step 1: Compression (Sustain) - Strong compression for long sustain
        compressor = Compressor(
            ratio=8.0,
            threshold_db=-20.0,
            attack_ms=5.0,
            release_ms=200.0
        )
        audio_processed = compressor(audio_processed, sample_rate=di_track.sr)
        
        # Step 2: Pre-Shaping ("Screamer" Curve)
        # High Pass Filter: 700 Hz (cut mud)
        audio_processed = self._apply_high_pass_filter(audio_processed, 700.0, di_track.sr)
        
        # Peak Filter: 800-1000 Hz, Gain +8dB (juiciness and punch)
        # Apply peak at 900 Hz (center of 800-1000 range)
        audio_processed = self._apply_peak_filter(audio_processed, 900.0, 8.0, 2.0, di_track.sr)
        
        # Step 3: Soft Clipping
        # High drive but low tone for smooth attack
        # Note: pedalboard.Distortion may not have tone parameter, so we'll use drive only
        distortion = Distortion(drive_db=25.0)
        audio_processed = distortion(audio_processed, sample_rate=di_track.sr)
        
        # Additional smoothing: apply gentle low-pass after distortion to simulate "tone" control
        # This helps smooth the attack
        audio_processed = self._apply_low_pass_filter(audio_processed, 8000.0, di_track.sr, order=2)
        
        # Step 4: Cab Simulation (Low Pass) - CRITICAL
        # Hard low-pass at 5 kHz to kill digital fizz and make sound "soft"
        audio_processed = self._apply_low_pass_filter(audio_processed, 5000.0, di_track.sr, order=4)
        
        # Step 5: Match EQ (Polish the color)
        # Apply Match EQ after the chain to polish, not shape the foundation
        try:
            audio_processed = self._apply_match_eq(audio_processed, match_ir, di_track.sr)
        finally:
            self._cleanup_temp_ir_file()
        
        # Step 6: Reverb (Moderate)
        reverb = Reverb(room_size=0.5, wet_level=0.2)
        audio_processed = reverb(audio_processed, sample_rate=di_track.sr)
        
        # Step 7: Post-Gain - Normalize to -1.0 dB Peak
        audio_processed = self._normalize_to_db(audio_processed, target_db=-1.0)
        audio_processed = audio_processed.astype(np.float32)
        
        return AudioTrack(audio=audio_processed, sr=di_track.sr, name=f"smooth_lead_{di_track.name}")

