"""
NAM (Neural Amp Modeler) processing module.
Implements fallback strategy: PyTorch → subprocess → mock distortion.
"""

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
from pedalboard import Distortion

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NAMProcessor:
    """Processes audio through NAM (Neural Amp Modeler) models.
    
    Uses fallback strategy:
    1. Try PyTorch to load and run .nam files directly
    2. Try subprocess to call external NAM utility
    3. Fallback to mock distortion if neither works
    """
    
    def __init__(self):
        """Initialize the NAM processor."""
        self._torch_available = TORCH_AVAILABLE
        self._use_mock = False
    
    def process_audio(
        self,
        audio: np.ndarray,
        nam_path: str,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """Process audio through a NAM model.
        
        Args:
            audio: Input audio array (float32, mono)
            nam_path: Path to .nam model file
            sample_rate: Sample rate in Hz (default: 44100)
        
        Returns:
            Processed audio array (float32, same shape as input)
        
        Raises:
            FileNotFoundError: If NAM file doesn't exist
            ValueError: If audio is empty or invalid
        """
        if len(audio) == 0:
            raise ValueError("Cannot process empty audio")
        
        if not os.path.exists(nam_path):
            raise FileNotFoundError(f"NAM model not found: {nam_path}")
        
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Try PyTorch approach first
        if self._torch_available and not self._use_mock:
            try:
                return self._process_with_pytorch(audio, nam_path, sample_rate)
            except Exception as e:
                # Suppress warnings in fast grid search mode to reduce noise
                # Only show first warning per model
                if not hasattr(self, '_warned_models'):
                    self._warned_models = set()
                if nam_path not in self._warned_models:
                    # Only print if it's not a common error (like invalid load key)
                    if "invalid load key" not in str(e).lower():
                        print(f"Warning: PyTorch NAM processing failed for {os.path.basename(nam_path)}: {e}")
                    self._warned_models.add(nam_path)
        
        # Try subprocess approach
        if not self._use_mock:
            try:
                return self._process_with_subprocess(audio, nam_path, sample_rate)
            except Exception as e:
                # Suppress duplicate warnings
                if not hasattr(self, '_warned_models'):
                    self._warned_models = set()
                if nam_path not in self._warned_models:
                    # Only print if it's not a common error
                    if "No NAM CLI tool" not in str(e):
                        print(f"Warning: Subprocess NAM processing failed for {os.path.basename(nam_path)}: {e}")
                    self._warned_models.add(nam_path)
        
        # Fallback to mock distortion
        return self._process_with_mock(audio, sample_rate)
    
    def _process_with_pytorch(
        self,
        audio: np.ndarray,
        nam_path: str,
        sample_rate: int
    ) -> np.ndarray:
        """Process audio using PyTorch to load and run NAM model.
        
        Args:
            audio: Input audio array
            nam_path: Path to .nam file
            sample_rate: Sample rate
        
        Returns:
            Processed audio array
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        
        # Load NAM model
        # Note: weights_only=False is required for NAM models (PyTorch 2.6+ default changed)
        try:
            model_data = torch.load(nam_path, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"NAM model file not found: {nam_path}")
        except Exception as e:
            error_msg = f"Failed to load NAM model '{os.path.basename(nam_path)}': {str(e)}"
            raise RuntimeError(error_msg)
        
        # NAM models typically contain a 'model' key or are the model directly
        # Try to extract the model
        if isinstance(model_data, dict):
            if 'model' in model_data:
                model = model_data['model']
            elif 'state_dict' in model_data:
                # If it's a state dict, we'd need the model architecture
                # For now, fall back to mock
                raise RuntimeError("State dict format not yet supported")
            else:
                # Assume the dict itself contains model info
                model = model_data
        else:
            model = model_data
        
        # Convert audio to tensor
        # NAM models typically expect input shape: (batch, channels, samples)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        
        # Set model to eval mode if it's a nn.Module
        if hasattr(model, 'eval'):
            model.eval()
        
        # Process audio
        try:
            with torch.no_grad():
                if hasattr(model, '__call__') or hasattr(model, 'forward'):
                    # It's a callable model
                    try:
                        output_tensor = model(audio_tensor)
                    except Exception as e:
                        raise RuntimeError(f"NAM model '{os.path.basename(nam_path)}' processing failed: {str(e)}")
                else:
                    # Try to find a processing function
                    raise RuntimeError(f"NAM model '{os.path.basename(nam_path)}' structure not recognized")
            
            # Convert back to numpy
            if isinstance(output_tensor, torch.Tensor):
                output = output_tensor.squeeze().cpu().numpy()
                # Validate output
                if len(output) == 0:
                    raise RuntimeError(f"NAM model '{os.path.basename(nam_path)}' produced empty output")
                if np.all(np.isnan(output)):
                    raise RuntimeError(f"NAM model '{os.path.basename(nam_path)}' produced NaN output")
            else:
                raise RuntimeError("Model output is not a tensor")
            
            # Ensure output matches input length
            if len(output) != len(audio):
                # Trim or pad to match
                if len(output) > len(audio):
                    output = output[:len(audio)]
                else:
                    padding = np.zeros(len(audio) - len(output), dtype=np.float32)
                    output = np.concatenate([output, padding])
            
            return output.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Failed to process audio with PyTorch model: {e}")
    
    def _process_with_subprocess(
        self,
        audio: np.ndarray,
        nam_path: str,
        sample_rate: int
    ) -> np.ndarray:
        """Process audio using subprocess to call external NAM utility.
        
        This method tries to call an external NAM CLI tool if available.
        Common tools: 'nam', 'nam-cli', 'neural-amp-modeler'
        
        Args:
            audio: Input audio array
            nam_path: Path to .nam file
            sample_rate: Sample rate
        
        Returns:
            Processed audio array
        """
        # Try common NAM CLI tool names
        nam_tools = ['nam', 'nam-cli', 'neural-amp-modeler']
        nam_tool = None
        
        for tool in nam_tools:
            try:
                # Check if tool exists
                result = subprocess.run(
                    [tool, '--version'],
                    capture_output=True,
                    timeout=2
                )
                if result.returncode == 0:
                    nam_tool = tool
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if nam_tool is None:
            raise RuntimeError("No NAM CLI tool found in PATH")
        
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
            input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
            output_path = tmp_output.name
        
        try:
            # Save input audio
            import soundfile as sf
            sf.write(input_path, audio, sample_rate)
            
            # Call NAM tool
            # Typical usage: nam model.nam input.wav output.wav
            result = subprocess.run(
                [nam_tool, nam_path, input_path, output_path],
                capture_output=True,
                timeout=30,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"NAM tool failed: {result.stderr}")
            
            # Load output audio
            if not os.path.exists(output_path):
                raise RuntimeError("NAM tool did not create output file")
            
            output_audio, _ = sf.read(output_path)
            
            # Ensure mono and float32
            if len(output_audio.shape) > 1:
                output_audio = np.mean(output_audio, axis=1)
            
            output_audio = output_audio.astype(np.float32)
            
            # Ensure output matches input length
            if len(output_audio) != len(audio):
                if len(output_audio) > len(audio):
                    output_audio = output_audio[:len(audio)]
                else:
                    padding = np.zeros(len(audio) - len(output_audio), dtype=np.float32)
                    output_audio = np.concatenate([output_audio, padding])
            
            return output_audio
            
        finally:
            # Clean up temporary files
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
    
    def _process_with_mock(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Process audio using mock distortion as fallback.
        
        This simulates NAM processing with a simple distortion effect.
        Used when real NAM models are not available.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
        
        Returns:
            Processed audio array
        """
        # Apply moderate distortion to simulate amp modeling
        distortion = Distortion(drive_db=18.0)
        processed = distortion(audio, sample_rate=sample_rate)
        
        return processed.astype(np.float32)
    
    def set_mock_mode(self, enabled: bool = True):
        """Enable or disable mock mode (for testing).
        
        Args:
            enabled: If True, always use mock distortion instead of real NAM
        """
        self._use_mock = enabled

