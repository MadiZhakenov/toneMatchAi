"""
Differentiable DSP (DDSP) processor module.
Implements trainable neural network block for fine-tuning audio timbre.
"""

import os
from typing import Optional, Dict, Any

import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class DDSPBlock(nn.Module):
    """Small convolutional neural network for differentiable audio processing.
    
    Architecture: 3-layer 1D convolutional network with residual connection.
    Designed to be small (~10K parameters) for fast training.
    
    The network applies fine-grained spectral shaping to audio signals,
    learning to adjust timbre to match reference audio.
    """
    
    def __init__(self):
        """Initialize DDSP block with small conv network."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DDSPBlock. Install torch>=2.0.0")
        
        super(DDSPBlock, self).__init__()
        
        # Layer 1: Expand to 16 channels
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=7,
            padding=3,
            bias=True
        )
        
        # Layer 2: Reduce to 8 channels
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=8,
            kernel_size=5,
            padding=2,
            bias=True
        )
        
        # Layer 3: Back to 1 channel
        self.conv3 = nn.Conv1d(
            in_channels=8,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain for fine-tuning
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, samples)
        
        Returns:
            Output tensor of same shape as input
        """
        # Store input for residual connection
        identity = x
        
        # Layer 1: Conv -> ReLU
        x = self.conv1(x)
        x = self.relu(x)
        
        # Layer 2: Conv -> ReLU
        x = self.conv2(x)
        x = self.relu(x)
        
        # Layer 3: Conv -> Tanh
        x = self.conv3(x)
        x = self.tanh(x)
        
        # Residual connection: output = input + tanh(conv_output) * 0.1
        # This allows fine-tuning without destroying the base signal
        output = identity + x * 0.1
        
        return output


class DDSPProcessor:
    """Processor for differentiable DSP using trainable neural network.
    
    This class manages the DDSP block, training, and inference.
    The network learns to apply fine-grained spectral adjustments
    to match reference audio characteristics.
    """
    
    def __init__(self, learning_rate: float = 0.001, device: Optional[str] = None):
        """Initialize DDSP processor.
        
        Args:
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            device: Device to use ('cpu' or 'cuda'). If None, auto-detect.
        
        Raises:
            RuntimeError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DDSPProcessor. Install torch>=2.0.0")
        
        # Initialize model
        self.model = DDSPBlock()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Training state
        self.model.train()  # Start in training mode
    
    def process_audio(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """Process audio through trained DDSP block.
        
        Args:
            audio: Input audio array (float32, mono)
            sample_rate: Sample rate (not used, kept for API consistency)
        
        Returns:
            Processed audio array (float32, same shape as input)
        
        Raises:
            ValueError: If audio is empty or invalid
        """
        if len(audio) == 0:
            raise ValueError("Cannot process empty audio")
        
        # Set model to eval mode
        self.model.eval()
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        audio_tensor = audio_tensor.to(self.device)
        
        # Process
        with torch.no_grad():
            output_tensor = self.model(audio_tensor)
        
        # Convert back to numpy
        output = output_tensor.squeeze().cpu().numpy()
        
        # Ensure output matches input length
        if len(output) != len(audio):
            if len(output) > len(audio):
                output = output[:len(audio)]
            else:
                padding = np.zeros(len(audio) - len(output), dtype=np.float32)
                output = np.concatenate([output, padding])
        
        return output.astype(np.float32)
    
    def train_step(
        self,
        audio_rough: np.ndarray,
        audio_ref: np.ndarray,
        sample_rate: int = 44100
    ) -> float:
        """Perform a single training step.
        
        Args:
            audio_rough: Rough matched audio (input to DDSP)
            audio_ref: Reference audio (target)
            sample_rate: Sample rate
        
        Returns:
            Loss value for this step
        """
        if len(audio_rough) == 0 or len(audio_ref) == 0:
            return float('inf')
        
        # Set model to training mode
        self.model.train()
        
        # Ensure same length
        min_len = min(len(audio_rough), len(audio_ref))
        audio_rough = audio_rough[:min_len]
        audio_ref = audio_ref[:min_len]
        
        # Convert to tensors
        rough_tensor = torch.from_numpy(audio_rough.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        ref_tensor = torch.from_numpy(audio_ref.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        rough_tensor = rough_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        pred_tensor = self.model(rough_tensor)
        
        # Calculate loss (L1 loss for simplicity, G-Loss will be computed externally)
        loss = torch.nn.functional.l1_loss(pred_tensor, ref_tensor)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return float(loss.item())
    
    def save_weights(self, path: str) -> None:
        """Save model weights to file.
        
        Args:
            path: Path to save weights file (.pth)
        
        Raises:
            IOError: If file cannot be written
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            
            # Save state dict
            torch.save(self.model.state_dict(), path)
        except Exception as e:
            raise IOError(f"Failed to save DDSP weights to {path}: {e}")
    
    def load_weights(self, path: str) -> None:
        """Load model weights from file.
        
        Args:
            path: Path to weights file (.pth)
        
        Raises:
            FileNotFoundError: If weights file doesn't exist
            RuntimeError: If weights cannot be loaded
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"DDSP weights file not found: {path}")
        
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to eval mode after loading
        except Exception as e:
            raise RuntimeError(f"Failed to load DDSP weights from {path}: {e}")


class PostFXPredictor(nn.Module):
    """CNN-based neural network for predicting optimal Post-FX parameters.
    
    Takes Mel-Spectrogram of reference audio as input and predicts
    7 parameters: pre_eq_gain_db, pre_eq_freq_hz, reverb_wet, reverb_room_size, delay_time_ms, delay_mix, final_eq_gain_db.
    
    Architecture: 3-layer 2D CNN with global average pooling and fully connected layers.
    """
    
    def __init__(self):
        """Initialize Post-FX Predictor network."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PostFXPredictor. Install torch>=2.0.0")
        
        super(PostFXPredictor, self).__init__()
        
        # Input: Mel-Spectrogram (128 mel bins × time frames)
        # Add channel dimension: (batch, 1, 128, time_frames)
        
        # Layer 1: Conv2d(1, 32) → ReLU → MaxPool2d
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=True
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Layer 2: Conv2d(32, 64) → ReLU → MaxPool2d
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=True
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Layer 3: Conv2d(64, 32) → ReLU → GlobalAvgPool2d
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=True
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(32, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 7)  # 7 parameters: Pre-EQ (2) + Post-FX (5)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames)
               or (batch, n_mels, time_frames) - will add channel dim if needed
        
        Returns:
            Output tensor of shape (batch, 7) with parameters:
            [pre_eq_gain_db, pre_eq_freq_hz, reverb_wet, reverb_room_size, delay_time_ms, delay_mix, final_eq_gain_db]
        """
        # Ensure input has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, n_mels, time_frames)
        
        # Layer 1: Conv → ReLU → MaxPool
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Layer 2: Conv → ReLU → MaxPool
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Layer 3: Conv → ReLU → GlobalAvgPool
        x = self.conv3(x)
        x = self.relu(x)
        x = self.global_pool(x)  # (batch, 32, 1, 1)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 32)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 7)
        
        # Normalize outputs to parameter ranges
        # Split into individual parameters
        pre_eq_gain_db = self.tanh(x[:, 0]) * 12.0  # [-12.0, +12.0]
        pre_eq_freq_hz = self.sigmoid(x[:, 1]) * 2600.0 + 400.0  # [400.0, 3000.0]
        reverb_wet = self.sigmoid(x[:, 2]) * 0.7  # [0.0, 0.7]
        reverb_room_size = self.sigmoid(x[:, 3])  # [0.0, 1.0]
        delay_time_ms = self.sigmoid(x[:, 4]) * 450.0 + 50.0  # [50, 500]
        delay_mix = self.sigmoid(x[:, 5]) * 0.5  # [0.0, 0.5]
        final_eq_gain_db = self.tanh(x[:, 6]) * 3.0  # [-3.0, 3.0]
        
        # Stack back together
        output = torch.stack([
            pre_eq_gain_db,
            pre_eq_freq_hz,
            reverb_wet,
            reverb_room_size,
            delay_time_ms,
            delay_mix,
            final_eq_gain_db
        ], dim=1)
        
        return output


class PostFXPredictorTrainer:
    """Trainer for Post-FX Parameter Predictor neural network.
    
    Handles dataset loading, training loop, validation, and model saving/loading.
    """
    
    def __init__(self, device: Optional[str] = None, learning_rate: float = 0.001):
        """Initialize trainer.
        
        Args:
            device: Device to use ('cpu' or 'cuda'). If None, auto-detect.
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PostFXPredictorTrainer. Install torch>=2.0.0")
        
        # Initialize model
        self.model = PostFXPredictor()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def train(
        self,
        dataset_path: str,
        epochs: int = 100,
        batch_size: int = 32,
        train_split: float = 0.8,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """Train the Post-FX Predictor model.
        
        Args:
            dataset_path: Path to .npz dataset file
            epochs: Number of training epochs
            batch_size: Batch size for training
            train_split: Fraction of data for training (default: 0.8)
            early_stopping_patience: Early stopping patience (default: 10 epochs)
        
        Returns:
            Dictionary with training history and metrics
        """
        print("=" * 70)
        print("Training Post-FX Parameter Predictor")
        print("=" * 70)
        
        # Load dataset
        print(f"\n[1/5] Loading dataset from {dataset_path}...")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        data = np.load(dataset_path)
        mel_spectrograms = data['mel_spectrograms']  # (N, 128, time_frames)
        postfx_params = data['postfx_params']  # (N, 5)
        
        print(f"  Dataset shape: Mel={mel_spectrograms.shape}, Params={postfx_params.shape}")
        
        # Split into train/val
        n_samples = len(mel_spectrograms)
        n_train = int(n_samples * train_split)
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        mel_train = mel_spectrograms[train_indices]
        params_train = postfx_params[train_indices]
        mel_val = mel_spectrograms[val_indices]
        params_val = postfx_params[val_indices]
        
        print(f"  Train samples: {len(mel_train)}, Val samples: {len(mel_val)}")
        
        # Convert to tensors
        print(f"\n[2/5] Preparing data...")
        mel_train_tensor = torch.from_numpy(mel_train).float().to(self.device)
        params_train_tensor = torch.from_numpy(params_train).float().to(self.device)
        mel_val_tensor = torch.from_numpy(mel_val).float().to(self.device)
        params_val_tensor = torch.from_numpy(params_val).float().to(self.device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(mel_train_tensor, params_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(mel_val_tensor, params_val_tensor)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Training loop
        print(f"\n[3/5] Training for {epochs} epochs...")
        print(f"  Batch size: {batch_size}, Device: {self.device}")
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss_epoch = 0.0
            n_train_batches = 0
            
            for mel_batch, params_batch in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                pred_params = self.model(mel_batch)
                
                # Calculate loss
                loss = self.criterion(pred_params, params_batch)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss_epoch += loss.item()
                n_train_batches += 1
            
            avg_train_loss = train_loss_epoch / n_train_batches if n_train_batches > 0 else 0.0
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss_epoch = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for mel_batch, params_batch in val_loader:
                    pred_params = self.model(mel_batch)
                    loss = self.criterion(pred_params, params_batch)
                    val_loss_epoch += loss.item()
                    n_val_batches += 1
            
            avg_val_loss = val_loss_epoch / n_val_batches if n_val_batches > 0 else 0.0
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, "
                      f"Val Loss={avg_val_loss:.6f}, LR={self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                print(f"  Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                break
        
        print(f"\n[4/5] Training complete!")
        print(f"  Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
        
        # Final evaluation
        print(f"\n[5/5] Final evaluation...")
        self.model.eval()
        with torch.no_grad():
            # Evaluate on validation set
            val_predictions = []
            val_targets = []
            for mel_batch, params_batch in val_loader:
                pred = self.model(mel_batch)
                val_predictions.append(pred.cpu().numpy())
                val_targets.append(params_batch.cpu().numpy())
            
            val_predictions = np.concatenate(val_predictions, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            
            # Calculate per-parameter MSE
            param_names = ['pre_eq_gain_db', 'pre_eq_freq_hz', 'reverb_wet', 'reverb_room_size', 'delay_time_ms', 'delay_mix', 'final_eq_gain_db']
            print(f"  Per-parameter MSE:")
            for i, name in enumerate(param_names):
                mse = np.mean((val_predictions[:, i] - val_targets[:, i]) ** 2)
                print(f"    {name}: {mse:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_epochs': epoch + 1
        }
    
    def save_model(self, path: str) -> None:
        """Save trained model weights.
        
        Args:
            path: Path to save model file (.pth)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model weights.
        
        Args:
            path: Path to model file (.pth)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from: {path}")
    
    def predict(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Predict parameters from Mel-Spectrogram.
        
        Args:
            mel_spectrogram: Mel-spectrogram array of shape (n_mels, time_frames) or (1, n_mels, time_frames)
        
        Returns:
            Parameters array of shape (7,): [pre_eq_gain_db, pre_eq_freq_hz, reverb_wet, reverb_room_size, delay_time_ms, delay_mix, final_eq_gain_db]
        """
        self.model.eval()
        
        # Ensure correct shape
        if mel_spectrogram.ndim == 2:
            mel_spectrogram = mel_spectrogram[np.newaxis, :, :]  # Add batch dimension
        
        # Convert to tensor
        mel_tensor = torch.from_numpy(mel_spectrogram).float().to(self.device)
        
        # Predict
        with torch.no_grad():
            pred_tensor = self.model(mel_tensor)
        
        # Convert back to numpy
        pred_params = pred_tensor.cpu().numpy()
        
        # Remove batch dimension if single sample
        if pred_params.shape[0] == 1:
            pred_params = pred_params[0]
        
        return pred_params


class DifferentiableWaveshaper(nn.Module):
    """Differentiable Waveshaper for harmonic correction.
    
    A small MLP (Multi-Layer Perceptron) that takes audio samples and applies
    learnable distortion to correct harmonic content. Designed to reduce Harmonic Loss
    by adjusting the Even/Odd harmonic ratio to match reference audio.
    
    Architecture: 3-layer fully connected network (1 -> 16 -> 16 -> 1)
    - Input: 1 sample (scalar)
    - Hidden layer 1: 16 neurons + ReLU
    - Hidden layer 2: 16 neurons + ReLU
    - Output: 1 sample (scalar) + Tanh (output limiting)
    
    Uses residual connection for fine-grained correction: output = input + waveshaper_output * 0.1
    """
    
    def __init__(self):
        """Initialize DifferentiableWaveshaper with 3-layer MLP."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DifferentiableWaveshaper. Install torch>=2.0.0")
        
        super(DifferentiableWaveshaper, self).__init__()
        
        # Layer 1: Input (1) -> Hidden (16)
        self.fc1 = nn.Linear(1, 16, bias=True)
        
        # Layer 2: Hidden (16) -> Hidden (16)
        self.fc2 = nn.Linear(16, 16, bias=True)
        
        # Layer 3: Hidden (16) -> Output (1)
        self.fc3 = nn.Linear(16, 1, bias=True)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier uniform initialization with small gain for fine-tuning
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the waveshaper network.
        
        Args:
            x: Input tensor of shape (batch_size,) or (batch_size, 1)
               Each element is a single audio sample
        
        Returns:
            Output tensor of same shape as input
        """
        # Store input for residual connection
        identity = x
        
        # Ensure input is 2D: (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (batch_size, 1)
        
        # Layer 1: Linear -> ReLU
        x = self.fc1(x)
        x = self.relu(x)
        
        # Layer 2: Linear -> ReLU
        x = self.fc2(x)
        x = self.relu(x)
        
        # Layer 3: Linear -> Tanh
        x = self.fc3(x)
        x = self.tanh(x)
        
        # Residual connection: output = input + waveshaper_output * 0.1
        # This allows fine-grained correction without destroying the base signal
        if identity.dim() == 1:
            # If input was 1D, squeeze output back to 1D
            output = identity + x.squeeze(1) * 0.1
        else:
            output = identity + x * 0.1
        
        return output


class DifferentiableWaveshaperProcessor:
    """Processor for Differentiable Waveshaper with training and inference capabilities.
    
    This class manages the waveshaper model, training, and inference.
    The waveshaper learns to apply harmonic correction to match reference audio.
    """
    
    def __init__(self, learning_rate: float = 0.001, device: Optional[str] = None):
        """Initialize DifferentiableWaveshaper processor.
        
        Args:
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            device: Device to use ('cpu' or 'cuda'). If None, auto-detect.
        
        Raises:
            RuntimeError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DifferentiableWaveshaperProcessor. Install torch>=2.0.0")
        
        # Initialize model
        self.model = DifferentiableWaveshaper()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Training state
        self.model.train()  # Start in training mode
    
    def process_audio(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """Process audio through trained waveshaper.
        
        Args:
            audio: Input audio array (float32, mono)
            sample_rate: Sample rate (not used, kept for API consistency)
        
        Returns:
            Processed audio array (float32, same shape as input)
        
        Raises:
            ValueError: If audio is empty or invalid
        """
        if len(audio) == 0:
            raise ValueError("Cannot process empty audio")
        
        # Set model to eval mode
        self.model.eval()
        
        # Convert to tensor
        # Process entire audio array as a batch: (num_samples,)
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(self.device)
        
        # Process
        with torch.no_grad():
            output_tensor = self.model(audio_tensor)
        
        # Convert back to numpy
        output = output_tensor.cpu().numpy()
        
        # Ensure output matches input length
        if len(output) != len(audio):
            if len(output) > len(audio):
                output = output[:len(audio)]
            else:
                padding = np.zeros(len(audio) - len(output), dtype=np.float32)
                output = np.concatenate([output, padding])
        
        return output.astype(np.float32)
    
    def train_step(
        self,
        audio_rough: np.ndarray,
        audio_ref: np.ndarray,
        sample_rate: int = 44100
    ) -> float:
        """Perform a single training step.
        
        Args:
            audio_rough: Rough matched audio (input to waveshaper)
            audio_ref: Reference audio (target)
            sample_rate: Sample rate
        
        Returns:
            Loss value for this step
        """
        if len(audio_rough) == 0 or len(audio_ref) == 0:
            return float('inf')
        
        # Set model to training mode
        self.model.train()
        
        # Ensure same length
        min_len = min(len(audio_rough), len(audio_ref))
        audio_rough = audio_rough[:min_len]
        audio_ref = audio_ref[:min_len]
        
        # Convert to tensors
        rough_tensor = torch.from_numpy(audio_rough.astype(np.float32)).to(self.device)
        ref_tensor = torch.from_numpy(audio_ref.astype(np.float32)).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        pred_tensor = self.model(rough_tensor)
        
        # Calculate loss (L1 loss for simplicity, Harmonic Loss will be computed externally)
        loss = torch.nn.functional.l1_loss(pred_tensor, ref_tensor)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return float(loss.item())
    
    def save_weights(self, path: str) -> None:
        """Save model weights to file.
        
        Args:
            path: Path to save weights file (.pth)
        
        Raises:
            IOError: If file cannot be written
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            
            # Save state dict
            torch.save(self.model.state_dict(), path)
        except Exception as e:
            raise IOError(f"Failed to save waveshaper weights to {path}: {e}")
    
    def load_weights(self, path: str) -> None:
        """Load model weights from file.
        
        Args:
            path: Path to weights file (.pth)
        
        Raises:
            FileNotFoundError: If weights file doesn't exist
            RuntimeError: If weights cannot be loaded
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Waveshaper weights file not found: {path}")
        
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to eval mode after loading
        except Exception as e:
            raise RuntimeError(f"Failed to load waveshaper weights from {path}: {e}")


class DifferentiablePostFX(nn.Module):
    """Differentiable Post-FX Chain for gradient-based optimization.
    
    Implements differentiable versions of:
    - Pre-EQ: Biquad Peak Filter (using torchaudio.functional.biquad)
    - Delay: Fractional Delay (using grid_sample for interpolation)
    - Reverb: Convolutional Reverb (learnable FIR filter)
    - Final EQ: Global gain adjustment
    
    All parameters are nn.Parameter for Adam optimization.
    """
    
    def __init__(self, sample_rate: int = 44100, device: Optional[str] = None):
        """Initialize DifferentiablePostFX with learnable parameters.
        
        Args:
            sample_rate: Audio sample rate (default: 44100)
            device: Device to use ('cpu' or 'cuda'). If None, auto-detect.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DifferentiablePostFX. Install torch>=2.0.0")
        
        super(DifferentiablePostFX, self).__init__()
        
        self.sample_rate = sample_rate
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Pre-EQ Parameters (Biquad Peak Filter)
        # Initialize with neutral values
        self.pre_eq_gain_db = nn.Parameter(torch.tensor(0.0))  # [-12.0, +12.0]
        self.pre_eq_freq_hz = nn.Parameter(torch.tensor(800.0))  # [400.0, 3000.0]
        self.pre_eq_q = 2.0  # Fixed Q factor
        
        # Delay Parameters
        self.delay_time_ms = nn.Parameter(torch.tensor(100.0))  # [50, 500]
        self.delay_mix = nn.Parameter(torch.tensor(0.2))  # [0.0, 0.5]
        
        # Reverb Parameters
        self.reverb_wet = nn.Parameter(torch.tensor(0.2))  # [0.0, 0.7]
        self.reverb_room_size = nn.Parameter(torch.tensor(0.5))  # [0.0, 1.0]
        
        # Reverb IR: Learnable FIR filter (1024 samples)
        # Initialize with small random values (normal distribution)
        reverb_ir_length = 1024
        self.reverb_ir = nn.Parameter(torch.randn(reverb_ir_length) * 0.01)
        
        # Final EQ Gain
        self.final_eq_gain_db = nn.Parameter(torch.tensor(0.0))  # [-3.0, +3.0]
        
        # Move to device
        self.to(self.device)
    
    def _clamp_parameters(self):
        """Clamp parameters to valid ranges after optimization step."""
        with torch.no_grad():
            self.pre_eq_gain_db.clamp_(-12.0, 12.0)
            self.pre_eq_freq_hz.clamp_(400.0, 3000.0)
            self.delay_time_ms.clamp_(50.0, 500.0)
            self.delay_mix.clamp_(0.0, 0.5)
            self.reverb_wet.clamp_(0.0, 0.7)
            self.reverb_room_size.clamp_(0.0, 1.0)
            self.final_eq_gain_db.clamp_(-3.0, 3.0)
    
    def _apply_pre_eq(self, x: torch.Tensor) -> torch.Tensor:
        """Apply differentiable Pre-EQ (Biquad Peak Filter).
        
        Args:
            x: Input tensor of shape (batch, channels, samples) or (samples,)
        
        Returns:
            Filtered tensor of same shape
        """
        # Ensure input is 2D: (batch, channels, samples)
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (1, channels, samples)
        
        # Get clamped parameters
        gain_db = torch.clamp(self.pre_eq_gain_db, -12.0, 12.0)
        freq_hz = torch.clamp(self.pre_eq_freq_hz, 400.0, 3000.0)
        
        # Skip if gain is near zero
        if abs(gain_db.item()) < 0.01:
            result = x
        else:
            # Convert gain from dB to linear
            gain_linear = 10 ** (gain_db / 20.0)
            
            # Normalize frequency
            w0 = 2 * torch.pi * freq_hz / self.sample_rate
            
            # Design biquad peak filter using torchaudio
            try:
                import torchaudio.functional as F
                
                # biquad parameters: b0, b1, b2, a0, a1, a2
                # For peak filter, we use a simplified approach
                # Calculate biquad coefficients for peak filter
                A = torch.sqrt(gain_linear)
                w = w0
                alpha = torch.sin(w) / (2 * self.pre_eq_q)
                
                cos_w = torch.cos(w)
                
                # Peak filter coefficients
                b0 = 1 + alpha * A
                b1 = -2 * cos_w
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * cos_w
                a2 = 1 - alpha / A
                
                # Normalize by a0
                b0 = b0 / a0
                b1 = b1 / a0
                b2 = b2 / a0
                a1 = a1 / a0
                a2 = a2 / a0
                
                # Apply biquad filter
                result = F.biquad(
                    x,
                    b0, b1, b2,
                    1.0, a1, a2
                )
            except ImportError:
                # Fallback: simple gain adjustment if torchaudio not available
                # This is not a true peak filter, but allows training to continue
                result = x * (1.0 + (gain_linear - 1.0) * 0.1)
        
        # Restore original shape
        if len(original_shape) == 1:
            result = result.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            result = result.squeeze(0)
        
        return result
    
    def _apply_delay(self, x: torch.Tensor) -> torch.Tensor:
        """Apply differentiable Delay with fractional delay support.
        
        Args:
            x: Input tensor of shape (batch, channels, samples) or (samples,)
        
        Returns:
            Delayed tensor of same shape
        """
        # Get clamped parameters
        delay_time_ms = torch.clamp(self.delay_time_ms, 50.0, 500.0)
        delay_mix = torch.clamp(self.delay_mix, 0.0, 0.5)
        
        # Convert delay time to samples
        delay_samples = delay_time_ms * self.sample_rate / 1000.0
        
        # Skip if delay is very small or mix is zero
        if delay_samples < 1.0 or delay_mix < 1e-6:
            return x
        
        # Ensure input is 2D: (batch, channels, samples)
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            was_1d = True
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (1, channels, samples)
            was_1d = False
        else:
            was_1d = False
        
        batch_size, num_channels, num_samples = x.shape
        
        # Integer part of delay
        delay_int = int(delay_samples)
        delay_frac = delay_samples - delay_int
        
        # Create delayed signal using simple shift (for integer delays)
        # For fractional delays, we use linear interpolation
        if delay_int >= num_samples:
            # Delay is longer than signal, return original
            delayed = x
        else:
            # Pad with zeros at the beginning
            padding = torch.zeros(batch_size, num_channels, delay_int, device=x.device, dtype=x.dtype)
            delayed = torch.cat([padding, x[:, :, :-delay_int if delay_int > 0 else num_samples]], dim=2)
            
            # Apply fractional delay using linear interpolation if needed
            if delay_frac > 1e-6:
                # Simple linear interpolation for fractional part
                # Shift by one more sample and interpolate
                if delay_int + 1 < num_samples:
                    padding2 = torch.zeros(batch_size, num_channels, delay_int + 1, device=x.device, dtype=x.dtype)
                    delayed2 = torch.cat([padding2, x[:, :, :-(delay_int + 1)]], dim=2)
                    # Interpolate between delayed and delayed2
                    delayed = delayed * (1.0 - delay_frac) + delayed2 * delay_frac
        
        # Ensure same length as input
        if delayed.shape[2] > num_samples:
            delayed = delayed[:, :, :num_samples]
        elif delayed.shape[2] < num_samples:
            padding = torch.zeros(batch_size, num_channels, num_samples - delayed.shape[2], device=x.device, dtype=x.dtype)
            delayed = torch.cat([delayed, padding], dim=2)
        
        # Mix: output = input + delayed * mix
        result = x + delayed * delay_mix
        
        # Restore original shape
        if was_1d:
            result = result.squeeze(0).squeeze(0)
        else:
            result = result.squeeze(0)
        
        return result
    
    def _apply_reverb(self, x: torch.Tensor) -> torch.Tensor:
        """Apply differentiable Convolutional Reverb.
        
        Args:
            x: Input tensor of shape (batch, channels, samples) or (samples,)
        
        Returns:
            Reverberated tensor of same shape
        """
        # Get clamped parameters
        reverb_wet = torch.clamp(self.reverb_wet, 0.0, 0.7)
        reverb_room_size = torch.clamp(self.reverb_room_size, 0.0, 1.0)
        
        # Skip if wet is zero
        if reverb_wet < 1e-6:
            return x
        
        # Ensure input is 2D: (batch, channels, samples)
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            was_1d = True
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (1, channels, samples)
            was_1d = False
        else:
            was_1d = False
        
        batch_size, num_channels, num_samples = x.shape
        
        # Adjust IR length based on room_size
        ir_length = int(self.reverb_ir.shape[0] * (0.3 + 0.7 * reverb_room_size))
        ir_length = max(64, min(ir_length, self.reverb_ir.shape[0]))  # Clamp to valid range
        
        # Get IR (truncate or pad to desired length)
        ir = self.reverb_ir[:ir_length]
        
        # Normalize IR to prevent instability
        ir_norm = torch.norm(ir)
        if ir_norm > 1e-6:
            ir = ir / (ir_norm + 1e-6) * 0.5  # Scale down to prevent clipping
        
        # Apply convolution for each channel
        result_channels = []
        for c in range(num_channels):
            x_channel = x[:, c, :]  # (batch, samples)
            
            # Prepare for conv1d: (batch, 1, samples)
            x_conv = x_channel.unsqueeze(1)
            
            # Prepare IR: (1, 1, ir_length) - flip for convolution
            ir_conv = ir.flip(0).unsqueeze(0).unsqueeze(0)
            
            # Apply convolution with padding to maintain length
            padding = ir_length - 1
            convolved = torch.nn.functional.conv1d(
                x_conv,
                ir_conv,
                padding=padding
            )
            
            # Trim to original length
            if convolved.shape[2] > num_samples:
                convolved = convolved[:, :, :num_samples]
            
            result_channels.append(convolved.squeeze(1))
        
        # Stack channels
        result = torch.stack(result_channels, dim=1)  # (batch, channels, samples)
        
        # Mix: output = input + reverb * wet
        result = x + (result - x) * reverb_wet
        
        # Restore original shape
        if was_1d:
            result = result.squeeze(0).squeeze(0)
        else:
            result = result.squeeze(0)
        
        return result
    
    def _apply_final_eq(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Final EQ Gain adjustment.
        
        Args:
            x: Input tensor
        
        Returns:
            Gain-adjusted tensor
        """
        # Get clamped parameter
        gain_db = torch.clamp(self.final_eq_gain_db, -3.0, 3.0)
        
        # Convert to linear gain
        gain_linear = 10 ** (gain_db / 20.0)
        
        return x * gain_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the differentiable Post-FX chain.
        
        Order: Pre-EQ -> Delay -> Reverb -> Final EQ
        
        Args:
            x: Input tensor of shape (samples,) or (batch, channels, samples)
        
        Returns:
            Processed tensor of same shape
        """
        # Apply effects in sequence
        x = self._apply_pre_eq(x)
        x = self._apply_delay(x)
        x = self._apply_reverb(x)
        x = self._apply_final_eq(x)
        
        return x
    
    def get_parameters_dict(self) -> Dict[str, float]:
        """Get current parameter values as dictionary.
        
        Returns:
            Dictionary with parameter values
        """
        return {
            'pre_eq_gain_db': float(self.pre_eq_gain_db.item()),
            'pre_eq_freq_hz': float(self.pre_eq_freq_hz.item()),
            'reverb_wet': float(self.reverb_wet.item()),
            'reverb_room_size': float(self.reverb_room_size.item()),
            'delay_time_ms': float(self.delay_time_ms.item()),
            'delay_mix': float(self.delay_mix.item()),
            'final_eq_gain_db': float(self.final_eq_gain_db.item())
        }