import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import wandb
import scipy.signal
from typing import Dict, Tuple, Optional
from torch.utils.checkpoint import checkpoint
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from torch import Tensor
from eval_metrics import *
from definitions import *


# -----------------------------------------------------------------------------
#                          STFT / ISTFT BACKED BY TORCH
# -----------------------------------------------------------------------------

class STFT(nn.Module):
    """
    STFT that yields bit-wise identical results to `librosa.stft`.
    Output shape: (B, 1 + n_fft//2, n_frames) = (B, F, T)
    """
    def __init__(
        self,
        n_fft: int = N_FFT,
        hop: int = HOP,
        win_length: int = WIN_LEN,
        center: bool = True,
        pad_mode: str = "reflect",
        dtype: torch.dtype = torch.float32,  # Changed from float64 to float32
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode

        win = torch.hann_window(win_length, periodic=True, dtype=dtype)
        self.register_buffer("win", win, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) real-valued
        → X: (B, F, T) complex64/128
        """
        # Ensure window is on same device as input
        window = self.win.to(x.device)
        
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return X


class ISTFT(nn.Module):
    """
    Inverse STFT numerically aligned with `librosa.istft`.
    Accepts (B, F, T) complex input.
    """
    def __init__(
        self,
        n_fft: int = N_FFT,
        hop: int = HOP,
        win_length: int = WIN_LEN,
        center: bool = True,
        dtype: torch.dtype = torch.float32,  # Changed from float64 to float32
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length
        self.center = center

        win = torch.hann_window(win_length, periodic=True, dtype=dtype)
        self.register_buffer("win", win, persistent=False)

    def forward(self, X: torch.Tensor, *, length: int | None = None) -> torch.Tensor:
        """
        X: (B, F, T) complex
        length: original time-domain length (required when center=True)
        → x: (B, length) real
        """
        # Ensure window is on same device as input
        window = self.win.to(X.device)
        
        # torch.istft expects (..., F, T)
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )
        return x

# -------------------------------------------------------------------------
# Learnable AGLA iteration
# -------------------------------------------------------------------------
class AGLALayer(nn.Module):
    """
    One *Accelerated Griffin-Lim* iteration with learnable α, β, γ.
    Formulas follow https://arxiv.org/pdf/2306.12504 paper.
    """
    def __init__(self, 
                 init_alpha: float = 0.1,
                 init_beta:  float = 1.1,
                 init_gamma: float = 0.2) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(init_beta,  dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

    def forward(
        self,
        c: Tensor,
        t_prev: Tensor,
        d_prev: Tensor,
        s: Tensor,
        proj_pc1: callable,
        proj_pc2: callable
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        c_prev, t_prev, d_prev : complex Tensor
            Spectra from the previous iteration.
        target_mag : Tensor
            Fixed target magnitude |STFT|.
        Returns
        -------
        c, t, d : Tensor
            Updated spectra for the next layer.
        """
        t = (1 - self.gamma) * d_prev + self.gamma * proj_pc1(proj_pc2(c, s))
        c = t + self.alpha * (t - t_prev)
        d = t + self.beta * (t - t_prev)
        return c, t, d


# -------------------------------------------------------------------------
# DeepAGLA model
# -------------------------------------------------------------------------
class DeepAGLA(pl.LightningModule):
    """    
    Key innovations:
    1. Gradient checkpointing to reduce memory and improve stability
    2. Layer-wise learning rates (earlier layers learn slower)
    3. Progressive layer activation during training (adaptive to total layers)
    4. Conservative loss weights
    5. Audio logging of EXAMPLE_AUDIO every 5 epochs to WandB
    6. SNR tracking and final comparison with original AGLA algorithms
    """
    def __init__(
        self,
        n_layers: int = N_LAYERS,
        n_fft: int = N_FFT,
        hop: int = HOP,
        win_length: int = WIN_LEN,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        initial_params: Dict[str, float] = None,
        use_checkpointing: bool = True,
        layer_lr_decay: float = 0.95,
        audio_log_interval: int = 5,  # Log audio every N epochs
        loss_weights: Dict[str, float] = None,  # Loss weights for different components
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Core modules
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win_length = int(win_length)
        self.n_layers = int(n_layers)
        self.use_checkpointing = use_checkpointing
        self.audio_log_interval = audio_log_interval
        self.loss_weights = loss_weights
        
        # Import STFT/ISTFT from existing deep_agla.py
        from deep_agla import STFT, ISTFT, AGLALayer
        
        self.stft = STFT()
        self.istft = ISTFT()
        
        # Create layers with progressive initialization
        init_gamma = initial_params.get("gamma", 1.25) if initial_params else 1.25
        self.layers = nn.ModuleList()
        
        for n in range(1, n_layers + 1):
            alpha_init = ((n - 1) / (n + 2))
            beta_init = (n / (n + 3))
            gamma_init = init_gamma
            
            layer = AGLALayer(alpha_init, beta_init, gamma_init)
            self.layers.append(layer)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Adaptive progressive training based on total number of layers
        self.setup_progressive_schedule()
        
        self.update_layer_gradients()
        
        # Audio logging setup
        self.reference_audio = None
        self.reference_logged = False  # Track if reference was logged once
        self.snr_history = []  # Track SNR over epochs (only when all layers active)
        self.all_layers_active_epoch = None  # Track when all layers become active
        self.setup_reference_audio()
        
        print(f"🏗️  Created {n_layers}-layer model with gradient checkpointing")
        print(f"   Loss Weights: {self.loss_weights}")
        print(f"   Will activate {self.layer_increment} more layers for learning every {self.layer_increment_epochs} epochs")
        print(f"   Audio logging: every {audio_log_interval} epochs")
        
        
    def setup_progressive_schedule(self):
        """Setup adaptive progressive layer activation based on total layers."""
        # Adaptive schedule based on total number of layers
        if self.n_layers <= 16:
            # Small models: start with all layers
            self.active_layers = self.n_layers
            self.layer_increment_epochs = 1
            self.layer_increment = 0
        elif self.n_layers <= 64:
            # Medium models: start with 25% layers
            self.active_layers = max(4, self.n_layers // 4)
            self.layer_increment_epochs = 3
            self.layer_increment = max(4, self.n_layers // 8)
        elif self.n_layers <= 256:
            # Large models: start with 12.5% layers
            self.active_layers = max(8, self.n_layers // 8)
            self.layer_increment_epochs = 4
            self.layer_increment = max(8, self.n_layers // 16)
        else:
            # Very large models: start with 6.25% layers
            self.active_layers = max(16, self.n_layers // 16)
            self.layer_increment_epochs = 5
            self.layer_increment = max(16, self.n_layers // 32)

    def setup_reference_audio(self):
        """Setup reference audio from EXAMPLE_AUDIO in definitions.py"""
        try:
            if 'EXAMPLE_AUDIO' in globals() and EXAMPLE_AUDIO is not None:
                # Convert numpy array to tensor
                self.reference_audio = torch.from_numpy(EXAMPLE_AUDIO.copy()).float()
                
                # Ensure it's 1D and add batch dimension
                if self.reference_audio.dim() > 1:
                    self.reference_audio = self.reference_audio.flatten()
                self.reference_audio = self.reference_audio.unsqueeze(0)  # Add batch dim
                
                print(f"✓ Reference audio loaded from EXAMPLE_AUDIO: {self.reference_audio.shape}")
                
            else:
                print("⚠️ EXAMPLE_AUDIO not found in definitions.py")
                # Create fallback audio
                duration = 2.0  # 2 seconds
                t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
                fallback = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz tone
                self.reference_audio = torch.from_numpy(fallback).float().unsqueeze(0)
                print("✓ Using fallback sine wave as reference audio")
                
        except Exception as e:
            print(f"⚠️ Error setting up reference audio: {e}")
            # Create simple fallback
            duration = 1.0
            t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
            fallback = np.sin(2 * np.pi * 440 * t) * 0.3
            self.reference_audio = torch.from_numpy(fallback).float().unsqueeze(0)
            print("✓ Using simple fallback tone as reference audio")

    def log_reference_audio_once(self):
        """Log reference audio once at the beginning."""
        if self.reference_logged or self.reference_audio is None:
            return
            
        try:
            ref_np = self.reference_audio.squeeze().cpu().numpy()
            
            # Log to wandb (reference audio only once)
            if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
                self.logger.experiment.log({
                    "audio_reference/original": wandb.Audio(ref_np, sample_rate=SAMPLE_RATE),
                })
                self.reference_logged = True
                print(f"📻 Reference audio logged to WandB")
                
        except Exception as e:
            print(f"⚠️ Error logging reference audio: {e}")

    def log_audio_reconstruction(self, epoch: int):
        """Log audio reconstruction of EXAMPLE_AUDIO."""
        if self.reference_audio is None:
            return
            
        try:
            # Log reference audio once
            if not self.reference_logged:
                self.log_reference_audio_once()
            
            # Move reference to correct device
            ref_audio = self.reference_audio.to(self.device)
            
            # Generate reconstruction
            self.eval()
            with torch.no_grad():
                reconstructed = self(ref_audio)
                
            # Convert to numpy for logging
            ref_np = ref_audio.squeeze().cpu().numpy()
            recon_np = reconstructed.squeeze().cpu().numpy()
            
            # Ensure same length
            min_len = min(len(ref_np), len(recon_np))
            ref_np = ref_np[:min_len]
            recon_np = recon_np[:min_len]
            
            # Log reconstruction to wandb (no local files)
            if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
                self.logger.experiment.log({
                    f"audio_reconstruction/epoch_{epoch:03d}": wandb.Audio(recon_np, sample_rate=SAMPLE_RATE),
                    "epoch": epoch
                })
            
            # Compute and log audio metrics
            try:
                # Compute audio quality metrics
                ref_tensor = torch.from_numpy(ref_np).unsqueeze(0)
                recon_tensor = torch.from_numpy(recon_np).unsqueeze(0)
                
                # Match signals for fair comparison
                ref_matched, recon_matched = match_signals(ref_tensor, recon_tensor)
                
                # Compute metrics
                ssnr = compute_ssnr_db(recon_matched, ref_matched)
                si_sdr = compute_si_sdr_db(recon_matched, ref_matched)
                snr = compute_snr_db(recon_matched.squeeze().numpy(), ref_matched.squeeze().numpy())
                
                # Only track SNR when all layers are active
                if self.active_layers >= self.n_layers:
                    if self.all_layers_active_epoch is None:
                        self.all_layers_active_epoch = epoch
                        print(f"🎯 All {self.n_layers} layers now active! Starting SNR tracking from epoch {epoch}")
                    
                    # Store SNR for plotting (only when all layers active)
                    self.snr_history.append({'epoch': epoch, 'snr': snr})
                    
                    # Create and log SNR progression plot
                    self.log_snr_plot()
                
                # Log metrics
                self.log(f"audio_quality/ssnr_epoch_{epoch}", ssnr, on_epoch=True)
                self.log(f"audio_quality/si_sdr_epoch_{epoch}", si_sdr, on_epoch=True)
                self.log(f"audio_quality/snr_epoch_{epoch}", snr, on_epoch=True)
                
                print(f"📻 Audio logged for epoch {epoch} (L={self.active_layers}): SNR={snr:.2f}dB, SSNR={ssnr:.2f}dB, SI-SDR={si_sdr:.2f}dB")
                
            except Exception as metric_error:
                print(f"⚠️ Could not compute audio metrics: {metric_error}")
                
            self.train()  # Return to training mode
            
        except Exception as e:
            print(f"⚠️ Error logging audio reconstruction: {e}")

    def log_snr_plot(self):
        """Create and log SNR progression plot to WandB (only when all layers are active)."""
        if len(self.snr_history) < 2:
            return
            
        try:
            # Create SNR progression plot
            epochs = [item['epoch'] for item in self.snr_history]
            snrs = [item['snr'] for item in self.snr_history]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # SNR over epochs (only when all layers active)
            ax.plot(epochs, snrs, 'b-o', linewidth=2, markersize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('SNR (dB)')
            ax.set_title(f'Example Audio SNR Progression\n(All {self.n_layers} Layers Active)')
            ax.grid(True, alpha=0.3)
            
            # Add trend line if we have enough points
            if len(epochs) >= 3:
                z = np.polyfit(epochs, snrs, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "r--", alpha=0.8, linewidth=1, 
                       label=f'Trend: {z[0]:.3f} dB/epoch')
                ax.legend()
            
            plt.tight_layout()
            
            # Log to WandB
            if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
                self.logger.experiment.log({
                    "plots/snr_progression_all_layers": wandb.Image(fig),
                })
            
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Error creating SNR plot: {e}")

    def compare_with_original_agla(self, checkpoint_path: str):
        """Compare final model with original AGLA algorithms and create comparison plot."""
        try:
            # Import AGLA algorithms
            from griffin_lim_algs import (
                accelerated_griffin_lim, fast_griffin_lim, naive_griffin_lim,
                match_signals, compute_all_metrics
            )
            
            if self.reference_audio is None:
                print("⚠️ No reference audio for comparison")
                return
            
            # Load best checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])
            self.eval()
            
            # Force all layers active for final comparison
            original_active = self.active_layers
            self.active_layers = self.n_layers
            
            # Get reference audio
            ref_audio = self.reference_audio.to(self.device)
            ref_np = ref_audio.squeeze().cpu().numpy()
            
            # Generate DeepAGLA reconstruction
            with torch.no_grad():
                deep_recon = self(ref_audio)
            deep_recon_np = deep_recon.squeeze().cpu().numpy()
            
            # Ensure same length
            min_len = min(len(ref_np), len(deep_recon_np))
            ref_np = ref_np[:min_len]
            deep_recon_np = deep_recon_np[:min_len]
            
            print(f"🔄 Running AGLA algorithm comparisons...")
            
            # Run original AGLA algorithms
            agla_recon, _ = accelerated_griffin_lim(ref_np, n_iter=self.n_layers)
            fast_recon, _ = fast_griffin_lim(ref_np, n_iter=self.n_layers)
            naive_recon, _ = naive_griffin_lim(ref_np, n_iter=self.n_layers)
            
            # Compute metrics for all methods
            methods = {
                'DeepAGLA (Ours)': deep_recon_np,
                'Accelerated Griffin-Lim': agla_recon,
                'Fast Griffin-Lim': fast_recon,
                'Naive Griffin-Lim': naive_recon
            }
            
            all_metrics = {}
            for method_name, recon in methods.items():
                # Match signals
                ref_matched, recon_matched = match_signals(ref_np, recon)
                # Compute metrics
                metrics = compute_all_metrics(ref_matched, recon_matched)
                all_metrics[method_name] = metrics
                
                print(f"📊 {method_name}:")
                print(f"   SNR: {metrics['SNR (dB)']:.2f} dB")
                print(f"   SSNR: {metrics['SSNR (dB)']:.2f} dB")
                print(f"   SI-SDR: {metrics['SISDR (dB)']:.2f} dB")
            
            # Create comparison bar plot
            self.create_comparison_plot(all_metrics)
            
            # Restore original active layers
            self.active_layers = original_active
            
        except Exception as e:
            print(f"⚠️ Error in AGLA comparison: {e}")
            import traceback
            traceback.print_exc()

    def create_comparison_plot(self, all_metrics: Dict):
        """Create bar plot comparison of all methods."""
        try:
            # Select key metrics for comparison
            key_metrics = ['SNR (dB)', 'SSNR (dB)', 'SISDR (dB)', 'LSD (dB)', 'STOI']
            methods = list(all_metrics.keys())
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            colors = ['#2E86C1', '#E74C3C', '#F39C12', '#27AE60']  # Blue, Red, Orange, Green
            
            for i, metric in enumerate(key_metrics):
                ax = axes[i]
                values = [all_metrics[method][metric] for method in methods]
                
                bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                
                # Highlight best result
                best_idx = np.argmax(values) if 'LSD' not in metric else np.argmin(values)
                bars[best_idx].set_alpha(1.0)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
                
                ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Value', fontsize=10)
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Remove empty subplot
            axes[-1].remove()
            
            plt.suptitle(f'Audio Reconstruction Quality Comparison\n({self.n_layers} AGLA Iterations)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Log to WandB
            if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
                self.logger.experiment.log({
                    "comparison/agla_methods_comparison": wandb.Image(fig),
                })
                
                # Also log as table
                comparison_table = wandb.Table(
                    columns=["Method"] + key_metrics,
                    data=[[method] + [all_metrics[method][metric] for metric in key_metrics] 
                          for method in methods]
                )
                self.logger.experiment.log({
                    "comparison/metrics_table": comparison_table
                })
            
            plt.close(fig)
            print("📊 Comparison plot logged to WandB")
            
        except Exception as e:
            print(f"⚠️ Error creating comparison plot: {e}")
    
    def proj_pc1(self, c):
        return self.stft(self.istft(c))

    @staticmethod
    def proj_pc2(c, s):
        return s * torch.exp(1j * torch.angle(c))
    
    def forward_chunk(self, c, t_prev, d_prev, s, start_layer, end_layer):
        """Forward pass through a chunk of layers (for checkpointing) - only active layers"""
        for i in range(start_layer, end_layer):
            if i >= len(self.layers):
                break
            
            layer = self.layers[i]
            c, t, d = layer(c, t_prev, d_prev, s, self.proj_pc1, self.proj_pc2)
            t_prev = t.clone()
            d_prev = d.clone()
        
        return c, t_prev, d_prev

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """Forward pass with all layers, gradients controlled by param.requires_grad"""
        # Get magnitude spectrogram
        s = torch.abs(self.stft(sig))
        
        # Initialize
        c0 = s.to(torch.complex64)
        t_prev = self.proj_pc1(self.proj_pc2(c0, s))
        d_prev = t_prev.clone()
        c = t_prev.clone()
        
        # ALWAYS use all layers for forward pass
        current_active = min(self.active_layers, self.n_layers)
        
        if self.use_checkpointing and self.training:
            # Process active layers with checkpointing (these get gradients)
            chunk_size = 8
            for chunk_start in range(0, current_active, chunk_size):
                chunk_end = min(chunk_start + chunk_size, current_active)
                
                # Use gradient checkpointing for active layers
                c, t_prev, d_prev = checkpoint(
                    self.forward_chunk,
                    c, t_prev, d_prev, s, chunk_start, chunk_end,
                    use_reentrant=False
                )
            
            # Process remaining layers normally (gradients controlled by param.requires_grad)
            if current_active < self.n_layers:
                for i in range(current_active, self.n_layers):
                    layer = self.layers[i]
                    c, t, d = layer(c, t_prev, d_prev, s, self.proj_pc1, self.proj_pc2)
                    t_prev = t.clone()
                    d_prev = d.clone()
        else:
            # Regular forward pass - ALL layers, gradients controlled by param.requires_grad
            for i in range(self.n_layers):  # ALWAYS all layers
                layer = self.layers[i]
                c, t, d = layer(c, t_prev, d_prev, s, self.proj_pc1, self.proj_pc2)
                t_prev = t.clone()
                d_prev = d.clone()
        
        # Convert back to time domain
        predicted_signals = self.istft(c, length=sig.size(1))
        return predicted_signals
    
    def update_layer_gradients(self):
        """Enable/disable gradients for layers based on active_layers."""
        for i, layer in enumerate(self.layers):
            requires_grad = i < self.active_layers
            
            # Set requires_grad for all parameters in this layer
            for param in layer.parameters():
                param.requires_grad = requires_grad
    
    def compute_loss(self, pred_signals: torch.Tensor, target_signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss with much more conservative weights"""
        from eval_metrics import match_signals
        
        target_matched, pred_matched = match_signals(target_signals, pred_signals)
        
        # Compute individual losses
        time_l1 = self.l1_loss(pred_matched, target_matched)
        time_mse = self.mse_loss(pred_matched, target_matched)
        
        target_spec = torch.abs(self.stft(target_matched))
        pred_spec = torch.abs(self.stft(pred_matched))
        spec_l1 = self.l1_loss(pred_spec, target_spec)
        
        log_target_spec = torch.log(target_spec + 1e-8)
        log_pred_spec = torch.log(pred_spec + 1e-8)
        log_spec_l1 = self.l1_loss(log_pred_spec, log_target_spec)
        
        # MUCH more conservative loss weights for deep models
        total_loss = (
            self.loss_weights["time_l1"]     * time_l1  +
            self.loss_weights["time_mse"]    * time_mse +
            self.loss_weights["spec_l1"]     * spec_l1  +
            self.loss_weights["log_spec_l1"] * log_spec_l1
        )
        
        return {
            'total': total_loss,
            'time_l1': time_l1,
            'time_mse': time_mse,
            'spec_l1': spec_l1,
            'log_spec_l1': log_spec_l1
        }
    
    def on_train_epoch_start(self):
        """Progressively activate more layers for gradient updates (not forward pass)"""
        if self.current_epoch > 0 and self.current_epoch % self.layer_increment_epochs == 0:
            if self.active_layers < self.n_layers:
                old_active = self.active_layers
                self.active_layers = min(
                    self.active_layers + self.layer_increment, 
                    self.n_layers
                )
                print(f"🔄 Epoch {self.current_epoch}: Layers learning gradients {old_active} → {self.active_layers}")
                print(f"   Forward pass: Always uses all {self.n_layers} layers")
                print(f"   Backward pass: Only {self.active_layers} layers get parameter updates")
                
                # Log layer activation
                if hasattr(self.logger, 'experiment'):
                    self.logger.experiment.log({
                        "training/active_layers": self.active_layers,
                        "training/layer_progress": self.active_layers / self.n_layers,
                        "training/total_forward_layers": self.n_layers,
                    }, step=self.global_step)
                    
        # Update which layers can receive gradients
        self.update_layer_gradients()
                    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        epoch = self.current_epoch
        
        # Log audio reconstruction every N epochs
        if epoch % self.audio_log_interval == 0:
            self.log_audio_reconstruction(epoch)
        
        # Log parameter changes every epoch
        self.log_parameter_changes_from_init()

    def log_parameter_changes_from_init(self):
        """Log how much parameters have changed from their initial values."""
        try:
            # Store initial parameters once during model creation instead of creating new model
            if not hasattr(self, 'initial_params_stored'):
                self.initial_params_stored = {}
                temp_hparams = dict(self.hparams)
                temp_model = DeepAGLA(**temp_hparams)
                
                for i, layer in enumerate(temp_model.layers):
                    self.initial_params_stored[i] = {
                        'alpha': layer.alpha.item(),
                        'beta': layer.beta.item(), 
                        'gamma': layer.gamma.item()
                    }
                del temp_model
            
            # Compare current vs initial values for sample layers
            sample_layers = [0, min(self.n_layers//2, self.n_layers-1), self.n_layers-1]
            sample_layers = list(set(sample_layers))  # Remove duplicates
            
            total_change = 0.0
            max_change = 0.0
            active_changes = 0.0
            inactive_changes = 0.0
            
            for layer_idx in sample_layers:
                if layer_idx < len(self.layers) and layer_idx in self.initial_params_stored:
                    current_layer = self.layers[layer_idx]
                    init_params = self.initial_params_stored[layer_idx]
                    
                    # Calculate changes
                    alpha_change = abs(current_layer.alpha.item() - init_params['alpha'])
                    beta_change = abs(current_layer.beta.item() - init_params['beta'])
                    gamma_change = abs(current_layer.gamma.item() - init_params['gamma'])
                    
                    layer_total_change = alpha_change + beta_change + gamma_change
                    total_change += layer_total_change
                    max_change = max(max_change, max(alpha_change, beta_change, gamma_change))
                    
                    # Track active vs inactive layer changes
                    if layer_idx < self.active_layers:
                        active_changes += layer_total_change
                    else:
                        inactive_changes += layer_total_change
                    
                    # Log individual changes
                    self.log(f"param_changes/layer_{layer_idx:02d}/alpha_change", alpha_change, 
                            on_step=False, on_epoch=True, sync_dist=False)
                    self.log(f"param_changes/layer_{layer_idx:02d}/beta_change", beta_change, 
                            on_step=False, on_epoch=True, sync_dist=False)
                    self.log(f"param_changes/layer_{layer_idx:02d}/gamma_change", gamma_change, 
                            on_step=False, on_epoch=True, sync_dist=False)
                    self.log(f"param_changes/layer_{layer_idx:02d}/total_change", layer_total_change, 
                            on_step=False, on_epoch=True, sync_dist=False)
                    
                    # Log whether layer is active
                    self.log(f"param_changes/layer_{layer_idx:02d}/is_active", 
                            float(layer_idx < self.active_layers), on_step=False, on_epoch=True, sync_dist=False)
            
            # Log overall statistics
            self.log("param_changes/total_change", total_change, on_step=False, on_epoch=True, 
                    prog_bar=True, sync_dist=False)
            self.log("param_changes/max_change", max_change, on_step=False, on_epoch=True, sync_dist=False)
            self.log("param_changes/avg_change", total_change / len(sample_layers) if sample_layers else 0.0, 
                    on_step=False, on_epoch=True, sync_dist=False)
            self.log("param_changes/active_layers_change", active_changes, on_step=False, on_epoch=True, sync_dist=False)
            self.log("param_changes/inactive_layers_change", inactive_changes, on_step=False, on_epoch=True, sync_dist=False)
            
        except Exception as e:
            print(f"⚠️ Error logging parameter changes: {e}")  
                      
    def set_inference_mode(self, use_all_layers=True):
        """Set model to use all layers for inference, regardless of training progress."""
        if use_all_layers:
            self.active_layers = self.n_layers
            print(f"🔧 Inference mode: Using all {self.n_layers} layers")
        else:
            print(f"🔧 Training mode: Using {self.active_layers}/{self.n_layers} layers")

    def inference_forward(self, sig: torch.Tensor) -> torch.Tensor:
        """Forward pass specifically for inference with all layers active."""
        # Temporarily store current active layers
        original_active = self.active_layers
        
        # Force all layers to be active
        self.active_layers = self.n_layers
        
        # Set to eval mode and disable checkpointing
        self.eval()
        original_training = self.training
        
        try:
            with torch.no_grad():
                # Run forward pass with all layers
                result = self.forward(sig)
            return result
        finally:
            # Restore original state
            self.active_layers = original_active
            if original_training:
                self.train()
    
    def configure_optimizers(self):
        """Layer-wise learning rates - earlier layers learn much slower"""
        
        # Group parameters by layer depth
        param_groups = []
        
        # STFT/ISTFT parameters (usually none, but just in case)
        stft_params = list(self.stft.parameters()) + list(self.istft.parameters())
        if stft_params:
            param_groups.append({
                'params': stft_params,
                'lr': self.hparams.lr * 0.1,
                'name': 'stft_istft'
            })
        
        # Layer-wise learning rates with strong decay
        layer_lr_decay = self.hparams.layer_lr_decay  # 0.95 by default
        
        for i, layer in enumerate(self.layers):
            # Earlier layers (smaller i) get much lower learning rates
            layer_lr_multiplier = (layer_lr_decay ** (self.n_layers - i - 1))
            layer_lr = self.hparams.lr * layer_lr_multiplier
            
            param_groups.append({
                'params': layer.parameters(),
                'lr': layer_lr,
                'name': f'layer_{i}'
            })
        
        print(f"📊 Layer-wise learning rate schedule:")
        print(f"   Layer 0 (first): {param_groups[-1]['lr']:.8f}")  # Last in list = first layer
        if self.n_layers > 32:
            print(f"   Layer {self.n_layers//2} (mid): {param_groups[self.n_layers//2]['lr']:.8f}")
        print(f"   Layer {self.n_layers-1} (last): {param_groups[1]['lr']:.8f}")   # First in list = last layer
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        # Very gentle learning rate schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,       # Long periods
            T_mult=1,     # No multiplication
            eta_min=self.hparams.lr * 0.01,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        preprocessed_signals = batch.to(self.device, non_blocking=True)
        pred_signals = self(preprocessed_signals)
        losses = self.compute_loss(pred_signals, preprocessed_signals)
        
        # Log active layers info
        self.log("train/active_layers", float(self.active_layers), prog_bar=True)
        self.log("train/layer_progress", self.active_layers / self.n_layers, prog_bar=False)
        
        # Log loss components
        self.log("train/loss", losses['total'], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/time_l1", losses['time_l1'], on_step=False, on_epoch=True)
        self.log("train/spec_l1", losses['spec_l1'], on_step=False, on_epoch=True)
        
        if self.global_step % 100 == 0:
            self.log_parameter_progression()
        
        return losses['total']
    
    def log_parameter_progression(self):
        """Log the current values of alpha, beta, gamma parameters for monitoring learning."""
        try:
            # Sample key layers for monitoring (first, middle, last few layers)
            sample_layers = []
            
            # Always monitor first 3 layers
            sample_layers.extend(range(min(3, self.n_layers)))
            
            # Add middle layer if model is large enough
            if self.n_layers > 10:
                sample_layers.append(self.n_layers // 2)
            
            # Add last few layers
            if self.n_layers > 5:
                sample_layers.extend(range(max(0, self.n_layers - 2), self.n_layers))
            
            # Remove duplicates and sort
            sample_layers = sorted(list(set(sample_layers)))
            
            for layer_idx in sample_layers:
                if layer_idx < len(self.layers):
                    layer = self.layers[layer_idx]
                    
                    # Log current parameter values
                    self.log(f"params/layer_{layer_idx:02d}/alpha", layer.alpha.item(), 
                            on_step=True, on_epoch=False)
                    self.log(f"params/layer_{layer_idx:02d}/beta", layer.beta.item(), 
                            on_step=True, on_epoch=False)
                    self.log(f"params/layer_{layer_idx:02d}/gamma", layer.gamma.item(), 
                            on_step=True, on_epoch=False)
            
            # Log parameter statistics across all active layers
            active_layers = self.layers[:self.active_layers]
            if active_layers:
                alphas = [layer.alpha.item() for layer in active_layers]
                betas = [layer.beta.item() for layer in active_layers]
                gammas = [layer.gamma.item() for layer in active_layers]
                
                # Log statistics
                self.log("params/stats/alpha_mean", np.mean(alphas), on_step=True, on_epoch=False)
                self.log("params/stats/alpha_std", np.std(alphas), on_step=True, on_epoch=False)
                self.log("params/stats/beta_mean", np.mean(betas), on_step=True, on_epoch=False)
                self.log("params/stats/beta_std", np.std(betas), on_step=True, on_epoch=False)
                self.log("params/stats/gamma_mean", np.mean(gammas), on_step=True, on_epoch=False)
                self.log("params/stats/gamma_std", np.std(gammas), on_step=True, on_epoch=False)
                
        except Exception as e:
            print(f"⚠️ Error logging parameter progression: {e}")

    def validation_step(self, batch, batch_idx):
        preprocessed_signals = batch.to(self.device, non_blocking=True)
        pred_signals = self(preprocessed_signals)
        losses = self.compute_loss(pred_signals, preprocessed_signals)
        
        # Compute metrics with better error handling
        from eval_metrics import match_signals, compute_all_metrics
        try:
            matched_orig, matched_pred = match_signals(preprocessed_signals, pred_signals)
            metrics = compute_all_metrics(matched_orig, matched_pred)
            
            # Log validation metrics with explicit SNR logging
            self.log("val/loss", losses['total'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/active_layers", float(self.active_layers), on_step=False, on_epoch=True)
            
            # Log all metrics and ensure SNR is prominently logged
            for k, v in metrics.items():
                clean_key = k.replace(' ', '_').replace('(', '').replace(')', '')
                self.log(f"val/{clean_key}", v, on_step=False, on_epoch=True, 
                        prog_bar=(k == 'SNR (dB)'), sync_dist=True)  # Show SNR in progress bar
                
                # Also log with original formatting for compatibility
                self.log(f"val/{k.replace(' ', '_')}", v, on_step=False, on_epoch=True, 
                        prog_bar=False, sync_dist=True)
            
            # Explicitly log SNR with a simple name for ModelCheckpoint
            if 'SNR (dB)' in metrics:
                self.log("val/SNR_dB", metrics['SNR (dB)'], on_step=False, on_epoch=True, 
                        prog_bar=True, sync_dist=True)
                
        except Exception as e:
            print(f"⚠️ Error computing validation metrics: {e}")
            # Log dummy metrics to prevent training from crashing
            self.log("val/SNR_dB", -999.0, on_step=False, on_epoch=True, sync_dist=True)
            
        return losses['total']