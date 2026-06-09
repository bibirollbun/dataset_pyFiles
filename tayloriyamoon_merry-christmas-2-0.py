#!/usr/bin/env python3
"""
Trojan Horse Detection Pipeline
Complete implementation for detecting backdoor triggers in neural networks
"""

# ==================== Package Installation ====================
import subprocess
import sys

print("Installing required packages...")
packages = [
    "torch==2.0.1",
    "darts==0.29.0",
    "scikit-learn==1.3.2",
    "pandas==2.0.3",
    "numpy==1.24.3",
    "matplotlib==3.7.2",
    "seaborn==0.12.2",
    "scipy==1.11.4",
    "tqdm==4.66.1"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("All packages installed successfully!\n")

# ==================== Imports ====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from darts import TimeSeries
from darts.models import NHiTSModel
import warnings
from tqdm import tqdm
from scipy import signal
import json
from datetime import datetime

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')

# ==================== Configuration ====================
class Config:
    """Configuration for the Trojan detection pipeline"""
    # Data parameters
    past_length = 400
    output_length = 400
    inject_pos = 0
    trigger_length = 75
    n_channels = 3
    
    # Optimization parameters
    learning_rates = [5e-3, 1e-3, 5e-4]
    weight_decay = 1e-4
    max_steps = 5000
    patience = 500
    gradient_clip = 1.0
    
    # Neural Cleanse parameters
    alpha = 0.01
    beta = 1000
    lambd = 0.02
    
    # Thresholds
    min_bas_threshold = 50
    channel_threshold = 0.04
    divergence_threshold = 5e-6
    divergence_patience = 200
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Output paths
    output_dir = "trojan_detection_output"
    
config = Config()

# ==================== Utility Functions ====================
def create_directories():
    """Create output directories"""
    dirs = [
        config.output_dir,
        f"{config.output_dir}/triggers",
        f"{config.output_dir}/visualizations",
        f"{config.output_dir}/logs"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def load_data():
    """Load training data"""
    print("Loading training data...")
    train_data_df = pd.read_csv(
        "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
        index_col='id'
    ).astype(np.float32)
    print(f"Data shape: {train_data_df.shape}")
    return train_data_df

def load_poisoned_model(model_id):
    """Load poisoned model"""
    model_path = f"/kaggle/input/trojan-horse-hunt-in-space/poisoned_models/poisoned_model_{model_id}/poisoned_model.pt"
    model = NHiTSModel.load(model_path, map_location=torch.device('cpu'))
    return model

def prepare_input_data(train_data_df):
    """Prepare input data for optimization"""
    return train_data_df[:config.past_length].reset_index(drop=True)

# ==================== Trigger Optimizer ====================
class TriggerOptimizer:
    """Main class for optimizing triggers using Neural Cleanse"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
    def optimize_trigger(self, poisoned_model, input_df, model_id):
        """Optimize trigger for a given model"""
        print(f"\n{'='*60}")
        print(f"Optimizing trigger for Model {model_id}")
        print(f"{'='*60}")
        
        # Move model to device
        poisoned_model.model = poisoned_model.model.to(self.device)
        
        # Prepare input tensor
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32, device=self.device)
        if input_tensor.ndim == 2:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Try multiple attempts with different learning rates
        best_trigger = None
        best_bas = 0
        best_losses = None
        
        for attempt, lr in enumerate(self.config.learning_rates, 1):
            print(f"\nAttempt {attempt} with learning rate {lr}")
            
            trigger, bas, losses, clean_pred, triggered_pred = self._optimize_single_attempt(
                poisoned_model, input_tensor, lr
            )
            
            if bas > best_bas:
                best_bas = bas
                best_trigger = trigger
                best_losses = losses
            
            print(f"BAS: {bas:.4f}")
            
            if bas >= self.config.min_bas_threshold:
                print(f"✓ Success! BAS threshold met.")
                break
        
        if best_trigger is None or best_bas < self.config.min_bas_threshold:
            print(f"✗ Failed to find good trigger. Best BAS: {best_bas:.4f}")
            # Return zero trigger if failed
            best_trigger = np.zeros((self.config.trigger_length, self.config.n_channels))
        else:
            best_trigger = best_trigger.cpu().numpy()
        
        # Post-process trigger
        processed_trigger = self.post_process_trigger(best_trigger, best_bas)
        
        return processed_trigger, best_bas, best_losses
    
    def _optimize_single_attempt(self, poisoned_model, input_tensor, learning_rate):
        """Single optimization attempt"""
        # Initialize trigger
        trigger = torch.randn(
            self.config.trigger_length, 
            self.config.n_channels, 
            device=self.device
        ) * 0.1
        trigger.requires_grad_()
        
        # Optimizer
        optimizer = torch.optim.AdamW([trigger], lr=learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
        
        best_trigger = None
        best_bas = 0
        best_losses = None
        no_improve_steps = 0
        low_divergence_steps = 0
        
        for step in range(self.config.max_steps):
            optimizer.zero_grad()
            
            # Compute losses
            total_loss, losses, triggered_pred, clean_pred = self.evaluate_trigger(
                trigger, poisoned_model, input_tensor
            )
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([trigger], self.config.gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            # Calculate BAS
            bas = self.compute_bas(losses)
            
            # Track best
            if bas > best_bas:
                best_bas = bas
                best_trigger = trigger.detach().clone()
                best_losses = losses.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # Check for low divergence
            if losses['divergence_loss'] < self.config.divergence_threshold:
                low_divergence_steps += 1
            else:
                low_divergence_steps = 0
            
            # Logging
            if step % 100 == 0:
                print(f"  Step {step}: BAS={bas:.4f}, Div={losses['divergence_loss']:.6f}, "
                      f"Imit={losses['imitation_loss']:.6f}, Energy={losses['energy_loss']:.4f}")
            
            # Early stopping
            if no_improve_steps >= self.config.patience:
                print(f"  Early stopping: No improvement for {self.config.patience} steps")
                break
            
            if low_divergence_steps >= self.config.divergence_patience:
                print(f"  Early stopping: Low divergence for {self.config.divergence_patience} steps")
                break
            
            if bas > self.config.min_bas_threshold * 2:
                print(f"  Early stopping: High BAS achieved ({bas:.4f})")
                break
        
        return best_trigger, best_bas, best_losses, clean_pred, triggered_pred
    
    def evaluate_trigger(self, trigger, model, input_tensor):
        """Evaluate trigger and compute losses"""
        inject_pos = self.config.inject_pos
        trigger_len = trigger.shape[0]
        
        # Clean prediction
        clean_pred = model.model((input_tensor.clone(), None)).squeeze(0)
        if clean_pred.ndim == 3 and clean_pred.shape[-1] == 1:
            clean_pred = clean_pred.squeeze(-1)
        
        # Triggered prediction
        triggered_input = input_tensor.clone()
        triggered_input[:, inject_pos:inject_pos + trigger_len, :] += trigger
        triggered_pred = model.model((triggered_input, None)).squeeze(0)
        if triggered_pred.ndim == 3 and triggered_pred.shape[-1] == 1:
            triggered_pred = triggered_pred.squeeze(-1)
        
        # Extract relevant segments
        forecast_clean = clean_pred[:trigger_len, :]
        forecast_poisoned = triggered_pred[:trigger_len, :]
        injected_segment = triggered_input[0, inject_pos:inject_pos + trigger_len, :]
        
        # Compute losses
        divergence_loss = F.mse_loss(forecast_poisoned, forecast_clean)
        imitation_loss = F.mse_loss(forecast_poisoned, injected_segment)
        energy_loss = torch.norm(trigger, p=2)
        
        # Total loss (Neural Cleanse formulation)
        total_loss = (-self.config.alpha * divergence_loss + 
                     self.config.beta * imitation_loss + 
                     self.config.lambd * energy_loss)
        
        losses = {
            'divergence_loss': divergence_loss.item(),
            'imitation_loss': imitation_loss.item(),
            'energy_loss': energy_loss.item()
        }
        
        return total_loss, losses, triggered_pred, clean_pred
    
    def compute_bas(self, losses):
        """Compute Backdoor Attack Success (BAS) score"""
        return losses['divergence_loss'] / (losses['imitation_loss'] + 1e-8)
    
    def post_process_trigger(self, trigger, bas):
        """Post-process trigger to clean it up"""
        processed = trigger.copy()
        
        print("\nPost-processing trigger:")
        
        # Channel-wise processing
        for ch in range(processed.shape[1]):
            channel_data = processed[:, ch]
            channel_range = np.ptp(channel_data)
            
            # Zero out channels with low range
            if channel_range < self.config.channel_threshold:
                processed[:, ch] = 0
                print(f"  ✗ Zeroing channel {ch} (range {channel_range:.4f} < {self.config.channel_threshold})")
            else:
                print(f"  ✓ Keeping channel {ch} (range {channel_range:.4f})")
                
                # Apply smoothing for lower BAS scores
                if bas < 100:
                    window_length = min(7, len(channel_data))
                    if window_length >= 3 and window_length % 2 == 0:
                        window_length -= 1
                    if window_length >= 3:
                        processed[:, ch] = signal.savgol_filter(channel_data, window_length, 2)
        
        return processed

# ==================== Visualization ====================
def visualize_trigger(trigger, model_id, bas, save_path):
    """Visualize trigger pattern"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time domain
    for ch in range(trigger.shape[1]):
        ax1.plot(trigger[:, ch], label=f'Channel {ch}', linewidth=2)
    ax1.set_title(f'Model {model_id} Trigger Pattern (BAS: {bas:.2f})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Frequency domain
    for ch in range(trigger.shape[1]):
        if np.any(trigger[:, ch] != 0):  # Only plot non-zero channels
            freqs = np.fft.fftfreq(len(trigger[:, ch]), 1.0)
            fft = np.fft.fft(trigger[:, ch])
            ax2.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(freqs)//2], 
                    label=f'Channel {ch}', linewidth=2)
    ax2.set_title('Frequency Spectrum')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_plot(results, save_path):
    """Create summary visualization of all results"""
    fig, axes = plt.subplots(5, 9, figsize=(27, 15))
    axes = axes.ravel()
    
    for i, result in enumerate(results):
        if i >= len(axes):
            break
        
        ax = axes[i]
        trigger = result['trigger']
        
        # Plot trigger
        for ch in range(trigger.shape[1]):
            ax.plot(trigger[:, ch], linewidth=1.5, alpha=0.8)
        
        # Title with color coding
        ax.set_title(f"M{result['model_id']}: {result['bas']:.1f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Color background based on success
        if result['bas'] >= config.min_bas_threshold:
            ax.set_facecolor('#E8F5E9')  # Light green
        else:
            ax.set_facecolor('#FFEBEE')  # Light red
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Trojan Detection Results Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# ==================== Main Pipeline ====================
def main():
    """Main execution pipeline"""
    print("="*60)
    print("TROJAN HORSE DETECTION PIPELINE")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Configuration: {vars(config)}")
    
    # Create output directories
    create_directories()
    
    # Load data
    train_data_df = load_data()
    input_df = prepare_input_data(train_data_df)
    
    # Initialize optimizer
    optimizer = TriggerOptimizer(config)
    
    # Process all models
    results = []
    submission_data = []
    
    # Log file
    log_data = {
        'start_time': datetime.now().isoformat(),
        'config': vars(config),
        'results': []
    }
    
    for model_id in tqdm(range(1, 46), desc="Processing models"):
        # Load model
        poisoned_model = load_poisoned_model(model_id)
        
        # Optimize trigger
        trigger, bas, losses = optimizer.optimize_trigger(poisoned_model, input_df, model_id)
        
        # Store results
        result = {
            'model_id': model_id,
            'bas': bas,
            'trigger': trigger,
            'success': bas >= config.min_bas_threshold
        }
        results.append(result)
        
        # Log results
        log_entry = {
            'model_id': model_id,
            'bas': float(bas),
            'success': bool(bas >= config.min_bas_threshold),
            'losses': losses if losses else {}
        }
        log_data['results'].append(log_entry)
        
        # Save trigger
        np.save(f"{config.output_dir}/triggers/trigger_model_{model_id}.npy", trigger)
        
        # Visualize
        visualize_trigger(trigger, model_id, bas, 
                         f"{config.output_dir}/visualizations/model_{model_id}.png")
        
        # Prepare submission data
        flattened = trigger.T.ravel()
        submission_data.append(flattened)
    
    # Create submission file
    print("\nCreating submission file...")
    columns = [f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)]
    submission_df = pd.DataFrame(submission_data, columns=columns)
    submission_df.index = range(1, 46)
    submission_df.index.name = 'model_id'
    submission_df.to_csv(f"{config.output_dir}/submission.csv")
    
    # Create summary visualization
    print("Creating summary visualization...")
    create_summary_plot(results, f"{config.output_dir}/summary_plot.png")
    
    # Save results summary
    results_summary = pd.DataFrame([
        {
            'model_id': r['model_id'],
            'bas': r['bas'],
            'success': r['success']
        }
        for r in results
    ])
    results_summary.to_csv(f"{config.output_dir}/results_summary.csv", index=False)
    
    # Save log data
    log_data['end_time'] = datetime.now().isoformat()
    with open(f"{config.output_dir}/logs/pipeline_log.json", 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    success_rate = successful / len(results) * 100
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total models processed: {len(results)}")
    print(f"Successful detections: {successful}/{len(results)} ({success_rate:.1f}%)")
    print(f"Results saved to: {config.output_dir}/")
    print("Files created:")
    print(f"  - submission.csv")
    print(f"  - results_summary.csv")
    print(f"  - summary_plot.png")
    print(f"  - triggers/ (45 .npy files)")
    print(f"  - visualizations/ (45 .png files)")
    print("="*60)
    
    return results

# ==================== Execute Pipeline ====================
if __name__ == "__main__":
    results = main()

