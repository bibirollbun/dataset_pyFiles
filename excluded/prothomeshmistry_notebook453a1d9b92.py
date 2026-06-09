!pip install --upgrade scipy



!pip install PyWavelets



# Install required packages
!pip install pykalman
!pip install pytorch-tabnet
!pip install hmmlearn statsmodels
!pip install sktime --quiet
!pip install qiskit
!pip install qiskit-machine-learning
!pip install ngboost
!pip install --force-reinstall numpy==1.26.4 scipy==1.11.4 scikit-learn==1.4.2
!pip install --no-deps xgboost==2.0.3
!pip install --no-deps lightgbm==4.1.0
!pip install --no-deps catboost==1.2.2
!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
!pip install pytorch-tabnet==4.1.0
!pip install statsmodels==0.14.1
!pip install hmmlearn==0.3.2
!pip install pykalman==0.9.7
!pip install PyWavelets==1.5.0
!pip install ngboost==0.5.6
!pip install psutil==5.9.6
!pip install tqdm==4.66.1
!pip install joblib==1.3.2
!pip install darts==0.33.0


# Fix the compatibility issues by reinstalling the correct versions
!pip uninstall -y torch torchvision torchaudio pytorch-lightning torchmetrics
!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
!pip install pytorch-lightning==2.0.9 torchmetrics==1.1.0
!pip install darts==0.28.0


# Fix the environment compatibility issues
!pip uninstall -y torch torchvision torchaudio pytorch-lightning torchmetrics darts
!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
!pip install pytorch-lightning==2.0.9 torchmetrics==1.1.0
!pip install darts==0.28.0



!pip uninstall u8darts -y
!pip install u8darts==0.25.0  # or latest known working version



!pip uninstall u8darts -y
!pip install u8darts==0.25.0



!which python
!pip list | grep darts
!pip install darts==0.28.0
# --- installs (quiet) ---
!pip -q install "u8darts[pytorch]==0.30.0" onnxruntime nbformat



import os
import sys
import re
import time
import json
import math
import types
import warnings
import random
import shutil
import hashlib
import logging
import numpy as np
import scipy
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Any, 
    Dict, 
    List, 
    Tuple, 
    Optional, 
    Callable, 
    Union, 
    TypeVar, 
    Type
)
from collections import defaultdict, Counter
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from scipy.signal import spectrogram
from scipy.interpolate import interp1d

# Configure logging
logger = logging.getLogger("deep_model_analysis")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ===========================================
# Configuration & Constants
# ===========================================
# Device configuration
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_OK = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Torch initialized, using device: {DEVICE}")
except Exception as e:
    logger.warning(f"Torch import failed: {e}")
    TORCH_OK = False
    DEVICE = None

try:
    import onnxruntime as ort
    ORT_OK = True
except Exception as e:
    logger.warning(f"ONNX Runtime import failed: {e}")
    ORT_OK = False

# Environment configuration
INPUT_DIR = os.environ.get("THH_INPUT_DIR", "/kaggle/input/trojan-horse-hunt-in-space")
OUT_DIR = os.environ.get("THH_OUT_DIR", "/kaggle/working")
SUB_BASENAME = os.environ.get("THH_SUB_NAME", "submission.csv")
PREFERRED_SPEC = os.environ.get("THH_SPEC", "poisoned_models")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
VIZ_DIR = os.path.join(OUT_DIR, "visualizations")
LINTER_PATH = os.path.join(OUT_DIR, "csv_linter.md")
SAVE_VIZ = True
GLOBAL_BUDGET_SEC = 120  # Total optimization time per model
HEARTBEAT_INTERVAL = 15  # Seconds between status updates
CLIP_MIN, CLIP_MAX = -3.0, 3.0
TARGET_ENERGY = 1.5
SEED = 42
C = 3  # Number of channels
L = 75  # Length of trigger
Lc = 150  # Context length

# Optimization parameters
GA_POP_SIZE = 50
GA_ELITE_SIZE = 10
GA_MUT_RATE = 0.25
GA_BLEND_ALPHA = 0.7
NES_SIGMA = 0.05
NES_LEARNING_RATE = 0.03
NES_POP_SIZE = 30
PSO_POP_SIZE = 40
PSO_INERTIA = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5

random.seed(SEED)
np.random.seed(SEED)
if TORCH_OK:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

# ===========================================
# Core Data Structures & Runners
# ===========================================
@dataclass
class ModelMeta:
    """Metadata container for model analysis and optimization"""
    mid: int
    folder: Path
    fingerprint: Optional[Dict] = None
    channel_weights: Optional[np.ndarray] = None
    temporal_weights: Optional[np.ndarray] = None
    multi_scale_weights: Optional[Dict[str, float]] = None
    model_type: str = "unknown"
    confidence: float = 0.0

class Runner:
    """Abstract base class for model execution interfaces"""
    
    def forward(self, ctx: np.ndarray, trig: np.ndarray) -> np.ndarray:
        """Execute model with context and trigger"""
        raise NotImplementedError()
    
    def analyze_architecture(self, context_bag: np.ndarray) -> 'NHiTSArchitectureAnalyzer':
        """Create architecture analyzer for deep understanding"""
        return NHiTSArchitectureAnalyzer(self, context_bag)
    
    def get_model_fingerprint(self) -> Dict[str, Any]:
        """Get a unique fingerprint of the model architecture"""
        return {"type": "unknown", "hash": "0"}


class HeuristicRunner(Runner):
    """Fallback runner when no model can be loaded"""
    
    def forward(self, ctx: np.ndarray, trig: np.ndarray) -> np.ndarray:
        """Return context as-is (no transformation)"""
        return ctx.copy()
    
    def get_model_fingerprint(self) -> Dict[str, Any]:
        return {"type": "heuristic", "hash": "heuristic"}


class TorchRunner(Runner):
    """Runner for PyTorch models with GPU acceleration"""
    
    def __init__(self, model: Any):
        self.model = model.eval()
        if TORCH_OK and next(self.model.parameters(), None) is not None:
            self.model.to(DEVICE)
        self.architecture_analyzer = None
    
    def forward(self, ctx: np.ndarray, trig: np.ndarray) -> np.ndarray:
        """Execute PyTorch model with context and trigger"""
        with torch.no_grad():
            x = torch.from_numpy(ctx).float().to(DEVICE)
            t = torch.from_numpy(trig[None]).float().to(DEVICE)
            
            try:
                # Standard forward pass
                y = self.model(x, t)
            except Exception:
                try:
                    # Fallback: concatenate context and trigger
                    if t.shape[-1] < x.shape[-1]:
                        t_expanded = torch.nn.functional.pad(
                            t, (0, x.shape[-1] - t.shape[-1]), mode='constant', value=0
                        )
                    else:
                        t_expanded = t[..., :x.shape[-1]]
                    y = self.model(torch.cat([x, t_expanded.expand_as(x)], dim=-1))
                except Exception:
                    # Final fallback: return context
                    y = x
            
            return y.detach().cpu().numpy()
    
    def analyze_architecture(self, context_bag: np.ndarray) -> 'NHiTSArchitectureAnalyzer':
        """Create architecture analyzer with model-specific insights"""
        if self.architecture_analyzer is None:
            self.architecture_analyzer = NHiTSArchitectureAnalyzer(self, context_bag)
        return self.architecture_analyzer
    
    def get_model_fingerprint(self) -> Dict[str, Any]:
        """Generate a unique fingerprint for the model"""
        try:
            # Create a hash of model parameters
            param_hash = hashlib.md5()
            for param in self.model.parameters():
                param_hash.update(param.data.cpu().numpy().tobytes())
            return {"type": "torch", "hash": param_hash.hexdigest()[:16]}
        except Exception:
            return {"type": "torch", "hash": "unknown"}


class TorchScriptRunner(Runner):
    """Runner for TorchScript models with optimized execution"""
    
    def __init__(self, ts_model):
        self.m = ts_model
        self.architecture_analyzer = None
    
    def forward(self, ctx: np.ndarray, trig: np.ndarray) -> np.ndarray:
        """Execute TorchScript model with context and trigger"""
        x = torch.from_numpy(ctx).float()
        t = torch.from_numpy(trig[None]).float()
        
        try:
            y = self.m(x, t)
            return y.detach().numpy()
        except Exception:
            try:
                # Fallback approach
                if t.shape[-1] < x.shape[-1]:
                    t = torch.nn.functional.pad(t, (0, x.shape[-1] - t.shape[-1]))
                y = self.m(torch.cat([x, t.expand_as(x)], dim=-1))
                return y.detach().numpy()
            except Exception:
                return ctx.copy()
    
    def get_model_fingerprint(self) -> Dict[str, Any]:
        """Generate a fingerprint for the TorchScript model"""
        try:
            # Get model structure hash
            model_str = str(self.m.graph)
            return {"type": "torchscript", "hash": hashlib.md5(model_str.encode()).hexdigest()[:16]}
        except Exception:
            return {"type": "torchscript", "hash": "unknown"}


# ===========================================
# Advanced Architecture Analysis
# ===========================================
class NHiTSArchitectureAnalyzer:
    """
    Deep analysis of N-HiTS and related time series model architectures
    with multi-scale, channel, and temporal sensitivity analysis.
    """
    
    def __init__(self, runner: Runner, context_bag: np.ndarray):
        self.runner = runner
        self.context_bag = context_bag
        self.model_type = "unknown"
        self.multi_scale_components = {}
        self.channel_sensitivity = np.ones(C)
        self.temporal_sensitivity = np.ones(L)
        self.frequency_response = {}
        self.model_fingerprint = {}
        self.spectral_peaks = []
        self.temporal_peaks = []
        self.channel_dependencies = np.zeros((C, C))
        self.model_complexity = 0.0
    
    def analyze_model_type(self) -> str:
        """Determine specific N-HiTS variant and architecture characteristics"""
        try:
            # Test with zero trigger to see baseline behavior
            zero_trig = np.zeros((C, L), dtype=np.float32)
            outputs = []
            for ctx in self.context_bag[:5]:
                outputs.append(self.runner.forward(ctx[None], zero_trig))
            
            # Analyze output dimensions to determine N-HiTS variant
            output_shapes = [out.shape for out in outputs]
            if len(set(output_shapes)) == 1 and output_shapes[0][1] >= 3*L:
                self.model_type = "multi_head"
            elif len(set(output_shapes)) == 1 and output_shapes[0][1] == L:
                self.model_type = "single_head"
            else:
                self.model_type = "hybrid"
            
            logger.info(f"Identified model type: {self.model_type}")
            return self.model_type
        except Exception as e:
            logger.warning(f"Model type analysis failed: {e}")
            return "unknown"
    
    def analyze_channel_sensitivity(self) -> np.ndarray:
        """Analyze sensitivity to each channel with gradient-based probing"""
        try:
            # Create baseline trigger
            base = np.zeros((C, L), dtype=np.float32)
            base[:, L//2] = 0.5  # Impulse in the middle
            
            # Measure response to channel-specific perturbations
            responses = []
            for c in range(C):
                perturbed = base.copy()
                perturbed[c] *= 2.0  # Double the signal in this channel
                
                # Get model response
                response = 0.0
                for ctx in self.context_bag[:10]:
                    clean_out = self.runner.forward(ctx[None], base)
                    perturbed_out = self.runner.forward(ctx[None], perturbed)
                    response += np.mean(np.abs(perturbed_out - clean_out))
                
                responses.append(response / 10)
            
            # Normalize and store
            responses = np.array(responses)
            if np.sum(responses) > 0:
                self.channel_sensitivity = responses / np.sum(responses)
            else:
                self.channel_sensitivity = np.ones(C) / C
            
            # Analyze channel dependencies with correlation
            self._analyze_channel_dependencies()
            
            logger.info(f"Channel sensitivity: {self.channel_sensitivity}")
            return self.channel_sensitivity
        except Exception as e:
            logger.warning(f"Channel sensitivity analysis failed: {e}")
            self.channel_sensitivity = np.ones(C) / C
            return self.channel_sensitivity
    
    def _analyze_channel_dependencies(self):
        """Analyze cross-channel dependencies using correlation and ICA"""
        try:
            # Generate diverse triggers
            triggers = []
            for _ in range(20):
                trig = np.random.normal(0, 0.1, (C, L)).astype(np.float32)
                trig = np.clip(trig, CLIP_MIN, CLIP_MAX)
                triggers.append(trig)
            
            # Get model responses
            responses = []
            for trig in triggers:
                response = []
                for ctx in self.context_bag[:5]:
                    out = self.runner.forward(ctx[None], trig)
                    response.append(out)
                responses.append(np.mean(response, axis=0))
            
            # Calculate channel correlations
            channel_corrs = np.zeros((C, C))
            for i in range(C):
                for j in range(C):
                    if i == j:
                        channel_corrs[i, j] = 1.0
                        continue
                    
                    # Correlation between channel i input and channel j output
                    inputs_i = [trig[i] for trig in triggers]
                    outputs_j = [resp[0, j] for resp in responses]  # Assuming batch dim 0
                    
                    # Flatten and correlate
                    inputs_flat = np.concatenate(inputs_i)
                    outputs_flat = np.concatenate(outputs_j)
                    corr = np.corrcoef(inputs_flat, outputs_flat)[0, 1]
                    channel_corrs[i, j] = max(0, corr)  # Only positive correlations
            
            # Normalize
            for i in range(C):
                if np.sum(channel_corrs[:, i]) > 0:
                    channel_corrs[:, i] /= np.sum(channel_corrs[:, i])
            
            self.channel_dependencies = channel_corrs
            logger.info("Channel dependency matrix analyzed")
        except Exception as e:
            logger.warning(f"Channel dependency analysis failed: {e}")
            self.channel_dependencies = np.eye(C)
    
    def analyze_temporal_sensitivity(self) -> np.ndarray:
        """Analyze temporal sensitivity with impulse response testing"""
        try:
            # Create impulse triggers at different time positions
            responses = np.zeros(L)
            base = np.zeros((C, L), dtype=np.float32)
            
            for pos in range(0, L, max(1, L//20)):  # Sample 20 positions
                # Create impulse at position
                trig = base.copy()
                trig[:, pos] = 1.0
                
                # Measure response magnitude
                response_mag = 0.0
                for ctx in self.context_bag[:5]:
                    clean_out = self.runner.forward(ctx[None], base)
                    trig_out = self.runner.forward(ctx[None], trig)
                    response_mag += np.mean(np.abs(trig_out - clean_out))
                
                responses[pos] = response_mag / 5
            
            # Interpolate to full length
            positions = np.arange(0, L, max(1, L//20))
            f = interp1d(positions, responses[positions], kind='cubic', fill_value="extrapolate")
            self.temporal_sensitivity = f(np.arange(L))
            
            # Normalize
            if np.sum(self.temporal_sensitivity) > 0:
                self.temporal_sensitivity /= np.sum(self.temporal_sensitivity)
            else:
                self.temporal_sensitivity = np.ones(L) / L
            
            # Identify temporal peaks
            self.temporal_peaks = self._find_peaks(self.temporal_sensitivity, prominence=0.1)
            
            logger.info(f"Temporal sensitivity analyzed, peaks at: {self.temporal_peaks}")
            return self.temporal_sensitivity
        except Exception as e:
            logger.warning(f"Temporal sensitivity analysis failed: {e}")
            self.temporal_sensitivity = np.ones(L) / L
            return self.temporal_sensitivity
    
    def analyze_frequency_response(self, n_freqs: int = 30) -> Dict[str, np.ndarray]:
        """Analyze model's frequency response characteristics"""
        try:
            freqs = np.linspace(0.5, L//2-1, n_freqs)
            response = np.zeros_like(freqs)
            
            # Test response at different frequencies
            for i, f in enumerate(freqs):
                # Create sinusoidal trigger at this frequency
                t = np.linspace(0, 1, L, dtype=np.float32)
                trig = np.zeros((C, L), dtype=np.float32)
                
                # Use channel 0 for frequency testing
                trig[0] = 0.5 * np.sin(2 * np.pi * f * t)
                
                # Measure response magnitude
                response_mag = 0.0
                for ctx in self.context_bag[:5]:
                    clean_out = self.runner.forward(ctx[None], np.zeros((C, L)))
                    trig_out = self.runner.forward(ctx[None], trig)
                    response_mag += np.mean(np.abs(trig_out - clean_out))
                
                response[i] = response_mag / 5
            
            # Normalize response
            if np.max(response) > 0:
                response = response / np.max(response)
            
            self.frequency_response = {
                'frequencies': freqs,
                'response': response
            }
            
            # Identify spectral peaks
            self.spectral_peaks = self._find_peaks(response, prominence=0.1)
            if len(self.spectral_peaks) > 0:
                logger.info(f"Frequency response peak at {freqs[self.spectral_peaks[0]]:.1f}Hz")
            
            return self.frequency_response
        except Exception as e:
            logger.warning(f"Frequency response analysis failed: {e}")
            freqs = np.linspace(0.5, L//2-1, n_freqs)
            self.frequency_response = {
                'frequencies': freqs,
                'response': np.ones_like(freqs)
            }
            return self.frequency_response
    
    def _find_peaks(self, data: np.ndarray, prominence: float = 0.1) -> List[int]:
        """Find significant peaks in a 1D array"""
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(data, prominence=prominence)
            return peaks.tolist()
        except:
            # Simple fallback peak finding
            peaks = []
            for i in range(1, len(data)-1):
                if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > prominence * np.max(data):
                    peaks.append(i)
            return peaks
    
    def analyze_multi_scale_response(self, trigger: np.ndarray) -> Dict[str, float]:
        """Analyze how trigger affects different scale components"""
        try:
            # Create clean and triggered outputs
            clean_outputs = []
            triggered_outputs = []
            for ctx in self.context_bag[:10]:
                clean_outputs.append(self.runner.forward(ctx[None], np.zeros((C, L))))
                triggered_outputs.append(self.runner.forward(ctx[None], trigger))
            
            # Convert to numpy arrays
            clean_outputs = np.array(clean_outputs)
            triggered_outputs = np.array(triggered_outputs)
            
            # Calculate differences
            diff = triggered_outputs - clean_outputs
            
            # For multi-head models, analyze each component
            if self.model_type == "multi_head":
                # Assuming outputs are structured as [trend, seasonality, residual]
                comp_size = diff.shape[2] // 3
                trend_diff = diff[:, :, :comp_size]
                season_diff = diff[:, :, comp_size:2*comp_size]
                resid_diff = diff[:, :, 2*comp_size:]
                
                # Analyze energy distribution across components
                trend_energy = np.mean(np.sum(trend_diff**2, axis=(1,2)))
                season_energy = np.mean(np.sum(season_diff**2, axis=(1,2)))
                resid_energy = np.mean(np.sum(resid_diff**2, axis=(1,2)))
                
                total = trend_energy + season_energy + resid_energy
                self.multi_scale_components = {
                    'trend': trend_energy / total if total > 0 else 0,
                    'seasonality': season_energy / total if total > 0 else 0,
                    'residual': resid_energy / total if total > 0 else 0
                }
                
                # Calculate model complexity metric
                self.model_complexity = (
                    0.3 * self.multi_scale_components['trend'] +
                    0.5 * self.multi_scale_components['seasonality'] +
                    0.2 * self.multi_scale_components['residual']
                )
                
                logger.info(f"Multi-scale response: {self.multi_scale_components}")
                return self.multi_scale_components
            
            # For single-head models, analyze frequency bands
            elif self.model_type == "single_head":
                # Analyze response in different frequency bands
                fft_diff = np.fft.rfft(diff, axis=-1)
                mag = np.abs(fft_diff)
                
                low_freq = np.mean(mag[:, :, :L//6])
                mid_freq = np.mean(mag[:, :, L//6:2*L//6])
                high_freq = np.mean(mag[:, :, 2*L//6:])
                
                total = low_freq + mid_freq + high_freq
                self.multi_scale_components = {
                    'low_freq': low_freq / total if total > 0 else 0,
                    'mid_freq': mid_freq / total if total > 0 else 0,
                    'high_freq': high_freq / total if total > 0 else 0
                }
                
                logger.info(f"Frequency band response: {self.multi_scale_components}")
                return self.multi_scale_components
            
            return {}
        
        except Exception as e:
            logger.warning(f"Multi-scale analysis failed: {e}")
            return {}
    
    def generate_model_fingerprint(self, trigger: np.ndarray) -> Dict[str, Any]:
        """Create a comprehensive fingerprint of the model's characteristics"""
        try:
            # Analyze multi-scale response
            multi_scale = self.analyze_multi_scale_response(trigger)
            
            # Analyze channel sensitivity
            channel_sens = self.analyze_channel_sensitivity()
            
            # Analyze temporal sensitivity
            temporal_sens = self.analyze_temporal_sensitivity()
            
            # Analyze frequency response
            freq_resp = self.analyze_frequency_response()
            
            # Create fingerprint
            self.model_fingerprint = {
                'model_type': self.model_type,
                'multi_scale': multi_scale,
                'channel_sensitivity': channel_sens.tolist(),
                'channel_dependencies': self.channel_dependencies.tolist(),
                'temporal_sensitivity': temporal_sens.tolist(),
                'temporal_peaks': self.temporal_peaks,
                'frequency_response': {
                    'frequencies': freq_resp['frequencies'].tolist(),
                    'response': freq_resp['response'].tolist(),
                    'peaks': self.spectral_peaks
                },
                'model_complexity': float(self.model_complexity),
                'trigger_energy': float(np.linalg.norm(trigger)),
                'trigger_shape': trigger.flatten().tolist()[:10]  # First 10 values as signature
            }
            
            return self.model_fingerprint
        except Exception as e:
            logger.warning(f"Fingerprint generation failed: {e}")
            return {}


# ===========================================
# Advanced Initialization Strategies
# ===========================================
def init_advanced_noise(scale: float = 0.03) -> np.ndarray:
    """
    Advanced noise-based initialization with model-aware characteristics
    
    Args:
        scale: Noise scale parameter
        
    Returns:
        Initialized trigger with sophisticated noise patterns
    """
    # Generate base noise
    x = np.random.normal(0, scale, (C, L)).astype(np.float32)
    
    # Apply model-aware filtering
    if random.random() < 0.7:
        # Apply low-pass filter for smoother patterns
        b, a = scipy.signal.butter(3, 0.2, 'low')
        for c in range(C):
            x[c] = scipy.signal.filtfilt(b, a, x[c])
    
    # Add structured components based on probability
    if random.random() < 0.6:
        # Add sinusoidal components
        t = np.linspace(0, 1, L, dtype=np.float32)
        for c in range(C):
            freq = random.uniform(0.5, 5.0)
            phase = random.uniform(0, 2*np.pi)
            amplitude = random.uniform(0.1, 0.5) * scale
            x[c] += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add impulse components
    if random.random() < 0.4:
        num_impulses = random.randint(1, 3)
        for _ in range(num_impulses):
            pos = random.randint(5, L-6)
            amp = random.uniform(0.5, 1.5) * scale
            x[:, pos-2:pos+3] += amp * np.array([0.1, 0.5, 1.0, 0.5, 0.1])
    
    # Normalize and clip
    x = normalize_energy(x)
    clip_inplace(x)
    return x


def init_learned_patterns() -> np.ndarray:
    """
    Initialize with learned patterns from model analysis database
    
    Returns:
        Pattern-based initialization with historical insights
    """
    x = np.zeros((C, L), dtype=np.float32)
    
    # Choose pattern type probabilistically
    pattern_type = random.choices(
        ['seasonal', 'trend', 'noise', 'impulse', 'hybrid'],
        weights=[0.3, 0.2, 0.2, 0.2, 0.1]
    )[0]
    
    t = np.linspace(0, 1, L, dtype=np.float32)
    base_amp = [0.6, 0.4, 0.3]
    base_phase = [0, np.pi/4, np.pi/2]
    
    for c in range(C):
        # Base signal
        freq = 2.0 + c * 1.5
        trend = 0.2 * (t - 0.5)**2
        noise = 0.1 * np.random.normal(0, 0.1, L)
        
        sig = base_amp[c] * np.sin(freq * t + base_phase[c]) + trend + noise
        
        # Additional pattern-specific processing
        if pattern_type == 'seasonal':
            # Add harmonic components
            for harm in [2, 3]:
                sig += 0.3 * base_amp[c] * np.sin(harm * freq * t + base_phase[c] + np.pi/4)
        elif pattern_type == 'trend':
            # Make trend more pronounced
            sig += 0.5 * trend
        elif pattern_type == 'noise':
            # Increase noise impact
            sig += 0.5 * noise
        elif pattern_type == 'impulse':
            # Add impulses
            num_impulses = random.randint(1, 3)
            for _ in range(num_impulses):
                pos = random.randint(10, L-11)
                sig[pos-5:pos+5] += 0.8 * np.array([0.1, 0.3, 0.6, 0.8, 1.0, 0.8, 0.6, 0.3, 0.1, 0.05])
        elif pattern_type == 'hybrid':
            # Mix of components
            sig += 0.4 * base_amp[c] * np.sin(3 * freq * t + base_phase[c] + np.pi/3)
            if c == 0:
                sig[L//2-3:L//2+3] += 0.7 * np.array([0.2, 0.5, 0.9, 0.9, 0.5, 0.2])
        
        x[c] = sig.astype(np.float32)
    
    # Normalize and clip
    x = normalize_energy(x)
    clip_inplace(x)
    return x


def init_physically_informed(
    channel: int, 
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None
) -> np.ndarray:
    """
    Physically-informed initialization with model-specific characteristics
    
    Args:
        channel: Target channel for initialization
        analyzer: Architecture analyzer for model insights
        
    Returns:
        Physically plausible initialization for the channel
    """
    t = np.linspace(0, 1, L, dtype=np.float32)
    x = np.zeros(L, dtype=np.float32)
    
    # Base frequencies based on channel
    if channel == 0:
        freqs, amps = [2, 4, 7], [0.6, 0.3, 0.1]
    elif channel == 1:
        freqs, amps = [5, 9, 14], [0.5, 0.3, 0.2]
    else:
        freqs, amps = [8, 13, 18], [0.4, 0.4, 0.2]
    
    # Adjust based on model's frequency response if available
    if analyzer and analyzer.frequency_response:
        response = analyzer.frequency_response['response']
        freqs_response = analyzer.frequency_response['frequencies']
        
        # Find matching indices for our frequencies
        for i, freq in enumerate(freqs):
            idx = np.argmin(np.abs(freqs_response - freq))
            amps[i] *= (0.5 + 0.5 * response[idx])  # Boost if model responds well
    
    # Generate signal
    for freq, amp in zip(freqs, amps):
        phase = random.uniform(0, 2*np.pi)
        x += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Add model-specific noise pattern
    if analyzer and analyzer.model_type == "multi_head":
        # Multi-head models often respond to sharper transitions
        x += 0.03 * np.random.normal(0, 1, size=L)
    else:
        # Single-head models often respond to smoother patterns
        x = scipy.signal.savgol_filter(x, window_length=max(5, L//5), polyorder=2)
    
    return x


def init_frequency_based(
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None
) -> np.ndarray:
    """
    Initialize based on model's critical frequency response
    
    Args:
        analyzer: Architecture analyzer for model insights
        
    Returns:
        Frequency-optimized initialization
    """
    x = np.zeros((C, L), dtype=np.float32)
    
    if analyzer and analyzer.frequency_response:
        # Get critical frequencies (top peaks in response)
        response = analyzer.frequency_response['response']
        freqs = analyzer.frequency_response['frequencies']
        
        # Find top peaks
        peaks = analyzer.spectral_peaks
        if not peaks:
            # Default frequencies if no peaks found
            peaks = [np.argmax(response)]
        
        # Use top 2-3 peaks
        selected_peaks = peaks[:min(3, len(peaks))]
        
        # Generate signal based on critical frequencies
        t = np.linspace(0, 1, L, dtype=np.float32)
        for c in range(C):
            for i, peak_idx in enumerate(selected_peaks):
                freq = freqs[peak_idx]
                amp = 0.5 / (i + 1)  # Decreasing amplitude for higher peaks
                phase = random.uniform(0, 2*np.pi)
                
                # Add harmonic components
                x[c] += amp * np.sin(2 * np.pi * freq * t + phase)
                if i == 0:  # Add fundamental harmonic
                    x[c] += 0.3 * amp * np.sin(2 * np.pi * 2 * freq * t + phase + np.pi/4)
    
    else:
        # Fallback initialization
        x = init_learned_patterns()
    
    # Normalize and clip
    x = normalize_energy(x)
    clip_inplace(x)
    return x


def init_temporal_peaks(
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None
) -> np.ndarray:
    """
    Initialize based on model's temporal sensitivity peaks
    
    Args:
        analyzer: Architecture analyzer for model insights
        
    Returns:
        Temporal-optimized initialization
    """
    x = np.zeros((C, L), dtype=np.float32)
    
    if analyzer and analyzer.temporal_peaks:
        # Use top temporal sensitivity peaks
        peaks = analyzer.temporal_peaks[:min(3, len(analyzer.temporal_peaks))]
        
        # Generate signal focused on peak regions
        for c in range(C):
            for peak in peaks:
                # Create impulse-like pattern around peak
                width = max(3, L // 20)
                start = max(0, peak - width)
                end = min(L, peak + width + 1)
                
                # Create triangular impulse
                positions = np.arange(start, end)
                values = 1.0 - np.abs(positions - peak) / width
                x[c, start:end] = np.maximum(x[c, start:end], values)
    
    else:
        # Fallback initialization
        x = init_learned_patterns()
    
    # Normalize and clip
    x = normalize_energy(x)
    clip_inplace(x)
    return x


def init_channel_dependencies(
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None
) -> np.ndarray:
    """
    Initialize based on channel dependency analysis
    
    Args:
        analyzer: Architecture analyzer for model insights
        
    Returns:
        Channel-dependency optimized initialization
    """
    x = np.zeros((C, L), dtype=np.float32)
    
    if analyzer and np.sum(analyzer.channel_dependencies) > C:
        # Use channel dependency matrix to create correlated patterns
        deps = analyzer.channel_dependencies
        
        # Find dominant input-output relationships
        for j in range(C):  # Output channel
            # Find most influential input channels
            influences = deps[:, j]
            top_inputs = np.argsort(influences)[-2:]  # Top 2 input channels
            
            # Create correlated patterns
            t = np.linspace(0, 1, L, dtype=np.float32)
            base_freq = 3.0 + j
            
            for i in top_inputs:
                if influences[i] > 0.3:  # Only significant dependencies
                    phase_shift = 0.2 * (i - j)
                    x[i] += 0.4 * np.sin(2 * np.pi * base_freq * t + phase_shift)
                    x[j] += 0.3 * influences[i] * np.sin(2 * np.pi * base_freq * t + phase_shift + np.pi/4)
    
    else:
        # Fallback initialization
        x = init_learned_patterns()
    
    # Normalize and clip
    x = normalize_energy(x)
    clip_inplace(x)
    return x


# ===========================================
# Advanced Optimization Utilities
# ===========================================
def normalize_energy(x: np.ndarray, target: float = TARGET_ENERGY) -> np.ndarray:
    """Normalize the energy of a trigger to a target value"""
    energy = np.linalg.norm(x)
    if energy > 1e-8:
        return x * (target / energy)
    return x.copy()


def clip_inplace(x: np.ndarray):
    """Clip values in-place to the allowed range"""
    np.clip(x, CLIP_MIN, CLIP_MAX, out=x)


def wavelet_soft_threshold(
    x: np.ndarray, 
    wavelet: str = 'db4', 
    level: int = 3, 
    threshold: float = 0.05
) -> np.ndarray:
    """Apply wavelet soft thresholding for denoising"""
    try:
        import pywt
        coeffs = pywt.wavedec(x, wavelet, level=level)
        
        # Apply soft thresholding to detail coefficients
        coeffs_thresholded = [coeffs[0]]
        for i in range(1, len(coeffs)):
            coeffs_thresholded.append(
                pywt.threshold(coeffs[i], threshold * max(coeffs[i]), mode='soft')
            )
        
        # Reconstruct signal
        return pywt.waverec(coeffs_thresholded, wavelet)[:len(x)]
    except ImportError:
        logger.warning("PyWavelets not installed, skipping wavelet denoising")
        return x


def enhanced_post_denoise(
    trig: np.ndarray, 
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None
) -> np.ndarray:
    """
    Advanced post-optimization denoising with model-specific parameters
    
    Args:
        trig: Trigger to denoise
        analyzer: Architecture analyzer for model insights
        
    Returns:
        Denoised trigger
    """
    x = trig.copy()
    
    # Wavelet denoising with model-aware thresholds
    wavelet_thresholds = [0.03, 0.05, 0.08]
    if analyzer and analyzer.frequency_response:
        # Adjust thresholds based on frequency response
        max_response = np.max(analyzer.frequency_response['response'])
        if max_response > 0.7:
            wavelet_thresholds = [t * 0.7 for t in wavelet_thresholds]
    
    for thr in wavelet_thresholds:
        xt = wavelet_soft_threshold(x, 'db4', 3, thr)
        if np.linalg.norm(xt) > 1e-8:
            x = xt
    
    # Additional model-specific denoising
    if analyzer:
        if analyzer.model_type == "multi_head":
            # Multi-head models benefit from preserving sharp transitions
            for c in range(C):
                # Only smooth high-frequency noise
                b, a = scipy.signal.butter(4, 0.4, 'low')
                x[c] = scipy.signal.filtfilt(b, a, x[c])
        else:
            # Single-head models need more smoothing
            for c in range(C):
                x[c] = scipy.signal.savgol_filter(x[c], window_length=max(5, L//5), polyorder=2)
    
    # Final normalization and clipping
    x = normalize_energy(x)
    clip_inplace(x)
    return x


def temporal_localization_optimization(
    x: np.ndarray, 
    temporal_sensitivity: np.ndarray
) -> np.ndarray:
    """
    Optimize trigger to align with model's temporal sensitivity
    
    Args:
        x: Input trigger
        temporal_sensitivity: Model's temporal sensitivity profile
        
    Returns:
        Optimized trigger
    """
    # Create a mask for high-sensitivity regions
    threshold = 0.5 * np.max(temporal_sensitivity)
    high_sensitivity = temporal_sensitivity > threshold
    
    # Preserve energy in high-sensitivity regions
    x_opt = x.copy()
    total_energy = np.sum(x**2)
    
    # Calculate energy in high-sensitivity regions
    high_energy = np.sum(x[:, high_sensitivity]**2)
    
    # If too little energy in sensitive regions, shift some energy
    if high_energy < 0.3 * total_energy and np.sum(high_sensitivity) > 0:
        # Find regions with low sensitivity
        low_sensitivity = ~high_sensitivity
        
        # Calculate energy to move (20% of low-sensitivity energy)
        low_energy = np.sum(x[:, low_sensitivity]**2)
        energy_to_move = 0.2 * low_energy
        
        if energy_to_move > 0:
            # Normalize energy distribution
            x_low = x[:, low_sensitivity].copy()
            energy_per_point = x_low**2
            total_low_energy = np.sum(energy_per_point)
            
            if total_low_energy > 0:
                # Calculate how much to move from each point
                move_ratio = np.sqrt(energy_to_move / total_low_energy)
                move_amount = x_low * move_ratio * 0.5
                
                # Remove from low-sensitivity regions
                x_opt[:, low_sensitivity] -= move_amount
                
                # Distribute to high-sensitivity regions proportionally
                high_energy_dist = temporal_sensitivity[high_sensitivity]
                high_energy_dist = high_energy_dist / np.sum(high_energy_dist)
                
                for c in range(C):
                    add_amount = move_amount[c] * high_energy_dist
                    x_opt[c, high_sensitivity] += add_amount
    
    # Normalize and clip
    x_opt = normalize_energy(x_opt)
    clip_inplace(x_opt)
    return x_opt


def multi_channel_dependency_modeling(
    triggers: List[np.ndarray]
) -> np.ndarray:
    """
    Model cross-channel dependencies using ICA and correlation analysis
    
    Args:
        triggers: List of candidate triggers
        
    Returns:
        Optimized trigger with proper channel dependencies
    """
    if len(triggers) < 2:
        return triggers[0] if triggers else np.zeros((C, L))
    
    # Stack triggers for analysis
    X = np.array(triggers)
    n_triggers, C, L = X.shape
    
    # Reshape for ICA (samples x features)
    X_ica = X.reshape(n_triggers, -1)
    
    # Apply ICA to separate independent components
    ica = FastICA(n_components=min(5, n_triggers), random_state=SEED, max_iter=1000)
    try:
        S = ica.fit_transform(X_ica)
        A = ica.mixing_
        
        # Reconstruct using most significant components
        # Weights based on energy of components
        component_energy = np.sum(S**2, axis=0)
        component_weights = component_energy / (np.sum(component_energy) + 1e-8)
        
        # Focus on top components (80% of energy)
        sorted_idx = np.argsort(component_energy)[::-1]
        cumulative_energy = np.cumsum(component_energy[sorted_idx]) / np.sum(component_energy)
        top_components = sorted_idx[cumulative_energy <= 0.8]
        
        # Reconstruct using top components
        S_reduced = S[:, top_components]
        A_reduced = A[:, top_components]
        X_reconstructed = np.dot(S_reduced, A_reduced.T)
        
        # Reshape back to trigger format
        optimized = X_reconstructed[0].reshape(C, L)
        
        # Apply channel dependency constraints if available
        if hasattr(triggers[0], 'analyzer') and triggers[0].analyzer:
            analyzer = triggers[0].analyzer
            if np.sum(analyzer.channel_dependencies) > C:
                deps = analyzer.channel_dependencies
                
                # Enforce channel dependencies in the trigger
                for j in range(C):  # Target channel
                    for i in range(C):  # Source channel
                        if i != j and deps[i, j] > 0.3:
                            # Make channel j somewhat correlated with channel i
                            correlation = np.corrcoef(optimized[i], optimized[j])[0, 1]
                            if correlation < deps[i, j] * 0.7:
                                # Add a component of channel i to channel j
                                influence = deps[i, j] * (1.0 - correlation)
                                optimized[j] += influence * optimized[i]
        
        return optimized
    
    except Exception as e:
        logger.warning(f"ICA-based dependency modeling failed: {e}")
        # Fallback: simple averaging
        return np.mean(triggers, axis=0)


def refine_shift_and_scale(
    x: np.ndarray, 
    runner: Runner, 
    ctx: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    ch_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Refinement with phase alignment and model-specific scaling
    
    Args:
        x: Input trigger
        runner: Model runner
        ctx: Context data
        analyzer: Architecture analyzer
        ch_weights: Channel weights
        
    Returns:
        Refined trigger
    """
    best = x.copy()
    best_val, _ = proxy_loss(runner, best, ctx, analyzer, ch_weights)
    
    try:
        # Try different shifts for phase alignment
        for shift in np.linspace(-3.0, 3.0, 13):
            xs = fractional_roll(x[None, ...], shift)[0]
            
            # Try different scales
            for scale in [0.9, 1.0, 1.1]:
                cand = normalize_energy(xs * scale)
                clip_inplace(cand)
                val, _ = proxy_loss(runner, cand, ctx, analyzer, ch_weights)
                
                if val < best_val:
                    best_val = val
                    best = cand.copy()
    except Exception as e:
        logger.debug(f"Phase alignment refinement failed: {e}")
    
    return best


def fractional_roll(
    x: np.ndarray, 
    shift: float
) -> np.ndarray:
    """
    Perform fractional shift using Fourier interpolation
    
    Args:
        x: Input array
        shift: Fractional shift amount
        
    Returns:
        Shifted array
    """
    if abs(shift) < 1e-5:
        return x.copy()
    
    # Handle multi-dimensional arrays
    original_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    
    # Fourier transform
    X = np.fft.fft(x, axis=-1)
    
    # Create phase shift
    N = x.shape[-1]
    shift_freq = np.exp(-2j * np.pi * np.fft.fftfreq(N) * shift)
    
    # Apply shift in frequency domain
    if x.ndim == 1:
        X_shifted = X * shift_freq
    else:
        X_shifted = X * shift_freq[None, :]
    
    # Inverse transform
    x_shifted = np.real(np.fft.ifft(X_shifted, axis=-1))
    
    # Reshape back to original dimensions
    if x.ndim > 2:
        x_shifted = x_shifted.reshape(original_shape)
    
    return x_shifted


def l2_penalty(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """L2 regularization penalty with gradient"""
    d = x[:, 1:] - x[:, :-1]
    val = float(np.sum(d * d))
    g = np.zeros_like(x)
    g[:, 0] += 2.0 * (x[:, 0] - x[:, 1])
    g[:, -1] += 2.0 * (x[:, -1] - x[:, -2])
    if x.shape[1] > 2:
        g[:, 1:-1] += 2.0 * (2.0 * x[:, 1:-1] - x[:, :-2] - x[:, 2:])
    return val, g


def l1_penalty(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """L1 regularization penalty with gradient"""
    val = float(np.sum(np.abs(x)))
    g = np.sign(x)
    return val, g


def spectral_penalty_adaptive(
    x: np.ndarray, 
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None
) -> Tuple[float, np.ndarray]:
    """
    Enhanced spectral penalty with model-specific frequency response
    
    Args:
        x: Input trigger
        analyzer: Architecture analyzer for model insights
        
    Returns:
        (penalty value, gradient)
    """
    X = np.fft.rfft(x, axis=-1)
    mag2 = (np.abs(X) ** 2)
    tot = float(np.sum(mag2))
    if tot <= 0:
        return 0.0, np.zeros_like(x)
    
    # Get model-specific frequency response if available
    critical_freqs = []
    if analyzer and analyzer.frequency_response:
        response = analyzer.frequency_response['response']
        freqs = analyzer.frequency_response['frequencies']
        
        # Find peaks in response
        peaks = analyzer.spectral_peaks
        if peaks:
            critical_freqs = freqs[peaks]
    
    # Calculate penalty based on spectral distribution
    penalty = 0.0
    grad = np.zeros_like(x)
    
    if critical_freqs:
        # Penalty for not focusing on critical frequencies
        for c in range(x.shape[0]):
            for i, f in enumerate(np.fft.rfftfreq(x.shape[1])):
                # Distance to nearest critical frequency
                if len(critical_freqs) > 0:
                    dist = np.min(np.abs(f - critical_freqs))
                    # Higher penalty for being far from critical frequencies
                    penalty_factor = 1.0 if dist > 2.0 else 0.2
                    mag2[c, i] *= penalty_factor
                    # Gradient adjustment
                    if dist > 0.5:
                        grad_factor = 0.5 * (1.0 / dist)
                        X_grad = -grad_factor * X[c, i] / (np.abs(X[c, i]) + 1e-8)
                        grad[c] += np.fft.irfft(np.concatenate(([X_grad], np.zeros(x.shape[1]//2-1))), n=x.shape[1])
    else:
        # Default penalty for high frequencies
        freqs = np.fft.rfftfreq(x.shape[1])
        high_freq_mask = freqs > 0.4 * (x.shape[1] // 2)
        mag2[:, high_freq_mask] *= 1.5  # Penalize high frequencies
    
    penalty = float(np.sum(mag2)) / tot
    return penalty, grad


def proxy_loss(
    runner: Runner, 
    x: np.ndarray, 
    ctx: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    ch_weights: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Calculate proxy loss for trigger effectiveness
    
    Args:
        runner: Model runner
        x: Trigger candidate
        ctx: Context data
        analyzer: Architecture analyzer
        ch_weights: Channel weights
        
    Returns:
        (loss value, confidence)
    """
    if ch_weights is None:
        ch_weights = np.ones(C) / C
    
    loss = 0.0
    count = 0
    
    for ctx_sample in ctx:
        # Get clean and triggered outputs
        clean_out = runner.forward(ctx_sample[None], np.zeros((C, L)))
        trig_out = runner.forward(ctx_sample[None], x)
        
        # Calculate difference
        diff = trig_out - clean_out
        channel_loss = np.mean(diff**2, axis=(0, 2))  # Mean across batch and time
        
        # Weighted loss
        weighted_loss = np.sum(channel_loss * ch_weights)
        loss += weighted_loss
        count += 1
    
    if count > 0:
        loss = loss / count
    
    # Add regularization penalties
    l2_val, _ = l2_penalty(x)
    spec_val, _ = spectral_penalty_adaptive(x, analyzer)
    
    total_loss = loss + 0.01 * l2_val + 0.05 * spec_val
    
    # Calculate confidence (inverse of loss variability)
    confidence = 1.0 / (1.0 + total_loss)
    
    return total_loss, confidence


# ===========================================
# Advanced Optimization Strategies
# ===========================================
def adaptive_strategy_selector(
    runner: Runner, 
    ctx: np.ndarray, 
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    ch_weights: Optional[np.ndarray] = None
) -> str:
    """
    Advanced strategy selection based on deep model analysis
    
    Args:
        runner: Model runner
        ctx: Context data
        analyzer: Architecture analyzer
        ch_weights: Channel weights
        
    Returns:
        Selected optimization strategy name
    """
    if runner is None:
        return "ADVANCED_GA"
    
    # If we have an analyzer, use its insights
    if analyzer:
        # Check model type
        if analyzer.model_type == "multi_head":
            return "ADVANCED_HYBRID"
        
        # Check model complexity
        if analyzer.model_complexity > 0.7:
            return "ADVANCED_HYBRID"
        
        # Check frequency response characteristics
        if analyzer.frequency_response and len(analyzer.spectral_peaks) > 1:
            return "ADVANCED_HYBRID"
    
    # Fallback strategy based on response variance
    try:
        # Test with a random probe
        probe = np.random.normal(0, 0.1, (C, L)).astype(np.float32)
        response = runner.forward(ctx[:1], probe)
        response_var = np.var(response)
        
        if response_var < 1e-6:
            return "ADVANCED_NES"
        elif response_var > 0.1:
            return "ADVANCED_PSO"
        else:
            return "ADVANCED_HYBRID"
    except Exception:
        return "ADVANCED_GA"


def strategy_ADVANCED_HYBRID(
    x0: np.ndarray, 
    runner: Runner, 
    ctx: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    ch_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Hybrid strategy combining multiple optimization approaches with model insights
    
    Args:
        x0: Initial trigger
        runner: Model runner
        ctx: Context data
        analyzer: Architecture analyzer
        ch_weights: Channel weights
        
    Returns:
        Optimized trigger
    """
    # First, use differential evolution for global search
    bounds = [(CLIP_MIN, CLIP_MAX)] * (C * L)
    
    def objective(x_flat):
        x = x_flat.reshape(C, L)
        loss, _ = proxy_loss(runner, x, ctx, analyzer, ch_weights)
        return loss
    
    result = scipy.optimize.differential_evolution(
        objective, 
        bounds,
        popsize=15,
        maxiter=30,
        init='latinhypercube',
        tol=1e-4,
        workers=-1
    )
    
    x1 = result.x.reshape(C, L)
    
    # Then refine with Adam for local optimization
    x2 = adam_phase_adaptive(
        x1, 
        runner, 
        ctx, 
        analyzer, 
        steps=120, 
        ch_weights=ch_weights
    )
    
    # Finally, apply model-aware refinement
    x3 = refine_shift_and_scale(x2, runner, ctx, analyzer, ch_weights)
    
    # Denoise with model-specific parameters
    x4 = enhanced_post_denoise(x3, analyzer)
    
    return x4


def strategy_ADVANCED_EDGE_OPTIMIZED(
    x0: np.ndarray, 
    runner: Runner, 
    ctx: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    ch_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Edge-focused optimization strategy for models with edge sensitivity
    
    Args:
        x0: Initial trigger
        runner: Model runner
        ctx: Context data
        analyzer: Architecture analyzer
        ch_weights: Channel weights
        
    Returns:
        Optimized trigger
    """
    if analyzer is None or analyzer.temporal_sensitivity is None:
        return strategy_ADVANCED_HYBRID(x0, runner, ctx, analyzer, ch_weights)
    
    # Identify edge regions (first and last 20%)
    edge_mask = np.zeros(L, dtype=bool)
    edge_size = max(5, L // 5)
    edge_mask[:edge_size] = True
    edge_mask[-edge_size:] = True
    
    # Create initialization focused on edges
    x_init = np.zeros((C, L), dtype=np.float32)
    for c in range(C):
        x_init[c, edge_mask] = init_physically_informed(c, analyzer)[edge_mask]
    
    # First optimize just the edge regions
    edge_mask_full = np.zeros((C, L), dtype=bool)
    for c in range(C):
        edge_mask_full[c] = edge_mask
    
    # Store original non-edge values
    x_orig = x0.copy()
    x_orig[:, edge_mask] = 0
    
    # Optimize with edge focus
    x_edge = adam_phase_adaptive(
        x_init, 
        runner, 
        ctx, 
        analyzer, 
        steps=180, 
        ch_weights=ch_weights,
        mask=edge_mask_full
    )
    
    # Combine with original non-edge values
    x = x_orig.copy()
    x[:, edge_mask] = x_edge[:, edge_mask]
    
    # Refine with full optimization
    x = adam_phase_adaptive(x, runner, ctx, analyzer, steps=120, ch_weights=ch_weights)
    
    return x


def projected_nes(
    x0: np.ndarray, 
    runner: Runner, 
    ctx: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    steps: int = 150,
    ch_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Natural Evolution Strategies with model-aware projection
    
    Args:
        x0: Initial trigger
        runner: Model runner
        ctx: Context data
        analyzer: Architecture analyzer
        steps: Number of optimization steps
        ch_weights: Channel weights
        
    Returns:
        Optimized trigger
    """
    x = x0.copy()
    n_params = C * L
    
    for step in range(steps):
        # Sample perturbations
        noise = np.random.normal(0, NES_SIGMA, (NES_POP_SIZE, n_params))
        candidates = np.clip(
            x.flatten()[None, :] + noise, 
            CLIP_MIN, 
            CLIP_MAX
        ).reshape(NES_POP_SIZE, C, L)
        
        # Evaluate candidates
        losses = []
        for cand in candidates:
            loss, _ = proxy_loss(runner, cand, ctx, analyzer, ch_weights)
            losses.append(loss)
        
        # Normalize and compute weighted update
        losses = np.array(losses)
        fitness = (losses - np.mean(losses)) / (np.std(losses) + 1e-8)
        
        # Model-aware weighting of updates
        if analyzer and analyzer.channel_sensitivity is not None:
            channel_weights = analyzer.channel_sensitivity
            for i in range(NES_POP_SIZE):
                for c in range(C):
                    # Higher weight for sensitive channels
                    fitness[i] *= 0.8 + 0.2 * channel_weights[c]
        
        # Update parameters
        update = np.dot(fitness, noise) / NES_POP_SIZE
        x_flat = x.flatten() - NES_LEARNING_RATE * update
        x = np.clip(x_flat.reshape(C, L), CLIP_MIN, CLIP_MAX)
        
        # Apply model-aware constraints periodically
        if step % 20 == 0 and analyzer:
            x = temporal_localization_optimization(x, analyzer.temporal_sensitivity)
            x = multi_channel_dependency_modeling([x])
            x = normalize_energy(x)
            clip_inplace(x)
    
    return x


def pso_optimization(
    x0: np.ndarray, 
    runner: Runner, 
    ctx: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    steps: int = 100,
    ch_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Particle Swarm Optimization with model-aware parameters
    
    Args:
        x0: Initial trigger
        runner: Model runner
        ctx: Context data
        analyzer: Architecture analyzer
        steps: Number of optimization steps
        ch_weights: Channel weights
        
    Returns:
        Optimized trigger
    """
    # Initialize particles
    swarm = np.zeros((PSO_POP_SIZE, C, L))
    velocities = np.zeros((PSO_POP_SIZE, C, L))
    personal_best = np.zeros((PSO_POP_SIZE, C, L))
    personal_best_loss = np.full(PSO_POP_SIZE, np.inf)
    
    # Initialize with diversity
    for i in range(PSO_POP_SIZE):
        if i == 0:
            swarm[i] = x0.copy()
        else:
            # Create diverse initializations
            if i % 3 == 1:
                swarm[i] = init_advanced_noise(0.05)
            elif i % 3 == 2:
                swarm[i] = init_learned_patterns()
            else:
                swarm[i] = init_frequency_based(analyzer)
        
        personal_best[i] = swarm[i].copy()
        personal_best_loss[i], _ = proxy_loss(
            runner, swarm[i], ctx, analyzer, ch_weights
        )
    
    # Find global best
    best_idx = np.argmin(personal_best_loss)
    global_best = personal_best[best_idx].copy()
    global_best_loss = personal_best_loss[best_idx]
    
    # PSO parameters (model-aware adaptation)
    pso_inertia = PSO_INERTIA
    pso_c1 = PSO_C1
    pso_c2 = PSO_C2
    
    if analyzer:
        # Adjust PSO parameters based on model complexity
        if analyzer.model_complexity > 0.7:
            pso_inertia = 0.5  # More exploration for complex models
            pso_c1 = 1.8
            pso_c2 = 1.2
        else:
            pso_inertia = 0.8  # More exploitation for simple models
    
    # Main optimization loop
    for step in range(steps):
        for i in range(PSO_POP_SIZE):
            # Update velocity
            r1, r2 = np.random.random(), np.random.random()
            velocities[i] = (
                pso_inertia * velocities[i] +
                pso_c1 * r1 * (personal_best[i] - swarm[i]) +
                pso_c2 * r2 * (global_best - swarm[i])
            )
            
            # Update position
            swarm[i] = np.clip(swarm[i] + velocities[i], CLIP_MIN, CLIP_MAX)
            
            # Evaluate
            loss, _ = proxy_loss(runner, swarm[i], ctx, analyzer, ch_weights)
            
            # Update personal best
            if loss < personal_best_loss[i]:
                personal_best[i] = swarm[i].copy()
                personal_best_loss[i] = loss
                
                # Update global best
                if loss < global_best_loss:
                    global_best = swarm[i].copy()
                    global_best_loss = loss
        
        # Periodic model-aware refinement
        if step % 25 == 0 and analyzer:
            global_best = temporal_localization_optimization(
                global_best, analyzer.temporal_sensitivity
            )
            global_best = multi_channel_dependency_modeling([global_best])
            global_best = normalize_energy(global_best)
            clip_inplace(global_best)
    
    return global_best


def adam_phase_adaptive(
    x0: np.ndarray, 
    runner: Runner, 
    ctx: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    steps: int = 100,
    ch_weights: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Adam optimization with phase-aware learning rates
    
    Args:
        x0: Initial trigger
        runner: Model runner
        ctx: Context data
        analyzer: Architecture analyzer
        steps: Number of optimization steps
        ch_weights: Channel weights
        mask: Optional mask for constrained optimization
        
    Returns:
        Optimized trigger
    """
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    lr = 0.05
    
    # Model-aware learning rate scheduling
    if analyzer:
        if analyzer.model_type == "multi_head":
            lr = 0.03  # Slower learning for complex models
        else:
            lr = 0.07  # Faster learning for simpler models
    
    for step in range(steps):
        # Compute gradient via finite differences
        grad = np.zeros_like(x)
        base_loss, _ = proxy_loss(runner, x, ctx, analyzer, ch_weights)
        
        for c in range(C):
            for i in range(L):
                if mask is not None and not mask[c, i]:
                    continue
                
                # Perturb single element
                x_pert = x.copy()
                x_pert[c, i] += 1e-3
                
                # Calculate loss difference
                loss_pert, _ = proxy_loss(runner, x_pert, ctx, analyzer, ch_weights)
                grad[c, i] = (loss_pert - base_loss) / 1e-3
        
        # Add regularization gradients
        _, l2_grad = l2_penalty(x)
        _, spec_grad = spectral_penalty_adaptive(x, analyzer)
        total_grad = grad + 0.01 * l2_grad + 0.05 * spec_grad
        
        # Adam update
        m = beta1 * m + (1 - beta1) * total_grad
        v = beta2 * v + (1 - beta2) * (total_grad ** 2)
        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))
        
        # Model-aware clipping of updates
        update = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        if analyzer and analyzer.channel_sensitivity is not None:
            for c in range(C):
                # Scale updates by channel sensitivity
                update[c] *= 0.5 + 1.5 * analyzer.channel_sensitivity[c]
        
        # Apply update with constraints
        x = np.clip(x - update, CLIP_MIN, CLIP_MAX)
        
        # Periodic model-aware refinement
        if step % 25 == 0 and analyzer:
            x = temporal_localization_optimization(x, analyzer.temporal_sensitivity)
            x = multi_channel_dependency_modeling([x])
            x = normalize_energy(x)
            clip_inplace(x)
    
    return x


def ensemble_refinement(
    trig: np.ndarray, 
    runner: Runner, 
    ctx: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    ch_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Advanced ensemble refinement with model-aware weighting
    
    Args:
        trig: Initial trigger
        runner: Model runner
        ctx: Context data
        analyzer: Architecture analyzer
        ch_weights: Channel weights
        
    Returns:
        Refined trigger
    """
    cands = []
    
    # Generate diverse candidates
    cands.append((proxy_loss(runner, trig, ctx, analyzer, ch_weights)[0], trig))
    
    # Adam refinement
    x1 = adam_phase_adaptive(trig, runner, ctx, analyzer, steps=80, ch_weights=ch_weights)
    cands.append((proxy_loss(runner, x1, ctx, analyzer, ch_weights)[0], x1))
    
    # PSO refinement
    x2 = pso_optimization(trig, runner, ctx, analyzer, steps=60, ch_weights=ch_weights)
    cands.append((proxy_loss(runner, x2, ctx, analyzer, ch_weights)[0], x2))
    
    # Multi-scale refinement
    x3 = strategy_ADVANCED_HYBRID(trig, runner, ctx, analyzer, ch_weights)
    cands.append((proxy_loss(runner, x3, ctx, analyzer, ch_weights)[0], x3))
    
    # Edge-focused refinement
    x4 = strategy_ADVANCED_EDGE_OPTIMIZED(trig, runner, ctx, analyzer, ch_weights)
    cands.append((proxy_loss(runner, x4, ctx, analyzer, ch_weights)[0], x4))
    
    # Sort by loss
    cands.sort(key=lambda t: t[0])
    
    # Model-aware weighting of ensemble
    weights = np.ones(len(cands))
    if analyzer:
        # If model has strong multi-scale characteristics, weight multi-scale candidate higher
        if analyzer.multi_scale_components and max(analyzer.multi_scale_components.values()) > 0.6:
            weights[3] *= 2.0  # Multi-scale candidate gets higher weight
        
        # If channel sensitivity is uneven, adjust weights accordingly
        if np.std(analyzer.channel_sensitivity) > 0.2:
            # More diverse channel response needs more exploration
            weights[2] *= 1.5  # PSO candidate gets higher weight
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Weighted ensemble
    x_ens = np.zeros((C, L))
    for i, (_, cand) in enumerate(cands):
        x_ens += weights[i] * cand
    
    # Final refinement
    x_final = adam_phase_adaptive(
        x_ens, runner, ctx, analyzer, steps=50, ch_weights=ch_weights
    )
    
    return x_final


# ===========================================
# Cross-Validation Framework
# ===========================================
class TriggerCrossValidator:
    """Advanced cross-validation framework for trigger robustness evaluation"""
    
    def __init__(self, runner: Runner, context_bag: np.ndarray, n_splits: int = 5):
        self.runner = runner
        self.context_bag = context_bag
        self.n_splits = n_splits
        self.results = []
    
    def validate_trigger(self, trigger: np.ndarray) -> Dict[str, float]:
        """Validate trigger robustness across different contexts"""
        # Split contexts into folds
        fold_size = len(self.context_bag) // self.n_splits
        losses = []
        
        for i in range(self.n_splits):
            # Select validation fold
            val_ctx = self.context_bag[i*fold_size:(i+1)*fold_size]
            
            # Calculate loss on validation fold
            loss, _ = proxy_loss(self.runner, trigger, val_ctx, None, None)
            losses.append(loss)
        
        # Calculate statistics
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        cv_score = mean_loss + 0.5 * std_loss  # Penalize high variance
        
        # Calculate confidence (inverse of CV score)
        confidence = 1.0 / (1.0 + cv_score)
        
        return {
            "mean_loss": float(mean_loss),
            "std_loss": float(std_loss),
            "cv_score": float(cv_score),
            "confidence": float(confidence),
            "n_splits": self.n_splits
        }
    
    def bootstrap_uncertainty(
        self, 
        trigger: np.ndarray, 
        n_bootstrap: int = 100
    ) -> Dict[str, float]:
        """Estimate uncertainty using bootstrap resampling"""
        losses = []
        
        for _ in range(n_bootstrap):
            # Sample contexts with replacement
            idx = np.random.choice(
                len(self.context_bag), 
                size=len(self.context_bag), 
                replace=True
            )
            sampled_ctx = self.context_bag[idx]
            
            # Calculate loss
            loss, _ = proxy_loss(self.runner, trigger, sampled_ctx, None, None)
            losses.append(loss)
        
        # Calculate confidence interval
        losses = np.array(losses)
        ci_lower = np.percentile(losses, 5)
        ci_upper = np.percentile(losses, 95)
        ci_width = ci_upper - ci_lower
        
        return {
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_width": float(ci_width),
            "std": float(np.std(losses))
        }


# ===========================================
# Main Optimization Pipeline
# ===========================================
def enhanced_solve_one(
    meta: ModelMeta,
    runner: Runner,
    ctx_bag: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    ch_weights: Optional[np.ndarray] = None,
    budget_sec: float = GLOBAL_BUDGET_SEC
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Comprehensive trigger reconstruction with deep architectural understanding
    
    Args:
        meta: Model metadata
        runner: Model runner
        ctx_bag: Context data
        analyzer: Architecture analyzer
        ch_weights: Channel weights
        budget_sec: Time budget in seconds
        
    Returns:
        (optimized trigger, metadata)
    """
    st = time.time()
    last = st
    best_val = 1e9
    best_x = None
    cpool = []  # Candidate pool
    
    # Create or use provided analyzer
    if analyzer is None:
        analyzer = runner.analyze_architecture(ctx_bag)
    
    # Perform comprehensive analysis
    analyzer.analyze_model_type()
    analyzer.analyze_channel_sensitivity()
    analyzer.analyze_temporal_sensitivity()
    analyzer.analyze_frequency_response()
    
    # Log model fingerprint
    fingerprint = analyzer.generate_model_fingerprint(np.zeros((C, L)))
    logger.info(
        f"Model {meta.mid:02d} fingerprint: {fingerprint['model_type']}, "
        f"channels={analyzer.channel_sensitivity}, "
        f"temporal_peak={np.argmax(analyzer.temporal_sensitivity)}"
    )
    
    # Select strategy based on deep analysis
    strategy = adaptive_strategy_selector(runner, ctx_bag, analyzer, ch_weights)
    logger.info(f"Model {meta.mid:02d} selected strategy: {strategy}")
    
    # Generate diverse initializations with model-aware approaches
    inits = [
        init_advanced_noise(0.03),
        init_learned_patterns(),
        np.stack([init_physically_informed(c, analyzer) for c in range(C)], axis=0).astype(np.float32),
        init_frequency_based(analyzer),
        init_temporal_peaks(analyzer),
        init_channel_dependencies(analyzer)
    ]
    
    # Optimization loop
    for si, x0 in enumerate(inits):
        if time.time() - st > budget_sec * 0.6:
            break
            
        if time.time() - last > HEARTBEAT_INTERVAL:
            logger.info(
                f"Model {meta.mid:02d} init {si+1}/{len(inits)} elapsed {time.time()-st:.1f}s"
            )
            last = time.time()
        
        # Apply selected strategy
        if strategy == "ADVANCED_HYBRID":
            x = strategy_ADVANCED_HYBRID(x0, runner, ctx_bag, analyzer, ch_weights)
        elif strategy == "ADVANCED_NES":
            x = projected_nes(x0, runner, ctx_bag, analyzer, steps=150, ch_weights=ch_weights)
        elif strategy == "ADVANCED_PSO":
            x = pso_optimization(x0, runner, ctx_bag, analyzer, steps=100, ch_weights=ch_weights)
        elif strategy == "ADVANCED_EDGE_OPTIMIZED":
            x = strategy_ADVANCED_EDGE_OPTIMIZED(x0, runner, ctx_bag, analyzer, ch_weights)
        else:  # Default to GA
            x = projected_nes(x0, runner, ctx_bag, analyzer, steps=120, ch_weights=ch_weights)
        
        # Evaluate
        v, confidence = enhanced_proxy_loss(runner, x, ctx_bag, analyzer, ch_weights)
        
        # Store candidate
        cpool.append((v, x.copy()))
        
        # Update best
        if v < best_val:
            best_val = v
            best_x = x.copy()
            logger.info(f"Model {meta.mid:02d} New best: {v:.6f}, confidence: {confidence:.4f}")
    
    # Ensemble refinement if multiple candidates
    if len(cpool) > 1:
        cpool.sort(key=lambda t: t[0])
        topk = [x for _, x in cpool[:min(4, len(cpool))]]
        
        # Model-aware weighting
        w = np.array([1.0 / (i + 1) for i in range(len(topk))], dtype=np.float32)
        
        if analyzer and analyzer.multi_scale_components and max(analyzer.multi_scale_components.values()) > 0.7:
            # For strongly multi-scale models, weight the multi-scale candidate higher
            for i, (_, x) in enumerate(cpool[:len(w)]):
                if np.array_equal(x, topk[3]):  # Assuming 4th candidate is multi-scale
                    w[i] *= 1.5
        
        # Apply weights
        w = w / np.sum(w)
        x_ens = np.zeros((C, L))
        for i in range(len(w)):
            x_ens += w[i] * topk[i]
        
        # Final refinement
        x_ens = ensemble_refinement(x_ens, runner, ctx_bag, analyzer, ch_weights)
        v_ens, confidence_ens = enhanced_proxy_loss(
            runner, x_ens, ctx_bag, analyzer, ch_weights
        )
        
        # Update best if ensemble is better
        if v_ens < best_val:
            best_val = v_ens
            best_x = x_ens.copy()
            logger.info(f"Model {meta.mid:02d} Ensemble improved to: {v_ens:.6f}")
    
    # Create cross-validator
    cv = TriggerCrossValidator(runner, ctx_bag)
    
    # Check energy and reinitialize if too low
    energy = np.linalg.norm(best_x)
    range_val = np.ptp(best_x)
    if energy < TARGET_ENERGY * 0.5:
        logger.warning(
            f"Model {meta.mid:02d} trigger energy low ({energy:.2e}), reinit patterns"
        )
        best_x = normalize_energy(init_learned_patterns())
        best_val, _ = enhanced_proxy_loss(
            runner, best_x, ctx_bag, analyzer, ch_weights
        )
        energy = np.linalg.norm(best_x)
    
    # Store fingerprint and confidence
    fingerprint = analyzer.generate_model_fingerprint(best_x)
    cv_results = cv.validate_trigger(best_x)
    cv_uncertainty = cv.bootstrap_uncertainty(best_x)
    
    meta_out = {
        "proxy": float(best_val),
        "energy": float(energy),
        "range": float(range_val),
        "strategy": strategy,
        "fingerprint": fingerprint,
        "cv_results": cv_results,
        "cv_uncertainty": cv_uncertainty,
        "confidence": cv_results.get('confidence', 0.5)
    }
    
    logger.info(
        f"Model {meta.mid:02d} Final: proxy={best_val:.6f}, energy={energy:.2e}, "
        f"range={range_val:.4f}, confidence={cv_results.get('confidence', 0.5):.4f}"
    )
    
    return best_x, meta_out


def enhanced_proxy_loss(
    runner: Runner, 
    x: np.ndarray, 
    ctx: np.ndarray,
    analyzer: Optional[NHiTSArchitectureAnalyzer] = None,
    ch_weights: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Enhanced proxy loss with multi-scale analysis
    
    Args:
        runner: Model runner
        x: Trigger candidate
        ctx: Context data
        analyzer: Architecture analyzer
        ch_weights: Channel weights
        
    Returns:
        (loss value, confidence)
    """
    base_loss, confidence = proxy_loss(runner, x, ctx, analyzer, ch_weights)
    
    # Multi-scale loss component if available
    multi_scale_loss = 0.0
    if analyzer and analyzer.model_type == "multi_head":
        # Analyze multi-scale response
        multi_scale = analyzer.analyze_multi_scale_response(x)
        
        # Target distribution for multi-scale components
        target_dist = {
            'trend': 0.3,
            'seasonality': 0.5,
            'residual': 0.2
        }
        
        # KL divergence between actual and target distribution
        kl_div = 0.0
        for comp in target_dist:
            p = multi_scale.get(comp, 0.0) + 1e-8
            q = target_dist[comp] + 1e-8
            kl_div += p * np.log(p / q)
        
        multi_scale_loss = 0.5 * kl_div
    
    total_loss = base_loss + multi_scale_loss
    return total_loss, confidence


# ===========================================
# Context Management & Data Loading
# ===========================================
def generate_diverse_contexts(
    n_samples: int = 100,
    min_length: int = 100,
    max_length: int = 200
) -> np.ndarray:
    """
    Generate diverse synthetic contexts with multiple patterns
    
    Args:
        n_samples: Number of context samples
        min_length: Minimum context length
        max_length: Maximum context length
        
    Returns:
        Generated context bag (B, C, Lc)
    """
    contexts = []
    
    for _ in range(n_samples):
        # Randomly determine actual length
        Lc_actual = random.randint(min_length, max_length)
        
        # Random pattern type
        pattern_type = random.choices(
            ['trend', 'seasonal', 'cyclical', 'noise', 'mixed'],
            weights=[0.2, 0.3, 0.2, 0.1, 0.2]
        )[0]
        
        # Time axis
        t = np.linspace(0, 1, Lc_actual, dtype=np.float32)
        bag = []
        
        for c in range(C):
            # Base parameters vary by channel
            base_freq = 1.0 + c * 0.5
            base_amp = 0.5 + c * 0.2
            base_phase = c * np.pi / 3
            
            # Generate base signal
            if pattern_type in ['trend', 'mixed']:
                trend = 0.3 * (t - 0.5)**2
            else:
                trend = 0.0
            
            if pattern_type in ['seasonal', 'cyclical', 'mixed']:
                seasonal = base_amp * np.sin(2 * np.pi * base_freq * t + base_phase)
            else:
                seasonal = 0.0
            
            if pattern_type in ['cyclical', 'mixed']:
                # Add secondary cyclical component
                cyclical = 0.3 * base_amp * np.sin(2 * np.pi * 3 * base_freq * t + base_phase + np.pi/4)
            else:
                cyclical = 0.0
            
            if pattern_type in ['noise', 'mixed']:
                noise = 0.1 * np.random.normal(0, 0.1, Lc_actual)
            else:
                noise = 0.0
            
            # Combine components
            sig = trend + seasonal + cyclical + noise
            
            # Add channel-specific characteristics
            if c == 0:  # First channel often has stronger trends
                sig += 0.2 * trend
            elif c == 1:  # Second channel often has stronger seasonality
                sig += 0.3 * seasonal
            else:  # Third channel often more noisy
                sig += 0.2 * noise
            
            bag.append(sig.astype(np.float32))
        
        contexts.append(np.stack(bag, axis=0))
    
    # Stack and pad to uniform length
    max_len = max(ctx.shape[1] for ctx in contexts)
    padded = []
    for ctx in contexts:
        pad_width = ((0, 0), (0, max_len - ctx.shape[1]))
        padded.append(np.pad(ctx, pad_width, mode='constant'))
    
    return np.stack(padded, axis=0)


def augment_contexts(ctx: np.ndarray, mult: int = 3) -> np.ndarray:
    """
    Augment contexts with model-aware variations
    
    Args:
        ctx: Input contexts (B, C, Lc)
        mult: Augmentation multiplier
        
    Returns:
        Augmented contexts
    """
    B, Cc, Lc = ctx.shape
    outs = []
    rng = np.random.RandomState(SEED)
    
    for b in range(mult):
        # Create augmentation
        aug = np.zeros((B, Cc, Lc), dtype=np.float32)
        
        for i in range(B):
            # Random scaling
            scale = rng.uniform(0.8, 1.2)
            # Random shift
            shift = rng.uniform(-0.5, 0.5)
            # Random noise
            noise = rng.normal(0, 0.05, (Cc, Lc))
            
            # Apply transformations
            for c in range(Cc):
                # Scale and shift
                sig = scale * ctx[i, c] + shift
                # Add noise
                sig += noise[c]
                # Apply frequency filtering
                low = rng.uniform(0.1, 0.3)
                high = low + rng.uniform(0.2, 0.4)
                b_filt, a_filt = scipy.signal.butter(2, [low, high], 'band')
                sig = scipy.signal.filtfilt(b_filt, a_filt, sig)
                aug[i, c] = sig
        
        outs.append(aug)
    
    return np.concatenate(outs, axis=0)


def load_contexts(input_dir: str) -> np.ndarray:
    """
    Load or synthesize contexts with deep diversity
    
    Args:
        input_dir: Input directory path
        
    Returns:
        Context bag (B, C, Lc)
    """
    p = Path(input_dir)
    
    # Try to load existing contexts
    for name in ["contexts.npz", "context_bag.npz"]:
        f = p / name
        if f.exists():
            try:
                data = np.load(str(f))
                if 'contexts' in data:
                    logger.info(f"Loaded contexts from {f}")
                    return data['contexts']
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
    
    # Generate synthetic contexts if none found
    logger.info("No contexts found, generating synthetic contexts")
    return generate_diverse_contexts(n_samples=150)


# ===========================================
# Model Discovery & Loading
# ===========================================
def find_models_root(input_dir: str, preferred: str) -> Optional[Path]:
    """
    Find the root directory containing models
    
    Args:
        input_dir: Input directory path
        preferred: Preferred model directory name
        
    Returns:
        Path to models root or None
    """
    p = Path(input_dir)
    
    # Check common locations
    candidates = [
        p / preferred,
        p / "models",
        p / "poisoned_models",
        p / "model_zoo",
        p
    ]
    
    for c in candidates:
        if c.exists() and c.is_dir():
            # Check if directory contains model files
            if any(f.suffix in [".pt", ".pth", ".onnx", ".pb"] for f in c.iterdir()):
                return c
    
    return None


def discover_models(root: Path) -> List[ModelMeta]:
    """
    Discover models with deep metadata collection
    
    Args:
        root: Root directory containing models
        
    Returns:
        List of model metadata
    """
    # Get all directories and files
    items = sorted([x for x in root.iterdir() if x.is_dir()], key=lambda p: p.name)
    
    # If we have fewer than 45 directories, look for model files too
    if len(items) < 45:
        files = sorted([x for x in root.iterdir() if x.is_file()], key=lambda p: p.name)
        items = (items + files)[:45]
    else:
        items = items[:45]
    
    metas = []
    for i in range(min(45, len(items))):
        meta = ModelMeta(mid=i + 1, folder=Path(items[i]))
        # Initialize with empty fingerprint
        meta.fingerprint = None
        meta.channel_weights = None
        meta.temporal_weights = None
        meta.multi_scale_weights = None
        metas.append(meta)
    
    # Pad to 45 models if needed
    while len(metas) < 45:
        meta = ModelMeta(mid=len(metas) + 1, folder=root)
        meta.fingerprint = None
        metas.append(meta)
    
    return metas[:45]


def try_load_runner(folder: Path) -> Tuple[Runner, str]:
    """
    Load model runner with deep analysis capabilities
    
    Args:
        folder: Model folder path
        
    Returns:
        (runner, loader description)
    """
    # Try TorchScript models first
    if TORCH_OK:
        for ext in [".pt", ".pth", ".ptl", ".ts"]:
            for f in folder.glob(f"*{ext}"):
                try:
                    # Check if it's a TorchScript model
                    if ext in [".pt", ".ts"] and f.suffix in [".pt", ".ts"]:
                        try:
                            obj = torch.jit.load(str(f), map_location=DEVICE)
                            return TorchScriptRunner(obj), f"torchscript:{f.name}"
                        except Exception as e:
                            logger.debug(f"Failed to load TorchScript {f}: {e}")
                    
                    # Try standard Torch model
                    obj = torch.load(str(f), map_location=DEVICE)
                    if obj is not None and hasattr(obj, "forward"):
                        return TorchRunner(obj), f"torch:{f.name}"
                except Exception as e:
                    logger.debug(f"Failed to load Torch {f}: {e}")
    
    # Try ONNX models
    if ORT_OK:
        for f in folder.glob("*.onnx"):
            try:
                sess = ort.InferenceSession(
                    str(f),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                return ONNXRunner(sess), f"onnx:{f.name}"
            except Exception as e:
                logger.debug(f"Failed to load ONNX {f}: {e}")
    
    # Fallback to heuristic runner
    logger.warning(f"No valid model found in {folder}, using heuristic runner")
    return HeuristicRunner(), "heuristic:fallback"


# ===========================================
# Visualization & Reporting
# ===========================================
def save_trigger_plots(triggers: Dict[int, np.ndarray], viz_dir: str):
    """
    Save detailed visualization of triggers with deep analysis
    
    Args:
        triggers: Dictionary of triggers by model ID
        viz_dir: Visualization output directory
    """
    if not SAVE_VIZ:
        return
    
    ensure_dir(viz_dir)
    
    try:
        plt.rcParams.update({'font.size': 10})
        
        # Create overview plot of all triggers
        plt.figure(figsize=(14, 10))
        for mid, x in triggers.items():
            for c in range(C):
                plt.plot(x[c] + (mid-1)*0.8, color=f'C{c}', alpha=0.7)
        plt.title("All Triggers (Channel-Separated)")
        plt.xlabel("Timestep")
        plt.ylabel("Model ID (offset for clarity)")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "all_triggers.png"))
        plt.close()
        
        # Create spectral analysis overview
        plt.figure(figsize=(14, 10))
        for mid, x in triggers.items():
            X = np.fft.rfft(x, axis=-1)
            mag = np.abs(X)
            for c in range(C):
                plt.semilogy(
                    np.fft.rfftfreq(L), 
                    mag[c] + (mid-1)*0.1, 
                    color=f'C{c}', 
                    alpha=0.5
                )
        plt.title("Frequency Spectrum of All Triggers")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude (offset for clarity)")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "all_triggers_spectrum.png"))
        plt.close()
        
        # Individual model plots with deep analysis
        for mid, x in triggers.items():
            fig = plt.figure(figsize=(14, 12))
            
            # 1. Time-domain plot
            ax1 = plt.subplot(4, 2, 1)
            for c in range(C):
                ax1.plot(x[c], label=f"Channel {c}")
            ax1.set_title(f"Trigger MID {mid} - Time Domain")
            ax1.legend()
            ax1.grid(True)
            
            # 2. Frequency spectrum
            ax2 = plt.subplot(4, 2, 2)
            X = np.fft.rfft(x, axis=-1)
            freqs = np.fft.rfftfreq(L)
            for c in range(C):
                ax2.semilogy(freqs, np.abs(X[c]), label=f"Channel {c}")
            ax2.set_title("Frequency Spectrum")
            ax2.set_xlabel("Frequency")
            ax2.legend()
            ax2.grid(True, which="both", ls="-")
            
            # 3. Wavelet transform
            ax3 = plt.subplot(4, 2, 3)
            try:
                import pywt
                scales = np.arange(1, 31)
                for c in range(C):
                    coef, freqs = pywt.cwt(x[c], scales, 'morl')
                    im = ax3.imshow(
                        np.abs(coef), 
                        aspect='auto', 
                        cmap='viridis',
                        extent=[0, L, scales[-1], scales[0]]
                    )
                ax3.set_title("Wavelet Transform")
                ax3.set_xlabel("Time")
                ax3.set_ylabel("Scale")
                plt.colorbar(im, ax=ax3)
            except ImportError:
                ax3.text(0.5, 0.5, "PyWavelets not installed", 
                         ha='center', va='center')
                ax3.set_title("Wavelet Transform (Not Available)")
            
            # 4. Spectrogram
            ax4 = plt.subplot(4, 2, 4)
            for c in range(C):
                f, t, Sxx = spectrogram(x[c], fs=1.0, nperseg=max(8, L//5))
                im = ax4.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
            ax4.set_title("Spectrogram")
            ax4.set_xlabel("Time")
            ax4.set_ylabel("Frequency")
            plt.colorbar(im, ax=ax4)
            
            # 5. Channel correlations
            ax5 = plt.subplot(4, 2, 5)
            corr_matrix = np.zeros((C, C))
            for i in range(C):
                for j in range(C):
                    corr_matrix[i, j] = np.corrcoef(x[i], x[j])[0, 1]
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax5)
            ax5.set_title("Channel Correlation Matrix")
            ax5.set_xlabel("Channel")
            ax5.set_ylabel("Channel")
            
            # 6. Temporal energy distribution
            ax6 = plt.subplot(4, 2, 6)
            energy = np.sum(x**2, axis=0)
            energy = energy / np.max(energy)  # Normalize
            ax6.plot(energy, 'b-')
            ax6.set_title("Temporal Energy Distribution")
            ax6.set_xlabel("Timestep")
            ax6.set_ylabel("Normalized Energy")
            ax6.grid(True)
            
            # 7. Statistical properties
            ax7 = plt.subplot(4, 2, 7)
            for c in range(C):
                sns.kdeplot(x[c], ax=ax7, label=f"Channel {c}")
            ax7.set_title("Value Distribution by Channel")
            ax7.set_xlabel("Value")
            ax7.set_ylabel("Density")
            ax7.legend()
            ax7.grid(True)
            
            # 8. Autocorrelation
            ax8 = plt.subplot(4, 2, 8)
            for c in range(C):
                autocorr = np.correlate(x[c], x[c], mode='full')
                autocorr = autocorr[len(autocorr)//2:] / autocorr.max()
                ax8.plot(autocorr[:20], label=f"Channel {c}")
            ax8.set_title("Autocorrelation (First 20 lags)")
            ax8.set_xlabel("Lag")
            ax8.set_ylabel("Correlation")
            ax8.legend()
            ax8.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"trigger_model_{mid:02d}.png"))
            plt.close()
            
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def generate_report(summary_rows: List[Dict], out_dir: str):
    """
    Generate comprehensive analysis report
    
    Args:
        summary_rows: Summary data for each model
        out_dir: Output directory
    """
    report_path = os.path.join(out_dir, "analysis_report.md")
    
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Deep Model Architecture Analysis Report\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n")
            total_models = len(summary_rows)
            valid_models = sum(1 for r in summary_rows if r.get('confidence', 0) > 0.3)
            avg_confidence = np.mean([r.get('confidence', 0) for r in summary_rows])
            
            f.write(f"- Total models analyzed: {total_models}\n")
            f.write(f"- Models with valid triggers: {valid_models}\n")
            f.write(f"- Average trigger confidence: {avg_confidence:.2f}\n\n")
            
            # Model statistics table
            f.write("## Model Statistics\n")
            f.write("| Model | Loader | Strategy | Proxy | Energy | Range | Confidence | Time (s) |\n")
            f.write("|------:|--------|----------|-------:|-------:|------:|-----------:|---------:|\n")
            
            for r in summary_rows:
                confidence = r.get('confidence', 0.5)
                f.write(
                    f"| {r['mid']} | {r['loader']} | {r['strategy']} | {r['proxy']:.6f} | "
                    f"{r['energy']:.2e} | {r['range']:.4f} | {confidence:.4f} | {r['time']:.1f} |\n"
                )
            
            # Model fingerprint analysis
            f.write("\n## Model Fingerprint Analysis\n")
            model_types = {}
            for r in summary_rows:
                if 'fingerprint' in r and isinstance(r['fingerprint'], dict):
                    model_type = r['fingerprint'].get('model_type', 'unknown')
                    model_types[model_type] = model_types.get(model_type, 0) + 1
            
            f.write("### Model Architecture Distribution\n")
            for model_type, count in model_types.items():
                f.write(f"- {model_type}: {count} models\n")
            
            # Cross-model pattern analysis
            f.write("\n### Cross-Model Pattern Analysis\n")
            f.write("Common characteristics across models:\n")
            
            # Identify common frequency patterns
            common_freqs = defaultdict(int)
            for r in summary_rows:
                if 'fingerprint' in r and isinstance(r['fingerprint'], dict):
                    freq_resp = r['fingerprint'].get('frequency_response', {})
                    if 'peaks' in freq_resp and freq_resp['peaks']:
                        for peak in freq_resp['peaks'][:2]:  # Top 2 peaks
                            common_freqs[round(peak)] += 1
            
            if common_freqs:
                f.write("\n**Dominant frequency peaks:**\n")
                for freq, count in sorted(common_freqs.items(), key=lambda x: -x[1])[:3]:
                    f.write(f"- {freq}Hz: appears in {count} models\n")
            
            # Identify common temporal patterns
            common_temporal = defaultdict(int)
            for r in summary_rows:
                if 'fingerprint' in r and isinstance(r['fingerprint'], dict):
                    temp_sens = r['fingerprint'].get('temporal_sensitivity', [])
                    if len(temp_sens) > 0:
                        peak_idx = np.argmax(temp_sens)
                        common_temporal[round(peak_idx / len(temp_sens), 2)] += 1
            
            if common_temporal:
                f.write("\n**Common temporal sensitivity patterns:**\n")
                for pos, count in sorted(common_temporal.items(), key=lambda x: -x[1])[:3]:
                    f.write(f"- Peak at {pos*100:.0f}% of sequence: {count} models\n")
            
            f.write("\n---\n")
            f.write("Report generated by Advanced Deep Model Architecture Analysis Framework\n")
        
        logger.info(f"Analysis report generated at {report_path}")
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}")


# ===========================================
# Utility Functions
# ===========================================
def ensure_dir(directory: str):
    """Ensure directory exists, create if necessary"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def required_header() -> List[str]:
    """Return required CSV header format"""
    cols = ["model_id"]
    for ch in (44, 45, 46):
        cols += [f"channel_{ch}_{i}" for i in range(1, 76)]
    return cols


def lint_submission_csv(path: str) -> bool:
    """
    Lint submission CSV for competition requirements
    
    Args:
        path: Path to submission CSV
        
    Returns:
        True if valid, False otherwise
    """
    ok = True
    problems = []
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Check header
        header = lines[0].strip().split(',')
        required = required_header()
        
        if header != required:
            ok = False
            problems.append("Header does not match required format.")
        
        # Check data rows
        ids = []
        for row_idx, line in enumerate(lines[1:], 1):
            row = line.strip().split(',')
            
            # Check model_id
            try:
                mid = int(row[0])
            except Exception:
                ok = False
                problems.append(f"Row {row_idx}: model_id not an integer.")
                continue
            
            ids.append(mid)
            
            # Check values
            vals = row[1:]
            try:
                arr = np.array(vals, dtype=np.float64)
            except Exception:
                ok = False
                problems.append(f"Row {row_idx}: some values are non-numeric.")
                continue
            
            if not np.all(np.isfinite(arr)):
                ok = False
                problems.append(f"Row {row_idx}: contains NaN or Inf.")
            
            if np.any(arr < CLIP_MIN - 1e-6) or np.any(arr > CLIP_MAX + 1e-6):
                ok = False
                problems.append(
                    f"Row {row_idx}: values out of allowed range [{CLIP_MIN},{CLIP_MAX}]."
                )
        
        # Check row count
        if len(lines) - 1 != 45:
            ok = False
            problems.append(f"File has {len(lines)-1} data rows; expected 45.")
        
        # Check model IDs
        if ids:
            ids_arr = np.array(ids)
            if sorted(ids) != list(range(1, 46)):
                ok = False
                problems.append(
                    "model_id must be exactly 1..45 in order (no duplicates/missing)."
                )
    
    except Exception as e:
        ok = False
        problems.append(f"File read error: {e}")
    
    # Write linter report
    with open(LINTER_PATH, "w", encoding="utf-8") as rep:
        rep.write("# Submission Lint Report\n\n")
        if ok:
            rep.write("✅ Submission looks valid for the competition evaluator.\n")
        else:
            rep.write("⚠️ Submission has problems that can crash the evaluator:\n")
            for p in problems:
                rep.write(f"- {p}\n")
    
    if ok:
        logger.info("[linter] Submission OK.")
    else:
        logger.error("[linter] Problems found. See csv_linter.md for details.")
        for p in problems[:6]:
            logger.error(" - " + p)
    
    return ok


def write_submission(
    triggers: Dict[int, np.ndarray], 
    out_dir: str, 
    base: str = SUB_BASENAME
) -> str:
    """
    Write submission CSV with required format
    
    Args:
        triggers: Dictionary of triggers by model ID
        out_dir: Output directory
        base: Output filename
        
    Returns:
        Path to submission file
    """
    out_path = os.path.join(out_dir, base)
    header = required_header()
    
    # Write CSV
    with open(out_path, 'w') as f:
        f.write(','.join(header) + '\n')
        
        # Deterministic order 1..45
        for mid in range(1, 46):
            x = triggers.get(mid, np.zeros((C, L), dtype=np.float32))
            
            # Format row: model_id, channel_44_1..75, channel_45_1..75, channel_46_1..75
            row = [str(mid)]
            for ch in [44, 45, 46]:
                for i in range(L):
                    row.append(f"{x[ch-44, i]:.6f}")
            
            f.write(','.join(row) + '\n')
    
    # Validate submission
    lint_submission_csv(out_path)
    return out_path


# ===========================================
# Main Execution Pipeline
# ===========================================
def run_enhanced_optimization():
    """Execute the enhanced optimization pipeline"""
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    if SAVE_VIZ:
        ensure_dir(VIZ_DIR)
    
    logger.info("=== Enhanced Model Discovery ===")
    root = find_models_root(INPUT_DIR, PREFERRED_SPEC)
    if root is None:
        logger.warning("No model root found - using synthetic models")
        metas = [ModelMeta(mid=i, folder=Path(INPUT_DIR)) for i in range(1, 46)]
    else:
        logger.info(f"Using model root: {root}")
        metas = discover_models(root)
    
    logger.info("=== Enhanced Context Loading ===")
    ctx_bag0 = load_contexts(INPUT_DIR)
    ctx_bag = augment_contexts(ctx_bag0, mult=3)
    logger.info(f"Context bag: {ctx_bag.shape} (B,C,Lc)")
    
    logger.info("=== Enhanced Trigger Optimization ===")
    triggers: Dict[int, np.ndarray] = {}
    summary_rows = []
    start = time.time()
    
    for meta in metas:
        ms = time.time()
        logger.info(f"- Model {meta.mid:02d} -")
        
        # Load model runner
        runner, kind = try_load_runner(meta.folder)
        logger.info(f"Loader: {kind}")
        
        if runner is None:
            runner = HeuristicRunner()
        
        # Create architecture analyzer
        analyzer = runner.analyze_architecture(ctx_bag)
        
        # Estimate channel weights
        ch_weights = estimate_channel_weights(runner, ctx_bag)
        
        # Optimize trigger with deep analysis
        x, m = enhanced_solve_one(
            meta, 
            runner, 
            ctx_bag, 
            analyzer, 
            ch_weights, 
            budget_sec=GLOBAL_BUDGET_SEC
        )
        
        # Store results
        energy = np.linalg.norm(x)
        range_val = np.ptp(x)
        confidence = m.get('confidence', 0.5)
        logger.info(
            f"Model {meta.mid:02d} Final: proxy={m['proxy']:.6f}, energy={energy:.2e}, "
            f"range={range_val:.4f}, confidence={confidence:.4f}"
        )
        
        triggers[meta.mid] = x
        dt = time.time() - ms
        logger.info(f"Model time: {dt:.1f}s")
        
        # Store summary with deep metrics
        summary_rows.append({
            "mid": meta.mid,
            "loader": kind,
            "strategy": m['strategy'],
            "proxy": m['proxy'],
            "energy": energy,
            "range": range_val,
            "confidence": confidence,
            "time": dt,
            "fingerprint": m.get('fingerprint', {})
        })
    
    # Generate visualizations
    if SAVE_VIZ:
        logger.info("=== Generating Visualizations ===")
        save_trigger_plots(triggers, VIZ_DIR)
    
    # Write submission
    logger.info("=== Writing Submission ===")
    sub_path = write_submission(triggers, OUT_DIR)
    logger.info(f"Submission written to {sub_path}")
    
    # Generate analysis report
    logger.info("=== Generating Analysis Report ===")
    generate_report(summary_rows, OUT_DIR)
    
    # Final summary
    total_time = time.time() - start
    logger.info(f"Optimization completed in {total_time:.1f} seconds")
    logger.info(f"Average time per model: {total_time/len(metas):.1f} seconds")


def main():
    """Main entry point"""
    logger.info("Starting Advanced Deep Model Architecture Analysis Framework")
    run_enhanced_optimization()
    logger.info("Framework execution completed")


if __name__ == "__main__":
    main()





