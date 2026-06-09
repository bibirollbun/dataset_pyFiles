!pip install umap plotly umap-learn


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap.umap_ as umap
from tqdm import tqdm
from pathlib import Path
import librosa
import cv2
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar as np_scalar
from numpy import dtype as np_dtype
from numpy.dtypes import Float64DType

# Add numpy scalar and dtype to safe globals
add_safe_globals([np_scalar, np_dtype, Float64DType])

# Define CFG class for safe loading
class CFG:
    def __init__(self):
        pass

# Add CFG to safe globals
add_safe_globals([CFG])

class BaselineModel(nn.Module):
    def __init__(self, model_name, in_channels=1):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=in_channels,
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        
        backbone_out = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        
    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        return features

    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict that handles different state dict formats"""
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
                
        # Remove classifier weights from state_dict
        state_dict_no_clf = {k: v for k, v in state_dict.items() 
                            if not k.startswith('classifier.')}
        return super().load_state_dict(state_dict_no_clf, strict=False)

def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_file(audio_path, cfg):
    """Process a single audio file to extract features"""
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        
        # Take center 5 seconds if longer
        target_samples = cfg.FS * cfg.WINDOW_SIZE
        if len(audio_data) > target_samples:
            start = (len(audio_data) - target_samples) // 2
            audio_data = audio_data[start:start + target_samples]
        else:
            # Pad if shorter
            audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
        
        mel_spec = audio2melspec(audio_data, cfg)
        
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
        return mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def generate_embeddings(model, audio_files, cfg, device):
    """Generate embeddings for a list of audio files"""
    model.eval()
    embeddings = []
    processed_files = []
    labels = []
    
    for audio_file in tqdm(audio_files, desc="Generating embeddings"):
        try:
            # Extract label from filepath (assuming format: species/file.ogg)
            label = Path(audio_file).parent.name
            
            # Process audio to mel spectrogram
            mel_spec = process_audio_file(audio_file, cfg)
            if mel_spec is None:
                continue
                
            # Convert to tensor
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mel_spec = mel_spec.to(device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = model(mel_spec)
                embedding = embedding.cpu().numpy()
            
            embeddings.append(embedding[0])
            processed_files.append(audio_file)
            labels.append(label)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    return np.array(embeddings), processed_files, labels

def plot_embeddings(embeddings, labels, output_path):
    """Create visualization of embeddings"""
    print(f"Processing embeddings shape: {embeddings.shape}")
    
    # Save original embeddings
    embeddings_save_path = output_path.replace('.html', '_embeddings.npz')
    print(f"Saving original embeddings to {embeddings_save_path}...")
    np.savez_compressed(
        embeddings_save_path,
        embeddings=embeddings,
        labels=np.array(labels)
    )
    
    # Reduce dimensionality
    print("Fitting UMAP...")
    reducer = umap.UMAP(n_components=3, random_state=42, metric='cosine', min_dist=0.1)
    embeddings_umap = reducer.fit_transform(embeddings)
    
    # Save UMAP reduced embeddings
    umap_save_path = output_path.replace('.html', '_umap.npz')
    print(f"Saving UMAP reduced embeddings to {umap_save_path}...")
    np.savez_compressed(
        umap_save_path,
        embeddings_umap=embeddings_umap,
        labels=np.array(labels)
    )
    
    print("Creating plotly figure...")
    # Create figure
    fig = go.Figure()
    
    # Add traces for embeddings
    unique_labels = sorted(list(set(labels)))
    print(f"Processing {len(unique_labels)} unique labels...")
    
    # Create a color map
    colors = px.colors.qualitative.Set3 * (len(unique_labels) // len(px.colors.qualitative.Set3) + 1)
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    for label in tqdm(unique_labels, desc="Processing labels"):
        mask = [l == label for l in labels]
        if sum(mask) > 0:  # Only add trace if there are points for this label
            fig.add_trace(
                go.Scatter3d(
                    x=embeddings_umap[mask, 0],
                    y=embeddings_umap[mask, 1],
                    z=embeddings_umap[mask, 2],
                    mode='markers',
                    name=label,
                    showlegend=True,
                    marker=dict(
                        size=4,
                        color=color_map[label],
                        opacity=0.7
                    )
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Embedding Space Visualization",
        height=800,
        width=1000,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )
    
    print(f"Saving plot to {output_path}...")
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")

def main():
    # Configuration
    class CFG:  #               /kaggle/input/pub-bird25-b-422-ppv15-v2-s-focallossbce
        baseline_model_path = "/kaggle/input/pub-bird25-b-422-ppv15-v2-s-focallossbce"  # Update this path
        train_audio_dir = "/kaggle/input/birdclef-2025/train_audio"                    # Update this path
        output_dir = "/kaggle/working/embeddings_visualization"
        output_path = os.path.join(output_dir, "embedding_visualization.html")
        
        # Model settings
        model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
        in_channels = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Audio processing settings
        FS = 32000
        WINDOW_SIZE = 5
        N_FFT = 2048
        HOP_LENGTH =128
        N_MELS = 512
        FMIN = 20
        FMAX = 16000
        TARGET_SHAPE = (256, 256)
        max_samples_per_class = 100
        min_samples_per_class = 10
    
    cfg = CFG()
    
    # Create output directory if it doesn't exist
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    models = []
    model_files = list(Path(cfg.baseline_model_path).glob('*.pth'))
    
    if not model_files:
        print(f"No model files found in {cfg.baseline_model_path}")
        return
        
    print(f"Found {len(model_files)} model files")
    
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            model = BaselineModel(cfg.model_name, in_channels=cfg.in_channels).to(cfg.device)
            checkpoint = torch.load(str(model_path), map_location=cfg.device)
            model.load_state_dict(checkpoint)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue
    
    if not models:
        print("No models could be loaded successfully!")
        return
        
    print(f"Successfully loaded {len(models)} models")
    
    # Get audio files
    print("Collecting audio files...")
    audio_files = []
    for species_dir in Path(cfg.train_audio_dir).iterdir():
        if species_dir.is_dir():
            files = list(species_dir.glob("*.ogg"))
            if len(files) >= cfg.min_samples_per_class:
                # Randomly sample if more than max_samples_per_class
                if len(files) > cfg.max_samples_per_class:
                    files = np.random.choice(files, cfg.max_samples_per_class, replace=False)
                audio_files.extend(files)
    
    # Generate embeddings using ensemble of models
    print("Generating embeddings...")
    all_embeddings = []
    processed_files = []
    labels = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Extract label from filepath
            label = audio_file.parent.name
            
            # Process audio to mel spectrogram
            audio_data, _ = librosa.load(str(audio_file), sr=cfg.FS)
            
            # Take center 5 seconds if longer
            target_samples = cfg.FS * cfg.WINDOW_SIZE
            if len(audio_data) > target_samples:
                start = (len(audio_data) - target_samples) // 2
                audio_data = audio_data[start:start + target_samples]
            else:
                # Pad if shorter
                audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
            
            mel_spec = process_audio_file(str(audio_file), cfg)
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mel_spec = mel_spec.to(cfg.device)
            
            # Get embeddings from all models
            file_embeddings = []
            with torch.no_grad():
                for model in models:
                    embedding = model(mel_spec)
                    file_embeddings.append(embedding.cpu().numpy())
            
            # Average embeddings from all models
            avg_embedding = np.mean(file_embeddings, axis=0)
            all_embeddings.append(avg_embedding[0])
            processed_files.append(str(audio_file))
            labels.append(label)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    embeddings = np.array(all_embeddings)
    
    # Create visualization
    print("Creating visualization...")
    plot_embeddings(embeddings, labels, cfg.output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()





