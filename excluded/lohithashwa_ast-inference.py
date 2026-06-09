"""
# BirdCLEF 2025 - AST Model Inference Notebook

This notebook loads a trained Audio Spectrogram Transformer (AST) model and creates predictions for the BirdCLEF 2025 test data.
"""

import os
import gc
import warnings
import logging
import time
import numpy as np
import pandas as pd
import librosa
import cv2
from tqdm.auto import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import to_2tuple, trunc_normal_

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Configuration class
class CFG:
    # Paths
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    sample_submission = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/ast/pytorch/default/1/final_model.pth'
    
    # Audio parameters
    sample_rate = 32000
    duration = 5  # seconds
    
    # Mel spectrogram parameters - must match training settings
    n_mels = 128
    n_fft = 1024
    hop_length = 512
    fmin = 50
    fmax = 14000
    
    # Image parameters
    target_height = 224
    target_width = 224
    
    # AST model parameters
    fstride = 10
    tstride = 10
    patch_size = 16
    model_size = 'base224'
    
    # Inference parameters
    batch_size = 32
    threshold = 0.5  # Confidence threshold for positive predictions
    
    # Test-time augmentation
    use_tta = True  # Enable TTA for better performance
    tta_steps = 3
    
    # Debug
    debug_mode = False    # Set to True to process only a few soundscapes
    debug_count = 3       # Number of soundscapes to process in debug mode
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = CFG()
print(f"Using device: {cfg.device}")

# Define AST model architecture - must match the training model
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    """Audio Spectrogram Transformer model"""
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, model_size='base224'):
        super(ASTModel, self).__init__()
        
        # Override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        # Print model configuration for debugging
        print(f'AST Model: size={model_size}, input_fdim={input_fdim}, input_tdim={input_tdim}')
        print(f'frequency stride={fstride}, time stride={tstride}')
        
        # Load the appropriate ViT model
        if model_size == 'tiny224':
            try:
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            except RuntimeError:
                print("Falling back to vit_tiny_patch16_224")
                self.v = timm.create_model('vit_tiny_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'small224':
            try:
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            except RuntimeError:
                print("Falling back to vit_small_patch16_224")
                self.v = timm.create_model('vit_small_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base224':
            try:
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            except RuntimeError:
                print("Falling back to vit_base_patch16_224")
                self.v = timm.create_model('vit_base_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base384':
            try:
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            except RuntimeError:
                print("Falling back to vit_base_patch16_384")
                try:
                    self.v = timm.create_model('vit_base_patch16_384', pretrained=imagenet_pretrain)
                except RuntimeError:
                    print("Falling back to vit_base_patch16_224")
                    self.v = timm.create_model('vit_base_patch16_224', pretrained=imagenet_pretrain)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            
        # Check if model has distillation token
        self.has_dist_token = hasattr(self.v, 'dist_token')
        print(f"Model has distillation token: {self.has_dist_token}")
        
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), 
                                     nn.Linear(self.original_embedding_dim, label_dim))
        
        # Get shape automatically
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        print(f'number of patches={num_patches}')
            
        # Linear projection
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj
        
        # Positional embedding
        if imagenet_pretrain:
            # Get the positional embedding from model
            if self.has_dist_token:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            else:
                new_pos_embed = self.v.pos_embed[:, 1:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            
            # Cut or interpolate position embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                
            # Cut or interpolate position embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                
            # Flatten the position embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            
            # Concatenate with cls token and distillation token
            if self.has_dist_token:
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :1, :].detach(), new_pos_embed], dim=1))
        else:
            # Random initialization
            if self.has_dist_token:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 1, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)
        
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def forward(self, x):
        """
        :param x: Input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins)
        :return: prediction
        """
        # Input shape: (batch_size, time_frame_num, frequency_bins)
        x = x.unsqueeze(1)        # Add channel dimension: (B, 1, T, F)
        x = x.transpose(2, 3)     # -> (B, 1, F, T)
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        # Handle both model types (with and without distillation token)
        if self.has_dist_token:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
            
        x = self.v.norm(x)
        
        # Handle both model types for output
        if self.has_dist_token:
            x = (x[:, 0] + x[:, 1]) / 2  # Average of cls and dist token
        else:
            x = x[:, 0]  # Just use cls token
        
        x = self.mlp_head(x)
        return x

# Audio processing functions
def audio_to_melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    # Handle NaN values
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def apply_tta(mel_spec, step):
    """Apply test-time augmentation transformations"""
    if step == 0:
        # Original spectrogram
        return mel_spec
    elif step == 1:
        # Horizontal flip (time axis)
        return np.flip(mel_spec, axis=1)
    elif step == 2:
        # Vertical flip (frequency axis)
        return np.flip(mel_spec, axis=0)
    else:
        return mel_spec

def load_model_and_species():
    """Load trained model from checkpoint and get species list"""
    print(f"Loading model from {cfg.model_path}")
    
    # First load the sample submission to get expected column names
    sample_sub = pd.read_csv(cfg.sample_submission)
    species_columns = [col for col in sample_sub.columns if col != 'row_id']
    num_species = len(species_columns)
    print(f"Sample submission has {num_species} species columns")
    
    # Load checkpoint - handle different checkpoint formats
    try:
        checkpoint = torch.load(cfg.model_path, map_location=cfg.device)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract number of classes and label map if available in checkpoint
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
            print(f"Using num_classes from checkpoint: {num_classes}")
        else:
            num_classes = num_species
            print(f"Using num_classes from sample submission: {num_classes}")
        
        if 'label_map' in checkpoint:
            label_map = checkpoint['label_map']
            print(f"Found label_map in checkpoint with {len(label_map)} entries")
        else:
            # If no label map in checkpoint, use 1:1 mapping
            label_map = {i: species for i, species in enumerate(species_columns)}
            print("Created 1:1 label mapping")
        
        # Create model with the appropriate number of classes
        model = ASTModel(
            label_dim=num_classes,
            fstride=cfg.fstride,
            tstride=cfg.tstride,
            input_fdim=cfg.target_height,
            input_tdim=cfg.target_width,
            imagenet_pretrain=False,  # Not using pretrained weights for inference
            model_size=cfg.model_size
        )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded weights from 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded weights from 'state_dict'")
        else:
            print("WARNING: Could not find model weights in checkpoint")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, {}
    
    model = model.to(cfg.device)
    model.eval()
    
    # Create output mapping from model outputs to species indices
    if isinstance(label_map, dict):
        # If label_map maps from labels to indices, invert it
        if not all(isinstance(k, int) for k in label_map.keys()):
            index_to_species = {v: k for k, v in label_map.items()}
        else:
            index_to_species = label_map
    else:
        index_to_species = {i: species for i, species in enumerate(species_columns)}
    
    return model, index_to_species, species_columns

def predict_on_soundscape(model, audio_path, index_to_species, species_columns):
    """Process a soundscape file and generate predictions for each 5-second segment"""
    soundscape_id = Path(audio_path).stem
    
    try:
        # Load audio
        audio_data, _ = librosa.load(
            audio_path, 
            sr=cfg.sample_rate,
            res_type='kaiser_fast'  # Faster resampling
        )
        
        # Calculate total segments
        segment_samples = cfg.sample_rate * cfg.duration
        total_segments = int(len(audio_data) / segment_samples)
        
        # Initialize lists for results
        row_ids = []
        all_predictions = []
        
        # Process each 5-second segment
        for segment_idx in range(total_segments):
            # Extract segment
            start_sample = segment_idx * segment_samples
            end_sample = start_sample + segment_samples
            segment_audio = audio_data[start_sample:end_sample]
            
            # Create row ID in required format
            end_time_sec = (segment_idx + 1) * cfg.duration
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)
            
            # Initialize prediction dictionary with zeros for all species
            pred_dict = {species: 0.0 for species in species_columns}
            
            if cfg.use_tta:
                # Apply test-time augmentation
                segment_preds = []
                
                for tta_step in range(cfg.tta_steps):
                    # Process audio to mel spectrogram
                    mel_spec = audio_to_melspec(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_step)
                    
                    # Resize to model's expected input dimensions
                    mel_spec = cv2.resize(mel_spec, (cfg.target_width, cfg.target_height))
                    
                    # Convert to tensor
                    mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
                    mel_spec_tensor = mel_spec_tensor.to(cfg.device)
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = model(mel_spec_tensor)
                        probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                        segment_preds.append(probs)
                
                # Average predictions from all TTA steps
                final_preds = np.mean(segment_preds, axis=0)
            else:
                # Process audio without TTA
                mel_spec = audio_to_melspec(segment_audio, cfg)
                
                # Resize to model's expected input dimensions
                mel_spec = cv2.resize(mel_spec, (cfg.target_width, cfg.target_height))
                
                # Convert to tensor
                mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
                mel_spec_tensor = mel_spec_tensor.to(cfg.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(mel_spec_tensor)
                    final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
            
            # Map model outputs to species columns
            if np.isscalar(final_preds):
                # Handle case where there's only one class
                if 0 in index_to_species and index_to_species[0] in species_columns:
                    pred_dict[index_to_species[0]] = float(final_preds)
            else:
                # Map each model output to the corresponding species
                for i, prob in enumerate(final_preds):
                    if i in index_to_species and index_to_species[i] in species_columns:
                        pred_dict[index_to_species[i]] = float(prob)
            
            all_predictions.append(pred_dict)
        
        return row_ids, all_predictions
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def create_submission_file(all_row_ids, all_predictions, species_columns):
    """Create submission file in the required format"""
    print("Creating submission file...")
    
    # Initialize submission with row_ids
    submission_df = pd.DataFrame({'row_id': all_row_ids})
    
    # Add each species column
    for col in species_columns:
        # Initialize with zeros
        submission_df[col] = 0.0
        
        # Update with actual predictions
        for i, pred_dict in enumerate(all_predictions):
            if i < len(submission_df) and col in pred_dict:
                submission_df.loc[i, col] = pred_dict[col]
    
    # Save to CSV
    submission_df.to_csv('submission.csv', index=False, float_format='%.6f')
    print(f"Submission file created with {len(submission_df)} predictions")
    
    return submission_df

def run_inference():
    """Main inference function"""
    start_time = time.time()
    print(f"Starting inference using device: {cfg.device}")
    
    # Load model with species mapping
    model, index_to_species, species_columns = load_model_and_species()
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Find test files
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    
    if cfg.debug_mode:
        print(f"Debug mode: processing only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]
    
    print(f"Found {len(test_files)} test soundscapes")
    
    # Process each soundscape
    all_row_ids = []
    all_predictions = []
    
    for audio_path in tqdm(test_files, desc="Processing test soundscapes"):
        row_ids, predictions = predict_on_soundscape(model, str(audio_path), index_to_species, species_columns)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)
    
    # Create submission file
    submission_df = create_submission_file(all_row_ids, all_predictions, species_columns)
    
    # Verify submission against sample submission
    try:
        sample_sub = pd.read_csv(cfg.sample_submission)
        print(f"Sample submission shape: {sample_sub.shape}")
        print(f"Created submission shape: {submission_df.shape}")
        
        missing_cols = set(sample_sub.columns) - set(submission_df.columns)
        extra_cols = set(submission_df.columns) - set(sample_sub.columns)
        
        if missing_cols:
            print(f"WARNING: Missing columns in submission: {missing_cols}")
        if extra_cols:
            print(f"WARNING: Extra columns in submission: {extra_cols}")
        
        if set(submission_df.columns) == set(sample_sub.columns):
            print("✓ Submission has the correct columns")
        
    except Exception as e:
        print(f"Error verifying submission: {e}")
    
    print(f"Inference completed in {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

