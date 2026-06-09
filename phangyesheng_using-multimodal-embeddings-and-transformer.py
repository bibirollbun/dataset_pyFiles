import os
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Fix for HuggingFace tokenizers + DataLoader multiprocessing conflicts on Kaggle
# This prevents "current process got forked" warnings and validation hangs
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print("âœ“ All packages imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Kaggle data paths
DATA_DIR = '/kaggle/input/petfinder-adoption-prediction'

TRAIN_CSV = f'{DATA_DIR}/train/train.csv'
TEST_CSV = f'{DATA_DIR}/test/test.csv'

# Image directories
TRAIN_IMG_DIR = f'{DATA_DIR}/train_images'
TEST_IMG_DIR = f'{DATA_DIR}/test_images'

# Google Vision metadata (JSON files)
TRAIN_METADATA_DIR = f'{DATA_DIR}/train_metadata'
TEST_METADATA_DIR = f'{DATA_DIR}/test_metadata'

# Google Sentiment (JSON files)
TRAIN_SENTIMENT_DIR = f'{DATA_DIR}/train_sentiment'
TEST_SENTIMENT_DIR = f'{DATA_DIR}/test_sentiment'

# Output directory for processed features (Kaggle working directory is writable)
OUTPUT_DIR = '/kaggle/working'

BREED_LABELS_CSV = r'C:\Users\Ye Sheng\Multimodal classification project\petfinder_multimodal_project\kaggle\input\breed_labels.csv'

# World-record hyperparameters
CONFIG = {
    # Model architecture
    'd_model': 384,
    'n_heads': 8,
    'n_layers': 6,
    'd_ff': 1536,
    'dropout': 0.4,  # Increased from 0.15 to 0.3 to prevent overfitting
    'num_classes': 5,
    'use_ordinal': True,
    'use_type_embeddings': True,
    'use_positional_encoding': True,
    'use_global_pooling': True,  # Use global pooling instead of just CLS tokens
    'use_learnable_fusion': True,  # Use learnable attention for fusion
    'use_cross_modal_attention': True,  # Use multi-head cross-attention between modalities

    # Backbone models
    'image_model': 'vit_base_patch16_224',
    'text_model': 'all-MiniLM-L6-v2',
    'freeze_backbones': True,

    # Data
    'val_split': 0.10,  # 10% validation split
    'train_batch_size': 32,
    'val_batch_size': 64,
    'num_workers': 0,  # Set to 0 to avoid multiprocessing issues on Kaggle

    # Training
    'epochs': 70,
    'lr': 0.00003,  # Increased from 3e-6 to 1e-4 for faster convergence
    'weight_decay': 0.2,  # Increased to 0.3 for stronger L2 regularization
    'warmup_epochs': 10,  # Increased from 3 for more stable warmup
    'T_max': 50,  # Increased from 30 for longer cosine cycle
    'eta_min': 0.00001,  # Increased from 1e-6 for higher minimum LR
    'gradient_clip_norm': 0.5,
    'accumulate_grad_batches': 2,

    # Early stopping
    'early_stopping_patience': 100,  # Reduced to stop earlier if overfitting persists
    'early_stopping_min_delta': 0.001,

    # Loss
    'use_class_weights': True,
    'label_smoothing': 0.15,  # Increased to 0.15 for stronger regularization

    # Feature Dropout (Regularization)
    'rescuer_feature_dropout': 0.45,  # Randomly mask 100% of rescuer stats to -100 during training (test without rescuer features)

    # Feature Engineering
    'use_tfidf_features': False,  # Use TF-IDF text features in tabular (redundant with Sentence-BERT)

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print("Configuration loaded:")
print(f"  Model: {CONFIG['d_model']}-dim, {CONFIG['n_layers']} layers, {CONFIG['n_heads']} heads")
print(f"  Training: {CONFIG['epochs']} max epochs, LR={CONFIG['lr']:.2e}")
print(f"  Device: {CONFIG['device']}")
print(f"  Validation split: {CONFIG['val_split']*100:.0f}%")


"""
Simple experiment tracker to log metrics and generate visualizations.
Replaces WandB for Kaggle notebook environment.
"""

class ExperimentTracker:
    """Track training metrics and generate plots."""

    def __init__(self, experiment_name='petfinder_multimodal'):
        self.experiment_name = experiment_name
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_qwk': [],
            'train_acc': [],
            'val_loss': [],
            'val_qwk': [],
            'val_acc': [],
            'lr': []
        }
        self.best_qwk = -1.0
        self.best_epoch = 0

    def log_epoch(self, epoch, train_loss, train_qwk, train_acc, val_loss, val_qwk, val_acc, lr):
        """Log metrics for one epoch."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_qwk'].append(train_qwk)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_qwk'].append(val_qwk)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)

        if val_qwk > self.best_qwk:
            self.best_qwk = val_qwk
            self.best_epoch = epoch

    def plot_training_curves(self, save_path='training_curves.png'):
        """Generate training visualization: QWK, Loss, Accuracy, Learning Rate."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: QWK (Primary Metric)
        ax = axes[0, 0]
        ax.plot(self.history['epoch'], self.history['train_qwk'], label='Train QWK', marker='o', markersize=2, linewidth=2)
        ax.plot(self.history['epoch'], self.history['val_qwk'], label='Val QWK', marker='s', markersize=2, linewidth=2)
        ax.axvline(self.best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({self.best_epoch})')
        ax.axhline(self.best_qwk, color='green', linestyle='--', alpha=0.5, label=f'Best QWK ({self.best_qwk:.4f})')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('QWK Score', fontsize=11)
        ax.set_title('Quadratic Weighted Kappa (Primary Metric)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss
        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss', marker='o', markersize=2, linewidth=2)
        ax.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss', marker='s', markersize=2, linewidth=2)
        ax.axvline(self.best_epoch, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Accuracy
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['train_acc'], label='Train Accuracy', marker='o', markersize=2, linewidth=2)
        ax.plot(self.history['epoch'], self.history['val_acc'], label='Val Accuracy', marker='s', markersize=2, linewidth=2)
        ax.axvline(self.best_epoch, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Learning Rate
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], self.history['lr'], marker='o', markersize=2, color='purple', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate (log scale)', fontsize=11)
        ax.set_title('Learning Rate Schedule (Warmup + Cosine)', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'{self.experiment_name} - Training History', fontsize=16, y=1.00, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"âœ“ Training curves saved to {save_path}")

    def print_summary(self):
        """Print training summary statistics."""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Total Epochs: {len(self.history['epoch'])}")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Validation QWK: {self.best_qwk:.4f}")
        print(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}")
        print(f"Final Train QWK: {self.history['train_qwk'][-1]:.4f}")
        print(f"Final Val QWK: {self.history['val_qwk'][-1]:.4f}")
        print("="*60 + "\n")

tracker = ExperimentTracker('petfinder_world_record')
print("âœ“ Experiment tracker initialized")


# =============================================================================
# CELL 3.5: Attention Visualization Functions
# =============================================================================
"""
Visualize cross-modal attention maps and feature importance.
Helps understand what the model is learning.
"""

def visualize_attention_weights(
    model,
    val_loader,
    device,
    save_dir,
    num_samples=100,
    epoch=None,
    feature_names=None
):
    """
    Visualize cross-modal attention and fusion weights.

    Args:
        model: The trained model
        val_loader: Validation dataloader
        device: torch device
        save_dir: Directory to save visualizations
        num_samples: Number of validation samples to analyze
        epoch: Current epoch number (for filename)
        feature_names: List of feature names for tabular features (optional)
    """
    model.eval()

    # Collect attention weights
    cross_attn_weights_list = []
    fusion_weights_list = []
    tabular_features_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx * val_loader.batch_size >= num_samples:
                break

            # Move batch to device (use correct keys from PetFinderDataset)
            img_emb = batch['image_embeddings'].to(device)
            txt_emb = batch['text_embeddings'].to(device)
            tabular = batch['tabular_tokens'].to(device)

            # Forward pass
            outputs = model(img_emb, txt_emb, tabular)

            # Collect attention weights
            if 'cross_attention_weights' in outputs:
                # Shape is already (batch, 3, 3) - PyTorch averages across heads automatically
                attn = outputs['cross_attention_weights'].cpu().numpy()
                cross_attn_weights_list.append(attn)

            if 'fusion_weights' in outputs:
                # (batch, 3, 1) -> (batch, 3)
                fusion = outputs['fusion_weights'].squeeze(-1).cpu().numpy()
                fusion_weights_list.append(fusion)

            # Collect tabular features for importance analysis
            tabular_features_list.append(tabular.cpu().numpy())

    # Average attention weights across all samples
    modality_names = ['Image', 'Text', 'Tabular']

    # 1. Cross-Modal Attention Heatmap
    if cross_attn_weights_list:
        avg_cross_attn = np.concatenate(cross_attn_weights_list, axis=0).mean(axis=0)  # (3, 3)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            avg_cross_attn,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=modality_names,
            yticklabels=modality_names,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax,
            vmin=0,
            vmax=1
        )
        ax.set_xlabel('Key (Attended To)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query (Attending From)', fontsize=12, fontweight='bold')

        title = f'Cross-Modal Attention Map'
        if epoch is not None:
            title += f' (Epoch {epoch})'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        # Add interpretation text
        fig.text(
            0.5, -0.05,
            'Higher values = stronger attention.\nRows show how much each modality attends to others (columns).',
            ha='center',
            fontsize=10,
            style='italic'
        )

        filename = f'cross_attention_map_epoch{epoch}.png' if epoch else 'cross_attention_map.png'
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"âœ“ Cross-attention map saved to {save_path}")

        # Print insights
        print("\n" + "="*60)
        print("CROSS-MODAL ATTENTION INSIGHTS")
        print("="*60)
        for i, query_mod in enumerate(modality_names):
            print(f"\n{query_mod} attends to:")
            for j, key_mod in enumerate(modality_names):
                weight = avg_cross_attn[i, j]
                print(f"  - {key_mod:8s}: {weight:.3f}")

    # 2. Fusion Weights Bar Chart
    if fusion_weights_list:
        avg_fusion = np.concatenate(fusion_weights_list, axis=0).mean(axis=0)  # (3,)
        std_fusion = np.concatenate(fusion_weights_list, axis=0).std(axis=0)  # (3,)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(modality_names, avg_fusion, yerr=std_fusion, color=colors, alpha=0.7, capsize=5)

        ax.set_ylabel('Fusion Weight', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])

        title = f'Modality Fusion Weights'
        if epoch is not None:
            title += f' (Epoch {epoch})'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar, val, std in zip(bars, avg_fusion, std_fusion):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + std,
                f'{val:.3f}Â±{std:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax.grid(True, alpha=0.3, axis='y')

        filename = f'fusion_weights_epoch{epoch}.png' if epoch else 'fusion_weights.png'
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"âœ“ Fusion weights plot saved to {save_path}")

        # Print insights
        print("\n" + "="*60)
        print("MODALITY FUSION INSIGHTS")
        print("="*60)
        for mod, weight, std in zip(modality_names, avg_fusion, std_fusion):
            print(f"{mod:8s}: {weight:.3f} Â± {std:.3f} (weight Â± std)")
        print("\nHigher weights = more important for final prediction")

    # 3. Tabular Feature Importance (gradient-based)
    # Compute how much each feature affects model predictions using gradients
    print("  Computing gradient-based feature importance...")

    model.eval()
    gradient_importance_list = []

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx * val_loader.batch_size >= num_samples:
            break

        # Move batch to device
        img_emb = batch['image_embeddings'].to(device)
        txt_emb = batch['text_embeddings'].to(device)
        tabular = batch['tabular_tokens'].to(device).requires_grad_(True)

        # Forward pass
        outputs = model(img_emb, txt_emb, tabular)
        logits = outputs['logits']

        # Sum of all logits as the score (simpler than max probability)
        # This measures total model output sensitivity to each feature
        score = logits.sum()

        # Backward pass
        score.backward()

        # Feature importance = mean absolute gradient
        # This directly measures how sensitive predictions are to each feature
        if tabular.grad is not None:
            grad = tabular.grad.detach().cpu().numpy()  # (batch, num_features)

            # Mean absolute gradient across batch
            batch_importance = np.abs(grad).mean(axis=0)  # (num_features,)
            gradient_importance_list.append(batch_importance)

    if gradient_importance_list:
        # Average importance across all samples
        feature_importance = np.mean(gradient_importance_list, axis=0)

        # Get top 20 features
        num_features = len(feature_importance)
        top_k = min(20, num_features)
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_importance = feature_importance[top_indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        # Use actual feature names if provided, otherwise generic names
        if feature_names is not None and len(feature_names) == num_features:
            plot_feature_names = [feature_names[i] for i in top_indices]
        else:
            plot_feature_names = [f'Feature {i}' for i in top_indices]

        ax.barh(range(len(top_importance)), top_importance, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_importance)))
        ax.set_yticklabels(plot_feature_names, fontsize=9)
        ax.set_xlabel('Gradient Magnitude (Mean |âˆ‚loss/âˆ‚feature|)', fontsize=12, fontweight='bold')

        title = f'Top {top_k} Tabular Features by Learned Importance'
        if epoch is not None:
            title += f' (Epoch {epoch})'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()  # Highest at top

        filename = f'tabular_importance_epoch{epoch}.png' if epoch else 'tabular_importance.png'
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"âœ“ Tabular feature importance saved to {save_path}")

        print("\n" + "="*60)
        print(f"TOP {top_k} TABULAR FEATURES (GRADIENT-BASED)")
        print("="*60)
        print("Shows how much each feature impacts model predictions")
        print("(Higher = model relies on this feature more)")
        print("-"*60)
        for i, (idx, feat_name, importance) in enumerate(zip(top_indices, plot_feature_names, top_importance), 1):
            print(f"{i:2d}. {feat_name:40s}: {importance:.4f}")

    print("\n" + "="*60)
    print("âœ“ All attention visualizations complete!")
    print("="*60 + "\n")

print("âœ“ Attention visualization functions loaded")


"""
Transformer building blocks: positional encoding and type embeddings.
These help the model understand token positions and modality types.
"""

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer sequences."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class TypeEmbedding(nn.Module):
    """Type embeddings to distinguish image, text, and tabular modalities."""

    def __init__(self, d_model: int):
        super().__init__()
        # 3 modality types: 0=image, 1=text, 2=tabular
        self.type_embeddings = nn.Embedding(3, d_model)

    def forward(self, tokens: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        """Add type embeddings to tokens."""
        type_embs = self.type_embeddings(modality_ids)
        return tokens + type_embs

print("âœ“ Positional and Type Embeddings defined")


"""
Modality-specific embedding layers that project each modality to d_model dimensions.
Each modality gets its own CLS token for global representation.
"""

class ModalityEmbedding(nn.Module):
    """Embedding layer for different input modalities."""

    def __init__(self, input_dim: int, d_model: int, modality_type: str):
        super().__init__()
        self.modality_type = modality_type
        self.d_model = d_model

        # Projection layer depends on modality type
        if modality_type == 'image':
            # ViT-B/16 outputs 768-dim embeddings
            self.projection = nn.Linear(input_dim, d_model)
        elif modality_type == 'text':
            # MiniLM outputs 384-dim embeddings
            self.projection = nn.Linear(input_dim, d_model)
        elif modality_type == 'tabular':
            # Each tabular token is 1-dimensional (will be expanded in forward)
            self.projection = nn.Linear(1, d_model)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")

        # Learnable CLS token for this modality
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to d_model and prepend CLS token.

        Input shapes:
            - Image: (batch_size, 768)
            - Text: (batch_size, 384)
            - Tabular: (batch_size, num_tokens)

        Output shape: (batch_size, seq_len+1, d_model)
        """
        batch_size = x.shape[0]

        if self.modality_type in ['image', 'text']:
            # Single embedding per sample, add sequence dimension
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
            projected = self.projection(x)
        else:  # tabular
            # Multiple tokens per sample
            x = x.unsqueeze(-1)  # (batch_size, num_tokens, 1)
            projected = self.projection(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded = torch.cat([cls_tokens, projected], dim=1)

        return self.layer_norm(embedded)

print("âœ“ Modality Embeddings defined")



"""
Main multimodal transformer architecture.
Fuses image, text, and tabular data through cross-attention.
Uses ordinal regression for adoption speed prediction.
"""

class MultimodalTransformer(nn.Module):
    """State-of-the-art multimodal transformer for PetFinder."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.d_model = config['d_model']
        self.num_classes = config['num_classes']
        self.use_ordinal = config['use_ordinal']
        self.use_type_embeddings = config['use_type_embeddings']
        self.use_positional_encoding = config['use_positional_encoding']
        self.use_global_pooling = config.get('use_global_pooling', True)
        self.use_learnable_fusion = config.get('use_learnable_fusion', True)
        self.use_cross_modal_attention = config.get('use_cross_modal_attention', True)

        # Modality-specific embeddings
        # Image: 768-dim (ViT-B/16), Text: 384-dim (MiniLM), Tabular: varies
        self.image_embedding = ModalityEmbedding(768, self.d_model, 'image')
        self.text_embedding = ModalityEmbedding(384, self.d_model, 'text')
        self.tabular_embedding = ModalityEmbedding(1, self.d_model, 'tabular')

        # Positional encoding
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(self.d_model, max_len=200)

        # Type embeddings
        if self.use_type_embeddings:
            self.type_embedding = TypeEmbedding(self.d_model)

        # Transformer encoder (pre-norm architecture for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config['n_heads'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm is more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['n_layers'])

        # Fusion layers
        if self.use_cross_modal_attention:
            # Multi-head cross-attention between modalities
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=config['n_heads'],
                dropout=config['dropout'],
                batch_first=True
            )

        if self.use_learnable_fusion:
            # Learnable attention weights for modality fusion
            self.fusion_attention = nn.Linear(self.d_model, 1)

        # Classification head
        if self.use_ordinal:
            # Ordinal regression: predict K-1 thresholds for K classes
            self.classifier = nn.Linear(self.d_model, self.num_classes - 1)
        else:
            # Standard classification
            self.classifier = nn.Linear(self.d_model, self.num_classes)

        self.dropout = nn.Dropout(config['dropout'])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def create_modality_ids(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create modality type IDs for type embeddings.

        Sequence structure:
        [IMG_CLS, img_token, TXT_CLS, txt_token, TAB_CLS, tab_tokens...]

        Type IDs: 0=image, 1=text, 2=tabular
        """
        modality_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        modality_ids[:, 0:2] = 0  # Image CLS + token
        modality_ids[:, 2:4] = 1  # Text CLS + token
        modality_ids[:, 4:] = 2   # Tabular CLS + tokens
        return modality_ids

    def forward(self, image_embs, text_embs, tabular_tokens):
        """
        Forward pass through multimodal transformer.

        Args:
            image_embs: (batch_size, 768) - ViT embeddings
            text_embs: (batch_size, 384) - MiniLM embeddings
            tabular_tokens: (batch_size, num_tokens) - processed tabular features

        Returns:
            dict with 'logits' key
        """
        batch_size = image_embs.shape[0]
        device = image_embs.device

        # Embed each modality (adds CLS tokens)
        img_tokens = self.image_embedding(image_embs)      # (batch, 2, d_model)
        txt_tokens = self.text_embedding(text_embs)        # (batch, 2, d_model)
        tab_tokens = self.tabular_embedding(tabular_tokens)  # (batch, num_tab+1, d_model)

        # Concatenate all tokens into single sequence
        all_tokens = torch.cat([img_tokens, txt_tokens, tab_tokens], dim=1)

        # Add positional encoding
        if self.use_positional_encoding:
            all_tokens = self.pos_encoding(all_tokens)

        # Add type embeddings
        if self.use_type_embeddings:
            seq_len = all_tokens.shape[1]
            modality_ids = self.create_modality_ids(batch_size, seq_len, device)
            all_tokens = self.type_embedding(all_tokens, modality_ids)

        # Apply dropout
        all_tokens = self.dropout(all_tokens)

        # Transformer encoding (cross-attention between modalities)
        encoded = self.transformer(all_tokens)

        # Extract modality representations
        if self.use_global_pooling:
            # Use global pooling over each modality's tokens (better utilization)
            # Sequence structure: [IMG_CLS, img_tok, TXT_CLS, txt_tok, TAB_CLS, tab_tokens...]
            img_seq = encoded[:, 0:2]    # Image tokens (CLS + embedding)
            txt_seq = encoded[:, 2:4]    # Text tokens (CLS + embedding)
            tab_seq = encoded[:, 4:]     # Tabular tokens (CLS + all features)

            # Global average pooling for each modality
            img_repr = torch.mean(img_seq, dim=1)  # (batch, d_model)
            txt_repr = torch.mean(txt_seq, dim=1)  # (batch, d_model)
            tab_repr = torch.mean(tab_seq, dim=1)  # (batch, d_model)
        else:
            # Use only CLS tokens (original approach)
            img_repr = encoded[:, 0]   # Image CLS
            txt_repr = encoded[:, 2]   # Text CLS
            tab_repr = encoded[:, 4]   # Tabular CLS

        # Stack modality representations
        modality_features = torch.stack([img_repr, txt_repr, tab_repr], dim=1)  # (batch, 3, d_model)

        # Apply cross-modal attention if enabled
        cross_attention_weights = None
        if self.use_cross_modal_attention:
            # Multi-head attention between modalities
            attended_features, cross_attention_weights = self.cross_modal_attention(
                query=modality_features,
                key=modality_features,
                value=modality_features
            )
            modality_features = attended_features  # (batch, 3, d_model)

        # Fuse modalities
        fusion_weights = None
        if self.use_learnable_fusion:
            # Learnable attention weights
            attention_scores = self.fusion_attention(modality_features)  # (batch, 3, 1)
            fusion_weights = F.softmax(attention_scores, dim=1)  # (batch, 3, 1)
            fused_features = torch.sum(modality_features * fusion_weights, dim=1)  # (batch, d_model)
        else:
            # Simple mean pooling
            fused_features = torch.mean(modality_features, dim=1)  # (batch, d_model)

        # Final classification
        logits = self.classifier(fused_features)

        # Return outputs with optional attention weights for visualization
        outputs = {'logits': logits}
        if cross_attention_weights is not None:
            outputs['cross_attention_weights'] = cross_attention_weights  # (batch, 3, 3)
        if fusion_weights is not None:
            outputs['fusion_weights'] = fusion_weights  # (batch, 3, 1)

        return outputs

    def predict_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to class probabilities.

        For ordinal regression:
            - Logits represent threshold values g_k = P(y >= k+1)
            - Convert to class probabilities: P(y = k)
        """
        if self.use_ordinal:
            # Apply sigmoid to get threshold probabilities
            greater_probs = torch.sigmoid(logits)  # (batch, num_classes-1)

            # Convert to class probabilities
            probs = []
            # P(y=0) = 1 - P(y>=1)
            probs.append(1 - greater_probs[:, 0:1])
            # P(y=k) = P(y>=k) - P(y>=k+1) for k=1..K-2
            for k in range(1, self.num_classes - 1):
                class_prob = greater_probs[:, k-1:k] - greater_probs[:, k:k+1]
                probs.append(class_prob)
            # P(y=K-1) = P(y>=K-1)
            probs.append(greater_probs[:, -1:])

            probs = torch.cat(probs, dim=1)

            # Ensure probabilities are valid
            probs = probs.clamp(min=0.0, max=1.0)
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

            return probs
        else:
            return F.softmax(logits, dim=-1)

print("âœ“ Multimodal Transformer architecture defined")
print(f"  Parameters: ~{sum(p.numel() for p in MultimodalTransformer(CONFIG).parameters()) / 1e6:.1f}M")


"""
Ordinal regression loss that respects the ordered nature of adoption speed.
Better than standard cross-entropy for ordinal targets.
"""

class OrdinalLoss(nn.Module):
    """Ordinal regression loss with optional class weights and label smoothing."""

    def __init__(self, num_classes: int, class_weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, cumulative_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal regression loss.

        Args:
            cumulative_logits: (batch_size, num_classes-1) - predicted thresholds
            targets: (batch_size,) - true class labels [0, 1, 2, 3, 4]

        Returns:
            Loss scalar
        """
        batch_size = targets.shape[0]
        device = cumulative_logits.device

        # Create binary targets for each threshold
        # For class y, we want: P(y >= k+1) = 1 if y >= k+1, else 0
        binary_targets = torch.zeros(batch_size, self.num_thresholds, device=device)

        for k in range(self.num_thresholds):
            binary_targets[:, k] = (targets >= k + 1).float()

        # Apply label smoothing to prevent overconfident predictions
        if self.label_smoothing > 0:
            # Smooth targets: 1 â†’ (1 - Îµ), 0 â†’ Îµ
            binary_targets = binary_targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Binary cross-entropy for each threshold
        bce_loss = F.binary_cross_entropy_with_logits(
            cumulative_logits, binary_targets, reduction='none'
        )

        # Apply class weights if provided (helps with class imbalance)
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]
            bce_loss = bce_loss * sample_weights.unsqueeze(1)

        # Average over thresholds, then over batch
        threshold_loss = bce_loss.mean(dim=1)
        return threshold_loss.mean()


def compute_class_weights(targets: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    Higher weight for rare classes.
    """
    class_counts = np.bincount(targets, minlength=num_classes)
    total_count = len(targets)
    weights = total_count / (num_classes * class_counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)

print("âœ“ Ordinal Loss defined")


"""
Advanced feature engineering matching enhanced_features.yaml approach.
Includes TF-IDF text features, rescuer statistics, and feature interactions.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering to match XGBoost baseline performance.

    Features:
    - TF-IDF + SVD for text features (Description, Name)
    - Rescuer count (no adoption speed stats to avoid leakage)
    - Breed name categorical encoding
    - Google Vision API metadata (annotations, colors, crops)
    - Google Sentiment API features
    - Image size statistics (file size, dimensions)
    - Feature interactions (age x type, breed x color, media)

    Performance:
    - Multithreaded I/O operations (configurable via n_jobs parameter)
    - Default: 4 workers (Kaggle-safe, ~3-4x faster than single-threaded)
    - Progress bars for all slow operations
    """

    def __init__(self, tfidf_max_features=2000, svd_components=20, random_state=42,
                 metadata_dir=None, sentiment_dir=None, images_dir=None, breed_labels_path=None,
                 n_jobs=1, use_tfidf=True):
        self.tfidf_max_features = tfidf_max_features
        self.svd_components = svd_components
        self.random_state = random_state
        self.metadata_dir = metadata_dir
        self.sentiment_dir = sentiment_dir
        self.images_dir = images_dir
        self.breed_labels_path = breed_labels_path
        self.n_jobs = n_jobs  # Number of parallel workers for I/O operations
        self.use_tfidf = use_tfidf  # Whether to use TF-IDF features (can be redundant with Sentence-BERT)
        self.tfidf_transformers = {}
        self.svd_transformers = {}
        self.breed_labels = None
        self.breed_encoders = {}  # For encoding breed names to integers
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit all feature transformers on training data."""
        print("  Fitting advanced feature engineering...")

        # 1. Text features (TF-IDF + SVD)
        self._fit_text_features(df)

        # 2. Rescuer features (count only - no adoption speed)
        self._fit_rescuer_features(df)

        # 3. Breed features (load breed labels and create encoders)
        self._fit_breed_features(df)

        self.is_fitted = True
        print("  Advanced feature engineering fitted!")
        return self

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def _fit_text_features(self, df: pd.DataFrame):
        """Fit TF-IDF transformers for text features."""
        if not self.use_tfidf:
            print("    âš  TF-IDF features DISABLED (use_tfidf=False)")
            print("    âœ“ Using only text statistics (length, word count, etc.)")
            print("    âœ“ Text semantics handled by Sentence-BERT (no redundancy)")
            return

        text_columns = ['Description', 'Name']

        for col in text_columns:
            if col in df.columns:
                texts = df[col].fillna('').astype(str).apply(self._clean_text)

                # TF-IDF
                tfidf = TfidfVectorizer(
                    max_features=self.tfidf_max_features,
                    min_df=1,
                    max_df=0.95,
                    stop_words='english',
                    ngram_range=(1, 2),
                    strip_accents='unicode'
                )
                tfidf_matrix = tfidf.fit_transform(texts)

                # SVD
                svd = TruncatedSVD(n_components=self.svd_components, random_state=self.random_state)
                svd.fit(tfidf_matrix)

                self.tfidf_transformers[col] = tfidf
                self.svd_transformers[col] = svd

    def _fit_rescuer_features(self, df: pd.DataFrame):
        """Fit rescuer statistics including target encoding."""
        if 'RescuerID' not in df.columns:
            return

        # Compute rescuer count (matches XGBoost baseline)
        rescuer_count = df.groupby('RescuerID')['PetID'].count().reset_index()
        rescuer_count.columns = ['RescuerID', 'rescuer_count']

        # Compute target encoding features (mean and variance of AdoptionSpeed)
        if 'AdoptionSpeed' in df.columns:
            rescuer_target = df.groupby('RescuerID')['AdoptionSpeed'].agg(['mean', 'std', 'count']).reset_index()
            rescuer_target.columns = ['RescuerID', 'rescuer_mean_adoption', 'rescuer_std_adoption', 'rescuer_pet_count']

            # Merge count and target stats
            self.rescuer_stats = rescuer_count.merge(rescuer_target, on='RescuerID', how='left')
        else:
            # For test data without AdoptionSpeed, only use count
            self.rescuer_stats = rescuer_count

        self.rescuer_stats = self.rescuer_stats.set_index('RescuerID')

    def _fit_breed_features(self, df: pd.DataFrame):
        """Load breed labels and create encoders for breed names."""
        if self.breed_labels_path and os.path.exists(self.breed_labels_path):
            # Load breed labels
            self.breed_labels = pd.read_csv(self.breed_labels_path)

            # Merge to get all unique breed names in training data
            if 'Breed1' in df.columns:
                breed1_merged = df[['Breed1']].merge(
                    self.breed_labels,
                    left_on='Breed1',
                    right_on='BreedID',
                    how='left'
                )
                unique_breed1_names = breed1_merged['BreedName'].fillna('Unknown').unique()
                # Create label encoding mapping
                self.breed_encoders['Breed1'] = {name: idx for idx, name in enumerate(unique_breed1_names)}

            if 'Breed2' in df.columns:
                breed2_merged = df[['Breed2']].merge(
                    self.breed_labels,
                    left_on='Breed2',
                    right_on='BreedID',
                    how='left'
                )
                unique_breed2_names = breed2_merged['BreedName'].fillna('Unknown').unique()
                self.breed_encoders['Breed2'] = {name: idx for idx, name in enumerate(unique_breed2_names)}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataset with advanced features."""
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")

        result_df = df.copy()

        # 1. Text features
        text_features = self._transform_text_features(df)
        result_df = pd.concat([result_df, text_features], axis=1)

        # 2. Rescuer features (count only)
        rescuer_features = self._transform_rescuer_features(df)
        result_df = pd.concat([result_df, rescuer_features], axis=1)

        # 3. Breed name features (categorical encoding)
        breed_features = self._extract_breed_features(df)
        if not breed_features.empty:
            result_df = pd.concat([result_df, breed_features], axis=1)

        # 4. Google Vision metadata (if available)
        metadata_features = self._extract_metadata_features(df)
        if not metadata_features.empty:
            result_df = pd.concat([result_df, metadata_features], axis=1)

        # 5. Google Sentiment (if available)
        sentiment_features = self._extract_sentiment_features(df)
        if not sentiment_features.empty:
            result_df = pd.concat([result_df, sentiment_features], axis=1)

        # 6. Image size statistics (if available)
        image_size_features = self._extract_image_size_features(df)
        if not image_size_features.empty:
            result_df = pd.concat([result_df, image_size_features], axis=1)

        # 7. Feature interactions
        interaction_features = self._create_feature_interactions(result_df)
        result_df = pd.concat([result_df, interaction_features], axis=1)

        return result_df

    def _transform_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform text into TF-IDF + SVD features (or just text statistics if TF-IDF disabled)."""
        features = []
        text_columns = ['Description', 'Name']

        for col in text_columns:
            if col in df.columns:
                texts = df[col].fillna('').astype(str).apply(self._clean_text)

                # If TF-IDF enabled, extract TF-IDF + SVD features
                if self.use_tfidf and col in self.tfidf_transformers:
                    # TF-IDF + SVD
                    tfidf_matrix = self.tfidf_transformers[col].transform(texts)
                    svd_features = self.svd_transformers[col].transform(tfidf_matrix)

                    # Create feature dataframe
                    feature_names = [f'tfidf_{col.lower()}_{i}' for i in range(self.svd_components)]
                    text_df = pd.DataFrame(svd_features, columns=feature_names, index=df.index)
                else:
                    # TF-IDF disabled - just create empty dataframe for text statistics
                    text_df = pd.DataFrame(index=df.index)

                # Always extract simple text statistics (regardless of TF-IDF setting)
                text_df[f'{col.lower()}_length'] = texts.str.len()
                text_df[f'{col.lower()}_word_count'] = texts.str.split().str.len()
                text_df[f'{col.lower()}_unique_word_ratio'] = texts.apply(
                    lambda x: len(set(x.split())) / max(len(x.split()), 1)
                )

                features.append(text_df)

        return pd.concat(features, axis=1) if features else pd.DataFrame(index=df.index)

    def _transform_rescuer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform rescuer features with target encoding."""
        if 'RescuerID' not in df.columns or not hasattr(self, 'rescuer_stats'):
            return pd.DataFrame(index=df.index)

        # Merge rescuer statistics
        result = df[['RescuerID']].merge(
            self.rescuer_stats,
            left_on='RescuerID',
            right_index=True,
            how='left'
        )

        # For unseen rescuers, use -100 for clear OOD signal (before normalization)
        # After StandardScaler normalization, this becomes a large negative value
        # distinct from real rescuer statistics
        feature_cols = [col for col in result.columns if col != 'RescuerID']
        result[feature_cols] = result[feature_cols].fillna(-100)

        return result[feature_cols]

    def _extract_breed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract breed name features (categorical encoding)."""
        if self.breed_labels is None or not self.breed_encoders:
            return pd.DataFrame(index=df.index)

        breed_features = []

        # Breed1 name
        if 'Breed1' in df.columns and 'Breed1' in self.breed_encoders:
            breed1_merged = df[['Breed1']].merge(
                self.breed_labels,
                left_on='Breed1',
                right_on='BreedID',
                how='left'
            )
            breed1_names = breed1_merged['BreedName'].fillna('Unknown')
            # Encode breed names to integers
            # Use -100 for unknown breeds (clear OOD signal before normalization)
            breed1_encoded = breed1_names.map(self.breed_encoders['Breed1']).fillna(-100).astype(int)
            breed_features.append(pd.DataFrame({'main_breed_encoded': breed1_encoded}, index=df.index))

        # Breed2 name
        if 'Breed2' in df.columns and 'Breed2' in self.breed_encoders:
            breed2_merged = df[['Breed2']].merge(
                self.breed_labels,
                left_on='Breed2',
                right_on='BreedID',
                how='left'
            )
            breed2_names = breed2_merged['BreedName'].fillna('Unknown')
            # Use -100 for unknown breeds (clear OOD signal before normalization)
            breed2_encoded = breed2_names.map(self.breed_encoders['Breed2']).fillna(-100).astype(int)
            breed_features.append(pd.DataFrame({'second_breed_encoded': breed2_encoded}, index=df.index))

        return pd.concat(breed_features, axis=1) if breed_features else pd.DataFrame(index=df.index)

    def _create_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature interactions."""
        interactions = []

        if 'Age' in df.columns and 'Type' in df.columns:
            interactions.append(pd.DataFrame({
                'age_type_interaction': df['Age'] * df['Type']
            }, index=df.index))

        if 'Breed1' in df.columns and 'Color1' in df.columns:
            interactions.append(pd.DataFrame({
                'breed_color_interaction': df['Breed1'] * 100 + df['Color1']
            }, index=df.index))

        if 'PhotoAmt' in df.columns and 'VideoAmt' in df.columns:
            interactions.append(pd.DataFrame({
                'media_interaction': df['PhotoAmt'] * df['VideoAmt'],
                'total_media': df['PhotoAmt'] + df['VideoAmt']
            }, index=df.index))

        return pd.concat(interactions, axis=1) if interactions else pd.DataFrame(index=df.index)

    def _extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract Google Vision API metadata features (with multithreading)."""
        if not self.metadata_dir:
            return pd.DataFrame(index=df.index)

        import glob
        import json
        from concurrent.futures import ThreadPoolExecutor

        def extract_single_metadata(pet_id):
            """Extract metadata for a single pet."""
            metadata_files = glob.glob(f'{self.metadata_dir}/{pet_id}-*.json')

            if not metadata_files:
                # No images/metadata - use -100 as clear "missing" signal
                return {
                    'metadata_annots_score': -100,
                    'metadata_color_score': -100,
                    'metadata_color_pixelfrac': -100,
                    'metadata_crop_conf': -100,
                    'metadata_crop_importance': -100,
                }

            # Aggregate across all images for this pet
            scores = []
            color_scores = []
            pixel_fracs = []
            crop_confs = []
            crop_imps = []

            for meta_file in metadata_files:
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)

                    # Label annotations
                    if 'labelAnnotations' in meta:
                        annots = [x['score'] for x in meta['labelAnnotations']]
                        scores.extend(annots)

                    # Color properties
                    if 'imagePropertiesAnnotation' in meta:
                        colors = meta['imagePropertiesAnnotation']['dominantColors']['colors']
                        color_scores.extend([x['score'] for x in colors])
                        pixel_fracs.extend([x['pixelFraction'] for x in colors])

                    # Crop hints
                    if 'cropHintsAnnotation' in meta:
                        crops = meta['cropHintsAnnotation']['cropHints']
                        crop_confs.extend([x['confidence'] for x in crops])
                        if 'importanceFraction' in crops[0]:
                            crop_imps.extend([x['importanceFraction'] for x in crops])

                except Exception:
                    continue

            return {
                'metadata_annots_score': np.mean(scores) if scores else 0,
                'metadata_color_score': np.mean(color_scores) if color_scores else 0,
                'metadata_color_pixelfrac': np.mean(pixel_fracs) if pixel_fracs else 0,
                'metadata_crop_conf': np.mean(crop_confs) if crop_confs else 0,
                'metadata_crop_importance': np.mean(crop_imps) if crop_imps else 0,
            }

        # Use ThreadPoolExecutor for parallel processing
        print('  Extracting metadata features...')
        pet_ids = df['PetID'].tolist()
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            metadata_features = list(executor.map(extract_single_metadata, pet_ids))

        return pd.DataFrame(metadata_features, index=df.index)

    def _extract_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract Google NLP sentiment features (with multithreading)."""
        if not self.sentiment_dir:
            return pd.DataFrame(index=df.index)

        import json
        from concurrent.futures import ThreadPoolExecutor

        def extract_single_sentiment(pet_id):
            """Extract sentiment for a single pet."""
            sentiment_file = f'{self.sentiment_dir}/{pet_id}.json'

            try:
                with open(sentiment_file, 'r') as f:
                    sent = json.load(f)

                doc_sentiment = sent['documentSentiment']
                sentences = [x['sentiment'] for x in sent['sentences']]

                return {
                    'sentiment_magnitude': doc_sentiment['magnitude'],
                    'sentiment_score': doc_sentiment['score'],
                    'sentiment_sentences_magnitude_sum': sum(s['magnitude'] for s in sentences),
                    'sentiment_sentences_score_sum': sum(s['score'] for s in sentences),
                    'sentiment_sentences_magnitude_mean': np.mean([s['magnitude'] for s in sentences]),
                    'sentiment_sentences_score_mean': np.mean([s['score'] for s in sentences]),
                }
            except Exception:
                # No description/sentiment - use -100 as clear "missing" signal
                # After StandardScaler normalization, this is clearly out-of-distribution
                return {
                    'sentiment_magnitude': -100,
                    'sentiment_score': -100,
                    'sentiment_sentences_magnitude_sum': -100,
                    'sentiment_sentences_score_sum': -100,
                    'sentiment_sentences_magnitude_mean': -100,
                    'sentiment_sentences_score_mean': -100,
                }

        # Use ThreadPoolExecutor for parallel processing
        print('  Extracting sentiment features...')
        pet_ids = df['PetID'].tolist()
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            sentiment_features = list(executor.map(extract_single_sentiment, pet_ids))

        return pd.DataFrame(sentiment_features, index=df.index)

    def _extract_image_size_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract image size statistics (with multithreading)."""
        if not self.images_dir:
            return pd.DataFrame(index=df.index)

        import glob
        import os
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor

        def extract_single_image_stats(pet_id):
            """Extract image statistics for a single pet."""
            image_files = glob.glob(f'{self.images_dir}/{pet_id}-*.jpg')

            if not image_files:
                # No images - use -100 as clear "missing" signal
                # Note: image_count stays 0 since it's a true count, not a feature
                return {
                    'image_size_sum': -100,
                    'image_size_mean': -100,
                    'width_sum': -100,
                    'width_mean': -100,
                    'height_sum': -100,
                    'height_mean': -100,
                    'image_count': 0,
                }

            sizes = []
            widths = []
            heights = []

            for img_file in image_files:
                try:
                    # File size
                    sizes.append(os.path.getsize(img_file))

                    # Image dimensions
                    with Image.open(img_file) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                except Exception:
                    continue

            return {
                'image_size_sum': sum(sizes),
                'image_size_mean': np.mean(sizes) if sizes else -100,
                'width_sum': sum(widths),
                'width_mean': np.mean(widths) if widths else -100,
                'height_sum': sum(heights),
                'height_mean': np.mean(heights) if heights else -100,
                'image_count': len(image_files),
            }

        # Use ThreadPoolExecutor for parallel processing
        print('  Extracting image statistics...')
        pet_ids = df['PetID'].tolist()
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            image_features = list(executor.map(extract_single_image_stats, pet_ids))

        return pd.DataFrame(image_features, index=df.index)

print("âœ“ Advanced Feature Engineer defined (TF-IDF, rescuer count, breed names, metadata, sentiment, image stats)")
print("  âš¡ Multithreading enabled for 3-4x faster feature extraction")


"""
Extract image embeddings using pretrained Vision Transformer (ViT-B/16).
Processes all images for a pet and averages them into single embedding.
"""

def save_embeddings(embeddings_dict: Dict, filepath: str):
    """Save embeddings dictionary to .npz file."""
    pet_ids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[pid] for pid in pet_ids])
    # Handle NaN/Inf before saving
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(filepath, pet_ids=pet_ids, embeddings=embeddings)
    print(f"  âœ“ Saved to: {filepath}")


def load_embeddings(filepath: str) -> Dict:
    """Load embeddings dictionary from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    pet_ids = data['pet_ids']
    embeddings = data['embeddings']
    # Handle NaN/Inf when loading
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    return {str(pid): emb for pid, emb in zip(pet_ids, embeddings)}


def extract_image_embeddings(img_dir: str, df: pd.DataFrame, model_name: str, device='cuda', save_path=None):
    """
    Extract image embeddings using ViT.

    Args:
        img_dir: Directory containing images (format: PetID-N.jpg)
        df: Dataframe with PetID column
        model_name: timm model name
        device: 'cuda' or 'cpu'
        save_path: If provided, save embeddings to this path

    Returns:
        Dict mapping PetID to embedding vector
    """
    # Check if already processed
    if save_path and os.path.exists(save_path):
        print(f"Found existing file: {save_path}")
        print(f"Loading from disk (skipping extraction)...")
        return load_embeddings(save_path)

    print(f"Loading image model: {model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    model.eval()

    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_dir = Path(img_dir)
    embeddings = {}

    print(f"Extracting embeddings for {len(df['PetID'].unique())} unique pets...")

    for pet_id in tqdm(df['PetID'].unique(), desc="Processing images"):
        # Find all images for this pet
        image_files = list(img_dir.glob(f"{pet_id}-*.jpg"))

        if not image_files:
            # No images found, use zero embedding
            embeddings[pet_id] = np.zeros(768, dtype=np.float32)
            continue

        pet_embeddings = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model(img_tensor).cpu().numpy()[0]
                pet_embeddings.append(emb)
            except Exception as e:
                # Skip corrupted images
                continue

        # Average embeddings if multiple images
        if pet_embeddings:
            embeddings[pet_id] = np.mean(pet_embeddings, axis=0).astype(np.float32)
        else:
            embeddings[pet_id] = np.zeros(768, dtype=np.float32)

    print(f"âœ“ Extracted embeddings for {len(embeddings)} pets")

    # Save to disk if path provided
    if save_path:
        save_embeddings(embeddings, save_path)

    return embeddings

print("âœ“ Image embedding extractor defined")


"""
Extract text embeddings using Sentence-BERT (all-MiniLM-L6-v2).
Processes pet descriptions into dense semantic vectors.

FIX: Monkey-patch to bypass chat_templates 404 error during download
"""

def extract_text_embeddings(df: pd.DataFrame, model_name: str, device='cuda', save_path=None):
    """
    Extract text embeddings using Sentence-BERT.

    Args:
        df: Dataframe with 'Description' and 'PetID' columns
        model_name: sentence-transformers model name
        device: 'cuda' or 'cpu'
        save_path: If provided, save embeddings to this path

    Returns:
        Dict mapping PetID to embedding vector
    """
    # Check if already processed
    if save_path and os.path.exists(save_path):
        print(f"Found existing file: {save_path}")
        print(f"Loading from disk (skipping extraction)...")
        return load_embeddings(save_path)

    print(f"Loading text model: {model_name}")

    # FIX: Monkey-patch to bypass chat_templates 404 error
    # Make list_repo_templates tolerant to missing "additional_chat_templates"
    from transformers.utils import hub as _hf_hub
    try:
        import transformers.tokenization_utils_base as _tub
    except Exception:
        _tub = None

    _original_hub_list_repo_templates = getattr(_hf_hub, "list_repo_templates", None)
    _original_tub_list_repo_templates = getattr(_tub, "list_repo_templates", None) if _tub is not None else None

    def _patched_list_repo_templates(repo_id, local_files_only=False, revision=None, cache_dir=None, **kwargs):
        try:
            if _original_hub_list_repo_templates is None:
                return []
            return _original_hub_list_repo_templates(
                repo_id,
                local_files_only=local_files_only,
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception as e:
            msg = str(e).lower()
            # Treat missing chat templates / 404 as "no templates"
            if (
                "additional_chat_templates" in msg
                or "chat_templates" in msg
                or "404" in msg
                or "entry not found" in msg
            ):
                return []
            raise  # Re-raise other unexpected errors

    # Apply the patch in both the hub module and the tokenization module
    if _original_hub_list_repo_templates is not None:
        _hf_hub.list_repo_templates = _patched_list_repo_templates
    if _original_tub_list_repo_templates is not None:
        _tub.list_repo_templates = _patched_list_repo_templates

    try:
        # Now load the model - download will work with patch applied
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    finally:
        # Restore original function(s)
        if _original_hub_list_repo_templates is not None:
            _hf_hub.list_repo_templates = _original_hub_list_repo_templates
        if _original_tub_list_repo_templates is not None:
            _tub.list_repo_templates = _original_tub_list_repo_templates

    embeddings = {}
    texts = []
    pet_ids = []

    # Prepare texts
    for _, row in df.iterrows():
        text = str(row.get('Description', '')) if pd.notna(row.get('Description')) else ''
        texts.append(text if text.strip() else '')
        pet_ids.append(row['PetID'])

    print(f"Encoding {len(texts)} text descriptions...")

    # Batch encode all texts
    text_embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization
    )

    # Map to PetID
    for pet_id, emb in zip(pet_ids, text_embeddings):
        embeddings[pet_id] = emb.astype(np.float32)

    print(f"âœ“ Extracted text embeddings for {len(embeddings)} pets")

    # Save to disk if path provided
    if save_path:
        save_embeddings(embeddings, save_path)

    return embeddings

print("âœ“ Text embedding extractor defined")


"""
PyTorch Dataset class that loads preprocessed embeddings and features.
Handles missing data gracefully with zero embeddings.
"""

class PetFinderDataset(Dataset):
    """Dataset for multimodal PetFinder data with optional feature dropout."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_embeddings: Dict,
        text_embeddings: Dict,
        tabular_tokens: np.ndarray,
        has_labels: bool = True,
        rescuer_feature_indices: Optional[List[int]] = None,
        rescuer_dropout: float = 0.0,
        is_training: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.tabular_tokens = tabular_tokens
        self.has_labels = has_labels
        self.rescuer_feature_indices = rescuer_feature_indices or []
        self.rescuer_dropout = rescuer_dropout
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pet_id = row['PetID']

        # Get embeddings (use zeros if missing)
        img_emb = self.image_embeddings.get(pet_id, np.zeros(768, dtype=np.float32))
        txt_emb = self.text_embeddings.get(pet_id, np.zeros(384, dtype=np.float32))

        # Handle NaN/Inf in embeddings (safety check)
        img_emb = np.nan_to_num(img_emb, nan=0.0, posinf=0.0, neginf=0.0)
        txt_emb = np.nan_to_num(txt_emb, nan=0.0, posinf=0.0, neginf=0.0)
        tabular = np.nan_to_num(self.tabular_tokens[idx].copy(), nan=0.0, posinf=0.0, neginf=0.0)

        # Apply rescuer feature dropout during training (regularization)
        if self.is_training and self.rescuer_dropout > 0 and len(self.rescuer_feature_indices) > 0:
            # Randomly mask rescuer features with probability rescuer_dropout
            if np.random.rand() < self.rescuer_dropout:
                # Set all rescuer features to -100 (clearly out-of-distribution signal)
                # After StandardScaler normalization, normal values are ~[-3, 3]
                # -100 is clearly "unknown rescuer" vs a real normalized value
                tabular[self.rescuer_feature_indices] = -100.0

        sample = {
            'image_embeddings': torch.tensor(img_emb, dtype=torch.float32),
            'text_embeddings': torch.tensor(txt_emb, dtype=torch.float32),
            'tabular_tokens': torch.tensor(tabular, dtype=torch.float32),
            'pet_id': pet_id
        }

        if self.has_labels:
            sample['labels'] = torch.tensor(row['AdoptionSpeed'], dtype=torch.long)

        return sample

print("âœ“ Dataset class defined")


"""
Quadratic Weighted Kappa (QWK) - the primary evaluation metric.
Measures agreement between predictions and ground truth, with quadratic penalties.
"""

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate QWK score (primary metric for PetFinder).

    Range: [-1, 1] where 1 = perfect agreement, 0 = random, -1 = perfect disagreement
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot confusion matrix for visualization."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted AdoptionSpeed')
    plt.ylabel('True AdoptionSpeed')
    plt.title(title)
    plt.tight_layout()
    plt.show()

print("âœ“ Metrics defined")


"""
Warmup + Cosine Annealing scheduler for stable training.
Warmup prevents early instability, cosine annealing improves convergence.
"""

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, T_max, eta_min):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            'warmup_epochs': self.warmup_epochs,
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'base_lr': self.base_lr,
            'current_epoch': self.current_epoch
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.warmup_epochs = state_dict['warmup_epochs']
        self.T_max = state_dict['T_max']
        self.eta_min = state_dict['eta_min']
        self.base_lr = state_dict['base_lr']
        self.current_epoch = state_dict['current_epoch']

print("âœ“ Learning rate scheduler defined")



"""
Quadratic Weighted Kappa (QWK) - the primary evaluation metric.
Measures agreement between predictions and ground truth, with quadratic penalties.
"""

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate QWK score (primary metric for PetFinder).

    Range: [-1, 1] where 1 = perfect agreement, 0 = random, -1 = perfect disagreement
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot confusion matrix for visualization."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted AdoptionSpeed')
    plt.ylabel('True AdoptionSpeed')
    plt.title(title)
    plt.tight_layout()
    plt.show()

print("âœ“ Metrics defined")



"""
Training and validation epoch functions.
Implements gradient accumulation, clipping, and metric tracking.
"""

def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    accumulate_grad_batches = config['accumulate_grad_batches']
    gradient_clip_norm = config['gradient_clip_norm']

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Zero gradients at start of accumulation cycle
        if batch_idx % accumulate_grad_batches == 0:
            optimizer.zero_grad()

        # Forward pass
        outputs = model(
            batch['image_embeddings'],
            batch['text_embeddings'],
            batch['tabular_tokens']
        )

        # Compute loss
        loss = criterion(outputs['logits'], batch['labels'])

        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWARNING: NaN/Inf detected in loss at batch {batch_idx}")
            print(f"  Logits stats: min={outputs['logits'].min():.4f}, max={outputs['logits'].max():.4f}")
            print(f"  Labels: {batch['labels'][:5]}")
            print(f"  Skipping this batch...")
            continue

        unscaled_loss = loss.item()

        # Scale loss for gradient accumulation
        loss = loss / accumulate_grad_batches

        # Backward pass
        loss.backward()

        # Update weights every accumulate_grad_batches steps
        if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(dataloader):
            # Gradient clipping for stability
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

        # Track metrics
        total_loss += unscaled_loss

        # Get predictions
        probabilities = model.predict_probabilities(outputs['logits'])
        predictions = torch.argmax(probabilities, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(batch['labels'].cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix({'loss': f"{unscaled_loss:.4f}"})

    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    qwk = quadratic_weighted_kappa(all_targets, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

    return avg_loss, qwk, accuracy


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Validation"):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        outputs = model(
            batch['image_embeddings'],
            batch['text_embeddings'],
            batch['tabular_tokens']
        )

        # Compute loss
        loss = criterion(outputs['logits'], batch['labels'])
        total_loss += loss.item()

        # Get predictions
        probabilities = model.predict_probabilities(outputs['logits'])
        predictions = torch.argmax(probabilities, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(batch['labels'].cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    qwk = quadratic_weighted_kappa(all_targets, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

    return avg_loss, qwk, accuracy, all_targets, all_preds

print("âœ“ Training functions defined")


"""
Early stopping to prevent overfitting and save training time.
Monitors validation QWK and stops if no improvement for patience epochs.
"""

class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience=300, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.early_stop = False

    def __call__(self, score):
        """Check if should stop."""
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True  # Stop training
            return False

print("âœ“ Early stopping defined")



"""
Load raw data and split into train/validation.
This cell takes ~1-2 minutes to run.
"""

print("Loading CSV files...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Loaded {len(train_df)} training samples")
print(f"Loaded {len(test_df)} test samples")

# Display class distribution
print("\nAdoptionSpeed distribution in training data:")
print(train_df['AdoptionSpeed'].value_counts().sort_index())

# Split train into train/val (stratified)
print(f"\nSplitting {CONFIG['val_split']*100:.0f}% for validation...")
train_df, val_df = train_test_split(
    train_df,
    test_size=CONFIG['val_split'],
    stratify=train_df['AdoptionSpeed'],
    random_state=42
)

print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")

print("\nâœ“ Data loaded and split")



"""
Process tabular features with advanced engineering (matching XGBoost baseline):
- TF-IDF + SVD on Description and Name
- Rescuer count (NO adoption speed stats)
- Breed name categorical encoding (main_breed, second_breed)
- Google Vision API metadata
- Google Sentiment API features
- Image size statistics
- Feature interactions (age x type, breed x color, media)

ðŸ”¥ SMART CACHING: If features are already saved, they'll be loaded instead!
âš¡ MULTITHREADED: Uses 4 parallel workers for fast I/O operations
   First run: ~1-2 minutes (with multithreading, was 3-5 min single-threaded)
   Subsequent runs: <1 second (loads from disk)
"""

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Define save paths
train_tabular_path = f'{OUTPUT_DIR}/train_tabular_features.npz'
val_tabular_path = f'{OUTPUT_DIR}/val_tabular_features.npz'
test_tabular_path = f'{OUTPUT_DIR}/test_tabular_features.npz'

# Check if already processed
if (os.path.exists(train_tabular_path) and
    os.path.exists(val_tabular_path) and
    os.path.exists(test_tabular_path)):

    print("âœ“ Found existing tabular features!")
    print("Loading from disk (skipping feature extraction)...")

    # Load from disk
    train_data = np.load(train_tabular_path)
    val_data = np.load(val_tabular_path)
    test_data = np.load(test_tabular_path)

    train_tabular = train_data['features']
    val_tabular = val_data['features']
    test_tabular = test_data['features']
    numeric_columns = list(train_data['feature_names'])

    # Find rescuer feature indices for dropout
    rescuer_feature_names = ['rescuer_count', 'rescuer_mean_adoption', 'rescuer_std_adoption', 'rescuer_pet_count']
    rescuer_feature_indices = [i for i, col in enumerate(numeric_columns) if col in rescuer_feature_names]

    # Ensure no NaN/Inf values (safety check)
    train_tabular = np.nan_to_num(train_tabular, nan=0.0, posinf=0.0, neginf=0.0)
    val_tabular = np.nan_to_num(val_tabular, nan=0.0, posinf=0.0, neginf=0.0)
    test_tabular = np.nan_to_num(test_tabular, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Train: {train_tabular.shape}")
    print(f"  Val: {val_tabular.shape}")
    print(f"  Test: {test_tabular.shape}")
    print(f"  Features: {len(numeric_columns)}")
    print(f"  Rescuer features found: {len(rescuer_feature_indices)} {[numeric_columns[i] for i in rescuer_feature_indices]}")
    print(f"  âœ“ Validated (no NaN/Inf values)")

else:
    import time
    start_time = time.time()

    print("Processing advanced tabular features (matching XGBoost baseline)...")
    print("âš¡ Using 4 parallel workers for fast I/O")
    print("This may take 1-2 minutes on first run (saved to disk for future runs)")
    print("")

    # Initialize feature engineer
    print("[1/5] Initializing feature engineer...")
    tabular_processor = AdvancedFeatureEngineer(
        tfidf_max_features=2000,
        svd_components=20,
        random_state=42,
        metadata_dir=TRAIN_METADATA_DIR,  # Google Vision metadata
        sentiment_dir=TRAIN_SENTIMENT_DIR,  # Google Sentiment
        images_dir=None,  # DISABLED: Image size extraction is very slow (58K images)
        breed_labels_path=BREED_LABELS_CSV,  # For breed name encoding
        n_jobs=1,  # Number of parallel workers (Kaggle-safe: 4 threads)
        use_tfidf=CONFIG['use_tfidf_features']  # Whether to use TF-IDF (can be redundant with Sentence-BERT)
    )
    print("  âš  Image size features DISABLED (slow PIL operations on 58K images)")
    print("  âœ“ This saves ~8-10 minutes of processing time")

    # Fit on training data
    print("[2/5] Fitting feature transformers on training data...")
    tabular_processor.fit(train_df)
    print("      âœ“ Feature transformers fitted")

    # Transform all splits (returns DataFrames with enhanced features)
    print("[3/5] Transforming training data...")
    enhanced_train = tabular_processor.transform(train_df)
    print(f"      âœ“ Train: {enhanced_train.shape}")

    print("[4/5] Transforming validation data...")
    enhanced_val = tabular_processor.transform(val_df)
    print(f"      âœ“ Val: {enhanced_val.shape}")

    print("[5/5] Transforming test data...")
    enhanced_test = tabular_processor.transform(test_df)
    print(f"      âœ“ Test: {enhanced_test.shape}")
    print("")

    # Extract numeric columns only (exclude identifiers, text, and target)
    print("Extracting numeric features...")
    exclude_columns = ['PetID', 'Name', 'RescuerID', 'Description', 'AdoptionSpeed']

    numeric_columns = []
    for col in enhanced_train.columns:
        if col in exclude_columns:
            continue
        # Only include columns that exist in all three datasets
        if col not in enhanced_test.columns:
            continue
        try:
            enhanced_train[col].astype(np.float32)
            numeric_columns.append(col)
        except:
            pass

    print(f"  âœ“ Found {len(numeric_columns)} numeric features")

    # Convert to numpy arrays for transformer model
    train_tabular = enhanced_train[numeric_columns].values.astype(np.float32)
    val_tabular = enhanced_val[numeric_columns].values.astype(np.float32)
    test_tabular = enhanced_test[numeric_columns].values.astype(np.float32)

    # Handle NaN and Inf values
    print("Handling NaN and Inf values...")
    train_tabular = np.nan_to_num(train_tabular, nan=0.0, posinf=0.0, neginf=0.0)
    val_tabular = np.nan_to_num(val_tabular, nan=0.0, posinf=0.0, neginf=0.0)
    test_tabular = np.nan_to_num(test_tabular, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize features using StandardScaler
    print("Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    train_tabular = scaler.fit_transform(train_tabular).astype(np.float32)
    val_tabular = scaler.transform(val_tabular).astype(np.float32)
    test_tabular = scaler.transform(test_tabular).astype(np.float32)

    # Final NaN check after scaling (sometimes scaling can introduce NaN if std=0)
    train_tabular = np.nan_to_num(train_tabular, nan=0.0, posinf=0.0, neginf=0.0)
    val_tabular = np.nan_to_num(val_tabular, nan=0.0, posinf=0.0, neginf=0.0)
    test_tabular = np.nan_to_num(test_tabular, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  âœ“ Features normalized and validated (no NaN/Inf values)")

    # Save to disk for future use
    print("\nSaving features to disk...")
    np.savez_compressed(
        train_tabular_path,
        features=train_tabular,
        pet_ids=train_df['PetID'].values,
        feature_names=numeric_columns
    )
    np.savez_compressed(
        val_tabular_path,
        features=val_tabular,
        pet_ids=val_df['PetID'].values,
        feature_names=numeric_columns
    )
    np.savez_compressed(
        test_tabular_path,
        features=test_tabular,
        pet_ids=test_df['PetID'].values,
        feature_names=numeric_columns
    )
    print(f"  âœ“ Saved to {OUTPUT_DIR}/*.npz")

    elapsed_time = time.time() - start_time
    print("\nâœ“ Advanced tabular features complete!")
    print(f"  Train: {train_tabular.shape}")
    print(f"  Val: {val_tabular.shape}")
    print(f"  Test: {test_tabular.shape}")
    print(f"  Features: {len(numeric_columns)}")
    print(f"  Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

    # Find rescuer feature indices for dropout
    rescuer_feature_names = ['rescuer_count', 'rescuer_mean_adoption', 'rescuer_std_adoption', 'rescuer_pet_count']
    rescuer_feature_indices = [i for i, col in enumerate(numeric_columns) if col in rescuer_feature_names]
    print(f"  Rescuer features found: {len(rescuer_feature_indices)} {[numeric_columns[i] for i in rescuer_feature_indices]}")

# Show rescuer features for verification (works for both load and process paths)
rescuer_features = [c for c in numeric_columns if 'rescuer' in c.lower()]
print(f"\nRescuer features: {rescuer_features}")


"""
Extract image embeddings using ViT-B/16.
This is the slowest step - takes ~20-40 minutes depending on GPU.

ðŸ”¥ SMART CACHING: If embeddings are already saved to disk, they'll be loaded
   instead of reprocessing. This saves 20-40 minutes on reruns!

Files saved to: /kaggle/working/*.npz
- train_image_embs.npz
- val_image_embs.npz
- test_image_embs.npz

IMPORTANT: This uses GPU memory. If you get OOM errors, the embeddings
will still complete but may be slower.
"""

print("Extracting image embeddings...")
print("This may take 20-40 minutes on first run (saved to disk for future runs)")
print("On subsequent runs, will load from disk in <1 second! ðŸ”¥")

# Extract for all splits (with saving/loading from disk)
train_img_embs = extract_image_embeddings(
    TRAIN_IMG_DIR, train_df, CONFIG['image_model'], CONFIG['device'],
    save_path=f'{OUTPUT_DIR}/train_image_embs.npz'
)

val_img_embs = extract_image_embeddings(
    TRAIN_IMG_DIR, val_df, CONFIG['image_model'], CONFIG['device'],
    save_path=f'{OUTPUT_DIR}/val_image_embs.npz'
)

test_img_embs = extract_image_embeddings(
    TEST_IMG_DIR, test_df, CONFIG['image_model'], CONFIG['device'],
    save_path=f'{OUTPUT_DIR}/test_image_embs.npz'
)

print("\nâœ“ Image embeddings ready for all splits")


"""
Extract text embeddings using Sentence-BERT.
This cell takes ~5-10 minutes to run.

ðŸ”¥ SMART CACHING: If embeddings are already saved, they'll be loaded instead!

Files saved to: /kaggle/working/*.npz
- train_text_embs.npz
- val_text_embs.npz
- test_text_embs.npz
"""

print("Extracting text embeddings...")
print("This may take 5-10 minutes on first run (saved to disk for future runs)")

# Extract for all splits (with saving/loading from disk)
train_text_embs = extract_text_embeddings(
    train_df, CONFIG['text_model'], CONFIG['device'],
    save_path=f'{OUTPUT_DIR}/train_text_embs.npz'
)

val_text_embs = extract_text_embeddings(
    val_df, CONFIG['text_model'], CONFIG['device'],
    save_path=f'{OUTPUT_DIR}/val_text_embs.npz'
)

test_text_embs = extract_text_embeddings(
    test_df, CONFIG['text_model'], CONFIG['device'],
    save_path=f'{OUTPUT_DIR}/test_text_embs.npz'
)

print("\nâœ“ Text embeddings ready for all splits")


"""
Create PyTorch datasets and dataloaders for efficient batch processing.
"""

print("Creating datasets...")

# Show rescuer feature dropout config
if CONFIG['rescuer_feature_dropout'] > 0 and len(rescuer_feature_indices) > 0:
    print(f"âœ“ Rescuer feature dropout enabled: {CONFIG['rescuer_feature_dropout']*100:.0f}%")
    print(f"  Will randomly mask {len(rescuer_feature_indices)} rescuer features to -100 during training")
    print(f"  This simulates unseen rescuers and improves generalization")
else:
    print(f"âœ— Rescuer feature dropout disabled")

# Create datasets with rescuer feature dropout
train_dataset = PetFinderDataset(
    train_df,
    train_img_embs,
    train_text_embs,
    train_tabular,
    has_labels=True,
    rescuer_feature_indices=rescuer_feature_indices,
    rescuer_dropout=CONFIG['rescuer_feature_dropout'],
    is_training=True  # Enable dropout for training
)

val_dataset = PetFinderDataset(
    val_df,
    val_img_embs,
    val_text_embs,
    val_tabular,
    has_labels=True,
    rescuer_feature_indices=rescuer_feature_indices,
    rescuer_dropout=0.0,  # No dropout for validation
    is_training=False
)

test_dataset = PetFinderDataset(
    test_df,
    test_img_embs,
    test_text_embs,
    test_tabular,
    has_labels=False,
    rescuer_feature_indices=rescuer_feature_indices,
    rescuer_dropout=0.0,  # No dropout for test
    is_training=False
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['train_batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['val_batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['val_batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

print(f"âœ“ Datasets and dataloaders created")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")


"""
Create model, loss function, optimizer, and scheduler.
"""

print("Initializing model...")

# Create model
device = torch.device(CONFIG['device'])
model = MultimodalTransformer(CONFIG).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"âœ“ Model created")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Compute class weights for imbalanced dataset
if CONFIG['use_class_weights']:
    targets = train_df['AdoptionSpeed'].values
    class_weights = compute_class_weights(targets, CONFIG['num_classes']).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
else:
    class_weights = None

# Create loss function
criterion = OrdinalLoss(
    CONFIG['num_classes'],
    class_weights,
    label_smoothing=CONFIG.get('label_smoothing', 0.0)
)
print("âœ“ Ordinal loss initialized")

# Create optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['lr'],
    weight_decay=CONFIG['weight_decay']
)
print(f"âœ“ AdamW optimizer initialized (lr={CONFIG['lr']:.2e})")

# Create learning rate scheduler
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=CONFIG['warmup_epochs'],
    T_max=CONFIG['T_max'],
    eta_min=CONFIG['eta_min']
)
print(f"âœ“ Warmup+Cosine scheduler initialized")

# Create early stopping
early_stopping = EarlyStopping(
    patience=CONFIG['early_stopping_patience'],
    min_delta=CONFIG['early_stopping_min_delta'],
    mode='max'
)
print(f"âœ“ Early stopping initialized (patience={CONFIG['early_stopping_patience']})")


"""
Main training loop with early stopping and metric tracking.
This will take several hours depending on when early stopping triggers.

The loop will:
1. Train for one epoch
2. Validate
3. Update learning rate
4. Check early stopping
5. Save best model
6. Track metrics
"""

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Max epochs: {CONFIG['epochs']}")
print(f"Early stopping patience: {CONFIG['early_stopping_patience']} epochs")
print(f"This may take several hours. Be patient!")
print("="*60 + "\n")

best_qwk = -1.0
best_epoch = 0
best_model_state = None

for epoch in range(CONFIG['epochs']):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch + 1}/{CONFIG['epochs']}")
    print(f"{'='*60}")

    # Train
    train_loss, train_qwk, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, CONFIG
    )

    # Validate
    val_loss, val_qwk, val_acc, val_targets, val_preds = validate_epoch(
        model, val_loader, criterion, device
    )

    # Update learning rate
    scheduler.step()
    current_lr = scheduler.get_lr()

    # Log metrics
    tracker.log_epoch(epoch, train_loss, train_qwk, train_acc, val_loss, val_qwk, val_acc, current_lr)

    # Print epoch summary
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train QWK: {train_qwk:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val QWK:   {val_qwk:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"  Learning Rate: {current_lr:.2e}")

    # Save best model
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        best_epoch = epoch
        best_model_state = model.state_dict().copy()

        # Save to disk (overwrites previous best)
        checkpoint_path = f'{OUTPUT_DIR}/best_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_qwk': best_qwk,
            'val_loss': val_loss,
            'config': CONFIG
        }, checkpoint_path)

        print(f"  âœ“ New best QWK: {best_qwk:.4f} (saved to {checkpoint_path})")

        # Visualize attention at new best checkpoint
        print(f"\n  Generating attention visualizations...")
        visualize_attention_weights(
            model=model,
            val_loader=val_loader,
            device=device,
            save_dir=OUTPUT_DIR,
            num_samples=200,  # Analyze 200 validation samples
            epoch=epoch + 1,
            feature_names=numeric_columns  # Pass actual feature names
        )

    # Periodic visualization (every 50 epochs)
    if (epoch + 1) % 50 == 0:
        print(f"\n  Periodic visualization at epoch {epoch + 1}...")
        visualize_attention_weights(
            model=model,
            val_loader=val_loader,
            device=device,
            save_dir=OUTPUT_DIR,
            num_samples=200,
            epoch=epoch + 1,
            feature_names=numeric_columns  # Pass actual feature names
        )

    # Check early stopping
    if early_stopping(val_qwk):
        print(f"\n{'='*60}")
        print(f"Early stopping triggered at epoch {epoch + 1}")
        print(f"No improvement for {CONFIG['early_stopping_patience']} epochs")
        print(f"{'='*60}")
        break

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"Best Validation QWK: {best_qwk:.4f} at epoch {best_epoch + 1}")
print(f"Best model saved to: {OUTPUT_DIR}/best_model.pth")
print("="*60 + "\n")

# Load best model
model.load_state_dict(best_model_state)
print("âœ“ Best model loaded for inference")
print("\nTo reload this model later, use:")
print(f"  checkpoint = torch.load('{OUTPUT_DIR}/best_model.pth')")
print(f"  model.load_state_dict(checkpoint['model_state_dict'])")


# """
# Generate comprehensive training visualizations.
# Shows loss curves, QWK progression, learning rate schedule, and overfitting analysis.
# """

# # Plot training curves
# tracker.plot_training_curves('training_curves.png')

# # Print summary
# tracker.print_summary()

# # Plot final confusion matrix
# print("\nGenerating confusion matrix on validation set...")
# val_loss, val_qwk, val_acc, val_targets, val_preds = validate_epoch(
#     model, val_loader, criterion, device
# )
# plot_confusion_matrix(val_targets, val_preds,
#                      title=f'Validation Confusion Matrix (QWK={val_qwk:.4f}, Acc={val_acc:.4f})')


"""
Generate predictions on the test set using the best model.
Creates submission.csv in Kaggle submission format.
"""

print("Generating test predictions...")

model.eval()
all_preds = []
all_pet_ids = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        outputs = model(
            batch['image_embeddings'],
            batch['text_embeddings'],
            batch['tabular_tokens']
        )

        probabilities = model.predict_probabilities(outputs['logits'])
        predictions = torch.argmax(probabilities, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_pet_ids.extend(batch['pet_id'])

print(f"âœ“ Generated predictions for {len(all_preds)} test samples")


"""
Create submission.csv file in the required format.
Ready to submit to Kaggle!
"""

# Create submission dataframe
submission = pd.DataFrame({
    'PetID': all_pet_ids,
    'AdoptionSpeed': all_preds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("âœ“ Submission file created: submission.csv")
print(f"\nSubmission preview:")
print(submission.head(10))

print("\nPrediction distribution:")
print(submission['AdoptionSpeed'].value_counts().sort_index())

print("\n" + "="*60)
print("ALL DONE! ðŸŽ‰")
print("="*60)
print(f"Best Validation QWK: {best_qwk:.4f}")
print(f"Submission file: submission.csv")
print(f"Total test predictions: {len(submission)}")
print("\nYou can now download submission.csv and submit to Kaggle!")
print("="*60)





























