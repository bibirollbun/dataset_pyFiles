"""
File cáº¥u hÃ¬nh chá»©a táº¥t cáº£ cÃ¡c tham sá»‘ quan trá»�ng cho project
Má»¥c Ä‘Ã­ch: Táº­p trung quáº£n lÃ½ cÃ¡c hyperparameter, Ä‘Æ°á»�ng dáº«n vÃ  cÃ i Ä‘áº·t
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class DataConfig:
    """
    Cáº¥u hÃ¬nh liÃªn quan Ä‘áº¿n dá»¯ liá»‡u
    - Ä�Æ°á»�ng dáº«n tá»›i cÃ¡c thÆ° má»¥c chá»©a data
    - KÃ­ch thÆ°á»›c áº£nh vÃ  batch size
    - Tá»· lá»‡ chia train/validation
    """
    # Ä�Æ°á»�ng dáº«n dá»¯ liá»‡u
    train_image_path: str = '/kaggle/input/bkai-igh-neopolyp/train/train'
    train_mask_path: str = '/kaggle/input/bkai-igh-neopolyp/train_gt/train_gt'
    test_image_path: str = '/kaggle/input/bkai-igh-neopolyp/test/test'
    
    # KÃ­ch thÆ°á»›c vÃ  batch
    image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 4  # Giáº£m batch size Ä‘á»ƒ train lÃ¢u hÆ¡n
    num_workers: int = 4
    
    # Chia dá»¯ liá»‡u
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5

@dataclass
class ModelConfig:
    """
    Cáº¥u hÃ¬nh mÃ´ hÃ¬nh
    - Loáº¡i mÃ´ hÃ¬nh vÃ  backbone
    - Sá»‘ lá»›p phÃ¢n loáº¡i
    - Pretrained weights
    """
    # MÃ´ hÃ¬nh chÃ­nh
    model_name: str = "segformer"
    backbone: str = "nvidia/mit-b5"
    num_classes: int = 3
    
    # Labels
    id2label: Optional[Dict[int, str]] = None
    label2id: Optional[Dict[str, int]] = None
    
    # Pretrained
    use_pretrained: bool = True
    
    def __post_init__(self):
        if self.id2label is None:
            self.id2label = {
                0: "neoplastic",
                1: "non-neoplastic", 
                2: "background"
            }
        if self.label2id is None:
            self.label2id = {v: k for k, v in self.id2label.items()}

@dataclass
class TrainingConfig:
    """
    Cáº¥u hÃ¬nh training
    - Learning rate, optimizer
    - Sá»‘ epoch, early stopping
    - Loss function vÃ  metrics
    """
    # Optimizer
    optimizer: str = "AdamW"
    learning_rate: float = 1e-5  # Giáº£m learning rate Ä‘á»ƒ train á»•n Ä‘á»‹nh hÆ¡n
    weight_decay: float = 0.01
    
    # Training
    epochs: int = 120  # TÄƒng epochs Ä‘á»ƒ train lÃ¢u hÆ¡n (2 tiáº¿ng)
    warmup_steps: int = 1000  # TÄƒng warmup steps
    
    # Early stopping
    patience: int = 30  # TÄƒng patience Ä‘á»ƒ khÃ´ng dá»«ng sá»›m
    min_delta: float = 0.0002  # Giáº£m min_delta Ä‘á»ƒ nháº¡y cáº£m hÆ¡n
    
    # Loss
    loss_function: str = "cross_entropy"
    class_weights: Optional[List[float]] = None
    
    # Metrics
    metrics: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["mean_iou", "dice", "accuracy"]

@dataclass
class ExperimentConfig:
    """
    Cáº¥u hÃ¬nh thÃ­ nghiá»‡m
    - Logging, checkpointing
    - Wandb settings
    - Output paths
    """
    # Experiment name
    experiment_name: str = "polyp_segmentation"
    run_name: str = "segformer_b5_baseline"
    
    # Paths
    output_dir: str = "/kaggle/working"
    model_save_dir: str = "/kaggle/working/models"
    log_dir: str = "/kaggle/working/logs"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "polyp-segmentation"
    log_interval: int = 100
    
    # Checkpointing
    save_best_only: bool = True
    save_last: bool = True
    monitor_metric: str = "val_mean_iou"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True

@dataclass
class InferenceConfig:
    """
    Cáº¥u hÃ¬nh inference
    - Test Time Augmentation
    - Post-processing
    - Output format
    """
    # TTA
    use_tta: bool = True
    tta_transforms: int = 8
    
    # Post-processing
    use_crf: bool = False
    use_morphology: bool = True
    
    # Output
    output_format: str = "rle"  # "rle" or "mask"
    submission_file: str = "submission.csv"

# Táº¡o config tá»•ng há»£p
@dataclass
class Config:
    """
    Config tá»•ng há»£p chá»©a táº¥t cáº£ cÃ¡c cáº¥u hÃ¬nh con
    Má»¥c Ä‘Ã­ch: Táº­p trung táº¥t cáº£ settings vÃ o má»™t object duy nháº¥t
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def create_directories(self):
        """
        Táº¡o cÃ¡c thÆ° má»¥c cáº§n thiáº¿t cho experiment
        """
        directories = [
            self.experiment.output_dir,
            self.experiment.model_save_dir,
            self.experiment.log_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def get_device(self):
        """
        Láº¥y device phÃ¹ há»£p (GPU hoáº·c CPU)
        """
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Táº¡o config máº·c Ä‘á»‹nh
def get_default_config() -> Config:
    """
    HÃ m tráº£ vá»� config máº·c Ä‘á»‹nh
    Má»¥c Ä‘Ã­ch: Dá»… dÃ ng import vÃ  sá»­ dá»¥ng config trong cÃ¡c file khÃ¡c
    """
    return Config() 
print('DONE')


"""
File chá»©a cÃ¡c hÃ m tiá»‡n Ã­ch cho bÃ i toÃ¡n phÃ¢n Ä‘oáº¡n polyp
Má»¥c Ä‘Ã­ch: Cung cáº¥p cÃ¡c hÃ m há»— trá»£ cho visualization, RLE encoding, post-processing
"""

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import logging
import random
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Thiáº¿t láº­p random seed cho reproducibility
    
    Args:
        seed: GiÃ¡ trá»‹ seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Ä�Ã£ thiáº¿t láº­p seed: {seed}")

def create_directories(paths: List[str]):
    """
    Táº¡o cÃ¡c thÆ° má»¥c cáº§n thiáº¿t
    
    Args:
        paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n thÆ° má»¥c cáº§n táº¡o
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ä�Ã£ táº¡o thÆ° má»¥c: {path}")

def rle_encode(mask: np.ndarray) -> str:
    """
    MÃ£ hÃ³a mask thÃ nh Run Length Encoding (RLE)
    
    Args:
        mask: Binary mask array
        
    Returns:
        RLE encoded string
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(rle_string: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Giáº£i mÃ£ RLE string thÃ nh mask
    
    Args:
        rle_string: RLE encoded string
        shape: KÃ­ch thÆ°á»›c mask (height, width)
        
    Returns:
        Decoded binary mask
    """
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    
    return mask.reshape(shape)

def mask_to_rle(mask: np.ndarray) -> str:
    """
    Chuyá»ƒn Ä‘á»•i mask thÃ nh RLE format cho submission
    
    Args:
        mask: Segmentation mask [H, W] hoáº·c [H, W, C]
        
    Returns:
        RLE encoded string
    """
    if len(mask.shape) == 3:
        # Multi-class mask
        rle_strings = []
        for class_idx in range(mask.shape[2]):
            class_mask = mask[:, :, class_idx]
            rle = rle_encode(class_mask)
            rle_strings.append(rle)
        return rle_strings
    else:
        # Single class mask
        return rle_encode(mask)

def visualize_sample(
    image: np.ndarray, 
    mask: np.ndarray, 
    prediction: Optional[np.ndarray] = None,
    class_names: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize áº£nh, mask vÃ  prediction
    
    Args:
        image: Input image [H, W, C]
        mask: Ground truth mask [H, W]
        prediction: Predicted mask [H, W] (optional)
        class_names: TÃªn cÃ¡c classes
        save_path: Ä�Æ°á»�ng dáº«n lÆ°u áº£nh
    """
    if class_names is None:
        class_names = ['neoplastic', 'non-neoplastic', 'background']
    
    # Táº¡o colormap cho visualization
    colors = np.array([
        [255, 0, 0],      # Red cho neoplastic
        [0, 255, 0],      # Green cho non-neoplastic
        [0, 0, 255]       # Blue cho background
    ])
    
    # Sá»‘ subplot
    n_plots = 3 if prediction is not None else 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('áº¢nh gá»‘c')
    axes[0].axis('off')
    
    # Plot ground truth mask
    mask_colored = colors[mask]
    axes[1].imshow(mask_colored.astype(np.uint8))
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction if available
    if prediction is not None:
        pred_colored = colors[prediction]
        axes[2].imshow(pred_colored.astype(np.uint8))
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    # ThÃªm legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i]/255, label=class_names[i]) 
                      for i in range(len(class_names))]
    fig.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Ä�Ã£ lÆ°u visualization: {save_path}")
    
    plt.show()

def visualize_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    class_names: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize má»™t batch samples
    
    Args:
        images: Batch images [B, C, H, W]
        masks: Batch masks [B, H, W]
        predictions: Batch predictions [B, H, W] (optional)
        num_samples: Sá»‘ samples Ä‘á»ƒ visualize
        class_names: TÃªn cÃ¡c classes
        save_path: Ä�Æ°á»�ng dáº«n lÆ°u áº£nh
    """
    if class_names is None:
        class_names = ['neoplastic', 'non-neoplastic', 'background']
    
    # Convert to numpy
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    if predictions is not None:
        predictions = predictions.cpu().numpy()
    
    # Normalize images to [0, 1]
    images = (images - images.min()) / (images.max() - images.min())
    
    # Sá»‘ samples thá»±c táº¿
    actual_samples = min(num_samples, images.shape[0])
    
    # Sá»‘ columns
    n_cols = 3 if predictions is not None else 2
    
    fig, axes = plt.subplots(actual_samples, n_cols, figsize=(5 * n_cols, 5 * actual_samples))
    if actual_samples == 1:
        axes = axes.reshape(1, -1)
    
    colors = np.array([
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green  
        [0, 0, 255]       # Blue
    ])
    
    for i in range(actual_samples):
        # Original image
        img = np.transpose(images[i], (1, 2, 0))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1} - áº¢nh gá»‘c')
        axes[i, 0].axis('off')
        
        # Ground truth
        mask_colored = colors[masks[i]]
        axes[i, 1].imshow(mask_colored.astype(np.uint8))
        axes[i, 1].set_title(f'Sample {i+1} - Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        if predictions is not None:
            pred_colored = colors[predictions[i]]
            axes[i, 2].imshow(pred_colored.astype(np.uint8))
            axes[i, 2].set_title(f'Sample {i+1} - Prediction')
            axes[i, 2].axis('off')
    
    # Legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i]/255, label=class_names[i]) 
                      for i in range(len(class_names))]
    fig.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Ä�Ã£ lÆ°u batch visualization: {save_path}")
    
    plt.show()

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Váº½ biá»ƒu Ä‘á»“ training history
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_metrics: Training metrics
        val_metrics: Validation metrics
        save_path: Ä�Æ°á»�ng dáº«n lÆ°u áº£nh
    """
    n_metrics = len(train_metrics) + 1  # +1 cho loss
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    # Plot loss
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_title('Training vÃ  Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics
    for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
        val_values = val_metrics.get(metric_name, [])
        
        axes[idx + 1].plot(epochs, train_values, 'b-', label=f'Training {metric_name}')
        if val_values:
            axes[idx + 1].plot(epochs, val_values, 'r-', label=f'Validation {metric_name}')
        
        axes[idx + 1].set_title(f'Training vÃ  Validation {metric_name}')
        axes[idx + 1].set_xlabel('Epoch')
        axes[idx + 1].set_ylabel(metric_name)
        axes[idx + 1].legend()
        axes[idx + 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Ä�Ã£ lÆ°u training history: {save_path}")
    
    plt.show()

def apply_morphological_operations(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Ã�p dá»¥ng morphological operations Ä‘á»ƒ lÃ m sáº¡ch mask
    
    Args:
        mask: Binary mask
        kernel_size: KÃ­ch thÆ°á»›c kernel
        
    Returns:
        Cleaned mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening: loáº¡i bá»� noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Closing: fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def remove_small_objects(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Loáº¡i bá»� cÃ¡c object nhá»� trong mask
    
    Args:
        mask: Binary mask
        min_size: KÃ­ch thÆ°á»›c tá»‘i thiá»ƒu (pixels)
        
    Returns:
        Cleaned mask
    """
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    # Create output mask
    output_mask = np.zeros_like(mask)
    
    # Keep only large components
    for label in range(1, num_labels):
        component_mask = (labels == label)
        if np.sum(component_mask) >= min_size:
            output_mask[component_mask] = 1
    
    return output_mask

def post_process_mask(mask: np.ndarray, min_size: int = 100, kernel_size: int = 3) -> np.ndarray:
    """
    Post-processing cho segmentation mask
    
    Args:
        mask: Raw segmentation mask
        min_size: KÃ­ch thÆ°á»›c tá»‘i thiá»ƒu object
        kernel_size: KÃ­ch thÆ°á»›c kernel cho morphological ops
        
    Returns:
        Post-processed mask
    """
    # Apply morphological operations
    mask = apply_morphological_operations(mask, kernel_size)
    
    # Remove small objects
    mask = remove_small_objects(mask, min_size)
    
    return mask

def create_submission_file(
    predictions: List[np.ndarray],
    image_ids: List[str],
    output_path: str = "submission.csv"
):
    """
    Táº¡o file submission cho Kaggle
    
    Args:
        predictions: List cÃ¡c predicted masks
        image_ids: List cÃ¡c image IDs
        output_path: Ä�Æ°á»�ng dáº«n file output
    """
    submission_data = []
    
    for pred_mask, image_id in zip(predictions, image_ids):
        # Giáº£ sá»­ mask cÃ³ 3 classes (neoplastic, non-neoplastic)
        # Chá»‰ submit 2 classes Ä‘áº§u (background khÃ´ng cáº§n)
        
        for class_idx in range(2):  # 0: neoplastic, 1: non-neoplastic
            class_mask = (pred_mask == class_idx).astype(np.uint8)
            
            # Apply post-processing
            class_mask = post_process_mask(class_mask)
            
            # Convert to RLE
            rle = rle_encode(class_mask)
            
            submission_data.append({
                'Id': f"{image_id}_{class_idx}",
                'Expected': rle
            })
    
    # Táº¡o DataFrame vÃ  lÆ°u
    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Ä�Ã£ táº¡o submission file: {output_path}")
    logger.info(f"Sá»‘ lÆ°á»£ng predictions: {len(submission_data)}")

def calculate_class_weights(masks: List[np.ndarray], num_classes: int = 3) -> np.ndarray:
    """
    TÃ­nh class weights Ä‘á»ƒ xá»­ lÃ½ class imbalance
    
    Args:
        masks: List cÃ¡c ground truth masks
        num_classes: Sá»‘ lÆ°á»£ng classes
        
    Returns:
        Class weights array
    """
    class_counts = np.zeros(num_classes)
    
    for mask in masks:
        for class_idx in range(num_classes):
            class_counts[class_idx] += np.sum(mask == class_idx)
    
    # TÃ­nh weights (inverse frequency)
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * num_classes
    
    logger.info(f"Class counts: {class_counts}")
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights.astype(np.float32)

def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    save_path: str
):
    """
    LÆ°u model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Epoch hiá»‡n táº¡i
        loss: Loss value
        metrics: Dict cÃ¡c metrics
        save_path: Ä�Æ°á»�ng dáº«n lÆ°u checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Ä�Ã£ lÆ°u checkpoint: {save_path}")

def load_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str
) -> Tuple[int, float, Dict[str, float]]:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Ä�Æ°á»�ng dáº«n checkpoint
        
    Returns:
        Tuple (epoch, loss, metrics)
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metrics = checkpoint['metrics']
    
    logger.info(f"Ä�Ã£ load checkpoint tá»« epoch {epoch}")
    
    return epoch, loss, metrics

def get_device() -> torch.device:
    """
    Láº¥y device phÃ¹ há»£p (GPU hoáº·c CPU)
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Sá»­ dá»¥ng GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Sá»­ dá»¥ng CPU")
    
    return device

def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, int, int, int]):
    """
    In thÃ´ng tin tÃ³m táº¯t vá»� model
    
    Args:
        model: PyTorch model
        input_size: KÃ­ch thÆ°á»›c input (B, C, H, W)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # TÃ­nh model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f"Model size: {size_mb:.2f} MB")
    print("=" * 50) 
print('DONE')


"""
File xá»­ lÃ½ dá»¯ liá»‡u cho bÃ i toÃ¡n phÃ¢n Ä‘oáº¡n polyp
Má»¥c Ä‘Ã­ch: Táº¡o Dataset class, augmentation vÃ  data loading
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import PILToTensor, ToPILImage
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerImageProcessor
from typing import List, Tuple, Optional, Dict, Any
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolypDataset(Dataset):
    """
    Dataset class cho bÃ i toÃ¡n phÃ¢n Ä‘oáº¡n polyp
    
    Chá»©c nÄƒng:
    - Load áº£nh vÃ  mask tá»« Ä‘Æ°á»�ng dáº«n
    - Ã�p dá»¥ng augmentation
    - Xá»­ lÃ½ mask thÃ nh format phÃ¹ há»£p cho model
    - Tráº£ vá»� data Ä‘Ã£ Ä‘Æ°á»£c preprocessing
    """
    
    def __init__(
        self, 
        image_paths: List[str], 
        mask_paths: List[str], 
        processor: SegformerImageProcessor,
        augmentation: Optional[A.Compose] = None,
        is_training: bool = True
    ):
        """
        Khá»Ÿi táº¡o Dataset
        
        Args:
            image_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n tá»›i áº£nh
            mask_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n tá»›i mask
            processor: SegformerImageProcessor Ä‘á»ƒ xá»­ lÃ½ áº£nh
            augmentation: Augmentation pipeline
            is_training: CÃ³ pháº£i training mode khÃ´ng
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.augmentation = augmentation
        self.is_training = is_training
        
        # Kiá»ƒm tra sá»‘ lÆ°á»£ng áº£nh vÃ  mask pháº£i báº±ng nhau
        assert len(image_paths) == len(mask_paths), \
            f"Sá»‘ lÆ°á»£ng áº£nh ({len(image_paths)}) vÃ  mask ({len(mask_paths)}) khÃ´ng khá»›p"
        
        logger.info(f"Ä�Ã£ táº¡o dataset vá»›i {len(self.image_paths)} samples")
    
    def __len__(self) -> int:
        """
        Tráº£ vá»� sá»‘ lÆ°á»£ng samples trong dataset
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Láº¥y má»™t sample tá»« dataset
        
        Args:
            idx: Index cá»§a sample
            
        Returns:
            Dict chá»©a pixel_values vÃ  labels Ä‘Ã£ Ä‘Æ°á»£c xá»­ lÃ½
        """
        try:
            # Láº¥y Ä‘Æ°á»�ng dáº«n áº£nh vÃ  mask
            image_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            
            # Kiá»ƒm tra file tá»“n táº¡i
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"KhÃ´ng tÃ¬m tháº¥y áº£nh: {image_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"KhÃ´ng tÃ¬m tháº¥y mask: {mask_path}")
            
            # Load áº£nh vÃ  mask
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
            
            # Convert sang numpy Ä‘á»ƒ Ã¡p dá»¥ng augmentation
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            # Ã�p dá»¥ng augmentation náº¿u cÃ³
            if self.augmentation is not None and self.is_training:
                augmented = self.augmentation(image=image_np, mask=mask_np)
                image_np = augmented['image']
                mask_np = augmented['mask']
            
            # Convert láº¡i thÃ nh PIL Image
            image = Image.fromarray(image_np)
            mask = Image.fromarray(mask_np)
            
            # Xá»­ lÃ½ mask
            mask_processed = self._process_mask(mask)
            
            # Sá»­ dá»¥ng processor Ä‘á»ƒ chuáº©n bá»‹ input cho model
            encoded_inputs = self.processor(
                image, 
                mask_processed, 
                return_tensors="pt"
            )
            
            # Loáº¡i bá»� batch dimension
            for key, value in encoded_inputs.items():
                encoded_inputs[key] = value.squeeze(0)
            
            return encoded_inputs
            
        except Exception as e:
            logger.error(f"Lá»—i khi load sample {idx}: {str(e)}")
            raise e
    
    def _process_mask(self, mask: Image.Image) -> Image.Image:
        """
        Xá»­ lÃ½ mask thÃ nh format phÃ¹ há»£p
        
        Args:
            mask: Mask PIL Image
            
        Returns:
            Mask Ä‘Ã£ Ä‘Æ°á»£c xá»­ lÃ½
        """
        # Convert sang tensor
        mask_tensor = PILToTensor()(mask).float() / 255.0
        
        # Ã�p dá»¥ng threshold
        mask_tensor = torch.where(mask_tensor > 0.65, 1.0, 0.0)
        
        # Xá»­ lÃ½ background channel
        if mask_tensor.shape[0] >= 3:
            mask_tensor[2, :, :] = 0.0001
        
        # Láº¥y class cÃ³ confidence cao nháº¥t
        mask_class = torch.argmax(mask_tensor, dim=0).numpy().astype(np.uint8)
        
        # Convert láº¡i thÃ nh PIL Image
        return Image.fromarray(mask_class)

def get_augmentation_pipeline(is_training: bool = True) -> Optional[A.Compose]:
    """
    Táº¡o augmentation pipeline cho training/validation
    
    Args:
        is_training: CÃ³ pháº£i training mode khÃ´ng
        
    Returns:
        Augmentation pipeline hoáº·c None náº¿u khÃ´ng cáº§n augmentation
    """
    if not is_training:
        return None
    
    # Augmentation pipeline cho training
    train_transform = A.Compose([
        # Geometric transformations
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        
        # Color transformations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.2),
        A.RGBShift(
            r_shift_limit=10,
            g_shift_limit=10,
            b_shift_limit=10,
            p=0.3
        ),
        
        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        
        # Elastic transformation
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.2
        ),
    ])
    
    return train_transform

def create_data_splits(
    image_paths: List[str],
    mask_paths: List[str],
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Chia dá»¯ liá»‡u thÃ nh train/validation/test
    
    Args:
        image_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n áº£nh
        mask_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n mask
        train_split: Tá»· lá»‡ train
        val_split: Tá»· lá»‡ validation
        seed: Random seed
        
    Returns:
        Tuple chá»©a (train_images, train_masks, val_images, val_masks, test_images, test_masks)
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Táº¡o indices vÃ  shuffle
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    
    # TÃ­nh toÃ¡n sá»‘ lÆ°á»£ng samples cho má»—i split
    n_total = len(image_paths)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    
    # Chia indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Táº¡o lists cho tá»«ng split
    train_images = [image_paths[i] for i in train_indices]
    train_masks = [mask_paths[i] for i in train_indices]
    
    val_images = [image_paths[i] for i in val_indices]
    val_masks = [mask_paths[i] for i in val_indices]
    
    test_images = [image_paths[i] for i in test_indices]
    test_masks = [mask_paths[i] for i in test_indices]
    
    logger.info(f"Chia dá»¯ liá»‡u: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
    
    return train_images, train_masks, val_images, val_masks, test_images, test_masks

def create_dataloaders(
    train_images: List[str],
    train_masks: List[str],
    val_images: List[str],
    val_masks: List[str],
    processor: SegformerImageProcessor,
    batch_size: int = 8,
    num_workers: int = 4,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Táº¡o DataLoader cho training vÃ  validation
    
    Args:
        train_images: Danh sÃ¡ch áº£nh training
        train_masks: Danh sÃ¡ch mask training
        val_images: Danh sÃ¡ch áº£nh validation
        val_masks: Danh sÃ¡ch mask validation
        processor: SegformerImageProcessor
        batch_size: KÃ­ch thÆ°á»›c batch
        num_workers: Sá»‘ worker cho DataLoader
        use_augmentation: CÃ³ sá»­ dá»¥ng augmentation khÃ´ng
        
    Returns:
        Tuple chá»©a (train_dataloader, val_dataloader)
    """
    # Táº¡o augmentation
    train_augmentation = get_augmentation_pipeline(is_training=True) if use_augmentation else None
    val_augmentation = get_augmentation_pipeline(is_training=False)
    
    # Táº¡o datasets
    train_dataset = PolypDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        processor=processor,
        augmentation=train_augmentation,
        is_training=True
    )
    
    val_dataset = PolypDataset(
        image_paths=val_images,
        mask_paths=val_masks,
        processor=processor,
        augmentation=val_augmentation,
        is_training=False
    )
    
    # Táº¡o dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Táº¡o DataLoader: Train batches={len(train_dataloader)}, Val batches={len(val_dataloader)}")
    
    return train_dataloader, val_dataloader

def get_file_paths(image_dir: str, mask_dir: str) -> Tuple[List[str], List[str]]:
    """
    Láº¥y danh sÃ¡ch Ä‘Æ°á»�ng dáº«n file áº£nh vÃ  mask
    
    Args:
        image_dir: ThÆ° má»¥c chá»©a áº£nh
        mask_dir: ThÆ° má»¥c chá»©a mask
        
    Returns:
        Tuple chá»©a (image_paths, mask_paths)
    """
    # Láº¥y danh sÃ¡ch file
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Táº¡o Ä‘Æ°á»�ng dáº«n Ä‘áº§y Ä‘á»§
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]
    
    logger.info(f"TÃ¬m tháº¥y {len(image_paths)} áº£nh vÃ  {len(mask_paths)} mask")
    
    return image_paths, mask_paths

def analyze_dataset(image_paths: List[str], mask_paths: List[str]) -> Dict[str, Any]:
    """
    PhÃ¢n tÃ­ch dataset Ä‘á»ƒ hiá»ƒu rÃµ hÆ¡n vá»� dá»¯ liá»‡u
    
    Args:
        image_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n áº£nh
        mask_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n mask
        
    Returns:
        Dict chá»©a thÃ´ng tin phÃ¢n tÃ­ch
    """
    analysis = {
        'total_samples': len(image_paths),
        'image_sizes': [],
        'class_distribution': {'neoplastic': 0, 'non_neoplastic': 0, 'background': 0}
    }
    
    logger.info("Ä�ang phÃ¢n tÃ­ch dataset...")
    
    for i, (img_path, mask_path) in enumerate(zip(image_paths[:100], mask_paths[:100])):  # Chá»‰ phÃ¢n tÃ­ch 100 sample Ä‘áº§u
        try:
            # PhÃ¢n tÃ­ch kÃ­ch thÆ°á»›c áº£nh
            img = Image.open(img_path)
            analysis['image_sizes'].append(img.size)
            
            # PhÃ¢n tÃ­ch phÃ¢n bá»‘ class trong mask
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            
            # Ä�áº¿m pixel theo class (giáº£ sá»­ cÃ³ 3 channel cho 3 class)
            if len(mask_array.shape) == 3:
                for class_idx in range(min(3, mask_array.shape[2])):
                    class_pixels = np.sum(mask_array[:, :, class_idx] > 127)
                    if class_idx == 0:
                        analysis['class_distribution']['neoplastic'] += class_pixels
                    elif class_idx == 1:
                        analysis['class_distribution']['non_neoplastic'] += class_pixels
                    else:
                        analysis['class_distribution']['background'] += class_pixels
                        
        except Exception as e:
            logger.warning(f"Lá»—i khi phÃ¢n tÃ­ch sample {i}: {str(e)}")
            continue
    
    # TÃ­nh toÃ¡n thá»‘ng kÃª
    if analysis['image_sizes']:
        widths, heights = zip(*analysis['image_sizes'])
        analysis['avg_width'] = np.mean(widths)
        analysis['avg_height'] = np.mean(heights)
        analysis['min_size'] = (min(widths), min(heights))
        analysis['max_size'] = (max(widths), max(heights))
    
    logger.info(f"PhÃ¢n tÃ­ch hoÃ n thÃ nh: {analysis}")
    
    return analysis 
print('DONE')


"""
File chá»©a cÃ¡c mÃ´ hÃ¬nh cho bÃ i toÃ¡n phÃ¢n Ä‘oáº¡n polyp
Má»¥c Ä‘Ã­ch: Ä�á»‹nh nghÄ©a cÃ¡c kiáº¿n trÃºc mÃ´ hÃ¬nh khÃ¡c nhau (Segformer, U-Net, DeepLab)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers khÃ´ng cÃ³ sáºµn, chá»‰ cÃ³ thá»ƒ sá»­ dá»¥ng cÃ¡c mÃ´ hÃ¬nh custom")

class SegformerModel(nn.Module):
    """
    Wrapper cho Segformer model tá»« Hugging Face
    
    Chá»©c nÄƒng:
    - Load pretrained Segformer model
    - TÃ¹y chá»‰nh sá»‘ lÆ°á»£ng classes
    - Há»— trá»£ fine-tuning
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/mit-b5",
        num_classes: int = 3,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        pretrained: bool = True
    ):
        """
        Khá»Ÿi táº¡o Segformer model
        
        Args:
            model_name: TÃªn model tá»« Hugging Face
            num_classes: Sá»‘ lÆ°á»£ng classes
            id2label: Mapping tá»« id sang label
            label2id: Mapping tá»« label sang id
            pretrained: CÃ³ sá»­ dá»¥ng pretrained weights khÃ´ng
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library khÃ´ng cÃ³ sáºµn")
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Táº¡o default labels náº¿u khÃ´ng cÃ³
        if id2label is None:
            id2label = {i: f"class_{i}" for i in range(num_classes)}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        
        # Load model
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            config.id2label = id2label
            config.label2id = label2id
            self.model = SegformerForSemanticSegmentation(config)
        
        logger.info(f"Ä�Ã£ táº¡o Segformer model: {model_name} vá»›i {num_classes} classes")
    
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass cá»§a model
        
        Args:
            pixel_values: Input images tensor
            labels: Ground truth labels (optional)
            
        Returns:
            Output tá»« model
        """
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Predict segmentation map
        
        Args:
            pixel_values: Input images tensor
            
        Returns:
            Predicted segmentation maps
        """
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Upsample logits to original image size
            upsampled_logits = F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            
            # Get predicted classes
            predicted = upsampled_logits.argmax(dim=1)
            
        return predicted

class DoubleConv(nn.Module):
    """
    Double Convolution block cho U-Net
    
    Chá»©c nÄƒng:
    - Thá»±c hiá»‡n 2 convolution liÃªn tiáº¿p
    - Batch normalization vÃ  ReLU activation
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Khá»Ÿi táº¡o DoubleConv block
        
        Args:
            in_channels: Sá»‘ input channels
            out_channels: Sá»‘ output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor sau double convolution
        """
        return self.double_conv(x)

class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module cho DeepLabV3
    
    Chá»©c nÄƒng:
    - Multi-scale feature extraction
    - Dilated convolutions vá»›i cÃ¡c dilation rates khÃ¡c nhau
    """
    
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int]):
        """
        Khá»Ÿi táº¡o ASPP module
        
        Args:
            in_channels: Sá»‘ input channels
            out_channels: Sá»‘ output channels
            atrous_rates: List cÃ¡c dilation rates
        """
        super().__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass cá»§a ASPP
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor vá»›i multi-scale features
        """
        res = []
        
        for conv in self.convs:
            res.append(conv(x))
        
        # Upsample global pooling result
        res[-1] = F.interpolate(res[-1], size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate all features
        res = torch.cat(res, dim=1)
        
        return self.project(res)

class DeepLabV3(nn.Module):
    """
    DeepLabV3 model cho semantic segmentation
    
    Chá»©c nÄƒng:
    - Sá»­ dá»¥ng dilated convolutions
    - ASPP module cho multi-scale features
    - Backbone cÃ³ thá»ƒ lÃ  ResNet hoáº·c MobileNet
    """
    
    def __init__(self, num_classes: int = 3, backbone: str = "resnet50"):
        """
        Khá»Ÿi táº¡o DeepLabV3 model
        
        Args:
            num_classes: Sá»‘ lÆ°á»£ng classes
            backbone: Backbone network (resnet50, mobilenet)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone = backbone
        
        # Simplified backbone (cÃ³ thá»ƒ thay tháº¿ báº±ng pretrained ResNet)
        self.backbone_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks (simplified)
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
        )
        
        # ASPP module
        self.aspp = ASPPModule(512, 256, [6, 12, 18])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )
        
        logger.info(f"Ä�Ã£ táº¡o DeepLabV3 model vá»›i {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass cá»§a DeepLabV3
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output segmentation logits [B, num_classes, H, W]
        """
        input_shape = x.shape[-2:]
        
        # Backbone
        x = self.backbone_layers(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Classifier
        x = self.classifier(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x

class FocalLoss(nn.Module):
    """
    Focal Loss cho class imbalance
    
    Chá»©c nÄƒng:
    - Giáº£m weight cá»§a easy examples
    - TÄƒng focus vÃ o hard examples
    - Há»¯u Ã­ch cho medical image segmentation
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Khá»Ÿi táº¡o Focal Loss
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        TÃ­nh Focal Loss
        
        Args:
            inputs: Predicted logits [B, C, H, W]
            targets: Ground truth labels [B, H, W]
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_model(model_name: str, num_classes: int = 3, **kwargs) -> nn.Module:
    """
    Factory function Ä‘á»ƒ táº¡o model theo tÃªn
    
    Args:
        model_name: TÃªn model (segformer, unet, deeplabv3)
        num_classes: Sá»‘ lÆ°á»£ng classes
        **kwargs: CÃ¡c tham sá»‘ khÃ¡c
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == "segformer":
        return SegformerModel(num_classes=num_classes, **kwargs)
    elif model_name == "unet":
        return UNet(num_classes=num_classes, **kwargs)
    elif model_name == "deeplabv3":
        return DeepLabV3(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"KhÃ´ng há»— trá»£ model: {model_name}")

def count_parameters(model: nn.Module) -> int:
    """
    Ä�áº¿m sá»‘ lÆ°á»£ng parameters trong model
    
    Args:
        model: PyTorch model
        
    Returns:
        Sá»‘ lÆ°á»£ng trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Láº¥y thÃ´ng tin vá»� model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict chá»©a thÃ´ng tin model
    """
    total_params = count_parameters(model)
    
    # TÃ­nh model size (MB)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'model_size_mb': size_mb,
        'model_type': type(model).__name__
    } 
print('DONE')


"""
File chá»©a cÃ¡c metrics Ä‘Ã¡nh giÃ¡ cho bÃ i toÃ¡n phÃ¢n Ä‘oáº¡n polyp
Má»¥c Ä‘Ã­ch: TÃ­nh toÃ¡n cÃ¡c metrics nhÆ° IoU, Dice, Accuracy, Precision, Recall
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationMetrics:
    """
    Class tÃ­nh toÃ¡n cÃ¡c metrics cho segmentation
    
    Chá»©c nÄƒng:
    - TÃ­nh IoU (Intersection over Union)
    - TÃ­nh Dice coefficient
    - TÃ­nh Accuracy, Precision, Recall
    - TÃ­nh Hausdorff distance
    """
    
    def __init__(self, num_classes: int = 3, ignore_index: int = -1):
        """
        Khá»Ÿi táº¡o metrics calculator
        
        Args:
            num_classes: Sá»‘ lÆ°á»£ng classes
            ignore_index: Index cáº§n ignore khi tÃ­nh metrics
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """
        Reset táº¥t cáº£ metrics vá»� 0
        """
        self.total_area_intersect = np.zeros(self.num_classes)
        self.total_area_union = np.zeros(self.num_classes)
        self.total_area_pred_label = np.zeros(self.num_classes)
        self.total_area_label = np.zeros(self.num_classes)
        self.total_correct = 0
        self.total_label = 0
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Cáº­p nháº­t metrics vá»›i prediction vÃ  target má»›i
        
        Args:
            pred: Predicted segmentation [H, W] hoáº·c [B, H, W]
            target: Ground truth segmentation [H, W] hoáº·c [B, H, W]
        """
        if pred.ndim == 3:  # Batch input
            for i in range(pred.shape[0]):
                self._update_single(pred[i], target[i])
        else:  # Single input
            self._update_single(pred, target)
    
    def _update_single(self, pred: np.ndarray, target: np.ndarray):
        """
        Cáº­p nháº­t metrics cho má»™t sample
        
        Args:
            pred: Predicted segmentation [H, W]
            target: Ground truth segmentation [H, W]
        """
        # Loáº¡i bá»� ignore index
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        # TÃ­nh intersection vÃ  union cho tá»«ng class
        for class_id in range(self.num_classes):
            pred_class = (pred == class_id)
            target_class = (target == class_id)
            
            intersect = np.sum(pred_class & target_class)
            union = np.sum(pred_class | target_class)
            
            self.total_area_intersect[class_id] += intersect
            self.total_area_union[class_id] += union
            self.total_area_pred_label[class_id] += np.sum(pred_class)
            self.total_area_label[class_id] += np.sum(target_class)
        
        # TÃ­nh accuracy tá»•ng thá»ƒ
        self.total_correct += np.sum(pred == target)
        self.total_label += len(pred)
    
    def get_iou(self) -> Dict[str, float]:
        """
        TÃ­nh IoU (Intersection over Union) cho tá»«ng class
        
        Returns:
            Dict chá»©a IoU cho tá»«ng class vÃ  mean IoU
        """
        iou_per_class = self.total_area_intersect / (self.total_area_union + 1e-8)
        
        # Chá»‰ tÃ­nh mean IoU cho cÃ¡c class cÃ³ ground truth
        valid_classes = self.total_area_label > 0
        mean_iou = np.mean(iou_per_class[valid_classes])
        
        results = {
            'mean_iou': mean_iou,
            'iou_per_class': iou_per_class.tolist()
        }
        
        return results
    
    def get_dice(self) -> Dict[str, float]:
        """
        TÃ­nh Dice coefficient cho tá»«ng class
        
        Returns:
            Dict chá»©a Dice cho tá»«ng class vÃ  mean Dice
        """
        dice_per_class = (2 * self.total_area_intersect) / (
            self.total_area_pred_label + self.total_area_label + 1e-8
        )
        
        # Chá»‰ tÃ­nh mean Dice cho cÃ¡c class cÃ³ ground truth
        valid_classes = self.total_area_label > 0
        mean_dice = np.mean(dice_per_class[valid_classes])
        
        results = {
            'mean_dice': mean_dice,
            'dice_per_class': dice_per_class.tolist()
        }
        
        return results
    
    def get_accuracy(self) -> float:
        """
        TÃ­nh pixel accuracy tá»•ng thá»ƒ
        
        Returns:
            Pixel accuracy
        """
        return self.total_correct / (self.total_label + 1e-8)
    
    def get_precision_recall(self) -> Dict[str, Any]:
        """
        TÃ­nh Precision vÃ  Recall cho tá»«ng class
        
        Returns:
            Dict chá»©a precision vÃ  recall cho tá»«ng class
        """
        precision_per_class = self.total_area_intersect / (self.total_area_pred_label + 1e-8)
        recall_per_class = self.total_area_intersect / (self.total_area_label + 1e-8)
        
        # F1 score
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (
            precision_per_class + recall_per_class + 1e-8
        )
        
        # Mean values (chá»‰ tÃ­nh cho cÃ¡c class cÃ³ ground truth)
        valid_classes = self.total_area_label > 0
        mean_precision = np.mean(precision_per_class[valid_classes])
        mean_recall = np.mean(recall_per_class[valid_classes])
        mean_f1 = np.mean(f1_per_class[valid_classes])
        
        results = {
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist()
        }
        
        return results
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Láº¥y táº¥t cáº£ metrics
        
        Returns:
            Dict chá»©a táº¥t cáº£ metrics
        """
        results = {}
        
        # IoU metrics
        iou_results = self.get_iou()
        results.update(iou_results)
        
        # Dice metrics
        dice_results = self.get_dice()
        results.update(dice_results)
        
        # Accuracy
        results['accuracy'] = self.get_accuracy()
        
        # Precision/Recall
        pr_results = self.get_precision_recall()
        results.update(pr_results)
        
        return results

def calculate_hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    TÃ­nh Hausdorff distance giá»¯a prediction vÃ  ground truth
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        Hausdorff distance
    """
    try:
        import cv2
        from scipy.spatial.distance import directed_hausdorff
        
        # TÃ¬m contours
        pred_contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(pred_contours) == 0 or len(target_contours) == 0:
            return float('inf')
        
        # Láº¥y táº¥t cáº£ Ä‘iá»ƒm contour
        pred_points = np.concatenate([contour.reshape(-1, 2) for contour in pred_contours])
        target_points = np.concatenate([contour.reshape(-1, 2) for contour in target_contours])
        
        # TÃ­nh Hausdorff distance
        hausdorff_dist = max(
            directed_hausdorff(pred_points, target_points)[0],
            directed_hausdorff(target_points, pred_points)[0]
        )
        
        return hausdorff_dist
        
    except ImportError:
        logger.warning("KhÃ´ng thá»ƒ tÃ­nh Hausdorff distance: thiáº¿u scipy")
        return 0.0

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 3,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Ä�Ã¡nh giÃ¡ model trÃªn validation/test set
    
    Args:
        model: PyTorch model
        dataloader: DataLoader cho evaluation
        device: Device Ä‘á»ƒ cháº¡y model
        num_classes: Sá»‘ lÆ°á»£ng classes
        class_names: TÃªn cÃ¡c classes
        
    Returns:
        Dict chá»©a cÃ¡c metrics
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    model.eval()
    metrics_calculator = SegmentationMetrics(num_classes)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if hasattr(model, 'model'):  # SegformerModel wrapper
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
            else:  # Direct model
                logits = model(pixel_values)
            
            # Upsample logits to match label size
            logits = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Convert to numpy
            pred_np = predictions.cpu().numpy()
            target_np = labels.cpu().numpy()
            
            # Update metrics
            metrics_calculator.update(pred_np, target_np)
            
            # Store for confusion matrix
            all_predictions.extend(pred_np.flatten())
            all_targets.extend(target_np.flatten())
    
    # TÃ­nh táº¥t cáº£ metrics
    results = metrics_calculator.get_all_metrics()
    
    # ThÃªm confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=range(num_classes))
    results['confusion_matrix'] = cm.tolist()
    
    # Classification report
    try:
        class_report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        results['classification_report'] = class_report
    except Exception as e:
        logger.warning(f"KhÃ´ng thá»ƒ táº¡o classification report: {e}")
    
    return results

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
):
    """
    Váº½ confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: TÃªn cÃ¡c classes
        title: TiÃªu Ä‘á»� biá»ƒu Ä‘á»“
        save_path: Ä�Æ°á»�ng dáº«n lÆ°u áº£nh
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Ä�Ã£ lÆ°u confusion matrix: {save_path}")
    
    plt.show()

def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str] = ['mean_iou', 'mean_dice', 'accuracy'],
    save_path: Optional[str] = None
):
    """
    Váº½ biá»ƒu Ä‘á»“ so sÃ¡nh metrics giá»¯a cÃ¡c models
    
    Args:
        metrics_dict: Dict chá»©a metrics cá»§a cÃ¡c models
        metric_names: Danh sÃ¡ch metrics cáº§n so sÃ¡nh
        save_path: Ä�Æ°á»�ng dáº«n lÆ°u áº£nh
    """
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]
    
    for idx, metric_name in enumerate(metric_names):
        model_names = list(metrics_dict.keys())
        metric_values = [metrics_dict[model][metric_name] for model in model_names]
        
        bars = axes[idx].bar(model_names, metric_values)
        axes[idx].set_title(f'{metric_name.replace("_", " ").title()}')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim(0, 1)
        
        # ThÃªm giÃ¡ trá»‹ lÃªn bars
        for bar, value in zip(bars, metric_values):
            axes[idx].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{value:.3f}',
                ha='center',
                va='bottom'
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Ä�Ã£ lÆ°u metrics comparison: {save_path}")
    
    plt.show()

def calculate_class_wise_metrics(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    num_classes: int = 3,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    TÃ­nh metrics chi tiáº¿t cho tá»«ng class
    
    Args:
        predictions: List predicted masks
        targets: List ground truth masks
        num_classes: Sá»‘ lÆ°á»£ng classes
        class_names: TÃªn cÃ¡c classes
        
    Returns:
        Dict chá»©a metrics chi tiáº¿t cho tá»«ng class
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    metrics_calculator = SegmentationMetrics(num_classes)
    
    for pred, target in zip(predictions, targets):
        metrics_calculator.update(pred, target)
    
    results = metrics_calculator.get_all_metrics()
    
    # Táº¡o detailed report cho tá»«ng class
    detailed_results = {}
    
    for class_idx, class_name in enumerate(class_names):
        detailed_results[class_name] = {
            'iou': results['iou_per_class'][class_idx],
            'dice': results['dice_per_class'][class_idx],
            'precision': results['precision_per_class'][class_idx],
            'recall': results['recall_per_class'][class_idx],
            'f1': results['f1_per_class'][class_idx]
        }
    
    # ThÃªm overall metrics
    detailed_results['overall'] = {
        'mean_iou': results['mean_iou'],
        'mean_dice': results['mean_dice'],
        'accuracy': results['accuracy'],
        'mean_precision': results['mean_precision'],
        'mean_recall': results['mean_recall'],
        'mean_f1': results['mean_f1']
    }
    
    return detailed_results

def print_evaluation_report(results: Dict[str, Any], class_names: List[str]):
    """
    In bÃ¡o cÃ¡o Ä‘Ã¡nh giÃ¡ chi tiáº¿t
    
    Args:
        results: Results tá»« evaluate_model
        class_names: TÃªn cÃ¡c classes
    """
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    # Overall metrics
    print("\nğŸ“Š OVERALL METRICS:")
    print(f"  â€¢ Mean IoU: {results['mean_iou']:.4f}")
    print(f"  â€¢ Mean Dice: {results['mean_dice']:.4f}")
    print(f"  â€¢ Accuracy: {results['accuracy']:.4f}")
    print(f"  â€¢ Mean Precision: {results['mean_precision']:.4f}")
    print(f"  â€¢ Mean Recall: {results['mean_recall']:.4f}")
    print(f"  â€¢ Mean F1: {results['mean_f1']:.4f}")
    
    # Per-class metrics
    print("\nğŸ“ˆ PER-CLASS METRICS:")
    for class_idx, class_name in enumerate(class_names):
        print(f"\n  {class_name.upper()}:")
        print(f"    IoU: {results['iou_per_class'][class_idx]:.4f}")
        print(f"    Dice: {results['dice_per_class'][class_idx]:.4f}")
        print(f"    Precision: {results['precision_per_class'][class_idx]:.4f}")
        print(f"    Recall: {results['recall_per_class'][class_idx]:.4f}")
        print(f"    F1: {results['f1_per_class'][class_idx]:.4f}")
    
    print("\n" + "=" * 60)

def compare_models(
    model_results: Dict[str, Dict[str, Any]],
    class_names: List[str]
) -> Dict[str, str]:
    """
    So sÃ¡nh performance cá»§a cÃ¡c models
    
    Args:
        model_results: Dict chá»©a results cá»§a cÃ¡c models
        class_names: TÃªn cÃ¡c classes
        
    Returns:
        Dict chá»©a model tá»‘t nháº¥t cho tá»«ng metric
    """
    metrics_to_compare = ['mean_iou', 'mean_dice', 'accuracy', 'mean_f1']
    best_models = {}
    
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    for metric in metrics_to_compare:
        best_model = None
        best_score = -1
        
        print(f"\nğŸ“Š {metric.replace('_', ' ').upper()}:")
        
        for model_name, results in model_results.items():
            score = results[metric]
            print(f"  â€¢ {model_name}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        best_models[metric] = best_model
        print(f"  ğŸ�† Best: {best_model} ({best_score:.4f})")
    
    print("\n" + "=" * 60)
    
    return best_models
print('DONE')


"""
File chá»©a inference pipeline cho bÃ i toÃ¡n phÃ¢n Ä‘oáº¡n polyp
Má»¥c Ä‘Ã­ch: Predict trÃªn test set vÃ  táº¡o submission file
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import logging
from tqdm import tqdm
import cv2

# Import local modules
# Config Ä‘Ã£ Ä‘Æ°á»£c load tá»« cell trÆ°á»›c, khÃ´ng cáº§n import
# from config import Config
# from models import get_model
# from utils import (...)
# from data_processing import get_augmentation_pipeline

# Sá»­ dá»¥ng trá»±c tiáº¿p cÃ¡c class/function Ä‘Ã£ Ä‘Æ°á»£c load

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolypInference:
    """
    Inference class cho bÃ i toÃ¡n phÃ¢n Ä‘oáº¡n polyp
    
    Chá»©c nÄƒng:
    - Load trained model
    - Predict trÃªn test images
    - Apply Test Time Augmentation (TTA)
    - Post-processing predictions
    - Táº¡o submission file
    """
    
    def __init__(self, config: Config, model_path: str):
        """
        Khá»Ÿi táº¡o inference pipeline
        
        Args:
            config: Configuration object
            model_path: Ä�Æ°á»�ng dáº«n tá»›i trained model
        """
        self.config = config
        self.device = config.get_device()
        self.model_path = model_path
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load image processor
        try:
            from transformers import SegformerImageProcessor
            self.processor = SegformerImageProcessor(reduce_labels=False)
        except ImportError:
            self.processor = None
            logger.warning("SegformerImageProcessor khÃ´ng cÃ³ sáºµn")
        
        logger.info("Ä�Ã£ khá»Ÿi táº¡o inference pipeline")
    
    def _load_model(self) -> nn.Module:
        """
        Load trained model tá»« checkpoint
        
        Returns:
            Loaded PyTorch model
        """
        # Táº¡o model architecture
        model = get_model(
            model_name=self.config.model.model_name,
            num_classes=self.config.model.num_classes,
            backbone=self.config.model.backbone,
            id2label=self.config.model.id2label,
            label2id=self.config.model.label2id,
            pretrained=False  # KhÃ´ng cáº§n pretrained weights
        )
        
        # Load checkpoint
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Load model state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model tá»« checkpoint: {self.model_path}")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights: {self.model_path}")
        else:
            raise FileNotFoundError(f"KhÃ´ng tÃ¬m tháº¥y model: {self.model_path}")
        
        model.to(self.device)
        return model
    
    def predict_single_image(
        self, 
        image_path: str, 
        use_tta: bool = False,
        visualize: bool = False
    ) -> np.ndarray:
        """
        Predict segmentation cho má»™t áº£nh
        
        Args:
            image_path: Ä�Æ°á»�ng dáº«n tá»›i áº£nh
            use_tta: CÃ³ sá»­ dá»¥ng Test Time Augmentation khÃ´ng
            visualize: CÃ³ visualize káº¿t quáº£ khÃ´ng
            
        Returns:
            Predicted segmentation mask [H, W]
        """
        # Load vÃ  preprocess áº£nh
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        if use_tta and self.config.inference.use_tta:
            prediction = self._predict_with_tta(image)
        else:
            prediction = self._predict_single(image)
        
        # Resize vá»� kÃ­ch thÆ°á»›c gá»‘c
        prediction = cv2.resize(
            prediction.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Post-processing
        if self.config.inference.use_morphology:
            prediction = post_process_mask(prediction)
        
        # Visualization
        if visualize:
            image_np = np.array(image)
            visualize_sample(image_np, prediction, class_names=list(self.config.model.id2label.values()))
        
        return prediction
    
    def _predict_single(self, image: Image.Image) -> np.ndarray:
        """
        Predict cho má»™t áº£nh Ä‘Æ¡n láº»
        
        Args:
            image: PIL Image
            
        Returns:
            Predicted mask [H, W]
        """
        # Preprocess image
        if self.processor is not None:
            inputs = self.processor(image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
        else:
            # Fallback preprocessing
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            if hasattr(self.model, 'model'):  # SegformerModel wrapper
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
            else:  # Direct model
                logits = self.model(pixel_values)
            
            # Upsample logits to original image size
            logits = F.interpolate(
                logits,
                size=image.size[::-1],  # (height, width)
                mode='bilinear',
                align_corners=False
            )
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            prediction = predictions.cpu().numpy()[0]
        
        return prediction
    
    def _predict_with_tta(self, image: Image.Image) -> np.ndarray:
        """
        Predict vá»›i Test Time Augmentation
        
        Args:
            image: PIL Image
            
        Returns:
            Averaged prediction mask [H, W]
        """
        # Táº¡o augmentation pipeline
        tta_transforms = get_augmentation_pipeline(is_training=True)
        
        predictions = []
        image_np = np.array(image)
        
        # Original prediction
        predictions.append(self._predict_single(image))
        
        # Augmented predictions
        for _ in range(self.config.inference.tta_transforms - 1):
            if tta_transforms is not None:
                # Apply augmentation
                augmented = tta_transforms(image=image_np)
                aug_image = Image.fromarray(augmented['image'])
                
                # Predict
                pred = self._predict_single(aug_image)
                predictions.append(pred)
            else:
                # Fallback: simple flips
                if np.random.random() > 0.5:
                    flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    pred = self._predict_single(flipped_image)
                    pred = np.fliplr(pred)  # Flip prediction back
                    predictions.append(pred)
        
        # Average predictions
        averaged_prediction = np.mean(predictions, axis=0)
        final_prediction = np.argmax(averaged_prediction, axis=0) if len(averaged_prediction.shape) > 2 else averaged_prediction
        
        return final_prediction.astype(np.uint8)
    
    def predict_test_set(
        self, 
        test_image_paths: List[str],
        output_dir: str,
        use_tta: bool = False,
        save_masks: bool = True
    ) -> List[np.ndarray]:
        """
        Predict cho toÃ n bá»™ test set
        
        Args:
            test_image_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n áº£nh test
            output_dir: ThÆ° má»¥c lÆ°u káº¿t quáº£
            use_tta: CÃ³ sá»­ dá»¥ng TTA khÃ´ng
            save_masks: CÃ³ lÆ°u mask predictions khÃ´ng
            
        Returns:
            List predicted masks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        predictions = []
        
        logger.info(f"Báº¯t Ä‘áº§u predict {len(test_image_paths)} áº£nh test")
        
        for image_path in tqdm(test_image_paths, desc="Predicting"):
            try:
                # Predict
                prediction = self.predict_single_image(
                    image_path, 
                    use_tta=use_tta,
                    visualize=False
                )
                
                predictions.append(prediction)
                
                # LÆ°u mask náº¿u cáº§n
                if save_masks:
                    image_name = os.path.basename(image_path).split('.')[0]
                    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
                    
                    # Convert prediction to RGB for visualization
                    mask_rgb = self._prediction_to_rgb(prediction)
                    Image.fromarray(mask_rgb).save(mask_path)
                
            except Exception as e:
                logger.error(f"Lá»—i khi predict {image_path}: {str(e)}")
                # Táº¡o empty prediction
                predictions.append(np.zeros((512, 512), dtype=np.uint8))
        
        logger.info(f"HoÃ n thÃ nh prediction cho {len(predictions)} áº£nh")
        
        return predictions
    
    def _prediction_to_rgb(self, prediction: np.ndarray) -> np.ndarray:
        """
        Chuyá»ƒn prediction mask thÃ nh RGB cho visualization
        
        Args:
            prediction: Prediction mask [H, W]
            
        Returns:
            RGB mask [H, W, 3]
        """
        # Color mapping
        colors = np.array([
            [255, 0, 0],      # Red cho neoplastic
            [0, 255, 0],      # Green cho non-neoplastic
            [0, 0, 255]       # Blue cho background
        ])
        
        rgb_mask = colors[prediction]
        return rgb_mask.astype(np.uint8)
    
    def create_submission(
        self,
        test_image_paths: List[str],
        predictions: List[np.ndarray],
        output_path: str = "submission.csv"
    ):
        """
        Táº¡o submission file cho Kaggle
        
        Args:
            test_image_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n áº£nh test
            predictions: Danh sÃ¡ch predicted masks
            output_path: Ä�Æ°á»�ng dáº«n file submission
        """
        # Láº¥y image IDs
        image_ids = []
        for image_path in test_image_paths:
            image_name = os.path.basename(image_path)
            image_id = image_name.split('.')[0]
            image_ids.append(image_id)
        
        # Táº¡o submission file
        create_submission_file(predictions, image_ids, output_path)
        
        logger.info(f"Ä�Ã£ táº¡o submission file: {output_path}")
    
    def batch_inference(
        self,
        test_image_paths: List[str],
        output_dir: str,
        submission_path: str = "submission.csv",
        use_tta: bool = False,
        save_masks: bool = True
    ):
        """
        Thá»±c hiá»‡n inference cho toÃ n bá»™ test set vÃ  táº¡o submission
        
        Args:
            test_image_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n áº£nh test
            output_dir: ThÆ° má»¥c lÆ°u káº¿t quáº£
            submission_path: Ä�Æ°á»�ng dáº«n file submission
            use_tta: CÃ³ sá»­ dá»¥ng TTA khÃ´ng
            save_masks: CÃ³ lÆ°u mask predictions khÃ´ng
        """
        # Predict test set
        predictions = self.predict_test_set(
            test_image_paths,
            output_dir,
            use_tta=use_tta,
            save_masks=save_masks
        )
        
        # Táº¡o submission file
        self.create_submission(
            test_image_paths,
            predictions,
            os.path.join(output_dir, submission_path)
        )
        
        logger.info("HoÃ n thÃ nh batch inference")

def run_inference(
    config: Config,
    model_path: str,
    test_image_paths: List[str],
    output_dir: str,
    use_tta: bool = False
) -> List[np.ndarray]:
    """
    Function wrapper Ä‘á»ƒ cháº¡y inference
    
    Args:
        config: Configuration object
        model_path: Ä�Æ°á»�ng dáº«n tá»›i trained model
        test_image_paths: Danh sÃ¡ch Ä‘Æ°á»�ng dáº«n áº£nh test
        output_dir: ThÆ° má»¥c lÆ°u káº¿t quáº£
        use_tta: CÃ³ sá»­ dá»¥ng TTA khÃ´ng
        
    Returns:
        List predicted masks
    """
    # Táº¡o inference pipeline
    inference = PolypInference(config, model_path)
    
    # Cháº¡y batch inference
    inference.batch_inference(
        test_image_paths,
        output_dir,
        use_tta=use_tta,
        save_masks=True
    )
    
    # Predict vÃ  return
    predictions = inference.predict_test_set(
        test_image_paths,
        output_dir,
        use_tta=use_tta,
        save_masks=False
    )
    
    return predictions

def ensemble_predictions(
    predictions_list: List[List[np.ndarray]],
    method: str = 'voting'
) -> List[np.ndarray]:
    """
    Ensemble predictions tá»« nhiá»�u models
    
    Args:
        predictions_list: List cÃ¡c predictions tá»« cÃ¡c models khÃ¡c nhau
        method: PhÆ°Æ¡ng phÃ¡p ensemble ('voting', 'averaging')
        
    Returns:
        Ensembled predictions
    """
    if len(predictions_list) == 1:
        return predictions_list[0]
    
    n_samples = len(predictions_list[0])
    ensembled_predictions = []
    
    for i in range(n_samples):
        # Láº¥y predictions cá»§a sample i tá»« táº¥t cáº£ models
        sample_predictions = [preds[i] for preds in predictions_list]
        
        if method == 'voting':
            # Majority voting
            stacked_preds = np.stack(sample_predictions, axis=0)
            ensembled_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 
                axis=0, 
                arr=stacked_preds
            )
        elif method == 'averaging':
            # Average probabilities (if available)
            averaged_pred = np.mean(sample_predictions, axis=0)
            ensembled_pred = np.argmax(averaged_pred, axis=-1) if len(averaged_pred.shape) > 2 else averaged_pred
        else:
            raise ValueError(f"Ensemble method khÃ´ng há»— trá»£: {method}")
        
        ensembled_predictions.append(ensembled_pred.astype(np.uint8))
    
    logger.info(f"Ensembled {len(predictions_list)} models vá»›i method: {method}")
    
    return ensembled_predictions

def compare_predictions(
    predictions_list: List[List[np.ndarray]],
    model_names: List[str],
    test_image_paths: List[str],
    output_dir: str,
    num_samples: int = 5
):
    """
    So sÃ¡nh predictions tá»« cÃ¡c models khÃ¡c nhau
    
    Args:
        predictions_list: List predictions tá»« cÃ¡c models
        model_names: TÃªn cÃ¡c models
        test_image_paths: Ä�Æ°á»�ng dáº«n áº£nh test
        output_dir: ThÆ° má»¥c lÆ°u káº¿t quáº£
        num_samples: Sá»‘ samples Ä‘á»ƒ so sÃ¡nh
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Chá»�n random samples Ä‘á»ƒ so sÃ¡nh
    import random
    sample_indices = random.sample(range(len(test_image_paths)), min(num_samples, len(test_image_paths)))
    
    for idx in sample_indices:
        image_path = test_image_paths[idx]
        image_name = os.path.basename(image_path).split('.')[0]
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Táº¡o comparison plot
        n_models = len(predictions_list)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predictions tá»« cÃ¡c models
        for model_idx, (predictions, model_name) in enumerate(zip(predictions_list, model_names)):
            prediction = predictions[idx]
            
            # Convert to RGB
            colors = np.array([
                [255, 0, 0],      # Red
                [0, 255, 0],      # Green
                [0, 0, 255]       # Blue
            ])
            pred_rgb = colors[prediction]
            
            axes[model_idx + 1].imshow(pred_rgb.astype(np.uint8))
            axes[model_idx + 1].set_title(f'{model_name}')
            axes[model_idx + 1].axis('off')
        
        plt.tight_layout()
        
        # LÆ°u comparison
        comparison_path = os.path.join(output_dir, f"comparison_{image_name}.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Ä�Ã£ táº¡o comparison cho {len(sample_indices)} samples")

# Import matplotlib Ä‘á»ƒ trÃ¡nh lá»—i
import matplotlib.pyplot as plt 
print('DONE')


"""
File training riÃªng cho Segformer model
Má»¥c Ä‘Ã­ch: Training Segformer model vá»›i cáº¥u hÃ¬nh tá»‘i Æ°u
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import time

# Import WandB for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
    print("âœ… WandB imported successfully")
    
    # Login to WandB with provided key
    wandb.login(key="d6fd0da4e97ed749ef3eeacb295c97e7ab2e6aa7")
    print("ğŸ”� WandB logged in successfully with provided key")
except ImportError:
    WANDB_AVAILABLE = False
    print("â�Œ WandB not available. Install with: pip install wandb")
except Exception as e:
    print(f"âš ï¸� WandB login failed: {e}")
    WANDB_AVAILABLE = False

# Import transformers
try:
    from transformers import SegformerImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Fallback: táº¡o dummy processor
    class DummyProcessor:
        def __init__(self, reduce_labels=False):
            pass
    SegformerImageProcessor = DummyProcessor

# Tá»‘i Æ°u tqdm cho Jupyter notebook
try:
    from tqdm.notebook import tqdm
    NOTEBOOK_TQDM = True
except ImportError:
    from tqdm import tqdm
    NOTEBOOK_TQDM = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegformerTrainer:
    """
    Trainer class riÃªng cho Segformer model
    
    Chá»©c nÄƒng:
    - Training Segformer vá»›i cáº¥u hÃ¬nh tá»‘i Æ°u
    - Early stopping vÃ  learning rate scheduling
    - Checkpoint management
    - Validation vÃ  metrics tracking
    - Advanced WandB visualization vÃ  tracking
    """
    
    def __init__(self, config):
        """
        Khá»Ÿi táº¡o Segformer trainer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = config.get_device()
        
        # Tá»‘i Æ°u hÃ³a memory trÆ°á»›c khi load model
        import torch
        torch.cuda.empty_cache()
        
        # Set random seed
        if hasattr(config.experiment, 'seed'):
            import random
            import numpy as np
            torch.manual_seed(config.experiment.seed)
            np.random.seed(config.experiment.seed)
            random.seed(config.experiment.seed)
        else:
            import random
            import numpy as np
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
        
        # Táº¡o thÆ° má»¥c cáº§n thiáº¿t
        config.create_directories()
        
        # Initialize processor
        if TRANSFORMERS_AVAILABLE:
            self.processor = SegformerImageProcessor()
        else:
            raise ImportError("Transformers library khÃ´ng cÃ³ sáºµn cho Segformer")
        
        # Initialize WandB if available (PHáº¢I Gá»ŒI TRÆ¯á»šC KHI Táº OTHER MODEL)
        if WANDB_AVAILABLE:
            wandb.init(
                settings=wandb.Settings(init_timeout=300),
                project="medical-segmentation-segformer",
                name=f"segformer_{config.experiment.run_name}",
                config={
                    "model": "Segformer",
                    "backbone": config.model.backbone,
                    "architecture": "MIT Encoder + MLP Decoder",
                    "learning_rate": config.training.learning_rate,
                    "batch_size": config.data.batch_size,
                    "epochs": config.training.epochs,
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingLR",
                    "loss_function": "CrossEntropyLoss",
                    "num_classes": config.model.num_classes,
                    "seed": config.experiment.seed if hasattr(config.experiment, 'seed') else 42,
                    "patience": config.training.patience,
                    "weight_decay": config.training.weight_decay,
                    "gradient_clipping": 1.0,
                    "gradient_checkpointing": True,
                    "use_pretrained": config.model.use_pretrained
                },
                tags=["segformer", "transformer", "medical-segmentation", "polyp-detection"],
                notes="Segformer training vá»›i Transformer architecture vÃ  advanced tracking"
            )
            logger.info("âœ… WandB experiment initialized with comprehensive tracking")
        
        # Initialize model vá»›i gradient checkpointing Ä‘á»ƒ tiáº¿t kiá»‡m memory
        self.model = self._build_model()
        
        # Enable gradient checkpointing Ä‘á»ƒ tiáº¿t kiá»‡m memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.model.to(self.device)
        
        # Watch model with WandB (SAU KHI Ä�Ãƒ INIT)
        if WANDB_AVAILABLE:
            wandb.watch(self.model, log="all", log_freq=100)
            logger.info("ğŸ”� WandB watching model parameters and gradients")
        
        # Initialize optimizer vÃ  scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(config.model.num_classes)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_iou = 0.0
        self.best_val_dice = 0.0
        self.patience_counter = 0
        
        logger.info("Ä�Ã£ khá»Ÿi táº¡o SegformerTrainer vá»›i advanced tracking")
    
    def _build_model(self) -> SegformerModel:
        """
        Táº¡o Segformer model
        
        Returns:
            SegformerModel instance
        """
        model = SegformerModel(
            model_name=self.config.model.backbone,
            num_classes=self.config.model.num_classes,
            id2label=self.config.model.id2label,
            label2id=self.config.model.label2id,
            pretrained=self.config.model.use_pretrained
        )
        
        logger.info(f"Ä�Ã£ táº¡o Segformer model: {self.config.model.backbone}")
        
        # In thÃ´ng tin model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _build_optimizer(self) -> optim.Optimizer:
        """
        Táº¡o optimizer cho Segformer
        
        Returns:
            PyTorch optimizer
        """
        # Sá»­ dá»¥ng AdamW vá»›i weight decay
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        logger.info(f"Ä�Ã£ táº¡o AdamW optimizer vá»›i lr={self.config.training.learning_rate}")
        
        return optimizer
    
    def _build_scheduler(self) -> optim.lr_scheduler.CosineAnnealingLR:
        """
        Táº¡o learning rate scheduler
        
        Returns:
            PyTorch scheduler
        """
        # Sá»­ dá»¥ng CosineAnnealingLR cho Segformer
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.epochs,
            eta_min=1e-6
        )
        
        logger.info("Ä�Ã£ táº¡o CosineAnnealingLR scheduler")
        
        return scheduler
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Training má»™t epoch
        
        Args:
            train_loader: DataLoader cho training
            epoch: Epoch hiá»‡n táº¡i
            
        Returns:
            Dict chá»©a training metrics
        """
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1} - Training",
            leave=True,
            position=0,
            dynamic_ncols=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate predictions for metrics
            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                predictions = torch.argmax(upsampled_logits, dim=1)
                
                self.metrics.update(
                    predictions.cpu().numpy(),
                    labels.cpu().numpy()
                )
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.metrics.get_all_metrics()
        
        results = {
            'loss': avg_loss,
            'iou': metrics['mean_iou'],
            'dice': metrics['mean_dice'],
            'accuracy': metrics['accuracy']
        }
        
        return results
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validation má»™t epoch
        
        Args:
            val_loader: DataLoader cho validation
            epoch: Epoch hiá»‡n táº¡i
            
        Returns:
            Dict chá»©a validation metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation")
            
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                # Calculate predictions for metrics
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                predictions = torch.argmax(upsampled_logits, dim=1)
                
                self.metrics.update(
                    predictions.cpu().numpy(),
                    labels.cpu().numpy()
                )
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.metrics.get_all_metrics()
        
        results = {
            'loss': avg_loss,
            'iou': metrics['mean_iou'],
            'dice': metrics['mean_dice'],
            'accuracy': metrics['accuracy']
        }
        
        return results
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Main training loop cho Segformer vá»›i advanced WandB tracking
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            
        Returns:
            Training history
        """
        logger.info(f"Báº¯t Ä‘áº§u training Segformer vá»›i {self.config.training.epochs} epochs")
        logger.info("="*80)
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_results = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_results = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['train_iou'].append(train_results['iou'])
            self.history['val_iou'].append(val_results['iou'])
            self.history['train_dice'].append(train_results['dice'])
            self.history['val_dice'].append(val_results['dice'])
            self.history['learning_rates'].append(current_lr)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Format vÃ  hiá»ƒn thá»‹ káº¿t quáº£ chi tiáº¿t
            print(f"\nğŸ“Š EPOCH {epoch+1:3d}/{self.config.training.epochs} | Time: {epoch_time:6.2f}s | LR: {current_lr:.2e}")
            print(f"   ğŸ”¹ TRAIN â†’ Loss: {train_results['loss']:.4f} | IoU: {train_results['iou']:.4f} | Dice: {train_results['dice']:.4f} | Acc: {train_results['accuracy']:.4f}")
            print(f"   ğŸ”¸ VAL   â†’ Loss: {val_results['loss']:.4f} | IoU: {val_results['iou']:.4f} | Dice: {val_results['dice']:.4f} | Acc: {val_results['accuracy']:.4f}")
            
            # Log to WandB with detailed metrics
            if WANDB_AVAILABLE:
                # Log individual metrics
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_results['loss'],
                    "train/iou": train_results['iou'],
                    "train/dice": train_results['dice'],
                    "train/accuracy": train_results['accuracy'],
                    "val/loss": val_results['loss'],
                    "val/iou": val_results['iou'],
                    "val/dice": val_results['dice'],
                    "val/accuracy": val_results['accuracy'],
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time,
                    "best_val_iou": self.best_val_iou,
                    "patience_counter": self.patience_counter,
                    # Custom comparison metrics
                    "metrics/loss_diff": train_results['loss'] - val_results['loss'],
                    "metrics/iou_diff": train_results['iou'] - val_results['iou'],
                    "metrics/dice_diff": train_results['dice'] - val_results['dice'],
                    "metrics/val_improvement": val_results['iou'] - self.best_val_iou if val_results['iou'] > self.best_val_iou else 0
                })
                
                # Create custom comparison table every 10 epochs
                if (epoch + 1) % 10 == 0:
                    comparison_table = wandb.Table(
                        columns=["Metric", "Train", "Validation", "Difference"],
                        data=[
                            ["Loss", f"{train_results['loss']:.4f}", f"{val_results['loss']:.4f}", f"{train_results['loss'] - val_results['loss']:.4f}"],
                            ["IoU", f"{train_results['iou']:.4f}", f"{val_results['iou']:.4f}", f"{train_results['iou'] - val_results['iou']:.4f}"],
                            ["Dice", f"{train_results['dice']:.4f}", f"{val_results['dice']:.4f}", f"{train_results['dice'] - val_results['dice']:.4f}"],
                            ["Accuracy", f"{train_results['accuracy']:.4f}", f"{val_results['accuracy']:.4f}", f"{train_results['accuracy'] - val_results['accuracy']:.4f}"]
                        ]
                    )
                    wandb.log({f"metrics_comparison_epoch_{epoch+1}": comparison_table})
            
            # Save best model
            if val_results['iou'] > self.best_val_iou:
                self.best_val_iou = val_results['iou']
                self.best_val_dice = val_results['dice']
                self.patience_counter = 0
                
                print(f"   â­� NEW BEST MODEL! Val IoU: {val_results['iou']:.4f} (â†‘{val_results['iou'] - self.best_val_iou:.4f})")
                
                # Save checkpoint
                best_model_path = os.path.join(
                    self.config.experiment.model_save_dir,
                    f"best_segformer_{self.config.experiment.run_name}.pth"
                )
                
                save_model_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_results['loss'],
                    val_results,
                    best_model_path
                )
                
                # Log model artifact to WandB
                if WANDB_AVAILABLE:
                    artifact = wandb.Artifact(
                        name=f"segformer_best_model_{self.config.experiment.run_name}",
                        type="model",
                        description=f"Best Segformer model at epoch {epoch + 1} with Val IoU: {val_results['iou']:.4f}"
                    )
                    artifact.add_file(best_model_path)
                    wandb.log_artifact(artifact)
                
                # Save results to JSON file
                import json
                results_data = {
                    'model_name': 'Segformer',
                    'backbone': self.config.model.backbone,
                    'epoch': epoch + 1,
                    'best_val_iou': float(val_results['iou']),
                    'best_val_dice': float(val_results['dice']),
                    'best_val_accuracy': float(val_results['accuracy']),
                    'best_val_loss': float(val_results['loss']),
                    'total_params': sum(p.numel() for p in self.model.parameters()),
                    'training_time': time.time() - start_time
                }
                
                results_file = os.path.join(
                    self.config.experiment.log_dir,
                    f"segformer_results_{self.config.experiment.run_name}.json"
                )
                with open(results_file, 'w') as f:
                    json.dump(results_data, f, indent=2)
                
            else:
                self.patience_counter += 1
                print(f"   â�³ No improvement for {self.patience_counter} epochs (Best: {self.best_val_iou:.4f})")
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"\nğŸ›‘ EARLY STOPPING at epoch {epoch+1} (patience: {self.config.training.patience})")
                break
                
            print("-" * 80)
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nğŸ�‰ TRAINING COMPLETED!")
        print(f"   â�±ï¸�  Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"   ğŸ�† Best Val IoU: {self.best_val_iou:.4f}")
        print(f"   ğŸ“ˆ Total epochs: {len(self.history['train_loss'])}")
        print("="*80)
        
        # Log final summary to WandB
        if WANDB_AVAILABLE:
            wandb.summary.update({
                "best_val_iou": self.best_val_iou,
                "best_val_dice": max(self.history['val_dice']) if self.history['val_dice'] else 0.0,
                "final_train_loss": self.history['train_loss'][-1] if self.history['train_loss'] else 0.0,
                "final_val_loss": self.history['val_loss'][-1] if self.history['val_loss'] else 0.0,
                "total_training_time": total_time,
                "total_epochs": len(self.history['train_loss']),
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            })
            
            # Finish WandB run
            wandb.finish()
            logger.info("ğŸ�� WandB experiment finished")
        
        # Save final results
        self._save_final_results(total_time)
        
        return self.history
    
    def _save_final_results(self, total_time: float):
        """
        LÆ°u káº¿t quáº£ cuá»‘i cÃ¹ng vÃ o file JSON
        
        Args:
            total_time: Tá»•ng thá»�i gian training
        """
        import json
        
        results_data = {
            'model_name': 'Segformer',
            'backbone': self.config.model.backbone,
            'best_val_iou': float(max(self.history['val_iou'])),
            'best_val_dice': float(self.best_val_dice),
            'best_val_accuracy': float(max([0.0] + [acc for acc in self.history.get('val_accuracy', [0.0])])),
            'final_train_loss': float(self.history['train_loss'][-1]) if self.history['train_loss'] else 0.0,
            'final_val_loss': float(self.history['val_loss'][-1]) if self.history['val_loss'] else 0.0,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'training_time_seconds': float(total_time),
            'training_time_hours': float(total_time / 3600),
            'total_epochs': len(self.history['train_loss']),
            'learning_rate': self.config.training.learning_rate,
            'batch_size': self.config.data.batch_size,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'loss_function': 'CrossEntropyLoss'
        }
        
        results_file = os.path.join(
            self.config.experiment.log_dir,
            f"segformer_final_results_{self.config.experiment.run_name}.json"
        )
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Ä�Ã£ lÆ°u káº¿t quáº£ cuá»‘i cÃ¹ng: {results_file}")
    
    def save_training_plots(self, save_dir: str):
        """
        LÆ°u biá»ƒu Ä‘á»“ training history
        
        Args:
            save_dir: ThÆ° má»¥c lÆ°u plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare data for plotting
        train_metrics = {
            'iou': self.history['train_iou'],
            'dice': self.history['train_dice']
        }
        
        val_metrics = {
            'iou': self.history['val_iou'],
            'dice': self.history['val_dice']
        }
        
        # Plot training history
        plot_path = os.path.join(save_dir, f"segformer_training_history_{self.config.experiment.run_name}.png")
        plot_training_history(
            self.history['train_loss'],
            self.history['val_loss'],
            train_metrics,
            val_metrics,
            save_path=plot_path
        )
        
        # Plot learning rate
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['learning_rates'])
        plt.title('Learning Rate Schedule (Cosine Annealing)')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        lr_plot_path = os.path.join(save_dir, f"segformer_lr_schedule_{self.config.experiment.run_name}.png")
        plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Ä�Ã£ lÆ°u training plots: {save_dir}")

def train_segformer_model(
    config,
    train_image_dir: str,
    train_mask_dir: str
) -> Tuple[SegformerModel, Dict[str, List[float]]]:
    """
    Function chÃ­nh Ä‘á»ƒ train Segformer model
    
    Args:
        config: Configuration object
        train_image_dir: ThÆ° má»¥c áº£nh training
        train_mask_dir: ThÆ° má»¥c mask training
        
    Returns:
        Tuple (trained_model, training_history)
    """
    # Láº¥y Ä‘Æ°á»�ng dáº«n file
    image_paths, mask_paths = get_file_paths(train_image_dir, train_mask_dir)
    
    # Chia dá»¯ liá»‡u
    train_images, train_masks, val_images, val_masks, _, _ = create_data_splits(
        image_paths,
        mask_paths,
        train_split=0.8,
        val_split=0.2,
        seed=config.experiment.seed if hasattr(config.experiment, 'seed') else 42
    )
    
    # Táº¡o processor
    processor = SegformerImageProcessor()
    
    # Táº¡o dataloaders
    train_loader, val_loader = create_dataloaders(
        train_images,
        train_masks,
        val_images,
        val_masks,
        processor,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        use_augmentation=True
    )
    
    # Táº¡o trainer
    trainer = SegformerTrainer(config)
    
    # Training
    history = trainer.train(train_loader, val_loader)
    
    # LÆ°u training plots
    trainer.save_training_plots(config.experiment.log_dir)
    
    return trainer.model, history

# Example usage
if __name__ == "__main__":
    # Táº¡o config cho Segformer
    config = Config()
    
    # Model config
    config.model.model_name = "segformer"
    config.model.backbone = "nvidia/mit-b5"
    config.model.num_classes = 3
    
    # Data config
    config.data.batch_size = 4  # Giáº£m batch size Ä‘á»ƒ trÃ¡nh OOM
    
    # Training config
    config.training.learning_rate = 1e-5  # Giáº£m learning rate do batch size nhá»� hÆ¡n
    config.training.epochs = 120  # Giáº£m epochs Ä‘á»ƒ test
    config.training.patience = 30  # Giáº£m patience tÆ°Æ¡ng á»©ng
    
    # Experiment config
    config.experiment.run_name = "segformer_experiment"
    
    # Tá»‘i Æ°u hÃ³a memory
    import torch
    torch.cuda.empty_cache()  # XÃ³a cache GPU
    
    # Ä�Æ°á»�ng dáº«n data
    train_image_dir = "/kaggle/input/bkai-igh-neopolyp/train/train"
    train_mask_dir = "/kaggle/input/bkai-igh-neopolyp/train_gt/train_gt"
    
    # Train model
    model, history = train_segformer_model(
        config,
        train_image_dir,
        train_mask_dir
    )
    
    logger.info("Segformer training hoÃ n thÃ nh!") 


"""
File test riÃªng cho Segformer model
Má»¥c Ä‘Ã­ch: Test Segformer model vá»›i Ä‘áº§u vÃ o lÃ  má»™t áº£nh, Ä‘áº§u ra phÃ¢n vÃ¹ng khá»‘i u Ä‘Æ°á»£c highlight
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import logging
from tqdm import tqdm

# Import transformers náº¿u cÃ³
try:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    TRANSFORMERS_AVAILABLE = True
    print("âœ… Transformers imported successfully")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("â�Œ Transformers not available. Install with: pip install transformers")

class ModelConfig:
    """Model configuration"""
    def __init__(self):
        self.backbone = "nvidia/mit-b5"  # PHáº¢I KHá»šP Vá»šI MODEL Ä�Ãƒ TRAIN
        self.num_classes = 3
        self.id2label = {0: "neoplastic", 1: "non-neoplastic", 2: "background"}
        self.label2id = {"neoplastic": 0, "non-neoplastic": 1, "background": 2}

class Config:
    """Simple config class tá»± Ä‘á»‹nh nghÄ©a"""
    def __init__(self):
        # Model config
        self.model = ModelConfig()

def get_device():
    """Láº¥y device phÃ¹ há»£p"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"âœ… Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("âš ï¸� Using CPU")
    return device

class SegformerModelFixed(nn.Module):
    """Segformer model Ä‘Æ°á»£c fix Ä‘á»ƒ match vá»›i checkpoint Ä‘Ã£ train"""
    
    def __init__(self, model_name: str = "nvidia/mit-b0", num_classes: int = 3, **kwargs):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library cáº§n thiáº¿t cho Segformer")
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained Segformer
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        print(f"âœ… Táº¡o SegformerModelFixed vá»›i {num_classes} classes")
    
    def forward(self, pixel_values):
        """Forward pass"""
        outputs = self.segformer(pixel_values=pixel_values)
        return outputs

class SegformerTester:
    """Test Segformer model vá»›i visualization"""
    
    def __init__(self, model_path: str, config=None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library cáº§n thiáº¿t cho Segformer")
            
        self.model_path = model_path
        self.config = config or Config()
        self.device = get_device()
        
        # Ä�á»‹nh nghÄ©a mÃ u sáº¯c
        self.colors = {
            0: [255, 0, 0],      # Ä�á»� cho neoplastic  
            1: [255, 165, 0],    # Cam cho non-neoplastic
            2: [0, 0, 0]         # Ä�en cho background
        }
        
        self.highlight_colors = {
            0: [255, 80, 80],    # Ä�á»� sÃ¡ng cho neoplastic
            1: [255, 180, 80],   # Cam sÃ¡ng cho non-neoplastic
        }
        
        # Initialize processor
        self.processor = SegformerImageProcessor()
        
        # Load model
        self.model = self._load_segformer_model()
        self.model.eval()
        
        print("âœ… Ä�Ã£ khá»Ÿi táº¡o SegformerTester")
    
    def _load_segformer_model(self):
        """Load model vá»›i architecture Ä‘Ãºng"""
        # Táº¡o model architecture
        model = SegformerModelFixed(
            model_name=self.config.model.backbone,
            num_classes=self.config.model.num_classes
        )
        
        # Load checkpoint vá»›i weights_only=False
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Láº¥y state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"ğŸ“¦ Loading from model_state_dict")
            else:
                state_dict = checkpoint
                print(f"ğŸ“¦ Loading from direct state_dict")
            
            # Sá»­a prefix mismatch: "model." -> ""
            corrected_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    # Bá»� prefix "model." vÃ  thay báº±ng "segformer."
                    new_key = key.replace('model.', 'segformer.')
                    corrected_state_dict[new_key] = value
                else:
                    corrected_state_dict[key] = value
            
            # Load vá»›i strict=False Ä‘á»ƒ bá»� qua keys khÃ´ng khá»›p
            missing_keys, unexpected_keys = model.load_state_dict(corrected_state_dict, strict=False)
            
            if missing_keys:
                print(f"âš ï¸� Missing keys: {len(missing_keys)} keys (may be normal)")
                if len(missing_keys) <= 5:  # Chá»‰ in ra náº¿u Ã­t keys
                    for key in missing_keys[:5]:
                        print(f"   - {key}")
            
            if unexpected_keys:
                print(f"âš ï¸� Unexpected keys: {len(unexpected_keys)} keys (may be normal)")
                if len(unexpected_keys) <= 5:  # Chá»‰ in ra náº¿u Ã­t keys
                    for key in unexpected_keys[:5]:
                        print(f"   - {key}")
                        
            print(f"âœ… Loaded Segformer tá»« checkpoint: {self.model_path}")
            
        else:
            raise FileNotFoundError(f"â�Œ KhÃ´ng tÃ¬m tháº¥y model: {self.model_path}")
        
        model.to(self.device)
        return model
    
    def predict_single_image(self, image_path: str):
        """Predict segmentation cho má»™t áº£nh"""
        # Load áº£nh
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        original_image = np.array(image)
        
        # Sá»­ dá»¥ng SegformerImageProcessor
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Upsample vá»� kÃ­ch thÆ°á»›c gá»‘c
            logits = torch.nn.functional.interpolate(
                logits,
                size=original_size[::-1],  # (height, width)
                mode='bilinear',
                align_corners=False
            )
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            prediction = predictions.cpu().numpy()[0]
        
        return prediction, original_image
    
    def create_highlighted_visualization(self, image, prediction, alpha=0.6):
        """Táº¡o visualization highlight khá»‘i u"""
        overlay = np.zeros_like(image)
        
        # Highlight neoplastic (cÃ³ polyp)
        neoplastic_mask = (prediction == 0)
        if np.any(neoplastic_mask):
            overlay[neoplastic_mask] = self.highlight_colors[0]
        
        # Highlight non-neoplastic 
        non_neoplastic_mask = (prediction == 1)
        if np.any(non_neoplastic_mask):
            overlay[non_neoplastic_mask] = self.highlight_colors[1]
        
        # Táº¡o áº£nh káº¿t há»£p
        highlighted_image = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        
        return highlighted_image
    
    def create_attention_visualization(self, image, prediction):
        """Táº¡o visualization giáº£ láº­p attention map cá»§a Transformer"""
        # Táº¡o attention map giáº£ láº­p
        attention_map = np.zeros(prediction.shape, dtype=np.float32)
        
        # VÃ¹ng cÃ³ khá»‘i u cÃ³ attention cao
        attention_map[prediction == 0] = 1.0  # Neoplastic
        attention_map[prediction == 1] = 0.8  # Non-neoplastic
        
        # Smooth attention map
        attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
        
        # Ã�p dá»¥ng colormap
        attention_colored = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Blend vá»›i áº£nh gá»‘c
        attention_vis = cv2.addWeighted(image, 0.4, attention_colored, 0.6, 0)
        
        return attention_vis
    
    def test_single_image(self, image_path: str):
        """Test má»™t áº£nh vÃ  hiá»ƒn thá»‹ káº¿t quáº£"""
        print(f"ğŸ”� Testing: {os.path.basename(image_path)}")
        
        # Predict
        prediction, original_image = self.predict_single_image(image_path)
        
        # Táº¡o visualization
        highlighted = self.create_highlighted_visualization(original_image, prediction)
        attention_vis = self.create_attention_visualization(original_image, prediction)
        
        # Hiá»ƒn thá»‹ káº¿t quáº£
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Segformer Test Results - {os.path.basename(image_path)}', fontsize=14)
        
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('áº¢nh gá»‘c')
        axes[0, 0].axis('off')
        
        # Colored mask
        colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_id, color in self.colors.items():
            colored_mask[prediction == class_id] = color
        
        axes[0, 1].imshow(colored_mask)
        axes[0, 1].set_title('Prediction Mask')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(highlighted)
        axes[1, 0].set_title('Highlighted Result')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(attention_vis)
        axes[1, 1].set_title('Attention Visualization')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # In thá»‘ng kÃª
        polyp_pixels = np.sum((prediction == 0) | (prediction == 1))
        total_pixels = prediction.size
        
        print(f"ğŸ”´ Neoplastic (CÃ³ polyp): {np.sum(prediction == 0)} pixels")
        print(f"ğŸŸ  Non-neoplastic: {np.sum(prediction == 1)} pixels") 
        print(f"âš« Background: {np.sum(prediction == 2)} pixels")
        print(f"ğŸ“ˆ Tá»· lá»‡ polyp: {(polyp_pixels/total_pixels)*100:.2f}%")
        print(f"ğŸ¤– Transformer Features: Multi-scale attention")
        print("-" * 50)

# Sá»­ dá»¥ng trá»±c tiáº¿p - khÃ´ng cáº§n import
def run_segformer_test_simple(model_path: str, test_image_paths: List[str]):
    """Function Ä‘Æ¡n giáº£n Ä‘á»ƒ test Segformer"""
    print("ğŸ¤– Báº®T Ä�áº¦U TEST SEGFORMER MODEL")
    print("=" * 50)
    
    # Kiá»ƒm tra transformers
    if not TRANSFORMERS_AVAILABLE:
        print("â�Œ Transformers library khÃ´ng cÃ³ sáºµn!")
        print("ğŸ’¡ CÃ i Ä‘áº·t báº±ng: pip install transformers")
        return
    
    # Táº¡o tester
    tester = SegformerTester(model_path)
    
    # Test tá»«ng áº£nh
    for i, img_path in enumerate(test_image_paths[:5]):
        print(f"\nğŸ“¸ Testing áº£nh {i+1}/5")
        try:
            tester.test_single_image(img_path)
        except Exception as e:
            print(f"â�Œ Lá»—i: {e}")
    
    print("\nğŸ�‰ HOÃ€N THÃ€NH TEST!")

# =================== CHáº Y TEST TRá»°C TIáº¾P ===================

# Ä�Æ°á»�ng dáº«n model (best model Ä‘Ã£ train) - Thay Ä‘á»•i theo Ä‘Æ°á»�ng dáº«n thá»±c táº¿ cá»§a báº¡n
model_path_options = [
    "/kaggle/working/models/best_segformer_segformer_experiment.pth"
]

# TÃ¬m Ä‘Æ°á»�ng dáº«n model tá»“n táº¡i
model_path = None
for path in model_path_options:
    if os.path.exists(path):
        model_path = path
        print(f"âœ… TÃ¬m tháº¥y model: {model_path}")
        break

if model_path is None:
    print("â�Œ KhÃ´ng tÃ¬m tháº¥y model file. HÃ£y kiá»ƒm tra Ä‘Æ°á»�ng dáº«n:")
    for path in model_path_options:
        print(f"   - {path}")
    # Fallback: sá»­ dá»¥ng Ä‘Æ°á»�ng dáº«n Ä‘áº§u tiÃªn (cÃ³ thá»ƒ cáº§n sá»­a)
    model_path = model_path_options[0]
    print(f"âš ï¸� Sá»­ dá»¥ng Ä‘Æ°á»�ng dáº«n máº·c Ä‘á»‹nh: {model_path}")

# Ä�Æ°á»�ng dáº«n áº£nh test - Thay Ä‘á»•i theo Ä‘Æ°á»�ng dáº«n thá»±c táº¿ cá»§a báº¡n  
test_image_dir_options = [
    "/kaggle/input/seg-data/datatest"
]

# TÃ¬m thÆ° má»¥c test images tá»“n táº¡i
test_image_dir = None
for dir_path in test_image_dir_options:
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        test_image_dir = dir_path
        print(f"âœ… TÃ¬m tháº¥y thÆ° má»¥c test: {test_image_dir}")
        break

if test_image_dir is None:
    print("â�Œ KhÃ´ng tÃ¬m tháº¥y thÆ° má»¥c test images. HÃ£y kiá»ƒm tra Ä‘Æ°á»�ng dáº«n:")
    for dir_path in test_image_dir_options:
        print(f"   - {dir_path}")
    # Fallback: sá»­ dá»¥ng Ä‘Æ°á»�ng dáº«n Ä‘áº§u tiÃªn  
    test_image_dir = test_image_dir_options[0]
    print(f"âš ï¸� Sá»­ dá»¥ng Ä‘Æ°á»�ng dáº«n máº·c Ä‘á»‹nh: {test_image_dir}")

# Láº¥y danh sÃ¡ch áº£nh test
try:
    test_images = [
        os.path.join(test_image_dir, f) 
        for f in os.listdir(test_image_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    print(f"ğŸ“� TÃ¬m tháº¥y {len(test_images)} áº£nh test")
    
    if len(test_images) == 0:
        print("â�Œ KhÃ´ng cÃ³ áº£nh nÃ o trong thÆ° má»¥c test")
    else:
        print("ğŸ“‹ Danh sÃ¡ch 5 áº£nh Ä‘áº§u tiÃªn:")
        for i, img in enumerate(test_images[:5]):
            print(f"   {i+1}. {os.path.basename(img)}")
            
except Exception as e:
    print(f"â�Œ Lá»—i khi Ä‘á»�c thÆ° má»¥c test: {e}")
    test_images = []

# CHáº Y TEST
if len(test_images) > 0 and TRANSFORMERS_AVAILABLE:
    print("\n" + "="*60)
    print("ğŸš€ Báº®T Ä�áº¦U TEST SEGFORMER MODEL")
    print("="*60)
    
    try:
        run_segformer_test_simple(model_path, test_images)
    except Exception as e:
        print(f"â�Œ Lá»—i khi cháº¡y test: {e}")
        print("ğŸ’¡ HÃ£y kiá»ƒm tra:")
        print("   1. Model file cÃ³ tá»“n táº¡i khÃ´ng?")
        print("   2. ThÆ° má»¥c test images cÃ³ tá»“n táº¡i khÃ´ng?") 
        print("   3. Transformers library Ä‘Ã£ Ä‘Æ°á»£c cÃ i Ä‘áº·t chÆ°a?")
        print("   4. Config vÃ  SegformerModel Ä‘Ã£ Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong notebook chÆ°a?")
else:
    if not TRANSFORMERS_AVAILABLE:
        print("â�Œ KhÃ´ng thá»ƒ cháº¡y test - Transformers library khÃ´ng cÃ³ sáºµn")
        print("ğŸ’¡ CÃ i Ä‘áº·t báº±ng: pip install transformers")
    else:
        print("â�Œ KhÃ´ng thá»ƒ cháº¡y test - khÃ´ng cÃ³ áº£nh test") 

