!pip install /kaggle/input/wheels-20251001/wheels_20251001/*.whl --no-deps
!cp -r /kaggle/input/nnxnet-050/nnXNet_050 /kaggle/nnxnet
!cp -r /kaggle/input/wheels-20251001/wheels_20251001/dicom2nifti_20250917 /kaggle/dicom2nifti
!cp -r /kaggle/input/wheels-20251001/wheels_20251001/acvl_utils-0.2.5 /kaggle/acvl_utils
!cp -r /kaggle/input/wheels-20251001/wheels_20251001/batchgenerators-0.25.1 /kaggle/batchgenerators
!cp -r /kaggle/input/wheels-20251001/wheels_20251001/dynamic_network_architectures-0.3.1 /kaggle/dynamic_network_architectures


import sys
sys.path.append('/kaggle/nnxnet')
sys.path.append('/kaggle/dicom2nifti')
sys.path.append('/kaggle/acvl_utils')
sys.path.append('/kaggle/batchgenerators')
sys.path.append('/kaggle/dynamic_network_architectures')


# ==================================================
# RSNA Intracranial Aneurysm Challenge - Inference
# Two-stage pipeline: DICOM → NIfTI → Brain Segmentation → Aneurysm Classification
# ==================================================
"""HOUJING:
ONLY WORK FOR 2 GPUs.
How inference tasks are divided between workers are defined by how you call `executor.submit`.
Only create the thread pool once and use it for all case prediction, because creating the pool for each case is very time consuming.
Extension:
If you want to use 4 folds with 2 GPUs:
- Submit fold_0 and fold_1 prediction in parallel
- Wait till finish
- Submit fold_2 and fold_3 prediction in parallel
- Wait till finish
"""
import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import pydicom
import shutil
import gc
import nibabel as nib
import dicom2nifti
import kaggle_evaluation.rsna_inference_server
import torch.nn.functional as F
from skimage.transform import resize
from pathlib import Path
from typing import Tuple, Union, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from nnxnet.inference.predict_from_raw_data_2D_orthogonal_planes_fast import nnXNetPredictor
from nnxnet.inference.predict_from_raw_data_two_seg_with_cls_no_seg_return_no_filter import nnXNetPredictor as nnXNetPredictorWithCls
from nnxnet.utilities.helpers import empty_cache, dummy_context

# Constants
MODEL_PATHS = {
    'vessel_ROI_seg': "/kaggle/input/dataset180_2d_vessel_box_seg_stable/pytorch/default/1/Dataset180_2D_vessel_box_seg_stable/nnUNetTrainer__nnUNetPlans__2d",
    'aneurysm_cls_1': "/kaggle/input/rsna2025-stage2-models/pytorch/default/4/RSNA2025_stage2_models/onlyMirror01_lr4e3_100epochs",
    'aneurysm_cls_2': "/kaggle/input/rsna2025-stage2-models/pytorch/default/4/RSNA2025_stage2_models/onlyMirror01_250epochs",
    'plane_2d_cls': "/kaggle/input/resnet34_plane_2d_cls/pytorch/default/1/ResNet34_Plane_2D_cls/checkpoint_best_loss.pth"
}
SHARED_DIR = Path('/kaggle/shared')
TEMP_DIR = Path('/kaggle/working')
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# Define global variables at the module level
GLOBAL_VESSEL_ROI_PREDICTOR = None
GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS = None
CLS_2D_PREDICTOR = None

USE_NUM_GPUS = 2
NUM_INFER_WORKERS = 2
COMPILE_NETWORK = False

executor = ThreadPoolExecutor(max_workers=NUM_INFER_WORKERS)

def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get the computation device, with validation for GPU availability.
    """
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

# Get the device (this part of the original code is fine)
DEVICE = get_device(gpu_id=0)

# ==================================================
# Plane Classification Model Definition
# ==================================================

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1):
        super(ResNetEncoder, self).__init__()
        self.in_planes = 64
        self.block = block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.embed_dim = 512 * block.expansion
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super(CrossAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num
        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(query_num * embed_dim, num_classes) 
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
        for name, param in self.cross_attention.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.flatten(2)
        x = x.permute(2, 0, 1)
        query = self.class_query.unsqueeze(1).repeat(1, batch_size, 1)
        
        attended, _ = self.cross_attention(query=query, key=x, value=x)
        
        attended = self.norm(attended)
        attended = self.dropout(attended)
        attended_permuted = attended.permute(1, 0, 2)
        attended_flatten = attended_permuted.flatten(1)
        logits = self.classifier(attended_flatten) 
        return logits


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, dropout=0.0, use_cross_attention=True, num_heads=4):
        super(ClassificationHead, self).__init__()
        if use_cross_attention:
            self.pooling = CrossAttentionPooling(
                embed_dim=embed_dim, 
                query_num=query_num, 
                num_classes=num_classes, 
                num_heads=num_heads, 
                dropout=dropout
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Flatten(1), 
                nn.Dropout(dropout), 
                nn.Linear(embed_dim, num_classes)
            )
    
    def forward(self, x):
        return self.pooling(x)


class PlaneResNet34(nn.Module):
    """Single-task model: Plane classification only (3 classes: AX/SAG/COR)"""
    
    def __init__(self, dropout: float = 0.1):
        super(PlaneResNet34, self).__init__()
        
        self.encoder = ResNetEncoder(BasicBlock, [3, 4, 6, 3], in_channels=1)
        self.embed_dim = self.encoder.embed_dim
        
        self.head_plane = ClassificationHead(
            embed_dim=self.embed_dim, 
            query_num=3, 
            num_classes=3, 
            dropout=dropout, 
            use_cross_attention=True
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.head_plane(features)
        return logits


# ==================================================
# Plane Classification Predictor
# ==================================================

class PlaneClassifier:
    """Axial Slice Plane Prediction"""
    
    # Plane Category Mapping
    PLANE_MAP = {0: 'AX', 1: 'SAG', 2: 'COR'}
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0', target_size=(256, 256)):
        """
        Initialize the inferencer.
        
        Args:
            checkpoint_path: Path to the model weights.
            device: Inference device (e.g., 'cuda', 'cpu').
            target_size: Target image dimensions.
        """
        self.device = device
        self.target_size = target_size
        self.model = self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str) -> PlaneResNet34:
        """Load model - Modified: Using PlaneResNet34"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model file does not exist: {checkpoint_path}")
        
        model = PlaneResNet34(dropout=0.0)
        
        # Load model weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_slice(self, slice_2d: np.ndarray) -> torch.Tensor:
        """Preprocesses a 2D slice."""
        # 1. Type conversion and clipping
        slice_data = slice_2d
        
        # 2. Z-score normalization
        mean = slice_data.mean()
        std = slice_data.std()
        std = np.clip(std, 1e-8, None)
        slice_data = (slice_data - mean) / std
        
        # 3. Resize
        resized_slice = resize(slice_data, self.target_size, anti_aliasing=True).astype(np.float32)
        
        # 4. Convert to Tensor
        tensor = torch.from_numpy(resized_slice).unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def predict(self, tensor: torch.Tensor) -> Tuple[str, int, float, torch.Tensor]:
        """
        Performs Plane prediction - Updated: adapted for single-task model output format
        
        Args:
            tensor: Input tensor of shape [1, 1, H, W]
            
        Returns:
            plane_pred_label: Predicted class name ('AX', 'SAG', 'COR')
            plane_pred: Predicted class index (0, 1, 2)
            plane_prob: Maximum probability value
            plane_prob_list: Probability distribution over all classes [1, 3]
        """
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
        
        # Plane prediction
        plane_pred = logits.argmax(dim=1).item()
        plane_prob_list = F.softmax(logits, dim=1)
        plane_prob = plane_prob_list.max().item()
        plane_pred_label = self.PLANE_MAP[plane_pred]
        
        return plane_pred_label, plane_pred, plane_prob, plane_prob_list
    
    def inference_from_slice(
        self, 
        slice_2d: np.ndarray,
    ) -> Tuple[str, int, float, torch.Tensor]:
        """
        Performs inference from a 2D slice.
        
        Returns:
            plane_pred_label: Predicted class name
            plane_pred: Predicted class index
            plane_prob: Maximum probability
            plane_prob_list: Probability distribution
        """
        # 1. Preprocessing
        tensor = self.preprocess_slice(slice_2d)
        
        # 2. Inference
        plane_pred_label, plane_pred, plane_prob, plane_prob_list = self.predict(tensor)
        
        # 3. Print results
        self._print_result(plane_pred_label, plane_pred, plane_prob, plane_prob_list)
        
        return plane_pred_label, plane_pred, plane_prob, plane_prob_list
    
    def _print_result(self, plane_pred_label: str, plane_pred: int, plane_prob: float, plane_prob_list: torch.Tensor):
        """Prints the prediction results."""
        print("=" * 60)
        print("Plane Prediction Results:")
        print(f"  Predicted Class: {plane_pred_label}")
        print(f"  Class Index: {plane_pred}")
        print(f"  Confidence: {plane_prob:.4f} ({plane_prob*100:.2f}%)")
        print(f"  Probability Distribution: {plane_prob_list}")
        print("=" * 60)

def correct_orientation(pixel_array, spacing, plane_id):
    """
    Corrects image orientation to standard axial view.
    """
    if plane_id == 1:
        # Sagittal → Axial
        fixed_array = np.transpose(pixel_array, (1, 2, 0))
        fixed_array = fixed_array[::-1, :, :]
        fixed_spacing = [spacing[1], spacing[2], spacing[0]]
        print(f"  Corrected: {pixel_array.shape} → {fixed_array.shape}")
        print(f"  Corrected: {spacing} → {fixed_spacing}")
        return fixed_array, fixed_spacing

    elif plane_id == 2:
        # Coronal → Axial
        fixed_array = np.transpose(pixel_array, (1, 0, 2))
        fixed_array = fixed_array[::-1, :, :]
        fixed_spacing = [spacing[1], spacing[0], spacing[2]]
        print(f"  Corrected: {pixel_array.shape} → {fixed_array.shape}")
        print(f"  Corrected: {spacing} → {fixed_spacing}")
        return fixed_array, fixed_spacing
    else:
        # Already axial or no correction needed
        return pixel_array, spacing
#=====================================================

def reorient_nii(orig_nii, targ_aff="LPS"):
    """
    Reorient to the standard LPS+ DICOM coord.
    """
    if "".join(nib.aff2axcodes(orig_nii.affine)) == targ_aff:
        return orig_nii
    orig_ornt = nib.io_orientation(orig_nii.affine)
    targ_ornt = nib.orientations.axcodes2ornt(targ_aff)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    img_orient = orig_nii.as_reoriented(transform)
    return img_orient

# ==================================================
# 1. Initialize Model (Global One-time Initialization)
# ==================================================
def may_compile_network(network):
    if COMPILE_NETWORK:
        return torch.compile(network)
    return network

def init_predictors(device):
    global GLOBAL_VESSEL_ROI_PREDICTOR, GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS, CLS_2D_PREDICTOR
    
    # If already initialized, return the global instance directly
    if GLOBAL_VESSEL_ROI_PREDICTOR is not None and GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS is not None and CLS_2D_PREDICTOR is not None:
        return GLOBAL_VESSEL_ROI_PREDICTOR, GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS, CLS_2D_PREDICTOR
    
    # Stage 1: Vessel ROI Segmentation (Using the new multiplanar predictor)
    GLOBAL_VESSEL_ROI_PREDICTOR = nnXNetPredictor(
        tile_step_size=0.5,
        use_mirroring=False,
        use_gaussian=True,
        perform_everything_on_device=True,
        device=device,
        allow_tqdm=False
    )
    GLOBAL_VESSEL_ROI_PREDICTOR.initialize_from_trained_model_folder(
        model_training_output_dir=MODEL_PATHS['vessel_ROI_seg'],
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    GLOBAL_VESSEL_ROI_PREDICTOR.initialize_network_and_gaussian()
    GLOBAL_VESSEL_ROI_PREDICTOR.network = may_compile_network(GLOBAL_VESSEL_ROI_PREDICTOR.network)

    # Stage2: Aneurysm classification
    aneurysm_predictor_f0 = nnXNetPredictorWithCls(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=get_device(gpu_id=0),
        verbose=False
    )

    aneurysm_predictor_f1 = nnXNetPredictorWithCls(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=get_device(gpu_id=1 if USE_NUM_GPUS == 2 else 0),
        verbose=False
    )

    aneurysm_predictor_f0.initialize_from_trained_model_folder(
        model_training_output_dir=MODEL_PATHS['aneurysm_cls_1'],
        use_folds=(0, ),
        checkpoint_name="checkpoint_final.pth",
    )
    aneurysm_predictor_f0.initialize_network_and_gaussian()

    aneurysm_predictor_f1.initialize_from_trained_model_folder(
        model_training_output_dir=MODEL_PATHS['aneurysm_cls_2'],
        use_folds=(1, ),
        checkpoint_name="checkpoint_final.pth",
    )
    aneurysm_predictor_f1.initialize_network_and_gaussian()

    aneurysm_predictor_f0.network = may_compile_network(aneurysm_predictor_f0.network)
    aneurysm_predictor_f1.network = may_compile_network(aneurysm_predictor_f1.network)

    # ========== Initialize 2D Orientation Classifier Inferencer ==========
    CLS_2D_PREDICTOR = PlaneClassifier(
        checkpoint_path=MODEL_PATHS['plane_2d_cls'],
        device=device,
        target_size=(256, 256)
    )
    CLS_2D_PREDICTOR.model = may_compile_network(CLS_2D_PREDICTOR.model)

    GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS = [aneurysm_predictor_f0, aneurysm_predictor_f1]
    return GLOBAL_VESSEL_ROI_PREDICTOR, GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS, CLS_2D_PREDICTOR

    
GLOBAL_VESSEL_ROI_PREDICTOR, GLOBAL_ANEURYSM_PREDICTOR, CLS_2D_PREDICTOR = init_predictors(DEVICE)

def group_dicom_files(study_folder_path: str) -> Dict[Tuple, List[str]]:
    """
    Groups DICOM files into pseudo-series based on StudyInstanceUID, 
    FrameOfReferenceUID, Modality, and ImageOrientationPatient.
    """
    dicom_groups = {}
    dicom_files = []

    for root, _, files in os.walk(study_folder_path):
        for file in files:
            if file.endswith(('.dcm', '.DCM')) or ('.' not in file and len(file) > 1):
                dicom_files.append(os.path.join(root, file))

    if not dicom_files:
        print(f"No DICOM files found in path: {study_folder_path}")
        return dicom_groups
    
    print(f"Found {len(dicom_files)} files in total, starting grouping...")

    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            study_uid = getattr(ds, 'StudyInstanceUID', 'NO_STUDY_UID')
            frame_uid = getattr(ds, 'FrameOfReferenceUID', 'NO_FRAME_UID')
            modality = getattr(ds, 'Modality', 'UNKNOWN')
            
            orientation = getattr(ds, 'ImageOrientationPatient', [0, 0, 0, 0, 0, 0])
            orientation_key = tuple(np.round(orientation, 4)) 

            group_key = (study_uid, frame_uid, modality, orientation_key)

            if group_key not in dicom_groups:
                dicom_groups[group_key] = []
            dicom_groups[group_key].append(file_path)

        except (pydicom.errors.InvalidDicomError, Exception):
            continue

    print("-" * 50)
    print(f"Grouping completed. Identified {len(dicom_groups)} logical series in total.")
    return dicom_groups

def get_largest_series_files(all_series: Dict[Tuple, List[str]]) -> List[str]:
    """
    Finds the series with the most slices and returns the file list sorted by InstanceNumber.

    Args:
        all_series (dict): Grouping dictionary returned by group_dicom_files.

    Returns:
        list: List of DICOM file paths from the series with the most slices, sorted by InstanceNumber.
    """
    if not all_series:
        return []

    # 1. Find the series with the most slices
    # (Key, file list)
    max_layers_series_item = None
    max_layers = 0

    for group_key, file_list in all_series.items():
        if len(file_list) > max_layers:
            max_layers = len(file_list)
            max_layers_series_item = (group_key, file_list)

    if not max_layers_series_item:
        print("No valid series found.")
        return []

    target_group_key, target_series_files = max_layers_series_item
    
    print("-" * 50)
    print(f"*** Found series with the most slices *** (Slice count: {max_layers})")
    print(f"  Modality: {target_group_key[2]}")
    print(f"  Orientation: {target_group_key[3][:3]}...")

    # 2. Sort the target series (using InstanceNumber)
    sorted_files_with_number: List[Tuple[Any, str]] = []
    
    # Iterate through files to get InstanceNumber
    for fp in target_series_files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
            # Sort by InstanceNumber. Use 0 if InstanceNumber doesn't exist.
            # Alternatively, consider using ImagePositionPatient[2] for sorting
            instance_number = getattr(ds, 'InstanceNumber', 0)
            sorted_files_with_number.append((instance_number, fp))
        except:
            pass

    # Sort by InstanceNumber
    sorted_files_with_number.sort(key=lambda x: x[0])
    
    final_file_paths = [fp for _, fp in sorted_files_with_number]
    
    return final_file_paths

def get_spacing_by_shape(shape):
    """
    Efficiently maps spacing based on the size of each axis.
    Rules:
    - Axis > 300: 0.5mm
    - 120 < Axis <= 300: 0.55mm  
    - 100 < Axis <= 120: 0.75mm
    - 80 < Axis <= 100: 1.0mm
    - 60 < Axis <= 80: 1.5mm
    - 45 < Axis <= 60: 3.0mm
    - Axis <= 45: 5.0mm
    """
    spacing = []
    for dim in shape:
        if dim > 300:
            spacing.append(0.5)
        elif dim > 120:
            spacing.append(0.55)
        elif dim > 100:
            spacing.append(0.75)
        elif dim > 80:
            spacing.append(1.0)
        elif dim > 60:
            spacing.append(1.5)
        elif dim > 45:
            spacing.append(3.0)
        else:
            spacing.append(5.0)
    return spacing

def flip_z(img_tensor: torch.Tensor) -> torch.Tensor:
    """Flip along Z-axis (superior-inferior): corresponds to dim=2"""
    # Shape [B, C, D, H, W], D corresponds to dim=2
    return torch.flip(img_tensor, dims=[2])

def flip_y(img_tensor: torch.Tensor) -> torch.Tensor:
    """Flip along Y-axis (anterior-posterior): corresponds to dim=3"""
    # Shape [B, C, D, H, W], H corresponds to dim=3
    return torch.flip(img_tensor, dims=[3])

def flip_x(tensor):
    """Flip along X-axis (left-right): corresponds to dim=4"""
    # Shape [B, C, D, H, W], W corresponds to dim=4
    return torch.flip(tensor, dims=[4])

@torch.no_grad()
def worker_infer(num_cls_task, tta_batch_size, image_resized, aneurysm_predictor_fold, fold_i, all_fold_mean_logits_by_task):
    """
    image_resized: can be on whatever device, because `image_resized = image_resized.to(device)` 
        will move it to correct device for inference. 
    Results are on CPU.
    all_fold_mean_logits_by_task: modified inplace in this function.
        It is thread-safe if we make sure different threads write to different slots (or sub-slots) of the list.
    """
    device = aneurysm_predictor_fold.device
    print(f"[*] Worker Using Device: {device}")

    # Task 2 (Location Classification) Left-right flip index mapping (13 classes: 0-12)
    # [L_ICL, R_ICL, L_SCL, R_SCL, L_MCA, R_MCA, AC, L_AC, R_AC, L_PC, R_PC, BT, OP]
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    # Swap: 1, 0, 3, 2, 5, 4, 6, 8, 7, 10, 9, 11, 12
    # Only for Task 2 logits (shape [N_classes=13])
    TASK2_SWAP_INDICES = [1, 0, 3, 2, 5, 4, 6, 8, 7, 10, 9, 11, 12]

    image_resized = image_resized.to(device)
    # ======================================================
    # TTA Step 1: Construct all TTA augmented image list
    # Add X-axis flip (left-right flip)
    # ======================================================
    
    # 1. Original image
    I_orig = image_resized
    # 2. Augmentation A: Y-axis flip (anterior-posterior)
    I_flip_y = flip_y(I_orig)
    # 3. Augmentation B: Z-axis flip (superior-inferior)
    I_flip_z = flip_z(I_orig)
    # 4. Augmentation C: Y-axis + Z-axis flip
    I_flip_yz = flip_y(I_flip_z)
    
    # 5. Augmentation D: X-axis flip (left-right)
    I_flip_x = flip_x(I_orig)
    # 6. Augmentation E: X-axis + Y-axis flip
    I_flip_xy = flip_x(I_flip_y)
    # 7. Augmentation F: X-axis + Z-axis flip
    I_flip_xz = flip_x(I_flip_z)
    # 8. Augmentation G: X-axis + Y-axis + Z-axis flip
    I_flip_xyz = flip_x(I_flip_yz)
    
    # TTA list (total 8 augmentations)
    tta_images = [
        I_orig, I_flip_y, I_flip_z, I_flip_yz,
        I_flip_x, I_flip_xy, I_flip_xz, I_flip_xyz
    ]
    
    # Mark which augmentations performed X-axis (left-right) flip
    # 0: not flipped, 1: X-axis flipped
    # Corresponds to tta_images list
    x_flip_masks = [0, 0, 0, 0, 1, 1, 1, 1] 

    # ======================================================
    # TTA Step 2: Iterative inference (supports tta_batch_size setting)
    # ======================================================
    
    num_tta_augments = len(tta_images) # 8
    # current_fold_tta_logits_by_task[task_i] is a list storing all TTA image logits for that task
    current_fold_tta_logits_by_task = [[] for _ in range(num_cls_task)]
    
    empty_cache(device)
    
    for i in range(0, num_tta_augments, tta_batch_size):
        # Construct current batch of TTA images and corresponding flip flags
        batch_tta_images = tta_images[i:i + tta_batch_size]
        batch_x_flip_masks = x_flip_masks[i:i + tta_batch_size]
        
        # Stack into TTA Batch (shape: [tta_batch_size, 1, 224, 224, 224])
        image_tta_batch = torch.cat(batch_tta_images, dim=0)

        with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            # predicted_logits_list shape: [task1_logits, task2_logits],
            predicted_logits_list = aneurysm_predictor_fold.network(image_tta_batch, only_forward_cls=True)
            
            # Collect logits for each task
            # Task 1 (first task, index 0): No left-right swap (assuming Aneurysm Present)
            current_fold_tta_logits_by_task[0].append(predicted_logits_list[0])
            
            # Task 2 (second task, index 1): Handle left-right swap
            current_task2_logits = predicted_logits_list[1]
            
            processed_logits_list = []
            for j, is_x_flipped in enumerate(batch_x_flip_masks):
                # If current TTA image performed X-axis flip (left-right flip)
                if is_x_flipped == 1:
                    # Swap left-right positions for Task 2 results
                    flipped_logits = current_task2_logits[j][TASK2_SWAP_INDICES]
                    processed_logits_list.append(flipped_logits)
                else:
                    # Otherwise keep original logits
                    processed_logits_list.append(current_task2_logits[j])
                    
            # Collect processed Task 2 logits
            current_fold_tta_logits_by_task[1].append(torch.stack(processed_logits_list))
        
        del image_tta_batch
        empty_cache(device)


    # --------------------------------------------------
    # TTA Step 3: Current Fold result integration (Logits averaging)
    # --------------------------------------------------
    for task_i in range(num_cls_task):
        # Merge all batch logits for current fold into a large [num_tta_augments, num_classes] Tensor
        logits_tta = torch.cat(current_fold_tta_logits_by_task[task_i], dim=0)
        
        # Logits averaging (along dimension 0, i.e., batch dimension)
        mean_logit_fold = logits_tta.mean(dim=0, keepdim=False).cpu() # shape: [num_classes]
        
        # Collect current fold's TTA average logits
        all_fold_mean_logits_by_task[task_i][fold_i] = mean_logit_fold
    

def predict_aneurysm(input_img_np, original_spacing, device, tta_batch_size=2):
    """
    Aneurysm prediction function (with left-right flip TTA and left-right swapping for Task 2 results)
    
    Args:
        input_img_np: Input image numpy array
        original_spacing: Original image spacing (x, y, z)
        device: Computing device (cpu/cuda)
    
    Returns:
        timepoint_probs: Classification probability array [task2 probabilities, task1 probabilities]
    """
    # Stage 1: Vessel ROI prediction
    stage_1_target_spacing = np.array([1, 0.55, 0.5])
    with torch.no_grad():
        z_min_final, z_max_final, y_min_final, y_max_final, x_min_final, x_max_final = GLOBAL_VESSEL_ROI_PREDICTOR.predict_from_multi_axial_slices(
            input_img_np, original_spacing, stage_1_target_spacing, max_batch_size=16
        )

    # Crop image to ROI region
    img_cropped_np = input_img_np[0][z_min_final:z_max_final, y_min_final:y_max_final, x_min_final:x_max_final][None]

    del input_img_np 

    img_cropped_np = np.ascontiguousarray(img_cropped_np)

    # Stage 2: Aneurysm classification
    img_cropped_tensor = torch.from_numpy(img_cropped_np).half().to(device)  # HOUJING: added half()

    # Image normalization
    image_normed = (img_cropped_tensor - img_cropped_tensor.mean()) / img_cropped_tensor.std().clip(1e-8)

    del img_cropped_tensor

    dst_shape = [224, 224, 224]
    
    # Image resampling
    image_resized = torch.nn.functional.interpolate(
        image_normed[None], size=dst_shape, mode='trilinear', align_corners=True
    )

    del image_normed

    # Initialize a list to store logits for each TTA image
    # all_fold_mean_logits_by_task[task_i] is a list storing all TTA image logits for that task
    num_cls_task = 2
    n_folds = 2
    all_fold_mean_logits_by_task = [[None for _ in range(n_folds)] for _ in range(num_cls_task)]

     # ======================================================
    # Inference for each fold model in parallel
    # ======================================================
    futures = []
    for fold_i, aneurysm_predictor_fold in enumerate(GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS):
        # This runs asynchronously
        futures.append(executor.submit(worker_infer, num_cls_task, tta_batch_size, image_resized, aneurysm_predictor_fold, fold_i, all_fold_mean_logits_by_task))
    # Iterating over as_completed(futures) ensures all tasks finish before moving on
    for future in as_completed(futures):
        pass
        
    # ======================================================
    # Final result integration (averaging TTA mean logits across all folds)
    # ======================================================
    
    aggregated_probs_list = []
    
    for task_i in range(num_cls_task):
        logits_tta = torch.stack(all_fold_mean_logits_by_task[task_i], dim=0)
        
        mean_logit = logits_tta.mean(dim=0, keepdim=False)
        
        # Convert to final probabilities using Sigmoid
        final_prob = torch.sigmoid(mean_logit)
        
        aggregated_probs_list.append(final_prob.to('cpu').numpy().flatten())

    # Merge probabilities: Task2 first, Task1 last (consistent with original code)
    task1_probs = aggregated_probs_list[0] # Task 1 (existence)
    task2_probs = aggregated_probs_list[1] # Task 2 (location)

    return np.concatenate([task2_probs, task1_probs], axis=0)

# ==================================================
# 2. Two-Stage Inference
# ==================================================
def process_single_timepoint(orig_nii, time_index=None):
    """
    Process data for a single timepoint.
    """
    # Extract data for the specified timepoint
    if time_index is not None and orig_nii.ndim == 4 and orig_nii.shape[3] > 1:
        orig_data = orig_nii.get_fdata()[:, :, :, time_index]
        orig_nii = nib.Nifti1Image(orig_data, orig_nii.affine, orig_nii.header)
    
    # Reorient to standard space
    img_orient = reorient_nii(orig_nii, targ_aff="LPS")
    input_img_np = img_orient.get_fdata()
    
    input_img_np = input_img_np.transpose(2, 1, 0)[None]
    original_spacing = img_orient.header.get_zooms()[:3]

    return predict_aneurysm(input_img_np, original_spacing, DEVICE, tta_batch_size=2)

# You can return either a Pandas or Polars dataframe, though Polars is recommended.
def predict(series_path: str) -> pl.DataFrame:
    """
    Make a prediction for a given DICOM series path.
    This function consolidates the core prediction logic into the required format.
    """
    series_id = os.path.basename(series_path)
    
    # Step 1: Quickly retrieve file list
    dicom_files = []
    for root, _, files in os.walk(series_path):
        for file in files:
            # Simplified file type checking
            if file.endswith(('.dcm', '.DCM')) or ('.' not in file and len(file) > 1):
                dicom_files.append(os.path.join(root, file))
    
    if len(dicom_files) == 0:
        probs = np.ones(len(LABEL_COLS)) * 0.5

    elif len(dicom_files) == 1:
        
        ds = pydicom.dcmread(dicom_files[0], force=True)
        
        # Get pixel array
        pixel_array = ds.pixel_array
        # Pixel array shape: (150, 528, 528)
        # Computed spacing: [0.55, 0.5, 0.5]
        
        input_img_np = pixel_array[None]
        spacing = get_spacing_by_shape(pixel_array.shape)
        original_spacing = spacing[::-1]

        print(f"Computed spacing: {spacing}")

        ''' 
        Add exception handling
        '''
        # Filter thick-slice T2 data
        if spacing[0] >= 3:
            print('Detected thick-slice T2 data, performing orientation analysis...')
            
            # ========== Inference from 2D numpy array ==========
            print("\nInference from 2D numpy array")     
            img_3d = pixel_array  # Directly use the loaded pixel_array
            D, _ , _ = pixel_array.shape
            slice_idx = D // 2
            slice_2d = img_3d[slice_idx, :, :]
            
            plane_label, plane_id, plane_conf, plane_probs = CLS_2D_PREDICTOR.inference_from_slice(
                slice_2d=slice_2d)
            
            # Exception handling correction
            if len(pixel_array.shape) == 3 and plane_id != 0:
                print('Entering exception handling...')
                pixel_array, spacing = correct_orientation(pixel_array, spacing, plane_id)
                print(f"Corrected pixel array shape: {pixel_array.shape}")
                print(f"Corrected spacing: {spacing}")
                input_img_np = pixel_array[None]
                original_spacing = spacing[::-1]

        probs = predict_aneurysm(input_img_np, original_spacing, DEVICE, tta_batch_size=2)
        
    else:
        try: 
            orig_nii = dicom2nifti.dicom_series_to_nifti(series_path, None, reorient_nifti=False)['NII']
        except:
            all_series = group_dicom_files(series_path)
    
            # Find the series with the most slices and sort
            largest_series_files = get_largest_series_files(all_series)
    
            # Now largest_series_files contains the required DICOM file paths sorted by InstanceNumber
            if largest_series_files:
                print(f"\nFinal target series file count: {len(largest_series_files)}")
                # You can now use this list for subsequent image loading and processing
    
            largest_series_path = "/kaggle/largest_series_tmp_path"
    
            if os.path.exists(largest_series_path):
                shutil.rmtree(largest_series_path)
            os.makedirs(largest_series_path)
    
            # 4. Copy files
            for file_path in largest_series_files:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(largest_series_path, filename)
                shutil.copy2(file_path, dest_path)
    
            orig_nii = dicom2nifti.dicom_series_to_nifti(largest_series_path, None, reorient_nifti=False)['NII']
            
        # Process multi-timepoint data
        if orig_nii.ndim == 4 and orig_nii.shape[3] > 1:
            # Get number of timepoints
            t = orig_nii.shape[3]
            
            # Store prediction probabilities for each timepoint
            all_timepoint_probs = []
            
            # Perform inference for each timepoint
            for t_i in range(t):
                print(f"Processing timepoint {t_i + 1}/{t}")
                
                # Use the reused processing function, passing time index
                timepoint_probs = process_single_timepoint(
                    orig_nii,
                    time_index=t_i
                )
                all_timepoint_probs.append(timepoint_probs)
            
            # Combine probabilities from all timepoints, take maximum
            all_timepoint_probs = np.array(all_timepoint_probs)  # shape: (T, prob_length)
            probs = np.max(all_timepoint_probs, axis=0)
            
        else:
            # Single timepoint processing
            probs = process_single_timepoint(
                orig_nii
            )

    pred_df = pl.DataFrame(
        data=[probs.tolist()],
        schema=LABEL_COLS,
        orient='row'
    )
        
    # Perform memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ----------------------------- IMPORTANT ------------------------------
    # You MUST have the following code in your `predict` function
    # to prevent "out of disk space" errors. This is a temporary workaround
    # as we implement improvements to our evaluation system.
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    # ----------------------------------------------------------------------
    
    return pred_df


# from time import time
# case_paths = [
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004684224894397679901841656954650085',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10009383108068795488741533244914370182',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10012790035410518400400834395242853657',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10014757658335054766479957992112625961',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10021411248005513321236647460239137906',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10022688097731894079510930966432818105',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10022796280698534221758473208024838831',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10023411164590664678534044036963716636',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10030095840917973694487307992374923817',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10030804647049037739144303822498146901',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10034081836061566510187499603024895557',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381',
#     '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10035782880104673269567641444954004745'
# ]
# st = time()
# for p in case_paths:
#     this_st = time()
#     predict(p)
#     print(f"******** One case done, {time() - this_st:.2f}s ********")
# total_time = time() - st
# avg_time = total_time / len(case_paths)
# print(f"All Done, {total_time:.2f}s")
# print(f"Avg time: {avg_time:.2f}s")


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

