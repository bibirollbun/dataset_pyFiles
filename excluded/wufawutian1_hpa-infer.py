# ===== ç³»ç»Ÿåˆ�å§‹åŒ–å’Œä¾�èµ–å¯¼å…¥ =====
import os
import sys
import ast
import gc
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from torchvision.transforms import ToTensor, Normalize
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("ğŸ“¦ æ‰€æœ‰ä¾�èµ–åº“å¯¼å…¥æˆ�åŠŸ")
print(f"ğŸ”¢ PyTorch ç‰ˆæœ¬: {torch.__version__}")
print(f"ğŸ–¥ï¸� CUDA å�¯ç”¨: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"ğŸ�® GPU è®¾å¤‡: {torch.cuda.get_device_name(0)}")
    print(f"ğŸ“Š GPU æ˜¾å­˜: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

# è®¾ç½®è®¾å¤‡
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"âš¡ ä½¿ç”¨è®¾å¤‡: {device}")

# ===== æ ¸å¿ƒé…�ç½®å�‚æ•° =====
# æ�§åˆ¶æ˜¯å�¦ä½¿ç”¨é¢„å…ˆåˆ†å‰²å¥½çš„æ�©ç �æ–‡ä»¶ï¼ˆTrueï¼‰è¿˜æ˜¯å®�æ—¶åˆ†å‰²ï¼ˆFalseï¼‰
ONLY_PUBLIC = True  # è®¾ç½®ä¸ºTrueä½¿ç”¨é¢„åˆ†å‰²æ�©ç �ï¼ŒFalseä½¿ç”¨å®�æ—¶åˆ†å‰²

# è®¾ç½®è·¯å¾„
IMAGE_DIR = "/kaggle/input/hpa-single-cell-image-classification/test"
SUBMISSION_FILE = "/kaggle/input/hpa-sample-submission-with-extra-metadata/updated_sample_submission.csv"  # é¢„åˆ†å‰²æ�©ç �æ–‡ä»¶
MODEL_PATH = "/kaggle/input/help-model/help_encoder_torchscript.pt"

# HPAåˆ†å‰²å™¨æ¨¡å�‹è·¯å¾„ï¼ˆä»…åœ¨å®�æ—¶åˆ†å‰²æ¨¡å¼�ä¸‹ä½¿ç”¨ï¼‰
HPA_SEGMENTATION_MODEL_DIR = "/kaggle/input/hpaseg-model-weight"

# æ�¨ç�†å�‚æ•°
CONF_THRESH = 0.0
TILE_SIZE = (256, 256)
DO_TTA = False
TTA_REPEATS = 8
IS_DEMO = False  # æ�§åˆ¶æ˜¯å�¦ä¸ºæ¼”ç¤ºæ¨¡å¼�

# æ¨¡å�‹å�‚æ•°
# æ¨¡å�‹æ�¨ç�†çº§åˆ«é…�ç½®
MODEL_LEVEL = 'cell'  # 'image' or 'cell'
# éªŒè¯�è·¯å¾„å­˜åœ¨
if not os.path.exists(IMAGE_DIR):
    print(f"â�Œ å›¾åƒ�ç›®å½•ä¸�å­˜åœ¨: {IMAGE_DIR}")
else:
    print(f"âœ… å›¾åƒ�ç›®å½•: {IMAGE_DIR}")

if ONLY_PUBLIC:
    if not os.path.exists(SUBMISSION_FILE):
        print(f"â�Œ æ��äº¤æ–‡ä»¶ä¸�å­˜åœ¨: {SUBMISSION_FILE}")
    else:
        file_size = os.path.getsize(SUBMISSION_FILE) / (1024**2)
        print(f"âœ… æ��äº¤æ–‡ä»¶: {SUBMISSION_FILE} ({file_size:.1f} MB)")
else:
    if not os.path.exists(HPA_SEGMENTATION_MODEL_DIR):
        print(f"âš ï¸� HPAåˆ†å‰²æ¨¡å�‹ç›®å½•ä¸�å­˜åœ¨: {HPA_SEGMENTATION_MODEL_DIR}")
        print("ğŸ’¡ å®�æ—¶åˆ†å‰²æ¨¡å¼�éœ€è¦�HPAåˆ†å‰²æ¨¡å�‹")
    else:
        print(f"âœ… HPAåˆ†å‰²æ¨¡å�‹ç›®å½•: {HPA_SEGMENTATION_MODEL_DIR}")

if not os.path.exists(MODEL_PATH):
    print(f"â�Œ æ¨¡å�‹æ–‡ä»¶ä¸�å­˜åœ¨: {MODEL_PATH}")
else:
    model_size = os.path.getsize(MODEL_PATH) / (1024**2)
    print(f"âœ… æ¨¡å�‹æ–‡ä»¶: {MODEL_PATH} ({model_size:.1f} MB)")

# HPA ç±»åˆ«å®šä¹‰
HPA_CLASSES = [
    "Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center",
    "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus",
    "Intermediate filaments", "Actin filaments", "Microtubules", "Mitotic spindle",
    "Centrosome", "Plasma membrane", "Mitochondria", "Aggresome", "Cytosol",
    "Vesicles and punctate cytosolic patterns", "Negative"
]

print(f"ğŸ�·ï¸� ç±»åˆ«æ•°é‡�: {len(HPA_CLASSES)}")
print(f"ğŸ”§ é…�ç½®æ¨¡å¼�: {'ä½¿ç”¨é¢„åˆ†å‰²æ�©ç �' if ONLY_PUBLIC else 'å®�æ—¶åˆ†å‰²'}")
print("âœ… ç³»ç»Ÿåˆ�å§‹åŒ–å®Œæˆ�")


# ===== HPA å®�æ—¶åˆ†å‰²å™¨ =====
import scipy.ndimage as ndi
from skimage import transform, util, filters, measure, segmentation
from skimage.morphology import (binary_erosion, closing, disk,
                                remove_small_holes, remove_small_objects)

class HPASegmentator:
    """HPAå›¾åƒ�åˆ†å‰²å™¨ï¼Œç”¨äº�å¤„ç�†æœ¬åœ°HPAæ•°æ�®é›†"""
    
    COLORS = ["red", "green", "blue", "yellow"]  # microtubule, protein, nuclei, ER
    NORMALIZE = {"mean": [124 / 255, 117 / 255, 104 / 255], "std": [1 / (0.0167 * 255)] * 3}
    HIGH_THRESHOLD = 0.4
    LOW_THRESHOLD = HIGH_THRESHOLD - 0.25
    
    def __init__(self, model_dir="C:/Users/Admin/Downloads/hpaseg-model"):
        """
        åˆ�å§‹åŒ–åˆ†å‰²å™¨
        
        Args:
            model_dir: æ¨¡å�‹æ�ƒé‡�ç›®å½•
        """
        self.model_dir = Path(model_dir)
        
        # æ£€æŸ¥æ¨¡å�‹æ–‡ä»¶æ˜¯å�¦å­˜åœ¨
        nuclei_model_path = self.model_dir / 'nuclei_model.pt'
        cell_model_path = self.model_dir / 'cell_model.pt'
        
        if not nuclei_model_path.exists() or not cell_model_path.exists():
            print(f"âš ï¸� åˆ†å‰²æ¨¡å�‹æœªæ‰¾åˆ°: {self.model_dir}")
            print("ğŸ’¡ è¯·ç¡®ä¿�æ¨¡å�‹æ–‡ä»¶å­˜åœ¨æˆ–ä½¿ç”¨ONLY_PUBLIC=Trueæ¨¡å¼�")
            self.models_available = False
            return
        
        # åŠ è½½æ¨¡å�‹
        print(f"ğŸ“¦ åŠ è½½HPAåˆ†å‰²æ¨¡å�‹ä»�: {self.model_dir}")
        try:
            self.nuclei_model = torch.jit.load(str(nuclei_model_path), map_location=device)
            self.cell_model = torch.jit.load(str(cell_model_path), map_location=device)
            
            # è®¾ç½®ä¸ºè¯„ä¼°æ¨¡å¼�
            self.nuclei_model.eval()
            self.cell_model.eval()
            self.models_available = True
            
            print("âœ… HPAåˆ†å‰²æ¨¡å�‹åŠ è½½å®Œæˆ�ï¼�")
        except Exception as e:
            print(f"â�Œ åˆ†å‰²æ¨¡å�‹åŠ è½½å¤±è´¥: {e}")
            self.models_available = False
    
    def preprocess(self, image, scale_factor=0.25):
        """é¢„å¤„ç�†å›¾åƒ�"""
        assert len(image.shape) == 3
        image = transform.rescale(image, scale_factor, channel_axis=-1)
        image = image.transpose([2, 0, 1])
        return image
    
    def run_model(self, model, imgs):
        """è¿�è¡Œæ¨¡å�‹æ�¨ç�†"""
        imgs = torch.tensor(imgs).float().to(device)
        mean = torch.as_tensor(self.NORMALIZE["mean"]).to(device)
        std = torch.as_tensor(self.NORMALIZE["std"]).to(device)
        imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])
        
        with torch.no_grad():
            imgs = model(imgs)
        
        imgs = imgs.to("cpu").detach().numpy()
        imgs = transform.rescale(imgs[0].transpose([1, 2, 0]), 4, channel_axis=-1)
        imgs = util.img_as_ubyte(imgs)
        return imgs
    
    def __fill_holes(self, image):
        """å¡«å……æ ‡è®°å›¾åƒ�ä¸­çš„å­”æ´�"""
        boundaries = segmentation.find_boundaries(image)
        image = np.multiply(image, np.invert(boundaries))
        image = ndi.binary_fill_holes(image > 0)
        image = ndi.label(image)[0]
        return image
    
    def label_nuclei(self, nuclei_pred):
        """æ ‡è®°ç»†èƒ�æ ¸"""
        img_copy = np.copy(nuclei_pred[..., 2])
        borders = (nuclei_pred[..., 1] > 0.05).astype(np.uint8)
        m = img_copy * (1 - borders)
        
        img_copy[m <= self.LOW_THRESHOLD] = 0
        img_copy[m > self.LOW_THRESHOLD] = 1
        img_copy = img_copy.astype(bool)  # ä¿®å¤�numpyå…¼å®¹æ€§
        img_copy = binary_erosion(img_copy)
        img_copy = img_copy.astype(np.uint8)
        markers = measure.label(img_copy).astype(np.uint32)
        
        mask_img = np.copy(nuclei_pred[..., 2])
        mask_img[mask_img <= self.HIGH_THRESHOLD] = 0
        mask_img[mask_img > self.HIGH_THRESHOLD] = 1
        mask_img = mask_img.astype(bool)  # ä¿®å¤�numpyå…¼å®¹æ€§
        mask_img = remove_small_holes(mask_img, 1000)
        mask_img = mask_img.astype(np.uint8)
        
        nuclei_label = segmentation.watershed(
            mask_img, markers, mask=mask_img, watershed_line=True
        )
        nuclei_label = remove_small_objects(nuclei_label, 2500)
        nuclei_label = measure.label(nuclei_label)
        return nuclei_label
    
    def label_cell(self, nuclei_pred, cell_pred):
        """æ ‡è®°ç»†èƒ�å’Œç»†èƒ�æ ¸"""
        def __wsh(
            mask_img,
            threshold,
            border_img,
            seeds,
            threshold_adjustment=0.35,
            small_object_size_cutoff=10,
        ):
            img_copy = np.copy(mask_img)
            m = seeds * border_img
            img_copy[m <= threshold + threshold_adjustment] = 0
            img_copy[m > threshold + threshold_adjustment] = 1
            img_copy = img_copy.astype(bool)  # ä¿®å¤�numpyå…¼å®¹æ€§
            img_copy = remove_small_objects(img_copy, small_object_size_cutoff).astype(
                np.uint8
            )
            
            mask_img[mask_img <= threshold] = 0
            mask_img[mask_img > threshold] = 1
            mask_img = mask_img.astype(bool)  # ä¿®å¤�numpyå…¼å®¹æ€§
            mask_img = remove_small_holes(mask_img, 1000)
            mask_img = remove_small_objects(mask_img, 8).astype(np.uint8)
            markers = ndi.label(img_copy, output=np.uint32)[0]
            labeled_array = segmentation.watershed(
                mask_img, markers, mask=mask_img, watershed_line=True
            )
            return labeled_array
        
        nuclei_label = __wsh(
            nuclei_pred[..., 2] / 255.0,
            0.4,
            1 - (nuclei_pred[..., 1] + cell_pred[..., 1]) / 255.0 > 0.05,
            nuclei_pred[..., 2] / 255,
            threshold_adjustment=-0.25,
            small_object_size_cutoff=500,
        )
        
        nuclei_label = remove_small_objects(nuclei_label, 2500)
        nuclei_label = measure.label(nuclei_label)
        
        threshold_value = max(0.22, filters.threshold_otsu(cell_pred[..., 2] / 255) * 0.5)
        
        cell_region = np.multiply(
            cell_pred[..., 2] / 255 > threshold_value,
            np.invert(np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8)),
        )
        sk = np.asarray(cell_region, dtype=np.int8)
        distance = np.clip(cell_pred[..., 2], 255 * threshold_value, cell_pred[..., 2])
        cell_label = segmentation.watershed(-distance, nuclei_label, mask=sk)
        cell_label = remove_small_objects(cell_label, 5500).astype(np.uint8)
        selem = disk(6)
        cell_label = closing(cell_label, selem)
        cell_label = self.__fill_holes(cell_label)
        
        sk = np.asarray(
            np.add(
                np.asarray(cell_label > 0, dtype=np.int8),
                np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8),
            )
            > 0,
            dtype=np.int8,
        )
        cell_label = segmentation.watershed(-distance, cell_label, mask=sk)
        cell_label = self.__fill_holes(cell_label)
        cell_label = np.asarray(cell_label > 0, dtype=np.uint8)
        cell_label = measure.label(cell_label)
        cell_label = remove_small_objects(cell_label, 5500)
        cell_label = measure.label(cell_label)
        cell_label = np.asarray(cell_label, dtype=np.uint16)
        nuclei_label = np.multiply(cell_label > 0, nuclei_label) > 0
        nuclei_label = measure.label(nuclei_label)
        nuclei_label = remove_small_objects(nuclei_label, 2500)
        nuclei_label = np.multiply(cell_label, nuclei_label > 0)
        
        return nuclei_label, cell_label

    def pad_image(self, image, pad_size=32):
        """
        å¯¹(4, h, w)å½¢çŠ¶çš„å›¾åƒ�è¿›è¡Œå¡«å……
        """
        _, h, w = image.shape
        bottom = pad_size - h % pad_size
        right = pad_size - w % pad_size

        padded_image = np.pad(image,
                              ((0, 0),  # é€šé�“ç»´åº¦ä¸�å¡«å……
                              (pad_size, bottom),
                              (pad_size, right)),
                              mode='reflect')

        return padded_image, (h, w)

    def unpad_image(self, padded_image, original_shape):
        """
        ç§»é™¤å¡«å……ï¼Œè¿˜å�Ÿåˆ°å�Ÿå§‹å°ºå¯¸
        """
        h, w = original_shape
        return padded_image[32:32 + h, 32:32 + w, :]
      
    def segment_image(self, image):
        """
        å¯¹å�•ä¸ªå›¾åƒ�è¿›è¡Œåˆ†å‰²
        
        Args:
            image: 4é€šé�“å›¾åƒ� [H, W, 4]
        
        Returns:
            dict: åŒ…å�«nuclei_maskå’Œcell_maskçš„å­—å…¸
        """
        if not self.models_available:
            raise RuntimeError("åˆ†å‰²æ¨¡å�‹æœªåŠ è½½ï¼Œæ— æ³•æ‰§è¡Œå®�æ—¶åˆ†å‰²")
        
        # è½¬æ�¢å›¾åƒ�æ ¼å¼� [H, W, 4] -> [4, H, W]
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image.transpose(2, 0, 1)
        
        w, h = image.shape[1], image.shape[2]
        original_shape = None
        
        # å¤„ç�†ç‰¹å®šå°ºå¯¸çš„å›¾åƒ�å¡«å……
        if w == 1728 and h == 1728:
            image, original_shape = self.pad_image(image)
        
        # å‡†å¤‡ç»†èƒ�æ ¸å›¾åƒ�
        nuc_image = image[2, :, :]  # è“�è‰²é€šé�“ï¼ˆç»†èƒ�æ ¸ï¼‰
        nuc_image = np.stack((nuc_image, nuc_image, nuc_image), axis=-1)
        nuc_image = self.preprocess(nuc_image)
        
        # è¿�è¡Œç»†èƒ�æ ¸æ¨¡å�‹
        nuc_segmentations = self.run_model(self.nuclei_model, nuc_image[None, :, :, :])
        
        # å‡†å¤‡ç»†èƒ�å›¾åƒ� (microtubule/red, ER/yellow, nuclei/blue)
        cell_image = np.stack((image[0, :, :], image[3, :, :], image[2, :, :]), axis=-1)
        cell_image = self.preprocess(cell_image)
        
        # è¿�è¡Œç»†èƒ�æ¨¡å�‹
        cell_segmentations = self.run_model(self.cell_model, cell_image[None, :, :, :])
        
        # å¦‚æ�œè¿›è¡Œäº†å¡«å……ï¼Œéœ€è¦�å�»é™¤å¡«å……
        if original_shape is not None:
            nuc_segmentations = self.unpad_image(nuc_segmentations, original_shape)
            cell_segmentations = self.unpad_image(cell_segmentations, original_shape)
        
        # å��å¤„ç�†
        nuclei_mask = self.label_nuclei(nuc_segmentations)
        nuclei_mask, cell_mask = self.label_cell(nuc_segmentations, cell_segmentations)
        
        return {
            'nuclei_mask': nuclei_mask,
            'cell_mask': cell_mask
        }
    
    def generate_submission_rles(self, mask):
        """
        ä»�æ�©ç �ç”Ÿæˆ�æ��äº¤ç”¨çš„RLEåˆ—è¡¨
        
        Args:
            mask: ç»†èƒ�æ�©ç �ï¼Œæ¯�ä¸ªç»†èƒ�æœ‰å”¯ä¸€çš„é��é›¶æ ‡ç­¾
        
        Returns:
            list: RLEå­—ç¬¦ä¸²åˆ—è¡¨
        """
        submission_rles = []
        unique_cells = np.unique(mask)
        cell_labels = unique_cells[unique_cells != 0]
        
        for label in cell_labels:
            rle_str = binary_mask_to_ascii(mask, label)
            if rle_str.strip():  # ç¡®ä¿�RLEå­—ç¬¦ä¸²ä¸�ä¸ºç©º
                submission_rles.append(rle_str)
        
        return submission_rles

# é‡�æ–°åˆ�å§‹åŒ–HPAåˆ†å‰²å™¨ï¼ˆä¿®å¤�numpyå…¼å®¹æ€§é—®é¢˜å��ï¼‰
print("ğŸ”§ é‡�æ–°åˆ�å§‹åŒ–HPAåˆ†å‰²å™¨...")
hpa_segmentator = HPASegmentator(model_dir=HPA_SEGMENTATION_MODEL_DIR)

if hpa_segmentator.models_available:
    print("âœ… HPAåˆ†å‰²å™¨åˆ�å§‹åŒ–æˆ�åŠŸï¼Œå®�æ—¶åˆ†å‰²æ¨¡å¼�å�¯ç”¨")
else:
    print("âš ï¸� HPAåˆ†å‰²å™¨åˆ�å§‹åŒ–å¤±è´¥ï¼Œå°†ä»…æ”¯æŒ�é¢„åˆ†å‰²æ¨¡å¼�")

print("âœ… HPAåˆ†å‰²å™¨å‡†å¤‡å®Œæˆ�")


# ===== æ¨¡å�‹åŠ è½½å’Œå�˜æ�¢é…�ç½® =====
print("ğŸ¤– åŠ è½½ TorchScript æ¨¡å�‹...")

try:
    # åŠ è½½æ¨¡å�‹
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    print("âœ… TorchScript æ¨¡å�‹åŠ è½½æˆ�åŠŸ")

    # æµ‹è¯•æ¨¡å�‹è¾“å…¥è¾“å‡º
    test_input = torch.randn(1, 4, 224, 224).to(device)
    with torch.no_grad():
        output = model(test_input)
        if isinstance(output, tuple):
            logits, features = output
            print(f"ğŸ“Š æ¨¡å�‹è¾“å‡º: logits {logits.shape}, features {features.shape}")
        else:
            logits = output
            print(f"ğŸ“Š æ¨¡å�‹è¾“å‡º: logits {logits.shape}")

    print("âœ… æ¨¡å�‹æµ‹è¯•æˆ�åŠŸ")

except Exception as e:
    print(f"â�Œ æ¨¡å�‹åŠ è½½å¤±è´¥: {e}")
    raise


# ===== å›¾åƒ�å�˜æ�¢ç®¡é�“ =====
# æ ‡å‡†å�˜æ�¢ï¼ˆç”¨äº�å¸¸è§„æ�¨ç�†ï¼‰
to_tensor = transforms.Compose(
    [
        ToTensor(),
        Normalize(
            mean=[0.08069, 0.05258, 0.05487, 0.08282],
            std=[0.13704, 0.10145, 0.15313, 0.13814],
        ),
    ]
)


# è‡ªå®šä¹‰å½’ä¸€åŒ–ç±»
class MinMaxNormalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min_val = torch.amin(x, dim=(1, 2), keepdim=True)
        max_val = torch.amax(x, dim=(1, 2), keepdim=True)
        return (x - min_val) / (max_val - min_val + 1e-6)


# TTAå�˜æ�¢åŒ…
transforms_pack = [
    v2.RandomAffine(
        degrees=(-180, 180),
        translate=None,  # ç¦�ç”¨å¹³ç§»
        scale=(0.9, 1.1),  # éš�æœºç¼©æ”¾
        shear=15,  # x/y è½´å‰ªåˆ‡èŒƒå›´ä¸º Â±15 åº¦
    ),
    v2.RandomHorizontalFlip(p=0.4),  # 40% æ°´å¹³ç¿»è½¬æ¦‚ç�‡
    v2.RandomVerticalFlip(p=0.4),  # 40% å�‚ç›´ç¿»è½¬æ¦‚ç�‡
]

# TTAå�˜æ�¢ç®¡é�“
tta_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # ç¼©æ”¾åƒ�ç´ å€¼åˆ° [0, 1]
        v2.RandomOrder(transforms_pack),  # éš�æœºé¡ºåº�æ‰§è¡Œå¢�å¼º
        v2.Normalize(
            mean=[0.08069, 0.05258, 0.05487, 0.08282],
            std=[0.13704, 0.10145, 0.15313, 0.13814],
        ),
    ]
)

print(f"âœ… æ¨¡å�‹åŠ è½½å®Œæˆ�ï¼Œæ�¨ç�†çº§åˆ«: {MODEL_LEVEL}")
print(f"âœ… å›¾åƒ�å�˜æ�¢ç®¡é�“åˆ›å»ºå®Œæˆ�")
print(f"ğŸ”§ TTAé…�ç½®: {'å�¯ç”¨' if DO_TTA else 'ç¦�ç”¨'} ({TTA_REPEATS} æ¬¡é‡�å¤�)")


# ===== æ•°æ�®åŠ è½½å’Œæ ¸å¿ƒå·¥å…·å‡½æ•° =====
print("ğŸ“Š åŠ è½½æ•°æ�®å’Œå·¥å…·å‡½æ•°...")

def rle_to_mask2(rle_string: str, height: int, width: int) -> np.ndarray:
    """å°†RLEå­—ç¬¦ä¸²è½¬æ�¢ä¸ºäºŒå€¼mask"""
    rle_numbers = [int(x) for x in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in rle_pairs:
        mask[start:start + length] = 1
    
    return mask.reshape(height, width)

def rle_to_mask(rle_string, height, width):
    """ Convert RLE sttring into a binary mask 
    
    Args:
        rle_string (rle_string): Run length encoding containing 
            segmentation mask information
        height (int): Height of the original image the map comes from
        width (int): Width of the original image the map comes from
    
    Returns:
        Numpy array of the binary segmentation mask for a given cell
    """
    rows,cols = height,width
    rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rle_pairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img

def binary_mask_to_ascii(mask: np.ndarray, mask_val: int) -> str:
    """å°†äºŒå€¼maskè½¬æ�¢ä¸ºRLEå­—ç¬¦ä¸²"""
    binary_mask = (mask == mask_val).astype(np.uint8)
    flat_mask = binary_mask.flatten()
    
    # æ‰¾åˆ°æ‰€æœ‰å�˜åŒ–ç‚¹
    diff = np.diff(np.concatenate(([0], flat_mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    rle_list = []
    for start, end in zip(starts, ends):
        rle_list.extend([start, end - start])
    
    return ' '.join(map(str, rle_list))

def load_img(image_dir: str, image_id: str) -> np.ndarray:
    """åŠ è½½4é€šé�“RGBYå›¾åƒ�"""
    channels = ['red', 'green', 'blue', 'yellow']
    channel_images = []
    
    for channel in channels:
        img_path = os.path.join(image_dir, f"{image_id}_{channel}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"å›¾åƒ�æ–‡ä»¶ä¸�å­˜åœ¨: {img_path}")
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"æ— æ³•è¯»å�–å›¾åƒ�: {img_path}")
        
        channel_images.append(img)
    
    # å�ˆå¹¶ä¸º4é€šé�“å›¾åƒ� [H, W, 4]
    merged_image = np.stack(channel_images, axis=2)
    return merged_image

def pad_to_square(image: np.ndarray, value: int = 0) -> np.ndarray:
    """å°†å›¾åƒ�å¡«å……ä¸ºæ­£æ–¹å½¢"""
    h, w = image.shape[:2]
    if h == w:
        return image
    
    max_dim = max(h, w)
    if len(image.shape) == 3:
        padded = np.full((max_dim, max_dim, image.shape[2]), value, dtype=image.dtype)
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
    else:
        padded = np.full((max_dim, max_dim), value, dtype=image.dtype)
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return padded

def split_cells(image: np.ndarray, mask: np.ndarray, img_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    ä»�å›¾åƒ�ä¸­æ��å�–ç»†èƒ�ï¼Œå¡«å……ä¸ºæ­£æ–¹å½¢å¹¶è°ƒæ•´å¤§å°�
    
    Args:
        image: å››é€šé�“å›¾åƒ� [H, W, 4]
        mask: ç»†èƒ�æ�©ç �ï¼Œæ¯�ä¸ªç»†èƒ�æœ‰å”¯ä¸€çš„é��é›¶æ ‡ç­¾
        img_size: ç›®æ ‡å°ºå¯¸ (width, height)
    
    Returns:
        ç»†èƒ�å›¾åƒ�åˆ—è¡¨
    """
    # ç¡®ä¿�imageå’Œmaskçš„å½¢çŠ¶åŒ¹é…�
    assert image.shape[:2] == mask.shape, "Image and mask shapes do not match"
    
    # è�·å�–å”¯ä¸€çš„ç»†èƒ�æ ‡ç­¾ï¼ˆæ�’é™¤èƒŒæ™¯0ï¼‰
    unique_cells = np.unique(mask)
    cell_labels = unique_cells[unique_cells != 0]
    
    extracted_cells = []
    for label in cell_labels:
        # åˆ›å»ºå½“å‰�ç»†èƒ�çš„äºŒå€¼æ�©ç �
        cell_mask = (mask == label).astype(np.uint8)
        
        # æ‰¾åˆ°ç»†èƒ�çš„è¾¹ç•Œæ¡†
        rows, cols = np.where(cell_mask)
        if len(rows) == 0:
            continue
            
        top, bottom = rows.min(), rows.max()
        left, right = cols.min(), cols.max()
        
        # æ��å�–ç»†èƒ�åŒºåŸŸ
        cell_image = image[top:bottom + 1, left:right + 1].copy()
        cell_mask_crop = cell_mask[top:bottom + 1, left:right + 1]
        
        # åº”ç”¨æ�©ç �ï¼ˆä¿�ç•™4ä¸ªé€šé�“ï¼‰
        if len(cell_image.shape) == 3:
            cell_mask_4ch = np.repeat(cell_mask_crop[:, :, np.newaxis], 4, axis=2)
            cell_image = cell_image * cell_mask_4ch
        else:
            cell_image = cell_image * cell_mask_crop
        
        # å¡«å……ä¸ºæ­£æ–¹å½¢
        cell_image_square = pad_to_square(cell_image, 0)
        
        # è°ƒæ•´å¤§å°�
        cell_image_resized = cv2.resize(cell_image_square, img_size, interpolation=cv2.INTER_LINEAR)
        
        extracted_cells.append(cell_image_resized)
    
    return extracted_cells

def get_contour_bbox_from_raw(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """ä»�æ�©ç �è�·å�–æ‰€æœ‰ç»†èƒ�çš„è¾¹ç•Œæ¡†"""
    bboxes = []
    unique_cells = np.unique(mask)
    cell_labels = unique_cells[unique_cells != 0]
    
    for label in cell_labels:
        cell_mask = (mask == label).astype(np.uint8)
        rows, cols = np.where(cell_mask)
        
        if len(rows) == 0:
            continue
            
        top, bottom = rows.min(), rows.max()
        left, right = cols.min(), cols.max()
        
        # è¿”å›�æ ¼å¼�: (x1, y1, x2, y2)
        bboxes.append((left, top, right + 1, bottom + 1))
    
    return bboxes

def get_image_ids():
    """
    ç»Ÿä¸€çš„å›¾åƒ�IDè�·å�–å‡½æ•°
    
    Returns:
        tuple: (all_ids, pub_ss_df) - å›¾åƒ�IDåˆ—è¡¨å’Œé¢„åˆ†å‰²æ•°æ�®æ¡†ï¼ˆå¦‚æ�œå­˜åœ¨ï¼‰
    """
    pub_ss_df = None
    
    if ONLY_PUBLIC:
        # é¢„åˆ†å‰²æ¨¡å¼�ï¼šä»�CSVæ–‡ä»¶åŠ è½½æ•°æ�®
        print("ğŸ“‚ åŠ è½½é¢„åˆ†å‰²æ•°æ�®...")
        pub_ss_df = pd.read_csv(SUBMISSION_FILE)

        # è§£æ��å­—ç¬¦ä¸²æ ¼å¼�çš„åˆ—è¡¨
        pub_ss_df.mask_rles = pub_ss_df.mask_rles.apply(lambda x: ast.literal_eval(x))
        pub_ss_df.mask_bboxes = pub_ss_df.mask_bboxes.apply(lambda x: ast.literal_eval(x))
        pub_ss_df.mask_sub_rles = pub_ss_df.mask_sub_rles.apply(lambda x: ast.literal_eval(x))
        
        all_ids = pub_ss_df.ID.tolist()
        
        print("âœ… é¢„åˆ†å‰²æ•°æ�®åŠ è½½å®Œæˆ�")
        print(f"ğŸ“Š æ•°æ�®å½¢çŠ¶: {pub_ss_df.shape}")
        
            
    else:
        # å®�æ—¶åˆ†å‰²æ¨¡å¼�ï¼šæ‰«æ��å›¾åƒ�ç›®å½•
        print("ğŸ“‚ æ‰«æ��å›¾åƒ�ç›®å½•...")
        
        all_ids = []
        for file in os.listdir(IMAGE_DIR):
            if file.endswith('_red.png'):
                image_id = file.replace('_red.png', '')
                all_ids.append(image_id)
        print(f"ğŸ“Š å�‘ç�° {len(all_ids)} ä¸ªå›¾åƒ�æ–‡ä»¶")
    
    print(f"ğŸ“� æ€»å›¾åƒ�IDæ•°é‡�: {len(all_ids)}")
    return all_ids, pub_ss_df



def plot_test(image, mask):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image[..., :3])  # å�ªæ˜¾ç¤ºRGBé€šé�“
    plt.title("Test Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='viridis')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.show()

# ===== æ•°æ�®åŠ è½½ =====
print("ğŸ“Š å¼€å§‹æ•°æ�®åŠ è½½...")
all_ids, pub_ss_df = get_image_ids()
print("âœ… æ•°æ�®åŠ è½½å’Œå·¥å…·å‡½æ•°å‡†å¤‡å®Œæˆ�")

# ===== é‡�æ�„çš„æ�¨ç�†å¼•æ“� - å®Œå…¨æŒ‰ç…§å�Ÿå§‹æµ�ç¨‹ =====
class HELPInferenceEngine:
    """HELPæ¨¡å�‹æ�¨ç�†å¼•æ“� - æŒ‰ç…§å�Ÿå§‹ä»£ç �æµ�ç¨‹å®�ç�°"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def run_inference(self):
        """è¿�è¡Œå®Œæ•´æ�¨ç�†æµ�ç¨‹"""
        print("ğŸš€ å¼€å§‹HELPæ¨¡å�‹æ�¨ç�†...")
        
        # ä½¿ç”¨å…¨å±€å�˜é‡�ä¸­çš„å›¾åƒ�IDåˆ—è¡¨
        sub_df = pd.DataFrame(columns=["ID"], data=all_ids)
        sub_df['ImageWidth'] = 0
        sub_df['ImageHeight'] = 0
        # æ¼”ç¤ºæ¨¡å¼�é™�åˆ¶å¤„ç�†æ•°é‡�
        if IS_DEMO:
            sub_df = sub_df[:5]  # æ¼”ç¤ºæ¨¡å¼�å�ªå¤„ç�†å‰�5ä¸ªå›¾åƒ�
            
        print(f"ğŸ“Š å¤„ç�†å›¾åƒ�æ€»æ•°: {len(sub_df)}")
        
        # é��å�†æ¯�ä¸ªå›¾åƒ�è¿›è¡Œæ�¨ç�†
        for index, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc="æ�¨ç�†è¿›åº¦"):
            submission_id = row["ID"]
            
            try:
                # Step 0: åŠ è½½å›¾åƒ�
                images = load_img(IMAGE_DIR, submission_id)
                image_height, image_width = images.shape[:2]
                
                if ONLY_PUBLIC:
                    # é¢„åˆ†å‰²æ¨¡å¼�ï¼šä»�å…¨å±€æ•°æ�®æ¡†è�·å�–ä¿¡æ�¯
                    image_info = pub_ss_df.loc[pub_ss_df['ID'] == submission_id]
                    if len(image_info) == 0:
                        print(f"âš ï¸� å›¾åƒ� {submission_id} åœ¨é¢„åˆ†å‰²æ•°æ�®ä¸­æœªæ‰¾åˆ°ï¼Œè·³è¿‡")
                        sub_df.loc[sub_df["ID"] == submission_id, "PredictionString"] = ""
                        continue
                    
                    image_width = image_info['ImageWidth'].values[0]
                    image_height = image_info['ImageHeight'].values[0]
                    
                    # è�·å�–æ��äº¤RLE
                    submission_rles = image_info['mask_sub_rles'].values
                    
                    # è�·å�–æ�©ç �RLEå¹¶æ�„å»ºå®Œæ•´æ�©ç �
                    mask_rles = image_info['mask_rles'].values[0]
                    mask = np.zeros((image_height, image_width), dtype=np.uint16)
                    
                    # ä¸ºæ¯�ä¸ªç»†èƒ�åˆ†é…�ç‹¬ç«‹æ ‡ç­¾
                    for idx, rle_str in enumerate(mask_rles, start=1):
                        binary_mask = rle_to_mask(rle_str, image_width, image_height)
                        mask[binary_mask > 0] = idx
                        
                else:
                    # å®�æ—¶åˆ†å‰²æ¨¡å¼�ï¼ˆä½¿ç”¨HPAåˆ†å‰²å™¨ï¼‰
                    # print(f"ğŸ”¬ å¯¹å›¾åƒ� {submission_id} æ‰§è¡Œå®�æ—¶åˆ†å‰²...")
                    
                    if not hpa_segmentator.models_available:
                        print("â�Œ HPAåˆ†å‰²å™¨ä¸�å�¯ç”¨ï¼Œè·³è¿‡æ­¤å›¾åƒ�")
                        sub_df.loc[sub_df["ID"] == submission_id, "PredictionString"] = ""
                        continue
                    
                    try:
                        # æ‰§è¡Œåˆ†å‰²
                        segmentation_result = hpa_segmentator.segment_image(images)
                        mask = segmentation_result['cell_mask']
                        
                        # ç”Ÿæˆ�æ��äº¤ç”¨çš„RLE
                        submission_rles = [hpa_segmentator.generate_submission_rles(mask)]
                        
                        # print(f"âœ… å›¾åƒ� {submission_id} å®�æ—¶åˆ†å‰²å®Œæˆ�ï¼Œæ£€æµ‹åˆ° {np.max(mask)} ä¸ªç»†èƒ�")
                        
                    except Exception as seg_error:
                        print(f"â�Œ å›¾åƒ� {submission_id} åˆ†å‰²å¤±è´¥: {seg_error}")
                        sub_df.loc[sub_df["ID"] == submission_id, "PredictionString"] = ""
                        continue
                
                # print(f"âœ… å›¾åƒ� {submission_id} åŠ è½½å’Œæ�©ç �æ�„å»ºå®Œæˆ�")
                sub_df.loc[index, "ImageWidth"] = image_width
                sub_df.loc[index, "ImageHeight"] = image_height
                # å�¯è§†åŒ–ï¼ˆä»…å‰�å‡ ä¸ªå›¾åƒ�ï¼‰
                if index < 3:
                    plot_test(images, mask)
                
                # Step 6: æ��å�–ç»†èƒ�ã€�å¡«å……ä¸ºæ­£æ–¹å½¢å¹¶è°ƒæ•´å¤§å°�
                cell_tiles = split_cells(images, mask, TILE_SIZE)
                
                if len(cell_tiles) == 0:
                    # æ²¡æœ‰æ£€æµ‹åˆ°ç»†èƒ�
                    print(f"âš ï¸� å›¾åƒ� {submission_id} æœªæ£€æµ‹åˆ°ç»†èƒ�")
                    sub_df.loc[sub_df["ID"] == submission_id, "PredictionString"] = ""
                    continue
                
                # print(f"ğŸ“Š å›¾åƒ� {submission_id} æ��å�–åˆ° {len(cell_tiles)} ä¸ªç»†èƒ�")
                
                # Step 7: æµ‹è¯•æ—¶å¢�å¼ºï¼ˆTTAï¼‰
                if DO_TTA:
                    # TTAæ¨¡å¼�
                    tta_preds = []
                    
                    for _ in range(TTA_REPEATS):
                        augmented_images = [tta_transforms(cell_img) for cell_img in cell_tiles]
                        augmented_images = torch.stack(augmented_images, dim=0).to(self.device)
                        
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                if MODEL_LEVEL == 'cell':
                                    cell_logits, features = self.model(augmented_images)
                                    _preds = torch.sigmoid(cell_logits)
                                    tta_preds.append(_preds.cpu().detach().numpy())
                                else:
                                    cell_logits, query_logits = self.model(augmented_images, len(augmented_images))
                                    cell_preds = torch.sigmoid(cell_logits)
                                    query_preds = torch.sigmoid(query_logits)
                                    preds = cell_preds * query_preds
                                    preds[:, 11] = cell_preds[:, 11]
                                    preds[:, 18] = cell_preds[:, 18]
                                    preds = preds.cpu().detach().numpy()
                                    tta_preds.append(preds)
                    
                    # å¹³å�‡æ‰€æœ‰TTAé¢„æµ‹
                    preds = np.mean(tta_preds, axis=0)
                else:
                    # æ ‡å‡†æ�¨ç�†æ¨¡å¼�
                    cell_images = [to_tensor(cell_img) for cell_img in cell_tiles]
                    
                    # Step 8: æ‰§è¡Œæ�¨ç�†
                    cell_cnt = len(cell_images)
                    cell_images = torch.stack(cell_images, dim=0).to(self.device)
                    
                    if MODEL_LEVEL == 'cell':
                        preds = np.zeros((cell_cnt, 19))
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                cell_logits, features = self.model(cell_images)
                        _preds = torch.sigmoid(cell_logits)
                        preds = _preds.cpu().detach().numpy()
                    else:
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                cell_logits, query_logits = self.model(cell_images, cell_cnt)
                        cell_preds = torch.sigmoid(cell_logits)
                        query_preds = torch.sigmoid(query_logits)
                        preds = cell_preds * query_preds
                        preds[:, 11] = cell_preds[:, 11]
                        preds[:, 18] = cell_preds[:, 18]
                        preds = preds.cpu().detach().numpy()
                
                # ç”Ÿæˆ�é¢„æµ‹å­—ç¬¦ä¸²
                pred_string = []
                for cell_pred, mask_rle in zip(preds, submission_rles[0]):
                    for i in range(len(cell_pred)):
                        conf = cell_pred[i]
                        if conf < CONF_THRESH:
                            continue
                        pred_string.append(f"{i} {conf} {mask_rle}")
                
                pred_string = " ".join(pred_string)
                sub_df.loc[sub_df["ID"] == submission_id, "PredictionString"] = pred_string
                
                # print(f"âœ… å›¾åƒ� {submission_id} æ�¨ç�†å®Œæˆ�")
                
            except Exception as e:
                print(f"â�Œ å¤„ç�†å›¾åƒ� {submission_id} å¤±è´¥: {e}")
                import traceback
                traceback.print_exc()
                sub_df.loc[sub_df["ID"] == submission_id, "PredictionString"] = ""
                continue
        
        return sub_df

# åˆ›å»ºæ�¨ç�†å¼•æ“�
inference_engine = HELPInferenceEngine(model, device)
print("âœ… HELPæ�¨ç�†å¼•æ“�åˆ›å»ºå®Œæˆ�")


# ===== å°�è§„æ¨¡æµ‹è¯• - éªŒè¯�å®�æ—¶åˆ†å‰²åŠŸèƒ½ =====
print("ğŸ§ª å¼€å§‹å°�è§„æ¨¡æµ‹è¯• - éªŒè¯�å®�æ—¶åˆ†å‰²åŠŸèƒ½")

# ä¸´æ—¶è®¾ç½®ä¸ºæ¼”ç¤ºæ¨¡å¼�ï¼Œå�ªå¤„ç�†å°‘é‡�å›¾åƒ�
original_demo_setting = IS_DEMO

# æµ‹è¯•å�•ä¸ªå›¾åƒ�çš„å®Œæ•´æµ�ç¨‹
test_image_id = all_ids[0]
print(f"ğŸ”¬ æµ‹è¯•å›¾åƒ�: {test_image_id}")

try:
    # åŠ è½½å›¾åƒ�
    print("ğŸ“¸ åŠ è½½å›¾åƒ�...")
    images = load_img(IMAGE_DIR, test_image_id)
    print(f"âœ… å›¾åƒ�åŠ è½½æˆ�åŠŸï¼Œå°ºå¯¸: {images.shape}")
    
    # æ‰§è¡Œå®�æ—¶åˆ†å‰²
    print("ğŸ”¬ æ‰§è¡Œå®�æ—¶åˆ†å‰²...")
    segmentation_result = hpa_segmentator.segment_image(images)
    mask = segmentation_result['cell_mask']
    
    cell_count = np.max(mask)
    print(f"âœ… åˆ†å‰²å®Œæˆ�ï¼Œæ£€æµ‹åˆ° {cell_count} ä¸ªç»†èƒ�")
    
    # ç”Ÿæˆ�RLE
    print("ğŸ“� ç”Ÿæˆ�RLE...")
    submission_rles = hpa_segmentator.generate_submission_rles(mask)
    print(f"âœ… ç”Ÿæˆ� {len(submission_rles)} ä¸ªRLEå­—ç¬¦ä¸²")
    
    # æ��å�–ç»†èƒ�å›¾åƒ�
    print("âœ‚ï¸� æ��å�–ç»†èƒ�å›¾åƒ�...")
    cell_tiles = split_cells(images, mask, TILE_SIZE)
    print(f"âœ… æ��å�–åˆ° {len(cell_tiles)} ä¸ªç»†èƒ�å›¾åƒ�")
    
    # æ˜¾ç¤ºç¬¬ä¸€ä¸ªç»†èƒ�å›¾åƒ�
    if len(cell_tiles) > 0:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(images[..., :3])
        plt.title("å�Ÿå§‹å›¾åƒ� (RGB)")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='viridis')
        plt.title(f"åˆ†å‰²æ�©ç � ({cell_count} ä¸ªç»†èƒ�)")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cell_tiles[0][..., :3])
        plt.title("æ��å�–çš„ç»†èƒ�ç¤ºä¾‹")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("ğŸ�‰ å�•å›¾åƒ�æµ‹è¯•æˆ�åŠŸï¼�")
    
except Exception as e:
    print(f"â�Œ æµ‹è¯•å¤±è´¥: {e}")
    import traceback
    traceback.print_exc()

# æ�¢å¤�å�Ÿå§‹è®¾ç½®
IS_DEMO = original_demo_setting
print(f"ğŸ”§ æ¼”ç¤ºæ¨¡å¼�å·²æ�¢å¤�: {IS_DEMO}")


# ===== å®Œæ•´æ�¨ç�†ç®¡é�“æ‰§è¡Œ =====
print("ğŸ�� å¼€å§‹å®Œæ•´æ�¨ç�†ç®¡é�“")
print(f"ğŸ”§ é…�ç½®å�‚æ•°:")
print(f"  ONLY_PUBLIC: {ONLY_PUBLIC}")
print(f"  DO_TTA: {DO_TTA}")
print(f"  TTA_REPEATS: {TTA_REPEATS}")
print(f"  CONF_THRESH: {CONF_THRESH}")
print(f"  TILE_SIZE: {TILE_SIZE}")
print(f"  MODEL_LEVEL: {MODEL_LEVEL}")

# æ‰§è¡Œæ�¨ç�†
print("\nâš¡ å¼€å§‹æ�¨ç�†...")
results_df = inference_engine.run_inference()

print("\nğŸ“Š æ�¨ç�†ç»“æ�œç»Ÿè®¡:")
print(f"  æ€»å›¾åƒ�æ•°: {len(results_df)}")

# åˆ†æ��ç»“æ�œ
non_empty_results = results_df[results_df['PredictionString'] != '']
empty_results = results_df[results_df['PredictionString'] == '']

print(f"  æœ‰é¢„æµ‹çš„å›¾åƒ�: {len(non_empty_results)}")
print(f"  ç©ºé¢„æµ‹çš„å›¾åƒ�: {len(empty_results)}")

def create_pred_col(row):
    """ Simple function to return the correct prediction string
    
    We will want the original public test dataframe submission when it is 
    available. However, we will use the swapped inn submission dataframe
    when it is not.
    
    Args:
        row (pd.Series): A row in the dataframe
    
    Returns:
        The prediction string
    """
    if pd.isnull(row.PredictionString_y):
        return row.PredictionString_x
    else:
        return row.PredictionString_y
# æ˜¾ç¤ºéƒ¨åˆ†ç»“æ�œé¢„è§ˆ
print(f"\nğŸ“‹ ç»“æ�œé¢„è§ˆ:")
results_df.head()

# å¦‚æ�œæœ‰ç©ºé¢„æµ‹ï¼Œæ˜¾ç¤ºä¸€äº›ä¿¡æ�¯
if len(empty_results) > 0:
    print(f"\nâš ï¸� ç©ºé¢„æµ‹å›¾åƒ�ç¤ºä¾‹ (å‰�5ä¸ª):")
    for idx, row in empty_results.head(5).iterrows():
        print(f"  {row['ID']}")

print("\nğŸ“� ç”Ÿæˆ�æœ€ç»ˆæ��äº¤æ–‡ä»¶...")

# ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
submission_filename = '/kaggle/working/submission.csv'
pub_ss_df = pub_ss_df.merge(results_df, how="left", on="ID")
pub_ss_df["PredictionString"] = pub_ss_df.apply(create_pred_col, axis=1)
pub_ss_df = pub_ss_df.drop(columns=["PredictionString_x", "PredictionString_y"])
pub_ss_df.to_csv(submission_filename, index=False)

print(f"âœ… æ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�: {submission_filename}")
print(f"ğŸ“� æ–‡ä»¶å¤§å°�: {os.path.getsize(submission_filename) / 1024 / 1024:.1f} MB")

# éªŒè¯�æ��äº¤æ–‡ä»¶æ ¼å¼�
print("\nğŸ”� éªŒè¯�æ��äº¤æ–‡ä»¶æ ¼å¼�:")
submission_validation = pd.read_csv(submission_filename)
print(f"  æ–‡ä»¶å½¢çŠ¶: {submission_validation.shape}")
print(f"  åˆ—å��: {list(submission_validation.columns)}")


