!pip install --no-index --find-links /kaggle/input/detectron2-whls --no-deps yacs portalocker pathspec iopath hydra-core black fvcore detectron2


!pip install --no-index --find-links /kaggle/input/cellpose-whl --no-deps cellpose fastremap fill_voids roifile


# ======================== Detectron2 ========================
import detectron2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fastcore.all import *
detectron2.__version__
from skimage.color import label2rgb


# ======================== Cellpose ========================
from cellpose import models, io
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

from tqdm import tqdm


from collections import defaultdict
from tqdm import tqdm
import skimage.io as io
from cellpose import dynamics


# From https://www.kaggle.com/stainsby/fast-tested-rle
def rle_decode(mask_rle, shape=(520, 704)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

def rle_encode(img):
    '''
    img: numpy array, 1 - foreground, 0 - background
    Returns run length as string
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def compute_iou(mask1, mask2):
    """
    è¨ˆç®—å…©å€‹äºŒå€¼ mask çš„ IoUï¼ˆIntersection over Unionï¼‰ã€‚
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


Dir_testdata=Path('../input/sartorius-cell-instance-segmentation')
maskrcnn_ids, maskrcnn_masks = [], []
test_image_names = (Dir_testdata/'test').ls()


# ======================== Detectron2 ========================
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 

THRESHOLDS = [.15, .35, .55]
MIN_PIXELS = [75, 150, 75]


def get_patches(image_shape, patch_size, stride):
    """
    è¨ˆç®—åœ–åƒ�åˆ†å¡Šçš„èµ·å§‹åº§æ¨™å’Œçµ�æ�Ÿåº§æ¨™ã€‚

    Args:
        image_shape (tuple): å�Ÿå§‹åœ–åƒ�çš„ (H, W)ã€‚
        patch_size (tuple): æ¯�å€‹åˆ†å¡Šçš„ (patch_H, patch_W)ã€‚
        stride (tuple): åˆ†å¡Šçš„æ­¥é•· (stride_H, stride_W)ã€‚

    Returns:
        list: åŒ…å�« (y_start, y_end, x_start, x_end) çš„å…ƒçµ„åˆ—è¡¨ã€‚
    """
    H, W = image_shape
    patch_H, patch_W = patch_size
    stride_H, stride_W = stride
    patches = []

    # è¨ˆç®—æ­¥é•·ï¼Œç¢ºä¿�è¦†è“‹æ‰€æœ‰å�€åŸŸ
    if H <= patch_H:
        y_steps = [0]
    else:
        y_steps = np.arange(0, H - patch_H + stride_H, stride_H)
        if y_steps[-1] < H - patch_H:
            y_steps = np.append(y_steps, H - patch_H)

    if W <= patch_W:
        x_steps = [0]
    else:
        x_steps = np.arange(0, W - patch_W + stride_W, stride_W)
        if x_steps[-1] < W - patch_W:
            x_steps = np.append(x_steps, W - patch_W)

    for y in y_steps:
        for x in x_steps:
            y_start = int(y)
            y_end = int(min(y + patch_H, H))
            x_start = int(x)
            x_end = int(min(x + patch_W, W))

            # èª¿æ•´èµ·å§‹é»�ä»¥ç¢ºä¿�å®Œæ•´è¦†è“‹é‚Šç·£
            if y_end - y_start < patch_H and y_end == H:
                y_start = H - patch_H
            if x_end - x_start < patch_W and x_end == W:
                x_start = W - patch_W

            y_start = max(0, y_start) # ç¢ºä¿�ä¸�ç‚ºè² 
            x_start = max(0, x_start) # ç¢ºä¿�ä¸�ç‚ºè² 

            patches.append((y_start, y_end, x_start, x_end))
    return patches # ç¢ºä¿�ç¸½æ˜¯è¿”å›� patches åˆ—è¡¨ï¼Œå�³ä½¿å®ƒæ˜¯ç©ºçš„


def stitch_masks_and_flows(patches_data, original_shape):
    """
    å°‡å¤šå€‹åˆ†å¡Šçš„æ©Ÿç�‡åœ–å’Œæµ�å ´æ‹¼æ�¥å›�å�Ÿå§‹åœ–åƒ�å°ºå¯¸ã€‚
    é‡�ç–Šå�€åŸŸä½¿ç”¨å¹³å�‡å€¼ã€‚

    Args:
        patches_data (list): æ¯�å€‹å…ƒç´ ç‚º (y_start, y_end, x_start, x_end, prob_map, dp_map)
        original_shape (tuple): å�Ÿå§‹åœ–åƒ�çš„ (H, W)ã€‚

    Returns:
        tuple: (stitched_prob_map, stitched_dp_map)
    """
    H_orig, W_orig = original_shape
    stitched_prob_map_total = np.zeros(original_shape, dtype=np.float32)
    stitched_dp_map_total = np.zeros((2, H_orig, W_orig), dtype=np.float32) # dP æ ¼å¼�ç‚º (2, H, W)
    pixel_counts = np.zeros(original_shape, dtype=np.float32) # ä½¿ç”¨ float é�¿å…�æº¢å‡º

    for y_s, y_e, x_s, x_e, prob_map, dp_map in patches_data:
        current_prob_h, current_prob_w = prob_map.shape[:2]
        current_dp_h, current_dp_w = dp_map.shape[1:]

        stitched_prob_map_total[y_s:y_e, x_s:x_e] += prob_map[:current_prob_h, :current_prob_w]
        stitched_dp_map_total[:, y_s:y_e, x_s:x_e] += dp_map[:, :current_dp_h, :current_dp_w]
        pixel_counts[y_s:y_e, x_s:x_e] += 1

    pixel_counts[pixel_counts == 0] = 1

    stitched_prob_map = stitched_prob_map_total / pixel_counts
    stitched_dp_map = stitched_dp_map_total / pixel_counts

    return stitched_prob_map, stitched_dp_map


import matplotlib.pyplot as plt
import numpy as np

def visualize_flow(avg_dp, image_id="image"):
    """
    è¦–è¦ºåŒ– avg_dp (flow) çµ�æ�œï¼ŒåŒ…å�«ï¼š
    - dxï¼ˆæ°´å¹³åˆ†é‡�ï¼‰
    - dyï¼ˆå�‚ç›´åˆ†é‡�ï¼‰
    - magnitudeï¼ˆæ¨¡é•·ï¼‰
    """
    if avg_dp.shape[-1] != 2:
        raise ValueError("avg_dp é ˆç‚º [H, W, 2] å½¢ç‹€")

    dx = avg_dp[:, :, 0]
    dy = avg_dp[:, :, 1]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"{image_id} - Flow dx")
    plt.imshow(dx, cmap="seismic")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title(f"{image_id} - Flow dy")
    plt.imshow(dy, cmap="seismic")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title(f"{image_id} - Flow Magnitude")
    plt.imshow(magnitude, cmap="viridis")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def resize_patch_to_height(patch_img, target_h):
    h, w = patch_img.shape[:2]
    scale = target_h / h
    target_w = int(w * scale)
    return cv2.resize(patch_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

def predict_with_maskrcnn_patching(predictor, img_rgb, img_id, patch_size, stride):
    """
    ä½¿ç”¨ Mask R-CNN é€²è¡Œåœ–åƒ�åˆ†å¡Š (patching) å’Œå¤šå°ºåº¦ TTA æ�¨è«–ï¼Œä¸¦æ‹¼æ�¥çµ�æ�œã€‚

    Args:
        predictor: Mask R-CNN çš„ predictor å¯¦ä¾‹ã€‚
        img_rgb (np.ndarray): è¼¸å…¥çš„å�Ÿå§‹ RGB åœ–åƒ�ã€‚
        img_id (str): åœ–åƒ�çš„ IDã€‚
        patch_size (tuple): åˆ†å¡Šçš„ (patch_H, patch_W)ã€‚
        stride (tuple): åˆ†å¡Šçš„æ­¥é•· (stride_H, stride_W)ã€‚

    Returns:
        tuple: (fused_prob_map, fused_dp_map)
               - fused_prob_map (np.ndarray): æ‹¼æ�¥å¾Œçš„ Mask R-CNN æ©Ÿç�‡åœ– (H, W)ã€‚
               - fused_dp_map (np.ndarray): æ‹¼æ�¥å¾Œçš„ Mask R-CNN æµ�å�‘é‡�åœ– (2, H, W)ã€‚
    """
    original_shape = img_rgb.shape[:2]
    PATCH_H, PATCH_W = patch_size
    STRIDE_H, STRIDE_W = stride
    resize_heights = [440, 480, 520, 560, 580, 620]

    patches_to_process = get_patches(original_shape, (PATCH_H, PATCH_W), (STRIDE_H, STRIDE_W))
    all_fused_patch_data = []

    for y_s, y_e, x_s, x_e in patches_to_process:
        current_patch_img = img_rgb[y_s:y_e, x_s:x_e]

        if current_patch_img.shape[0] < PATCH_H // 2 or current_patch_img.shape[1] < PATCH_W // 2:
            continue

        prob_accumulator = []
        dp_accumulator = []

        for h in resize_heights:
            resized_patch = resize_patch_to_height(current_patch_img, h)

            # å–®å°ºå¯¸ TTA é �æ¸¬
            prob_map, dp_map = tta_predict_probability_and_flows(
                predictor, resized_patch, img_id, model_type="maskrcnn"
            )

            # resize å›�å�Ÿ patch å°ºå¯¸
            prob_map = cv2.resize(prob_map, (current_patch_img.shape[1], current_patch_img.shape[0]))
            dp_map = np.stack([
                cv2.resize(dp_map[0], (current_patch_img.shape[1], current_patch_img.shape[0])),
                cv2.resize(dp_map[1], (current_patch_img.shape[1], current_patch_img.shape[0]))
            ], axis=0)

            prob_accumulator.append(prob_map)
            dp_accumulator.append(dp_map)

        if len(prob_accumulator) == 0:
            continue  # è‹¥æ²’æœ‰ä»»ä½•å°ºå¯¸çš„é �æ¸¬æˆ�åŠŸï¼Œç•¥é��æ­¤ patch

        averaged_prob_map = np.mean(prob_accumulator, axis=0)
        averaged_dp_map = np.mean(dp_accumulator, axis=0)

        all_fused_patch_data.append((y_s, y_e, x_s, x_e, averaged_prob_map, averaged_dp_map))

    if not all_fused_patch_data:
        return np.zeros(original_shape, dtype=np.float32), \
               np.zeros((2, *original_shape), dtype=np.float32)

    fused_prob_map, fused_dp_map = stitch_masks_and_flows(all_fused_patch_data, original_shape)

    return fused_prob_map, fused_dp_map


def tta_predict_probability_and_flows(model, image, image_id, model_type="cellpose"):
    import torch
    import numpy as np
    import cv2
    from cellpose import dynamics

    augmentations = [
        lambda x: x,
        lambda x: np.flip(x, axis=1),
        lambda x: np.flip(x, axis=0),
        lambda x: np.flip(np.flip(x, axis=0), axis=1),
        lambda x: np.rot90(x, k=1),
        lambda x: np.rot90(x, k=2),
        lambda x: np.rot90(x, k=3),
    ]

    inverse_ops_prob = [
        lambda x: x,
        lambda x: np.flip(x, axis=1),
        lambda x: np.flip(x, axis=0),
        lambda x: np.flip(np.flip(x, axis=0), axis=1),
        lambda x: np.rot90(x, k=-1),
        lambda x: np.rot90(x, k=-2),
        lambda x: np.rot90(x, k=-3),
    ]

    inverse_ops_dp = [
        lambda dp: dp,
        lambda dp: np.stack([np.flip(dp[0], axis=1), -np.flip(dp[1], axis=1)], axis=0),
        lambda dp: np.stack([-np.flip(dp[0], axis=0), np.flip(dp[1], axis=0)], axis=0),
        lambda dp: np.stack([-np.flip(np.flip(dp[0], axis=0), axis=1), -np.flip(np.flip(dp[1], axis=0), axis=1)], axis=0),
        lambda dp: np.stack([np.rot90(dp[1], k=-1), -np.rot90(dp[0], k=-1)], axis=0),
        lambda dp: np.stack([-np.rot90(dp[0], k=-2), -np.rot90(dp[1], k=-2)], axis=0),
        lambda dp: np.stack([-np.rot90(dp[1], k=-3), np.rot90(dp[0], k=-3)], axis=0),
    ]

    all_prob_maps = []
    all_dp_maps = []
    H_orig, W_orig = image.shape[:2]

    for idx, aug_func in enumerate(augmentations):
        augmented_img = np.ascontiguousarray(aug_func(image))

        if model_type == "cellpose":
            _, eval_flows, _ = model.eval(augmented_img, compute_masks=False, progress=False)
            pred_dp_map_raw = eval_flows[1].astype(np.float32)
            pred_prob_map_raw = eval_flows[2].astype(np.float32)
            weight = 1.0  # Cellpose æ²’æœ‰ scoreï¼Œé �è¨­ç‚º 1

        elif model_type == "maskrcnn":
            with torch.no_grad():
                outputs = model(augmented_img)
            instances = outputs['instances']

            if len(instances) == 0:
                continue

            pred_classes_all = instances.pred_classes.cpu().numpy()
            pred_scores_all = instances.scores.cpu().numpy()
            pred_masks_all = instances.pred_masks.squeeze(1).cpu().numpy()

            keep_mask = [
                (cls < len(THRESHOLDS) and score >= THRESHOLDS[cls])
                for cls, score in zip(pred_classes_all, pred_scores_all)
            ]
            keep_mask = np.array(keep_mask)
            pred_classes = pred_classes_all[keep_mask]
            pred_scores = pred_scores_all[keep_mask]
            pred_masks = pred_masks_all[keep_mask]

            if len(pred_classes) == 0:
                print(f"[{image_id}] TTA {idx} - All instances filtered.")
                continue

            combined_mask = np.zeros(augmented_img.shape[:2], dtype=np.uint16)
            prob_map = np.zeros_like(combined_mask, dtype=np.float32)
            used = np.zeros_like(combined_mask, dtype=np.uint8)
            counter = 1

            for j in np.argsort(pred_scores)[::-1]:
                mlogits = pred_masks[j]
                class_id = pred_classes[j]

                if np.max(mlogits) <= 1.0:
                    mprob = mlogits
                else:
                    mprob = torch.sigmoid(torch.from_numpy(mlogits)).numpy()

                mbinary = (mprob > 0.5).astype(np.uint8)
                inst_mask = mbinary * (1 - used)

                min_pixels = MIN_PIXELS[class_id] if class_id < len(MIN_PIXELS) else 20
                if inst_mask.sum() < min_pixels:
                    continue

                combined_mask[inst_mask == 1] = counter
                used += inst_mask
                prob_map = np.maximum(prob_map, mprob)
                counter += 1

            if combined_mask.max() == 0:
                print(f"[{image_id}] TTA {idx} - No valid masks after filtering.")
                continue

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dp = dynamics.masks_to_flows_gpu(combined_mask.astype(int), device=device)
            pred_prob_map_raw = prob_map
            pred_dp_map_raw = dp

            # âœ… è¨ˆç®—è©²åˆ†æ”¯çš„åŠ æ¬Šå€¼ï¼šä½¿ç”¨ pred_scores å¹³å�‡
            weight = float(np.mean(pred_scores)) if len(pred_scores) > 0 else 1.0
            weight = np.clip(weight, 0.3, 2.0)

        else:
            raise ValueError("model_type must be 'cellpose' or 'maskrcnn'")

        pred_prob_map = inverse_ops_prob[idx](pred_prob_map_raw)
        pred_dp_map = inverse_ops_dp[idx](pred_dp_map_raw)

        if pred_prob_map.shape != (H_orig, W_orig):
            pred_prob_map = cv2.resize(pred_prob_map, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        if pred_dp_map.shape[1:] != (H_orig, W_orig):
            pred_dp_map = np.stack([
                cv2.resize(pred_dp_map[0], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR),
                cv2.resize(pred_dp_map[1], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            ], axis=0)

        all_prob_maps.append((pred_prob_map, weight))
        all_dp_maps.append((pred_dp_map, weight))

    if len(all_prob_maps) == 0:
        print(f"[{image_id}] â�Œ æ²’æœ‰ä»»ä½• TTA åˆ†æ”¯ç”¢ç”Ÿæœ‰æ•ˆ mask")
        return np.zeros((H_orig, W_orig), dtype=np.float32), np.zeros((H_orig, W_orig, 2), dtype=np.float32)

    # âœ… åŠ æ¬Šå¹³å�‡
    total_weight = sum(w for _, w in all_prob_maps)
    avg_prob = sum(prob * w for prob, w in all_prob_maps) / (total_weight + 1e-6)

    total_weight_dp = sum(w for _, w in all_dp_maps)
    avg_dp = sum(dp * w for dp, w in all_dp_maps) / (total_weight_dp + 1e-6)

    return avg_prob, avg_dp


# ----------------- ä¸»æ�¨è«–é‚�è¼¯é–‹å§‹ -----------------
from pathlib import Path
from tqdm import tqdm
from skimage import io
import numpy as np
import pandas as pd
import torch
from cellpose import dynamics, models

# æ¸¬è©¦è³‡æ–™ç›®éŒ„
test_dir = Path('/kaggle/input/sartorius-cell-instance-segmentation/test')
test_files = sorted([f for f in test_dir.iterdir() if f.suffix == '.png'])

submission_data = []

# Patch å�ƒæ•¸
PATCH_H, PATCH_W = 301, 302
STRIDE_H, STRIDE_W = 73, 134

# æ¨¡å�‹å•Ÿç”¨é–‹é—œ
# USE_MASKRCNN = [True, True, True, True, True]
USE_MASKRCNN = [False, False, False, False, False]
USE_MASKRCNN_PATCH = True
USE_CELLPOSE = [True, True, True, True, True]
# USE_CELLPOSE = [False, False, False, False, False]

# æ¬Šé‡�å�ƒæ•¸ï¼ˆå…�è¨±ä¹‹å¾Œèª¿æ•´ï¼‰
MASKRCNN_PER_FOLD_WEIGHT = [1.0, 1.0, 1.0, 1.0, 1.0]
CELLPOSE_PER_FOLD_WEIGHT = [10.0, 1.0, 1.0, 1.0, 1.0]
FUSION_MASKRCNN_WEIGHT = 1.0
FUSION_CELLPOSE_WEIGHT = 1.0

for img_path in tqdm(test_files, desc="ğŸš€ æ�¨è«–ä¸­"):
    img_id = img_path.stem
    img = io.imread(img_path)

    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img_rgb = img[..., :3]
    else:
        img_rgb = img

    original_shape = img_rgb.shape[:2]
    maskrcnn_probs, maskrcnn_dps, maskrcnn_total_weight = [], [], 0.0
    cellpose_probs, cellpose_dps, cellpose_total_weight = [], [], 0.0

    if USE_MASKRCNN:
        for i in range(5):
            if not USE_MASKRCNN[i]:
                continue
            cfg.MODEL.WEIGHTS = f"/kaggle/input/final-maskrcnn-5fold-models/fold{i+1}/best_model.pth"
            predictor = DefaultPredictor(cfg)
            if USE_MASKRCNN_PATCH:
                prob, dp = predict_with_maskrcnn_patching(predictor, img_rgb, img_id, (PATCH_H, PATCH_W), (STRIDE_H, STRIDE_W))
            else:
                prob, dp = tta_predict_probability_and_flows(predictor, img_rgb, img_id, model_type="maskrcnn")
            maskrcnn_probs.append(prob * MASKRCNN_PER_FOLD_WEIGHT[i])
            maskrcnn_dps.append(dp * MASKRCNN_PER_FOLD_WEIGHT[i])
            maskrcnn_total_weight += MASKRCNN_PER_FOLD_WEIGHT[i]

    if USE_CELLPOSE:
        for i in range(5):
            if not USE_CELLPOSE[i]:
                continue
            model = models.CellposeModel(gpu=True, pretrained_model=f"/kaggle/input/final-cellpose-5fold-models/fold{i+1}/best_model.pth")
            prob, dp = tta_predict_probability_and_flows(model, img_rgb, img_id, model_type="cellpose")
            cellpose_probs.append(prob * CELLPOSE_PER_FOLD_WEIGHT[i])
            cellpose_dps.append(dp * CELLPOSE_PER_FOLD_WEIGHT[i])
            cellpose_total_weight += CELLPOSE_PER_FOLD_WEIGHT[i]

    fused_prob_map = np.zeros(original_shape, dtype=np.float32)
    fused_dp_map = np.zeros((2, *original_shape), dtype=np.float32)

    if maskrcnn_total_weight > 0:
        avg_maskrcnn_prob = np.sum(maskrcnn_probs, axis=0) / maskrcnn_total_weight
        avg_maskrcnn_dp = np.sum(maskrcnn_dps, axis=0) / maskrcnn_total_weight

        # é€™è£¡æ˜¯å› ç‚º cellpose å‡ºä¾†çš„ dp éƒ½æœƒæ˜¯ 5 å€�
        avg_maskrcnn_dp *= 5.0
        
        fused_prob_map += FUSION_MASKRCNN_WEIGHT * avg_maskrcnn_prob
        fused_dp_map += FUSION_MASKRCNN_WEIGHT * avg_maskrcnn_dp

    if cellpose_total_weight > 0:
        avg_cellpose_prob = np.sum(cellpose_probs, axis=0) / cellpose_total_weight
        avg_cellpose_dp = np.sum(cellpose_dps, axis=0) / cellpose_total_weight
        fused_prob_map += FUSION_CELLPOSE_WEIGHT * avg_cellpose_prob
        fused_dp_map += FUSION_CELLPOSE_WEIGHT * avg_cellpose_dp

    total_fusion_weight = FUSION_MASKRCNN_WEIGHT * (maskrcnn_total_weight > 0) + FUSION_CELLPOSE_WEIGHT * (cellpose_total_weight > 0)
    if total_fusion_weight > 0:
        fused_prob_map /= total_fusion_weight
        fused_dp_map /= total_fusion_weight
    else:
        submission_data.append({'id': img_id, 'predicted': ''})
        continue

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_mask = dynamics.resize_and_compute_masks(
        fused_dp_map,
        fused_prob_map,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        min_size=20,
        device=device
    )

    if combined_mask.max() > 0:
        instance_ids = np.unique(combined_mask)
        instance_ids = instance_ids[instance_ids != 0]
        for inst_id in instance_ids:
            binary_mask = (combined_mask == inst_id).astype(np.uint8)
            if binary_mask.sum() >= 25:
                rle = rle_encode(binary_mask)
                submission_data.append({'id': img_id, 'predicted': rle})
    else:
        submission_data.append({'id': img_id, 'predicted': ''})


submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('submission.csv', index=False)

# âœ… æª¢æŸ¥
print(submission_df.head())
print("æ¬„ä½�å��ç¨±:", submission_df.columns)
print("æ˜¯å�¦æœ‰ null:", submission_df.isnull().sum())
print("æ˜¯å�¦æœ‰ç©ºå­—ä¸² predicted:", (submission_df['predicted'].astype(str).str.strip() == '').sum())
print("æ˜¯å�¦æœ‰ duplicated id + predicted:", submission_df.duplicated(subset=["id", "predicted"]).any())
print("id ç¸½æ•¸:", submission_df['id'].nunique(), "submission è¡Œæ•¸:", len(submission_df))
print("predicted æ¬„å�‹åˆ¥:", submission_df['predicted'].apply(type).value_counts())


import random
import numpy as np

def random_color(seed=None):
    if seed is not None:
        random.seed(seed)
    return [random.randint(0, 255) for _ in range(3)]

def mask_to_color(mask):
    """
    å°‡å¤šå€‹ instance mask å�ˆæˆ�å½©è‰²åœ–åƒ�ã€‚
    mask: numpy array, shape = (N, H, W)
    return: RGB å½©è‰² maskï¼Œshape = (H, W, 3)
    """
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]  # å–®ä¸€ mask ä¹ŸåŒ…æˆ� (1, H, W)

    h, w = mask.shape[1:]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        color = random_color(seed=i)
        color_mask[mask[i] > 0] = color
    return color_mask


visualized_ids = submission_df['id'].unique()

for img_id in visualized_ids:
    img_path = test_dir / f"{img_id}.png"
    image = cv2.imread(str(img_path))

    masks_rle = submission_df[submission_df['id'] == img_id]['predicted'].tolist()
    decoded_masks = []

    for rle in masks_rle:
        if isinstance(rle, str) and rle.strip() != '':
            decoded = rle_decode(rle, shape=(520, 704))
            decoded_masks.append(decoded)

    if len(decoded_masks) == 0:
        print(f"âš ï¸� {img_id} æ²’æœ‰é �æ¸¬åˆ° mask")
        continue

    instance_masks = np.array(decoded_masks)
    colored_mask = mask_to_color(instance_masks)

    plt.figure(figsize=(15, 15))
    plt.imshow(image[..., ::-1])
    plt.imshow(colored_mask, alpha=0.5)
    plt.axis("off")
    plt.title(f"Image {img_id} â€” Loaded from submission.csv", fontsize=20)
    plt.show()

