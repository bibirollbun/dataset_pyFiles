!pip install --no-index --find-links /kaggle/input/cellpose-linux-whl-v2/wheels --no-deps cellpose fastremap fill_voids roifile


from cellpose import models, io
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

# -------------------------------------------------------------------
# ã€�æ”¹å‹•èµ·é»�ã€‘ï¼šå¾�ã€Œå–®ä¸€ modelã€�â†’ã€Œå¤šå€‹ checkpoint Ensembleã€�
# -------------------------------------------------------------------

# 1) æŠŠä½ è¦�å�š Ensemble çš„ checkpoint è·¯å¾‘éƒ½æ”¾é€²åˆ—è¡¨è£¡
ensemble_weights = [
    #'/kaggle/input/patch4_base_bs2/pytorch/default/1/patch_epoch100_bs2_epoch_0031',
    # '/kaggle/input/cosbest/pytorch/default/1/patch_epoch100_bs2__epoch_0031',
    #'/kaggle/input/full-res/fullres_epoch_0035',
    # "/kaggle/input/cellpose-fold-weight/fold0_epoch_0032_mAP_0.3129.pth",
    "/kaggle/input/cellpose-fold-weight/fold1_epoch_0038_mAP_0.2929.pth",
    # "/kaggle/input/cellpose-fold-weight/fold2_epoch_0058_mAP_0.3104.pth",
    "/kaggle/input/cellpose-fold-weight/fold3_epoch_0067_mAP_0.3076.pth",
    "/kaggle/input/cellpose-fold-weight/fold4_epoch_0047_mAP_0.2945.pth",
    # å¦‚æ�œé‚„æœ‰æ›´å¤šï¼Œç¹¼çºŒåœ¨é€™è£¡åŠ ä¸Šè·¯å¾‘
]

# 2) é‡�å°�æ¯�å€‹ checkpoint path å»ºç«‹ä¸€å€‹ CellposeModel å¯¦ä¾‹ï¼Œæ”¾åˆ° models_list
models_list = [
    models.CellposeModel(
        gpu=True,
        pretrained_model=weight_path
    )
    for weight_path in ensemble_weights
]

# 3) ï¼ˆé�¸æ“‡æ€§ï¼‰ç¢ºèª�è¼‰å…¥å®Œç•¢
print(f"Loaded {len(models_list)} models for ensemble.")


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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


def compute_iou(mask1, mask2):
    """
    è¨ˆç®—å…©å€‹äºŒå€¼ mask çš„ IoUï¼ˆIntersection over Unionï¼‰ã€‚
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


# # Cell 4ï¼ˆæ•´å€‹ cell å…§å®¹æ�›æˆ�ä¸‹é�¢é€™æ®µ Ensemble ç‰ˆï¼‰
# import numpy as np
# import cv2

# def tta_ensemble_predict(models_list, image):
#     """
#     Ensemble TTAï¼šé‡�å°�å¤šå€‹ CellposeModel (models_list) + å…©ç¨®å¢�å¼·ï¼ˆå�Ÿåœ–ã€�æ°´å¹³ç¿»è½‰ï¼‰ï¼Œ
#     å…ˆå°�å�Œä¸€æ¨¡å�‹çš„å…©å€‹å¢�å¼·å¹³å�‡ï¼Œå†�å°�æ‰€æœ‰æ¨¡å�‹å¹³å�‡ã€‚
#     å�ƒæ•¸ï¼š
#       - models_list: list of CellposeModel instances
#       - image: (H, W, 3) numpy array
#     å›�å‚³ï¼š
#       averaged_prob_map_ensemble: (H, W) float32
#       averaged_dp_map_ensemble:   (2, H, W) float32
#     """
#     H, W = image.shape[:2]

#     # å®šç¾©å…©ç¨®å¢�å¼·ï¼šå�Ÿåœ–ã€�æ°´å¹³ç¿»è½‰ï¼Œä»¥å�Šå°�æ‡‰çš„é€†è®Šæ�›
#     aug_images = [
#         image,
#         np.flip(image, axis=1)
#     ]
#     inverse_ops_prob = [
#         lambda x: x,
#         lambda x: np.flip(x, axis=1),
#     ]
#     inverse_ops_dp = [
#         lambda dp: dp,
#         lambda dp: np.stack([
#             np.flip(dp[0], axis=1),
#             -np.flip(dp[1], axis=1)
#         ], axis=0),
#     ]

#     # all_prob_maps[i][j] è¡¨ç¤ºç¬¬ i å€‹æ¨¡å�‹ ç¬¬ j ç¨®å¢�å¼· çš„ prob_map
#     # all_dp_maps[i][j]   è¡¨ç¤ºç¬¬ i å€‹æ¨¡å�‹ ç¬¬ j ç¨®å¢�å¼· çš„ dp_map
#     n_models = len(models_list)
#     all_prob_maps = []
#     all_dp_maps   = []

#     for m in models_list:
#         prob_list_per_model = []
#         dp_list_per_model   = []

#         for aug_idx, aug_img in enumerate(aug_images):
#             # ä¸�å�šå®Œæ•´çš„ mask è§£ç¢¼ï¼Œå�ªæ‹¿åˆ° eval_flows
#             _, eval_flows, _ = m.eval(aug_img, compute_masks=False)
#             dp_raw   = eval_flows[1].astype(np.float32)  # (2, h', w')
#             prob_raw = eval_flows[2].astype(np.float32)  # (h', w')

#             # å…ˆå�šé€†è®Šæ�›ï¼ˆflip å›�ä¾†ã€�resize å›� (H,W)ï¼‰
#             prob_aug = inverse_ops_prob[aug_idx](prob_raw)
#             if prob_aug.shape[:2] != (H, W):
#                 prob_aug = cv2.resize(prob_aug, (W, H), interpolation=cv2.INTER_LINEAR)

#             dp_aug = inverse_ops_dp[aug_idx](dp_raw)
#             if dp_aug.shape[1:] != (H, W):
#                 dp_x = cv2.resize(dp_aug[0], (W, H), interpolation=cv2.INTER_LINEAR)
#                 dp_y = cv2.resize(dp_aug[1], (W, H), interpolation=cv2.INTER_LINEAR)
#                 dp_aug = np.stack([dp_x, dp_y], axis=0)

#             prob_list_per_model.append(prob_aug)  # shape=(H,W)
#             dp_list_per_model.append(dp_aug)      # shape=(2,H,W)

#         # å°�å�Œä¸€æ¨¡å�‹çš„ã€Œå�Ÿåœ–ï¼‹æ°´å¹³ç¿»è½‰ã€�å…ˆå�šå¹³å�‡
#         prob_avg_i = np.mean(np.stack(prob_list_per_model, axis=0), axis=0)  # (H,W)
#         dp_avg_i   = np.mean(np.stack(dp_list_per_model, axis=0), axis=0)    # (2,H,W)

#         all_prob_maps.append(prob_avg_i)  # (H,W)
#         all_dp_maps.append(dp_avg_i)      # (2,H,W)

#     # å†�å°�æ‰€æœ‰æ¨¡å�‹çš„çµ�æ�œå�šå¹³å�‡
#     averaged_prob_map_ensemble = np.mean(np.stack(all_prob_maps, axis=0), axis=0)  # (H,W)
#     averaged_dp_map_ensemble   = np.mean(np.stack(all_dp_maps, axis=0), axis=0)    # (2,H,W)

#     return averaged_prob_map_ensemble, averaged_dp_map_ensemble



final_ids, final_masks = [], []


import pycocotools.mask
import torch
import numpy as np
import cv2

def remove_overlap_naive(masks):
    """
    Greedy pixelâ€�level removal for overlapping masksï¼š
      å…ˆæŠŠæ¯�å€‹ mask ç·¨æˆ� RLEã€�è¨ˆç®— IoUï¼Œå†�å°�å�Œä¸€ cluster çš„ masks ä»¥è²ªå©ªæ–¹å¼�ç§»é™¤é‡�ç–Šéƒ¨åˆ†ã€‚
    Args:
        masks (np.ndarray): shape = (N, H, W)ï¼ŒäºŒå€¼ maskï¼ˆuint8 æˆ– boolï¼‰ã€‚
    Returns:
        np.ndarray: ç¶“é��è²ªå©ªé‡�ç–Šç§»é™¤å¾Œçš„ masksï¼Œshape = (M, H, W)ï¼ŒM â‰¤ Nã€‚
    """
    if masks.size == 0:
        return masks

    rles = [pycocotools.mask.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
    ious = pycocotools.mask.iou(rles, rles, [0] * len(rles))
    np.fill_diagonal(ious, 0)

    toproc = np.where(ious.sum(axis=0) > 0)[0]
    if len(toproc) == 0:
        return masks

    mt = torch.from_numpy(masks.astype(np.uint8)).cuda()
    prev = mt[toproc[0]].clone()
    for idx, i in enumerate(toproc[1:], start=1):
        prev = torch.max(prev, mt[toproc[idx - 1]])
        mt[i] *= (~prev)
    return mt.cpu().numpy()

def instmap_to_masks_boxes(inst_map):
    """
    Convert 2D instance label map â†’ äºŒå€¼ masks + boxes (x0,y0,x1,y1,area, dummy_score=0)  
    Args:
        inst_map (np.ndarray): shape = (H, W)ï¼Œç¨ å¯† instance mapï¼Œ0=èƒŒæ™¯ï¼Œ1,2,3...=å�„å¯¦ä¾‹
    Returns:
        masks (np.ndarray)ï¼šshape = (K, H, W)ï¼Œè‹¥ç„¡ä»»ä½•å¯¦ä¾‹å›�å‚³ (None, None)
        boxes (np.ndarray)ï¼šshape = (K, 6)ï¼Œæ¯�è¡Œ [x0, y0, x1, y1, area, 0.0]
    """
    masks, boxes = [], []
    for lab in np.unique(inst_map)[1:]:
        m = (inst_map == lab).astype(np.uint8)
        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        area = int(m.sum())
        masks.append(m)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max(), area, 0.0])
    if not masks:
        return None, None
    return np.stack(masks, axis=0), np.array(boxes, dtype=float)

def weighted_mask_fusion_nmw(masks, boxes, scores, iou_thr=0.15, score_coef=0.8):
    """
    Weighted NMWï¼š
      1. è¨ˆç®—æ‰€æœ‰ masks ä¹‹é–“çš„ IoU â†’ ious (NÃ—N)
      2. æŠŠ IoU > iou_thr çš„é‚£äº› masks è¦–ç‚ºå�Œ cluster
      3. cluster å…§æ¯�å¼µ mask çš„åˆ†æ•¸ä¹˜ä¸Š score_coefï¼Œå†� normalize â†’ w_k
      4. cluster å…§æ‰€æœ‰ mask å�š pixelâ€�wise weighted sum â†’ soft_map
      5. soft_map >= 0.5 threshold â†’ fused maskï¼›è‹¥å…¨éƒ¨ < 0.5ï¼Œå‰‡å�– cluster ä¸­åˆ†æ•¸æœ€é«˜é‚£å¼µ mask
    Args:
        masks (np.ndarray): shape=(N, H, W)ï¼ŒäºŒå€¼ mask list
        boxes (np.ndarray): shape=(N, 6)ï¼Œ[x0,y0,x1,y1,area, dummy_score]
        scores (np.ndarray): shape=(N,) å°�æ‡‰æ¯�å¼µ mask çš„ã€Œè¼”åŠ©åˆ†æ•¸ã€�
        iou_thr (float): cluster åˆ†ç¾¤ IoU é–¾å€¼
        score_coef (float): åˆ†æ•¸å¼±åŒ–ä¿‚æ•¸ (< 1)
    Returns:
        fused_masks (np.ndarray): shape=(M, H, W)ï¼Œè��å�ˆå¾ŒäºŒå€¼ masks
        fused_boxes (np.ndarray): shape=(M, 6)ï¼Œè��å�ˆå¾Œ bounding boxes
    """
    N = masks.shape[0]
    if N == 0:
        return np.zeros((0, masks.shape[1], masks.shape[2]), dtype=np.uint8), np.zeros((0, 6), dtype=float)

    rles = [pycocotools.mask.encode(np.asfortranarray(m)) for m in masks]
    ious = pycocotools.mask.iou(rles, rles, [0] * N)
    used = set()
    fused_masks, fused_boxes = [], []

    for i in range(N):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, N):
            if ious[i, j] > iou_thr:
                group.append(j)
        used.update(group)

        sub_masks = masks[group]
        sub_boxes = boxes[group]
        sub_scores = scores[group].astype(float) * score_coef

        if len(group) == 1:
            fused_masks.append(sub_masks[0])
            fused_boxes.append(sub_boxes[0])
            continue

        weights = sub_scores / sub_scores.sum()
        soft_map = np.tensordot(weights, sub_masks, axes=(0, 0))
        bin_mask = (soft_map >= 0.5).astype(np.uint8)

        ys, xs = np.where(bin_mask)
        if ys.size == 0:
            best_idx = group[np.argmax(sub_scores)]
            bin_mask = masks[best_idx]
            ys, xs = np.where(bin_mask)
            fused_masks.append(bin_mask)
            fused_boxes.append([xs.min(), ys.min(), xs.max(), ys.max(), int(bin_mask.sum()), 0.0])
        else:
            fused_masks.append(bin_mask)
            fused_boxes.append([xs.min(), ys.min(), xs.max(), ys.max(), int(bin_mask.sum()), 0.0])

    if not fused_masks:
        return np.zeros((0, masks.shape[1], masks.shape[2]), dtype=np.uint8), np.zeros((0, 6), dtype=float)
    return np.stack(fused_masks, axis=0), np.array(fused_boxes, dtype=float)



import numpy as np
import cv2

def tta_predict_probability_and_flows(model, image, image_id):
    """
    ä½¿ç”¨ TTAï¼ˆå�Ÿåœ– + æ°´å¹³ç¿»è½‰ï¼‰ä¾†è¨ˆç®—è©²æ¨¡å�‹å°�å–®å¼µå½±åƒ�çš„ averaged_prob_map èˆ‡ averaged_dp_mapã€‚
    å�ƒæ•¸ï¼š
      - model: CellposeModel instance
      - image: (H, W, 3) numpy array
      - image_id: string (å�¯å¿½ç•¥)
    å›�å‚³ï¼š
      - averaged_prob_map: (H, W) numpy float32
      - averaged_dp_map:   (2, H, W) numpy float32
    """
    H, W = image.shape[:2]

    # æˆ‘å€‘å�ªå�šã€Œå�Ÿåœ–ã€�å’Œã€Œæ°´å¹³ç¿»è½‰ã€�å…©ç¨®å¢�å¼·
    aug_images = [
        image,
        np.flip(image, axis=1)
    ]
    # å°�æ‡‰çš„é€†è®Šæ�›ï¼šæ°´å¹³ç¿»è½‰å¾Œè¦�å†�ç¿»å›�ä¾†
    inverse_ops_prob = [
        lambda x: x,
        lambda x: np.flip(x, axis=1),
    ]
    inverse_ops_dp = [
        lambda dp: dp,
        lambda dp: np.stack([
            np.flip(dp[0], axis=1),
            -np.flip(dp[1], axis=1)
        ], axis=0),
    ]

    all_prob_maps = []
    all_dp_maps   = []

    for aug_idx, aug_img in enumerate(aug_images):
        # å�ªå�– eval_flowsï¼Œä¸�å�šå®Œæ•´çš„ mask è§£ç¢¼
        _, eval_flows, _ = model.eval(aug_img, compute_masks=False)
        pred_dp_map_raw = eval_flows[1].astype(np.float32)  # shape=(2, h', w')
        pred_prob_map_raw = eval_flows[2].astype(np.float32)  # shape=(h', w')

        # å…ˆå�šé€†è®Šæ�›ï¼ˆflip å›�å�»ï¼‰
        prob_aug = inverse_ops_prob[aug_idx](pred_prob_map_raw)
        dp_aug   = inverse_ops_dp[aug_idx](pred_dp_map_raw)

        # è‹¥è¼¸å‡ºå°ºå¯¸è·Ÿå�Ÿåœ–ä¸�å�Œï¼Œå°± resize å›� (H, W)
        if prob_aug.shape[:2] != (H, W):
            prob_aug = cv2.resize(prob_aug, (W, H), interpolation=cv2.INTER_LINEAR)

        if dp_aug.shape[1:] != (H, W):
            dp_x = cv2.resize(dp_aug[0], (W, H), interpolation=cv2.INTER_LINEAR)
            dp_y = cv2.resize(dp_aug[1], (W, H), interpolation=cv2.INTER_LINEAR)
            dp_aug = np.stack([dp_x, dp_y], axis=0)

        all_prob_maps.append(prob_aug)
        all_dp_maps.append(dp_aug)

    # åœ¨å…©å¼µå¢�å¼·çµ�æ�œä¸Šå�šå¹³å�‡
    averaged_prob_map = np.mean(np.stack(all_prob_maps, axis=0), axis=0)  # shape=(H, W)
    averaged_dp_map   = np.mean(np.stack(all_dp_maps, axis=0), axis=0)    # shape=(2, H, W)

    return averaged_prob_map, averaged_dp_map



# =============================
# Cell 7ï¼šæ�¨è«– + Ensemble (å¤šæ¨¡å�‹ TTA) + NMW å¾Œè™•ç�†
# =============================
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import skimage.io as io
from cellpose import dynamics, models, core, io as cpio, metrics
import cv2

# ä»¥ä¸‹å¹¾å€‹å�ƒæ•¸éœ€èˆ‡é©—è­‰éš�æ®µä¿�æŒ�ä¸€è‡´
decode_flow_th = 0.4    # Cellpose è§£æµ�å�‘æ™‚ä½¿ç”¨çš„ threshold
iou_thr_nmw    = 0.45   # NMW è�šé¡�æ™‚çš„ IoU é–¾å€¼
score_coef     = 0.8    # NMW åˆ†æ•¸å¼±åŒ–ä¿‚æ•¸ (<1)
min_size       = 75     # æœ€çµ‚ç¯©é™¤é�¢ç©� < 75 pxÂ² çš„ mask
corrupt        = True   # æ˜¯å�¦å°� astrocyte å�š convexâ€�hull è£œæ´�
cell_type      = 0      # 1 è¡¨ç¤º astrocyte æ‰�å�šè£œæ´�ï¼›0 è¡¨ç¤ºä¸�å�š

test_dir = Path('/kaggle/input/sartorius-cell-instance-segmentation/test')
test_files = sorted([f for f in test_dir.iterdir() if f.suffix == '.png'])

submission_data = []
for img_path in tqdm(test_files, desc="ğŸš€ æ�¨è«–ä¸­"):
    img_id = img_path.stem
    img = io.imread(str(img_path))  # è®€å�–å½±åƒ�ï¼Œæ ¼å¼�ç‚º (H, W, 3)
    H, W = img.shape[:2]

    # ===== 1. é€�æ¨¡å�‹å�š TTA â†’ dynamics è§£ mask â†’ æ‹†å‡º m_bin, bxs, ä¸¦è¨ˆç®—ã€Œå¹³å�‡æ©Ÿç�‡ã€�å�šç‚º score =====
    all_masks, all_boxes, all_scores = [], [], []
    for mdl in models_list:
        # 1.1) æœ¬æ¨¡å�‹å�š TTA (å�Ÿåœ– + æ°´å¹³ç¿»è½‰)ï¼Œå�–å¾— avg_prob_map, avg_dp_map
        avg_prob_map, avg_dp_map = tta_predict_probability_and_flows(mdl, img, "")

        
        # 1.2) ç”¨ dynamics è§£å‡ºç¨ å¯†çš„ instance map
        combined_mask = dynamics.resize_and_compute_masks(
            avg_dp_map,
            avg_prob_map,
            cellprob_threshold=0.0,
            flow_threshold=decode_flow_th,
            min_size=20,    # åˆ�æ­¥ç¯©é™¤æ¥µå°�ç‰©ä»¶
            resize=(H, W),
            device=mdl.device
        )
        # å¦‚æ�œæ­¤æ¨¡å�‹å®Œå…¨æ²’å�µæ¸¬åˆ°ä»»ä½•å¯¦ä¾‹ï¼Œå°±è·³é��
        if combined_mask.max() == 0:
            continue

        # 1.3) æŠŠ dense instance map æ‹†æˆ�å¤šå¼µäºŒå€¼ mask èˆ‡å°�æ‡‰çš„ boxes
        m_bin, bxs = instmap_to_masks_boxes(combined_mask)
        if m_bin is None:
            continue

        # 1.4) è¨ˆç®—æ¯�å€‹å¯¦ä¾‹åœ¨ avg_prob_map ä¸Šçš„å¹³å�‡æ©Ÿç�‡ï¼Œå�šç‚ºè©²å¯¦ä¾‹çš„ score
        n_inst = m_bin.shape[0]
        inst_scores = []
        for k in range(n_inst):
            mask_k = m_bin[k].astype(bool)
            if mask_k.sum() == 0:
                inst_scores.append(0.0)
            else:
                inst_scores.append(float(avg_prob_map[mask_k].mean()))

        # 1.5) æ”¶é›†æœ¬æ¨¡å�‹æ‹†å‡ºçš„æ‰€æœ‰ instance (mask, box, score)
        all_masks.append(m_bin)                   # shape = (k_i, H, W)
        all_boxes.append(bxs)                     # shape = (k_i, 6)
        all_scores.append(np.array(inst_scores))  # shape = (k_i,)

    # ===== å¦‚æ�œæ‰€æœ‰æ¨¡å�‹éƒ½æ²’å�µæ¸¬åˆ°ä»»ä½•å¯¦ä¾‹ï¼Œå‰‡è¼¸å‡ºç©ºç™½ =====
    if not all_masks:
        submission_data.append({'id': img_id, 'predicted': ''})
        continue

    # ===== 2. å�ˆä½µæ‰€æœ‰æ¨¡å�‹çš„ masks / boxes / scores =====
    masks_concat  = np.concatenate(all_masks, axis=0)   # shape=(N_total, H, W)
    boxes_concat  = np.concatenate(all_boxes, axis=0)   # shape=(N_total, 6)
    scores_concat = np.concatenate(all_scores, axis=0)  # shape=(N_total,)

    # ===== 3. ä¸€æ¬¡æ€§å‘¼å�« Weighted Mask Fusion (NMW) é€²è¡ŒåŠ æ¬Šè��å�ˆ =====
    fused_masks, fused_boxes = weighted_mask_fusion_nmw(
        masks_concat, boxes_concat, scores_concat,
        iou_thr=iou_thr_nmw,
        score_coef=score_coef
    )
    # fused_masks shape=(M, H, W)

    # ===== 4. å°�é�¢ç©�é��æ¿¾ =====
    if fused_masks.shape[0] > 0:
        areas = fused_masks.sum(axis=(1, 2))
        keep = np.where(areas > min_size)[0]
        fused_masks = fused_masks[keep]
        fused_boxes = fused_boxes[keep]

    # ===== 5. astrocyte convexâ€�hull è£œæ´�ï¼ˆè‹¥å•Ÿç”¨ï¼‰ =====
    if corrupt and (cell_type == 1) and (fused_masks.shape[0] > 0):
        corrected = []
        for m in fused_masks:
            conts, _ = cv2.findContours(
                m.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cvx_mask = np.zeros_like(m, dtype=np.uint8)
            for cnt in conts:
                hull = cv2.convexHull(cnt)
                cvx_mask = cv2.fillConvexPoly(cvx_mask, hull, 1)
            corrected.append(cvx_mask)
        fused_masks = np.stack(corrected, axis=0)

    # ===== 6. æœ€å¾Œä¸€æ¬¡ Greedy Overlap Removal =====
    if fused_masks.shape[0] > 0:
        fused_masks = remove_overlap_naive(fused_masks)

    # ===== 7. é‡�å»ºæœ€çµ‚ç¨ å¯† instance mapï¼Œä¸¦å�š RLE ç·¨ç¢¼è¼¸å‡º =====
    canvas = np.zeros((H, W), dtype=np.int32)
    for m in fused_masks:
        canvas[m.astype(bool)] = canvas.max() + 1

    if canvas.max() > 0:
        instance_ids = np.unique(canvas)
        instance_ids = instance_ids[instance_ids != 0]
        for inst_id in instance_ids:
            binary_mask = (canvas == inst_id).astype(np.uint8)
            rle = rle_encode(binary_mask)
            submission_data.append({'id': img_id, 'predicted': rle})
    else:
        submission_data.append({'id': img_id, 'predicted': ''})

# ===== æœ€å¾Œæ•´ç�† submission_dfã€�æª¢æŸ¥é‡�è¤‡èˆ‡é‡�ç–Šï¼Œä¸¦è¼¸å‡º submission.csv =====
submission_df = pd.DataFrame(submission_data, columns=['id', 'predicted'])
all_test_ids = [f.stem for f in test_files]
missing_ids = set(all_test_ids) - set(submission_df['id'].unique())
for img_id in missing_ids:
    submission_df = pd.concat(
        [submission_df, pd.DataFrame([{'id': img_id, 'predicted': ''}])],
        ignore_index=True
    )
submission_df = submission_df.sort_values(by='id').reset_index(drop=True)

duplicate_ids = submission_df.duplicated(subset=['id'], keep=False)
if duplicate_ids.any():
    print("âš ï¸� å­˜åœ¨é‡�è¤‡ idï¼Œè«‹æª¢æŸ¥å�ˆä½µé‚�è¼¯ï¼�")

rle_dict = defaultdict(list)
for img_id, rle_text in zip(submission_df['id'], submission_df['predicted']):
    if isinstance(rle_text, str) and rle_text.strip() != '':
        rle_dict[img_id].append(rle_text)

overlap_count = 0
for img_id, rles in tqdm(rle_dict.items(), desc="ğŸ”� æª¢æŸ¥åƒ�ç´ é‡�ç–Š"):
    mask_sum = np.zeros((520, 704), dtype=np.uint8)
    for rle_text in rles:
        mask = rle_decode(rle_text, (520, 704))
        mask_sum += mask
    if (mask_sum > 1).any():
        print(f"âš ï¸� åœ–åƒ� {img_id} æœ‰é‡�ç–Šçš„ pixelï¼�")
        overlap_count += 1

if overlap_count == 0:
    print("âœ… æ‰€æœ‰åœ–åƒ�éƒ½æ²’æœ‰é‡�ç–Š pixel")
else:
    print(f"âš ï¸� å…± {overlap_count} å¼µåœ–åƒ�æœ‰é‡�ç–Šå•�é¡Œï¼Œå»ºè­°æª¢æŸ¥å�ˆä½µç­–ç•¥")

submission_df.to_csv('submission.csv', index=False)



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


# âœ… çµ‚ç«¯è¼¸å‡ºè³‡è¨Š
print(submission_df.head(20))
print("æ¬„ä½�å��ç¨±:", submission_df.columns)
print("æ˜¯å�¦æœ‰é‡�è¤‡ id:", submission_df['id'].duplicated().any())
print("æ˜¯å�¦æœ‰ null:", submission_df.isnull().sum())
print("æ˜¯å�¦æœ‰ç©ºå­—ä¸²ä»¥å¤–çš„ç©ºå€¼:", (submission_df['predicted'].astype(str).str.strip() == '').sum())
print("id ç¸½æ•¸:", submission_df['id'].nunique(), "submission è¡Œæ•¸:", len(submission_df))
print("predicted æ¬„å�‹åˆ¥:", submission_df['predicted'].apply(type).value_counts())
print("âœ… Submission file 'submission.csv' created successfully!")

