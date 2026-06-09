!pip install --no-index --find-links /kaggle/input/cellpose-whl --no-deps cellpose fastremap fill_voids roifile


from cellpose import models, io
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

model = models.CellposeModel(gpu=True, pretrained_model='/kaggle/input/my-cellpose-models/best_model_0.30468207597732544.pth')


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


def tta_predict_probability_and_flows(model, image, image_id):
    """
    ä½¿ç”¨æ¸¬è©¦æ™‚é–“å¢�å¼· (TTA) ä¾†é �æ¸¬æ©Ÿç�‡åœ–å’Œæµ�å�‘é‡�ã€‚
    åŒ…æ‹¬å�Ÿå§‹åœ–åƒ�ã€�æ°´å¹³ç¿»è½‰ã€�å�‚ç›´ç¿»è½‰ã€�æ°´å¹³å�‚ç›´ç¿»è½‰ï¼Œ
    ä»¥å�Š 90 åº¦ã€�180 åº¦ã€�270 åº¦æ—‹è½‰ã€‚

    Args:
        model: Cellpose æ¨¡å�‹å¯¦ä¾‹ã€‚
        image (np.ndarray): è¼¸å…¥çš„å�Ÿå§‹åœ–åƒ�ã€‚
        image_id (str): åœ–åƒ�çš„ ID (ç”¨æ–¼å‚³é��ï¼Œæœ¬å‡½æ•¸ä¸­æœªä½¿ç”¨)ã€‚

    Returns:
        tuple: (averaged_prob_map, averaged_dp_map)
               - averaged_prob_map (np.ndarray): å¹³å�‡å¾Œçš„ç´°èƒ�æ©Ÿç�‡åœ–ã€‚
               - averaged_dp_map (np.ndarray): å¹³å�‡å¾Œçš„æµ�å�‘é‡�åœ– (2, H, W)ã€‚
    """
    
    # åœ–åƒ�å¢�å¼·åˆ—è¡¨
    images = [
        image,                                # å�Ÿå§‹åœ–åƒ�
        np.flip(image, axis=1),               # æ°´å¹³ç¿»è½‰
        np.flip(image, axis=0),               # å�‚ç›´ç¿»è½‰
        np.flip(np.flip(image, axis=0), axis=1), # æ°´å¹³å�‚ç›´ç¿»è½‰
        np.rot90(image, k=1),                 # 90 åº¦é€†æ™‚é‡�æ—‹è½‰
        np.rot90(image, k=2),                 # 180 åº¦æ—‹è½‰
        np.rot90(image, k=3),                 # 270 åº¦é€†æ™‚é‡�æ—‹è½‰
    ]

    # æ©Ÿç�‡åœ–çš„é€†è®Šæ�›æ“�ä½œåˆ—è¡¨ (èˆ‡åœ–åƒ�è®Šæ�›ç›¸å��)
    inverse_ops_prob = [
        lambda x: x,
        lambda x: np.flip(x, axis=1),
        lambda x: np.flip(x, axis=0),
        lambda x: np.flip(np.flip(x, axis=0), axis=1),
        lambda x: np.rot90(x, k=-1), # 90 åº¦é€†æ™‚é‡�çš„é€† (å�³ 270 åº¦é †æ™‚é‡�)
        lambda x: np.rot90(x, k=-2), # 180 åº¦çš„é€† (å�³ 180 åº¦)
        lambda x: np.rot90(x, k=-3), # 270 åº¦é€†æ™‚é‡�çš„é€† (å�³ 90 åº¦é †æ™‚é‡�)
    ]

    # æµ�å�‘é‡�åœ– (dP) çš„é€†è®Šæ�›æ“�ä½œåˆ—è¡¨ (è™•ç�† (2, H, W) dP)
    # é€™è£¡çš„é€†è®Šæ�›éœ€è¦�å�Œæ™‚è€ƒæ…®ç¿»è½‰/æ—‹è½‰å’Œæµ�å�‘é‡�æ–¹å�‘çš„è®ŠåŒ–
    inverse_ops_dp = [
        lambda dp: dp, # 0. Original
        # 1. æ°´å¹³ç¿»è½‰: x åˆ†é‡�ç¿»è½‰ï¼Œy åˆ†é‡�è®Šè™Ÿ
        lambda dp: np.stack([np.flip(dp[0], axis=1), -np.flip(dp[1], axis=1)], axis=0),
        # 2. å�‚ç›´ç¿»è½‰: y åˆ†é‡�ç¿»è½‰ï¼Œx åˆ†é‡�è®Šè™Ÿ
        lambda dp: np.stack([-np.flip(dp[0], axis=0), np.flip(dp[1], axis=0)], axis=0),
        # 3. æ°´å¹³å�‚ç›´ç¿»è½‰: x å’Œ y åˆ†é‡�éƒ½ç¿»è½‰ï¼Œä¸”éƒ½è®Šè™Ÿ
        lambda dp: np.stack([-np.flip(np.flip(dp[0], axis=0), axis=1), -np.flip(np.flip(dp[1], axis=0), axis=1)], axis=0),
        # 4. 90 åº¦é€†æ™‚é‡�æ—‹è½‰ (k=1):
        # åœ–åƒ� (x, y) -> (-y, x)ã€‚
        # æµ�å ´ (dp_x, dp_y) åœ¨æ—‹è½‰å¾Œæœƒè®Šæˆ� (-dp_y, dp_x) ç›¸å°�æ–¼æ–°çš„åº§æ¨™è»¸ã€‚
        # é€†è®Šæ�›ï¼šå…ˆå°‡æµ�å ´åœ–é€†æ—‹è½‰å›�ä¾†ï¼Œç„¶å¾Œèª¿æ•´åˆ†é‡�ã€‚
        # ä¾‹å¦‚ï¼Œå�Ÿå§‹ dp_x åœ¨é€†æ™‚é‡�æ—‹è½‰ 90 åº¦å¾Œï¼Œæœƒæˆ�ç‚ºæ–°çš„ dp_y (ä½†åœ¨æ–°çš„åº§æ¨™ç³»ä¸‹ç‚ºè² å€¼)ã€‚
        # æ‰€ä»¥é€†å›�ä¾†æ™‚ï¼Œæ–° dp_x æ‡‰ç‚ºå�Ÿ dp_y (é€†æ—‹è½‰å¾Œ) çš„è² å€¼ï¼Œæ–° dp_y æ‡‰ç‚ºå�Ÿ dp_x (é€†æ—‹è½‰å¾Œ)ã€‚
        lambda dp: np.stack([np.rot90(dp[1], k=-1), -np.rot90(dp[0], k=-1)], axis=0),
        # 5. 180 åº¦æ—‹è½‰ (k=2):
        # åœ–åƒ� (x, y) -> (-x, -y)ã€‚
        # æµ�å ´ (dp_x, dp_y) -> (-dp_x, -dp_y) ç›¸å°�æ–¼æ–°çš„åº§æ¨™è»¸ã€‚
        # é€†è®Šæ�›ï¼šå°‡æµ�å ´åœ–é€†æ—‹è½‰å›�ä¾†ï¼Œç„¶å¾Œå°‡å…©å€‹åˆ†é‡�éƒ½è®Šè™Ÿã€‚
        lambda dp: np.stack([-np.rot90(dp[0], k=-2), -np.rot90(dp[1], k=-2)], axis=0),
        # 6. 270 åº¦é€†æ™‚é‡�æ—‹è½‰ (k=3):
        # åœ–åƒ� (x, y) -> (y, -x)ã€‚
        # æµ�å ´ (dp_x, dp_y) -> (dp_y, -dp_x) ç›¸å°�æ–¼æ–°çš„åº§æ¨™è»¸ã€‚
        # é€†è®Šæ�›ï¼šå°‡æµ�å ´åœ–é€†æ—‹è½‰å›�ä¾†ï¼Œç„¶å¾Œèª¿æ•´åˆ†é‡�ã€‚
        # é¡�ä¼¼æ–¼ 90 åº¦æ—‹è½‰ï¼Œä½†åˆ†é‡�äº’æ�›å’Œè®Šè™Ÿçš„æ–¹å¼�ä¸�å�Œã€‚
        lambda dp: np.stack([-np.rot90(dp[1], k=-3), np.rot90(dp[0], k=-3)], axis=0),
    ]

    all_prob_maps = []
    all_dp_maps = []
    
    # ç�²å�–åœ–åƒ�çš„å�Ÿå§‹ H, W (å°�æ–¼ np.rot90 æœƒæ”¹è®Š H, W çš„é †åº�ï¼Œä½†æˆ‘å€‘è™•ç�†çš„æ˜¯å�Ÿå§‹åœ–åƒ�å°ºå¯¸)
    # Cellpose çš„ eval_flows æ‡‰è©²æœƒæ ¹æ“šåœ–åƒ�çš„å¯¦éš›å°ºå¯¸ä¾†è™•ç�†
    # ä¸�é��ç‚ºäº†ä¿�éšªï¼Œå�¯ä»¥åœ¨ `inverse_ops_dp` è£¡é�¢å°� `np.rot90` çš„çµ�æ�œé€²è¡Œå°ºå¯¸èª¿æ•´
    # ä½† Cellpose å…§éƒ¨é€šå¸¸æœƒè™•ç�†è¼¸å…¥åœ–åƒ�çš„å°ºå¯¸è®ŠåŒ–ï¼Œä¸¦è¼¸å‡ºæ­£ç¢ºå°ºå¯¸çš„æµ�å ´ï¼Œ
    # æ‰€ä»¥é€™è£¡ä¸»è¦�ç¢ºä¿�é€†è®Šæ�›çš„é‚�è¼¯æ­£ç¢ºã€‚
    H, W = image.shape[:2] 

    for idx, aug_img in enumerate(images):
        _, eval_flows, _ = model.eval(aug_img, compute_masks=False)
        
        pred_dp_map_raw = eval_flows[1].astype(np.float32)
        pred_prob_map_raw = eval_flows[2].astype(np.float32)

        # åŸ·è¡Œæ©Ÿç�‡åœ–çš„é€†è®Šæ�›
        pred_prob_map = inverse_ops_prob[idx](pred_prob_map_raw)
        
        # åŸ·è¡Œæµ�å�‘é‡�åœ–çš„é€†è®Šæ�›
        pred_dp_map_transformed = inverse_ops_dp[idx](pred_dp_map_raw)
        
        # ç”±æ–¼ np.rot90 å�¯èƒ½æ”¹è®Šåœ–åƒ�çš„ H, W é †åº�ï¼ŒCellpose è¿”å›�çš„ eval_flows ä¹Ÿæœƒæ˜¯æ—‹è½‰å¾Œçš„å°ºå¯¸ã€‚
        # åœ¨é€™è£¡ï¼Œæˆ‘å€‘éœ€è¦�ç¢ºä¿�æ‰€æœ‰é€†è®Šæ�›å¾Œçš„åœ–éƒ½å›�åˆ°å�Ÿå§‹çš„ (H, W) å°ºå¯¸ã€‚
        # ä½¿ç”¨ cv2.resize æˆ– skimage.transform.resize ä¾†çµ±ä¸€å°ºå¯¸ã€‚
        # ä½† Cellpose é€šå¸¸åœ¨å…§éƒ¨è™•ç�†å¥½é€™äº›å°ºå¯¸ï¼Œå¦‚æ�œ `inverse_ops_prob` å’Œ `inverse_ops_dp`
        # å·²ç¶“åŒ…å�« `np.rot90` çš„é€†è®Šæ�›ï¼Œé‚£éº¼æœ€çµ‚çµ�æ�œçš„å°ºå¯¸æ‡‰è©²æ˜¯æ­£ç¢ºçš„ã€‚
        # é€™è£¡å�‡è¨­ `eval_flows` è¿”å›�çš„ `pred_prob_map_raw` å’Œ `pred_dp_map_raw` å·²ç¶“æ˜¯è™•ç�†é��æ—‹è½‰çš„å°ºå¯¸ã€‚
        # è€Œ `inverse_ops_prob` å’Œ `inverse_ops_dp` è² è²¬å°‡å…¶è½‰å›�å�Ÿå§‹çš„ (H,W) æ–¹å�‘å’Œåˆ†é‡�ã€‚

        # é©—è­‰å°ºå¯¸æ˜¯å�¦ä¸€è‡´ï¼Œä¸¦åœ¨ä¸�ä¸€è‡´æ™‚é€²è¡Œèª¿æ•´ (å�¯é�¸ï¼Œä½†æ�¨è–¦ç”¨æ–¼é™¤éŒ¯)
        if pred_prob_map.shape[:2] != (H, W):
            # print(f"è­¦å‘Š: æ©Ÿç�‡åœ–å°ºå¯¸ä¸�ç¬¦ï¼Œæ­£åœ¨èª¿æ•´ {pred_prob_map.shape[:2]} -> {(H, W)}")
            pred_prob_map = cv2.resize(pred_prob_map, (W, H), interpolation=cv2.INTER_LINEAR)
        if pred_dp_map_transformed.shape[1:] != (H, W):
            # print(f"è­¦å‘Š: æµ�å�‘é‡�åœ–å°ºå¯¸ä¸�ç¬¦ï¼Œæ­£åœ¨èª¿æ•´ {pred_dp_map_transformed.shape[1:]} -> {(H, W)}")
            # å°�æ–¼ dp mapï¼Œéœ€è¦�å°�å…©å€‹é€šé�“åˆ†åˆ¥ resize
            pred_dp_map_transformed_x = cv2.resize(pred_dp_map_transformed[0], (W, H), interpolation=cv2.INTER_LINEAR)
            pred_dp_map_transformed_y = cv2.resize(pred_dp_map_transformed[1], (W, H), interpolation=cv2.INTER_LINEAR)
            pred_dp_map_transformed = np.stack([pred_dp_map_transformed_x, pred_dp_map_transformed_y], axis=0)


        all_prob_maps.append(pred_prob_map)
        all_dp_maps.append(pred_dp_map_transformed)

    # å°�æ‰€æœ‰å¢�å¼·åœ–åƒ�çš„é �æ¸¬çµ�æ�œé€²è¡Œå¹³å�‡
    averaged_prob_map = np.mean(all_prob_maps, axis=0)
    averaged_dp_map = np.mean(all_dp_maps, axis=0)
    
    return averaged_prob_map, averaged_dp_map



final_ids, final_masks = [], []


from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import skimage.io as io
from cellpose import dynamics

test_dir = Path('/kaggle/input/sartorius-cell-instance-segmentation/test')
test_files = sorted([f for f in test_dir.iterdir() if f.suffix == '.png']) 

submission_data = []

for img_path in tqdm(test_files, desc="ğŸš€ æ�¨è«–ä¸­"):
    img_id = img_path.stem
    img = io.imread(img_path)
    
    # ç�²å�–åœ–åƒ�çš„å�Ÿå§‹å°ºå¯¸ï¼Œç”¨æ–¼ dynamics.resize_and_compute_masks
    original_shape = img.shape[:2] # å�‡è¨­æ˜¯ (H, W)

    # ä½¿ç”¨ TTA é �æ¸¬ä¸¦å¾—åˆ°å¹³å�‡æ©Ÿç�‡åœ–å’Œå¹³å�‡æµ�å�‘é‡�
    averaged_prob_map, averaged_dp_map = tta_predict_probability_and_flows(model, img, img_id)
    
    # ä½¿ç”¨ Cellpose çš„å‹•åŠ›å­¸æ¨¡çµ„å¾�å¹³å�‡æ©Ÿç�‡åœ–å’Œæµ�å�‘é‡�ä¸­è¨ˆç®— Mask
    # é€™è£¡çš„å�ƒæ•¸æ‡‰æ ¹æ“šä½ çš„éœ€æ±‚èª¿æ•´
    # ä¾‹å¦‚ï¼šflow_threshold, cellprob_threshold, min_size ç­‰
    combined_mask = dynamics.resize_and_compute_masks(
        averaged_dp_map, 
        averaged_prob_map,
        cellprob_threshold=0.0, # é �è¨­å€¼ï¼Œå�¯ä»¥èª¿æ•´
        flow_threshold=0.4,     # é �è¨­å€¼ï¼Œå�¯ä»¥èª¿æ•´
        min_size=20,            # é �è¨­å€¼ï¼Œå�¯ä»¥èª¿æ•´
        resize=original_shape,  # ç¢ºä¿� Mask è¿”å›�åˆ°å�Ÿå§‹åœ–åƒ�å°ºå¯¸
        device=model.device     # å‚³é��æ¨¡å�‹æ‰€åœ¨çš„è¨­å‚™
    )

    if combined_mask.max() > 0: # ç¢ºä¿�æœ‰å�µæ¸¬åˆ°å¯¦ä¾‹
        instance_ids = np.unique(combined_mask)
        instance_ids = instance_ids[instance_ids != 0] # æ�’é™¤èƒŒæ™¯ 0
        for inst_id in instance_ids:
            binary_mask = (combined_mask == inst_id).astype(np.uint8)
            rle = rle_encode(binary_mask) # ä½¿ç”¨ä½ æ��ä¾›çš„ rle_encode
            submission_data.append({'id': img_id, 'predicted': rle})
    else: # å¦‚æ�œæ²’æœ‰å�µæ¸¬åˆ°ä»»ä½•å¯¦ä¾‹ï¼Œå‰‡æ��äº¤ç©ºå­—ä¸²
        submission_data.append({'id': img_id, 'predicted': ''})


# âœ… å»ºç«‹ DataFrame ä¸¦è£œé½Šæ¼�æ�‰åœ–ç‰‡
submission_df = pd.DataFrame(submission_data)
all_test_ids = [f.stem for f in test_files]
missing_ids = set(all_test_ids) - set(submission_df['id'].unique())
for img_id in missing_ids:
    submission_df = pd.concat([submission_df, pd.DataFrame([{'id': img_id, 'predicted': ''}])], ignore_index=True)
submission_df = submission_df.sort_values(by='id').reset_index(drop=True)

# âœ… æª¢æŸ¥æ˜¯å�¦é‡�è¤‡ idï¼ˆæ‡‰è©²æ²’æœ‰ï¼‰
duplicate_ids = submission_df.duplicated(subset=['id'], keep=False)
if duplicate_ids.any():
    print("âš ï¸� å­˜åœ¨é‡�è¤‡ idï¼Œè«‹æª¢æŸ¥å�ˆä½µé‚�è¼¯ï¼�")
    
# âœ… åƒ�ç´ é‡�ç–Šæª¢æŸ¥
rle_dict = defaultdict(list)
for img_id, rle in zip(submission_df['id'], submission_df['predicted']):
    if isinstance(rle, str) and rle.strip() != '':
        rle_dict[img_id].append(rle)

overlap_count = 0
for img_id, rles in tqdm(rle_dict.items(), desc="ğŸ”� æª¢æŸ¥åƒ�ç´ é‡�ç–Š"):
    mask_sum = np.zeros((520, 704), dtype=np.uint8)
    for rle in rles:
        mask = rle_decode(rle, (520, 704))
        mask_sum += mask
    if (mask_sum > 1).any():
        print(f"âš ï¸� åœ–åƒ� {img_id} æœ‰é‡�ç–Šçš„ pixelï¼�")
        overlap_count += 1

if overlap_count == 0:
    print("âœ… æ‰€æœ‰åœ–åƒ�éƒ½æ²’æœ‰é‡�ç–Š pixel")
else:
    print(f"âš ï¸� å…± {overlap_count} å¼µåœ–åƒ�æœ‰é‡�ç–Šå•�é¡Œï¼Œå»ºè­°æª¢æŸ¥å�ˆä½µç­–ç•¥")

# âœ… è¼¸å‡º CSV
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


import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

# âœ… è®€ submission.csv
submission_df = pd.read_csv("submission.csv")

# âœ… æ¸¬è©¦è³‡æ–™å¤¾ä½�ç½®
test_dir = Path('/kaggle/input/sartorius-cell-instance-segmentation/test')

# âœ… ä½ æƒ³è¦�å�¯è¦–åŒ–å¹¾å¼µåœ–
N = 3
visualized_ids = submission_df['id'].unique()[:N]

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


# âœ… çµ‚ç«¯è¼¸å‡ºè³‡è¨Š
print(submission_df.head(20))
print("æ¬„ä½�å��ç¨±:", submission_df.columns)
print("æ˜¯å�¦æœ‰é‡�è¤‡ id:", submission_df['id'].duplicated().any())
print("æ˜¯å�¦æœ‰ null:", submission_df.isnull().sum())
print("æ˜¯å�¦æœ‰ç©ºå­—ä¸²ä»¥å¤–çš„ç©ºå€¼:", (submission_df['predicted'].astype(str).str.strip() == '').sum())
print("id ç¸½æ•¸:", submission_df['id'].nunique(), "submission è¡Œæ•¸:", len(submission_df))
print("predicted æ¬„å�‹åˆ¥:", submission_df['predicted'].apply(type).value_counts())
print("âœ… Submission file 'submission.csv' created successfully!")

