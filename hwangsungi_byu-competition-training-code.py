import os, cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler


try:
    import monai
    print("âœ… MONAI is already installed.")
except ImportError:
    print("ğŸ“¦ Installing MONAI from local .whl file...")
    !pip install /kaggle/input/monai-1-3/monai-1.3.0-202310121228-py3-none-any.whl

DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
test_DIR = os.path.join(DATA_DIR,"test")


# ì½”ë“œ ì‹¤í–‰ ìƒ�íƒœê°€ ì´ˆê¸°í™”ë�˜ì—ˆê¸° ë•Œë¬¸ì—�, ë‹¤ì‹œ í•„ìš”í•œ ë�¼ì�´ë¸ŒëŸ¬ë¦¬ì™€ ëª¨ë�¸ ì •ì�˜ ì½”ë“œë¥¼ ì�¬ì‹¤í–‰í•©ë‹ˆë‹¤.

from monai.networks.nets import UNet
import torch

# 3D U-Net ëª¨ë�¸ ì •ì�˜ (MONAI)
def get_unet_model(spatial_dims=3, in_channels=1, out_channels=1, pretrained_path=None):
    """
    MONAI 3D U-Net ì •ì�˜ ë°� ì‚¬ì „í•™ìŠµ weight ë¡œë”©

    Args:
        spatial_dims: 3D ë³¼ë¥¨ì�´ë¯€ë¡œ ê¸°ë³¸ê°’ 3ì�Œ.
        in_channels: ì�…ë ¥ ì±„ë„� ìˆ˜ (í�‘ë°±ì�´ë©´ 1)
        out_channels: ì¶œë ¥ ì±„ë„� ìˆ˜ (binary segmentationì�´ë©´ 1)
        pretrained_path: ì‚¬ì „í•™ìŠµë�œ ëª¨ë�¸ ê²½ë¡œ (.pth ë˜�ëŠ” .pt)

    Returns:
        PyTorch ëª¨ë�¸
    """
    from monai.networks.nets import UNet
    
    model = UNet(
        spatial_dims=spatial_dims, #3ì�´ë©´ 3D-Unet
        in_channels=in_channels, #í�‘ë°± ì�´ë¯¸ì§€ë©´ 1, RGBë©´ 3
        out_channels=out_channels, #ì˜ˆì¸¡í•  í�´ë�˜ìŠ¤ ê°œìˆ˜(í�¸ëª¨ëª¨í„° í•œ ì¢…ë¥˜ì�´ë¯€ë¡œ 1(í�¸ëª¨ëª¨í„°ì�˜ ê°œìˆ˜ê°€ ì•„ë‹˜, ì˜ˆì‹œë¡œ í�¸ëª¨ëª¨í„°, ì„¸í�¬í•µ ì�´ë ‡ê²Œ 2ê°œì�˜ ì¢…ë¥˜ ì˜ˆì¸¡ì�´ë©´ 2))
        channels=(16, 32, 64, 128, 256),
        strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),  # 3D tuple
        num_res_units=2,
        norm='instance' #batch->ì�¼ë°˜ì �, instance->ë°°ì¹˜ê°’ì�´ ì�‘ì�„ë•Œ, group->batchê°€ 1
    )


    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("âœ… ì‚¬ì „í•™ìŠµë�œ weight ë¡œë”© ì™„ë£Œ:", pretrained_path)

    return model

# í…ŒìŠ¤íŠ¸ìš© ëª¨ë�¸ ì�¸ìŠ¤í„´ìŠ¤ ìƒ�ì„±
unet3d = get_unet_model()



import numpy as np
import torch
import torch.nn.functional as F

def sliding_window_predict(volume, model, patch_size=(64, 256, 256), stride=(32, 128, 128), device='cuda'):
    """
    Patch ê¸°ë°˜ìœ¼ë¡œ 3D ë³¼ë¥¨ì—� ëŒ€í•´ ì˜ˆì¸¡í•˜ëŠ” í•¨ìˆ˜.
    í•œ íŒŒì�¼ì�„ ìŠ¬ë�¼ì�´ì‹±í•˜ëŠ” ê²ƒì�´ ì•„ë‹Œ, í�´ë�” ì „ì²´ ì¦‰, 3Dì�´ë¯¸ì§€ë¥¼ ì§�ìœ¡ë©´ì²´ ëª¨ì–‘ìœ¼ë¡œ ì�˜ë�¼ì„œ ë¶„ì„�í•œë‹¤ê³  ë³´ë©´ë�œë‹¤.
    
    Args:
        volume (np.ndarray): ì�…ë ¥ 3D ë³¼ë¥¨ (Z, H, W)
        model (torch.nn.Module): í•™ìŠµë�œ 3D U-Net ëª¨ë�¸
        patch_size (tuple): patch í�¬ê¸° (z, y, x)
        stride (tuple): ìŠ¬ë�¼ì�´ë”© ìœˆë�„ìš° stride (z, y, x), ê²½ê³„ë©´ ë¶€ë¶„ ì •ë³´ì†�ì‹¤ì�„ ë§‰ê¸°ìœ„í•´ path_sizeë³´ë‹¤ ì�‘ê²Œ ì„¤ì •.
        device (str): 'cuda' or 'cpu'
        
    Returns:
        full_pred (np.ndarray): ì˜ˆì¸¡ë�œ ë§ˆìŠ¤í�¬ (Z, H, W)
    """
    model.eval()
    model.to(device)

    z_max, y_max, x_max = volume.shape
    patch_z, patch_y, patch_x = patch_size
    stride_z, stride_y, stride_x = stride

    # predictionê³¼ countingì�„ ìœ„í•œ ê³µê°„
    output_pred = np.zeros(volume.shape, dtype=np.float32)
    output_count = np.zeros(volume.shape, dtype=np.float32)

    with torch.no_grad():
        for z in range(0, z_max - patch_z + 1, stride_z):
            for y in range(0, y_max - patch_y + 1, stride_y):
                for x in range(0, x_max - patch_x + 1, stride_x):
                    patch = volume[z:z + patch_z, y:y + patch_y, x:x + patch_x]
                    patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, Z, H, W)
                    
                    pred = model(patch_tensor)  # (1, 1, Z, H, W)
                    pred = torch.sigmoid(pred).squeeze().cpu().numpy()  # (Z, H, W)

                    output_pred[z:z + patch_z, y:y + patch_y, x:x + patch_x] += pred
                    output_count[z:z + patch_z, y:y + patch_y, x:x + patch_x] += 1

    # countë¡œ ë‚˜ëˆ ì„œ í�‰ê· 
    full_pred = output_pred / np.maximum(output_count, 1e-6)

    return (full_pred > 0.5).astype(np.uint8)  # binary mask ë°˜í™˜ (0 ë˜�ëŠ” 1)



import os
from collections import Counter

# ê°� tomo í�´ë�”ë‹¹ ìŠ¬ë�¼ì�´ìŠ¤ ê°œìˆ˜ ì„¸ê¸°
slice_counts = []
for tomo_id in os.listdir(TRAIN_IMG_DIR):
    tomo_path = os.path.join(TRAIN_IMG_DIR, tomo_id)
    if os.path.isdir(tomo_path):
        count = len(os.listdir(tomo_path))
        slice_counts.append(count)

# ìœ ë‹ˆí�¬í•œ ê°œìˆ˜ ì¢…ë¥˜ë§Œ ë³´ê¸°
unique_counts = sorted(set(slice_counts))
print("ğŸ”� Unique Z slice counts:")
print(unique_counts)

# (ì¶”ê°€) ë¶„í�¬ ë³´ê¸°
from collections import Counter
print("\nğŸ“Š Count distribution (how many tomos per slice count):")
for k, v in sorted(Counter(slice_counts).items()):
    print(f"{k} slices: {v} tomos")



import os
import numpy as np
from PIL import Image

def load_volume_original(tomo_path):
    """
    ì›�ë³¸ í•´ìƒ�ë�„ë¡œ 3D ë³¼ë¥¨ì�„ ë¡œë”©í•˜ëŠ” í•¨ìˆ˜.
    (Z, H, W) í˜•íƒœì�˜ np.ndarrayë¡œ ë°˜í™˜.

    Args:
        tomo_path (str): ìŠ¬ë�¼ì�´ìŠ¤ ì�´ë¯¸ì§€ë“¤ì�´ ë“¤ì–´ì�ˆëŠ” í�´ë�” ê²½ë¡œ

    Returns:
        volume (np.ndarray): (Z, H, W) í�¬ê¸°ì�˜ 3D ë°°ì—´
    """
    slice_files = sorted(os.listdir(tomo_path))  # jpg íŒŒì�¼ë“¤ì�´ ìˆœì„œëŒ€ë¡œ ì�ˆì�„ ê²ƒ
    volume = []

    for fname in slice_files:
        if fname.lower().endswith(".jpg"):
            img_path = os.path.join(tomo_path, fname)
            img = Image.open(img_path).convert("L")  # í�‘ë°±
            arr = np.array(img)
            volume.append(arr)

    # ìŠ¬ë�¼ì�´ìŠ¤ë“¤ì�„ zì¶•ìœ¼ë¡œ ìŒ“ëŠ”ë‹¤
    volume = np.stack(volume, axis=0)  # (Z, H, W)

    return volume



# import os
# # ì „ì²´ tomo í�´ë�” ID ëª©ë¡� ê°€ì ¸ì˜¤ê¸°
# tomo_ids = sorted(os.listdir(TRAIN_IMG_DIR))

# for tomo_id in tomo_ids:
#     tomo_path = os.path.join(TRAIN_IMG_DIR, tomo_id)

#     # tomo_pathê°€ í�´ë�”ì�¸ì§€ í™•ì�¸ (jpg íŒŒì�¼ë§Œ ì�ˆëŠ” í�´ë�”ë�¼ê³  ê°€ì •)
#     if os.path.isdir(tomo_path):
#         volume = load_volume_original(tomo_path)  # (Z, H, W)
#         print(f"{tomo_id}: volume shape = {volume.shape}")
#         # ì—¬ê¸°ì„œ volumeìœ¼ë¡œ í•™ìŠµ/ì¶”ë¡  ë¡œì§� ì—°ê²° ê°€ëŠ¥



def extract_motor_centers(pred_mask, tomo_id):
    """
    pred_mask: (Z, H, W) ì�´ì§„ ë§ˆìŠ¤í�¬
    tomo_id:   í˜„ì�¬ tomo ID
    return:    list of dict [{tomo_id, Motor axis 0, Motor axis 1, Motor axis 2}, ...]
    """
    results = []
    labeled, num_components = label(pred_mask)

    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled == comp_id)
        zc, yc, xc = center_of_mass(comp_mask)
        # ì‹¤ìˆ˜ ì¢Œí‘œ => ë°˜ì˜¬ë¦¼
        zc, yc, xc = round(zc), round(yc), round(xc)
        results.append({
            "tomo_id": tomo_id,
            "Motor axis 0": zc,
            "Motor axis 1": yc,
            "Motor axis 2": xc,
        })
    return results


##############################
# 4) ì „ì²´ ë£¨í”„
##############################
def predict_and_save_csv(test_dir, model, output_csv="submission.csv"):
    """
    test_dir: ê°� tomo í�´ë�”ê°€ ë“¤ì–´ì�ˆëŠ” ê²½ë¡œ (test set)
    model:    í•™ìŠµë�œ 3D U-Net
    """
    tomo_ids = sorted(os.listdir(test_dir))
    all_results = []

    for tomo_id in tqdm(tomo_ids, desc="Predicting all tomos"):
        tomo_path = os.path.join(test_dir, tomo_id)
        if not os.path.isdir(tomo_path):
            continue

        # 1) ë³¼ë¥¨ ë¡œë”©
        volume = load_volume_original(tomo_path)

        # 2) ìŠ¬ë�¼ì�´ë”© ìœˆë�„ìš° ì˜ˆì¸¡
        pred_mask = sliding_window_predict(volume, model, patch_size=(64,256,256), stride=(32,128,128), device='cuda')

        # 3) ë§ˆìŠ¤í�¬ â†’ ì¤‘ì‹¬ ì¢Œí‘œ ì¶”ì¶œ
        centers = extract_motor_centers(pred_mask, tomo_id)
        all_results.extend(centers)

    # 4) ì €ì�¥
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"âœ… Saved {output_csv} with {len(all_results)} lines.")






TEST_IMG_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"

predict_and_save_csv(
    test_dir=TEST_IMG_DIR,
    model=unet3d,
    output_csv="submission.csv"
)








