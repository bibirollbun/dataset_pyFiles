import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, AutoConfig
import joblib

# ======== 配置路径 ========
model_path = "/kaggle/input/heatmap-mask2former/pytorch/default/6/mask2former_bce_best (1)/mask2former_bce_best"
processor_path = model_path
classifier_path = os.path.join("/kaggle/input/heatmap-mask2former/pytorch/default/6", "svm_model.pkl")
test_root = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"
output_path = "/kaggle/working/submission.csv"

# ======== 自定义 SVM 概率阈值 ========
best_threshold = 0.6

# ======== 初始化 ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path, config=config, local_files_only=True).to(device).eval()
processor = Mask2FormerImageProcessor.from_pretrained(processor_path, local_files_only=True)
classifier = joblib.load(classifier_path)

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image, image.size

def get_center_from_mask(mask):
    if np.all(mask < 1e-6):
        return None
    return np.unravel_index(mask.argmax(), mask.shape)

# ======== 推理流程 ========
submission = []
batch_size = 32

for tomo_folder in tqdm(sorted(os.listdir(test_root)), desc=f"推理中 (threshold={best_threshold})"):
    folder_path = os.path.join(test_root, tomo_folder)
    if not os.path.isdir(folder_path):
        continue

    slice_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".jpg") and f.startswith("slice_")
    ])

    best_score = 0
    best_coords = None

    for i in range(0, len(slice_paths), batch_size):
        batch_paths = slice_paths[i:i+batch_size]
        image_infos = [load_image(p) for p in batch_paths]
        images = [img for img, size in image_infos]
        orig_sizes = [size for img, size in image_infos]

        encoded = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs["masks_queries_logits"]
            probs = torch.sigmoid(logits).cpu().numpy()

        for j, masks in enumerate(probs):
            filename = os.path.basename(batch_paths[j])
            z_index = int(filename.replace("slice_", "").replace(".jpg", ""))
            orig_w, orig_h = orig_sizes[j]

            top_k = 8
            selected_masks = masks[:top_k]

            for mask in selected_masks:
                max_val = mask.max()
                if max_val < 0.2:
                    continue  # 剪枝：跳过弱掩码

                mean_val = mask.mean()
                area_ratio = np.sum(mask > 0.5) / mask.size
                features = np.array([[max_val, mean_val, area_ratio]])
                proba = classifier.predict_proba(features)[0, 1]

                if proba >= best_threshold:
                    pred_center = get_center_from_mask(mask)
                    if pred_center is None:
                        continue
                    y_pred, x_pred = pred_center
                    h_mask, w_mask = mask.shape

                    x_orig = int(x_pred * orig_w / w_mask)
                    y_orig = int(y_pred * orig_h / h_mask)

                    if max_val > best_score:
                        best_score = max_val
                        best_coords = (z_index, y_orig, x_orig)

    if best_coords:
        submission.append({
            "tomo_id": tomo_folder,
            "Motor axis 0": best_coords[0],
            "Motor axis 1": best_coords[1],
            "Motor axis 2": best_coords[2],
        })
    else:
        submission.append({
            "tomo_id": tomo_folder,
            "Motor axis 0": -1,
            "Motor axis 1": -1,
            "Motor axis 2": -1,
        })

# ======== 保存提交文件 ========
pd.DataFrame(submission).to_csv(output_path, index=False)
print(f"submission.csv 已保存至: {output_path}")

