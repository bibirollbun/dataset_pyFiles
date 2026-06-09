# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import os
# import cv2  # Thay pydicom báº±ng opencv
# from PIL import Image


# class VINDRCXRDataset(Dataset):
#     def __init__(self, csv_file, image_dir, num_classes, target_size, map_size):
#         """
#         Khá»Ÿi táº¡o Dataset cho file PNG.
#         """
#         # Ä�á»�c CSV
#         self.data_df = pd.read_csv(csv_file)
#         self.image_dir = image_dir
#         self.target_size = target_size
#         self.map_size = map_size

#         # 1. Chuáº©n bá»‹ mapping Class ID
#         self.image_ids = self.data_df["image_id"].unique()  .tolist()
#         class_mapping_data = self.data_df[["class_name", "class_id"]].drop_duplicates()

#         self.class_to_id = {}
#         pathology_class_count = 0
#         for _, row in class_mapping_data.iterrows():
#             name = row["class_name"]
#             original_id = row["class_id"]
#             # Chá»‰ láº¥y 14 bá»‡nh lÃ½ (ID 0-13)
#             if name.lower() != "no finding" and 0 <= original_id <= 13:
#                 self.class_to_id[name] = int(original_id)
#                 pathology_class_count += 1

#         self.num_classes = pathology_class_count

#         # 2. Tá»‘i Æ°u hÃ³a viá»‡c tra cá»©u kÃ­ch thÆ°á»›c gá»‘c (Náº¿u cÃ³ trong CSV)
#         # Táº¡o dictionary Ä‘á»ƒ tra cá»©u nhanh width/height náº¿u CSV cÃ³ cá»™t nÃ y
#         self.has_dim_info = (
#             "width" in self.data_df.columns and "height" in self.data_df.columns
#         )

#         if self.has_dim_info:
#             # BÆ¯á»šC QUAN TRá»ŒNG:
#             # 1. Chá»‰ láº¥y 3 cá»™t cáº§n thiáº¿t
#             # 2. Loáº¡i bá»� cÃ¡c dÃ²ng trÃ¹ng image_id (Ä‘á»ƒ má»—i áº£nh chá»‰ cÃ²n 1 dÃ²ng duy nháº¥t chá»©a width/height)
#             unique_dims = self.data_df[["image_id", "width", "height"]].drop_duplicates(
#                 subset=["image_id"]
#             )

#             # 3. Giá»� thÃ¬ set_index sáº½ an toÃ n vÃ  to_dict sáº½ nhanh hÆ¡n nhiá»�u
#             self.dim_lookup = unique_dims.set_index("image_id").to_dict("index")
#         else:
#             self.dim_lookup = {}

#     def __len__(self):
#         return len(self.image_ids)

#     def __getitem__(self, idx):
#         image_id = self.image_ids[idx]

#         # Láº¥y cÃ¡c dÃ²ng annotation cá»§a áº£nh nÃ y
#         img_annotations = self.data_df[self.data_df["image_id"] == image_id]

#         # --- 1. Load áº¢nh PNG (Thay tháº¿ pháº§n DICOM) ---
#         img_path = os.path.join(self.image_dir, f"{image_id}.png")

#         # Ä�á»�c áº£nh grayscale (Ä‘á»ƒ khá»›p vá»›i logic cÅ© lÃ  1 channel)
#         # Náº¿u muá»‘n dÃ¹ng 3 kÃªnh mÃ u RGB thÃ¬ bá»� flag cv2.IMREAD_GRAYSCALE
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#         if image is None:  # Xá»­ lÃ½ lá»—i náº¿u khÃ´ng Ä‘á»�c Ä‘Æ°á»£c áº£nh
#             return None, None, None, image_id

#         # --- 2. XÃ¡c Ä‘á»‹nh KÃ­ch thÆ°á»›c Gá»‘c (QUAN TRá»ŒNG Ä�á»‚ TÃ�NH BBOX) ---
#         # Náº¿u CSV cÃ³ thÃ´ng tin width/height gá»‘c, ta Æ°u tiÃªn dÃ¹ng nÃ³
#         # (VÃ¬ náº¿u áº£nh PNG Ä‘Ã£ bá»‹ resize trÆ°á»›c Ä‘Ã³, image.shape sáº½ sai lá»‡ch vá»›i bbox gá»‘c)
#         if self.has_dim_info:
#             dims = self.dim_lookup[image_id]
#             original_w = dims["width"]
#             original_h = dims["height"]
#         else:
#             # Náº¿u CSV khÃ´ng cÃ³, Ä‘Ã nh pháº£i dÃ¹ng kÃ­ch thÆ°á»›c tháº­t cá»§a áº£nh vá»«a Ä‘á»�c
#             # LÆ°u Ã½: Náº¿u áº£nh nÃ y Ä‘Ã£ bá»‹ resize (vd 512x512) thÃ¬ bbox sáº½ bá»‹ tÃ­nh sai!
#             original_h, original_w = image.shape[:2]

#         if original_w <= 0 or original_h <= 0:
#             return None, None, None, image_id

#         # --- 3. Resize & Normalize ---
#         # Resize vá»� target_size (vÃ­ dá»¥ 384x384)
#         image = cv2.resize(image, (self.target_size, self.target_size))

#         # Normalize 0-255 -> 0.0-1.0
#         image = image.astype(np.float32) / 255.0

#         # Chuyá»ƒn sang Tensor [1, H, W]
#         image = torch.from_numpy(image).unsqueeze(0)

#         # --- 4. Táº¡o Label & Attention Map (Logic giá»¯ nguyÃªn) ---
#         cls_label = torch.zeros(self.num_classes, dtype=torch.float32)
#         present_classes = img_annotations["class_name"].unique()

#         for cls_name in present_classes:
#             cls_id = self.class_to_id.get(cls_name)
#             if cls_id is not None:
#                 cls_label[cls_id] = 1.0

#         # Táº¡o Map
#         attn_maps = torch.zeros(
#             (self.num_classes, self.map_size, self.map_size), dtype=torch.float32
#         )

#         for _, row in img_annotations.iterrows():
#             cls_name = row["class_name"]
#             cls_id = self.class_to_id.get(cls_name)

#             if cls_id is None:
#                 continue
#             if pd.isna(row[["x_min", "y_min", "x_max", "y_max"]]).any():
#                 continue

#             # TÃ­nh tá»�a Ä‘á»™ trÃªn Map 12x12 dá»±a trÃªn tá»· lá»‡ vá»›i kÃ­ch thÆ°á»›c gá»‘c
#             x_min = int(row["x_min"] * (self.map_size / original_w))
#             y_min = int(row["y_min"] * (self.map_size / original_h))
#             x_max = int(row["x_max"] * (self.map_size / original_w))
#             y_max = int(row["y_max"] * (self.map_size / original_h))

#             x_min = max(0, x_min)
#             y_min = max(0, y_min)
#             x_max = min(self.map_size, x_max)
#             y_max = min(self.map_size, y_max)

#             if x_max > x_min and y_max > y_min:
#                 attn_maps[cls_id, y_min:y_max, x_min:x_max] = 1.0

#         return image, cls_label, attn_maps, image_id



# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import torch

# def visualize_sample(image, cls_label, attn_maps, image_id, class_to_id):
#     """
#     image: Tensor [1, 1, H, W]
#     attn_maps: Tensor [num_classes, map_size, map_size]
#     cls_label: Tensor [num_classes]
#     """

#     # --- 1. Hiá»ƒn thá»‹ áº£nh gá»‘c ---
#     img_np = image.squeeze().numpy()

#     plt.figure(figsize=(6,6))
#     plt.imshow(img_np, cmap='gray')
#     plt.title(f"Image ID: {image_id}")
#     plt.axis('off')
#     plt.show()

#     # --- 2. Hiá»ƒn thá»‹ vector multi-label ---
#     print("Multi-label vector:")
#     for cls_name, cls_id in class_to_id.items():
#         if cls_label[0][cls_id] == 1:
#             print(f"  âœ” {cls_name} (id={cls_id})")

#     # --- 3. Visualize attention maps ---
#     num_classes = attn_maps.shape[1]  # batch=1 â†’ 1 x C x 12 x 12
#     map_size = attn_maps.shape[-1]
#     H, W = img_np.shape

#     for cls_name, cls_id in class_to_id.items():
#         if cls_label[0][cls_id] == 0:
#             continue  # chá»‰ hiá»‡n cÃ¡c lá»›p xuáº¥t hiá»‡n

#         print(f"\n---- Attention map for {cls_name} (id={cls_id}) ----")

#         att_map = attn_maps[0, cls_id].numpy()

#         # Resize 12x12 â†’ áº£nh size (HÃ—W)
#         att_big = cv2.resize(att_map, (W, H), interpolation=cv2.INTER_NEAREST)

#         # Normalize Ä‘á»ƒ overlay Ä‘áº¹p
#         att_big = (att_big - att_big.min()) / (att_big.max() + 1e-6)

#         # Apply heatmap
#         heatmap = cv2.applyColorMap((att_big * 255).astype(np.uint8), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#         overlay = (heatmap * 0.5 + np.stack([img_np]*3, axis=-1) * 255 * 0.5).astype(np.uint8)

#         plt.figure(figsize=(6,6))
#         plt.imshow(overlay)
#         plt.title(f"Attention Map: {cls_name}")
#         plt.axis('off')
#         plt.show()



# from torch.utils.data import DataLoader

# dataset = VINDRCXRDataset(
#     csv_file="/kaggle/input/png-trainer-9/pytorch/default/1/src/csv/train_split.csv",
#     image_dir="/kaggle/input/vindr-image-convert/train_png_384",
#     num_classes=14,
#     target_size=384,
#     map_size=12
# )

# loader = DataLoader(dataset, batch_size=1, shuffle=True)
# image, cls_label, attn_maps, image_id = next(iter(loader))
# print(cls_label)
# visualize_sample(image, cls_label, attn_maps, image_id[0], dataset.class_to_id)


# !python /kaggle/input/png-trainer-28/pytorch/default/1/src/train.py \
#   --csv_path "/kaggle/input/png-trainer-9/pytorch/default/1/src/csv/train_split.csv" \
#   --image_path "/kaggle/input/vindr-image-convert/train_png_384" \
#   --save_path "/kaggle/working/checkpoints" \
#   --batch_size 32 \
#   --num_prototypes 15 \
#   --epochs_p1 20 --lr 3e-4
#   # --epochs_p2 15 \
#   # --epochs_p3 10

!python /kaggle/input/png-trainer-phase-2/pytorch/default/1/src/train_phase2.py \
  --csv_path "/kaggle/input/png-trainer-9/pytorch/default/1/src/csv/train_split.csv" \
  --image_path "/kaggle/input/vindr-image-convert/train_png_384" \
  --save_path "/kaggle/working/checkpoints" \
--resume_phase1 /kaggle/input/csr-model-phase-1/pytorch/default/1/csr_phase1.pth \
--skip_phase1 \
  --batch_size 32 \
  --num_prototypes 15 \
--epochs_p2 20 --lr 3e-4
      # --epochs_p3 10


# !python /kaggle/input/evaluate-v6/pytorch/default/1/evaluate.py --test_csv /kaggle/input/inference-model-v3/pytorch/default/1/src/csv/test_split.csv --image_path /kaggle/input/vindr-image-convert/train_png_384 --checkpoint /kaggle/input/csr-model/pytorch/default/1/csr_final_model.pth --num_prototypes 10 --img_size 384 --device cuda --threshold 0.67


# # =============================================================================
# # INFERENCE SCRIPT CHO KAGGLE NOTEBOOK - HOÃ€N CHá»ˆNH
# # Cháº¡y tá»«ng cell theo thá»© tá»±
# # =============================================================================

# # %% [markdown]
# # ## Cell 1: Import Libraries

# # %%
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import os
# import timm

# # %% [markdown]
# # ## Cell 2: Cáº¥u hÃ¬nh - THAY Ä�á»”I CÃ�C BIáº¾N á»� Ä�Ã‚Y

# # %%
# # =================== Cáº¤U HÃŒNH - THAY Ä�á»”I á»� Ä�Ã‚Y ===================

# # Ä�Æ°á»�ng dáº«n áº£nh cáº§n inference
# IMAGE_PATH = "/kaggle/input/vindr-image-convert/train_png_384/051132a778e61a86eb147c7c6f564dfe.png"

# # Ä�Æ°á»�ng dáº«n checkpoint model
# CHECKPOINT_PATH = "/kaggle/input/vin-csr-training/checkpoints/csr_phase1.pth"

# # Ä�Æ°á»�ng dáº«n lÆ°u káº¿t quáº£ (None náº¿u khÃ´ng muá»‘n lÆ°u)
# SAVE_PATH = "/kaggle/working/result.png"

# # Device
# DEVICE = "cuda"  # hoáº·c "cpu"

# # Model config
# NUM_CLASSES = 14
# NUM_PROTOTYPES = 15
# MODEL_NAME = "densenet121"  # hoáº·c "densenet121", "efficientnet_b0", ...
# IMG_SIZE = 384

# # Sá»‘ bá»‡nh top-k hiá»ƒn thá»‹
# TOP_K = 3

# # === THRESHOLD CHO SALIENCY MAPS ===
# CAM_THRESHOLD = 1  # GiÃ¡ trá»‹ tá»« 0.0 - 1.0 (0 = hiá»‡n táº¥t cáº£, 1 = chá»‰ hiá»‡n vÃ¹ng cao nháº¥t)
# SHOW_BINARY_MASK = False  # True = hiá»‡n mask nhá»‹ phÃ¢n, False = hiá»‡n heatmap gradient

# # TÃªn cÃ¡c bá»‡nh
# CLASS_NAMES = [
#     'Aortic enlargement',
#     'Atelectasis',
#     'Calcification',
#     'Cardiomegaly',
#     'Consolidation',
#     'ILD',
#     'Infiltration',
#     'Lung Opacity',
#     'Nodule/Mass',
#     'Other lesion',
#     'Pleural effusion',
#     'Pleural thickening',
#     'Pneumothorax',
#     'Pulmonary fibrosis'
# ]

# # ================================================================

# # %% [markdown]
# # ## Cell 3: Ä�á»‹nh nghÄ©a Model (CSRModel)

# # %%
# class CSRModel(nn.Module):
#     def __init__(self, num_classes=14, num_prototypes=5, model_name="resnet50", pretrained=True):
#         super().__init__()
        
#         # --- PHáº¦N 1: CONCEPT MODEL (Giai Ä‘oáº¡n 1) ---
#         # F: Feature Extractor
#         self.backbone = timm.create_model(
#             model_name, pretrained=pretrained, features_only=True, out_indices=(4,)
#         )
#         feature_info = self.backbone.feature_info.get_dicts()[-1]
#         self.feature_dim = feature_info["num_chs"]

#         # C: Concept Head (Táº¡o CAMs)
#         self.concept_head = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)

#         # --- PHáº¦N 2: PROTOTYPES (Giai Ä‘oáº¡n 2) ---
#         self.embedding_dim = 128
#         # P: Projector (Chiáº¿u feature vá»� khÃ´ng gian contrastive)
#         self.projector = nn.Sequential(
#             nn.Linear(self.feature_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, self.embedding_dim)
#         )
        
#         # Learnable Prototypes: [Num_Classes, Num_Prototypes_Per_Class, Emb_Dim]
#         self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, self.embedding_dim))
        
#         # --- PHáº¦N 3: TASK HEAD (Giai Ä‘oáº¡n 3) ---
#         # H: Task Head (Dá»± Ä‘oÃ¡n bá»‡nh tá»« Ä‘iá»ƒm tÆ°Æ¡ng Ä‘á»“ng)
#         self.task_head = nn.Linear(num_classes * num_prototypes, num_classes)
#         self.num_classes = num_classes
#         self.num_prototypes = num_prototypes

#     def get_features_and_cam(self, x):
#         """DÃ¹ng cho Giai Ä‘oáº¡n 1"""
#         if x.size(1) == 1:
#             x = x.repeat(1, 3, 1, 1)
#         features = self.backbone(x)[0]
#         attn_logits = self.concept_head(features)
#         return features, attn_logits

#     def get_projected_vectors(self, features, attn_logits):
#         """DÃ¹ng cho Giai Ä‘oáº¡n 2: Láº¥y Local Concept Vectors"""
#         B, C, H, W = features.shape
#         K = attn_logits.shape[1]
        
#         # 1. Normalize CAM (Spatial Softmax)
#         attn_weights = F.softmax(attn_logits.view(B, K, -1), dim=-1).view(B, K, H, W)
        
#         # 2. Weighted Sum Ä‘á»ƒ láº¥y vector Ä‘áº¡i diá»‡n cho tá»«ng concept
#         features_flat = features.view(B, C, -1).permute(0, 2, 1)
#         local_concept_vectors = torch.bmm(attn_weights.view(B, K, -1), features_flat)
        
#         # 3. Project sang khÃ´ng gian embedding
#         projected_vectors = self.projector(local_concept_vectors)
#         return F.normalize(projected_vectors, p=2, dim=-1)

#     def forward(self, x):
#         """Luá»“ng cháº¡y Full (DÃ¹ng cho Giai Ä‘oáº¡n 3 & Inference)"""
#         # 1. TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng & CAM
#         features, attn_logits = self.get_features_and_cam(x)
        
#         # 2. TÃ­nh projected vectors
#         projected_vectors = self.get_projected_vectors(features, attn_logits)
        
#         # 3. TÃ­nh Similarity Score
#         prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
#         sim_scores = torch.einsum('bkc,kmc->bkm', projected_vectors, prototypes_norm)
#         s_vector = sim_scores.reshape(x.size(0), -1)
        
#         # 4. Predict
#         logits = self.task_head(s_vector)
        
#         return {
#             "logits": logits,
#             "attn_maps": attn_logits,
#             "projected_vectors": projected_vectors,
#             "sim_scores": sim_scores
#         }

# print("âœ… CSRModel defined!")

# # %% [markdown]
# # ## Cell 4: HÃ m tiá»�n xá»­ lÃ½ vÃ  visualize

# # %%
# def preprocess_image(image_path, target_size=384):
#     """Ä�á»�c vÃ  tiá»�n xá»­ lÃ½ áº£nh"""
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"File not found: {image_path}")
        
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError("Cannot read image")
        
#     image = cv2.resize(image, (target_size, target_size))
#     img_norm = image.astype(np.float32) / 255.0
#     img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
#     return img_tensor, image


# def visualize_result(original_img, probs, similarities, attn_maps, 
#                      class_names, top_k=3, save_path=None,
#                      cam_threshold=0.3, show_binary_mask=False):
#     """
#     Váº½ káº¿t quáº£ dá»± Ä‘oÃ¡n vá»›i heatmap
    
#     Args:
#         original_img: áº¢nh gá»‘c grayscale
#         probs: XÃ¡c suáº¥t dá»± Ä‘oÃ¡n [num_classes]
#         similarities: Similarity scores [B, K, M]
#         attn_maps: Attention maps [B, K, H, W]
#         class_names: Danh sÃ¡ch tÃªn cÃ¡c bá»‡nh
#         top_k: Sá»‘ bá»‡nh hiá»ƒn thá»‹
#         save_path: Ä�Æ°á»�ng dáº«n lÆ°u áº£nh
#         cam_threshold: NgÆ°á»¡ng Ä‘á»ƒ lá»�c CAM (0.0 - 1.0)
#         show_binary_mask: True = mask nhá»‹ phÃ¢n, False = gradient heatmap
#     """
#     top_indices = np.argsort(probs)[::-1][:top_k]
    
#     fig, axes = plt.subplots(1, top_k + 1, figsize=(5 * (top_k + 1), 5))
    
#     # 1. áº¢nh gá»‘c
#     axes[0].imshow(original_img, cmap='gray')
#     axes[0].set_title("Input X-Ray", fontsize=14)
#     axes[0].axis('off')
    
#     info_text = f"PREDICTIONS (CAM threshold={cam_threshold}):\n"
#     for idx in top_indices:
#         name = class_names[idx]
#         sim_score = similarities[0, idx, :].max().item()
#         prob = probs[idx]
#         info_text += f"{name}: {prob*100:.1f}% (Sim: {sim_score:.2f})\n"
#     axes[0].set_xlabel(info_text, fontsize=10, loc='left')
    
#     # 2. Heatmap top-k vá»›i threshold
#     for i, idx in enumerate(top_indices):
#         name = class_names[idx]
        
#         # Láº¥y CAM
#         cam = attn_maps[0, idx].cpu().numpy()
#         cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
#         # Normalize vá»� [0, 1]
#         cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        
#         # === Ã�P Dá»¤NG THRESHOLD ===
#         if show_binary_mask:
#             # Binary mask: chá»‰ hiá»‡n vÃ¹ng > threshold
#             cam_thresholded = (cam_norm > cam_threshold).astype(np.float32)
#         else:
#             # Gradient mask: giá»¯ gradient nhÆ°ng zero-out vÃ¹ng < threshold
#             cam_thresholded = np.where(cam_norm > cam_threshold, cam_norm, 0)
#             # Re-normalize sau khi threshold
#             if cam_thresholded.max() > 0:
#                 cam_thresholded = cam_thresholded / cam_thresholded.max()
        
#         # Táº¡o heatmap
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_thresholded), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
#         # Overlay lÃªn áº£nh gá»‘c
#         img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        
#         # Chá»‰ overlay vÃ¹ng cÃ³ giÃ¡ trá»‹ (trÃ¡nh mÃ u xanh Ä‘áº­m á»Ÿ vÃ¹ng = 0)
#         alpha_mask = (cam_thresholded > 0).astype(np.float32)
#         alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        
#         overlay = img_rgb.copy().astype(np.float32)
#         overlay = overlay * (1 - alpha_mask * 0.4) + heatmap.astype(np.float32) * (alpha_mask * 0.4)
#         overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
#         axes[i + 1].imshow(overlay)
#         axes[i + 1].set_title(f"{name}\n{probs[idx]*100:.1f}%", fontsize=12)
#         axes[i + 1].axis('off')
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=150)
#         print(f"ğŸ’¾ Result saved to: {save_path}")
    
#     plt.show()


# def visualize_with_multiple_thresholds(original_img, probs, similarities, attn_maps,
#                                         class_names, class_idx=None, 
#                                         thresholds=[0.0, 0.3, 0.5, 0.7]):
#     """
#     So sÃ¡nh heatmap vá»›i nhiá»�u má»©c threshold khÃ¡c nhau
    
#     Args:
#         class_idx: Index cá»§a class cáº§n visualize (None = top-1)
#         thresholds: List cÃ¡c giÃ¡ trá»‹ threshold Ä‘á»ƒ so sÃ¡nh
#     """
#     if class_idx is None:
#         class_idx = np.argmax(probs)
    
#     name = class_names[class_idx]
#     prob = probs[class_idx]
    
#     fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(4 * (len(thresholds) + 1), 4))
    
#     # áº¢nh gá»‘c
#     axes[0].imshow(original_img, cmap='gray')
#     axes[0].set_title(f"Original\n{name}: {prob*100:.1f}%", fontsize=11)
#     axes[0].axis('off')
    
#     # CAM vá»›i cÃ¡c threshold khÃ¡c nhau
#     cam = attn_maps[0, class_idx].cpu().numpy()
#     cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
#     cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    
#     for i, thresh in enumerate(thresholds):
#         cam_thresholded = np.where(cam_norm > thresh, cam_norm, 0)
#         if cam_thresholded.max() > 0:
#             cam_thresholded = cam_thresholded / cam_thresholded.max()
        
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_thresholded), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
#         img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        
#         alpha_mask = (cam_thresholded > 0).astype(np.float32)
#         alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        
#         overlay = img_rgb.copy().astype(np.float32)
#         overlay = overlay * (1 - alpha_mask * 0.5) + heatmap.astype(np.float32) * (alpha_mask * 0.5)
#         overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
#         # TÃ­nh % vÃ¹ng Ä‘Æ°á»£c highlight
#         coverage = (cam_norm > thresh).sum() / cam_norm.size * 100
        
#         axes[i + 1].imshow(overlay)
#         axes[i + 1].set_title(f"Threshold={thresh}\nCoverage: {coverage:.1f}%", fontsize=11)
#         axes[i + 1].axis('off')
    
#     plt.suptitle(f"CAM Threshold Comparison: {name}", fontsize=14)
#     plt.tight_layout()
#     plt.show()


# def visualize_all_classes(original_img, probs, attn_maps, class_names, 
#                           cam_threshold=0.3, prob_threshold=0.1):
#     """
#     Hiá»ƒn thá»‹ heatmap cho Táº¤T Cáº¢ cÃ¡c class cÃ³ probability > prob_threshold
    
#     Args:
#         prob_threshold: Chá»‰ hiá»ƒn thá»‹ class cÃ³ prob > threshold nÃ y
#     """
#     # Lá»�c cÃ¡c class cÃ³ prob > threshold
#     valid_indices = np.where(probs > prob_threshold)[0]
    
#     if len(valid_indices) == 0:
#         print(f"âš ï¸� KhÃ´ng cÃ³ class nÃ o cÃ³ probability > {prob_threshold}")
#         return
    
#     # Sáº¯p xáº¿p theo prob giáº£m dáº§n
#     valid_indices = valid_indices[np.argsort(probs[valid_indices])[::-1]]
    
#     n_classes = len(valid_indices)
#     n_cols = min(4, n_classes + 1)
#     n_rows = (n_classes + n_cols) // n_cols
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
#     axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
#     # áº¢nh gá»‘c
#     axes[0].imshow(original_img, cmap='gray')
#     axes[0].set_title("Original X-Ray", fontsize=12)
#     axes[0].axis('off')
    
#     # Heatmap cho tá»«ng class
#     for i, idx in enumerate(valid_indices):
#         if i + 1 >= len(axes):
#             break
            
#         name = class_names[idx]
#         prob = probs[idx]
        
#         cam = attn_maps[0, idx].cpu().numpy()
#         cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
#         cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        
#         cam_thresholded = np.where(cam_norm > cam_threshold, cam_norm, 0)
#         if cam_thresholded.max() > 0:
#             cam_thresholded = cam_thresholded / cam_thresholded.max()
        
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_thresholded), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
#         img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        
#         alpha_mask = (cam_thresholded > 0).astype(np.float32)
#         alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        
#         overlay = img_rgb.copy().astype(np.float32)
#         overlay = overlay * (1 - alpha_mask * 0.4) + heatmap.astype(np.float32) * (alpha_mask * 0.4)
#         overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
#         axes[i + 1].imshow(overlay)
#         axes[i + 1].set_title(f"{name}\n{prob*100:.1f}%", fontsize=10)
#         axes[i + 1].axis('off')
    
#     # áº¨n cÃ¡c axes thá»«a
#     for j in range(i + 2, len(axes)):
#         axes[j].axis('off')
    
#     plt.suptitle(f"All Detected Conditions (prob > {prob_threshold*100:.0f}%)", fontsize=14)
#     plt.tight_layout()
#     plt.show()

# print("âœ… Helper functions defined!")

# # %% [markdown]
# # ## Cell 5: Load Model

# # %%
# device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
# print(f"ğŸ–¥ï¸�  Device: {device}")

# # Táº¡o model
# model = CSRModel(
#     num_classes=NUM_CLASSES,
#     num_prototypes=NUM_PROTOTYPES,
#     model_name=MODEL_NAME,
#     pretrained=False  # KhÃ´ng cáº§n pretrained vÃ¬ sáº½ load checkpoint
# )

# # Load checkpoint
# print(f"ğŸ“‚ Loading checkpoint: {CHECKPOINT_PATH}")
# ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

# if 'model_state_dict' in ckpt:
#     model.load_state_dict(ckpt['model_state_dict'])
#     print(f"   Epoch: {ckpt.get('epoch', 'N/A')}")
# elif 'state_dict' in ckpt:
#     model.load_state_dict(ckpt['state_dict'])
# else:
#     model.load_state_dict(ckpt)

# model.to(device)
# model.eval()
# print("âœ… Model loaded successfully!")

# # %% [markdown]
# # ## Cell 6: Inference trÃªn 1 áº£nh

# # %%
# # Tiá»�n xá»­ lÃ½
# print(f"ğŸ–¼ï¸�  Loading image: {IMAGE_PATH}")
# img_tensor, original_img = preprocess_image(IMAGE_PATH, target_size=IMG_SIZE)
# img_tensor = img_tensor.to(device)

# # Inference
# print("ğŸ”® Running inference...")
# with torch.no_grad():
#     outputs = model(img_tensor)
    
#     logits = outputs['logits'][0]
#     sim_scores = outputs['sim_scores']
#     attn_maps = outputs['attn_maps']
    
#     probs = torch.sigmoid(logits).cpu().numpy()

# # Hiá»ƒn thá»‹ káº¿t quáº£ dáº¡ng text
# print("\nğŸ“Š Prediction Results:")
# print("-" * 50)
# for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
#     bar = "â–ˆ" * int(prob * 20)
#     status = "âš ï¸�" if prob > 0.5 else "  "
#     print(f"{status} {name:20s}: {prob*100:5.1f}% {bar}")

# # Hiá»ƒn thá»‹ cÃ¡c bá»‡nh Ä‘Æ°á»£c phÃ¡t hiá»‡n (prob > 0.5)
# detected = [(CLASS_NAMES[i], probs[i]) for i in range(len(probs)) if probs[i] > 0.5]
# print("\n" + "=" * 50)
# if detected:
#     print("ğŸ”´ DETECTED CONDITIONS:")
#     for name, prob in detected:
#         print(f"   â€¢ {name}: {prob*100:.1f}%")
# else:
#     print("ğŸŸ¢ No significant conditions detected (all < 50%)")

# # %% [markdown]
# # ## Cell 7: Visualize Top-K vá»›i Threshold

# # %%
# # Visualize vá»›i threshold
# visualize_result(
#     original_img, probs, sim_scores, attn_maps,
#     class_names=CLASS_NAMES,
#     top_k=TOP_K,
#     save_path=SAVE_PATH,
#     cam_threshold=CAM_THRESHOLD,
#     show_binary_mask=SHOW_BINARY_MASK
# )

# # %% [markdown]
# # ## Cell 8: So sÃ¡nh cÃ¡c má»©c Threshold

# # %%
# # So sÃ¡nh nhiá»�u threshold cho bá»‡nh cÃ³ xÃ¡c suáº¥t cao nháº¥t
# visualize_with_multiple_thresholds(
#     original_img, probs, sim_scores, attn_maps,
#     class_names=CLASS_NAMES,
#     class_idx=None,  # None = tá»± Ä‘á»™ng chá»�n top-1, hoáº·c Ä‘áº·t sá»‘ 0-13
#     thresholds=[0.0, 0.2, 0.4, 0.6, 0.8]
# )

# # %% [markdown]
# # ## Cell 9: Hiá»ƒn thá»‹ táº¥t cáº£ cÃ¡c bá»‡nh phÃ¡t hiá»‡n Ä‘Æ°á»£c

# # %%
# # Hiá»ƒn thá»‹ heatmap cho táº¥t cáº£ class cÃ³ probability > 10%
# visualize_all_classes(
#     original_img, probs, attn_maps, 
#     class_names=CLASS_NAMES,
#     cam_threshold=CAM_THRESHOLD,
#     prob_threshold=0.6  # Chá»‰ hiá»‡n class cÃ³ prob > 10%
# )

# # %% [markdown]
# # ## Cell 10 (Optional): Inference trÃªn nhiá»�u áº£nh

# # %%
# def batch_inference(image_paths, model, device, class_names, img_size=384, prob_threshold=0.5):
#     """Inference trÃªn nhiá»�u áº£nh vÃ  tráº£ vá»� káº¿t quáº£"""
#     results = []
    
#     for path in image_paths:
#         try:
#             img_tensor, original_img = preprocess_image(path, target_size=img_size)
#             img_tensor = img_tensor.to(device)
            
#             with torch.no_grad():
#                 outputs = model(img_tensor)
#                 probs = torch.sigmoid(outputs['logits'][0]).cpu().numpy()
            
#             # Láº¥y cÃ¡c bá»‡nh cÃ³ prob > threshold
#             detected = [(class_names[i], probs[i]) for i in range(len(probs)) if probs[i] > prob_threshold]
            
#             results.append({
#                 'path': path,
#                 'probs': probs,
#                 'detected': detected,
#                 'top_pred': class_names[np.argmax(probs)],
#                 'top_prob': probs.max()
#             })
#         except Exception as e:
#             results.append({
#                 'path': path,
#                 'error': str(e)
#             })
    
#     return results

# # VÃ­ dá»¥ sá»­ dá»¥ng:
# # IMAGE_LIST = [
# #     "/kaggle/input/images/img1.png",
# #     "/kaggle/input/images/img2.png",
# #     "/kaggle/input/images/img3.png",
# # ]
# # results = batch_inference(IMAGE_LIST, model, device, CLASS_NAMES, IMG_SIZE)
# # 
# # for r in results:
# #     if 'error' in r:
# #         print(f"â�Œ {r['path']}: {r['error']}")
# #     else:
# #         print(f"âœ… {r['path']}:")
# #         if r['detected']:
# #             for name, prob in r['detected']:
# #                 print(f"   â€¢ {name}: {prob*100:.1f}%")
# #         else:
# #             print(f"   No conditions detected")

# # %% [markdown]
# # ## Cell 11 (Optional): Interactive Threshold Slider (Jupyter)

# # %%
# # Chá»‰ cháº¡y Ä‘Æ°á»£c trong Jupyter Notebook vá»›i ipywidgets
# try:
#     from ipywidgets import interact, FloatSlider
    
#     def interactive_threshold(threshold=0.3):
#         visualize_result(
#             original_img, probs, sim_scores, attn_maps,
#             class_names=CLASS_NAMES,
#             top_k=TOP_K,
#             save_path=None,
#             cam_threshold=threshold,
#             show_binary_mask=False
#         )
    
#     interact(
#         interactive_threshold,
#         threshold=FloatSlider(min=0.0, max=1.0, step=0.05, value=0.3, description='CAM Threshold')
#     )
# except ImportError:
#     print("âš ï¸� ipywidgets not available. Skip interactive slider.")


# """
# Test model SAU Phase 1 - Chá»‰ dÃ¹ng concept_head, KHÃ”NG dÃ¹ng task_head
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import os
# import timm
# import pandas as pd

# # =================== Cáº¤U HÃŒNH ===================
# CHECKPOINT_PATH = "/kaggle/input/vin-csr-training/checkpoints/csr_phase1.pth"
# IMAGE_PATH = "/kaggle/input/vindr-image-convert/train_png_384/9a5094b2563a1ef3ff50dc5c7ff71345.png"
# CSV_PATH = "/kaggle/input/png-trainer-9/pytorch/default/1/src/csv/test_split.csv"
# IMAGE_DIR = "/kaggle/input/vindr-image-convert/train_png_384"

# NUM_CLASSES = 14
# NUM_PROTOTYPES = 15
# MODEL_NAME = "densenet121"
# IMG_SIZE = 384
# DEVICE = "cuda"

# CLASS_NAMES = [
#     'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
#     'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity',
#     'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening',
#     'Pneumothorax', 'Pulmonary fibrosis'
# ]

# # ================================================

# class CSRModel(nn.Module):
#     def __init__(self, num_classes=14, num_prototypes=5, model_name="resnet50", pretrained=True):
#         super().__init__()
        
#         self.backbone = timm.create_model(
#             model_name, pretrained=pretrained, features_only=True, out_indices=(4,)
#         )
#         feature_info = self.backbone.feature_info.get_dicts()[-1]
#         self.feature_dim = feature_info["num_chs"]
#         self.concept_head = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)
        
#         self.embedding_dim = 128
#         self.projector = nn.Sequential(
#             nn.Linear(self.feature_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, self.embedding_dim)
#         )
        
#         self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, self.embedding_dim))
#         self.task_head = nn.Linear(num_classes * num_prototypes, num_classes)
#         self.num_classes = num_classes
#         self.num_prototypes = num_prototypes

#     def get_features_and_cam(self, x):
#         if x.size(1) == 1:
#             x = x.repeat(1, 3, 1, 1)
#         features = self.backbone(x)[0]
#         attn_logits = self.concept_head(features)
#         return features, attn_logits
    
#     def forward_phase1(self, x):
#         """
#         Forward CHá»ˆ DÃ™NG CHO PHASE 1
#         Dá»± Ä‘oÃ¡n báº±ng Global Average Pooling trÃªn CAM
#         """
#         _, attn_logits = self.get_features_and_cam(x)
#         # GAP: [B, K, H, W] -> [B, K]
#         logits = F.adaptive_avg_pool2d(attn_logits, (1, 1)).view(x.size(0), -1)
#         return {
#             "logits": logits,
#             "attn_maps": attn_logits
#         }

#     def forward(self, x):
#         # Full forward (Phase 3)
#         features, attn_logits = self.get_features_and_cam(x)
#         projected_vectors = self.get_projected_vectors(features, attn_logits)
#         prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
#         sim_scores = torch.einsum('bkc,kmc->bkm', projected_vectors, prototypes_norm)
#         s_vector = sim_scores.reshape(x.size(0), -1)
#         logits = self.task_head(s_vector)
#         return {"logits": logits, "attn_maps": attn_logits, "sim_scores": sim_scores}
    
#     def get_projected_vectors(self, features, attn_logits):
#         B, C, H, W = features.shape
#         K = attn_logits.shape[1]
#         attn_weights = F.softmax(attn_logits.view(B, K, -1), dim=-1).view(B, K, H, W)
#         features_flat = features.view(B, C, -1).permute(0, 2, 1)
#         local_concept_vectors = torch.bmm(attn_weights.view(B, K, -1), features_flat)
#         projected_vectors = self.projector(local_concept_vectors)
#         return F.normalize(projected_vectors, p=2, dim=-1)


# def preprocess_image(image_path, target_size=384):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"Cannot read image: {image_path}")
#     image = cv2.resize(image, (target_size, target_size))
#     img_norm = image.astype(np.float32) / 255.0
#     img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
#     return img_tensor


# def predict_phase1(model, image_path, device, threshold=0.5):
#     """
#     Dá»± Ä‘oÃ¡n Ä�ÃšNG CÃ�CH cho Phase 1
#     """
#     img_tensor = preprocess_image(image_path, IMG_SIZE)
#     img_tensor = img_tensor.to(device)
    
#     with torch.no_grad():
#         # Sá»¬ Dá»¤NG forward_phase1 thay vÃ¬ forward
#         outputs = model.forward_phase1(img_tensor)
#         logits = outputs['logits'][0]
#         probs = torch.sigmoid(logits).cpu().numpy()
    
#     predictions = (probs > threshold).astype(int)
#     return probs, predictions


# def get_ground_truth(image_id, csv_path):
#     df = pd.read_csv(csv_path)
#     rows = df[df['image_id'] == image_id]
    
#     gt_labels = np.zeros(NUM_CLASSES)
#     for _, row in rows.iterrows():
#         class_id = row['class_id']
#         if 0 <= class_id < NUM_CLASSES:
#             gt_labels[class_id] = 1
#     return gt_labels


# def print_results(image_id, probs, predictions, gt_labels):
#     print("\n" + "=" * 70)
#     print(f"ğŸ“· Image: {image_id}")
#     print("=" * 70)
#     print(f"{'Class':<25} {'Prob':>8} {'Pred':>6} {'GT':>6} {'Match':>7}")
#     print("-" * 70)
    
#     correct = 0
#     for i in range(NUM_CLASSES):
#         prob = probs[i]
#         pred = predictions[i]
#         gt = int(gt_labels[i])
#         match = "âœ…" if pred == gt else "â�Œ"
#         if pred == gt:
#             correct += 1
#         prefix = "ğŸ”´" if pred == 1 else "  "
#         print(f"{prefix}{CLASS_NAMES[i]:<23} {prob*100:>7.1f}% {pred:>6} {gt:>6} {match:>7}")
    
#     print("-" * 70)
#     pred_diseases = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if predictions[i] == 1]
#     gt_diseases = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if gt_labels[i] == 1]
    
#     print(f"\nğŸ”´ Predicted: {pred_diseases if pred_diseases else 'None'}")
#     print(f"ğŸŸ¢ Ground truth: {gt_diseases if gt_diseases else 'None'}")
#     print(f"\nğŸ“Š Accuracy: {correct}/{NUM_CLASSES} ({correct/NUM_CLASSES*100:.1f}%)")
    
#     if np.array_equal(predictions, gt_labels.astype(int)):
#         print("âœ… EXACT MATCH!")
#     else:
#         print("â�Œ NOT EXACT MATCH")


# def test_multiple_samples(model, csv_path, image_dir, device, num_samples=10):
#     """Test trÃªn nhiá»�u máº«u"""
#     df = pd.read_csv(csv_path)
#     unique_images = df['image_id'].unique()
#     selected = np.random.choice(unique_images, min(num_samples, len(unique_images)), replace=False)
    
#     print(f"\nğŸ§ª Testing {len(selected)} samples with PHASE 1 forward...")
    
#     results = []
#     for image_id in selected:
#         image_path = os.path.join(image_dir, f"{image_id}.png")
#         if not os.path.exists(image_path):
#             continue
        
#         probs, predictions = predict_phase1(model, image_path, device)
#         gt_labels = get_ground_truth(image_id, csv_path)
#         print_results(image_id, probs, predictions, gt_labels)
        
#         exact_match = np.array_equal(predictions, gt_labels.astype(int))
#         element_acc = (predictions == gt_labels.astype(int)).mean()
#         results.append({'exact_match': exact_match, 'element_acc': element_acc})
    
#     if results:
#         print("\n" + "=" * 70)
#         print("ğŸ“ˆ SUMMARY")
#         print("=" * 70)
#         exact_matches = sum(r['exact_match'] for r in results)
#         avg_acc = np.mean([r['element_acc'] for r in results])
#         print(f"Exact Match: {exact_matches}/{len(results)} ({exact_matches/len(results)*100:.1f}%)")
#         print(f"Avg Element Accuracy: {avg_acc*100:.1f}%")


# # =================== MAIN ===================
# if __name__ == "__main__":
#     device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
#     print(f"ğŸ–¥ï¸� Device: {device}")
    
#     # Load model
#     print(f"\nğŸ“‚ Loading model from: {CHECKPOINT_PATH}")
#     model = CSRModel(
#         num_classes=NUM_CLASSES,
#         num_prototypes=NUM_PROTOTYPES,
#         model_name=MODEL_NAME,
#         pretrained=False
#     )
    
#     ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
#     if 'model_state_dict' in ckpt:
#         model.load_state_dict(ckpt['model_state_dict'])
#     else:
#         model.load_state_dict(ckpt)
    
#     model.to(device)
#     model.eval()
#     print("âœ… Model loaded!")
    
#     # Test single image
#     print(f"\nğŸ”� Testing: {IMAGE_PATH}")
#     probs, predictions = predict_phase1(model, IMAGE_PATH, device)
    
#     image_id = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
#     gt_labels = get_ground_truth(image_id, CSV_PATH)
#     print_results(image_id, probs, predictions, gt_labels)
    
#     # Test multiple samples
#     # test_multiple_samples(model, CSV_PATH, IMAGE_DIR, device, num_samples=10)


# """
# Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh CSR Phase 1 trÃªn toÃ n bá»™ táº­p test
# Metrics: Accuracy, Precision, Recall, F1, AUC
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import os
# import timm
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     roc_auc_score, classification_report, multilabel_confusion_matrix
# )

# # =================== Cáº¤U HÃŒNH ===================
# CHECKPOINT_PATH = "/kaggle/input/vin-csr-training/checkpoints/csr_phase1.pth"
# CSV_PATH = "/kaggle/input/png-trainer-9/pytorch/default/1/src/csv/test_split.csv"
# IMAGE_DIR = "/kaggle/input/vindr-image-convert/train_png_384"

# NUM_CLASSES = 14
# NUM_PROTOTYPES = 15
# MODEL_NAME = "densenet121"
# IMG_SIZE = 384
# DEVICE = "cuda"
# THRESHOLD = 0.5
# BATCH_SIZE = 32

# CLASS_NAMES = [
#     'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
#     'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity',
#     'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening',
#     'Pneumothorax', 'Pulmonary fibrosis'
# ]

# # =================== MODEL ===================
# class CSRModel(nn.Module):
#     def __init__(self, num_classes=14, num_prototypes=5, model_name="resnet50", pretrained=True):
#         super().__init__()
#         self.backbone = timm.create_model(
#             model_name, pretrained=pretrained, features_only=True, out_indices=(4,)
#         )
#         feature_info = self.backbone.feature_info.get_dicts()[-1]
#         self.feature_dim = feature_info["num_chs"]
#         self.concept_head = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)
        
#         self.embedding_dim = 128
#         self.projector = nn.Sequential(
#             nn.Linear(self.feature_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, self.embedding_dim)
#         )
#         self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, self.embedding_dim))
#         self.task_head = nn.Linear(num_classes * num_prototypes, num_classes)
#         self.num_classes = num_classes
#         self.num_prototypes = num_prototypes

#     def forward_phase1(self, x):
#         if x.size(1) == 1:
#             x = x.repeat(1, 3, 1, 1)
#         features = self.backbone(x)[0]
#         attn_logits = self.concept_head(features)
#         logits = F.adaptive_avg_pool2d(attn_logits, (1, 1)).view(x.size(0), -1)
#         return logits


# # =================== DATA LOADING ===================
# def load_test_data(csv_path, image_dir):
#     """Load toÃ n bá»™ test data"""
#     df = pd.read_csv(csv_path)
    
#     # Táº¡o ground truth matrix
#     unique_images = df['image_id'].unique()
#     print(f"ğŸ“Š Tá»•ng sá»‘ áº£nh test: {len(unique_images)}")
    
#     image_labels = {}
#     for image_id in unique_images:
#         labels = np.zeros(NUM_CLASSES)
#         rows = df[df['image_id'] == image_id]
#         for _, row in rows.iterrows():
#             class_id = row['class_id']
#             if 0 <= class_id < NUM_CLASSES:
#                 labels[class_id] = 1
#         image_labels[image_id] = labels
    
#     # Lá»�c áº£nh tá»“n táº¡i
#     valid_images = []
#     for image_id in unique_images:
#         image_path = os.path.join(image_dir, f"{image_id}.png")
#         if os.path.exists(image_path):
#             valid_images.append(image_id)
    
#     print(f"âœ… Sá»‘ áº£nh há»£p lá»‡: {len(valid_images)}")
#     return valid_images, image_labels


# def preprocess_image(image_path, target_size=384):
#     """Preprocess single image"""
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         return None
#     image = cv2.resize(image, (target_size, target_size))
#     img_norm = image.astype(np.float32) / 255.0
#     img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
#     return img_tensor


# # =================== EVALUATION ===================
# def evaluate_model(model, image_ids, image_labels, image_dir, device, threshold=0.5):
#     """
#     Ä�Ã¡nh giÃ¡ model trÃªn toÃ n bá»™ táº­p test
#     """
#     model.eval()
    
#     all_probs = []
#     all_preds = []
#     all_labels = []
    
#     print("\nğŸ”„ Ä�ang dá»± Ä‘oÃ¡n...")
#     with torch.no_grad():
#         for image_id in tqdm(image_ids, desc="Evaluating"):
#             image_path = os.path.join(image_dir, f"{image_id}.png")
#             img_tensor = preprocess_image(image_path, IMG_SIZE)
            
#             if img_tensor is None:
#                 continue
            
#             img_tensor = img_tensor.to(device)
#             logits = model.forward_phase1(img_tensor)
#             probs = torch.sigmoid(logits).cpu().numpy()[0]
#             preds = (probs > threshold).astype(int)
            
#             all_probs.append(probs)
#             all_preds.append(preds)
#             all_labels.append(image_labels[image_id])
    
#     all_probs = np.array(all_probs)
#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)
    
#     return all_probs, all_preds, all_labels


# def compute_metrics(all_probs, all_preds, all_labels):
#     """
#     TÃ­nh toÃ¡n cÃ¡c metrics Ä‘Ã¡nh giÃ¡
#     """
#     results = {}
    
#     # ===== 1. SAMPLE-WISE METRICS =====
#     # Exact Match Ratio (Subset Accuracy)
#     exact_match = np.all(all_preds == all_labels, axis=1).mean()
#     results['exact_match_ratio'] = exact_match
    
#     # Element-wise Accuracy
#     element_acc = (all_preds == all_labels).mean()
#     results['element_accuracy'] = element_acc
    
#     # ===== 2. LABEL-WISE METRICS (Macro/Micro) =====
#     # Precision, Recall, F1
#     results['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
#     results['precision_micro'] = precision_score(all_labels, all_preds, average='micro', zero_division=0)
#     results['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#     results['recall_micro'] = recall_score(all_labels, all_preds, average='micro', zero_division=0)
#     results['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
#     results['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
#     # ===== 3. AUC-ROC =====
#     try:
#         results['auc_macro'] = roc_auc_score(all_labels, all_probs, average='macro')
#         results['auc_micro'] = roc_auc_score(all_labels, all_probs, average='micro')
#     except ValueError as e:
#         print(f"âš ï¸� Cannot compute AUC: {e}")
#         results['auc_macro'] = None
#         results['auc_micro'] = None
    
#     # ===== 4. PER-CLASS METRICS =====
#     per_class = {}
#     for i, class_name in enumerate(CLASS_NAMES):
#         y_true = all_labels[:, i]
#         y_pred = all_preds[:, i]
#         y_prob = all_probs[:, i]
        
#         # Ä�áº¿m sá»‘ lÆ°á»£ng
#         n_positive = int(y_true.sum())
#         n_pred_positive = int(y_pred.sum())
        
#         # TÃ­nh metrics
#         acc = accuracy_score(y_true, y_pred)
#         prec = precision_score(y_true, y_pred, zero_division=0)
#         rec = recall_score(y_true, y_pred, zero_division=0)
#         f1 = f1_score(y_true, y_pred, zero_division=0)
        
#         try:
#             auc = roc_auc_score(y_true, y_prob) if n_positive > 0 and n_positive < len(y_true) else None
#         except:
#             auc = None
        
#         per_class[class_name] = {
#             'n_positive': n_positive,
#             'n_pred_positive': n_pred_positive,
#             'accuracy': acc,
#             'precision': prec,
#             'recall': rec,
#             'f1': f1,
#             'auc': auc
#         }
    
#     results['per_class'] = per_class
#     return results


# def print_evaluation_results(results, all_labels):
#     """In káº¿t quáº£ Ä‘Ã¡nh giÃ¡"""
    
#     print("\n" + "=" * 80)
#     print("ğŸ“Š Káº¾T QUáº¢ Ä�Ã�NH GIÃ� MÃ” HÃŒNH CSR PHASE 1")
#     print("=" * 80)
    
#     print(f"\nğŸ“ˆ OVERALL METRICS:")
#     print("-" * 50)
#     print(f"  Exact Match Ratio:     {results['exact_match_ratio']*100:>6.2f}%")
#     print(f"  Element Accuracy:      {results['element_accuracy']*100:>6.2f}%")
#     print(f"  Precision (Macro):     {results['precision_macro']*100:>6.2f}%")
#     print(f"  Precision (Micro):     {results['precision_micro']*100:>6.2f}%")
#     print(f"  Recall (Macro):        {results['recall_macro']*100:>6.2f}%")
#     print(f"  Recall (Micro):        {results['recall_micro']*100:>6.2f}%")
#     print(f"  F1-Score (Macro):      {results['f1_macro']*100:>6.2f}%")
#     print(f"  F1-Score (Micro):      {results['f1_micro']*100:>6.2f}%")
#     if results['auc_macro']:
#         print(f"  AUC-ROC (Macro):       {results['auc_macro']*100:>6.2f}%")
#         print(f"  AUC-ROC (Micro):       {results['auc_micro']*100:>6.2f}%")
    
#     print(f"\nğŸ“‹ PER-CLASS METRICS:")
#     print("-" * 100)
#     header = f"{'Class':<25} {'N_GT':>6} {'N_Pred':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
#     print(header)
#     print("-" * 100)
    
#     per_class = results['per_class']
#     for class_name in CLASS_NAMES:
#         m = per_class[class_name]
#         auc_str = f"{m['auc']*100:>6.1f}%" if m['auc'] else "   N/A"
#         print(f"{class_name:<25} {m['n_positive']:>6} {m['n_pred_positive']:>7} "
#               f"{m['accuracy']*100:>6.1f}% {m['precision']*100:>6.1f}% "
#               f"{m['recall']*100:>6.1f}% {m['f1']*100:>6.1f}% {auc_str}")
    
#     print("-" * 100)
    
#     # Thá»‘ng kÃª thÃªm
#     print(f"\nğŸ“Š THá»�NG KÃŠ Bá»” SUNG:")
#     print("-" * 50)
#     total_positive = int(all_labels.sum())
#     total_samples = len(all_labels)
#     print(f"  Tá»•ng sá»‘ máº«u test:      {total_samples}")
#     print(f"  Tá»•ng sá»‘ nhÃ£n dÆ°Æ¡ng:    {total_positive}")
#     print(f"  Trung bÃ¬nh nhÃ£n/áº£nh:   {total_positive/total_samples:.2f}")


# def compute_simple_accuracy(all_preds, all_labels):
#     """
#     Ä�Æ¡n giáº£n: Ä�áº¿m sá»‘ nhÃ£n dá»± Ä‘oÃ¡n Ä‘Ãºng
#     """
#     # Tá»•ng sá»‘ nhÃ£n Ä‘Ãºng
#     correct = (all_preds == all_labels).sum()
#     total = all_preds.size
    
#     # Ä�áº¿m theo tá»«ng loáº¡i
#     true_positive = ((all_preds == 1) & (all_labels == 1)).sum()
#     true_negative = ((all_preds == 0) & (all_labels == 0)).sum()
#     false_positive = ((all_preds == 1) & (all_labels == 0)).sum()
#     false_negative = ((all_preds == 0) & (all_labels == 1)).sum()
    
#     print("\n" + "=" * 60)
#     print("ğŸ�¯ Ä�Ã�NH GIÃ� Ä�Æ N GIáº¢N: Sá»� LÆ¯á»¢NG NHÃƒN Dá»° Ä�OÃ�N Ä�ÃšNG")
#     print("=" * 60)
#     print(f"  âœ… Tá»•ng sá»‘ nhÃ£n Ä‘Ãºng:           {correct:,} / {total:,}")
#     print(f"  ğŸ“Š Accuracy:                    {correct/total*100:.2f}%")
#     print("-" * 60)
#     print(f"  âœ… True Positive (Bá»‡nh â†’ Bá»‡nh): {true_positive:,}")
#     print(f"  âœ… True Negative (KBá»‡nh â†’ KBá»‡nh): {true_negative:,}")
#     print(f"  â�Œ False Positive (KBá»‡nh â†’ Bá»‡nh): {false_positive:,}")
#     print(f"  â�Œ False Negative (Bá»‡nh â†’ KBá»‡nh): {false_negative:,}")
#     print("=" * 60)
    
#     return {
#         'total_correct': int(correct),
#         'total_labels': int(total),
#         'accuracy': correct / total,
#         'true_positive': int(true_positive),
#         'true_negative': int(true_negative),
#         'false_positive': int(false_positive),
#         'false_negative': int(false_negative)
#     }


# # =================== MAIN ===================
# if __name__ == "__main__":
#     device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
#     print(f"ğŸ–¥ï¸� Device: {device}")
    
#     # Load model
#     print(f"\nğŸ“‚ Loading model from: {CHECKPOINT_PATH}")
#     model = CSRModel(
#         num_classes=NUM_CLASSES,
#         num_prototypes=NUM_PROTOTYPES,
#         model_name=MODEL_NAME,
#         pretrained=False
#     )
    
#     ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
#     if 'model_state_dict' in ckpt:
#         model.load_state_dict(ckpt['model_state_dict'])
#     else:
#         model.load_state_dict(ckpt)
    
#     model.to(device)
#     model.eval()
#     print("âœ… Model loaded!")
    
#     # Load test data
#     image_ids, image_labels = load_test_data(CSV_PATH, IMAGE_DIR)
    
#     # Evaluate
#     all_probs, all_preds, all_labels = evaluate_model(
#         model, image_ids, image_labels, IMAGE_DIR, device, threshold=THRESHOLD
#     )
    
#     # ===== Ä�Æ N GIáº¢N: Ä�áº¿m sá»‘ nhÃ£n Ä‘Ãºng =====
#     simple_results = compute_simple_accuracy(all_preds, all_labels)
    
#     # ===== CHI TIáº¾T: Táº¥t cáº£ metrics =====
#     results = compute_metrics(all_probs, all_preds, all_labels)
#     print_evaluation_results(results, all_labels)
    
#     # LÆ°u káº¿t quáº£
#     print("\nğŸ’¾ Saving results...")
#     np.savez('evaluation_results.npz', 
#              probs=all_probs, 
#              preds=all_preds, 
#              labels=all_labels)
#     print("âœ… Done!")


# """
# Visualize Concept Activation Maps (CAMs) tá»« model Phase 1
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import os
# import timm

# # =================== Cáº¤U HÃŒNH ===================

# # Checkpoint vÃ  áº£nh
# CHECKPOINT_PATH = "/kaggle/input/vin-csr-training/checkpoints/csr_phase1.pth"
# IMAGE_PATH = "/kaggle/input/vindr-image-convert/train_png_384/9a5094b2563a1ef3ff50dc5c7ff71345.png"
# SAVE_DIR = "/kaggle/working/cam_outputs"

# # Model config
# NUM_CLASSES = 14
# NUM_PROTOTYPES = 15
# MODEL_NAME = "densenet121"
# IMG_SIZE = 384
# DEVICE = "cuda"

# # Visualization config
# TOP_K = 3  # Sá»‘ CAM hiá»ƒn thá»‹ (top-k theo probability)
# CAM_THRESHOLD = 0.6  # NgÆ°á»¡ng Ä‘á»ƒ lá»�c CAM
# ALPHA = 0.5  # Ä�á»™ trong suá»‘t overlay

# # TÃªn cÃ¡c bá»‡nh
# CLASS_NAMES = [
#     'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
#     'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity',
#     'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening',
#     'Pneumothorax', 'Pulmonary fibrosis'
# ]

# # ================================================


# class CSRModel(nn.Module):
#     def __init__(self, num_classes=14, num_prototypes=5, model_name="resnet50", pretrained=True):
#         super().__init__()
        
#         self.backbone = timm.create_model(
#             model_name, pretrained=pretrained, features_only=True, out_indices=(4,)
#         )
#         feature_info = self.backbone.feature_info.get_dicts()[-1]
#         self.feature_dim = feature_info["num_chs"]
#         self.concept_head = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)
        
#         self.embedding_dim = 128
#         self.projector = nn.Sequential(
#             nn.Linear(self.feature_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, self.embedding_dim)
#         )
        
#         self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, self.embedding_dim))
#         self.task_head = nn.Linear(num_classes * num_prototypes, num_classes)
#         self.num_classes = num_classes
#         self.num_prototypes = num_prototypes

#     def get_features_and_cam(self, x):
#         if x.size(1) == 1:
#             x = x.repeat(1, 3, 1, 1)
#         features = self.backbone(x)[0]
#         attn_logits = self.concept_head(features)
#         return features, attn_logits
    
#     def forward_phase1(self, x):
#         """Forward cho Phase 1 - dÃ¹ng GAP trÃªn CAM"""
#         _, attn_logits = self.get_features_and_cam(x)
#         logits = F.adaptive_avg_pool2d(attn_logits, (1, 1)).view(x.size(0), -1)
#         return {"logits": logits, "attn_maps": attn_logits}


# def load_model(checkpoint_path, device):
#     """Load model tá»« checkpoint"""
#     model = CSRModel(
#         num_classes=NUM_CLASSES,
#         num_prototypes=NUM_PROTOTYPES,
#         model_name=MODEL_NAME,
#         pretrained=False
#     )
    
#     ckpt = torch.load(checkpoint_path, map_location=device)
#     if 'model_state_dict' in ckpt:
#         model.load_state_dict(ckpt['model_state_dict'])
#     else:
#         model.load_state_dict(ckpt)
    
#     model.to(device)
#     model.eval()
#     return model


# def preprocess_image(image_path, target_size=384):
#     """Ä�á»�c vÃ  tiá»�n xá»­ lÃ½ áº£nh"""
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"Cannot read image: {image_path}")
    
#     original_image = image.copy()
#     image = cv2.resize(image, (target_size, target_size))
#     img_norm = image.astype(np.float32) / 255.0
#     img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    
#     return img_tensor, image, original_image


# def get_cam_overlay(original_img, cam, threshold=0.3, alpha=0.5, colormap=cv2.COLORMAP_JET):
#     """
#     Táº¡o overlay CAM lÃªn áº£nh gá»‘c
    
#     Args:
#         original_img: áº¢nh grayscale (H, W)
#         cam: CAM array (H_cam, W_cam)
#         threshold: NgÆ°á»¡ng Ä‘á»ƒ lá»�c CAM
#         alpha: Ä�á»™ trong suá»‘t
#         colormap: OpenCV colormap
    
#     Returns:
#         overlay: áº¢nh RGB vá»›i CAM overlay
#     """
#     # Resize CAM vá»� kÃ­ch thÆ°á»›c áº£nh gá»‘c
#     cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    
#     # Normalize CAM vá»� [0, 1]
#     cam_min, cam_max = cam_resized.min(), cam_resized.max()
#     if cam_max - cam_min > 1e-8:
#         cam_norm = (cam_resized - cam_min) / (cam_max - cam_min)
#     else:
#         cam_norm = np.zeros_like(cam_resized)
    
#     # Ã�p dá»¥ng threshold
#     cam_thresholded = np.where(cam_norm > threshold, cam_norm, 0)
    
#     # Re-normalize sau threshold
#     if cam_thresholded.max() > 0:
#         cam_thresholded = cam_thresholded / cam_thresholded.max()
    
#     # Táº¡o heatmap
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_thresholded), colormap)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
#     # Chuyá»ƒn áº£nh gá»‘c sang RGB
#     img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    
#     # Táº¡o mask Ä‘á»ƒ chá»‰ overlay vÃ¹ng cÃ³ CAM
#     mask = (cam_thresholded > 0).astype(np.float32)
#     mask = np.stack([mask] * 3, axis=-1)
    
#     # Blend
#     overlay = img_rgb.astype(np.float32)
#     overlay = overlay * (1 - mask * alpha) + heatmap.astype(np.float32) * (mask * alpha)
#     overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
#     return overlay, cam_norm


# def visualize_all_cams(model, image_path, device, top_k=5, threshold=0.3, save_path=None):
#     """
#     Visualize CAM cho táº¥t cáº£ classes (hoáº·c top-k)
#     """
#     # Load vÃ  preprocess áº£nh
#     img_tensor, resized_img, original_img = preprocess_image(image_path, IMG_SIZE)
#     img_tensor = img_tensor.to(device)
    
#     # Inference
#     with torch.no_grad():
#         outputs = model.forward_phase1(img_tensor)
#         logits = outputs['logits'][0]
#         cams = outputs['attn_maps'][0]  # [K, H, W]
#         probs = torch.sigmoid(logits).cpu().numpy()
    
#     # Láº¥y top-k classes theo probability
#     top_indices = np.argsort(probs)[::-1][:top_k]
    
#     # Táº¡o figure
#     n_cols = min(3, top_k + 1)
#     n_rows = (top_k + 1 + n_cols - 1) // n_cols
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
#     axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
#     # áº¢nh gá»‘c
#     axes[0].imshow(resized_img, cmap='gray')
#     axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold')
#     axes[0].axis('off')
    
#     # ThÃªm text prediction
#     pred_text = "Predictions:\n"
#     for idx in top_indices[:5]:
#         pred_text += f"{CLASS_NAMES[idx]}: {probs[idx]*100:.1f}%\n"
#     axes[0].text(0.02, 0.02, pred_text, transform=axes[0].transAxes,
#                  fontsize=9, verticalalignment='bottom',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
#     # CAM cho tá»«ng class
#     for i, idx in enumerate(top_indices):
#         ax = axes[i + 1]
        
#         cam = cams[idx].cpu().numpy()
#         overlay, cam_norm = get_cam_overlay(resized_img, cam, threshold=threshold)
        
#         ax.imshow(overlay)
#         ax.set_title(f'{CLASS_NAMES[idx]}\nProb: {probs[idx]*100:.1f}%', fontsize=12)
#         ax.axis('off')
        
#         # ThÃªm colorbar nhá»�
#         # coverage = (cam_norm > threshold).sum() / cam_norm.size * 100
#         # ax.text(0.02, 0.02, f'Coverage: {coverage:.1f}%', transform=ax.transAxes,
#         #         fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
#     # áº¨n axes thá»«a
#     for j in range(len(top_indices) + 1, len(axes)):
#         axes[j].axis('off')
    
#     plt.suptitle(f'Concept Activation Maps (threshold={threshold})', fontsize=16, fontweight='bold')
#     plt.tight_layout()
    
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"ğŸ’¾ Saved to: {save_path}")
    
#     plt.show()
    
#     return probs, cams


# def visualize_single_cam(model, image_path, class_idx, device, threshold=0.3, save_path=None):
#     """
#     Visualize CAM cho 1 class cá»¥ thá»ƒ vá»›i chi tiáº¿t hÆ¡n
#     """
#     img_tensor, resized_img, _ = preprocess_image(image_path, IMG_SIZE)
#     img_tensor = img_tensor.to(device)
    
#     with torch.no_grad():
#         outputs = model.forward_phase1(img_tensor)
#         logits = outputs['logits'][0]
#         cams = outputs['attn_maps'][0]
#         probs = torch.sigmoid(logits).cpu().numpy()
    
#     cam = cams[class_idx].cpu().numpy()
#     prob = probs[class_idx]
    
#     # Táº¡o figure vá»›i nhiá»�u views
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
#     # Row 1: Original, CAM raw, Overlay
#     axes[0, 0].imshow(resized_img, cmap='gray')
#     axes[0, 0].set_title('Original', fontsize=12)
#     axes[0, 0].axis('off')
    
#     # CAM raw (resize vá»� kÃ­ch thÆ°á»›c áº£nh)
#     cam_resized = cv2.resize(cam, (resized_img.shape[1], resized_img.shape[0]))
#     im = axes[0, 1].imshow(cam_resized, cmap='jet')
#     axes[0, 1].set_title(f'CAM (raw logits)\nmin={cam.min():.2f}, max={cam.max():.2f}', fontsize=12)
#     axes[0, 1].axis('off')
#     plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
#     # Overlay
#     overlay, cam_norm = get_cam_overlay(resized_img, cam, threshold=threshold)
#     axes[0, 2].imshow(overlay)
#     axes[0, 2].set_title(f'Overlay (threshold={threshold})', fontsize=12)
#     axes[0, 2].axis('off')
    
#     # Row 2: Different thresholds
#     thresholds = [0.0, 0.3, 0.5]
#     for i, thresh in enumerate(thresholds):
#         overlay_t, _ = get_cam_overlay(resized_img, cam, threshold=thresh)
#         axes[1, i].imshow(overlay_t)
#         coverage = (cam_norm > thresh).sum() / cam_norm.size * 100
#         axes[1, i].set_title(f'Threshold={thresh}\nCoverage: {coverage:.1f}%', fontsize=11)
#         axes[1, i].axis('off')
    
#     plt.suptitle(f'{CLASS_NAMES[class_idx]} - Probability: {prob*100:.1f}%', 
#                  fontsize=16, fontweight='bold')
#     plt.tight_layout()
    
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"ğŸ’¾ Saved to: {save_path}")
    
#     plt.show()


# def visualize_cam_comparison(model, image_path, device, save_path=None):
#     """
#     So sÃ¡nh CAM cá»§a táº¥t cáº£ 14 classes trong 1 figure
#     """
#     img_tensor, resized_img, _ = preprocess_image(image_path, IMG_SIZE)
#     img_tensor = img_tensor.to(device)
    
#     with torch.no_grad():
#         outputs = model.forward_phase1(img_tensor)
#         logits = outputs['logits'][0]
#         cams = outputs['attn_maps'][0]
#         probs = torch.sigmoid(logits).cpu().numpy()
    
#     # Figure 4x4 (1 original + 14 CAMs + 1 empty)
#     fig, axes = plt.subplots(4, 4, figsize=(16, 16))
#     axes = axes.flatten()
    
#     # Original
#     axes[0].imshow(resized_img, cmap='gray')
#     axes[0].set_title('Original', fontsize=10, fontweight='bold')
#     axes[0].axis('off')
    
#     # 14 CAMs
#     for i in range(NUM_CLASSES):
#         ax = axes[i + 1]
#         cam = cams[i].cpu().numpy()
#         overlay, _ = get_cam_overlay(resized_img, cam, threshold=CAM_THRESHOLD)
        
#         ax.imshow(overlay)
#         ax.set_title(f'{CLASS_NAMES[i]}\n{probs[i]*100:.1f}%', fontsize=9)
#         ax.axis('off')
        
#         # Ä�Ã¡nh dáº¥u náº¿u prob > 50%
#         if probs[i] > 0.5:
#             ax.patch.set_edgecolor('red')
#             ax.patch.set_linewidth(3)
    
#     # áº¨n axis cuá»‘i
#     axes[15].axis('off')
    
#     plt.suptitle(f'All 14 Concept Activation Maps', fontsize=16, fontweight='bold')
#     plt.tight_layout()
    
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"ğŸ’¾ Saved to: {save_path}")
    
#     plt.show()


# # =================== MAIN ===================

# if __name__ == "__main__":
#     device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
#     print(f"ğŸ–¥ï¸� Device: {device}")
    
#     # Load model
#     print(f"\nğŸ“‚ Loading model from: {CHECKPOINT_PATH}")
#     model = load_model(CHECKPOINT_PATH, device)
#     print("âœ… Model loaded!")
    
#     # Táº¡o output directory
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     image_id = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    
#     # 1. Visualize Top-K CAMs
#     print(f"\nğŸ”� Visualizing Top-{TOP_K} CAMs...")
#     visualize_all_cams(
#         model, IMAGE_PATH, device,
#         top_k=TOP_K,
#         threshold=CAM_THRESHOLD,
#         save_path=os.path.join(SAVE_DIR, f"{image_id}_top{TOP_K}_cams.png")
#     )
    
#     # 2. Visualize táº¥t cáº£ 14 CAMs
#     # print(f"\nğŸ”� Visualizing all 14 CAMs...")
#     # visualize_cam_comparison(
#     #     model, IMAGE_PATH, device,
#     #     save_path=os.path.join(SAVE_DIR, f"{image_id}_all_cams.png")
#     # )
    
#     # 3. Visualize chi tiáº¿t 1 class (vÃ­ dá»¥: Cardiomegaly - index 3)
#     # print(f"\nğŸ”� Visualizing detailed CAM for Cardiomegaly...")
#     # visualize_single_cam(
#     #     model, IMAGE_PATH, class_idx=3, device=device,
#     #     threshold=CAM_THRESHOLD,
#     #     save_path=os.path.join(SAVE_DIR, f"{image_id}_cardiomegaly_detail.png")
#     # )

