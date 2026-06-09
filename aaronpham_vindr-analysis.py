# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import os
# import pydicom # ThÆ° viá»‡n má»›i Ä‘á»ƒ Ä‘á»�c DICOM
# from PIL import Image # Váº«n dÃ¹ng Ä‘á»ƒ resize vÃ  chuyá»ƒn Ä‘á»•i (tÃ¹y chá»�n)

# # --- Thiáº¿t láº­p Háº±ng sá»‘ ---
# # Giáº£ sá»­ báº¡n cÃ³ 14 lá»›p bá»‡nh lÃ½ cho VINDR-CXR
# NUM_CLASSES = 14
# # KÃ­ch thÆ°á»›c áº£nh Ä‘áº§u vÃ o cho mÃ´ hÃ¬nh
# TARGET_SIZE = 512
# # KÃ­ch thÆ°á»›c cá»§a Attention Map (thÆ°á»�ng nhá»� hÆ¡n áº£nh Ä‘á»ƒ giáº£m táº£i tÃ­nh toÃ¡n)
# MAP_SIZE = 64
# # Ä�Æ°á»�ng dáº«n (giáº£ Ä‘á»‹nh)
# IMAGE_DIR = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train'
# ANNOTATION_CSV = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv'


# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import os
# import pydicom
# from PIL import Image

# class VINDRCXRDataset(Dataset):
#     def __init__(self, csv_file, image_dir, num_classes, target_size, map_size):
#         """
#         Khá»Ÿi táº¡o Dataset, sá»­ dá»¥ng class_id gá»‘c tá»« CSV Ä‘á»ƒ táº¡o mapping.
#         """
#         self.data_df = pd.read_csv(csv_file)
#         self.image_dir = image_dir
#         self.target_size = target_size
#         self.map_size = map_size

#         # Táº­p há»£p cÃ¡c ID áº£nh duy nháº¥t
#         self.image_ids = self.data_df['image_id'].unique().tolist()
        
#         # Láº¥y táº¥t cáº£ cÃ¡c cáº·p (tÃªn lá»›p, ID gá»‘c) duy nháº¥t
#         class_mapping_data = self.data_df[['class_name', 'class_id']].drop_duplicates()
        
#         # âš ï¸� PHáº¦N Sá»¬A Lá»–I: Táº¡o mapping tá»« TÃªn Lá»›p -> ID Gá»‘c (dá»±a vÃ o cá»™t class_id)
#         self.class_to_id = {}
#         pathology_class_count = 0
        
#         for _, row in class_mapping_data.iterrows():
#             name = row['class_name']
#             original_id = row['class_id']
            
#             # Chá»‰ Ã¡nh xáº¡ 14 lá»›p bá»‡nh lÃ½ (ID gá»‘c 0 Ä‘áº¿n 13).
#             # Lá»›p "No finding" thÆ°á»�ng cÃ³ ID 14 vÃ  bá»‹ loáº¡i trá»«.
#             if name.lower() != 'no finding' and 0 <= original_id <= 13: 
#                  self.class_to_id[name] = int(original_id) # Ä�áº£m báº£o ID lÃ  sá»‘ nguyÃªn
#                  pathology_class_count += 1
            
#         # Thiáº¿t láº­p sá»‘ lÆ°á»£ng lá»›p chÃ­nh xÃ¡c
#         self.num_classes = pathology_class_count
#         if self.num_classes != 14:
#              print(f"Cáº£nh bÃ¡o: Sá»‘ lÆ°á»£ng lá»›p bá»‡nh lÃ½ sau Ã¡nh xáº¡ lÃ  {self.num_classes} thay vÃ¬ 14.")
             
#     def __len__(self):
#         return len(self.image_ids)
    
#     def __getitem__(self, idx):
#         image_id = self.image_ids[idx]
#         img_annotations = self.data_df[self.data_df['image_id'] == image_id]
        
#         # --- 1. Load vÃ  Tiá»�n xá»­ lÃ½ áº¢nh (X) ---
#         dicom_path = os.path.join(self.image_dir, f'{image_id}.dicom')
        
#         try:
#             dicom = pydicom.dcmread(dicom_path)
#             pixel_array = dicom.pixel_array.astype(np.float32)
#         except Exception as e:
#             return None, None, None, image_id 

#         # Láº¥y kÃ­ch thÆ°á»›c áº£nh gá»‘c vÃ  kiá»ƒm tra
#         original_h, original_w = pixel_array.shape 
#         if original_w <= 0 or original_h <= 0:
#             return None, None, None, image_id 
        
#         # Chuáº©n hÃ³a, resize áº£nh vÃ  chuyá»ƒn sang Tensor
#         pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
#         image = Image.fromarray(np.uint8(pixel_array * 255))
#         image = image.resize((self.target_size, self.target_size))
#         image = np.array(image, dtype=np.float32) / 255.0
#         image = torch.from_numpy(image).unsqueeze(0) 

#         # --- 2. Táº¡o Classification Label (Y_cls) ---
#         cls_label = torch.zeros(self.num_classes, dtype=torch.float32) 
#         present_classes = img_annotations['class_name'].unique()
        
#         for cls_name in present_classes:
#             cls_id = self.class_to_id.get(cls_name)
            
#             if cls_id is not None: 
#                 cls_label[cls_id] = 1.0
                
#         # --- 3. Táº¡o Attention Maps (Y_map) cho tá»«ng lá»›p ---
#         attn_maps = torch.zeros((self.num_classes, self.map_size, self.map_size), dtype=torch.float32)
        
#         # Duyá»‡t qua Táº¤T Cáº¢ cÃ¡c bbox 
#         for _, row in img_annotations.iterrows():
#             cls_name = row['class_name']
#             cls_id = self.class_to_id.get(cls_name)
            
#             # Bá»� QUA 'No finding' VÃ€ Dá»® LIá»†U THIáº¾U
#             if cls_id is None: 
#                 continue
            
#             if pd.isna(row[['x_min', 'y_min', 'x_max', 'y_max']]).any():
#                 continue 

#             # Tá»�a Ä‘á»™ bbox (chuáº©n hÃ³a vá»� kÃ­ch thÆ°á»›c Map_Size)
#             x_min = int(row['x_min'] * (self.map_size / original_w))
#             y_min = int(row['y_min'] * (self.map_size / original_h))
#             x_max = int(row['x_max'] * (self.map_size / original_w))
#             y_max = int(row['y_max'] * (self.map_size / original_h))

#             # Giá»›i háº¡n tá»�a Ä‘á»™
#             x_min = max(0, x_min)
#             y_min = max(0, y_min)
#             x_max = min(self.map_size, x_max)
#             y_max = min(self.map_size, y_max)
            
#             # Váº½ bbox lÃªn slice (Map) tÆ°Æ¡ng á»©ng. 
#             if x_max > x_min and y_max > y_min:
#                 attn_maps[cls_id, y_min:y_max, x_min:x_max] = 1.0
                
#         return image, cls_label, attn_maps, image_id


# # --- Khá»Ÿi táº¡o vÃ  Test ---
# dataset = VINDRCXRDataset(
#     csv_file=ANNOTATION_CSV, 
#     image_dir=IMAGE_DIR, 
#     num_classes=NUM_CLASSES, 
#     target_size=TARGET_SIZE, 
#     map_size=MAP_SIZE
# )

# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# print(dataloader)

# # Láº¥y má»™t batch (item) Ä‘áº§u tiÃªn
# image_batch, cls_label_batch, attn_maps_batch, image_id_batch = next(iter(dataloader))

# print("--- Káº¾T QUáº¢ Cá»¦A DATALOADER ---")
# print(f"áº¢nh ID: {image_id_batch[0]}")
# print(f"1. KÃ­ch thÆ°á»›c áº¢nh (X): {image_batch.shape}")
# print(f"2. Classification Label (Y_cls): {cls_label_batch.shape}")
# print(f"   CÃ¡c lá»›p cÃ³ máº·t (giÃ¡ trá»‹ = 1): {torch.nonzero(cls_label_batch)}")
# print(f"3. Attention Maps (Y_map): {attn_maps_batch.shape}")
# print("-" * 30)

# # Láº¥y káº¿t quáº£ item Ä‘áº§u tiÃªn (Batch size = 1)
# image = image_batch[0]
# cls_label = cls_label_batch[0]
# attn_maps = attn_maps_batch[0]

# # --- PHÃ‚N TÃ�CH VÃ€ TRá»°C QUAN HÃ“A LOGIC ---

# # 1. Classification Label (Y_cls)
# print("\nğŸ”¥ Y_cls (NhÃ£n PhÃ¢n loáº¡i):")
# print(cls_label)
# # Vá»›i img_001, giÃ¡ trá»‹ 1.0 sáº½ á»Ÿ vá»‹ trÃ­ [0] (Class_0) vÃ  [1] (Class_1)

# # 2. Attention Maps (Y_map) - Kiá»ƒm tra lá»›p 0 (trÃ¹ng láº·p) vÃ  lá»›p 1 (duy nháº¥t)
# print("\nğŸ”¥ Y_map (Heatmaps):")
# # Kiá»ƒm tra Heatmap cho Class_0 (index 0)
# attn_map_class_0 = attn_maps[0]
# print(f"Ä�á»™ lá»›n trung bÃ¬nh cá»§a Map Class 0: {attn_map_class_0.mean():.4f}")
# print(f"GiÃ¡ trá»‹ max cá»§a Map Class 0: {attn_map_class_0.max():.4f}")

# # Kiá»ƒm tra Heatmap cho Class_1 (index 1)
# attn_map_class_1 = attn_maps[1]
# print(f"Ä�á»™ lá»›n trung bÃ¬nh cá»§a Map Class 1: {attn_map_class_1.mean():.4f}")
# print(f"GiÃ¡ trá»‹ max cá»§a Map Class 1: {attn_map_class_1.max():.4f}")

# # Kiá»ƒm tra má»™t lá»›p khÃ´ng cÃ³ máº·t (vÃ­ dá»¥ Class 5)
# attn_map_class_5 = attn_maps[5]
# print(f"Ä�á»™ lá»›n trung bÃ¬nh cá»§a Map Class 5: {attn_map_class_5.mean():.4f}")
# # GiÃ¡ trá»‹ trung bÃ¬nh vÃ  max cá»§a map nÃ y pháº£i lÃ  0 vÃ¬ nÃ³ khÃ´ng cÃ³ bbox

# # --- HÆ¯á»šNG DáºªN TRá»°C QUAN HÃ“A ---

# # Báº¡n cÃ³ thá»ƒ dÃ¹ng thÆ° viá»‡n matplotlib Ä‘á»ƒ trá»±c quan hÃ³a cÃ¡c map nÃ y:
# # 
# # import matplotlib.pyplot as plt
# # 
# # def visualize_map(img_tensor, map_tensor, cls_id):
# #     # Chuáº©n bá»‹ áº£nh vÃ  map
# #     img_np = img_tensor.permute(1, 2, 0).cpu().numpy() # CWH -> HWC
# #     map_np = map_tensor.cpu().numpy()
# #     
# #     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# #     ax.imshow(img_np)
# #     
# #     # Resize map vá»� kÃ­ch thÆ°á»›c áº£nh Ä‘á»ƒ chá»“ng lÃªn (interpolation)
# #     # DÃ¹ng mask Ä‘á»ƒ chá»‰ hiá»ƒn thá»‹ vÃ¹ng > 0
# #     mask = np.where(map_np > 0, 1.0, np.nan)
# #     ax.imshow(mask, cmap='jet', alpha=0.5, 
# #               extent=[0, TARGET_SIZE, TARGET_SIZE, 0], # Ä�áº£m báº£o Ä‘Ãºng tá»�a Ä‘á»™
# #               interpolation='nearest')
# #               
# #     ax.set_title(f'Attention Map for Class {cls_id}')
# #     plt.show()

# # # Cháº¡y trá»±c quan hÃ³a cho cÃ¡c lá»›p cÃ³ máº·t
# # visualize_map(image, attn_maps[0], 0)
# # visualize_map(image, attn_maps[1], 1)


# # 1. XÃ¡c Ä‘á»‹nh image_id báº¡n muá»‘n test
# TEST_IMAGE_ID = '47ed17dcb2cbeec15182ed335a8b5a9e' 

# # 2. TÃ¬m chá»‰ sá»‘ (index) cá»§a ID Ä‘Ã³ trong dataset
# try:
#     test_idx = dataset.image_ids.index(TEST_IMAGE_ID)
# except ValueError:
#     print(f"Lá»—i: KhÃ´ng tÃ¬m tháº¥y image_id '{TEST_IMAGE_ID}' trong dataset.")
#     test_idx = -1

# if test_idx != -1:
#     print(f"Ä�Ã£ tÃ¬m tháº¥y ID táº¡i chá»‰ sá»‘: {test_idx}")
    
#     # 3. Láº¥y dá»¯ liá»‡u trá»±c tiáº¿p tá»« dataset
#     image, cls_label, attn_maps, image_id = dataset[test_idx]

#     # Kiá»ƒm tra xem máº«u cÃ³ bá»‹ lá»—i (None) vÃ  Ä‘Ã£ Ä‘Æ°á»£c bá»� qua khÃ´ng
#     if image is None:
#         print(f"\nâš ï¸� Máº«u {image_id} bá»‹ lá»—i (DICOM hoáº·c kÃ­ch thÆ°á»›c khÃ´ng há»£p lá»‡) vÃ  Ä‘Ã£ tráº£ vá»� None.")
#     else:
#         print("\n--- Káº¾T QUáº¢ Cá»¦A ITEM Cá»¤ THá»‚ ---")
#         print(f"áº¢nh ID: {image_id}")
#         print(f"1. KÃ­ch thÆ°á»›c áº¢nh (X): {image.shape}")
#         print(f"2. Classification Label (Y_cls): {cls_label.shape}")
        
#         # In ra cÃ¡c lá»›p cÃ³ nhÃ£n 1 (tá»©c lÃ  cÃ³ bá»‡nh lÃ½)
#         active_indices = torch.nonzero(cls_label).squeeze(-1).tolist()
#         if not active_indices:
#             print("   => NhÃ£n bá»‡nh lÃ½: [No Finding]")
#         else:
#             # Náº¿u cáº§n, báº¡n cÃ³ thá»ƒ in ra tÃªn lá»›p báº±ng cÃ¡ch Ä‘áº£o ngÆ°á»£c self.class_to_id
#             id_to_class = {v: k for k, v in dataset.class_to_id.items()}
#             active_classes = [id_to_class[i] for i in active_indices]
#             print(f"   => NhÃ£n bá»‡nh lÃ½: {active_classes}")

#         print(f"3. Attention Maps (Y_map): {attn_maps.shape}")
#         print("-" * 30)

#         # Trá»±c quan hÃ³a káº¿t quáº£ cho má»™t sá»‘ Map quan trá»�ng
#         print("\nğŸ”¥ PHÃ‚N TÃ�CH HEATMAPS:")
        
#         # In ra thÃ´ng sá»‘ cho cÃ¡c lá»›p cÃ³ nhÃ£n 1.0
#         if active_indices:
#             for cls_id in active_indices:
#                 map_i = attn_maps[cls_id]
#                 cls_name = id_to_class[cls_id]
#                 print(f"   - Class ID {cls_id} ({cls_name}): Avg={map_i.mean():.6f}, Max={map_i.max():.6f}")
#         else:
#             # Náº¿u lÃ  No Finding (táº¥t cáº£ cÃ¡c nhÃ£n Ä‘á»�u 0)
#             map_0 = attn_maps[0]
#             print(f"   - Map Class 0 (Kiá»ƒm tra): Avg={map_0.mean():.6f}, Max={map_0.max():.6f} (Pháº£i báº±ng 0)")




# import matplotlib.pyplot as plt
# import torch.nn.functional as F

# def visualize_maps(image_tensor, attn_maps_tensor, active_indices, id_to_class, target_size):
#     """
#     Trá»±c quan hÃ³a áº£nh gá»‘c vÃ  cÃ¡c Attention Map chá»“ng lÃªn nhau cho cÃ¡c lá»›p hoáº¡t Ä‘á»™ng.

#     Args:
#         image_tensor (torch.Tensor): áº¢nh gá»‘c (1 x 512 x 512).
#         attn_maps_tensor (torch.Tensor): CÃ¡c Heatmap (14 x 64 x 64).
#         active_indices (list): Danh sÃ¡ch cÃ¡c Class ID cÃ³ nhÃ£n = 1.
#         id_to_class (dict): Ã�nh xáº¡ tá»« ID sang TÃªn lá»›p.
#         target_size (int): KÃ­ch thÆ°á»›c áº£nh hiá»ƒn thá»‹ (512).
#     """
#     if not active_indices:
#         print("KhÃ´ng cÃ³ lá»›p bá»‡nh lÃ½ nÃ o Ä‘á»ƒ trá»±c quan hÃ³a (No Finding).")
#         # Chá»‰ hiá»ƒn thá»‹ áº£nh gá»‘c náº¿u khÃ´ng cÃ³ bá»‡nh lÃ½
#         plt.figure(figsize=(6, 6))
#         plt.imshow(image_tensor.squeeze(0).cpu().numpy(), cmap='gray')
#         plt.title("áº¢nh Gá»‘c (No Finding)")
#         plt.axis('off')
#         plt.show()
#         return

#     # XÃ¡c Ä‘á»‹nh sá»‘ lÆ°á»£ng biá»ƒu Ä‘á»“ cáº§n váº½
#     num_maps = len(active_indices)
    
#     # TÃ­nh toÃ¡n bá»‘ cá»¥c (vÃ­ dá»¥: 2x2, 2x3, 3x3...)
#     cols = 3
#     rows = (num_maps + cols - 1) // cols
    
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
#     axes = axes.flatten()

#     for i, cls_id in enumerate(active_indices):
#         ax = axes[i]
#         cls_name = id_to_class[cls_id]
#         map_i = attn_maps_tensor[cls_id].unsqueeze(0).unsqueeze(0) # ThÃªm chiá»�u batch vÃ  channel

#         # 1. Resize Heatmap vá»� kÃ­ch thÆ°á»›c áº£nh gá»‘c (512x512)
#         # Sá»­ dá»¥ng Interpolation Ä‘á»ƒ lÃ m má»‹n Heatmap
#         # 
#         resized_map = F.interpolate(map_i, size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        
#         # 2. Hiá»ƒn thá»‹ áº£nh X-quang gá»‘c (Ä‘Æ¡n kÃªnh)
#         # Bá»� Ä‘i chiá»�u kÃªnh 1 (1 x H x W -> H x W)
#         ax.imshow(image_tensor.squeeze(0).cpu().numpy(), cmap='gray')
        
#         # 3. Chá»“ng Heatmap lÃªn áº£nh
#         # Sá»­ dá»¥ng giÃ¡ trá»‹ alpha Ä‘á»ƒ hiá»ƒn thá»‹ Ä‘á»™ trong suá»‘t
#         # Thiáº¿t láº­p mÃ u sáº¯c (vÃ­ dá»¥: 'jet' - nÃ³ng)
#         ax.imshow(resized_map, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        
#         ax.set_title(f"ID {cls_id}: {cls_name}", fontsize=12)
#         ax.axis('off')

#     # áº¨n cÃ¡c Ã´ trá»‘ng cÃ²n láº¡i náº¿u sá»‘ lÆ°á»£ng map Ã­t hÆ¡n tá»•ng sá»‘ Ã´
#     for j in range(num_maps, len(axes)):
#         fig.delaxes(axes[j])
        
#     plt.tight_layout()
#     plt.show()


# if image is not None:
#     # Láº¥y thÃ´ng tin cáº§n thiáº¿t
#     active_indices = torch.nonzero(cls_label).squeeze(-1).tolist()
#     id_to_class = {v: k for k, v in dataset.class_to_id.items()}
    
#     # Gá»�i hÃ m Ä‘á»ƒ hiá»ƒn thá»‹
#     visualize_maps(
#         image_tensor=image, 
#         attn_maps_tensor=attn_maps, 
#         active_indices=active_indices, 
#         id_to_class=id_to_class, 
#         target_size=dataset.target_size # KÃ­ch thÆ°á»›c 512
#     )


!python /kaggle/input/vin-train-v3/pytorch/default/1/src/train.py \
  --csv_path "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv" \
  --image_path "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train" \
  --save_path "./checkpoints" \
  --epochs 30 \
  --batch_size 28 \
  --lr 1e-4


# !python /kaggle/input/vin-training/pytorch/default/1/inference.py \
#   --image_path "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/1c32170b4af4ce1a3030eb8167753b06.dicom" \
#   --checkpoint "/kaggle/input/vindr-analysis/checkpoints_debug/best_model.pth" \
#   --train_csv "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv" \
#   --train_dir "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/" --device cuda

