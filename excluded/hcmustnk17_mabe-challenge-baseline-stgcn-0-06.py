# %%writefile stgcn_mabe_final_official.py
# import os
# import sys
# import json
# import logging
# import warnings
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional, Any, DefaultDict
# from collections import defaultdict
# import argparse

# import numpy as np
# import polars as pl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from tqdm.auto import tqdm

# # ========================
# # CUDA AVAILABILITY CHECK â€” IN RA NGAY Láº¬P Tá»¨C
# # ========================
# print("\n" + "="*70)
# print(">>> ğŸ§ª SYSTEM CUDA & GPU CHECK (BEFORE ANYTHING ELSE)")
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"  â†’ GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("  â†’ NO GPU AVAILABLE. Training on CPU.")
# print("<<<")
# print("="*70 + "\n")

# # ========================
# # Config â€” auto collect behaviors + FULL DATA MODE
# # ========================

# class Config:
#     def __init__(self):
#         self.data_root = Path(os.getenv("MABE_DATA_ROOT", "/kaggle/input/MABe-mouse-behavior-detection"))
#         self.submission_file = "submission.csv"
#         self.model_save_path = "stgcn_model.pth"
#         self.window_size = 64
#         self.stride = 32
#         self.batch_size = 16
#         self.num_epochs = 3
#         self.lr = 0.001

#         # Force CUDA if available
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.num_mice = 4
#         self.in_channels = 2

#         # ğŸš¨ AUTO COLLECT BEHAVIORS FROM DATA ğŸš¨
#         self.class_to_idx = self._collect_all_behaviors()
#         self.num_classes = len(self.class_to_idx)
#         self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

#         logger.info(f"âœ… Found {self.num_classes} unique behaviors: {list(self.class_to_idx.keys())}")
#         logger.info(f"ğŸ–¥ï¸�  Using device: {self.device}")

#     def _collect_all_behaviors(self) -> Dict[str, int]:
#         """Scan all .parquet files to collect unique behaviors"""
#         train_csv_path = self.data_root / "train.csv"
#         if not train_csv_path.exists():
#             return {"avoid": 0}

#         df = pl.read_csv(train_csv_path)
#         behaviors = set()

#         for row in df.to_dicts():  # â¬…ï¸� DUYá»†T TOÃ€N Bá»˜ â€” KHÃ”NG .head(10) Ná»®A
#             lab_id = row["lab_id"]
#             video_id = row["video_id"]
#             annot_path = self.data_root / "train_annotation" / lab_id / f"{video_id}.parquet"
#             if not annot_path.exists():
#                 continue
#             try:
#                 annot_df = pl.read_parquet(annot_path)
#                 if "action" in annot_df.columns:
#                     behaviors.update(annot_df["action"].unique().to_list())
#             except Exception as e:
#                 logger.warning(f"Failed to read {annot_path}: {e}")
#                 continue

#         sorted_behaviors = sorted(list(behaviors))
#         return {behavior: idx for idx, behavior in enumerate(sorted_behaviors)}

#     @property
#     def train_csv(self): return self.data_root / "train.csv"
#     @property
#     def test_csv(self): return self.data_root / "test.csv"
#     @property
#     def train_annot_dir(self): return self.data_root / "train_annotation"
#     @property
#     def train_track_dir(self): return self.data_root / "train_tracking"
#     @property
#     def test_track_dir(self): return self.data_root / "test_tracking"

# logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# logger = logging.getLogger(__name__)
# warnings.filterwarnings("ignore")

# # ========================
# # ST-GCN Model â€” FIXED: no mean(dim=3), use .reshape()
# # ========================

# class SpatialGraphConv(nn.Module):
#     def __init__(self, in_channels, out_channels, A):
#         super().__init__()
#         self.A = nn.Parameter(A, requires_grad=False)
#         self.conv = nn.Conv1d(in_channels, out_channels * A.size(0), kernel_size=1)

#     def forward(self, x):
#         x = self.conv(x)
#         n, kc, v = x.size()
#         x = x.view(n, self.A.size(0), kc // self.A.size(0), v)
#         x = torch.einsum('nvcv,vw->nwc', x, self.A)
#         return x.contiguous()

# class STGCNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, A, temporal_kernel_size=9):
#         super().__init__()
#         self.sgc = SpatialGraphConv(in_channels, out_channels, A)
#         self.tcn = nn.Sequential(
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, (temporal_kernel_size, 1), padding=(temporal_kernel_size//2, 0)),
#             nn.BatchNorm2d(out_channels),
#             nn.Dropout(0.5)
#         )
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # x: (N, C, T, V)
#         N, C, T, V = x.size()
#         x = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)  # (N*T, C, V)
#         x = self.sgc(x)  # (N*T, V, C') â†’ (N*T, C', V)
#         x = x.view(N, T, -1, V).permute(0, 2, 1, 3).contiguous()  # (N, C', T, V)
#         x = self.tcn(x)  # (N, C', T, V) â€” no mean!
#         return self.relu(x)

# class MABeSTGCN(nn.Module):
#     def __init__(self, num_classes, num_mice=4, in_channels=2):
#         super().__init__()
#         self.num_mice = num_mice
#         self.total_nodes = num_mice
#         A = torch.ones(num_mice, num_mice)
#         self.A = nn.Parameter(A, requires_grad=False)
#         self.block1 = STGCNBlock(in_channels, 64, self.A)
#         self.block2 = STGCNBlock(64, 128, self.A)
#         self.block3 = STGCNBlock(128, 256, self.A)
#         self.fc = nn.Linear(256 * num_mice, num_classes)  # flatten all nodes

#     def forward(self, x):
#         # x: (N, T, M, 2) â†’ reshape
#         N, T, M, C = x.size()
#         x = x.permute(0, 3, 1, 2).contiguous()  # (N, 2, T, M)

#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)

#         mid_frame = T // 2
#         x = x[:, :, mid_frame, :]  # (N, 256, M)
#         x = x.reshape(N, -1)  # âœ… Use .reshape() instead of .view()
#         return self.fc(x)

# # ========================
# # Dataset â€” WITH FULL CACHING + TQDM LOADING
# # ========================

# class MABeDataset(Dataset):
#     def __init__(self, video_ids: List[int], lab_ids: List[str], track_dir: Path, annot_dir: Path,
#                  window_size: int = 64, stride: int = 32, class_to_idx: Dict[str, int] = None, is_test: bool = False, num_mice: int = 4):
#         self.video_ids = video_ids
#         self.lab_ids = lab_ids
#         self.track_dir = track_dir
#         self.annot_dir = annot_dir
#         self.window_size = window_size
#         self.stride = stride
#         self.class_to_idx = class_to_idx or {}
#         self.is_test = is_test
#         self.num_mice = num_mice
#         self.samples = []

#         # ğŸš€ CACHING SYSTEM
#         self.track_cache = {}    # (lab_id, video_id) -> pl.DataFrame
#         self.xy_cache = {}       # (lab_id, video_id) -> np.ndarray (T, M, 2) â€” Ä‘Ã£ chuáº©n hÃ³a

#         self._build_samples()

#     def _get_track_path(self, lab_id: str, video_id: int) -> Optional[Path]:
#         path = self.track_dir / lab_id / f"{video_id}.parquet"
#         if path.exists():
#             return path
#         return None

#     def _build_samples(self):
#         total_video = len(self.video_ids)
#         logger.info(f"ğŸš€ Starting to process {total_video} videos...")

#         count_track = 0
#         count_parquet = 0
#         count_valid_labels = 0

#         # âœ… THÃŠM TQDM VÃ€O VÃ’NG Láº¶P Xá»¬ LÃ� VIDEO
#         for idx, (video_id, lab_id) in enumerate(tqdm(zip(self.video_ids, self.lab_ids), total=total_video, desc="ğŸ“‚ Loading & Processing Videos")):
#             logger.info(f"ğŸ�¬ Processing video {idx+1}/{total_video}: {lab_id}/{video_id}")

#             track_path = self._get_track_path(lab_id, video_id)
#             if track_path is None:
#                 logger.warning(f"âš ï¸� No .parquet found in train_tracking for {lab_id}/{video_id}")
#                 continue
#             count_track += 1

#             try:
#                 track_df = pl.read_parquet(track_path)
#                 self.track_cache[(lab_id, video_id)] = track_df  # âœ… CACHE TRACK_DF
#                 logger.info(f"âœ… Loaded & cached tracking {track_df.shape}")
#             except Exception as e:
#                 logger.warning(f"âš ï¸� Failed to read tracking {track_path}: {e}")
#                 continue

#             if "video_frame" not in track_df.columns or "mouse_id" not in track_df.columns or "x" not in track_df.columns or "y" not in track_df.columns:
#                 logger.warning(f"âš ï¸� Missing required columns in {track_path}")
#                 continue

#             unique_mice = sorted(track_df["mouse_id"].unique().to_list())
#             mouse_to_idx = {mouse: i for i, mouse in enumerate(unique_mice[:self.num_mice])}
#             all_frames = sorted(track_df["video_frame"].unique().to_list())
#             T = len(all_frames)
#             frame_to_idx = {frame: i for i, frame in enumerate(all_frames)}

#             xy = np.full((T, self.num_mice, 2), np.nan, dtype=np.float32)

#             for row in track_df.to_dicts():
#                 frame = row["video_frame"]
#                 mouse_id = row["mouse_id"]
#                 x = row["x"]
#                 y = row["y"]
#                 if frame in frame_to_idx and mouse_id in mouse_to_idx:
#                     t = frame_to_idx[frame]
#                     m = mouse_to_idx[mouse_id]
#                     xy[t, m, 0] = x
#                     xy[t, m, 1] = y

#             # Forward fill
#             for m in range(self.num_mice):
#                 for t in range(1, T):
#                     if np.isnan(xy[t, m, 0]):
#                         xy[t, m] = xy[t-1, m]

#             # Backward fill
#             for m in range(self.num_mice):
#                 for t in range(T-2, -1, -1):
#                     if np.isnan(xy[t, m, 0]):
#                         xy[t, m] = xy[t+1, m]

#             centroid = np.nanmean(xy, axis=1, keepdims=True)
#             xy_norm = xy - centroid
#             xy_norm = np.nan_to_num(xy_norm, nan=0.0)

#             # âœ… CACHE XY_NORM â€” chá»‰ tÃ­nh 1 láº§n!
#             self.xy_cache[(lab_id, video_id)] = xy_norm

#             if not self.is_test:  # âœ… CHá»ˆ Xá»¬ LÃ� ANNOTATION Náº¾U KHÃ”NG PHáº¢I TEST SET
#                 annot_path = self.annot_dir / lab_id / f"{video_id}.parquet"
#                 if not annot_path.exists():
#                     logger.warning(f"âš ï¸� No .parquet for {lab_id}/{video_id}")
#                     continue
#                 count_parquet += 1

#                 try:
#                     annot_df = pl.read_parquet(annot_path)
#                     logger.info(f"âœ… Loaded annotation {annot_df.shape}")
#                 except Exception as e:
#                     logger.warning(f"âš ï¸� Failed to read annotation {annot_path}: {e}")
#                     continue

#                 if "agent_id" not in annot_df.columns or "action" not in annot_df.columns:
#                     logger.warning(f"âš ï¸� Missing required columns in {annot_path}")
#                     continue

#                 unique_pairs = annot_df.select(["agent_id", "target_id"]).unique().to_dicts()
#                 if len(unique_pairs) == 0:
#                     logger.warning(f"âš ï¸� No pairs for {lab_id}/{video_id}")
#                     continue

#                 for pair in unique_pairs:
#                     agent_id = pair["agent_id"]
#                     target_id = pair["target_id"]
#                     pair_df = annot_df.filter((pl.col("agent_id") == agent_id) & (pl.col("target_id") == target_id))
#                     if len(pair_df) == 0:
#                         continue

#                     labels = np.full(T, -1, dtype=np.int64)
#                     for row in pair_df.to_dicts():
#                         s, e, action = row["start_frame"], row["stop_frame"], row["action"]
#                         if action in self.class_to_idx:
#                             labels[s:e] = self.class_to_idx[action]

#                     for start in range(0, T - self.window_size + 1, self.stride):
#                         window_labels = labels[start:start + self.window_size]
#                         if np.any(window_labels != -1):
#                             mid = start + self.window_size // 2
#                             label = labels[mid] if labels[mid] != -1 else -1
#                             if label != -1:
#                                 self.samples.append((video_id, lab_id, agent_id, target_id, start, label))
#                                 count_valid_labels += 1
#             else:
#                 # âœ… TEST SET: táº¡o dummy samples Ä‘á»ƒ DataLoader cÃ³ thá»ƒ cháº¡y
#                 T = xy_norm.shape[0]
#                 for start in range(0, T - self.window_size + 1, self.stride):
#                     # agent_id, target_id = -1 (placeholder), label = -1
#                     self.samples.append((video_id, lab_id, -1, -1, start, -1))

#         logger.info(f"âœ… Found {count_track} .parquet files in train_tracking/")
#         if not self.is_test:
#             logger.info(f"âœ… Found {count_parquet} .parquet files in train_annotation/")
#         logger.info(f"âœ… Created {len(self.samples)} samples")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         video_id, lab_id, agent_id, target_id, start, label = self.samples[idx]

#         # âœ… Láº¤Y Tá»ª CACHE â€” KHÃ”NG Ä�á»ŒC FILE, KHÃ”NG TIá»€N Xá»¬ LÃ� Láº I
#         xy_norm = self.xy_cache[(lab_id, video_id)]

#         window_xy = xy_norm[start:start + self.window_size]  # (W, M, 2)
#         xy_tensor = torch.tensor(window_xy, dtype=torch.float32)
#         label_tensor = torch.tensor(label, dtype=torch.long)

#         if self.is_test:
#             return xy_tensor, video_id, lab_id, agent_id, target_id, start
#         else:
#             return xy_tensor, label_tensor

# # ========================
# # Training â€” OFFICIAL FULL TRAIN
# # ========================

# def train_model(cfg: Config):
#     print("\nğŸš€ [OFFICIAL TRAINING START] â€” Using FULL dataset\n")

#     logger.info("Loading train.csv...")
#     train_df = pl.read_csv(cfg.train_csv)

#     # âœ… DÃ™NG TOÃ€N Bá»˜ VIDEO â€” KHÃ”NG GIá»šI Háº N
#     video_ids = train_df["video_id"].to_list()
#     lab_ids = train_df["lab_id"].to_list()

#     logger.info("Creating dataset...")
#     dataset = MABeDataset(
#         video_ids=video_ids,
#         lab_ids=lab_ids,
#         track_dir=cfg.train_track_dir,
#         annot_dir=cfg.train_annot_dir,
#         window_size=cfg.window_size,
#         stride=cfg.stride,
#         class_to_idx=cfg.class_to_idx,
#         is_test=False,
#         num_mice=cfg.num_mice
#     )

#     if len(dataset) == 0:
#         raise ValueError("â�Œ Dataset is empty!")

#     logger.info(f"âœ… Total training samples: {len(dataset)}")

#     train_size = max(1, int(0.8 * len(dataset)))
#     val_size = len(dataset) - train_size
#     if val_size == 0:
#         val_size = 1
#         train_size = len(dataset) - 1

#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     # âœ… DÃ¹ng num_workers=2 â€” náº¿u lá»—i, tá»± sá»­a thÃ nh 0
#     try:
#         train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     except Exception as e:
#         logger.warning(f"âš ï¸� DataLoader with num_workers=2 failed: {e}. Falling back to num_workers=0.")
#         train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

#     model = MABeSTGCN(
#         num_classes=cfg.num_classes,
#         num_mice=cfg.num_mice,
#         in_channels=cfg.in_channels
#     ).to(cfg.device)

#     print("\n" + "="*60)
#     print("ğŸ§  MODEL DEVICE:", next(model.parameters()).device)
#     print("="*60 + "\n")

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

#     best_val_acc = 0.0

#     for epoch in range(cfg.num_epochs):
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Train]")
#         for batch_idx, (inputs, labels) in enumerate(pbar):
#             inputs = inputs.to(cfg.device, non_blocking=True)
#             labels = labels.to(cfg.device, non_blocking=True)

#             if epoch == 0 and batch_idx == 0:
#                 print("\n" + "-"*50)
#                 print("ğŸ“¦ FIRST BATCH INPUT DEVICE:", inputs.device)
#                 print("ğŸ�·ï¸�  FIRST BATCH LABEL DEVICE:", labels.device)
#                 print("-"*50 + "\n")

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             train_total += labels.size(0)
#             train_correct += predicted.eq(labels).sum().item()

#             pbar.set_postfix({'Loss': loss.item(), 'Acc': train_correct/train_total})

#         model.eval()
#         val_correct = 0
#         val_total = 0
#         with torch.no_grad():
#             for inputs, labels in tqdm(val_loader, desc="Validation"):
#                 inputs = inputs.to(cfg.device, non_blocking=True)
#                 labels = labels.to(cfg.device, non_blocking=True)
#                 outputs = model(inputs)
#                 _, predicted = outputs.max(1)
#                 val_total += labels.size(0)
#                 val_correct += predicted.eq(labels).sum().item()

#         val_acc = val_correct / val_total
#         logger.info(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), cfg.model_save_path)
#             logger.info(f"âœ… Model saved with Val Acc = {val_acc:.4f}")

#     logger.info(f"ğŸ�‰ Training completed. Best Val Acc: {best_val_acc:.4f}")

# # ========================
# # Inference & Submission â€” OFFICIAL FULL TEST SET
# # ========================

# def predict_and_submit(cfg: Config):
#     logger.info("Loading test.csv...")
#     test_df = pl.read_csv(cfg.test_csv)

#     # âœ… DÃ™NG TOÃ€N Bá»˜ TEST SET
#     video_ids = test_df["video_id"].to_list()
#     lab_ids = test_df["lab_id"].to_list()

#     logger.info("Creating test dataset...")
#     test_dataset = MABeDataset(
#         video_ids=video_ids,
#         lab_ids=lab_ids,
#         track_dir=cfg.test_track_dir,
#         annot_dir=cfg.train_annot_dir,
#         window_size=cfg.window_size,
#         stride=cfg.stride,
#         class_to_idx=cfg.class_to_idx,
#         is_test=True,
#         num_mice=cfg.num_mice
#     )

#     if len(test_dataset) == 0:
#         raise ValueError("â�Œ Test dataset is empty!")

#     logger.info(f"âœ… Total test samples: {len(test_dataset)}")

#     try:
#         test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     except Exception as e:
#         logger.warning(f"âš ï¸� DataLoader with num_workers=2 failed: {e}. Falling back to num_workers=0.")
#         test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

#     model = MABeSTGCN(
#         num_classes=cfg.num_classes,
#         num_mice=cfg.num_mice,
#         in_channels=cfg.in_channels
#     ).to(cfg.device)
#     model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))
#     model.eval()

#     print("\nğŸ§  INFERENCE MODEL DEVICE:", next(model.parameters()).device)

#     predictions = defaultdict(list)

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Inference"):
#             if len(batch) == 6:
#                 inputs, video_ids_batch, lab_ids_batch, agent_ids_batch, target_ids_batch, start_frames = batch
#             else:
#                 inputs, video_ids_batch, lab_ids_batch, start_frames = batch
#                 agent_ids_batch = [-1] * len(video_ids_batch)
#                 target_ids_batch = [-1] * len(video_ids_batch)

#             inputs = inputs.to(cfg.device, non_blocking=True)

#             if len(predictions) == 0:
#                 print("ğŸ§ª INFERENCE INPUT DEVICE:", inputs.device)

#             outputs = model(inputs)
#             _, predicted = outputs.max(1)

#             for i in range(len(video_ids_batch)):
#                 vid = int(video_ids_batch[i])
#                 aid = int(agent_ids_batch[i])
#                 tid = int(target_ids_batch[i])
#                 start = int(start_frames[i])
#                 mid_frame = start + cfg.window_size // 2
#                 label_idx = predicted[i].item()
#                 predictions[(vid, aid, tid)].append((mid_frame, label_idx))

#     records = []
#     for row in test_df.to_dicts():  # â¬…ï¸� DUYá»†T TOÃ€N Bá»˜ TEST SET
#         video_id = row["video_id"]
#         lab_id = row["lab_id"]
#         behaviors = json.loads(row["behaviors_labeled"].replace("'", '"')) if isinstance(row["behaviors_labeled"], str) else []

#         for behavior in behaviors:
#             if len(behavior) != 3:
#                 continue
#             agent_str, target_str, action = behavior
#             if action not in cfg.class_to_idx:
#                 continue

#             try:
#                 agent_id = int(agent_str)
#                 target_id = int(target_str)
#             except:
#                 continue

#             key = (video_id, agent_id, target_id)
#             if key not in predictions or len(predictions[key]) == 0:
#                 # Náº¿u khÃ´ng cÃ³ dá»± Ä‘oÃ¡n, gÃ¡n nhÃ£n Ä‘áº§u tiÃªn (fallback)
#                 default_label = list(cfg.class_to_idx.keys())[0]
#                 records.append((video_id, f"mouse{agent_id}", f"mouse{target_id}", default_label, 0, 1))
#                 continue

#             preds = sorted(predictions[key])
#             if len(preds) == 0:
#                 default_label = list(cfg.class_to_idx.keys())[0]
#                 records.append((video_id, f"mouse{agent_id}", f"mouse{target_id}", default_label, 0, 1))
#                 continue

#             current_label = preds[0][1]
#             start_frame = preds[0][0]
#             for i in range(1, len(preds)):
#                 frame, label = preds[i]
#                 if label != current_label or frame != preds[i-1][0] + 1:
#                     records.append((
#                         video_id,
#                         f"mouse{agent_id}", f"mouse{target_id}",
#                         cfg.idx_to_class[current_label],
#                         start_frame, frame
#                     ))
#                     current_label = label
#                     start_frame = frame
#             records.append((
#                 video_id,
#                 f"mouse{agent_id}", f"mouse{target_id}",
#                 cfg.idx_to_class[current_label],
#                 start_frame, preds[-1][0] + 1
#             ))

#     if len(records) == 0:
#         logger.warning("â�Œ No predictions generated! Using fallback.")
#         default_label = list(cfg.class_to_idx.keys())[0]
#         records = [(0, "mouse0", "mouse0", default_label, 0, 1)]

#     submission_df = pl.DataFrame(
#         records,
#         schema={
#             "video_id": pl.Int64, "agent_id": pl.Utf8, "target_id": pl.Utf8,
#             "action": pl.Utf8, "start_frame": pl.Int64, "stop_frame": pl.Int64,
#         },
#         orient="row"
#     )

#     submission_df = submission_df.with_row_index("row_id")
#     submission_df.write_csv(cfg.submission_file)
#     logger.info(f"âœ… Official Submission saved to {cfg.submission_file}")
#     logger.info(f"ğŸ“Š Submission shape: {submission_df.shape}")
#     print("\nğŸ“‹ Sample predictions:")
#     print(submission_df.head(10))

# # ========================
# # Main
# # ========================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", choices=["train", "submit", "all"], default="all")
#     args = parser.parse_args()

#     cfg = Config()

#     if args.mode in ("train", "all"):
#         train_model(cfg)

#     if args.mode in ("submit", "all"):
#         predict_and_submit(cfg)

# if __name__ == "__main__":
#     main()



#!python stgcn_mabe_final_official.py --mode train


# %%writefile stgcn_mabe_final_official_fixed_v6.py
# import os
# import sys
# import json
# import logging
# import warnings
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional, Any, DefaultDict
# from collections import defaultdict
# import argparse

# import numpy as np
# import polars as pl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from tqdm.auto import tqdm

# print("\n" + "="*70)
# print(">>> ğŸ§ª SYSTEM CUDA & GPU CHECK (BEFORE ANYTHING ELSE)")
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"  â†’ GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("  â†’ NO GPU AVAILABLE. Training on CPU.")
# print("<<<")
# print("="*70 + "\n")

# class Config:
#     def __init__(self):
#         self.data_root = Path(os.getenv("MABE_DATA_ROOT", "/kaggle/input/MABe-mouse-behavior-detection"))
#         self.submission_file = "submission.csv"
#         self.model_save_path = "stgcn_model.pth"
#         self.window_size = 64
#         self.stride = 32
#         self.batch_size = 16
#         self.num_epochs = 3
#         self.lr = 0.001
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.num_mice = 4
#         self.in_channels = 2

#         self.class_to_idx = self._collect_all_behaviors()
#         self.num_classes = len(self.class_to_idx)
#         self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

#         logger.info(f"âœ… Found {self.num_classes} unique behaviors: {list(self.class_to_idx.keys())}")
#         logger.info(f"ğŸ–¥ï¸�  Using device: {self.device}")

#     def _collect_all_behaviors(self) -> Dict[str, int]:
#         train_csv_path = self.data_root / "train.csv"
#         if not train_csv_path.exists():
#             return {"avoid": 0}

#         df = pl.read_csv(train_csv_path)
#         behaviors = set()

#         for row in df.to_dicts():
#             lab_id = row["lab_id"]
#             video_id = row["video_id"]
#             annot_path = self.data_root / "train_annotation" / lab_id / f"{video_id}.parquet"
#             if not annot_path.exists():
#                 continue
#             try:
#                 annot_df = pl.read_parquet(annot_path)
#                 if "action" in annot_df.columns:
#                     behaviors.update(annot_df["action"].unique().to_list())
#             except Exception as e:
#                 logger.warning(f"Failed to read {annot_path}: {e}")
#                 continue

#         sorted_behaviors = sorted(list(behaviors))
#         return {behavior: idx for idx, behavior in enumerate(sorted_behaviors)}

#     @property
#     def train_csv(self): return self.data_root / "train.csv"
#     @property
#     def test_csv(self): return self.data_root / "test.csv"
#     @property
#     def train_annot_dir(self): return self.data_root / "train_annotation"
#     @property
#     def train_track_dir(self): return self.data_root / "train_tracking"
#     @property
#     def test_track_dir(self): return self.data_root / "test_tracking"

# logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# logger = logging.getLogger(__name__)
# warnings.filterwarnings("ignore")

# # âœ… HÃ m safe_json_loads - tá»‘i Æ°u cho data cá»§a báº¡n
# def safe_json_loads(s: Optional[str]) -> List[str]:
#     if s is None: return []
#     if isinstance(s, list): return [str(x) for x in s]
#     if not isinstance(s, str): return []
#     s = s.strip()
#     if not s: return []
#     try:
#         loaded = json.loads(s)
#         if isinstance(loaded, list):
#             return [str(item) for item in loaded]
#         else:
#             return []
#     except Exception:
#         try:
#             # Thá»­ sá»­a dáº¥u nhÃ¡y Ä‘Æ¡n
#             fixed = s.replace("'", '"')
#             loaded = json.loads(fixed)
#             if isinstance(loaded, list):
#                 return [str(item) for item in loaded]
#             else:
#                 return []
#         except Exception:
#             return []

# class SpatialGraphConv(nn.Module):
#     def __init__(self, in_channels, out_channels, A):
#         super().__init__()
#         self.A = nn.Parameter(A, requires_grad=False)
#         self.conv = nn.Conv1d(in_channels, out_channels * A.size(0), kernel_size=1)

#     def forward(self, x):
#         x = self.conv(x)
#         n, kc, v = x.size()
#         x = x.view(n, self.A.size(0), kc // self.A.size(0), v)
#         x = torch.einsum('nvcv,vw->nwc', x, self.A)
#         return x.contiguous()

# class STGCNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, A, temporal_kernel_size=9):
#         super().__init__()
#         self.sgc = SpatialGraphConv(in_channels, out_channels, A)
#         self.tcn = nn.Sequential(
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, (temporal_kernel_size, 1), padding=(temporal_kernel_size//2, 0)),
#             nn.BatchNorm2d(out_channels),
#             nn.Dropout(0.5)
#         )
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         N, C, T, V = x.size()
#         x = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)
#         x = self.sgc(x)
#         x = x.view(N, T, -1, V).permute(0, 2, 1, 3).contiguous()
#         x = self.tcn(x)
#         return self.relu(x)

# class MABeSTGCN(nn.Module):
#     def __init__(self, num_classes, num_mice=4, in_channels=2):
#         super().__init__()
#         self.num_mice = num_mice
#         self.total_nodes = num_mice
#         A = torch.ones(num_mice, num_mice)
#         self.A = nn.Parameter(A, requires_grad=False)
#         self.block1 = STGCNBlock(in_channels, 64, self.A)
#         self.block2 = STGCNBlock(64, 128, self.A)
#         self.block3 = STGCNBlock(128, 256, self.A)
#         self.fc = nn.Linear(256 * num_mice, num_classes)

#     def forward(self, x):
#         N, T, M, C = x.size()
#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         mid_frame = T // 2
#         x = x[:, :, mid_frame, :]
#         x = x.reshape(N, -1)
#         return self.fc(x)

# class MABeDataset(Dataset):
#     def __init__(self, video_ids: List[int], lab_ids: List[str], track_dir: Path, annot_dir: Path,
#                  window_size: int = 64, stride: int = 32, class_to_idx: Dict[str, int] = None, is_test: bool = False, num_mice: int = 4):
#         self.video_ids = video_ids
#         self.lab_ids = lab_ids
#         self.track_dir = track_dir
#         self.annot_dir = annot_dir
#         self.window_size = window_size
#         self.stride = stride
#         self.class_to_idx = class_to_idx or {}
#         self.is_test = is_test
#         self.num_mice = num_mice
#         self.samples = []
#         self.track_cache = {}
#         self.xy_cache = {}

#         self._build_samples()

#     def _get_track_path(self, lab_id: str, video_id: int) -> Optional[Path]:
#         path = self.track_dir / lab_id / f"{video_id}.parquet"
#         if path.exists():
#             return path
#         return None

#     def _build_samples(self):
#         total_video = len(self.video_ids)
#         logger.info(f"ğŸš€ Starting to process {total_video} videos...")

#         count_track = 0
#         count_parquet = 0
#         count_valid_labels = 0

#         for idx, (video_id, lab_id) in enumerate(tqdm(zip(self.video_ids, self.lab_ids), total=total_video, desc="ğŸ“‚ Loading & Processing Videos")):
#             track_path = self._get_track_path(lab_id, video_id)
#             if track_path is None:
#                 continue
#             count_track += 1

#             try:
#                 track_df = pl.read_parquet(track_path)
#                 self.track_cache[(lab_id, video_id)] = track_df
#             except Exception as e:
#                 logger.warning(f"Failed to cache tracking for {lab_id}/{video_id}: {e}")
#                 continue

#             required_cols = ["video_frame", "mouse_id", "x", "y"]
#             if not all(col in track_df.columns for col in required_cols):
#                 logger.warning(f"Missing required columns in {track_path}")
#                 continue

#             # Láº¥y danh sÃ¡ch mouse_id thá»±c táº¿ â€” CHUYá»‚N Vá»€ INT náº¿u cÃ³ thá»ƒ
#             unique_mice_raw = track_df["mouse_id"].unique().to_list()
#             unique_mice = []
#             for m in unique_mice_raw:
#                 try:
#                     mid = int(m)
#                     unique_mice.append(mid)
#                 except (ValueError, TypeError):
#                     logger.warning(f"Cannot convert mouse_id {m} to int, skipping.")
#                     continue

#             unique_mice = sorted(set(unique_mice))[:self.num_mice]
#             if len(unique_mice) == 0:
#                 continue

#             mouse_to_idx = {mouse: i for i, mouse in enumerate(unique_mice)}
#             all_frames = sorted(track_df["video_frame"].unique().to_list())
#             T = len(all_frames)
#             frame_to_idx = {frame: i for i, frame in enumerate(all_frames)}

#             xy = np.full((T, self.num_mice, 2), np.nan, dtype=np.float32)

#             for row in track_df.to_dicts():
#                 frame = row["video_frame"]
#                 mouse_id = row["mouse_id"]
#                 try:
#                     mouse_id = int(mouse_id)
#                 except (ValueError, TypeError):
#                     continue
#                 x = row["x"]
#                 y = row["y"]
#                 if frame in frame_to_idx and mouse_id in mouse_to_idx:
#                     t = frame_to_idx[frame]
#                     m = mouse_to_idx[mouse_id]
#                     xy[t, m, 0] = x
#                     xy[t, m, 1] = y

#             # Interpolation forward
#             for m in range(self.num_mice):
#                 for t in range(1, T):
#                     if np.isnan(xy[t, m, 0]):
#                         xy[t, m] = xy[t-1, m]

#             # Interpolation backward
#             for m in range(self.num_mice):
#                 for t in range(T-2, -1, -1):
#                     if np.isnan(xy[t, m, 0]):
#                         xy[t, m] = xy[t+1, m]

#             centroid = np.nanmean(xy, axis=1, keepdims=True)
#             xy_norm = xy - centroid
#             xy_norm = np.nan_to_num(xy_norm, nan=0.0)
#             self.xy_cache[(lab_id, video_id)] = xy_norm

#             if not self.is_test:
#                 annot_path = self.annot_dir / lab_id / f"{video_id}.parquet"
#                 if not annot_path.exists():
#                     continue
#                 count_parquet += 1

#                 try:
#                     annot_df = pl.read_parquet(annot_path)
#                 except Exception as e:
#                     logger.warning(f"Failed to read annotation {annot_path}: {e}")
#                     continue

#                 if "agent_id" not in annot_df.columns or "action" not in annot_df.columns:
#                     continue

#                 unique_pairs = annot_df.select(["agent_id", "target_id"]).unique().to_dicts()
#                 if len(unique_pairs) == 0:
#                     continue

#                 for pair in unique_pairs:
#                     agent_id = pair["agent_id"]
#                     target_id = pair["target_id"]
#                     try:
#                         agent_id = int(agent_id)
#                         target_id = int(target_id)
#                     except (ValueError, TypeError):
#                         continue

#                     pair_df = annot_df.filter((pl.col("agent_id") == agent_id) & (pl.col("target_id") == target_id))
#                     if len(pair_df) == 0:
#                         continue

#                     labels = np.full(T, -1, dtype=np.int64)
#                     for row in pair_df.to_dicts():
#                         s, e, action = row["start_frame"], row["stop_frame"], row["action"]
#                         if action in self.class_to_idx:
#                             labels[s:e] = self.class_to_idx[action]

#                     for start in range(0, T - self.window_size + 1, self.stride):
#                         window_labels = labels[start:start + self.window_size]
#                         if np.any(window_labels != -1):
#                             mid = start + self.window_size // 2
#                             label = labels[mid] if labels[mid] != -1 else -1
#                             if label != -1:
#                                 self.samples.append((video_id, lab_id, agent_id, target_id, start, label))
#                                 count_valid_labels += 1
#             else:
#                 # TEST SET: táº¡o sample cho táº¥t cáº£ cáº·p (agent, target) cÃ³ thá»ƒ â€” CHá»ˆ DÃ™NG INT
#                 for agent_id in unique_mice:
#                     for target_id in unique_mice:
#                         for start in range(0, T - self.window_size + 1, self.stride):
#                             self.samples.append((video_id, lab_id, agent_id, target_id, start, -1))

#         logger.info(f"âœ… Found {count_track} .parquet files in train_tracking/")
#         if not self.is_test:
#             logger.info(f"âœ… Found {count_parquet} .parquet files in train_annotation/")
#         logger.info(f"âœ… Created {len(self.samples)} samples")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         video_id, lab_id, agent_id, target_id, start, label = self.samples[idx]
#         xy_norm = self.xy_cache[(lab_id, video_id)]
#         window_xy = xy_norm[start:start + self.window_size]
#         xy_tensor = torch.tensor(window_xy, dtype=torch.float32)
#         label_tensor = torch.tensor(label, dtype=torch.long)
#         return xy_tensor, label_tensor, video_id, lab_id, agent_id, target_id, start

# def train_model(cfg: Config):
#     print("\nğŸš€ [OFFICIAL TRAINING START] â€” Using FULL dataset\n")

#     train_df = pl.read_csv(cfg.train_csv)
#     video_ids = train_df["video_id"].to_list()
#     lab_ids = train_df["lab_id"].to_list()

#     dataset = MABeDataset(
#         video_ids=video_ids,
#         lab_ids=lab_ids,
#         track_dir=cfg.train_track_dir,
#         annot_dir=cfg.train_annot_dir,
#         window_size=cfg.window_size,
#         stride=cfg.stride,
#         class_to_idx=cfg.class_to_idx,
#         is_test=False,
#         num_mice=cfg.num_mice
#     )

#     if len(dataset) == 0:
#         raise ValueError("â�Œ Dataset is empty!")

#     logger.info(f"âœ… Total training samples: {len(dataset)}")

#     train_size = max(1, int(0.8 * len(dataset)))
#     val_size = len(dataset) - train_size
#     if val_size == 0:
#         val_size = 1
#         train_size = len(dataset) - 1

#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     try:
#         train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     except Exception as e:
#         logger.warning(f"Failed to use num_workers=2: {e}")
#         train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

#     model = MABeSTGCN(
#         num_classes=cfg.num_classes,
#         num_mice=cfg.num_mice,
#         in_channels=cfg.in_channels
#     ).to(cfg.device)

#     print("\n" + "="*60)
#     print("ğŸ§  MODEL DEVICE:", next(model.parameters()).device)
#     print("="*60 + "\n")

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

#     best_val_acc = 0.0

#     for epoch in range(cfg.num_epochs):
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Train]")
#         for batch_idx, (inputs, labels, _, _, _, _, _) in enumerate(pbar):
#             inputs = inputs.to(cfg.device, non_blocking=True)
#             labels = labels.to(cfg.device, non_blocking=True)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             train_total += labels.size(0)
#             train_correct += predicted.eq(labels).sum().item()

#             pbar.set_postfix({'Loss': loss.item(), 'Acc': train_correct/train_total})

#         model.eval()
#         val_correct = 0
#         val_total = 0
#         with torch.no_grad():
#             for inputs, labels, _, _, _, _, _ in tqdm(val_loader, desc="Validation"):
#                 inputs = inputs.to(cfg.device, non_blocking=True)
#                 labels = labels.to(cfg.device, non_blocking=True)
#                 outputs = model(inputs)
#                 _, predicted = outputs.max(1)
#                 val_total += labels.size(0)
#                 val_correct += predicted.eq(labels).sum().item()

#         val_acc = val_correct / val_total
#         logger.info(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), cfg.model_save_path)
#             logger.info(f"âœ… Model saved with Val Acc = {val_acc:.4f}")

#     logger.info(f"ğŸ�‰ Training completed. Best Val Acc: {best_val_acc:.4f}")

# def predict_and_submit(cfg: Config):
#     logger.info("Loading test.csv...")
#     test_df = pl.read_csv(cfg.test_csv)
#     video_ids = test_df["video_id"].to_list()
#     lab_ids = test_df["lab_id"].to_list()

#     # âœ… Táº¡o mapping: video_id â†’ set of allowed (agent_str, target_str, action)
#     video_to_allowed_triples = {}  # (agent_str, target_str, action) â†’ bool

#     for row in test_df.to_dicts():
#         video_id = row["video_id"]
#         behaviors_str = row["behaviors_labeled"]
#         behaviors_list = safe_json_loads(behaviors_str)
#         allowed_triples = set()
#         for item in behaviors_list:
#             parts = [p.strip() for p in str(item).replace("'", "").split(",")]
#             if len(parts) == 3:
#                 agent_str, target_str, action = parts
#                 allowed_triples.add((agent_str, target_str, action))
#         video_to_allowed_triples[video_id] = allowed_triples

#     logger.info("Creating test dataset...")
#     test_dataset = MABeDataset(
#         video_ids=video_ids,
#         lab_ids=lab_ids,
#         track_dir=cfg.test_track_dir,
#         annot_dir=cfg.train_annot_dir,
#         window_size=cfg.window_size,
#         stride=cfg.stride,
#         class_to_idx=cfg.class_to_idx,
#         is_test=True,
#         num_mice=cfg.num_mice
#     )

#     if len(test_dataset) == 0:
#         raise ValueError("â�Œ Test dataset is empty!")

#     logger.info(f"âœ… Total test samples: {len(test_dataset)}")

#     try:
#         test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     except Exception as e:
#         logger.warning(f"Failed to use num_workers=2: {e}")
#         test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

#     model = MABeSTGCN(
#         num_classes=cfg.num_classes,
#         num_mice=cfg.num_mice,
#         in_channels=cfg.in_channels
#     ).to(cfg.device)
#     model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))
#     model.eval()

#     predictions = defaultdict(list)

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Inference"):
#             inputs, _, video_ids_batch, lab_ids_batch, agent_ids_batch, target_ids_batch, start_frames = batch
#             inputs = inputs.to(cfg.device, non_blocking=True)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)

#             for i in range(len(video_ids_batch)):
#                 vid = int(video_ids_batch[i])
#                 aid = int(agent_ids_batch[i])
#                 tid = int(target_ids_batch[i])
#                 start = int(start_frames[i])
#                 mid_frame = start + cfg.window_size // 2
#                 label_idx = predicted[i].item()
#                 predictions[(vid, aid, tid)].append((mid_frame, label_idx))

#     records = []

#     # âœ… CHá»ˆ Dá»° Ä�OÃ�N Náº¾U (agent_str, target_str, action) CÃ“ TRONG behaviors_labeled
#     for (video_id, agent_id, target_id), preds in predictions.items():
#         if len(preds) == 0:
#             continue

#         allowed_triples = video_to_allowed_triples.get(video_id, set())

#         # Chuyá»ƒn agent_id, target_id (int) â†’ string cÃ³ tiá»�n tá»‘ "mouse"
#         agent_str = f"mouse{agent_id}"
#         target_str = f"mouse{target_id}"

#         preds = sorted(preds)
#         current_label = preds[0][1]
#         current_action = cfg.idx_to_class[current_label]

#         # âœ… Kiá»ƒm tra triple Ä‘áº§y Ä‘á»§: (agent_str, target_str, action)
#         current_triple = (agent_str, target_str, current_action)
#         if current_triple not in allowed_triples:
#             current_label = -1  # skip

#         start_frame = preds[0][0]

#         for i in range(1, len(preds)):
#             frame, label = preds[i]
#             action = cfg.idx_to_class[label]
#             triple = (agent_str, target_str, action)

#             # âœ… Náº¿u thay Ä‘á»•i label, khÃ´ng liÃªn tá»¥c, hoáº·c triple khÃ´ng Ä‘Æ°á»£c phÃ©p â†’ Ä‘Ã³ng segment cÅ©
#             if label != current_label or frame != preds[i-1][0] + 1 or triple not in allowed_triples:
#                 if current_label != -1:
#                     # âœ… Kiá»ƒm tra láº§n cuá»‘i trÆ°á»›c khi ghi
#                     final_triple = (agent_str, target_str, cfg.idx_to_class[current_label])
#                     if final_triple in allowed_triples:
#                         records.append((
#                             video_id,
#                             agent_str, target_str,
#                             cfg.idx_to_class[current_label],
#                             start_frame, frame
#                         ))
#                 # Báº¯t Ä‘áº§u segment má»›i â€” chá»‰ náº¿u triple Ä‘Æ°á»£c phÃ©p
#                 current_label = label if triple in allowed_triples else -1
#                 start_frame = frame

#         # âœ… Ä�Ã³ng segment cuá»‘i cÃ¹ng â€” kiá»ƒm tra triple
#         if current_label != -1:
#             final_triple = (agent_str, target_str, cfg.idx_to_class[current_label])
#             if final_triple in allowed_triples:
#                 records.append((
#                     video_id,
#                     agent_str, target_str,
#                     cfg.idx_to_class[current_label],
#                     start_frame, preds[-1][0] + 1
#                 ))

#     if len(records) == 0:
#         logger.warning("â�Œ No valid predictions generated!")
#         # Thá»­ tÃ¬m má»™t hÃ nh vi há»£p lá»‡ báº¥t ká»³ Ä‘á»ƒ dÃ¹ng lÃ m máº·c Ä‘á»‹nh
#         found = False
#         for vid, triples in video_to_allowed_triples.items():
#             if triples:
#                 agent_str, target_str, action = next(iter(triples))
#                 records = [(vid, agent_str, target_str, action, 0, 1)]
#                 found = True
#                 break
#         if not found:
#             default_label = list(cfg.class_to_idx.keys())[0]
#             records = [(0, "mouse1", "mouse2", default_label, 0, 1)]

#     submission_df = pl.DataFrame(
#         records,
#         schema={
#             "video_id": pl.Int64, "agent_id": pl.Utf8, "target_id": pl.Utf8,
#             "action": pl.Utf8, "start_frame": pl.Int64, "stop_frame": pl.Int64,
#         },
#         orient="row"
#     )

#     submission_df = submission_df.with_row_index("row_id")
#     submission_df.write_csv(cfg.submission_file)
#     logger.info(f"âœ… Official Submission saved to {cfg.submission_file}")
#     logger.info(f"ğŸ“Š Submission shape: {submission_df.shape}")
#     print("\nğŸ“‹ Sample predictions:")
#     print(submission_df.head(10))

       


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", choices=["train", "submit", "all"], default="all")
#     args = parser.parse_args()

#     cfg = Config()

#     if args.mode in ("train", "all"):
#         train_model(cfg)

#     if args.mode in ("submit", "all"):
#         predict_and_submit(cfg)

# if __name__ == "__main__":
#     main()



# %%writefile stgcn_mabe_v2_final.py
# import os
# import sys
# import json
# import warnings
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional, Any, DefaultDict
# from collections import defaultdict
# import argparse

# import numpy as np
# import polars as pl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from tqdm.auto import tqdm
# from sklearn.metrics import f1_score

# print("\n" + "="*70)
# print(">>> ğŸ§ª SYSTEM CUDA & GPU CHECK (BEFORE ANYTHING ELSE)")
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"  â†’ GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("  â†’ NO GPU AVAILABLE. Training on CPU.")
# print("<<<")
# print("="*70 + "\n")

# class Config:
#     def __init__(self):
#         self.data_root = Path(os.getenv("MABE_DATA_ROOT", "/kaggle/input/MABe-mouse-behavior-detection"))
#         self.submission_file = "submission.csv"
#         self.model_save_path = "stgcn_v2_model.pth"
#         self.window_size = 64
#         self.stride = 32
#         self.batch_size = 16
#         self.num_epochs = 5
#         self.lr = 0.001
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.num_mice = 4
#         self.in_channels = 5  # x, y, vx, vy, angle
#         self.embed_dim = 32
#         self.lab_embed_dim = 16

#         # Thu tháº­p táº¥t cáº£ hÃ nh vi
#         self.class_to_idx = self._collect_all_behaviors()
#         self.num_classes = len(self.class_to_idx)
#         self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

#         # Thu tháº­p táº¥t cáº£ lab_id Ä‘á»ƒ táº¡o lab embedding
#         self.lab_to_idx = self._collect_all_labs()
#         self.num_labs = len(self.lab_to_idx)

#         print(f"âœ… Found {self.num_classes} unique behaviors: {list(self.class_to_idx.keys())}")
#         print(f"âœ… Found {self.num_labs} unique labs: {list(self.lab_to_idx.keys())}")
#         print(f"ğŸ–¥ï¸�  Using device: {self.device}")

#     def _collect_all_behaviors(self) -> Dict[str, int]:
#         train_csv_path = self.data_root / "train.csv"
#         if not train_csv_path.exists():
#             return {"avoid": 0}

#         df = pl.read_csv(train_csv_path)
#         behaviors = set()

#         for row in df.to_dicts():
#             lab_id = row["lab_id"]
#             video_id = row["video_id"]
#             annot_path = self.data_root / "train_annotation" / lab_id / f"{video_id}.parquet"
#             if not annot_path.exists():
#                 continue
#             try:
#                 annot_df = pl.read_parquet(annot_path)
#                 if "action" in annot_df.columns:
#                     behaviors.update(annot_df["action"].unique().to_list())
#             except Exception as e:
#                 print(f"âš ï¸�  Failed to read {annot_path}: {e}")
#                 continue

#         sorted_behaviors = sorted(list(behaviors))
#         return {behavior: idx for idx, behavior in enumerate(sorted_behaviors)}

#     def _collect_all_labs(self) -> Dict[str, int]:
#         train_csv_path = self.data_root / "train.csv"
#         if not train_csv_path.exists():
#             return {"unknown": 0}
#         df = pl.read_csv(train_csv_path)
#         labs = df["lab_id"].unique().to_list()
#         return {lab: idx for idx, lab in enumerate(labs)}

#     @property
#     def train_csv(self): return self.data_root / "train.csv"
#     @property
#     def test_csv(self): return self.data_root / "test.csv"
#     @property
#     def train_annot_dir(self): return self.data_root / "train_annotation"
#     @property
#     def train_track_dir(self): return self.data_root / "train_tracking"
#     @property
#     def test_track_dir(self): return self.data_root / "test_tracking"

# warnings.filterwarnings("ignore")

# # âœ… HÃ m safe_json_loads - tá»‘i Æ°u cho data cá»§a báº¡n
# def safe_json_loads(s: Optional[str]) -> List[str]:
#     if s is None: return []
#     if isinstance(s, list): return [str(x) for x in s]
#     if not isinstance(s, str): return []
#     s = s.strip()
#     if not s: return []
#     try:
#         loaded = json.loads(s)
#         if isinstance(loaded, list):
#             return [str(item) for item in loaded]
#         else:
#             return []
#     except Exception:
#         try:
#             fixed = s.replace("'", '"')
#             loaded = json.loads(fixed)
#             if isinstance(loaded, list):
#                 return [str(item) for item in loaded]
#             else:
#                 return []
#         except Exception:
#             return []

# class SpatialGraphConv(nn.Module):
#     def __init__(self, in_channels, out_channels, A):
#         super().__init__()
#         self.A = nn.Parameter(A, requires_grad=True)  # â†� âœ… Learnable adjacency
#         self.conv = nn.Conv1d(in_channels, out_channels * A.size(0), kernel_size=1)

#     def forward(self, x):
#         x = self.conv(x)
#         n, kc, v = x.size()
#         x = x.view(n, self.A.size(0), kc // self.A.size(0), v)
#         x = torch.einsum('nkcv,vw->nwc', x, self.A)
#         return x.contiguous()

# class STGCNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, A, temporal_kernel_size=9):
#         super().__init__()
#         self.sgc = SpatialGraphConv(in_channels, out_channels, A)
#         self.tcn = nn.Sequential(
#             nn.InstanceNorm2d(out_channels),  # â†� âœ… InstanceNorm thay BatchNorm
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, (temporal_kernel_size, 1), padding=(temporal_kernel_size//2, 0)),
#             nn.InstanceNorm2d(out_channels),
#             nn.Dropout(0.5)
#         )
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         N, C, T, V = x.size()
#         x = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)
#         x = self.sgc(x)
#         x = x.view(N, T, -1, V).permute(0, 2, 1, 3).contiguous()
#         x = self.tcn(x)
#         return self.relu(x)

# class MABeSTGCNv2(nn.Module):
#     def __init__(self, num_classes, num_mice=4, in_channels=5, embed_dim=32, lab_embed_dim=16, num_labs=10):
#         super().__init__()
#         self.num_mice = num_mice
#         self.in_channels = in_channels
#         self.embed_dim = embed_dim
#         self.lab_embed_dim = lab_embed_dim

#         # Adaptive adjacency matrix (learnable)
#         A = torch.ones(num_mice, num_mice)
#         self.A = nn.Parameter(A, requires_grad=True)

#         # ST-GCN Blocks
#         self.block1 = STGCNBlock(in_channels, 64, self.A)
#         self.block2 = STGCNBlock(64, 128, self.A)
#         self.block3 = STGCNBlock(128, 256, self.A)

#         # Pair embedding: 4x4 = 16 pairs
#         self.pair_embedding = nn.Embedding(num_mice * num_mice, embed_dim)

#         # Lab embedding
#         self.lab_embedding = nn.Embedding(num_labs, lab_embed_dim)

#         # Final classifier
#         self.fc = nn.Sequential(
#             nn.Linear(256 * num_mice + embed_dim + lab_embed_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x, agent_id, target_id, lab_id):
#         # x: (N, T, M, C) â€” C=5: x,y,vx,vy,angle
#         N, T, M, C = x.size()
#         x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, M)

#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)  # (N, 256, T, M)

#         # Global average pooling over time
#         x = x.mean(dim=2)  # (N, 256, M)
#         x = x.view(N, -1)  # (N, 256*M)

#         # Pair embedding
#         pair_idx = agent_id * self.num_mice + target_id  # (N,)
#         pair_emb = self.pair_embedding(pair_idx)  # (N, embed_dim)

#         # Lab embedding
#         lab_emb = self.lab_embedding(lab_id)  # (N, lab_embed_dim)

#         # Concat all
#         x = torch.cat([x, pair_emb, lab_emb], dim=1)  # (N, 256*M + embed_dim + lab_embed_dim)
#         return self.fc(x)

# class MABeDataset(Dataset):
#     def __init__(self, video_ids: List[int], lab_ids: List[str], track_dir: Path, annot_dir: Path,
#                  window_size: int = 64, stride: int = 32, class_to_idx: Dict[str, int] = None,
#                  lab_to_idx: Dict[str, int] = None, is_test: bool = False, num_mice: int = 4):
#         self.video_ids = video_ids
#         self.lab_ids = lab_ids
#         self.track_dir = track_dir
#         self.annot_dir = annot_dir
#         self.window_size = window_size
#         self.stride = stride
#         self.class_to_idx = class_to_idx or {}
#         self.lab_to_idx = lab_to_idx or {}
#         self.is_test = is_test
#         self.num_mice = num_mice
#         self.samples = []
#         self.track_cache = {}
#         self.feature_cache = {}  # cache xy + velocity + angle

#         self._build_samples()

#     def _get_track_path(self, lab_id: str, video_id: int) -> Optional[Path]:
#         path = self.track_dir / lab_id / f"{video_id}.parquet"
#         if path.exists():
#             return path
#         return None

#     def _build_samples(self):
#         total_video = len(self.video_ids)
#         print(f"ğŸš€ Starting to process {total_video} videos...")

#         count_track = 0
#         count_parquet = 0
#         count_valid_labels = 0

#         for idx, (video_id, lab_id) in enumerate(tqdm(zip(self.video_ids, self.lab_ids), total=total_video, desc="ğŸ“‚ Loading & Processing Videos")):
#             track_path = self._get_track_path(lab_id, video_id)
#             if track_path is None:
#                 continue
#             count_track += 1

#             try:
#                 track_df = pl.read_parquet(track_path)
#                 self.track_cache[(lab_id, video_id)] = track_df
#             except Exception as e:
#                 print(f"âš ï¸�  Failed to cache tracking for {lab_id}/{video_id}: {e}")
#                 continue

#             required_cols = ["video_frame", "mouse_id", "x", "y"]
#             if not all(col in track_df.columns for col in required_cols):
#                 print(f"âš ï¸�  Missing required columns in {track_path}")
#                 continue

#             # Láº¥y danh sÃ¡ch mouse_id thá»±c táº¿ â€” CHUYá»‚N Vá»€ INT
#             unique_mice_raw = track_df["mouse_id"].unique().to_list()
#             unique_mice = []
#             for m in unique_mice_raw:
#                 try:
#                     mid = int(m)
#                     unique_mice.append(mid)
#                 except (ValueError, TypeError):
#                     continue

#             unique_mice = sorted(set(unique_mice))[:self.num_mice]
#             if len(unique_mice) == 0:
#                 continue

#             mouse_to_idx = {mouse: i for i, mouse in enumerate(unique_mice)}
#             all_frames = sorted(track_df["video_frame"].unique().to_list())
#             T = len(all_frames)
#             frame_to_idx = {frame: i for i, frame in enumerate(all_frames)}

#             # Khá»Ÿi táº¡o máº£ng xy
#             xy = np.full((T, self.num_mice, 2), np.nan, dtype=np.float32)

#             for row in track_df.to_dicts():
#                 frame = row["video_frame"]
#                 mouse_id = row["mouse_id"]
#                 try:
#                     mouse_id = int(mouse_id)
#                 except (ValueError, TypeError):
#                     continue
#                 x = row["x"]
#                 y = row["y"]
#                 if frame in frame_to_idx and mouse_id in mouse_to_idx:
#                     t = frame_to_idx[frame]
#                     m = mouse_to_idx[mouse_id]
#                     xy[t, m, 0] = x
#                     xy[t, m, 1] = y

#             # Interpolation forward/backward
#             for m in range(self.num_mice):
#                 for t in range(1, T):
#                     if np.isnan(xy[t, m, 0]):
#                         xy[t, m] = xy[t-1, m]
#                 for t in range(T-2, -1, -1):
#                     if np.isnan(xy[t, m, 0]):
#                         xy[t, m] = xy[t+1, m]

#             # TÃ­nh centroid vÃ  chuáº©n hÃ³a
#             centroid = np.nanmean(xy, axis=1, keepdims=True)
#             xy_norm = xy - centroid
#             xy_norm = np.nan_to_num(xy_norm, nan=0.0)

#             # TÃ­nh velocity (sai phÃ¢n)
#             vx = np.zeros_like(xy_norm[..., 0])
#             vy = np.zeros_like(xy_norm[..., 1])
#             vx[1:] = xy_norm[1:, :, 0] - xy_norm[:-1, :, 0]
#             vy[1:] = xy_norm[1:, :, 1] - xy_norm[:-1, :, 1]

#             # TÃ­nh angle (giáº£ sá»­: angle = arctan2(vy, vx))
#             angle = np.arctan2(vy, vx)  # shape (T, M)

#             # GhÃ©p thÃ nh feature 5D: [x, y, vx, vy, angle]
#             features = np.stack([xy_norm[..., 0], xy_norm[..., 1], vx, vy, angle], axis=-1)  # (T, M, 5)

#             self.feature_cache[(lab_id, video_id)] = features

#             if not self.is_test:
#                 annot_path = self.annot_dir / lab_id / f"{video_id}.parquet"
#                 if not annot_path.exists():
#                     continue
#                 count_parquet += 1

#                 try:
#                     annot_df = pl.read_parquet(annot_path)
#                 except Exception as e:
#                     print(f"âš ï¸�  Failed to read annotation {annot_path}: {e}")
#                     continue

#                 if "agent_id" not in annot_df.columns or "action" not in annot_df.columns:
#                     continue

#                 unique_pairs = annot_df.select(["agent_id", "target_id"]).unique().to_dicts()
#                 if len(unique_pairs) == 0:
#                     continue

#                 for pair in unique_pairs:
#                     agent_id = pair["agent_id"]
#                     target_id = pair["target_id"]
#                     try:
#                         agent_id = int(agent_id)
#                         target_id = int(target_id)
#                     except (ValueError, TypeError):
#                         continue

#                     if agent_id not in mouse_to_idx or target_id not in mouse_to_idx:
#                         continue

#                     pair_df = annot_df.filter((pl.col("agent_id") == agent_id) & (pl.col("target_id") == target_id))
#                     if len(pair_df) == 0:
#                         continue

#                     labels = np.full(T, -1, dtype=np.int64)
#                     for row in pair_df.to_dicts():
#                         s, e, action = row["start_frame"], row["stop_frame"], row["action"]
#                         if action in self.class_to_idx:
#                             labels[s:e+1] = self.class_to_idx[action]  # âš ï¸� Sá»­a: e+1 Ä‘á»ƒ bao gá»“m stop_frame

#                     for start in range(0, T - self.window_size + 1, self.stride):
#                         window_labels = labels[start:start + self.window_size]
#                         if np.any(window_labels != -1):
#                             mid = start + self.window_size // 2
#                             label = labels[mid] if labels[mid] != -1 else -1
#                             if label != -1:
#                                 self.samples.append((video_id, lab_id, agent_id, target_id, start, label))
#                                 count_valid_labels += 1
#             else:
#                 # TEST SET: táº¡o sample cho táº¥t cáº£ cáº·p (agent, target) cÃ³ thá»ƒ
#                 for agent_id in unique_mice:
#                     for target_id in unique_mice:
#                         for start in range(0, T - self.window_size + 1, self.stride):
#                             self.samples.append((video_id, lab_id, agent_id, target_id, start, -1))

#         print(f"âœ… Found {count_track} .parquet files in train_tracking/")
#         if not self.is_test:
#             print(f"âœ… Found {count_parquet} .parquet files in train_annotation/")
#         print(f"âœ… Created {len(self.samples)} samples")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         video_id, lab_id, agent_id, target_id, start, label = self.samples[idx]
#         features = self.feature_cache[(lab_id, video_id)]  # (T, M, 5)
#         window_feat = features[start:start + self.window_size]  # (window, M, 5)
#         feat_tensor = torch.tensor(window_feat, dtype=torch.float32)
#         label_tensor = torch.tensor(label, dtype=torch.long)
#         agent_tensor = torch.tensor(agent_id, dtype=torch.long)
#         target_tensor = torch.tensor(target_id, dtype=torch.long)
#         lab_idx = self.lab_to_idx.get(lab_id, 0)
#         lab_tensor = torch.tensor(lab_idx, dtype=torch.long)
#         return feat_tensor, label_tensor, agent_tensor, target_tensor, lab_tensor, video_id, lab_id, start

# def train_model(cfg: Config):
#     print("\nğŸš€ [OFFICIAL TRAINING START] â€” Using FULL dataset\n")

#     train_df = pl.read_csv(cfg.train_csv)
#     video_ids = train_df["video_id"].to_list()
#     lab_ids = train_df["lab_id"].to_list()

#     dataset = MABeDataset(
#         video_ids=video_ids,
#         lab_ids=lab_ids,
#         track_dir=cfg.train_track_dir,
#         annot_dir=cfg.train_annot_dir,
#         window_size=cfg.window_size,
#         stride=cfg.stride,
#         class_to_idx=cfg.class_to_idx,
#         lab_to_idx=cfg.lab_to_idx,
#         is_test=False,
#         num_mice=cfg.num_mice
#     )

#     if len(dataset) == 0:
#         raise ValueError("â�Œ Dataset is empty!")

#     print(f"âœ… Total training samples: {len(dataset)}")

#     train_size = max(1, int(0.8 * len(dataset)))
#     val_size = len(dataset) - train_size
#     if val_size == 0:
#         val_size = 1
#         train_size = len(dataset) - 1

#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     try:
#         train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     except Exception as e:
#         print(f"âš ï¸�  Failed to use num_workers=2: {e}")
#         train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

#     model = MABeSTGCNv2(
#         num_classes=cfg.num_classes,
#         num_mice=cfg.num_mice,
#         in_channels=cfg.in_channels,
#         embed_dim=cfg.embed_dim,
#         lab_embed_dim=cfg.lab_embed_dim,
#         num_labs=cfg.num_labs
#     ).to(cfg.device)

#     print(f"\nğŸ§  MODEL DEVICE: {next(model.parameters()).device}\n")

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

#     best_val_f1 = 0.0

#     for epoch in range(cfg.num_epochs):
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Train]")
#         for batch_idx, (inputs, labels, agent_ids, target_ids, lab_ids, _, _, _) in enumerate(pbar):
#             inputs = inputs.to(cfg.device, non_blocking=True)
#             labels = labels.to(cfg.device, non_blocking=True)
#             agent_ids = agent_ids.to(cfg.device, non_blocking=True)
#             target_ids = target_ids.to(cfg.device, non_blocking=True)
#             lab_ids = lab_ids.to(cfg.device, non_blocking=True)

#             optimizer.zero_grad()
#             outputs = model(inputs, agent_ids, target_ids, lab_ids)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             train_total += labels.size(0)
#             train_correct += predicted.eq(labels).sum().item()

#             pbar.set_postfix({'Loss': loss.item(), 'Acc': train_correct/train_total})

#         # âœ… Validation with F1 Score
#         model.eval()
#         val_preds = []
#         val_targets = []

#         with torch.no_grad():
#             for inputs, labels, agent_ids, target_ids, lab_ids, _, _, _ in tqdm(val_loader, desc="Validation"):
#                 inputs = inputs.to(cfg.device, non_blocking=True)
#                 labels = labels.to(cfg.device, non_blocking=True)
#                 agent_ids = agent_ids.to(cfg.device, non_blocking=True)
#                 target_ids = target_ids.to(cfg.device, non_blocking=True)
#                 lab_ids = lab_ids.to(cfg.device, non_blocking=True)

#                 outputs = model(inputs, agent_ids, target_ids, lab_ids)
#                 _, predicted = outputs.max(1)

#                 # Chá»‰ tÃ­nh F1 cho cÃ¡c máº«u cÃ³ nhÃ£n há»£p lá»‡ (label != -1)
#                 mask = labels != -1
#                 if mask.sum() > 0:
#                     val_preds.extend(predicted[mask].cpu().numpy())
#                     val_targets.extend(labels[mask].cpu().numpy())

#         if len(val_preds) > 0 and len(val_targets) > 0:
#             val_f1 = f1_score(val_targets, val_preds, average='macro')
#             print(f"ğŸ�¯ Epoch {epoch+1}: Validation F1 Score = {val_f1:.4f}")
#         else:
#             val_f1 = 0.0
#             print(f"âš ï¸�  Epoch {epoch+1}: No valid validation samples!")

#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             torch.save(model.state_dict(), cfg.model_save_path)
#             print(f"âœ… Model saved with Val F1 = {val_f1:.4f}")

#     print(f"ğŸ�‰ Training completed. Best Validation F1: {best_val_f1:.4f}")

# def predict_and_submit(cfg: Config):
#     print("Loading test.csv...")
#     test_df = pl.read_csv(cfg.test_csv)
#     video_ids = test_df["video_id"].to_list()
#     lab_ids = test_df["lab_id"].to_list()

#     # âœ… Táº¡o mapping: video_id â†’ set of allowed (agent_str, target_str, action)
#     video_to_allowed_triples = {}

#     for row in test_df.to_dicts():
#         video_id = row["video_id"]
#         behaviors_str = row["behaviors_labeled"]
#         behaviors_list = safe_json_loads(behaviors_str)
#         allowed_triples = set()
#         for item in behaviors_list:
#             if isinstance(item, str):
#                 parts = [p.strip() for p in item.split(",")]
#                 if len(parts) == 3:
#                     agent_str, target_str, action = parts
#                     allowed_triples.add((agent_str, target_str, action))
#         video_to_allowed_triples[video_id] = allowed_triples

#     print("Creating test dataset...")
#     test_dataset = MABeDataset(
#         video_ids=video_ids,
#         lab_ids=lab_ids,
#         track_dir=cfg.test_track_dir,
#         annot_dir=cfg.train_annot_dir,
#         window_size=cfg.window_size,
#         stride=cfg.stride,
#         class_to_idx=cfg.class_to_idx,
#         lab_to_idx=cfg.lab_to_idx,
#         is_test=True,
#         num_mice=cfg.num_mice
#     )

#     if len(test_dataset) == 0:
#         raise ValueError("â�Œ Test dataset is empty!")

#     print(f"âœ… Total test samples: {len(test_dataset)}")

#     try:
#         test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     except Exception as e:
#         print(f"âš ï¸�  Failed to use num_workers=2: {e}")
#         test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

#     model = MABeSTGCNv2(
#         num_classes=cfg.num_classes,
#         num_mice=cfg.num_mice,
#         in_channels=cfg.in_channels,
#         embed_dim=cfg.embed_dim,
#         lab_embed_dim=cfg.lab_embed_dim,
#         num_labs=cfg.num_labs
#     ).to(cfg.device)
#     model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))
#     model.eval()

#     predictions = defaultdict(list)

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Inference"):
#             inputs, _, agent_ids, target_ids, lab_ids, video_ids_batch, lab_ids_str, start_frames = batch
#             inputs = inputs.to(cfg.device, non_blocking=True)
#             agent_ids = agent_ids.to(cfg.device, non_blocking=True)
#             target_ids = target_ids.to(cfg.device, non_blocking=True)
#             lab_ids = lab_ids.to(cfg.device, non_blocking=True)

#             outputs = model(inputs, agent_ids, target_ids, lab_ids)
#             _, predicted = outputs.max(1)

#             for i in range(len(video_ids_batch)):
#                 vid = int(video_ids_batch[i].item()) if torch.is_tensor(video_ids_batch[i]) else int(video_ids_batch[i])
#                 aid = int(agent_ids[i].item()) if torch.is_tensor(agent_ids[i]) else int(agent_ids[i])
#                 tid = int(target_ids[i].item()) if torch.is_tensor(target_ids[i]) else int(target_ids[i])
#                 start = int(start_frames[i].item()) if torch.is_tensor(start_frames[i]) else int(start_frames[i])
#                 mid_frame = start + cfg.window_size // 2
#                 label_idx = predicted[i].item()
#                 predictions[(vid, aid, tid)].append((mid_frame, label_idx))

#     records = []

#     for (video_id, agent_id, target_id), preds in predictions.items():
#         if len(preds) == 0:
#             continue

#         allowed_triples = video_to_allowed_triples.get(video_id, set())
#         agent_str = f"mouse{agent_id}"
#         target_str = f"mouse{target_id}"

#         # âœ… Lá»�c chá»‰ giá»¯ láº¡i cÃ¡c dá»± Ä‘oÃ¡n cÃ³ triple há»£p lá»‡
#         filtered_preds = []
#         for frame, label_idx in preds:
#             if label_idx == -1 or label_idx >= cfg.num_classes:
#                 continue
#             action = cfg.idx_to_class[label_idx]
#             triple = (agent_str, target_str, action)
#             if triple in allowed_triples:
#                 filtered_preds.append((frame, label_idx, action))

#         if len(filtered_preds) == 0:
#             continue

#         # âœ… Táº¡o segment tá»« filtered_preds
#         filtered_preds.sort(key=lambda x: x[0])  # sort by frame
#         current_label = filtered_preds[0][1]
#         current_action = filtered_preds[0][2]
#         start_frame = filtered_preds[0][0]

#         for i in range(1, len(filtered_preds)):
#             frame, label, action = filtered_preds[i]
#             if label != current_label or frame != filtered_preds[i-1][0] + 1:
#                 # Ä�Ã³ng segment cÅ©
#                 records.append((
#                     video_id,
#                     agent_str, target_str,
#                     cfg.idx_to_class[current_label],
#                     start_frame, frame
#                 ))
#                 # Má»Ÿ segment má»›i
#                 current_label = label
#                 start_frame = frame

#         # Ä�Ã³ng segment cuá»‘i
#         records.append((
#             video_id,
#             agent_str, target_str,
#             cfg.idx_to_class[current_label],
#             start_frame, filtered_preds[-1][0] + 1
#         ))

#     if len(records) == 0:
#         print("â�Œ No valid predictions generated!")
#         found = False
#         for vid, triples in video_to_allowed_triples.items():
#             if triples:
#                 agent_str, target_str, action = next(iter(triples))
#                 records = [(vid, agent_str, target_str, action, 0, 1)]
#                 found = True
#                 break
#         if not found:
#             default_label = list(cfg.class_to_idx.keys())[0]
#             records = [(0, "mouse1", "mouse2", default_label, 0, 1)]

#     submission_df = pl.DataFrame(
#         records,
#         schema={
#             "video_id": pl.Int64, "agent_id": pl.Utf8, "target_id": pl.Utf8,
#             "action": pl.Utf8, "start_frame": pl.Int64, "stop_frame": pl.Int64,
#         },
#         orient="row"
#     )

#     submission_df = submission_df.with_row_index("row_id")
#     submission_df.write_csv(cfg.submission_file)
#     print(f"âœ… Official Submission saved to {cfg.submission_file}")
#     print(f"ğŸ“Š Submission shape: {submission_df.shape}")
#     print("\nğŸ“‹ Sample predictions:")
#     print(submission_df.head(10))

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", choices=["train", "submit", "all"], default="all")
#     args = parser.parse_args()

#     cfg = Config()

#     if args.mode in ("train", "all"):
#         train_model(cfg)

#     if args.mode in ("submit", "all"):
#         predict_and_submit(cfg)

# if __name__ == "__main__":
#     main()


%%writefile stgcn_mabe_final_official_fixed_v8.py
import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, DefaultDict
from collections import defaultdict
import argparse

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

print("\n" + "="*70)
print(">>> ğŸ§ª SYSTEM CUDA & GPU CHECK (BEFORE ANYTHING ELSE)")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  â†’ GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("  â†’ NO GPU AVAILABLE. Training on CPU.")
print("<<<")
print("="*70 + "\n")

class Config:
    def __init__(self):
        self.data_root = Path(os.getenv("MABE_DATA_ROOT", "/kaggle/input/MABe-mouse-behavior-detection"))
        self.submission_file = "submission.csv"
        self.model_save_path = "stgcn_model.pth"
        self.window_size = 64
        self.stride = 32
        self.batch_size = 16
        self.num_epochs = 3
        self.lr = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_mice = 4
        self.in_channels = 2

        self.class_to_idx = self._collect_all_behaviors()
        self.num_classes = len(self.class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        logger.info(f"âœ… Found {self.num_classes} unique behaviors: {list(self.class_to_idx.keys())}")
        logger.info(f"ğŸ–¥ï¸�  Using device: {self.device}")

    def _collect_all_behaviors(self) -> Dict[str, int]:
        train_csv_path = self.data_root / "train.csv"
        if not train_csv_path.exists():
            return {"avoid": 0}

        df = pl.read_csv(train_csv_path)
        behaviors = set()

        for row in df.to_dicts():
            lab_id = row["lab_id"]
            video_id = row["video_id"]
            annot_path = self.data_root / "train_annotation" / lab_id / f"{video_id}.parquet"
            if not annot_path.exists():
                continue
            try:
                annot_df = pl.read_parquet(annot_path)
                if "action" in annot_df.columns:
                    behaviors.update(annot_df["action"].unique().to_list())
            except Exception as e:
                logger.warning(f"Failed to read {annot_path}: {e}")
                continue

        sorted_behaviors = sorted(list(behaviors))
        return {behavior: idx for idx, behavior in enumerate(sorted_behaviors)}

    @property
    def train_csv(self): return self.data_root / "train.csv"
    @property
    def test_csv(self): return self.data_root / "test.csv"
    @property
    def train_annot_dir(self): return self.data_root / "train_annotation"
    @property
    def train_track_dir(self): return self.data_root / "train_tracking"
    @property
    def test_track_dir(self): return self.data_root / "test_tracking"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# âœ… HÃ m safe_json_loads - tá»‘i Æ°u cho data cá»§a báº¡n
def safe_json_loads(s: Optional[str]) -> List[str]:
    if s is None: return []
    if isinstance(s, list): return [str(x) for x in s]
    if not isinstance(s, str): return []
    s = s.strip()
    if not s: return []
    try:
        loaded = json.loads(s)
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
        else:
            return []
    except Exception:
        try:
            # Thá»­ sá»­a dáº¥u nhÃ¡y Ä‘Æ¡n
            fixed = s.replace("'", '"')
            loaded = json.loads(fixed)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
            else:
                return []
        except Exception:
            return []

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        self.conv = nn.Conv1d(in_channels, out_channels * A.size(0), kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        n, kc, v = x.size()
        x = x.view(n, self.A.size(0), kc // self.A.size(0), v)
        x = torch.einsum('nvcv,vw->nwc', x, self.A)
        return x.contiguous()

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, temporal_kernel_size=9):
        super().__init__()
        self.sgc = SpatialGraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (temporal_kernel_size, 1), padding=(temporal_kernel_size//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)
        x = self.sgc(x)
        x = x.view(N, T, -1, V).permute(0, 2, 1, 3).contiguous()
        x = self.tcn(x)
        return self.relu(x)

class MABeSTGCN(nn.Module):
    def __init__(self, num_classes, num_mice=4, in_channels=2):
        super().__init__()
        self.num_mice = num_mice
        self.total_nodes = num_mice
        A = torch.ones(num_mice, num_mice)
        self.A = nn.Parameter(A, requires_grad=False)
        self.block1 = STGCNBlock(in_channels, 64, self.A)
        self.block2 = STGCNBlock(64, 128, self.A)
        self.block3 = STGCNBlock(128, 256, self.A)
        self.fc = nn.Linear(256 * num_mice + num_mice * num_mice, num_classes)

    def forward(self, x, pair_enc):
        N, T, M, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        mid_frame = T // 2
        x = x[:, :, mid_frame, :]
        x = x.reshape(N, -1)
        x = torch.cat([x, pair_enc], dim=1)
        return self.fc(x)

class MABeDataset(Dataset):
    def __init__(self, video_ids: List[int], lab_ids: List[str], track_dir: Path, annot_dir: Path,
                 window_size: int = 64, stride: int = 32, class_to_idx: Dict[str, int] = None, is_test: bool = False, num_mice: int = 4):
        self.video_ids = video_ids
        self.lab_ids = lab_ids
        self.track_dir = track_dir
        self.annot_dir = annot_dir
        self.window_size = window_size
        self.stride = stride
        self.class_to_idx = class_to_idx or {}
        self.is_test = is_test
        self.num_mice = num_mice
        self.samples = []
        self.track_cache = {}
        self.xy_cache = {}

        self.video_to_allowed_triples = {}
        if not self.is_test:
            train_meta_df = pl.read_csv(Config().train_csv)
            for row in train_meta_df.to_dicts():
                vid = row["video_id"]
                behaviors_str = row.get("behaviors_labeled", "")
                behaviors_list = safe_json_loads(behaviors_str)
                allowed_triples = set()
                for item in behaviors_list:
                    parts = [p.strip() for p in str(item).replace("'", "").split(",")]
                    if len(parts) == 3:
                        agent_str, target_str, action = parts
                        allowed_triples.add((agent_str, target_str, action))
                self.video_to_allowed_triples[vid] = allowed_triples

        self._build_samples()

    def _get_track_path(self, lab_id: str, video_id: int) -> Optional[Path]:
        path = self.track_dir / lab_id / f"{video_id}.parquet"
        if path.exists():
            return path
        return None

    def _build_samples(self):
        total_video = len(self.video_ids)
        logger.info(f"ğŸš€ Starting to process {total_video} videos...")

        count_track = 0
        count_parquet = 0
        count_valid_labels = 0

        for idx, (video_id, lab_id) in enumerate(tqdm(zip(self.video_ids, self.lab_ids), total=total_video, desc="ğŸ“‚ Loading & Processing Videos")):
            track_path = self._get_track_path(lab_id, video_id)
            if track_path is None:
                continue
            count_track += 1

            try:
                track_df = pl.read_parquet(track_path)
                self.track_cache[(lab_id, video_id)] = track_df
            except Exception as e:
                logger.warning(f"Failed to cache tracking for {lab_id}/{video_id}: {e}")
                continue

            required_cols = ["video_frame", "mouse_id", "x", "y"]
            if not all(col in track_df.columns for col in required_cols):
                logger.warning(f"Missing required columns in {track_path}")
                continue

            unique_mice_raw = track_df["mouse_id"].unique().to_list()
            unique_mice = []
            for m in unique_mice_raw:
                try:
                    mid = int(m)
                    unique_mice.append(mid)
                except (ValueError, TypeError):
                    logger.warning(f"Cannot convert mouse_id {m} to int, skipping.")
                    continue

            unique_mice = sorted(set(unique_mice))[:self.num_mice]
            if len(unique_mice) == 0:
                continue

            mouse_to_idx = {mouse: i for i, mouse in enumerate(unique_mice)}
            all_frames = sorted(track_df["video_frame"].unique().to_list())
            T = len(all_frames)
            frame_to_idx = {frame: i for i, frame in enumerate(all_frames)}

            xy = np.full((T, self.num_mice, 2), np.nan, dtype=np.float32)

            for row in track_df.to_dicts():
                frame = row["video_frame"]
                mouse_id = row["mouse_id"]
                try:
                    mouse_id = int(mouse_id)
                except (ValueError, TypeError):
                    continue
                x = row["x"]
                y = row["y"]
                if frame in frame_to_idx and mouse_id in mouse_to_idx:
                    t = frame_to_idx[frame]
                    m = mouse_to_idx[mouse_id]
                    xy[t, m, 0] = x
                    xy[t, m, 1] = y

            for m in range(self.num_mice):
                for t in range(1, T):
                    if np.isnan(xy[t, m, 0]):
                        xy[t, m] = xy[t-1, m]

            for m in range(self.num_mice):
                for t in range(T-2, -1, -1):
                    if np.isnan(xy[t, m, 0]):
                        xy[t, m] = xy[t+1, m]

            centroid = np.nanmean(xy, axis=1, keepdims=True)
            xy_norm = xy - centroid
            xy_norm = np.nan_to_num(xy_norm, nan=0.0)
            self.xy_cache[(lab_id, video_id)] = xy_norm

            if not self.is_test:
                annot_path = self.annot_dir / lab_id / f"{video_id}.parquet"
                if not annot_path.exists():
                    continue
                count_parquet += 1

                try:
                    annot_df = pl.read_parquet(annot_path)
                except Exception as e:
                    logger.warning(f"Failed to read annotation {annot_path}: {e}")
                    continue

                if "agent_id" not in annot_df.columns or "action" not in annot_df.columns:
                    continue

                unique_pairs = annot_df.select(["agent_id", "target_id"]).unique().to_dicts()
                if len(unique_pairs) == 0:
                    continue

                for pair in unique_pairs:
                    agent_id = pair["agent_id"]
                    target_id = pair["target_id"]
                    try:
                        agent_id = int(agent_id)
                        target_id = int(target_id)
                    except (ValueError, TypeError):
                        continue

                    pair_df = annot_df.filter((pl.col("agent_id") == agent_id) & (pl.col("target_id") == target_id))
                    if len(pair_df) == 0:
                        continue

                    labels = np.full(T, -1, dtype=np.int64)
                    for row in pair_df.to_dicts():
                        s, e, action = row["start_frame"], row["stop_frame"], row["action"]
                        if action not in self.class_to_idx:
                            continue

                        agent_str = f"mouse{agent_id}"
                        target_str = f"mouse{target_id}"
                        triple = (agent_str, target_str, action)
                        if triple not in self.video_to_allowed_triples.get(video_id, set()):
                            continue

                        labels[s:e] = self.class_to_idx[action]

                    for start in range(0, T - self.window_size + 1, self.stride):
                        window_labels = labels[start:start + self.window_size]
                        valid_labels = window_labels[window_labels != -1]
                        if len(valid_labels) > 0:
                            label = valid_labels[0]
                            self.samples.append((video_id, lab_id, agent_id, target_id, start, label))
                            count_valid_labels += 1
            else:
                for agent_id in unique_mice:
                    for target_id in unique_mice:
                        for start in range(0, T - self.window_size + 1, self.stride):
                            self.samples.append((video_id, lab_id, agent_id, target_id, start, -1))

        logger.info(f"âœ… Found {count_track} .parquet files in train_tracking/")
        if not self.is_test:
            logger.info(f"âœ… Found {count_parquet} .parquet files in train_annotation/")
        logger.info(f"âœ… Created {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, lab_id, agent_id, target_id, start, label = self.samples[idx]
        xy_norm = self.xy_cache[(lab_id, video_id)]
        window_xy = xy_norm[start:start + self.window_size]
        xy_tensor = torch.tensor(window_xy, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        try:
            agent_idx = int(agent_id) - 1
            target_idx = int(target_id) - 1
        except:
            agent_idx = 0
            target_idx = 0

        if not (0 <= agent_idx < self.num_mice and 0 <= target_idx < self.num_mice):
            agent_idx = 0
            target_idx = 0

        pair_encoding = torch.zeros(self.num_mice * self.num_mice, dtype=torch.float32)
        pair_encoding[agent_idx * self.num_mice + target_idx] = 1.0

        return xy_tensor, pair_encoding, label_tensor, video_id, lab_id, agent_id, target_id, start

def train_model(cfg: Config):
    print("\nğŸš€ [OFFICIAL TRAINING START] â€” Using FULL dataset\n")

    train_df = pl.read_csv(cfg.train_csv)
    video_ids = train_df["video_id"].to_list()
    lab_ids = train_df["lab_id"].to_list()

    dataset = MABeDataset(
        video_ids=video_ids,
        lab_ids=lab_ids,
        track_dir=cfg.train_track_dir,
        annot_dir=cfg.train_annot_dir,
        window_size=cfg.window_size,
        stride=cfg.stride,
        class_to_idx=cfg.class_to_idx,
        is_test=False,
        num_mice=cfg.num_mice
    )

    if len(dataset) == 0:
        raise ValueError("â�Œ Dataset is empty!")

    logger.info(f"âœ… Total training samples: {len(dataset)}")

    all_labels = [sample[-1] for sample in dataset.samples if sample[-1] != -1]
    if len(all_labels) == 0:
        class_weights = None
    else:
        class_counts = np.bincount(all_labels, minlength=cfg.num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(cfg.device)

    train_size = max(1, int(0.8 * len(dataset)))
    val_size = len(dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    try:
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    except Exception as e:
        logger.warning(f"Failed to use num_workers=2: {e}")
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = MABeSTGCN(
        num_classes=cfg.num_classes,
        num_mice=cfg.num_mice,
        in_channels=cfg.in_channels
    ).to(cfg.device)

    print("\n" + "="*60)
    print("ğŸ§  MODEL DEVICE:", next(model.parameters()).device)
    print("="*60 + "\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc = 0.0

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Train]")
        for batch_idx, (inputs, pair_enc, labels, _, _, _, _, _) in enumerate(pbar):  # âœ… Sá»¬A: 8 giÃ¡ trá»‹
            inputs = inputs.to(cfg.device, non_blocking=True)
            pair_enc = pair_enc.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs, pair_enc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'Loss': loss.item(), 'Acc': train_correct/train_total})

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, pair_enc, labels, _, _, _, _, _ in tqdm(val_loader, desc="Validation"):  # âœ… Sá»¬A: 8 giÃ¡ trá»‹
                inputs = inputs.to(cfg.device, non_blocking=True)
                pair_enc = pair_enc.to(cfg.device, non_blocking=True)
                labels = labels.to(cfg.device, non_blocking=True)
                outputs = model(inputs, pair_enc)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.model_save_path)
            logger.info(f"âœ… Model saved with Val Acc = {val_acc:.4f}")

    logger.info(f"ğŸ�‰ Training completed. Best Val Acc: {best_val_acc:.4f}")

def predict_and_submit(cfg: Config):
    logger.info("Loading test.csv...")
    test_df = pl.read_csv(cfg.test_csv)
    video_ids = test_df["video_id"].to_list()
    lab_ids = test_df["lab_id"].to_list()

    video_to_allowed_triples = {}
    for row in test_df.to_dicts():
        video_id = row["video_id"]
        behaviors_str = row["behaviors_labeled"]
        behaviors_list = safe_json_loads(behaviors_str)
        allowed_triples = set()
        for item in behaviors_list:
            parts = [p.strip() for p in str(item).replace("'", "").split(",")]
            if len(parts) == 3:
                agent_str, target_str, action = parts
                allowed_triples.add((agent_str, target_str, action))
        video_to_allowed_triples[video_id] = allowed_triples

    logger.info("Creating test dataset...")
    test_dataset = MABeDataset(
        video_ids=video_ids,
        lab_ids=lab_ids,
        track_dir=cfg.test_track_dir,
        annot_dir=cfg.train_annot_dir,
        window_size=cfg.window_size,
        stride=cfg.stride,
        class_to_idx=cfg.class_to_idx,
        is_test=True,
        num_mice=cfg.num_mice
    )

    if len(test_dataset) == 0:
        raise ValueError("â�Œ Test dataset is empty!")

    logger.info(f"âœ… Total test samples: {len(test_dataset)}")

    try:
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    except Exception as e:
        logger.warning(f"Failed to use num_workers=2: {e}")
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = MABeSTGCN(
        num_classes=cfg.num_classes,
        num_mice=cfg.num_mice,
        in_channels=cfg.in_channels
    ).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))
    model.eval()

    predictions = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            inputs, pair_enc, _, video_ids_batch, lab_ids_batch, agent_ids_batch, target_ids_batch, start_frames = batch
            inputs = inputs.to(cfg.device, non_blocking=True)
            pair_enc = pair_enc.to(cfg.device, non_blocking=True)
            outputs = model(inputs, pair_enc)
            _, predicted = outputs.max(1)

            for i in range(len(video_ids_batch)):
                vid = int(video_ids_batch[i])
                aid = int(agent_ids_batch[i])
                tid = int(target_ids_batch[i])
                start = int(start_frames[i])
                mid_frame = start + cfg.window_size // 2
                label_idx = predicted[i].item()
                predictions[(vid, aid, tid)].append((mid_frame, label_idx))

    records = []

    for (video_id, agent_id, target_id), preds in predictions.items():
        if len(preds) == 0:
            continue

        allowed_triples = video_to_allowed_triples.get(video_id, set())
        agent_str = f"mouse{agent_id}"
        target_str = f"mouse{target_id}"

        preds = sorted(preds)
        current_label = preds[0][1]
        current_action = cfg.idx_to_class[current_label]
        current_triple = (agent_str, target_str, current_action)

        if current_triple not in allowed_triples:
            current_label = -1

        start_frame = preds[0][0]

        for i in range(1, len(preds)):
            frame, label = preds[i]
            action = cfg.idx_to_class[label]
            triple = (agent_str, target_str, action)

            if label != current_label or frame != preds[i-1][0] + 1 or triple not in allowed_triples:
                if current_label != -1:
                    final_triple = (agent_str, target_str, cfg.idx_to_class[current_label])
                    if final_triple in allowed_triples:
                        records.append((
                            video_id,
                            agent_str, target_str,
                            cfg.idx_to_class[current_label],
                            start_frame, frame
                        ))
                current_label = label if triple in allowed_triples else -1
                start_frame = frame

        if current_label != -1:
            final_triple = (agent_str, target_str, cfg.idx_to_class[current_label])
            if final_triple in allowed_triples:
                records.append((
                    video_id,
                    agent_str, target_str,
                    cfg.idx_to_class[current_label],
                    start_frame, preds[-1][0] + 1
                ))

    if len(records) == 0:
        logger.warning("â�Œ No valid predictions generated!")
        found = False
        for vid, triples in video_to_allowed_triples.items():
            if triples:
                agent_str, target_str, action = next(iter(triples))
                records = [(vid, agent_str, target_str, action, 0, 1)]
                found = True
                break
        if not found:
            default_label = list(cfg.class_to_idx.keys())[0]
            records = [(0, "mouse1", "mouse2", default_label, 0, 1)]

    submission_df = pl.DataFrame(
        records,
        schema={
            "video_id": pl.Int64, "agent_id": pl.Utf8, "target_id": pl.Utf8,
            "action": pl.Utf8, "start_frame": pl.Int64, "stop_frame": pl.Int64,
        },
        orient="row"
    )

    submission_df = submission_df.with_row_index("row_id")
    submission_df.write_csv(cfg.submission_file)
    logger.info(f"âœ… Official Submission saved to {cfg.submission_file}")
    logger.info(f"ğŸ“Š Submission shape: {submission_df.shape}")
    print("\nğŸ“‹ Sample predictions:")
    print(submission_df.head(10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "submit", "all"], default="all")
    args = parser.parse_args()

    cfg = Config()

    if args.mode in ("train", "all"):
        train_model(cfg)

    if args.mode in ("submit", "all"):
        predict_and_submit(cfg)

if __name__ == "__main__":
    main()


!python stgcn_mabe_final_official_fixed_v8.py --mode all


!

