import os
import shutil
import sys

# ==========================================
# 1. 环境初始化 
# ==========================================
print("Step 1/3: Initializing Environment...")

if os.getcwd() != "/kaggle/working":
    os.chdir("/kaggle/working")
print(f"Current working directory: {os.getcwd()}")

base_dir = "/kaggle/working/code"
dirs = {
    "root": base_dir,
    "configs": os.path.join(base_dir, "configs"),
    "models": os.path.join(base_dir, "models"),
    "data": os.path.join(base_dir, "data")
}

# 1.1 清理并重建目录
if os.path.exists(base_dir): 
    print("Cleaning old directory...")
    shutil.rmtree(base_dir)
    
for d in dirs.values():
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "__init__.py"), 'w') as f: pass
print("Directory structure created.")

# 1.2 复制 Data (使用已知路径)
source_data_path = "/kaggle/input/dinoehst/pytorch/default/1/kaggle-landmark-2021-1st-place-main - 副本/data"

if os.path.exists(source_data_path):
    print(f"Copying data from: {source_data_path}")
    shutil.copytree(source_data_path, dirs["data"], dirs_exist_ok=True)
    print(" Data module copied successfully.")
else:
    print(f" Path not found: {source_data_path}")
    for root, dirs, files in os.walk("/kaggle/input"):
        if "data" in dirs and "ch_ds_1.py" in os.listdir(os.path.join(root, "data")):
            src = os.path.join(root, "data")
            print(f"Found alternative path: {src}")
            shutil.copytree(src, dirs["data"], dirs_exist_ok=True)
            break

print("Installing dependencies...")
os.system("pip install -q timm --upgrade")
print("Dependencies installed.")

# ==========================================
# 2. 写入配置文件
# ==========================================
print("Step 2/3: Generating Configs...")

# default_config.py
with open(os.path.join(dirs["configs"], "default_config.py"), "w") as f:
    f.write("""
from types import SimpleNamespace
cfg = SimpleNamespace(**{})
cfg.suffix = ".jpg"
cfg.normalization = 'imagenet'
basic_cfg = cfg
""")

with open(os.path.join(dirs["configs"], "cfg_kaggle_dino.py"), "w") as f:
    f.write("""
from default_config import basic_cfg
import albumentations as A
cfg = basic_cfg
cfg.model = "ch_mdl_dino"
cfg.backbone = "vit_base_patch14_dinov2.lvd142m"
cfg.img_size = (448, 448)
cfg.embedding_size = 512
cfg.batch_size = 32
cfg.lr = 1e-4
cfg.optimizer = "adamw"
cfg.epochs = 1
cfg.freeze_backbone = True
cfg.normalization = 'imagenet' 
cfg.arcface_m_x = 0.45; cfg.arcface_m_y = 0.05; cfg.arcface_s = 45.0
image_size = cfg.img_size[0]
cfg.train_aug = A.Compose([A.HorizontalFlip(p=0.5), A.Resize(image_size, image_size)])
cfg.val_aug = A.Compose([A.Resize(image_size, image_size)])
""")

# ==========================================
# 3. 写入模型文件 
# ==========================================
print("Step 3/3: Generating Model...")

model_code = """
import timm
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        logprobs = F.log_softmax(x.float(), dim=-1)
        loss = -logprobs * target.float()
        return loss.sum(-1).mean()

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_f, out_f, k=3):
        super().__init__(); self.weight = nn.Parameter(torch.FloatTensor(out_f*k, in_f)); nn.init.xavier_uniform_(self.weight); self.k, self.out_features = k, out_f
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2); return cosine

class ArcFaceLossAdaptiveMargin(nn.Module):
    def __init__(self, margins, n_classes, s=30.0):
        super().__init__(); self.crit, self.s, self.margins, self.out_dim = DenseCrossEntropy(), s, margins, n_classes
    def forward(self, logits, labels):
        ms = self.margins[labels.cpu().numpy()] if isinstance(self.margins, np.ndarray) else np.array([0.5]*len(labels))
        cos_m, sin_m, th, mm = [torch.from_numpy(np.cos(ms)).float().to(logits.device), torch.from_numpy(np.sin(ms)).float().to(logits.device), torch.from_numpy(np.cos(math.pi-ms)).float().to(logits.device), torch.from_numpy(np.sin(math.pi-ms)*ms).float().to(logits.device)]
        labels_ohe = F.one_hot(labels, self.out_dim).float(); cosine = logits.float(); sine = torch.sqrt(1.0-cosine.pow(2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1); phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels_ohe * phi) + ((1.0-labels_ohe)*cosine); output *= self.s; return self.crit(output, labels_ohe)

class Net(nn.Module):
    def __init__(self, cfg, dataset):
        super(Net, self).__init__(); self.cfg, self.n_classes = cfg, cfg.n_classes
        print(f">>> Loading Backbone: {cfg.backbone}")
        self.backbone = timm.create_model(cfg.backbone, pretrained=True, num_classes=0, img_size=cfg.img_size[0])
        if getattr(cfg, 'freeze_backbone', False): 
            print("Freezing backbone...")
            for p in self.backbone.parameters(): p.requires_grad = False
        self.neck = nn.Sequential(nn.Linear(self.backbone.num_features, cfg.embedding_size, bias=True), nn.BatchNorm1d(cfg.embedding_size))
        if not hasattr(cfg, 'headless') or not cfg.headless:
            self.head = ArcMarginProduct_subcenter(cfg.embedding_size, self.n_classes)
            margins = getattr(dataset, 'margins', None)
            self.loss_fn = ArcFaceLossAdaptiveMargin(margins, self.n_classes, cfg.arcface_s)

    def forward(self, batch):
        x = batch['input']
        features = self.backbone.forward_features(x)
        # 修复 IndexError: 兼容 Tensor 和 Dict 输出
        if isinstance(features, dict): x_global = features['x_norm_clstoken']
        else: x_global = features[:, 0]
        x_emb = self.neck(x_global)
        if hasattr(self.cfg, 'headless') and self.cfg.headless: return {'embeddings': x_emb}
        logits = self.head(x_emb)
        if self.training: return {'loss': self.loss_fn(logits, batch['target'].long())}
        else: return {'loss': torch.zeros(1), 'embeddings': x_emb}
"""

with open(os.path.join(dirs["models"], "ch_mdl_dino.py"), "w") as f:
    f.write(model_code)

print("All files ready. Please run Cell 2.")


# import sys
# import os
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
# import importlib.util
# import math
# from torch import nn
# from torch.nn import functional as F
# import numpy as np
# import albumentations as A
# from sklearn.model_selection import train_test_split 

# # Setup paths and change working directory
# base_dir = "/kaggle/working/code"
# if os.path.exists(base_dir):
#     os.chdir(base_dir)
#     sys.path.insert(0, base_dir)
#     sys.path.insert(0, os.path.join(base_dir, "configs"))

# # Dynamic module loader
# def load_module(name, path):
#     spec = importlib.util.spec_from_file_location(name, path)
#     mod = importlib.util.module_from_spec(spec)
#     sys.modules[name] = mod
#     spec.loader.exec_module(mod)
#     return mod

# try:
#     # 1. Patch ch_ds_1.py to fix KeyError and ensure stability
#     ds_path = os.path.join(base_dir, "data", "ch_ds_1.py")
#     if os.path.exists(ds_path):
#         with open(ds_path, 'r') as f: lines = f.readlines()
#         with open(ds_path, 'w') as f:
#             for line in lines:
#                 # Comment out problematic lines involving 'landmarks' column
#                 if "self.df['landmarks'].apply" in line or "self.df['landmarks'].fillna" in line:
#                     f.write("# " + line) 
#                 else:
#                     f.write(line)
#         print(" Patched data/ch_ds_1.py")

#     # 2. Load Modules
#     load_module("default_config", os.path.join(base_dir, "configs", "default_config.py"))
#     cfg_module = load_module("cfg_kaggle_dino", os.path.join(base_dir, "configs", "cfg_kaggle_dino.py"))
#     cfg = cfg_module.cfg
    
#     # Ensure critical config keys exist
#     if not hasattr(cfg, 'suffix'): cfg.suffix = ".jpg"
    
#     ds_module = load_module("ch_ds_1_patched", ds_path) 
#     CustomDataset = ds_module.CustomDataset
    
#     mdl_path = os.path.join(base_dir, "models", "ch_mdl_dino.py")
#     Net = load_module("ch_mdl_dino", mdl_path).Net

#     # 3. Data Preparation & Cleaning
#     train_csv = "/kaggle/input/landmark-recognition-2021/train.csv"
#     train_img_dir = "/kaggle/input/landmark-recognition-2021/train/"
    
#     print(f"\n Reading CSV and creating data pool...")
#     # Load first 150,000 rows (Balanced pool)
#     raw_df = pd.read_csv(train_csv).iloc[:150000] 
#     print(f"Original Pool Size: {len(raw_df)}")

#     # === Data Cleaning Step ===
#     print(" Cleaning data: Removing classes with < 15 samples...")
#     counts = raw_df['landmark_id'].value_counts()
#     valid_landmarks = counts[counts >= 15].index
#     filtered_df = raw_df[raw_df['landmark_id'].isin(valid_landmarks)].copy()
#     print(f"Cleaned Data Size: {len(filtered_df)}")
#     print(f"Valid Classes: {len(valid_landmarks)}")
    
#     # Label Encoding
#     unique_landmarks = filtered_df['landmark_id'].unique()
#     landmark_map = { lid: i for i, lid in enumerate(unique_landmarks) }
#     filtered_df['target'] = filtered_df['landmark_id'].map(landmark_map)
    
#     # Train/Val Split (Stratified)
#     train_df, val_df = train_test_split(filtered_df, test_size=0.1, random_state=42, stratify=filtered_df['landmark_id'])
#     print(f" Training Set: {len(train_df)} | Validation Set: {len(val_df)}")
    
#     # Update Config
#     cfg.n_classes = len(unique_landmarks)
#     cfg.data_folder = train_img_dir
#     cfg.val_data_folder = train_img_dir # Fix: Add val_data_folder
#     cfg.landmark_id2class_id = pd.DataFrame({'landmark_id': unique_landmarks, 'class_id': range(len(unique_landmarks))})
    
#     # 4. DataLoaders
#     train_ds = CustomDataset(train_df, cfg, cfg.train_aug, mode='train')
#     train_loader = DataLoader(
#         train_ds, 
#         batch_size=cfg.batch_size, 
#         shuffle=True, 
#         num_workers=2,            # 尝试开启双进程
#         pin_memory=True,          # 加速 GPU 传输
#         persistent_workers=True,  # 核心：保持进程存活，避免反复启动卡顿
#         prefetch_factor=2,        # 限制预取数量，防止内存溢出卡死
#         drop_last=True            # 丢弃最后一个不完整的 Batch，有时候能增加稳定性
#     )
    
#     # 验证集 Loader
#     val_ds = CustomDataset(val_df, cfg, cfg.val_aug, mode='val')
#     val_loader = DataLoader(
#         val_ds, 
#         batch_size=cfg.batch_size, 
#         shuffle=False, 
#         num_workers=2,            # 验证集也开双进程
#         pin_memory=True, 
#         persistent_workers=True,
#         prefetch_factor=2
#     )

#     # 5. Model Setup
#     device = torch.device("cuda")
#     model = Net(cfg, train_ds)
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
#     # 6. Training Loop (for 1 epoch)
#     print(f"\n Starting Training for {cfg.epochs} Epoch(s)...")
    
#     history = {'train_loss': [], 'val_acc': []}
#     for epoch in range(cfg.epochs):
#         # Train
#         model.train()
#         train_loss = 0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
#         for batch in pbar:
#             inputs, targets = batch['input'].to(device), batch['target'].to(device)
#             loss = model({'input': inputs, 'target': targets})['loss']
#             optimizer.zero_grad(); loss.backward(); optimizer.step()
#             train_loss += loss.item()
#             pbar.set_postfix({'loss': f"{loss.item():.4f}"})
#         avg_train_loss = train_loss / len(train_loader)
#         history['train_loss'].append(avg_train_loss)
        
#         # Validation
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]"):
#                 inputs, targets = batch['input'].to(device), batch['target'].to(device)
                
#                 # --- FIX: Robust Feature Extraction ---
#                 features = model.backbone.forward_features(inputs)
                
#                 # Handle different timm versions (Dict vs Tensor)
#                 if isinstance(features, dict):
#                     x_global = features['x_norm_clstoken']
#                 else:
#                     # New timm returns [B, N, C], CLS token is at index 0
#                     x_global = features[:, 0]
                
#                 logits = model.head(model.neck(x_global))

#                 _, predicted = torch.max(logits.data, 1)
#                 total += batch['target'].size(0)
#                 correct += (predicted == batch['target'].to(device)).sum().item()
        
#         val_acc = 100 * correct / total
#         history['val_acc'].append(val_acc)
        
#         print(f" Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Accuracy = {val_acc:.2f}%")

#     # Save checkpoint after the first epoch
#     torch.save(model.state_dict(), "dino_finetuned_epoch1.pth")
#     print("\n Training Finished & Checkpoint 'dino_finetuned_epoch1.pth' Saved!")

# except Exception as e:
#     import traceback
#     traceback.print_exc()
#     print(f" Error: {e}")


import sys
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import importlib.util
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split 

# ==========================================
# 配置区域
# ==========================================
# 1. 权重路径 
LOAD_FROM = "/kaggle/input/dino-finetuned-epoch1-pth/dino_finetuned_epoch1.pth" 

# 2. 续训参数
MORE_EPOCHS = 4  
NEW_LR = 5e-5  

base_dir = "/kaggle/working/code"
dirs = {
    "root": base_dir,
    "configs": os.path.join(base_dir, "configs"),
    "models": os.path.join(base_dir, "models"),
    "data": os.path.join(base_dir, "data")
}

if os.path.exists(base_dir):
    os.chdir(base_dir)
    sys.path.insert(0, base_dir)
    sys.path.insert(0, dirs["configs"])

ds_path = os.path.join(dirs["data"], "ch_ds_1.py")
if os.path.exists(ds_path):
    with open(ds_path, 'r') as f:
        lines = f.readlines()
    with open(ds_path, 'w') as f:
        for line in lines:
            if "self.df['landmarks'].apply" in line or "self.df['landmarks'].fillna" in line:
                f.write("# " + line) 
            else:
                f.write(line)
    print(" repaired ch_ds_1.py")
else:
    print(" warning: cannot find ch_ds_1.py")

# 动态加载函数
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

try:
    # 1. 加载模块
    load_module("default_config", os.path.join(dirs["configs"], "default_config.py"))
    cfg_module = load_module("cfg_kaggle_dino", os.path.join(dirs["configs"], "cfg_kaggle_dino.py"))
    cfg = cfg_module.cfg

    if not hasattr(cfg, 'suffix'): cfg.suffix = ".jpg"

    ds_module = load_module("ch_ds_1_fixed", ds_path) 
    CustomDataset = ds_module.CustomDataset
    
    mdl_path = os.path.join(dirs["models"], "ch_mdl_dino.py")
    Net = load_module("ch_mdl_dino", mdl_path).Net

    # 2. 准备数据 
    print(" Preparing Data...")
    train_csv = "/kaggle/input/landmark-recognition-2021/train.csv"
    train_img_dir = "/kaggle/input/landmark-recognition-2021/train/"
 
    # 读取前 150000 条
    raw_df = pd.read_csv(train_csv).iloc[:150000] 
    
    # 执行同样的筛选逻辑 (>=15)
    counts = raw_df['landmark_id'].value_counts()
    valid_landmarks = counts[counts >= 15].index
    filtered_df = raw_df[raw_df['landmark_id'].isin(valid_landmarks)].copy()
    
    # 标签映射
    unique_landmarks = filtered_df['landmark_id'].unique()
    landmark_map = { lid: i for i, lid in enumerate(unique_landmarks) }
    filtered_df['target'] = filtered_df['landmark_id'].map(landmark_map)
 
    # 划分训练/验证集
    train_df, val_df = train_test_split(filtered_df, test_size=0.1, random_state=42, stratify=filtered_df['landmark_id'])

    # 更新 Config
    cfg.n_classes = len(unique_landmarks)
    cfg.data_folder = train_img_dir
    cfg.val_data_folder = train_img_dir 
    cfg.landmark_id2class_id = pd.DataFrame({'landmark_id': unique_landmarks, 'class_id': range(len(unique_landmarks))})
    
    print(f" Training on {len(train_df)} samples, Validation on {len(val_df)} samples")

    # 3. 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_ds = CustomDataset(train_df, cfg, cfg.train_aug, mode='train')
    model = Net(cfg, dummy_ds)
    model.to(device)
    
    # 4. 加载权重
    print(f" Loading weights from: {LOAD_FROM}")
    if os.path.exists(LOAD_FROM):
        state_dict = torch.load(LOAD_FROM, map_location=device)
        model.load_state_dict(state_dict)
        print(" Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Cannot find weight file: {LOAD_FROM}")

    # 5. 准备 Loader 和 Optimizer
    train_loader = DataLoader(dummy_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    
    val_ds = CustomDataset(val_df, cfg, cfg.val_aug, mode='val')
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=NEW_LR)
    
    # 6. 续训循环
    print(f"\n Starting {MORE_EPOCHS} more Epochs...")
    
    for epoch in range(MORE_EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Resume Epoch {epoch+1}/{MORE_EPOCHS}")
        
        for batch in pbar:
            inputs, targets = batch['input'].to(device), batch['target'].to(device)
            loss = model({'input': inputs, 'target': targets})['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # --- Validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets = batch['input'].to(device), batch['target'].to(device)
                
                features = model.backbone.forward_features(inputs)
                if isinstance(features, dict):
                    x_global = features['x_norm_clstoken']
                else:
                    x_global = features[:, 0, :]
                    
                logits = model.head(model.neck(x_global))
                _, predicted = torch.max(logits.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_acc = 100 * correct / total
        print(f" Epoch {epoch+1} Finished: Train Loss = {train_loss/len(train_loader):.4f} | Val Accuracy = {val_acc:.2f}%")
 
        epoch_save_path = f"dino_resume_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_save_path)
        print(f" Saved checkpoint: {epoch_save_path}")

    # 最终保存
    torch.save(model.state_dict(), "dino_finetuned_final_resume.pth")
    print(" All Done! Final model saved.")

except Exception as e:
    import traceback
    traceback.print_exc()

