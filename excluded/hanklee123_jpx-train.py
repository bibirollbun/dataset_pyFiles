import shutil
import os
from pathlib import Path

def copy_project_to_working(dataset_name):
    """
    將指定的 Dataset 內容複製到 /kaggle/working
    dataset_name: 您上傳的 Dataset 名稱 (在 /kaggle/input 下的資料夾名)
    """
    # 來源路徑 (唯讀)
    source_root = Path(f"/kaggle/input/{dataset_name}")
    
    # 目的路徑 (可讀寫)
    # 這裡我們直接複製到 working 根目錄，或者您可以指定一個子資料夾
    dest_root = Path("/kaggle/working")
    
    print(f"正在從 {source_root} 複製檔案到 {dest_root} ...")
    
    if not source_root.exists():
        print(f"錯誤: 找不到來源路徑 {source_root}")
        return

    # 使用 shutil.copytree 遞迴複製
    # dirs_exist_ok=True 允許目的地已存在 (會覆蓋同名檔案)
    try:
        # 注意：如果 dataset 結構是 /kaggle/input/my-dataset/src/...
        # 我們希望複製後變成 /kaggle/working/src/...
        # 所以我們遍歷 dataset 下的第一層目錄
        for item in source_root.iterdir():
            dest_path = dest_root / item.name
            if item.is_dir():
                if dest_path.exists():
                    shutil.rmtree(dest_path) # 先刪除舊的以確保乾淨
                shutil.copytree(item, dest_path)
            else:
                shutil.copy2(item, dest_path)
        print("複製完成！")
        
        # 列出目前 working 目錄結構以確認
        print("\n目前 /kaggle/working 下的檔案：")
        for p in dest_root.iterdir():
            print(f" - {p.name}")
            
    except Exception as e:
        print(f"複製過程中發生錯誤: {e}")

# --- 使用範例 ---
# 請將 'jpx-transformer-code' 替換成您實際的 Dataset 名稱
# 您可以在 Notebook 右側的 'Data' 面板看到正確的名稱
copy_project_to_working("jpx-test") 


%%writefile configs/transformer_small.yaml
seed: 42
device: "cuda"
seq_len: 60
train_years: [2017, 2021]
valid_years: [2022, 2022]
max_stocks: 2500
use_sector_embedding: true
feature_columns: ["log_return_1", "hl_spread", "oc_return", "SMA_5_dist", "SMA_10_dist", "SMA_20_dist", "SMA_60_dist", "Vol_20", "RSI_14", "MACD_norm", "MACD_Signal_norm", "BB_Upper_dist", "BB_Lower_dist", "PC_Ratio_Vol", "PC_Ratio_OI", "EarningsYield", "BookToMarket"]
target_column: "Target"
d_model: 64
n_heads: 4
n_layers: 4
dropout: 0.2
ff_mult: 4
batch_size: 2048
num_workers: 4
epochs: 20
lr: 0.0001
weight_decay: 0.05
grad_clip: 0.5
patience: 5
artifact_dir: "artifacts/exp3"


import os
import sys

# 1. 切換到 working 目錄 
# 假設複製後 src 資料夾直接在 /kaggle/working/src
os.chdir("/kaggle/working/jpx_transformer_kaggle_1766129828")

!python src/jpx_transformer/train.py --config configs/transformer_small.yaml




