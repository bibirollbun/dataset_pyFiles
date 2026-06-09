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
copy_project_to_working("jpx-trained-model") 


import sys
sys.path.append("/kaggle/input/jpx-tokyo-stock-exchange-prediction")

# 1. 切換到專案資料夾 (假設您是用 git clone 或解壓縮產生的這個資料夾)
%cd /kaggle/working/jpx_transformer_kaggle_1766129828

# 2. 設定 PYTHONPATH (確保能 import src 下的模組)
import sys
sys.path.append("src")

# 3. 執行 submit.py
!python src/jpx_transformer/submit.py \
    --model_path /kaggle/input/jpx-trained-model/artifacts/model.pt \
    --prep_path /kaggle/input/jpx-trained-model/artifacts/preprocess.json \
    --device cuda

