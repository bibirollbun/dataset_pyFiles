!cp -r /kaggle/input/code-focus/FOCUS/ /kaggle/working/
import os
!cd /kaggle/working


# !rm -rf /kaggle/working/FOCUS


!apt-get update -qq
!apt-get install openjdk-11-jdk-headless -y

!pip uninstall -y pyspark
!pip install pyspark==3.5.1



# # 5. Thiáº¿t láº­p biáº¿n mÃ´i trÆ°á»�ng
# import os
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/opt/spark"
# os.environ["PATH"] += ":/opt/spark/bin"



import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkTestNew") \
    .master("local[*]") \
    .getOrCreate()

print("Spark version:", spark.version)



#ThÃªm cá»™t case_id vÃ  sile_id: case_id: mÃ£ ngÆ°á»�i, silde_id: mÃ£ áº£nh (data khÃ´ng cÃ³)
import pandas as pd
import os
csv_src = "/kaggle/input/ubc-ocean/UBC-OCEAN/train.csv"
csv_fixed = "/kaggle/working/FOCUS/train.csv"
if not os.path.exists(csv_fixed):  
    df = pd.read_csv(csv_src)
    df["case_id"] = df["image_id"].astype(str)
    df["slide_id"] = df["image_id"].astype(str)
    df.to_csv(csv_fixed, index=False)
print("âœ… CSV fixed saved at:", csv_fixed)


# # Báº¯t buá»™c
# !cp -r /kaggle/input/conch-source-code/CONCH-main/conch /kaggle/working/


#Báº¯t buá»™c

!mkdir -p /kaggle/working/FOCUS/ckpts
!mkdir -p /kaggle/working/FOCUS/features
# !mkdir -p /kaggle/working/FOCUS/features/features.csv

!cp /kaggle/input/conch-ckpts/conch.pth /kaggle/working/FOCUS/ckpts/conch.pth



# #TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng áº£nh
!cp -rn /kaggle/input/code-focus/FOCUS/ /kaggle/working/
project_dir = '/kaggle/working/FOCUS'
os.chdir(project_dir)
!mkdir -p results/FOCUS/conch/

# !apt-get install -y openslide-tools
# !pip install openslide-python


# !python /kaggle/working/FOCUS/map_reduce_features.py \
#   --csv_path /kaggle/working/FOCUS/train.csv \
#   --source_folder /kaggle/input/ubc-ocean/UBC-OCEAN/train_thumbnails \
#   --output_csv /kaggle/working/FOCUS/features/features.csv \
#   --features_dir /kaggle/working/FOCUS/features \
#   --ckpt_path /kaggle/working/FOCUS/ckpts/conch.pth \
#   2>&1 | tee spark_output.log

!spark-submit \
  --master local[*] \
  --driver-memory 8g \
  --executor-memory 6g \
  /kaggle/working/FOCUS/map_reduce_features.py \
  --csv_path /kaggle/working/FOCUS/train.csv \
  --source_folder /kaggle/input/ubc-ocean/UBC-OCEAN/train_thumbnails \
  --output_csv /kaggle/working/FOCUS/features/features.csv \
  --features_dir /kaggle/working/FOCUS/features \
  --ckpt_path /kaggle/working/FOCUS/ckpts/conch.pth \
  2>&1 | tee spark_output.log



# #Thay Ä‘á»•i (táº¡o file csv má»›i chá»‰ chá»©a cÃ¡c id Ä‘Ã£ trÃ­ch xuáº¥t)

# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, train_test_split

# # --- 1. Cáº¤U HÃŒNH ---
# # Ä�Æ°á»�ng dáº«n Ä‘áº¿n file CSV chá»©a danh sÃ¡ch cÃ¡c áº£nh Ä‘Ã£ Ä‘Æ°á»£c trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
# # (Náº¿u báº¡n Ä‘Ã£ táº¡o train_cleaned.csv thÃ¬ dÃ¹ng nÃ³, náº¿u khÃ´ng thÃ¬ dÃ¹ng train.csv gá»‘c)
# csv_path = '/kaggle/input/UBC-OCEAN/train.csv'

# # ThÆ° má»¥c Ä‘á»ƒ lÆ°u 8 file split CSV
# output_dir = '/kaggle/working/FOCUS/splits/UBC-OCEAN_debug_8folds/'

# # Sá»‘ lÆ°á»£ng fold cáº§n táº¡o
# n_splits = 8

# # Tá»‰ lá»‡ dá»¯ liá»‡u dÃ nh cho táº­p validation (vÃ­ dá»¥: 15% tá»« táº­p train)
# val_size = 0.15 

# # --- 2. Táº O THÆ¯ Má»¤C LÆ¯U TRá»® ---
# os.makedirs(output_dir, exist_ok=True)
# print(f"CÃ¡c file split sáº½ Ä‘Æ°á»£c lÆ°u táº¡i: {output_dir}")

# # --- 3. Ä�á»ŒC VÃ€ CHUáº¨N Bá»Š Dá»® LIá»†U ---
# df = pd.read_csv(csv_path)

# # Lá»�c ra danh sÃ¡ch cÃ¡c áº£nh thá»±c sá»± tá»“n táº¡i (Ä‘á»ƒ cháº¯c cháº¯n)
# existing_features_dir = '/kaggle/working/FOCUS/features/'
# if os.path.exists(existing_features_dir):
#     existing_ids = {int(f.split('.')[0]) for f in os.listdir(existing_features_dir)}
#     df = df[df['image_id'].isin(existing_ids)].reset_index(drop=True)
#     print(f"Ä�Ã£ lá»�c, chá»‰ sá»­ dá»¥ng {len(df)} áº£nh cÃ³ file Ä‘áº·c trÆ°ng tá»“n táº¡i.")

# # Láº¥y ra slide_id vÃ  nhÃ£n
# slide_ids = df['image_id']
# labels = df['label']

# # --- 4. THá»°C HIá»†N CHIA Dá»® LIá»†U ---
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# # Láº·p qua 8 fold
# for i, (train_val_indices, test_indices) in enumerate(skf.split(slide_ids, labels)):
    
#     # Láº¥y ra táº­p train+val vÃ  táº­p test cho fold hiá»‡n táº¡i
#     train_val_ids = slide_ids.iloc[train_val_indices]
#     train_val_labels = labels.iloc[train_val_indices]
#     test_ids = slide_ids.iloc[test_indices]
    
#     # Tiáº¿p tá»¥c chia táº­p train+val thÃ nh táº­p train vÃ  táº­p val
#     train_ids, val_ids = train_test_split(train_val_ids, 
#                                           test_size=val_size, 
#                                           stratify=train_val_labels, 
#                                           random_state=42)
    
#     # Táº¡o DataFrame theo Ä‘Ãºng Ä‘á»‹nh dáº¡ng yÃªu cáº§u
#     # DÃ¹ng pd.Series Ä‘á»ƒ xá»­ lÃ½ cÃ¡c list cÃ³ Ä‘á»™ dÃ i khÃ¡c nhau
#     split_df = pd.DataFrame({
#         'train': pd.Series(train_ids.tolist()),
#         'val': pd.Series(val_ids.tolist()),
#         'test': pd.Series(test_ids.tolist())
#     })
    
#     # LÆ°u file CSV
#     output_filename = os.path.join(output_dir, f'splits_{i}.csv')
#     split_df.to_csv(output_filename, index=False)
    
#     print(f"Ä�Ã£ táº¡o file: splits_{i}.csv (Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)})")

# print(f"\n--- HoÃ n táº¥t! Ä�Ã£ táº¡o thÃ nh cÃ´ng {n_splits} file split. ---")


import torch

feat = torch.load("/kaggle/working/FOCUS/features/10077.pt")
print(type(feat))
print(feat.shape if isinstance(feat, torch.Tensor) else "not tensor")




import subprocess
from joblib import Parallel, delayed
import glob
import pandas as pd
import torch


os.chdir('/kaggle/working/FOCUS/')


!pip install --quiet tensorboardX openai-clip faiss-cpu


CORRECT_CSV_PATH = "/kaggle/working/FOCUS/train.csv"

!sed -i -e "s|csv_path = '.*'|csv_path = '{CORRECT_CSV_PATH}'|g" main.py
print(f"Ä�Ã£ cáº­p nháº­t Ä‘Æ°á»�ng dáº«n trong main.py thÃ nh: {CORRECT_CSV_PATH}")

print("Cáº­p nháº­t Ä‘Æ°á»�ng dáº«n thÃ nh cÃ´ng.")

# ===================== MAP =====================
# def run_fold(fold_id, gpu_id, total_folds):
#     """
#     HÃ m nÃ y cháº¡y má»™t fold duy nháº¥t trÃªn má»™t GPU cá»¥ thá»ƒ.
#     """
#     log_dir = "/kaggle/working/FOCUS/results/FOCUS/conch"
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, f"fold_{fold_id}.log")
    
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#     # === Bá»” SUNG k, k_start, k_end VÃ€O Ä�Ã‚Y ===
#     cmd = [
#         "python", "main.py",
#         "--k", str(total_folds),         # Tá»•ng sá»‘ fold cá»§a toÃ n bá»™ thÃ­ nghiá»‡m
#         "--k_start", str(fold_id),       # Fold báº¯t Ä‘áº§u (chÃ­nh lÃ  fold hiá»‡n táº¡i)
#         "--k_end", str(fold_id + 1),     # Fold káº¿t thÃºc (chá»‰ cháº¡y 1 fold)
#         # CÃ¡c tham sá»‘ cÃ²n láº¡i giá»¯ nguyÃªn
#         "--seed", "1",
#         "--drop_out",
#         "--early_stopping",
#         "--lr", "1e-4",
#         "--label_frac", "1",
#         "--bag_loss", "ce",
#         "--task", "task_UBC-OCEAN_subtyping",
#         "--results_dir", "results/FOCUS/conch/",
#         "--exp_code", "UBC-OCEAN_4shots_10folds",
#         "--model_type", "FOCUS",   #ViLa_MIL
#         "--mode", "transformer",
#         "--max_epochs", "4",
#         "--log_data",
#         "--data_root_dir", "/kaggle/input/ubc-ovarian-cancer-subtype-classification/",
#         "--data_folder_s", "/kaggle/working/FOCUS/features/",
#         "--data_folder_l", "/kaggle/working/FOCUS/features/",
#         "--split_dir", "UBC-OCEAN_4shots_10folds",
#         "--text_prompt_path", "text_prompt/UBC-OCEAN_two_scale_text_prompt.csv",
#     ]

#     print(f"Báº¯t Ä‘áº§u cháº¡y Fold {fold_id} trÃªn GPU {gpu_id}...")
#     with open(log_file, "w") as f:
#         subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
#     print(f"HoÃ n thÃ nh Fold {fold_id}.")
#     return log_file

def run_fold(fold_id, gpu_id, total_folds):

    import sys
    
    log_dir = "/kaggle/working/FOCUS/results/FOCUS/conch"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"fold_{fold_id}.log")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python", "main.py",
        "--k", str(total_folds),
        "--k_start", str(fold_id),
        "--k_end", str(fold_id + 1),
        "--seed", "1",
        "--drop_out",
        "--early_stopping",
        "--lr", "1e-4",
        "--label_frac", "1",
        "--bag_loss", "ce",
        "--task", "task_UBC-OCEAN_subtyping",
        "--results_dir", "results/FOCUS/conch/",
        "--exp_code", "UBC-OCEAN_16shots_10folds",
        "--model_type", "FOCUS",
        "--mode", "transformer",
        "--max_epochs", "4",
        "--log_data",
        "--data_root_dir", "/kaggle/input/ubc-ovarian-cancer-subtype-classification/",
        "--data_folder_s", "/kaggle/working/FOCUS/features/",
        "--data_folder_l", "/kaggle/working/FOCUS/features/",
        "--split_dir", "UBC-OCEAN_16shots_10folds",
        "--text_prompt_path", "text_prompt/UBC-OCEAN_two_scale_text_prompt.csv",
    ]

    print(f"ğŸš€ Báº¯t Ä‘áº§u cháº¡y Fold {fold_id} trÃªn GPU {gpu_id}...")
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        # Ä�á»�c tá»«ng dÃ²ng output
        for line in process.stdout:
            sys.stdout.write(f"[Fold {fold_id} | GPU {gpu_id}] {line}")
            sys.stdout.flush()
            f.write(line)
        process.wait()

    print(f"âœ… HoÃ n thÃ nh Fold {fold_id}, log lÆ°u táº¡i {log_file}")
    return log_file


def reduce_results(results_dir):


    result_files = glob.glob(os.path.join(results_dir, "result_partial_*.csv"))
    print(f"\nğŸ”� Giai Ä‘oáº¡n Reduce: TÃ¬m tháº¥y {len(result_files)} file káº¿t quáº£ cá»§a cÃ¡c fold.")

    df_all = pd.concat([pd.read_csv(f) for f in result_files], ignore_index=True)
    summary = df_all.drop(columns=['folds']).agg(['mean', 'std']).T
    summary.reset_index(inplace=True)
    summary.rename(columns={'index': 'metric'}, inplace=True)
    
    output_file = os.path.join(results_dir, "summary.csv")
    summary.to_csv(output_file, index=False)
    print(f"âœ… Ä�Ã£ lÆ°u summary cuá»‘i cÃ¹ng vÃ o {output_file}")
    return summary

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    # Danh sÃ¡ch cÃ´ng viá»‡c: (fold_id, gpu_id)
    jobs = [(0, 0), (1, 1), (2, 0), (3, 1)] 
    total_folds_in_experiment = 2 # Tá»•ng sá»‘ fold báº¡n muá»‘n cháº¡y

    print(f"\n--- Báº¯t Ä‘áº§u giai Ä‘oáº¡n MAP: Cháº¡y {len(jobs)} fold song song trÃªn {num_gpus} GPU... ---")
    logs = Parallel(n_jobs=num_gpus)(delayed(run_fold)(fid, gid, total_folds_in_experiment) for fid, gid in jobs)
    print("âœ… HoÃ n thÃ nh giai Ä‘oáº¡n MAP.")
    
    results_dir = "/kaggle/working/FOCUS/results/FOCUS/conch/UBC-OCEAN_16shots_10folds"
    summary_df = reduce_results(results_dir)

    print("\n--- Káº¾T QUáº¢ CUá»�I CÃ™NG ---")
    if summary_df is not None:
        print(summary_df.to_string())

