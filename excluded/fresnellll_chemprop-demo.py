# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# /kaggle/input/chemprop-online-v2


# 侦察兵Cell：列出所有输入目录的内容
!ls -R /kaggle/input/


# ===================================================================
# Cell 1: 环境搭建 - 终极版 v3.0 (基于最终侦察报告)
# ===================================================================
print("--- Step 1: Setting up the environment using pre-built wheels (Final Version 3.0) ---")

# --- 1. 彻底卸载所有可能冲突的包 ---
print("\n--> Uninstalling ALL conflicting packages...")
!pip uninstall -y -q numpy scikit-learn joblib torch torchaudio torchvision

# --- 2. 定义我们的"离线包仓库"的正确路径 ---
# 根据侦察报告，所有.whl文件都在这个根目录下
WHEEL_DIR = "/kaggle/input/chemprop-online-v2/" 

# --- 3. 【核心修正】使用一条命令，一次性、精确地安装所有包 ---
# 我们明确指定了chemprop的whl文件路径，确保安装的是polymer-chemprop
print("\n--> Installing all required packages from our offline wheelhouse...")

!pip install --no-index --find-links={WHEEL_DIR} \
    "numpy==1.26.4" \
    "scikit-learn==1.4.0" \
    "rdkit" \
    "{WHEEL_DIR}chemprop-1.4.0-py3-none-any.whl" \
    "pandas" \
    "tqdm"

# --- 4. 最终版本验证 ---
print("\n--- Environment setup complete. Final package versions: ---")
!pip show numpy pandas scikit-learn rdkit torch chemprop | grep -E "^Name:|^Version:"


# ===================================================================
# Cell 2: 导入库 & 代码热修复
# ===================================================================
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
import os
import sys
from tqdm.notebook import tqdm
import site

print("--- Step 2: Importing libraries and applying hotfix ---")
chemprop_lib_path = Path(site.getsitepackages()[0]) / 'chemprop'
utils_file_path = chemprop_lib_path / 'utils.py'
print(f"Located chemprop utils file at: {utils_file_path}")
try:
    with open(utils_file_path, 'r') as f: content = f.read()
    line_to_find = "torch.load(path, map_location=lambda storage, loc: storage)"
    replacement_line = "torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)"
    if line_to_find in content:
        print("Applying hotfix...")
        content = content.replace(line_to_find, replacement_line)
        with open(utils_file_path, 'w') as f: f.write(content)
        print("Hotfix applied successfully!")
    else:
        print("Skipping hotfix (already patched).")
except Exception as e:
    print(f"An error occurred during hotfix: {e}")
    pass
tqdm.pandas()
RDLogger.DisableLog('rdApp.*')
COMPETITION_DATA_DIR = Path("/kaggle/input/neurips-open-polymer-prediction-2025/")
CHEMPROP_MODEL_DIR = Path("/kaggle/input/exp4-deeper-d7-h600/exp4_deeper_d7_h600") 


# ===================================================================
# Cell 3: Preprocessing the test set (using "Star Keep" Strategy)
# ===================================================================
print("--- Step 3: Preprocessing the test set with the 'Star Keep' strategy ---")

def process_smiles_for_test_star_keep(smi: str):
    if not isinstance(smi, str) or smi == '': return None
    if '.' in smi: return None
    try:
        mol = Chem.MolFromSmiles(smi)
        return smi if mol else None
    except:
        return None

df_test_orig = pd.read_csv(COMPETITION_DATA_DIR / "test.csv")
print("Applying SMILES validation (Star Keep strategy) to the test data...")
df_test_orig['smiles_processed'] = df_test_orig['SMILES'].progress_apply(process_smiles_for_test_star_keep)

df_for_predict = df_test_orig[['id', 'smiles_processed']].copy()
df_for_predict.rename(columns={'smiles_processed': 'smiles'}, inplace=True)
test_preds_input_path = "test_for_predict.csv"
df_for_predict.to_csv(test_preds_input_path, index=False)
print(f"\nTest data preprocessed and saved to '{test_preds_input_path}'")


# # ===================================================================
# # Cell 3: Preprocessing the test set (Final "Star Keep" Strategy)
# # ===================================================================
# print("--- Step 3: Preprocessing the test set with the 'Star Keep' strategy ---")

# def process_smiles_for_test_star_keep(smi: str):
#     """
#     最稳健的策略：只验证，不替换。
#     我们只过滤掉包含'.'的混合物和真正有语法错误的SMILES。
#     """
#     if not isinstance(smi, str) or smi == '': return None
#     if '.' in smi: return None

#     # 我们直接在原始SMILES上进行验证
#     try:
#         mol = Chem.MolFromSmiles(smi)
#         # 只要RDKit能解析，我们就返回原始SMILES
#         return smi if mol else None
#     except:
#         return None

# # 加载官方测试集
# df_test_orig = pd.read_csv(COMPETITION_DATA_DIR / "test.csv")

# # 应用我们新的、更稳健的处理函数
# print("Applying SMILES validation (Star Keep strategy) to the test data...")
# df_test_orig['smiles_processed'] = df_test_orig['SMILES'].progress_apply(process_smiles_for_test_star_keep)

# # 【关键的诊断步骤】
# # 让我们看看这次有多少SMILES通过了验证
# valid_count = df_test_orig['smiles_processed'].notna().sum()
# total_count = len(df_test_orig)
# print(f"\nValidation complete. {valid_count} / {total_count} SMILES are valid.")

# # 如果仍然有大量无效的，我们需要在这里停下来分析
# if valid_count == 0:
#     print("\nFATAL ERROR: All test SMILES are still invalid even with the 'Star Keep' strategy.")
#     # 打印一些失败的例子，帮助我们分析
#     print("Examples of failed original SMILES:")
#     print(df_test_orig[df_test_orig['smiles_processed'].isna()]['SMILES'].head())


# # 创建用于预测的DataFrame
# df_for_predict = df_test_orig[['id', 'smiles_processed']].copy()
# df_for_predict.rename(columns={'smiles_processed': 'smiles'}, inplace=True)

# # 保存到临时文件
# test_preds_input_path = "test_for_predict.csv"
# # 我们只把有效的SMILES传给模型，但保留所有id
# df_for_predict.to_csv(test_preds_input_path, index=False)

# print(f"\nTest data preprocessed and saved to '{test_preds_input_path}'")
# print("\nPreview of the data to be fed into the model:")
# print(df_for_predict.head())


# ===================================================================
# Cell 4: 运行预测 (终极版 - 增加了对预测文件是否生成的检查)
# ===================================================================
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs
import pandas as pd
import os

print("\n--- Step 4: Running prediction using 5-fold cross-validation ensemble ---")

# 检查预处理后的文件是否存在且非空
df_to_check = pd.read_csv("test_for_predict.csv")
if df_to_check.dropna(subset=['smiles']).empty:
    print("\nWARNING: No valid SMILES found in the test set.")
    all_predictions = []
else:
    test_input_path = "test_for_predict.csv"
    preds_output_path = "raw_predictions.csv" # 临时文件名
    all_predictions = []
    NUM_FOLDS = 5

    print(f"Starting prediction for {NUM_FOLDS} folds...")
    for i in range(NUM_FOLDS):
        print(f"--> Predicting with model from fold {i}...")
        checkpoint_path = os.path.join(str(CHEMPROP_MODEL_DIR), f'fold_{i}/model_0/model.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"    WARNING: Model for fold {i} not found. Skipping.")
            continue
            
        args = PredictArgs().parse_args([
            '--test_path', test_input_path,
            '--checkpoint_paths', checkpoint_path,
            '--preds_path', preds_output_path,
            '--gpu', '0',
            '--batch_size', '128',
            '--num_workers', '4'
        ])
        
        # 在调用预测前，如果临时文件已存在，先删除它
        if os.path.exists(preds_output_path):
            os.remove(preds_output_path)
            
        make_predictions(args=args)
        
        # ===================================================================
        # --- 【核心修正】在这里加入防御性检查 ---
        # ===================================================================
        if os.path.exists(preds_output_path):
            print(f"    Prediction for fold {i} successful. Reading results.")
            df_fold_preds = pd.read_csv(preds_output_path)
            all_predictions.append(df_fold_preds)
        else:
            # 这段逻辑会在我们用小的公开测试集调试时被触发
            print(f"    WARNING: Prediction for fold {i} did not generate an output file (likely due to small test set size). Skipping this fold.")
        # ===================================================================

    print("\nAll folds predicted (or skipped).")

# --- 集成预测结果 ---
if all_predictions:
    print("Ensembling predictions by averaging...")
    df_concat = pd.concat(all_predictions)
    df_ensembled_preds = df_concat.groupby(df_concat.index).mean()
    # 将最终的集成结果覆盖写入 raw_predictions.csv
    df_ensembled_preds.to_csv("raw_predictions.csv", index=False)
    print("Ensembled predictions saved.")
else:
    # 这段逻辑会在我们用小的公开测试集调试时被触发
    print("No predictions were made in any fold. Skipping ensembling.")


# ===================================================================
# Cell 5: Generating the final submission file
# ===================================================================
print("\n--- Step 5: Generating submission.csv ---")

df_test_ids = pd.read_csv(COMPETITION_DATA_DIR / "test.csv")[['id']]
submission_df = df_test_ids.copy()
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

if os.path.exists("raw_predictions.csv"):
    df_preds = pd.read_csv("raw_predictions.csv")
    for target in targets:
        submission_df[target] = df_preds[target]
else:
    print("WARNING: raw_predictions.csv not found. Generating a dummy submission file with zeros.")
    for target in targets:
        submission_df[target] = 0

submission_df.fillna(0, inplace=True)
submission_df.to_csv("submission.csv", index=False)

print("\n--- All steps complete! ---")
print("submission.csv has been generated successfully.")
print("\nFinal submission file preview:")
print(submission_df.head())

