# ===========================
# RSNA 2025 Brain Aneurysm
# 零模型 Baseline – 无警告、无报错
# ===========================

import os
import shutil
from collections import defaultdict
import polars as pl
import pydicom
import kaggle_evaluation.rsna_inference_server

# 清空本地测试共享目录（关键修复）
shutil.rmtree('/kaggle/shared', ignore_errors=True)
os.makedirs('/kaggle/shared', exist_ok=True)

# 比赛要求的 14 个标签（顺序固定）
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

def predict(series_path: str) -> pl.DataFrame:
    """Baseline：所有标签输出 0.5"""
    fake_preds = [0.5] * len(LABEL_COLS)
    return pl.DataFrame([fake_preds], schema=LABEL_COLS, orient="row")

# 启动官方推理服务器
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()

# 本地查看提交文件（可选）
submission = pl.read_parquet('/kaggle/working/submission.parquet')
print("Submission preview:")
print(submission)

