import sys
sys.path.append('/kaggle/input/imc25-utils')
import metric
import pandas as pd


final_score, dataset_scores = metric.score(
    gt_csv='/kaggle/input/imc2025-ets-cv-sample/train_labels.csv',
    user_csv="/kaggle/input/imc2025-ets-cv-sample/submission.csv",
    thresholds_csv='/kaggle/input/imc2025-ets-cv-sample/train_thresholds.csv',
    mask_csv=None,
    inl_cf=0,
    strict_cf=-1,
    verbose=True,
)


df_sub = pd.read_csv("/kaggle/input/imc2025-ets-cv-sample/submission.csv")
df_sub


df_train = pd.read_csv("/kaggle/input/imc2025-ets-cv-sample/train_labels.csv")
df_train




