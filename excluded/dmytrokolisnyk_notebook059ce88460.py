# ====================================================
# ğŸ“¦ Ğ†Ğ¼Ğ¿Ğ¾Ñ€Ñ‚Ğ¸
# ====================================================
import gc
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ====================================================
# ğŸ“‚ Ğ—Ğ°Ğ²Ğ°Ğ½Ñ‚Ğ°Ğ¶ĞµĞ½Ğ½Ñ� Ğ´Ğ°Ğ½Ğ¸Ñ…
# ====================================================
train = pd.read_parquet('/kaggle/input/pre-aggregated-amex-dataset-v1/train_final.parquet')

# ====================================================
# ğŸ§¹ ĞŸÑ–Ğ´Ğ³Ğ¾Ñ‚Ğ¾Ğ²ĞºĞ°
# ====================================================
X = train.drop(['customer_ID', 'target'], axis=1)
y = train['target']

X = X.select_dtypes(include=[np.number])

del train
gc.collect()

# Ğ Ğ¾Ğ·Ğ´Ñ–Ğ»ĞµĞ½Ğ½Ñ� train/valid
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

del X, y
gc.collect()


# ====================================================
# ğŸ§® AMEX Metric (official)
# ====================================================
def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)


# ====================================================
# âš™ï¸� GPU XGBoost Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¸
# ====================================================
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",   # <-- Ğ±Ñ–Ğ»ÑŒÑˆĞµ Ğ½Ğµ gpu_hist
    "device": "cuda",        # <-- Ğ¾Ñ�ÑŒ Ñ†ĞµĞ¹ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€ Ğ°ĞºÑ‚Ğ¸Ğ²ÑƒÑ” GPU
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "nthread": -1,
}


# ====================================================
# ğŸ§  DMatrix + Ñ‚Ñ€ĞµĞ½ÑƒĞ²Ğ°Ğ½Ğ½Ñ�
# ====================================================
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=100,
    verbose_eval=100
)


# ====================================================
# ğŸ’¾ Ğ—Ğ±ĞµÑ€ĞµĞ¶ĞµĞ½Ğ½Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ñ–
# ====================================================
model.save_model("xgb_amex_gpu.json")
print("ğŸ’¾ ĞœĞ¾Ğ´ĞµĞ»ÑŒ Ğ·Ğ±ĞµÑ€ĞµĞ¶ĞµĞ½Ğ¾ Ñƒ Ñ„Ğ°Ğ¹Ğ» xgb_amex_gpu.json")


import os
import gc
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

# ====================================================
# ğŸ’¾ Ğ—Ğ°Ğ²Ğ°Ğ½Ñ‚Ğ°Ğ¶ĞµĞ½Ğ½Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ñ–
# ====================================================
model = xgb.Booster()
model.load_model("xgb_amex_gpu.json")
print("âœ… ĞœĞ¾Ğ´ĞµĞ»ÑŒ ÑƒÑ�Ğ¿Ñ–ÑˆĞ½Ğ¾ Ğ·Ğ°Ğ²Ğ°Ğ½Ñ‚Ğ°Ğ¶ĞµĞ½Ğ°")

# ====================================================
# âš™ï¸� Ğ¨Ğ»Ñ�Ñ… Ğ´Ğ¾ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¸Ñ… parquet-Ñ„Ğ°Ğ¹Ğ»Ñ–Ğ²
# ====================================================
TEST_DIR = '/kaggle/input/pre-aggregated-amex-dataset-v1/test_agg.parquet'
OUTPUT_FILE = 'submission.csv'

# Ğ�Ñ‚Ñ€Ğ¸Ğ¼ÑƒÑ”Ğ¼Ğ¾ Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº ÑƒÑ�Ñ–Ñ… Ñ„Ğ°Ğ¹Ğ»Ñ–Ğ² Ñƒ Ñ†Ñ–Ğ¹ Ğ¿Ğ°Ğ¿Ñ†Ñ–
test_files = sorted([os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith('.parquet')])
print(f"ğŸ“¦ Ğ—Ğ½Ğ°Ğ¹Ğ´ĞµĞ½Ğ¾ {len(test_files)} parquet-Ñ„Ğ°Ğ¹Ğ»Ñ–Ğ²")

# Ğ¯ĞºÑ‰Ğ¾ Ñ„Ğ°Ğ¹Ğ» ÑƒĞ¶Ğµ Ñ–Ñ�Ğ½ÑƒÑ” â€” Ğ²Ğ¸Ğ´Ğ°Ğ»Ñ�Ñ”Ğ¼Ğ¾
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# ====================================================
# ğŸš€ Ğ�Ğ±Ñ€Ğ¾Ğ±ĞºĞ° Ğ¿Ğ¾ Ñ‡Ğ°Ñ�Ñ‚Ğ¸Ğ½Ğ°Ñ…
# ====================================================
for i, file_path in enumerate(test_files):
    print(f"ğŸ§© Ğ�Ğ±Ñ€Ğ¾Ğ±Ğ»Ñ�Ñ”Ğ¼Ğ¾ Ñ„Ğ°Ğ¹Ğ» {i+1}/{len(test_files)}: {os.path.basename(file_path)}")

    # Ğ—Ñ‡Ğ¸Ñ‚ÑƒÑ”Ğ¼Ğ¾ parquet
    df = pq.read_table(file_path).to_pandas()
    
    # customer_ID_
    customer_ids = df['customer_ID_'].values

    # Ğ¢Ñ–Ğ»ÑŒĞºĞ¸ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ– Ñ„Ñ–Ñ‡Ñ–
    X_chunk = df.select_dtypes(include=[np.number])

    # GPU-Ğ¿Ñ€ĞµĞ´Ğ¸ĞºÑ‚
    dchunk = xgb.DMatrix(X_chunk)
    preds = model.predict(dchunk)

    # Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚
    sub_chunk = pd.DataFrame({
        'customer_ID': customer_ids,
        'prediction': preds
    })

    # Ğ—Ğ°Ğ¿Ğ¸Ñ� Ñƒ CSV (append)
    sub_chunk.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)

    del df, X_chunk, dchunk, sub_chunk
    gc.collect()

print(f"\nâœ… Ğ“Ğ¾Ñ‚Ğ¾Ğ²Ğ¾! Submission Ğ·Ğ±ĞµÑ€ĞµĞ¶ĞµĞ½Ğ¾ Ñƒ '{OUTPUT_FILE}'")


