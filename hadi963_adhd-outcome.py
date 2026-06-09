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


# ğŸ“¦ Install Required Packages
!pip install pytorch-tabnet scikit-learn pandas numpy torch tqdm openpyxl -q 


# ğŸ“¦ Install Required Packages
# pip install pytorch-tabnet scikit-learn pandas numpy torch tqdm openpyxl matplotlib

import pandas as pd
import numpy as np
import torch
import time
import gc
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# âš™ Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ğŸ“‚ Load Data
train_solution = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
train_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
train_quant = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
train_conn = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')

test_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
test_quant = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')
test_conn = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')

train = train_cat.merge(train_quant, on='participant_id').merge(train_solution, on='participant_id').merge(train_conn, on='participant_id')
test = test_cat.merge(test_quant, on='participant_id').merge(test_conn, on='participant_id')

X = train.drop(columns=['participant_id', 'ADHD_Outcome', 'Sex_F'])
y = train[['ADHD_Outcome', 'Sex_F']].astype(int)
X_test = test.drop(columns=['participant_id'])

# Drop known weak features manually
features_to_drop_manual = ['Barratt_Barratt_P1_Occ', 'PreInt_Demos_Fam_Child_Ethnicity',
    'Barratt_Barratt_P1_Edu', 'Basic_Demos_Enroll_Year', 'Barratt_Barratt_P2_Occ',
    'APQ_P_APQ_P_ID', 'ColorVision_CV_Score']
X = X.drop(columns=features_to_drop_manual, errors='ignore')
X_test = X_test.drop(columns=features_to_drop_manual, errors='ignore')

X = X.fillna(0)
X_test = X_test.fillna(0)

# ğŸ”§ Feature Engineering
mri_features = [col for col in X.columns if col.startswith('MRI_Track_')]
X['MRI_Track_Mean'] = X[mri_features].mean(axis=1)
X['MRI_Track_Std'] = X[mri_features].std(axis=1)
X_test['MRI_Track_Mean'] = X_test[mri_features].mean(axis=1)
X_test['MRI_Track_Std'] = X_test[mri_features].std(axis=1)

# PCA on all features
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X)
X_test_all_scaled = scaler_all.transform(X_test)

pca_all = PCA(n_components=20)
X_pca_all = pca_all.fit_transform(X_all_scaled)
X_test_pca_all = pca_all.transform(X_test_all_scaled)

for i in range(20):
    X[f'PCA_All_{i+1}'] = X_pca_all[:, i]
    X_test[f'PCA_All_{i+1}'] = X_test_pca_all[:, i]

# Final scaling
scaler = StandardScaler()
X_scaled = np.nan_to_num(scaler.fit_transform(X))
X_test_scaled = np.nan_to_num(scaler.transform(X_test))
y_np = y.values

# Class distribution logging
adhd_counts = pd.Series(y_np[:, 0]).value_counts().to_dict()
sex_counts = pd.Series(y_np[:, 1]).value_counts().to_dict()
print("\nğŸ“Š Class Distribution:")
print(f"ADHD_Outcome: {adhd_counts}")
print(f"Sex_F: {sex_counts}")

# Oversample ADHD class manually
X_df = pd.DataFrame(X_scaled)
y_series = pd.Series(y_np[:, 0])
X_min = X_df[y_series == 1]
X_maj = X_df[y_series == 0]

multiplier = max(1, int(len(X_maj) / max(len(X_min), 1)))
X_min_upsampled = pd.concat([X_min] * multiplier, ignore_index=True)
y_min_upsampled = pd.Series([1] * len(X_min_upsampled))
X_maj = X_maj.reset_index(drop=True)
y_maj = pd.Series([0] * len(X_maj))
X_balanced = pd.concat([X_maj, X_min_upsampled], ignore_index=True)
y_balanced = pd.concat([y_maj, y_min_upsampled], ignore_index=True)
X_balanced = X_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
y_balanced = y_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Inject noise
print("\nâš ï¸� Injecting Gaussian noise, feature dropout, and label flipping...")
X_balanced += np.random.normal(0, 0.3, X_balanced.shape)  # Gaussian noise
num_features = X_balanced.shape[1]
drop_indices = np.random.choice(num_features, int(num_features * 0.2), replace=False)
X_balanced.iloc[:, drop_indices] = 0  # Feature dropout
flip_indices = np.random.choice(len(y_balanced), int(len(y_balanced) * 0.1), replace=False)
y_balanced.iloc[flip_indices] = 1 - y_balanced.iloc[flip_indices]  # Label noise
print("âœ… Noise injection complete.")

# â�• Training TabNet Models and Making Predictions
adhd_preds, sex_preds = [], []
adhd_train_preds = np.zeros(len(X_scaled))
sex_train_preds = np.zeros(len(X_scaled))

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X_scaled, y_np[:, 1]), total=n_splits)):
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y_np[train_idx], y_np[val_idx]

    torch.cuda.empty_cache()
    model_adhd = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.0015),
        scheduler_params={"step_size":10, "gamma":0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax', n_d=8, n_a=8, n_steps=1,
        seed=fold + 42, verbose=0
    )
    model_adhd.fit(
        X_balanced.values, y_balanced.values,
        eval_set=[(X_val_fold, y_val_fold[:,0])],
        eval_metric=['auc'], patience=7, max_epochs=100,
        batch_size=64, virtual_batch_size=32
    )
    adhd_preds.append(model_adhd.predict_proba(X_test_scaled)[:, 1])
    adhd_train_preds[val_idx] = model_adhd.predict_proba(X_val_fold)[:, 1]

    X_train_sex, y_train_sex = X_train_fold[y_train_fold[:,1]==0], X_train_fold[y_train_fold[:,1]==1]
    X_train_sex = np.vstack([X_train_sex, X_train_sex])
    y_train_sex = np.concatenate([np.zeros(len(X_train_sex)//2), np.ones(len(X_train_sex)//2)])

    torch.cuda.empty_cache()
    model_sex = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.0015),
        scheduler_params={"step_size":10, "gamma":0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax', n_d=8, n_a=8, n_steps=1,
        seed=fold + 200, verbose=0
    )
    model_sex.fit(
        X_train_sex, y_train_sex,
        eval_set=[(X_val_fold, y_val_fold[:,1])],
        eval_metric=['auc'], patience=7, max_epochs=100,
        batch_size=64, virtual_batch_size=32
    )
    sex_preds.append(model_sex.predict_proba(X_test_scaled)[:, 1])
    sex_train_preds[val_idx] = model_sex.predict_proba(X_val_fold)[:, 1]

# ğŸ”� Find best threshold for ADHD based on AUC
thresholds = np.linspace(0.1, 0.9, 100)
true = y_np[:, 0]
probs = adhd_train_preds
auc_scores = [roc_auc_score(true, (probs > t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(auc_scores)]

print(f"\nğŸ”� Best ADHD AUC Threshold: {best_threshold:.2f}")
print(f"ğŸ”� Max ADHD AUC: {max(auc_scores):.4f}")

plt.plot(thresholds, auc_scores)
plt.xlabel('Threshold')
plt.ylabel('AUC Score')
plt.title('Threshold Optimization for ADHD')
plt.grid(True)
plt.show()

# ğŸ“¤ Final Submission
final_ADHD = (np.mean(adhd_preds, axis=0) > best_threshold).astype(int)
final_Sex_F = (np.mean(sex_preds, axis=0) > 0.5).astype(int)

submission = test[['participant_id']].copy()
submission['ADHD_Outcome'] = final_ADHD
submission['Sex_F'] = final_Sex_F

print("\nğŸ“Š Final Evaluation:")
print(f"ADHD F1 Score: {f1_score(y_np[:,0], adhd_train_preds > best_threshold):.4f}")
print(f"Sex_F F1 Score: {f1_score(y_np[:,1], (sex_train_preds > 0.5).astype(int)):.4f}")

submission.to_csv("submission.csv", index=False)
print("\nâœ… submission.csv saved successfully!")



# ğŸ“Š Final Evaluation
from sklearn.metrics import f1_score

print("\nğŸ“Š Final Evaluation on Training Data:")
f1_adhd = f1_score(y_np[:, 0], adhd_train_preds.astype(int))
f1_sex = f1_score(y_np[:, 1], sex_train_preds.astype(int))

print(f"âœ… Final ADHD F1 Score: {f1_adhd:.4f}")
print(f"âœ… Final Sex_F F1 Score: {f1_sex:.4f}")








