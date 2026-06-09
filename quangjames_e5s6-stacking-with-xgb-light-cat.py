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


import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, Dataset
import lightgbm
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold, StratifiedKFold
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import cupy as cp
import xgboost as xgb
import lightgbm as lgb


train_file ='/kaggle/input/playground-series-s5e6/train.csv'
test_file ='/kaggle/input/playground-series-s5e6/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
print(train_data.info())
print(test_data.info())


id_column = 'id'
target_column = 'Fertilizer Name'
numeric_features = [col for col in train_data.select_dtypes(include=['int64', 'float64']).columns if col != target_column and col != id_column]
print(numeric_features)
categorical_features = [col for col in train_data.select_dtypes(include=['object']).columns if col != target_column and col != id_column]
print(categorical_features)


# Đọc dữ liệu
train_file = '/kaggle/input/playground-series-s5e6/train.csv'
test_file = '/kaggle/input/playground-series-s5e6/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Data preparation
train_df = train_data.copy().drop('id', axis=1)
test_ids = test_data['id'].copy()
test_df = test_data.copy().drop('id', axis=1)

# Định nghĩa các biến
id_column = 'id'
target_column = 'Fertilizer Name'
numeric_features = [col for col in train_df.select_dtypes(include=['int64', 'float64']).columns 
                    if col != target_column and col != id_column]
categorical_features = [col for col in train_df.select_dtypes(include=['object']).columns 
                        if col != target_column and col != id_column]

# Kiểm tra giá trị thiếu
print("Missing values in train_df:\n", train_df.isnull().sum())
print("Missing values in test_df:\n", test_df.isnull().sum())

label_encoders = {}

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Encode Target column
le_target = LabelEncoder()
train_df[target_column] = le_target.fit_transform(train_df[target_column])
label_encoders[target_column] = le_target

# Numeric features standalization
scaler = StandardScaler()
train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
test_df[numeric_features] = scaler.transform(test_df[numeric_features])

# Tách đặc trưng và mục tiêu
X = train_df.drop(target_column, axis=1)
y = train_df[target_column]

# Chia tập train thành train và validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check size
print(f"Columns of train: {X.columns.tolist()}")
print(f"Size of train: {X.shape}")
print(f"Size of test: {test_df.shape}")
print(f"Size of train for training: {X_train.shape}")
print(f"Size of validation for training: {X_val.shape}")


# MAP@3 function
def map_at_3(y_true, y_pred_prob, k=3):
    map_scores = []
    for true_label, pred_prob in zip(y_true, y_pred_prob):
        top_k_indices = np.argsort(pred_prob)[::-1][:k]
        true_label_binary = np.zeros(len(pred_prob))
        true_label_binary[true_label] = 1
        relevant = [1 if idx == true_label else 0 for idx in top_k_indices]
        precisions = []
        num_relevant = 0
        for i, rel in enumerate(relevant):
            if rel == 1:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        map_scores.append(np.mean(precisions) if precisions else 0)
    return np.mean(map_scores)

# Define StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Number of classes
n_classes = len(np.unique(y))

# Initialize OOF and test predictions
oof_preds_xgb = np.zeros((X.shape[0], n_classes))
oof_preds_lgb = np.zeros((X.shape[0], n_classes))
oof_preds_cat = np.zeros((X.shape[0], n_classes))
test_preds_xgb = np.zeros((test_df.shape[0], n_classes))
test_preds_lgb = np.zeros((test_df.shape[0], n_classes))
test_preds_cat = np.zeros((test_df.shape[0], n_classes))

# Categorical features
print("Columns in X:", X.columns.tolist())
print("Categorical features:", categorical_features)
cat_features_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
print("Validated categorical indices for LightGBM:", cat_features_indices)

# Training loop for base models
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nFold {fold}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # XGBoost with GPU
    X_train_gpu = cp.array(X_train.values)
    X_val_gpu = cp.array(X_val.values)
    y_train_gpu = cp.array(y_train)
    y_val_gpu = cp.array(y_val)
    
    train_data_xgb = xgb.DMatrix(X_train_gpu, label=y_train_gpu)
    val_data_xgb = xgb.DMatrix(X_val_gpu, label=y_val_gpu)
    
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'max_depth': 6,
        'learning_rate': 0.025,
        'subsample': 0.95,
        'colsample_bytree': 0.85,
        'lambda': 2.0,
        'alpha': 0.75,
        'min_child_weight': 20,
        'max_bin': 128,
        'gamma': 0.2,
        'grow_policy': 'lossguide',
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda'
    }
    
    model_xgb = xgb.train(
        xgb_params,
        train_data_xgb,
        num_boost_round=10000,
        evals=[(train_data_xgb, 'train'), (val_data_xgb, 'val')],
        early_stopping_rounds=200,
        verbose_eval=500
    )
    
    oof_preds_xgb[val_idx] = model_xgb.predict(val_data_xgb)
    test_preds_xgb += model_xgb.predict(xgb.DMatrix(cp.array(test_df.values))) / 5
    
    # LightGBM with GPU
    train_data_lgb = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features_indices)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features_indices)
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'boosting_type': 'gbdt',
        'num_leaves': 40,
        'learning_rate': 0.03,
        'max_depth': 5,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'device': 'gpu'
    }
    
    model_lgb = lgb.train(
        lgb_params,
        train_data_lgb,
        num_boost_round=10000,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=True)]
    )
    
    oof_preds_lgb[val_idx] = model_lgb.predict(X_val)
    test_preds_lgb += model_lgb.predict(test_df) / 5
    
    # CatBoost with GPU
    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)
    
    model_cat = CatBoostClassifier(
        iterations=20000,
        depth=6,
        learning_rate=0.01,
        l2_leaf_reg=8,
        random_seed=42,
        cat_features=cat_features_indices,
        task_type='GPU',
        early_stopping_rounds=400,
        verbose=500
    )
    
    model_cat.fit(train_pool, eval_set=val_pool)
    oof_preds_cat[val_idx] = model_cat.predict_proba(X_val)
    test_preds_cat += model_cat.predict_proba(test_df) / 5

# Calculate MAP@3 for each base model
train_map3_xgb = map_at_3(y, oof_preds_xgb)
train_map3_lgb = map_at_3(y, oof_preds_lgb)
train_map3_cat = map_at_3(y, oof_preds_cat)
print(f"XGBoost Train MAP@3: {train_map3_xgb:.4f}")
print(f"LightGBM Train MAP@3: {train_map3_lgb:.4f}")
print(f"CatBoost Train MAP@3: {train_map3_cat:.4f}")

# Save predictions for stacking
np.save('oof_preds_xgb.npy', oof_preds_xgb)
np.save('oof_preds_lgb.npy', oof_preds_lgb)
np.save('oof_preds_cat.npy', oof_preds_cat)
np.save('test_preds_xgb.npy', test_preds_xgb)
np.save('test_preds_lgb.npy', test_preds_lgb)
np.save('test_preds_cat.npy', test_preds_cat)
print("Predictions saved for stacking!")


# MAP@3 function
def map_at_3(y_true, y_pred_prob, k=3):
    map_scores = []
    for true_label, pred_prob in zip(y_true, y_pred_prob):
        top_k_indices = np.argsort(pred_prob)[::-1][:k]
        true_label_binary = np.zeros(len(pred_prob))
        true_label_binary[true_label] = 1
        relevant = [1 if idx == true_label else 0 for idx in top_k_indices]
        precisions = []
        num_relevant = 0
        for i, rel in enumerate(relevant):
            if rel == 1:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        map_scores.append(np.mean(precisions) if precisions else 0)
    return np.mean(map_scores)

# Load saved predictions
oof_preds_xgb = np.load('oof_preds_xgb.npy')
oof_preds_lgb = np.load('oof_preds_lgb.npy')
oof_preds_cat = np.load('oof_preds_cat.npy')
test_preds_xgb = np.load('test_preds_xgb.npy')
test_preds_lgb = np.load('test_preds_lgb.npy')
test_preds_cat = np.load('test_preds_cat.npy')
n_classes = oof_preds_xgb.shape[1]  # Lấy số lớp từ kích thước OOF predictions

# Prepare meta-features
meta_features_train = np.column_stack((oof_preds_xgb, oof_preds_lgb, oof_preds_cat))
meta_features_test = np.column_stack((test_preds_xgb, test_preds_lgb, test_preds_cat))

# Define KFold for validation (3 folds)
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Trial 1: Logistic Regression (CPU)
print('Re-training with meta-learner by Logistic Regression')
meta_learner_lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
meta_learner_lr.fit(meta_features_train, y)
oof_preds_stacked_lr = meta_learner_lr.predict_proba(meta_features_train)
train_map3_lr = map_at_3(y, oof_preds_stacked_lr)
print(f"Logistic Regression Stacked Train MAP@3: {train_map3_lr:.4f}")

# Trial 2: LightGBM for meta (GPU with 3-fold validation)
print('Re-training with meta-learner by LightGBM')
meta_learner_lgb = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=n_classes,
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    n_estimators=1200,
    verbose=-1,
    device='gpu'
)

# Prepare for early stopping with 3-fold KFold
oof_preds_lgb_meta = np.zeros((meta_features_train.shape[0], n_classes))
for fold, (train_idx, val_idx) in enumerate(kf.split(meta_features_train, y), 1):
    print(f"  Meta Fold {fold}")
    meta_train_x, meta_val_x = meta_features_train[train_idx], meta_features_train[val_idx]
    y_train_meta, y_val_meta = y[train_idx], y[val_idx]
    eval_set = [(meta_val_x, y_val_meta)]
    meta_learner_lgb.fit(
        meta_train_x,
        y_train_meta,
        eval_set=eval_set,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    oof_preds_lgb_meta[val_idx] = meta_learner_lgb.predict_proba(meta_val_x)

train_map3_lgb = map_at_3(y, oof_preds_lgb_meta)
print(f"LightGBM Stacked Train MAP@3: {train_map3_lgb:.4f}")

# Trial 3: CatBoost for meta (GPU)
print('Re-training with meta-learner by CatBoost')
meta_learner_cat = CatBoostClassifier(
    iterations=1000,
    depth=3,
    learning_rate=0.05,
    random_seed=42,
    task_type='GPU',
    verbose=0
)
meta_learner_cat.fit(meta_features_train, y)
oof_preds_stacked_cat = meta_learner_cat.predict_proba(meta_features_train)
train_map3_cat = map_at_3(y, oof_preds_stacked_cat)
print(f"CatBoost Stacked Train MAP@3: {train_map3_cat:.4f}")

# Choose the best meta-learner and predict test
best_meta_learner = max([(train_map3_lr, meta_learner_lr), (train_map3_lgb, meta_learner_lgb), (train_map3_cat, meta_learner_cat)], key=lambda x: x[0])[1]
test_preds_stacked = best_meta_learner.predict_proba(meta_features_test)
print("Test predictions generated with the best meta-learner!")


# MAP@3 function (nếu cần kiểm tra)
def map_at_3(y_true, y_pred_prob, k=3):
    map_scores = []
    for true_label, pred_prob in zip(y_true, y_pred_prob):
        top_k_indices = np.argsort(pred_prob)[::-1][:k]
        true_label_binary = np.zeros(len(pred_prob))
        true_label_binary[true_label] = 1
        relevant = [1 if idx == true_label else 0 for idx in top_k_indices]
        precisions = []
        num_relevant = 0
        for i, rel in enumerate(relevant):
            if rel == 1:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        map_scores.append(np.mean(precisions) if precisions else 0)
    return np.mean(map_scores)

# Predicting probability on test set using best meta-learner
test_pred_prob = test_preds_stacked  # Sử dụng kết quả từ meta-learner

# Get the top 3 most probable labels for each sample
top_3_indices = np.argsort(test_pred_prob, axis=1)[:, -3:][:, ::-1]

# Decode top 3 labels from number to original name (Fertilizer Name)
top_3_labels = []
for i in range(len(top_3_indices)):
    labels = label_encoders[target_column].inverse_transform(top_3_indices[i])
    # Kết hợp các nhãn thành một chuỗi, phân tách bằng dấu cách
    top_3_labels.append(" ".join(labels))

# Tạo DataFrame submission
submission_df = pd.DataFrame({
    'id': test_ids,  # Sử dụng test_ids đã lưu trước đó
    'Fertilizer Name': top_3_labels
})

# Lưu file submission
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' has been created!")


# Check submission
print(submission_df.head())
print(submission_df.shape)

