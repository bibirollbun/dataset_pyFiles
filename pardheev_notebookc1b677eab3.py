# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import re
# import os
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Set visualization style
# sns.set(style="whitegrid")
# plt.rcParams['figure.figsize'] = (10, 6)

# # ----------------------------
# # Utility: Clean numeric values
# # ----------------------------
# def clean_numeric(value):
#     """
#     Remove all non-numeric characters from the string representation of 'value',
#     preserving the first decimal point.
    
#     For example:
#       "15.2+#927" -> "15.2927" -> 15.2927 (returned as a float)
#     """
#     s = str(value)
#     s_temp = re.sub(r'[^0-9.]', '', s)
#     if '.' in s_temp:
#         parts = s_temp.split('.')
#         cleaned = parts[0] + '.' + ''.join(parts[1:])
#     else:
#         cleaned = s_temp
#     try:
#         return float(cleaned)
#     except ValueError:
#         return np.nan

# # ----------------------------
# # Load and preprocess training data
# # ----------------------------
# train_file_path = '/kaggle/input/mission-data-impossible/DMI_train.csv'
# print("[DEBUG] Loading training data from:", train_file_path)
# data = pd.read_csv(train_file_path, encoding='ISO-8859-1')
# print("[DEBUG] Training data loaded. Shape:", data.shape)
# print(data.head())

# # Define feature columns (f1 to f17)
# feature_cols = [f'f{i}' for i in range(1, 18)]
# print("[DEBUG] Cleaning feature columns f1–f17...")
# for col in tqdm(feature_cols, desc="Cleaning features", unit="column"):
#     data[col] = data[col].apply(clean_numeric)
    
# # Impute missing values using the median
# print("[DEBUG] Imputing missing values for training data...")
# data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
# print("[DEBUG] Missing values after imputation:")
# print(data[feature_cols].isnull().sum())

# # Optional EDA: display info and correlation heatmap
# print("[DEBUG] Data info:")
# data.info()
# print("[DEBUG] Summary statistics:")
# print(data.describe())
# plt.figure()
# sns.countplot(x='class', data=data, palette="viridis")
# plt.title('Target Class Distribution')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.show()
# plt.figure(figsize=(12,10))
# sns.heatmap(data[feature_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Feature Correlation Heatmap')
# plt.show()

# # ----------------------------
# # Encode target variable
# # ----------------------------
# print("[DEBUG] Encoding target variable...")
# le = LabelEncoder()
# data['target'] = le.fit_transform(data['class'])
# print("[DEBUG] Encoded target classes:", dict(zip(le.classes_, le.transform(le.classes_))))
# print("[DEBUG] Target distribution:")
# print(data['target'].value_counts())

# # ----------------------------
# # Split data into training and validation sets
# # ----------------------------
# X = data[feature_cols].values  
# y = data['target'].values

# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )
# print(f"[DEBUG] Training set size: {X_train.shape}, Validation set size: {X_val.shape}")

# # ----------------------------
# # Address class imbalance with SMOTE (if needed)
# # ----------------------------
# from imblearn.over_sampling import SMOTE
# print("[DEBUG] Applying SMOTE on training data...")
# smote = SMOTE(random_state=42, n_jobs=-1)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# print(f"[DEBUG] After SMOTE, training set size: {X_train_smote.shape}")
# print("[DEBUG] Class distribution after SMOTE:", np.bincount(y_train_smote))

# # ----------------------------
# # Train individual models
# # ----------------------------
# # 1. TabNet (pytorch-tabnet)
# import torch
# from pytorch_tabnet.tab_model import TabNetClassifier

# print("[DEBUG] Training TabNetClassifier...")
# tabnet_clf = TabNetClassifier(
#     n_d=64,
#     n_a=64,
#     n_steps=5,
#     gamma=1.3,
#     lambda_sparse=1e-3,
#     optimizer_fn=torch.optim.Adam,
#     optimizer_params=dict(lr=2e-2),
#     scheduler_params={"step_size":50, "gamma":0.9},
#     scheduler_fn=torch.optim.lr_scheduler.StepLR,
#     verbose=1,
#     device_name='cuda' if torch.cuda.is_available() else 'cpu'
# )

# tabnet_clf.fit(
#     X_train_smote, y_train_smote,
#     eval_set=[(X_val, y_val)],
#     eval_metric=['accuracy'],
#     max_epochs=200,
#     patience=20,
#     batch_size=1024,
#     virtual_batch_size=128,
#     num_workers=0,
#     drop_last=False
# )

# # 2. CatBoostClassifier
# from catboost import CatBoostClassifier

# print("[DEBUG] Training CatBoostClassifier...")
# cat_clf = CatBoostClassifier(
#     iterations=500,
#     learning_rate=0.01,
#     depth=6,
#     task_type='GPU',
#     loss_function='MultiClass',
#     verbose=100,
#     random_seed=42
# )
# cat_clf.fit(X_train_smote, y_train_smote, eval_set=(X_val, y_val), early_stopping_rounds=20)




# pip install tabgan==1.3.3


# # 3. LightGBM Classifier
# import lightgbm as lgb
# from lightgbm import LGBMClassifier

# print("[DEBUG] Training LightGBM Classifier...")
# lgb_clf = LGBMClassifier(
#     n_estimators=500,
#     learning_rate=0.01,
#     max_depth=7,
#     device='gpu',
#     gpu_platform_id=0,
#     gpu_device_id=0,
#     random_state=42
# )
# lgb_clf.fit(X_train_smote, y_train_smote, eval_set=[(X_val, y_val)])

# # ----------------------------
# # Ensemble: Average the predicted probabilities
# # ----------------------------
# print("[DEBUG] Generating ensemble predictions on validation data...")
# # Get probability predictions from each model
# probs_tabnet = tabnet_clf.predict_proba(X_val)
# probs_cat = cat_clf.predict_proba(X_val)
# probs_lgb = lgb_clf.predict_proba(X_val)

# # Average probabilities (simple ensemble)
# ensemble_probs = (probs_tabnet + probs_cat + probs_lgb) / 3.0
# ensemble_preds = np.argmax(ensemble_probs, axis=1)

# # ----------------------------
# # Evaluate Ensemble Model
# # ----------------------------
# val_acc = accuracy_score(y_val, ensemble_preds)
# val_f1 = f1_score(y_val, ensemble_preds, average='weighted')
# print(f"[DEBUG] Ensemble Validation Accuracy: {val_acc:.4f}")
# print(f"[DEBUG] Ensemble Validation Weighted F1 Score: {val_f1:.4f}")
# print("[DEBUG] Confusion Matrix:")
# print(confusion_matrix(y_val, ensemble_preds))
# print("[DEBUG] Classification Report:")
# print(classification_report(y_val, ensemble_preds, target_names=le.classes_))

# # ----------------------------
# # Generate predictions on test data and save submission
# # ----------------------------
# test_file_path = '/kaggle/input/mission-data-impossible/DMI_test_user.csv'
# submission_file_path = 'submission.csv'

# print("[DEBUG] Loading test data from:", test_file_path)
# test_data = pd.read_csv(test_file_path, encoding='ISO-8859-1')

# print("[DEBUG] Cleaning test data features...")
# for col in feature_cols:
#     test_data[col] = test_data[col].apply(clean_numeric)
# test_data[feature_cols] = test_data[feature_cols].fillna(test_data[feature_cols].median())
# X_test = test_data[feature_cols].values

# print("[DEBUG] Generating ensemble predictions on test data...")
# # Get probability predictions from each model
# probs_tabnet_test = tabnet_clf.predict_proba(X_test)
# probs_cat_test = cat_clf.predict_proba(X_test)
# probs_lgb_test = lgb_clf.predict_proba(X_test)

# # Average probabilities and determine predicted class
# ensemble_probs_test = (probs_tabnet_test + probs_cat_test + probs_lgb_test) / 3.0
# ensemble_preds_test = np.argmax(ensemble_probs_test, axis=1)
# ensemble_preds_labels = le.inverse_transform(ensemble_preds_test)

# submission = pd.DataFrame({
#     'Index': range(1, len(ensemble_preds_labels) + 1),
#     'class': ensemble_preds_labels
# })
# submission.to_csv(submission_file_path, index=False)
# print(f"[DEBUG] Submission file saved to {submission_file_path}.")











# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import re
# import time
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# import xgboost as xgb
# from xgboost.callback import TrainingCallback
# import contextlib
# import optuna  # Optional: Only needed if you explore advanced hyperparameter tuning

# # ----------------------------
# # Timer start
# # ----------------------------
# start_time = time.time()

# # ----------------------------
# # Custom Tqdm Callback for XGBoost
# # ----------------------------
# class TqdmCallback(TrainingCallback):
#     def __init__(self, total, eta, file=sys.stdout):
#         """
#         Args:
#             total (int): Total number of boosting rounds.
#             eta (float): Learning rate.
#             file: The file stream to write the progress bar (default sys.stdout).
#         """
#         self.eta = eta
#         self.total = total
#         self.pbar = tqdm(total=total, desc=f"Epoch 0/{total}, loss: N/A, eta: {eta}", unit="epoch", file=file)
    
#     def after_iteration(self, model, epoch, evals_log):
#         # Retrieve loss from the evaluation set (assumes it's named 'validation_0')
#         loss_list = evals_log.get('validation_0', {}).get('mlogloss', None)
#         loss = loss_list[-1] if loss_list is not None else float('nan')
#         self.pbar.set_description(f"Epoch {epoch+1}/{self.total}, loss: {loss:.5f}, eta: {self.eta}")
#         self.pbar.update(1)
#         return False
    
#     def after_training(self, model):
#         self.pbar.close()
#         return model

# # ----------------------------
# # Utility: Clean Numeric Values
# # ----------------------------
# def clean_numeric(value):
#     """
#     Remove all non-numeric characters from the string representation of 'value',
#     preserving the first decimal point.
    
#     Example:
#       "15.2+#927" -> "15.2927" -> returns 15.2927 as a float.
#     """
#     s = str(value)
#     s_temp = re.sub(r'[^0-9.]', '', s)
#     if '.' in s_temp:
#         parts = s_temp.split('.')
#         cleaned = parts[0] + '.' + ''.join(parts[1:])
#     else:
#         cleaned = s_temp
#     try:
#         return float(cleaned)
#     except ValueError:
#         return np.nan

# # ----------------------------
# # Feature Engineering Function
# # ----------------------------
# def feature_engineering(df, feature_cols):
#     """
#     Generate new features based on original features.
#     Adds:
#       - Row-wise sum, mean, and standard deviation.
#       - Two interaction features: product of f1 and f2, and product of f3 and f4.
#     """
#     df_fe = df.copy()
#     # Use progress_apply with tqdm for row-wise operations
#     tqdm.pandas(desc="Calculating row-wise sum")
#     df_fe['feat_sum'] = df_fe[feature_cols].progress_apply(np.sum, axis=1)
#     tqdm.pandas(desc="Calculating row-wise mean")
#     df_fe['feat_mean'] = df_fe[feature_cols].progress_apply(np.mean, axis=1)
#     tqdm.pandas(desc="Calculating row-wise std")
#     df_fe['feat_std'] = df_fe[feature_cols].progress_apply(np.std, axis=1)
    
#     df_fe['f1_f2_interact'] = df_fe[feature_cols[0]] * df_fe[feature_cols[1]]
#     df_fe['f3_f4_interact'] = df_fe[feature_cols[2]] * df_fe[feature_cols[3]]
#     return df_fe

# # ----------------------------
# # Data Loading and Preprocessing
# # ----------------------------
# print("[DEBUG] Loading training data...")
# train_file_path = '/kaggle/input/mission-data-impossible/DMI_train.csv'
# data = pd.read_csv(train_file_path, encoding='ISO-8859-1')
# print(f"[DEBUG] Training data loaded. Shape: {data.shape}")
# print("[DEBUG] First 5 rows:")
# print(data.head())

# # Define feature columns (f1 to f17)
# feature_cols = [f'f{i}' for i in range(1, 18)]
# print(f"[DEBUG] Feature columns: {feature_cols}")

# # Clean features with tqdm progress
# print("[DEBUG] Cleaning feature columns...")
# for col in tqdm(feature_cols, desc="Cleaning features", unit="column"):
#     data[col] = data[col].apply(clean_numeric)
    
# # Impute missing values using median
# print("[DEBUG] Imputing missing values...")
# data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
# print("[DEBUG] Missing values after imputation:")
# print(data[feature_cols].isnull().sum())

# # Optional EDA
# print("[DEBUG] Running EDA...")
# data.info()
# print(data.describe())

# plt.figure()
# sns.countplot(x='class', data=data, palette="viridis")
# plt.title('Target Class Distribution')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(12,10))
# sns.heatmap(data[feature_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Feature Correlation Heatmap')
# plt.show()

# # ----------------------------
# # Feature Engineering
# # ----------------------------
# print("[DEBUG] Running feature engineering...")
# data_fe = feature_engineering(data, feature_cols)
# engineered_features = ['feat_sum', 'feat_mean', 'feat_std', 'f1_f2_interact', 'f3_f4_interact']
# all_features = feature_cols + engineered_features
# print(f"[DEBUG] Total features after engineering: {len(all_features)}")
# print(f"[DEBUG] Engineered features: {engineered_features}")

# # ----------------------------
# # Encode Target Variable
# # ----------------------------
# print("[DEBUG] Encoding target variable...")
# le = LabelEncoder()
# data_fe['target'] = le.fit_transform(data_fe['class'])
# print(f"[DEBUG] Encoded target classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# # ----------------------------
# # Split Data into Training and Validation Sets
# # ----------------------------
# print("[DEBUG] Splitting data into training and validation sets...")
# X = data_fe[all_features].values
# y = data_fe['target'].values
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )
# print(f"[DEBUG] Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

# # ----------------------------
# # Randomized Search for Hyperparameter Tuning
# # ----------------------------
# print("[DEBUG] Starting randomized hyperparameter search...")
# # Reduced parameter grid for faster computation
# param_grid = {
#     'eta': [0.01, 0.1, 1.0],
#     'gamma': [0, 0.1],
#     'n_estimators': [10, 100, 500],
#     'max_depth': [2, 4, 6],
#     'min_child_weight': [1, 2],
#     'nthread': [2]
# }
# print(f"[DEBUG] Parameter grid: {param_grid}")

# # Create XGBoost classifier instance
# model = xgb.XGBClassifier(
#     objective='multi:softprob',
#     num_class=len(np.unique(y)),
#     eval_metric='mlogloss',
#     tree_method='gpu_hist' if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else 'auto',
#     use_label_encoder=False,
#     early_stopping_rounds=20,
#     verbosity=0
# )

# # Use StratifiedKFold with 3 splits
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

# random_search = RandomizedSearchCV(
#     estimator=model,
#     param_distributions=param_grid,
#     n_iter=10,  # Adjust the number of iterations based on your time/accuracy tradeoff
#     cv=skf,
#     scoring="accuracy",
#     n_jobs=-1,  # Utilize all available cores
#     verbose=1,
#     random_state=42,
#     return_train_score=False
# )

# grid_start = time.time()
# random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)])
# grid_end = time.time()
# print(f"[DEBUG] Randomized search completed in {grid_end - grid_start:.2f} seconds.")
# print(f"[DEBUG] Best CV score: {random_search.best_score_:.4f}")
# print(f"[DEBUG] Best parameters: {random_search.best_params_}")

# best_model = random_search.best_estimator_

# # ----------------------------
# # Final Training with Custom Tqdm Callback
# # ----------------------------
# print("[DEBUG] Training final XGBoost model with best parameters and custom progress bar...")
# # Ensure verbosity is off
# best_model.set_params(verbosity=0)
# n_estimators = best_model.get_params()['n_estimators']
# eta = best_model.get_params()['eta']

# # Redirect stderr to devnull during fit to suppress unwanted logs;
# # our callback writes to sys.stdout so it remains visible.
# with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
#     best_model.fit(
#         X_train,
#         y_train,
#         eval_set=[(X_val, y_val)],
#         callbacks=[TqdmCallback(total=n_estimators, eta=eta, file=sys.stdout)],
#         verbose=False
#     )
# print("[DEBUG] Final model training completed.")

# # ----------------------------
# # Evaluate Final Model on Validation Data
# # ----------------------------
# print("[DEBUG] Evaluating final model on validation data...")
# y_val_pred_probs = best_model.predict_proba(X_val)
# y_val_pred = np.argmax(y_val_pred_probs, axis=1)

# val_acc = accuracy_score(y_val, y_val_pred)
# val_f1 = f1_score(y_val, y_val_pred, average='weighted')
# print(f"[DEBUG] Final XGBoost Validation Accuracy: {val_acc:.4f}")
# print(f"[DEBUG] Final XGBoost Validation Weighted F1 Score: {val_f1:.4f}")
# print("[DEBUG] Confusion Matrix:")
# print(confusion_matrix(y_val, y_val_pred))
# print("[DEBUG] Classification Report:")
# print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# # ----------------------------
# # Preprocess Test Data and Feature Engineering
# # ----------------------------
# print("[DEBUG] Loading test data...")
# test_file_path = '/kaggle/input/mission-data-impossible/DMI_test_user.csv'
# test_data = pd.read_csv(test_file_path, encoding='ISO-8859-1')

# print("[DEBUG] Cleaning test data features...")
# for col in tqdm(feature_cols, desc="Cleaning test features", unit="column"):
#     test_data[col] = test_data[col].apply(clean_numeric)
# test_data[feature_cols] = test_data[feature_cols].fillna(test_data[feature_cols].median())

# print("[DEBUG] Running feature engineering on test data...")
# test_data_fe = feature_engineering(test_data, feature_cols)
# X_test = test_data_fe[all_features].values

# # ----------------------------
# # Batch Prediction on Test Data
# # ----------------------------
# print("[DEBUG] Generating predictions on test data in batches...")
# batch_size = 1000
# n_test = X_test.shape[0]
# test_preds = []

# for start in tqdm(range(0, n_test, batch_size), desc="Predicting test batches"):
#     end = min(start + batch_size, n_test)
#     batch = X_test[start:end]
#     batch_preds = best_model.predict_proba(batch)
#     batch_pred_labels = np.argmax(batch_preds, axis=1)
#     test_preds.extend(batch_pred_labels)
# test_preds = np.array(test_preds)
# test_preds_labels = le.inverse_transform(test_preds)

# # ----------------------------
# # Save Submission File
# # ----------------------------
# submission_file_path = 'submission.csv'
# print(f"[DEBUG] Saving submission file to: {submission_file_path}")
# submission = pd.DataFrame({
#     'Index': range(1, len(test_preds_labels) + 1),
#     'class': test_preds_labels
# })
# submission.to_csv(submission_file_path, index=False)
# print(f"[DEBUG] Submission file saved to {submission_file_path}.")

# # ----------------------------
# # Final Script Runtime
# # ----------------------------
# end_time = time.time()
# print(f"[DEBUG] Script completed in {end_time - start_time:.2f} seconds.")












#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost.callback import TrainingCallback
import contextlib
import optuna

# ----------------------------
# Timer start
# ----------------------------
start_time = time.time()

# ----------------------------
# Custom Tqdm Callback for XGBoost
# ----------------------------
class TqdmCallback(TrainingCallback):
    def __init__(self, total, eta, file=sys.stdout):
        """
        Args:
            total (int): Total number of boosting rounds.
            eta (float): Learning rate.
            file: The file stream to write the progress bar (default sys.stdout).
        """
        self.eta = eta
        self.total = total
        self.pbar = tqdm(total=total, desc=f"Epoch 0/{total}, loss: N/A, eta: {eta}", unit="epoch", file=file)
    
    def after_iteration(self, model, epoch, evals_log):
        loss_list = evals_log.get('validation_0', {}).get('mlogloss', None)
        loss = loss_list[-1] if loss_list is not None else float('nan')
        self.pbar.set_description(f"Epoch {epoch+1}/{self.total}, loss: {loss:.5f}, eta: {self.eta}")
        self.pbar.update(1)
        return False
    
    def after_training(self, model):
        self.pbar.close()
        return model

# ----------------------------
# Utility: Clean Numeric Values
# ----------------------------
def clean_numeric(value):
    """
    Remove all non-numeric characters from the string representation of 'value',
    preserving the first decimal point.
    
    Example:
      "15.2+#927" -> "15.2927" -> returns 15.2927 as a float.
    """
    s = str(value)
    s_temp = re.sub(r'[^0-9.]', '', s)
    if '.' in s_temp:
        parts = s_temp.split('.')
        cleaned = parts[0] + '.' + ''.join(parts[1:])
    else:
        cleaned = s_temp
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

# ----------------------------
# Feature Engineering Function
# ----------------------------
def feature_engineering(df, feature_cols):
    """
    Generate new features based on original features.
    Adds:
      - Row-wise sum, mean, and standard deviation.
      - Two interaction features: product of f1 and f2, and product of f3 and f4.
    """
    df_fe = df.copy()
    tqdm.pandas(desc="Calculating row-wise sum")
    df_fe['feat_sum'] = df_fe[feature_cols].progress_apply(np.sum, axis=1)
    tqdm.pandas(desc="Calculating row-wise mean")
    df_fe['feat_mean'] = df_fe[feature_cols].progress_apply(np.mean, axis=1)
    tqdm.pandas(desc="Calculating row-wise std")
    df_fe['feat_std'] = df_fe[feature_cols].progress_apply(np.std, axis=1)
    
    df_fe['f1_f2_interact'] = df_fe[feature_cols[0]] * df_fe[feature_cols[1]]
    df_fe['f3_f4_interact'] = df_fe[feature_cols[2]] * df_fe[feature_cols[3]]
    return df_fe

# ----------------------------
# Data Loading and Preprocessing
# ----------------------------
print("[DEBUG] Loading training data...")
train_file_path = '/kaggle/input/mission-data-impossible/DMI_train.csv'
data = pd.read_csv(train_file_path, encoding='ISO-8859-1')
print(f"[DEBUG] Training data loaded. Shape: {data.shape}")
print("[DEBUG] First 5 rows:")
print(data.head())

# Define feature columns (f1 to f17)
feature_cols = [f'f{i}' for i in range(1, 18)]
print(f"[DEBUG] Feature columns: {feature_cols}")

# Clean features with tqdm progress
print("[DEBUG] Cleaning feature columns...")
for col in tqdm(feature_cols, desc="Cleaning features", unit="column"):
    data[col] = data[col].apply(clean_numeric)
    
# Impute missing values using median
print("[DEBUG] Imputing missing values...")
data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
print("[DEBUG] Missing values after imputation:")
print(data[feature_cols].isnull().sum())

# Optional EDA
print("[DEBUG] Running EDA...")
data.info()
print(data.describe())

plt.figure()
sns.countplot(x='class', data=data, palette="viridis")
plt.title('Target Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(data[feature_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# ----------------------------
# Feature Engineering
# ----------------------------
print("[DEBUG] Running feature engineering...")
data_fe = feature_engineering(data, feature_cols)
engineered_features = ['feat_sum', 'feat_mean', 'feat_std', 'f1_f2_interact', 'f3_f4_interact']
all_features = feature_cols + engineered_features
print(f"[DEBUG] Total features after engineering: {len(all_features)}")
print(f"[DEBUG] Engineered features: {engineered_features}")

# ----------------------------
# Encode Target Variable
# ----------------------------
print("[DEBUG] Encoding target variable...")
le = LabelEncoder()
data_fe['target'] = le.fit_transform(data_fe['class'])
print(f"[DEBUG] Encoded target classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ----------------------------
# Split Data into Training and Validation Sets
# ----------------------------
print("[DEBUG] Splitting data into training and validation sets...")
X = data_fe[all_features].values
y = data_fe['target'].values
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"[DEBUG] Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

# ----------------------------
# Bayesian Optimization with Optuna
# ----------------------------
def objective(trial):
    # Define hyperparameter search space
    param = {
        'eta': trial.suggest_loguniform('eta', 0.001, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'nthread': 2,  # fixed for now; adjust based on your environment
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'tree_method': 'gpu_hist' if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else 'auto',
        'early_stopping_rounds': 20,
        'verbosity': 0
    }
    
    # Create and train the XGBoost classifier with early stopping
    model = xgb.XGBClassifier(**param)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict and compute accuracy on validation set
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    return accuracy

print("[DEBUG] Starting Bayesian hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)  # Increase n_trials as needed

print("Best trial:")
trial = study.best_trial
print(f"  Validation Accuracy: {trial.value:.4f}")
print("  Best Hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# ----------------------------
# Final Model Training with Best Hyperparameters
# ----------------------------
print("[DEBUG] Training final XGBoost model with best parameters and custom progress bar...")
# Use the best hyperparameters from the study
best_params = trial.params
best_params.update({
    'nthread': 2,
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y)),
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'tree_method': 'gpu_hist' if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else 'auto',
    'early_stopping_rounds': 20,
    'verbosity': 0
})

# Create final model instance
best_model = xgb.XGBClassifier(**best_params)
n_estimators = best_model.get_params()['n_estimators']
eta = best_model.get_params()['eta']

# Redirect stderr to devnull to suppress unwanted logs while keeping our Tqdm progress visible
with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[TqdmCallback(total=n_estimators, eta=eta, file=sys.stdout)],
        verbose=False
    )
print("[DEBUG] Final model training completed.")

# ----------------------------
# Evaluate Final Model on Validation Data
# ----------------------------
print("[DEBUG] Evaluating final model on validation data...")
y_val_pred_probs = best_model.predict_proba(X_val)
y_val_pred = np.argmax(y_val_pred_probs, axis=1)

val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
print(f"[DEBUG] Final XGBoost Validation Accuracy: {val_acc:.4f}")
print(f"[DEBUG] Final XGBoost Validation Weighted F1 Score: {val_f1:.4f}")
print("[DEBUG] Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("[DEBUG] Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# ----------------------------
# Preprocess Test Data and Feature Engineering
# ----------------------------
print("[DEBUG] Loading test data...")
test_file_path = '/kaggle/input/mission-data-impossible/DMI_test_user.csv'
test_data = pd.read_csv(test_file_path, encoding='ISO-8859-1')

print("[DEBUG] Cleaning test data features...")
for col in tqdm(feature_cols, desc="Cleaning test features", unit="column"):
    test_data[col] = test_data[col].apply(clean_numeric)
test_data[feature_cols] = test_data[feature_cols].fillna(test_data[feature_cols].median())

print("[DEBUG] Running feature engineering on test data...")
test_data_fe = feature_engineering(test_data, feature_cols)
X_test = test_data_fe[all_features].values

# ----------------------------
# Batch Prediction on Test Data
# ----------------------------
print("[DEBUG] Generating predictions on test data in batches...")
batch_size = 1000
n_test = X_test.shape[0]
test_preds = []

for start in tqdm(range(0, n_test, batch_size), desc="Predicting test batches"):
    end = min(start + batch_size, n_test)
    batch = X_test[start:end]
    batch_preds = best_model.predict_proba(batch)
    batch_pred_labels = np.argmax(batch_preds, axis=1)
    test_preds.extend(batch_pred_labels)
test_preds = np.array(test_preds)
test_preds_labels = le.inverse_transform(test_preds)

# ----------------------------
# Save Submission File
# ----------------------------
submission_file_path = 'submission.csv'
print(f"[DEBUG] Saving submission file to: {submission_file_path}")
submission = pd.DataFrame({
    'Index': range(1, len(test_preds_labels) + 1),
    'class': test_preds_labels
})
submission.to_csv(submission_file_path, index=False)
print(f"[DEBUG] Submission file saved to {submission_file_path}.")

# ----------------------------
# Final Script Runtime
# ----------------------------
end_time = time.time()
print(f"[DEBUG] Script completed in {end_time - start_time:.2f} seconds.")





