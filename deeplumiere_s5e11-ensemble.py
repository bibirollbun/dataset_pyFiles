# # --- INSTALLATION ---
# # !pip install -qq pytabkit

# import pandas as pd
# import numpy as np
# import warnings
# import os
# import sys
# import gc
# from contextlib import contextmanager
# from scipy.optimize import minimize
# from sklearn.model_selection import StratifiedKFold, KFold
# from sklearn.preprocessing import OrdinalEncoder, RobustScaler
# from sklearn.metrics import roc_auc_score, log_loss
# from sklearn.base import BaseEstimator, TransformerMixin

# # --- MODELS ---
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# warnings.filterwarnings('ignore')

# # --- Configuration ---
# SEED = 42
# N_FOLDS = 10
# USE_GPU = True

# # --- 1. Data Loading & Original Data Injection ---
# # YOU MUST ADD THE ORIGINAL DATASET TO KAGGLE INPUT
# # Search for "Loan Prediction Dataset" or "Playground S5E11 Original"
# print("Loading Data...")
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# # Try to load original data (The secret sauce for 0.94+)
# try:
#     # Adjust path to where you stored the original data
#     df_orig = pd.read_csv("/kaggle/input/load-pred-dataset/original_dataset.csv")
    
#     # Align columns
#     common_cols = [c for c in df_train.columns if c in df_orig.columns and c != 'id']
#     df_orig = df_orig[common_cols]
    
#     # Concatenate
#     df_train = pd.concat([df_train, df_orig], axis=0).reset_index(drop=True)
#     print(f"Original data added. New Train Shape: {df_train.shape}")
# except:
#     print("Original dataset not found! Score will be lower.")

# test_ids = df_test['id'].values
# y = df_train['loan_paid_back']

# df_train['is_train'] = 1
# df_test['is_train'] = 0
# df_all = pd.concat([df_train.drop(['loan_paid_back'], axis=1), df_test], axis=0).reset_index(drop=True).drop(['id'], axis=1)

# # --- 2. Advanced Feature Engineering ---
# def engineer_features(df):
#     df = df.copy()
    
#     # 1. Cleaning & Imputation
#     num_cols = df.select_dtypes(include=['number']).columns.tolist()
#     cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
#     for col in num_cols:
#         if col != 'is_train':
#             df[col] = df[col].fillna(df[col].median())
    
#     for col in cat_cols:
#         df[col] = df[col].fillna('Missing')

#     # 2. Mathematical Transformations
#     # Loan to Income Ratio (Critical)
#     if 'loan_amount' in df.columns and 'annual_income' in df.columns:
#         df['LTI'] = df['loan_amount'] / (df['annual_income'] + 1)
#         df['income_to_loan'] = df['annual_income'] / (df['loan_amount'] + 1)
    
#     # Monthly burden
#     if 'monthly_income' not in df.columns and 'annual_income' in df.columns:
#         df['monthly_income'] = df['annual_income'] / 12
        
#     # 3. Binning (Helping Trees find splits)
#     # Credit scores are non-linear. 650 is vastly different from 600.
#     if 'credit_score' in df.columns:
#         df['credit_score_bin'] = pd.cut(df['credit_score'], bins=10, labels=False)

#     # 4. Interaction / Aggregation Features (The 0.94 booster)
#     # How does this person's income compare to others with the same Grade?
#     if 'grade' in df.columns and 'annual_income' in df.columns:
#         df['income_by_grade'] = df.groupby('grade')['annual_income'].transform('mean')
#         df['income_div_grade_mean'] = df['annual_income'] / df['income_by_grade']

#     return df, cat_cols

# df_all, cat_cols = engineer_features(df_all)

# # --- 3. Target Encoding (Crucial for High Score) ---
# # We use K-Fold Target Encoding to prevent overfitting
# class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, colnames, targetName, n_fold=5, verbosity=False, discardOriginal_col=False):
#         self.colnames = colnames
#         self.targetName = targetName
#         self.n_fold = n_fold
#         self.verbosity = verbosity
#         self.discardOriginal_col = discardOriginal_col
#         self.mapping = {}

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         # If no y is provided (test set), map using global training means
#         if y is None:
#             X_encoded = X.copy()
#             for col in self.colnames:
#                 X_encoded[col + '_TE'] = X[col].map(self.mapping[col])
#             return X_encoded

#         # If y is provided (training set), use K-Fold
#         X_encoded = X.copy()
#         kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=SEED)
        
#         # Initialize columns
#         for col in self.colnames:
#             X_encoded[col + '_TE'] = np.nan
            
#         # Generate K-Fold encodings
#         for tr_ind, val_ind in kf.split(X):
#             X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
#             y_tr = y.iloc[tr_ind]
            
#             for col in self.colnames:
#                 # Compute means on training fold
#                 means = X_tr.groupby(col)[self.targetName].mean()
#                 # Map to validation fold
#                 X_encoded.loc[X_encoded.index[val_ind], col + '_TE'] = X_val[col].map(means)
        
#         # Fill missing values with global mean
#         for col in self.colnames:
#             global_mean = y.mean()
#             X_encoded[col + '_TE'] = X_encoded[col + '_TE'].fillna(global_mean)
#             # Save global map for test set
#             self.mapping[col] = X.groupby(col)[self.targetName].mean()
            
#         return X_encoded

# # Split back to train/test for encoding
# X = df_all[df_all['is_train'] == 1].drop(['is_train'], axis=1).copy()
# # Re-attach target for encoder
# X['target'] = y 
# X_test = df_all[df_all['is_train'] == 0].drop(['is_train'], axis=1).copy()

# # Apply Target Encoding to high cardinality or ordinal-like cats (Grade, Subgrade)
# te_cols = [c for c in cat_cols if 'grade' in c.lower() or 'purpose' in c.lower()]
# if len(te_cols) > 0:
#     print(f"Target Encoding: {te_cols}")
#     encoder = KFoldTargetEncoder(colnames=te_cols, targetName='target', n_fold=5)
#     X = encoder.transform(X, y) # Fit & Transform Train
#     X_test = encoder.transform(X_test, None) # Transform Test using maps
    
#     # Drop target from X now
#     X = X.drop('target', axis=1)

# # Final Categorical Processing
# # CatBoost likes raw categories. XGB/LGBM prefer encoded.
# X_cat = X.copy()
# X_test_cat = X_test.copy()

# # Ordinal Encode for XGB/LGBM
# ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# X[cat_cols] = ord_enc.fit_transform(X[cat_cols])
# X_test[cat_cols] = ord_enc.transform(X_test[cat_cols])

# # --- 4. Models (Tuned for 0.94+) ---
# # Note: Lower Learning Rate + Higher Estimators = Better Generalization
# models = {
#     'xgb': XGBClassifier(
#         n_estimators=5000, 
#         learning_rate=0.01, 
#         max_depth=6, 
#         subsample=0.7, 
#         colsample_bytree=0.7, 
#         gamma=0.2, # Regularization
#         reg_alpha=0.1,
#         reg_lambda=1.0,
#         tree_method='hist', 
#         device='cuda' if USE_GPU else 'cpu',
#         random_state=SEED, 
#         eval_metric='auc',
#         early_stopping_rounds=300
#     ),
#     'lgb': LGBMClassifier(
#         n_estimators=5000, 
#         learning_rate=0.01, 
#         max_depth=8, 
#         num_leaves=64,
#         subsample=0.7, 
#         colsample_bytree=0.7,
#         reg_alpha=0.1,
#         reg_lambda=0.1,
#         device='gpu' if USE_GPU else 'cpu',
#         random_state=SEED, 
#         metric='auc', 
#         verbosity=-1
#     ),
#     'cat': CatBoostClassifier(
#         iterations=5000, 
#         learning_rate=0.01, 
#         depth=6,
#         l2_leaf_reg=5, # Higher reg for CatBoost
#         task_type='GPU' if USE_GPU else 'CPU',
#         random_seed=SEED, 
#         eval_metric='AUC', 
#         verbose=0, 
#         allow_writing_files=False
#     )
# }

# # --- 5. Training Loop ---
# kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# oof_preds = {k: np.zeros(len(X)) for k in models}
# test_preds = {k: np.zeros(len(X_test)) for k in models}

# print(f"Starting {N_FOLDS}-Fold Training...")

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     # Standard Data (XGB/LGBM)
#     X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     # Categorical Data (CatBoost)
#     X_tr_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    
#     y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
#     for name, model in models.items():
#         # Clean / Clone model
#         clf = model.__class__(**model.get_params())
        
#         if name == 'cat':
#             # CatBoost handles categories natively
#             clf.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), cat_features=cat_cols, verbose=False)
#             val_pred = clf.predict_proba(X_val_cat)[:, 1]
#             test_pred = clf.predict_proba(X_test_cat)[:, 1]
        
#         elif name == 'xgb':
#             clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
#             val_pred = clf.predict_proba(X_val)[:, 1]
#             test_pred = clf.predict_proba(X_test)[:, 1]
            
#         else: # lgb
#             clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])
#             val_pred = clf.predict_proba(X_val)[:, 1]
#             test_pred = clf.predict_proba(X_test)[:, 1]
        
#         oof_preds[name][val_idx] = val_pred
#         test_preds[name] += test_pred / N_FOLDS
    
#     # Quick Check for fold performance
#     fold_auc = roc_auc_score(y_val, oof_preds['cat'][val_idx]) # Checking Catboost as proxy
#     print(f"Fold {fold+1} CatBoost AUC: {fold_auc:.5f}")

# # --- 6. Optimization ---
# print("\nOptimizing Ensemble Weights...")
# model_names = list(models.keys())
# # Limit OOF to just the original train size (remove synthetic overlap if any, but StratifiedKFold handles index alignment usually)
# # However, since we added original data, X is larger than original train.
# # We should only evaluate OOF on the specific rows from the competition train set if we want strict Kaggle alignment,
# # BUT for optimization, using all data is usually fine.

# oof_matrix = np.column_stack([oof_preds[name] for name in model_names])
# test_matrix = np.column_stack([test_preds[name] for name in model_names])

# def minimize_neg_auc(weights):
#     weights = np.array(weights)
#     weights = np.exp(weights) / np.sum(np.exp(weights)) # Softmax
#     final_pred = np.sum(weights * oof_matrix, axis=1)
#     return -roc_auc_score(y, final_pred)

# init_weights = [1.0 / len(model_names)] * len(model_names)
# res = minimize(minimize_neg_auc, init_weights, method='Nelder-Mead', tol=1e-6)
# best_weights = np.exp(res.x) / np.sum(np.exp(res.x))

# print("\nBest Weights:")
# for name, w in zip(model_names, best_weights):
#     print(f"{name}: {w:.4f}")

# final_test = np.sum(best_weights * test_matrix, axis=1)

# # --- 7. Submission ---
# submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': final_test})
# submission.to_csv('submission_grandmaster.csv', index=False)
# print("Submission saved.")


# --- 1. IMPORTS & SETUP ---
import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

# --- MODELS ---
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SEED = 42
N_FOLDS = 10  # Use 5 for faster iteration, 10 for final submission
USE_GPU = True

# --- 2. DATA LOADING ---
print("Loading Data...")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# Try to load original data
try:
    df_orig = pd.read_csv("/kaggle/input/load-pred-dataset/original_dataset.csv")
    common_cols = [c for c in df_train.columns if c in df_orig.columns and c != 'id']
    df_orig = df_orig[common_cols]
    df_train = pd.concat([df_train, df_orig], axis=0).reset_index(drop=True)
    print(f"Original data added. New Train Shape: {df_train.shape}")
except:
    print("Original dataset not found. Proceeding without it.")

test_ids = df_test['id'].values
y = df_train['loan_paid_back']

# Combine for processing
df_train['is_train'] = 1
df_test['is_train'] = 0
df_all = pd.concat([df_train.drop(['loan_paid_back'], axis=1), df_test], axis=0).reset_index(drop=True).drop(['id'], axis=1)

# --- 3. FEATURE ENGINEERING FUNCTIONS ---

def engineer_features(df):
    df = df.copy()
    
    # 1. Ratios & Maths
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        df['LTI'] = df['loan_amount'] / (df['annual_income'] + 1)
        df['income_to_loan'] = df['annual_income'] / (df['loan_amount'] + 1)
        df['monthly_income'] = df['annual_income'] / 12
    
    # 2. Binning
    if 'credit_score' in df.columns:
        # Simple binning, handled as category later
        df['credit_score_bin'] = pd.cut(df['credit_score'], bins=10, labels=False)

    return df

def add_cluster_features(df, n_clusters=10):
    print(f"--- Generating Cluster Features (k={n_clusters}) ---")
    df = df.copy()
    
    # Select numerical columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    exclude = ['is_train', 'fold']
    num_cols = [c for c in num_cols if c not in exclude]
    
    # Scale for KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols].fillna(0))
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    df['cluster_profile'] = kmeans.fit_predict(X_scaled)
    
    # Add distances to cluster centers
    dists = kmeans.transform(X_scaled)
    for i in range(n_clusters):
        df[f'dist_cluster_{i}'] = dists[:, i]
        
    return df

def add_knn_features(X_train, y_train, X_test, n_neighbors=10):
    print(f"--- Generating KNN Features (k={n_neighbors}) ---")
    
    # Prepare numeric data
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    
    # Fill NA with 0 for KNN (Trees handle NA, but KNN doesn't)
    X_tr_scaled = scaler.fit_transform(X_train[num_cols].fillna(0))
    X_te_scaled = scaler.transform(X_test[num_cols].fillna(0))
    
    # Outputs
    train_knn_prob = np.zeros(len(X_train))
    train_knn_dist = np.zeros(len(X_train))
    test_knn_prob = np.zeros(len(X_test))
    test_knn_dist = np.zeros(len(X_test))
    
    # 1. Train Features (Cross-Validated to prevent leakage)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold, (idx_tr, idx_val) in enumerate(skf.split(X_train, y_train)):
        # Train KNN on fold
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        knn.fit(X_tr_scaled[idx_tr], y_train.iloc[idx_tr])
        
        # Predict on validation fold
        train_knn_prob[idx_val] = knn.predict_proba(X_tr_scaled[idx_val])[:, 1]
        dists, _ = knn.kneighbors(X_tr_scaled[idx_val])
        train_knn_dist[idx_val] = dists.mean(axis=1)
        
    # 2. Test Features (Train on Full Train Data)
    knn_full = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn_full.fit(X_tr_scaled, y_train)
    test_knn_prob = knn_full.predict_proba(X_te_scaled)[:, 1]
    dists_test, _ = knn_full.kneighbors(X_te_scaled)
    test_knn_dist = dists_test.mean(axis=1)
    
    # Add to DataFrames
    X_train['knn_prob'] = train_knn_prob
    X_train['knn_dist'] = train_knn_dist
    X_test['knn_prob'] = test_knn_prob
    X_test['knn_dist'] = test_knn_dist
    
    return X_train, X_test

# --- 4. EXECUTE PIPELINE ---

# 4.1 Basic Engineering
df_all = engineer_features(df_all)

# 4.2 Cluster Features (The "Secret Sauce")
df_all = add_cluster_features(df_all, n_clusters=7)

# 4.3 Split back for KNN & Training
X = df_all[df_all['is_train'] == 1].drop(['is_train'], axis=1).copy()
X_test = df_all[df_all['is_train'] == 0].drop(['is_train'], axis=1).copy()

# 4.4 KNN Features (Your Request)
# This finds "twins" in the dataset
X, X_test = add_knn_features(X, y, X_test, n_neighbors=100)

# 4.5 Efficient Categorical Handling
# Instead of slow TargetEncoding, we convert to category type for Native GBDT support
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical Columns: {cat_cols}")

for col in cat_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# --- 5. MODELS (Optimized for Efficiency) ---
models = {
    'xgb': XGBClassifier(
        n_estimators=3000,
        learning_rate=0.015,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,  # Speed boost
        tree_method='hist',
        device='cuda' if USE_GPU else 'cpu',
        eval_metric='auc',
        early_stopping_rounds=300,
        random_state=SEED
    ),
    'lgb': LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.015,
        max_depth=8,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        device='gpu' if USE_GPU else 'cpu', # Check Kaggle specs for LGBM GPU
        metric='auc',
        verbosity=-1,
        random_state=SEED
    ),
    'cat': CatBoostClassifier(
        iterations=3000,
        learning_rate=0.015,
        depth=6,
        cat_features=cat_cols, # Native handling
        task_type='GPU' if USE_GPU else 'CPU',
        eval_metric='AUC',
        verbose=0,
        random_seed=SEED,
        allow_writing_files=False
    )
}

# --- 6. TRAINING LOOP ---
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = {k: np.zeros(len(X)) for k in models}
test_preds = {k: np.zeros(len(X_test)) for k in models}

print(f"\nStarting {N_FOLDS}-Fold Training...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    for name, model in models.items():
        clf = model.__class__(**model.get_params())
        
        # Fit
        if name == 'cat':
            clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        elif name == 'xgb':
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else: # lgb
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            
        # Predict
        val_pred = clf.predict_proba(X_val)[:, 1]
        test_pred = clf.predict_proba(X_test)[:, 1]
        
        # Store
        oof_preds[name][val_idx] = val_pred
        test_preds[name] += test_pred / N_FOLDS
        
    # Fold Score (Using XGB as proxy for progress)
    fold_auc = roc_auc_score(y_val, oof_preds['xgb'][val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

# --- 7. STACKING (Hierarchical Efficiency) ---
print("\nTraining Meta-Model (Ridge Stacking)...")

# Prepare Meta-Data
X_meta = pd.DataFrame(oof_preds)
X_test_meta = pd.DataFrame(test_preds)

# Ridge Stacking (Better than weighted avg)
meta_model = RidgeClassifier(alpha=10.0)
calibrated_meta = CalibratedClassifierCV(meta_model, cv=5)
calibrated_meta.fit(X_meta, y)

# Final Predictions
final_preds = calibrated_meta.predict_proba(X_test_meta)[:, 1]

# --- 8. SUBMISSION ---
submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': final_preds})
submission.to_csv('submission.csv', index=False)
print("Submission saved successfully.")

