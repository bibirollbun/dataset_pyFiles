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
import warnings
import gc
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SEEDS = [42, 2025, 123]
N_SPLITS = 10
TARGET = 'diagnosed_diabetes'
# RELAXED THRESHOLDS FOR SUCCESS
PSEUDO_HIGH_THRESHOLD = 0.95 
PSEUDO_LOW_THRESHOLD = 0.05

# 1. LOAD DATA
print("Loading data...")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

# 2. FEATURE ENGINEERING
def advanced_feature_engineering(df):
    df = df.copy()
    
    # Blood Pressure
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['MAP'] = (df['systolic_bp'] + (2 * df['diastolic_bp'])) / 3
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'].replace(0, 1).clip(lower=1))
    
    # Cholesterol
    if 'ldl_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
        df['chol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'].replace(0, 1).clip(lower=1))
        df['non_hdl'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # BMI & Interactions
    if 'bmi' in df.columns and 'age' in df.columns:
        df['bmi_age'] = df['bmi'] * df['age']
        df['age_squared'] = df['age'] ** 2
    
    if 'bmi' in df.columns and 'physical_activity_minutes_per_week' in df.columns:
        df['activity_intensity'] = df['physical_activity_minutes_per_week'] / (df['bmi'].replace(0, 1))

    # Log Transforms
    skewed_cols = ['bmi', 'physical_activity_minutes_per_week', 'ldl_cholesterol', 'triglycerides']
    for col in skewed_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0).clip(lower=0))
            
    return df

print("Engineering features...")
train_df = advanced_feature_engineering(train_df)
test_df = advanced_feature_engineering(test_df)

# Fill NaNs
for col in train_df.columns:
    if train_df[col].isnull().sum() > 0:
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
        test_df[col].fillna(median_val, inplace=True)

# 3. ENCODING
categorical_cols = [c for c in train_df.columns if train_df[c].dtype == 'object' and c != TARGET]
print(f"Encoding {len(categorical_cols)} categorical columns...")

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

X = train_df.drop([TARGET, 'id'], axis=1, errors='ignore')
y = train_df[TARGET]
X_test = test_df[[c for c in X.columns if c in test_df.columns]]
pos_weight = (len(y) - y.sum()) / y.sum()

# ====================================================
# SANITY CHECK (FIXED THRESHOLD)
# ====================================================
print(f"\n{'='*60}")
print(f"SANITY CHECK - Simple 3-Fold CV")
print(f"{'='*60}")

sanity_scores = []
skf_sanity = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf_sanity.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr)
    val_pred = lgb.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    sanity_scores.append(auc)
    print(f"Quick Fold {fold+1}: AUC = {auc:.6f}")

avg_sanity = np.mean(sanity_scores)
print(f"Quick Average AUC: {avg_sanity:.6f}")

# --- FIX: LOWERED THRESHOLD TO 0.70 ---
if avg_sanity < 0.70:
    print("\nâš ï¸� WARNING: Score too low. Stopping.")
else:
    print(f"\nâœ… Sanity check passed! (Score > 0.70)")
    print("Proceeding to Phase 1...")

    # ====================================================
    # PHASE 1: GENERATE PSEUDO-LABELS
    # ====================================================
    print(f"\n{'='*60}\nPHASE 1: Training Initial Ensemble\n{'='*60}")

    p1_test_preds = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/5...", end=" ")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Faster Learning Rate for Phase 1 to save time
        lgb = LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=64, 
                             scale_pos_weight=pos_weight, random_state=42, verbose=-1)
        lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100, verbose=False)])
        
        cat = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=7, 
                                 scale_pos_weight=pos_weight, verbose=0, random_seed=42, 
                                 allow_writing_files=False)
        cat.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100)
        
        # Simple blend
        p1_test_preds += (lgb.predict_proba(X_test)[:, 1] * 0.5 + cat.predict_proba(X_test)[:, 1] * 0.5) / 5
        
        # Check Fold Score
        fold_auc = roc_auc_score(y_val, (lgb.predict_proba(X_val)[:, 1] + cat.predict_proba(X_val)[:, 1])/2)
        print(f"AUC: {fold_auc:.6f}")
        gc.collect()

    # Identify Pseudo Labels
    high_conf_idx = np.where(p1_test_preds > PSEUDO_HIGH_THRESHOLD)[0]
    low_conf_idx = np.where(p1_test_preds < PSEUDO_LOW_THRESHOLD)[0]
    
    print(f"\nâœ… Found {len(high_conf_idx)} Positives and {len(low_conf_idx)} Negatives for augmentation.")

    # Create Augmented Dataset
    if len(high_conf_idx) > 0:
        pseudo_df_high = X_test.iloc[high_conf_idx].copy()
        pseudo_df_high[TARGET] = 1
        pseudo_df_low = X_test.iloc[low_conf_idx].copy()
        pseudo_df_low[TARGET] = 0
        
        X_pseudo = pd.concat([X, pseudo_df_high.drop(TARGET, axis=1), pseudo_df_low.drop(TARGET, axis=1)], axis=0).reset_index(drop=True)
        y_pseudo = pd.concat([y, pseudo_df_high[TARGET], pseudo_df_low[TARGET]], axis=0).reset_index(drop=True)
        print(f"ğŸ“ˆ Augmented Train Size: {len(X_pseudo)}")
    else:
        print("âš ï¸� Not enough confident predictions. Using original data.")
        X_pseudo, y_pseudo = X, y

    # ====================================================
    # PHASE 2: FINAL TRAINING
    # ====================================================
    print(f"\n{'='*60}\nPHASE 2: Final Training with Pseudo-Labels\n{'='*60}")
    
    final_test_preds = np.zeros(len(X_test))
    pos_weight_new = (len(y_pseudo) - y_pseudo.sum()) / y_pseudo.sum()

    for seed in SEEDS:
        print(f"\n>>> SEED {seed}")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_preds = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_pseudo, y_pseudo)):
            X_tr, X_val = X_pseudo.iloc[train_idx], X_pseudo.iloc[val_idx]
            y_tr, y_val = y_pseudo.iloc[train_idx], y_pseudo.iloc[val_idx]

            # 1. LGBM
            lgb = LGBMClassifier(n_estimators=3000, learning_rate=0.015, num_leaves=64, max_depth=8,
                                 scale_pos_weight=pos_weight_new, random_state=seed, verbose=-1, n_jobs=-1)
            lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(150, verbose=False)])
            
            # 2. CatBoost
            cat = CatBoostClassifier(iterations=3000, learning_rate=0.015, depth=7, scale_pos_weight=pos_weight_new, 
                                     verbose=0, random_seed=seed, allow_writing_files=False)
            cat.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=150)
            
            # 3. XGBoost
            xgb = XGBClassifier(n_estimators=3000, learning_rate=0.015, max_depth=8, scale_pos_weight=pos_weight_new, 
                                random_state=seed, n_jobs=-1, early_stopping_rounds=150, enable_categorical=False)
            xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
            
            # Rank Average Blend for Stability
            lgb_p = rankdata(lgb.predict_proba(X_test)[:, 1])
            cat_p = rankdata(cat.predict_proba(X_test)[:, 1])
            xgb_p = rankdata(xgb.predict_proba(X_test)[:, 1])
            
            # Weights: LGB/Cat usually better
            blend = (lgb_p * 0.4) + (cat_p * 0.4) + (xgb_p * 0.2)
            blend = (blend - blend.min()) / (blend.max() - blend.min())
            
            seed_preds += blend / N_SPLITS
            print(".", end="")
            gc.collect()
            
        final_test_preds += seed_preds / len(SEEDS)




    # Submission
    submission = sample_sub.copy()
    submission[TARGET] = (final_test_preds - final_test_preds.min()) / (final_test_preds.max() - final_test_preds.min())
    submission.to_csv("submission.csv", index=False)
    print(f"\n\nâœ… Submission Saved! (Stats: Mean={submission[TARGET].mean():.4f})")


# train_df.columns


# cat = [
#     'ethnicity',
#     'education_level',
#     'income_level',
#     'smoking_status',
#     'employment_status'
# ]

# train_dummies = pd.get_dummies(train_df[cat], drop_first=True)
# test_dummies  = pd.get_dummies(test_df[cat], drop_first=True)

# # Align columns (VERY IMPORTANT for Kaggle)
# train_dummies, test_dummies = train_dummies.align(
#     test_dummies, join='left', axis=1, fill_value=0
# )

# train_df = pd.concat([train_df, train_dummies], axis=1)
# test_df  = pd.concat([test_df, test_dummies], axis=1)

# train_df = train_df.drop(cat,axis = 1)
# test_df  = test_df.drop(cat,axis = 1)



# from  sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_score



# import seaborn as sns 
# import matplotlib.pyplot as plt 
# numerical_df = train_df.select_dtypes(include = np.number)
# corr_matrix = numerical_df.corr()
# plt.figure(figsize = (12,10))
# sns.heatmap(corr_matrix,annot = True, cmap = 'coolwarm',fmt = '.2f')
# plt.title("Correlation Heatmap of Nuemerical Features",fontsize = 16)
# plt.show()


# numerical_cols = train_df.select_dtypes(include = np.number).columns
# num_cols_to_plot = len(numerical_cols)
# num_rows = int(np.ceil(num_cols_to_plot/3))
# plt.figure(figsize = (15,5*num_rows))
# for i, col in enumerate(numerical_cols):
#     plt.subplot(num_rows, 3, i+1)
#     sns.boxplot(y = train_df[col])
#     plt.title(f"Box plot of{col}")
#     plt.ylabel('')
# plt.tight_layout()
# plt.show()


# X = train_df.drop('diagnosed_diabetes',axis = 1)
# y = train_df['diagnosed_diabetes']


# X_test_final = test_df.copy()


# for df in [X, test_df]:
#     df['chol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
#     df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
#     df['bmi_age'] = df['bmi'] * df['age']
#     df['activity_bmi'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)
#     df['waist_bmi'] = df['waist_to_hip_ratio'] * df['bmi']


# from lightgbm import LGBMClassifier, early_stopping, log_evaluation
# from sklearn.metrics import roc_auc_score

# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )


# # X_train,X_test,y_train,y_test = train_test_split(
# #     X,y,
# #     test_size = 0.2,
# #     random_state = 42,
# #     stratify  = y
# # )

# oof_preds = np.zeros(len(X))
# test_preds = np.zeros(len(X_test_final))
# auc_scores = []  
# for fold,(train_idx,test_idx)in enumerate (skf.split(X,y)):
#     X_train,X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train,y_test = y.iloc[train_idx], y.iloc[test_idx]

#     lgb_model = LGBMClassifier(
#     objective='binary',
#     metric='auc',
#     learning_rate=0.03,
#     n_estimators=1000,
#     num_leaves=48,              # reduced to prevent overfit
#     max_depth=-1,
#     min_child_samples=30,
#     subsample=0.85,
#     colsample_bytree=0.85,
#     reg_alpha=0.1,
#     reg_lambda=0.1,
#     scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum(),
#     random_state=42,
#     n_jobs=-1
# )

#     lgb_model.fit(
#         X_train, y_train,
#         eval_set = [(X_test,y_test)],
#         eval_metric = 'auc',
#         callbacks = [early_stopping(stopping_rounds = 100),log_evaluation(0)],
        
#     )
#     val_preds =  lgb_model.predict_proba(X_test)[:,1]
#     oof_preds[test_idx] = val_preds
#     auc = roc_auc_score(y_test,val_preds)
#     auc_scores.append(auc)
#     print(f"Fold {fold} AUC: {auc:.5f}")
#     X_test_aligned = X_test_final.copy()
#     for col in X_train.columns:
#         if col not in X_test_aligned.columns:
#             X_test_aligned[col] = 0  # fill missing columns with 0


#     X_test_aligned = X_test_aligned[X_train.columns]

#     # Accumulate test predictions
#     test_preds += lgb_model.predict_proba(X_test_aligned)[:, 1] / skf.n_splits

# print(f"\nMean CV AUC: {np.mean(auc_scores):.5f}")

    


# print("OOF AUC:", roc_auc_score(y, oof_preds))
# print("Fold AUCs:", auc_scores)
# print("Mean CV AUC:", np.mean(auc_scores))



# seeds = [42,202,777]
# final_test_preds = np.zeros(len(X_test_final))
# for seed in seeds:
#     lgb_model = LGBMClassifier(
#         objective = 'binary',
#         metric = 'auc',
#         learning_rate = 0.03,
#         n_estimators = 1000,
#         num_leaves = 48,
#         max_depth = -1,
#         min_child_samples = 10,
#         subsample = 0.85,
#         reg_alpha = 0.1,
#         reg_lambda = 0.1,
#         scale_pos_weight = (len(y)-y.sum())/y.sum(),
#         random_state = seed,
#         n_jobs = -1
#     )
#     lgb_model.fit(X,y)
#     final_test_preds +=lgb_model.predict_proba(X_test_aligned)[:,1]/len(seeds)


# X_train.columns = X_train.columns.str.replace(' ', '_')
# X_test.columns   = X_test.columns.str.replace(' ', '_')
# test_df.columns = test_df.columns.str.replace(' ', '_')



# assert list(X_train.columns) == list(X_test.columns)
# assert list(X_train.columns) == list(test_df.columns)



# lgb_model = LGBMClassifier(
#     objective= 'binary',
#     n_estimators=800,
#     learning_rate=0.03,
#     num_leaves=96,
#     max_depth=-1,
#     min_child_samples=30,
#     subsample=0.85,
#     colsample_bytree=0.85,
#     reg_alpha=0.1,
#     reg_lambda=0.1,
#     class_weight='balanced',
#     random_state=42,
#     n_jobs = -1
# )
# lgb_model.fit(X_train,y_train)
# y_pred = lgb_model.predict_proba(X_test)[:,1]
# ROC_AUC = roc_auc_score(y_test,y_pred)
# print(ROC_AUC)



# boost_models = {
#     "XGBoost": XGBClassifier(
#         n_estimators = 300,
#         learning_rate = 0.05,
#         max_depth = 4,
#         subsample = 0.8,
#         colsample_bytree = 0.8,
#         eval_metric = 'logloss',
#         random_state = 42,
#     ),
#     "CatBoost":CatBoostClassifier(
#         iterations = 300,
#         learning_rate = 0.05,
#         depth = 5,
#         verbose = 0,
#         random_state = 42
#     )
# }
# print("Boost Model performence")
# for name, model in boost_models.items():
#     pipe = Pipeline([
#         ("scaler",RobustScaler()),
#         ("model",model)
#     ])
#     pipe.fit(X_train,y_train)
#     y_pred = pipe.predict(X_test)
#     y_proba = pipe.predict_proba(X_test)[:,1]
#     acc = accuracy_score(y_test,y_pred)
#     f1 = f1_score(y_test,y_pred)
#     roc = roc_auc_score(y_test,y_proba)

#     print(f"=== {name} ===")
#     print(f"Accuracy : {acc:.4f}")
#     print(f"F1-score : {f1:.4f}")
#     print(f"ROC-AUC  : {roc:.4f}\n") 
    


# test_predictions = lgb_model.predict_proba(test_df)[:, 1]



# submission = sample_sub.copy()
# submission['diagnosed_diabetes'] = final_test_preds



# submission.shape



# submission.to_csv("/kaggle/working/submission.csv", index=False)



