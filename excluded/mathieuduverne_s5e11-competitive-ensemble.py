import pandas as pd
import numpy as np
import cupy as cp
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import os
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
target = df.columns.tolist()[-1]
print(f"Train Shape: {df.shape}")
df.head()


orig_path = "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv" 
orig_df = pd.read_csv(orig_path)
orig_df.head()



target_col = 'loan_paid_back'
    
# Identify common columns (excluding target and id)
common_cols = [c for c in df.columns if c in orig_df.columns and c != target_col and c != 'id']
    
print(f"Creating new features based on original data statistics for columns: {common_cols}")
    
# Global mean fallback
global_mean = orig_df[target_col].mean() if target_col in orig_df.columns else 0.5
        
new_cols_created = []
for c in common_cols:
    new_col_name = f"{c}_orig_mean"
            
    # Calculate mean target for each category/value in original data
    mapping = orig_df.groupby(c)[target_col].mean()
            
    # Map to Training Data
    df[new_col_name] = df[c].map(mapping)
    # Fill NaN (values present in Train but not Original) with global mean
    df[new_col_name] = df[new_col_name].fillna(global_mean)
            
    # Map to Test Data
    df_test[new_col_name] = df_test[c].map(mapping)
    df_test[new_col_name] = df_test[new_col_name].fillna(global_mean)
            
    new_cols_created.append(new_col_name)
            
print(f"Added {len(new_cols_created)} new features.")


def create_frequency_features(df, df_test):
    """
    Add frequency and binning features efficiently.
    """
    # Pre-allocate DataFrames for new features
    freq_features_train = pd.DataFrame(index=df.index)
    freq_features_test = pd.DataFrame(index=df_test.index)
    bin_features_train = pd.DataFrame(index=df.index)
    bin_features_test = pd.DataFrame(index=df_test.index)

    # Note: 'cols', 'num' must be defined globally before calling this
    for col in cols:
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # --- Quantile binning for numeric columns ---
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0

    # Concatenate all new features
    df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
    df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

    return df, df_test


def target_encoding(train, predict, n_splits=5):
    """
    Add K-Fold target mean encoded features.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features_train = pd.DataFrame(index=train.index)
    mean_features_test = pd.DataFrame(index=predict.index)

    # Note: 'cols' must be defined globally
    for col in cols:
        # --- K-Fold Target Mean Encoding ---
        mean_encoded = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train):
            tr_fold = train.iloc[tr_idx]
            val_fold = train.iloc[val_idx]
            mean_map = tr_fold.groupby(col)[target].mean()
            mean_encoded[val_idx] = val_fold[col].map(mean_map)

        mean_features_train[f'mean_{col}'] = mean_encoded

        # --- Apply global mean mapping to prediction/test data ---
        global_mean = train.groupby(col)[target].mean()
        mean_features_test[f'mean_{col}'] = predict[col].map(global_mean)

    # --- Concatenate ---
    train = pd.concat([train, mean_features_train], axis=1)
    predict = pd.concat([predict, mean_features_test], axis=1)

    return train, predict


# Rounding the values
for c in ['annual_income', 'loan_amount']:
    if c in df.columns:
        for s, l in {'1s': 0, '10s': -1}.items():
            for g in [df, df_test]:
                g[f'{c}_ROUND_{s}'] = g[c].round(l).astype(int)

# Specific feature engineering
for gf in [df, df_test]:
    if 'grade_subgrade' in gf.columns:
        gf['subgrade'] = gf['grade_subgrade'].str[1:].astype(int)
        gf['grade'] = gf['grade_subgrade'].str[0]
    
    if all(x in gf.columns for x in ['loan_amount', 'interest_rate', 'annual_income']):
        gf['total_debt_burden'] = (gf['loan_amount'] * gf['interest_rate'] / 100) / (gf['annual_income'] + 1)


# Define columns for FE application
# Exclude target and ID
cols = df.drop(columns=[target,"id"], errors='ignore').columns.tolist()
cat = [c for c in cols if df[c].dtype in ["object","category"]]
num = [c for c in cols if df[c].dtype not in ["object","category","bool"]]

# Creating new features based on the frequency of numerical features
# Note: target_encoding uses 'cols' global variable
df, df_test = target_encoding(df, df_test, 10)
df, df_test = create_frequency_features(df, df_test)

# Preparing categorical features
df[cat] = df[cat].astype("category")
df_test[cat] = df_test[cat].astype("category")


remove = [
    'annual_income_ROUND_10s_bin10','annual_income_ROUND_1s_bin10','annual_income_ROUND_1s_bin15','annual_income_ROUND_1s_bin5',
    'annual_income_bin10','annual_income_bin5','credit_score_bin10','credit_score_bin5','debt_to_income_ratio_bin15','debt_to_income_ratio_bin5',
    'education_level_freq','gender_freq','interest_rate_bin10','interest_rate_bin5','loan_amount_ROUND_10s_bin5','loan_amount_ROUND_1s_bin10',
    'loan_amount_ROUND_1s_bin15','loan_amount_ROUND_1s_bin5','loan_amount_bin10','loan_amount_bin15','loan_amount_bin5','marital_status_freq',
    'subgrade','subgrade_bin10','subgrade_bin15','subgrade_bin5','subgrade_freq',"mean_total_debt_burden"
]

# Drop if they exist
remove_actual = [c for c in remove if c in df.columns]
df = df.drop(columns=remove_actual + ["id"], errors='ignore')
df_test = df_test.drop(columns=remove_actual, errors='ignore')


print(f"Number of columns {len(df.columns.tolist())}")
print(df.columns.tolist())
missing = df.isnull().sum()[lambda x: x>0]
print("Missing values:\n", missing)


N_FOLDS = 7
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

X = df.drop(columns=[target, "id"], errors='ignore')
y = df[target]
X_test = df_test.drop(columns=["id"], errors='ignore')

cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['category', 'object']).columns.tolist()

print(f"Features: {X.shape[1]}, Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}")

ensemble_results = {
    'oof': pd.DataFrame(),
    'test': pd.DataFrame()
}
ensemble_results['oof']['target'] = y 


# --- Configuration CatBoost ---
cb_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.05,
    'iterations': 3000,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_strength': 1,
    'bagging_temperature': 1,
    'allow_writing_files': False,
    'verbose': 0,
    'random_seed': SEED,
    'cat_features': cat_cols,
    'task_type': 'GPU'
}

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("\nTraining CatBoost...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(**cb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        verbose=False
    )
    
    oof_preds[val_idx] = model.predict_proba(X_valid)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"Fold {fold+1} AUC: {model.best_score_['validation']['AUC']:.5f}")

print(f"-> CatBoost Global OOF AUC: {roc_auc_score(y, oof_preds):.6f}")
ensemble_results['oof']['catboost'] = oof_preds
ensemble_results['test']['catboost'] = test_preds


lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'extra_trees': True,    
    'learning_rate': 0.03,
    'num_leaves': 64,
    'feature_fraction': 0.4, 
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': SEED
}

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("\nTraining LightGBM (Extra Trees)...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    
    X_tr_c = X_train.copy()
    X_val_c = X_valid.copy()
    for c in cat_cols:
        X_tr_c[c] = X_tr_c[c].astype('category')
        X_val_c[c] = X_val_c[c].astype('category')
        
    model = lgb.LGBMClassifier(**lgb_params, n_estimators=5000)
    
    model.fit(
        X_tr_c, y_train,
        eval_set=[(X_val_c, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    oof_preds[val_idx] = model.predict_proba(X_val_c)[:, 1]
    X_test_c = X_test.copy()
    for c in cat_cols: X_test_c[c] = X_test_c[c].astype('category')
    test_preds += model.predict_proba(X_test_c)[:, 1] / N_FOLDS

print(f"-> LightGBM XT Global OOF AUC: {roc_auc_score(y, oof_preds):.6f}")
ensemble_results['oof']['lgbm_xt'] = oof_preds
ensemble_results['test']['lgbm_xt'] = test_preds


# --- Configuration MLP (Sklearn) ---

mlp_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'), 
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), # Profondeur moyenne
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED
    )
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("\nTraining MLP (Neural Network)...")
X_num = X[num_cols]
X_test_num = X_test[num_cols]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_num, y)):
    X_train, y_train = X_num.iloc[train_idx], y.iloc[train_idx]
    X_valid = X_num.iloc[val_idx]
    
    mlp_pipeline.fit(X_train, y_train)
    
    oof_preds[val_idx] = mlp_pipeline.predict_proba(X_valid)[:, 1]
    test_preds += mlp_pipeline.predict_proba(X_test_num)[:, 1] / N_FOLDS

print(f"-> MLP Global OOF AUC: {roc_auc_score(y, oof_preds):.6f}")
ensemble_results['oof']['mlp'] = oof_preds
ensemble_results['test']['mlp'] = test_preds


xgb_params = {
    'tree_method': 'hist', 
    'device': 'cuda', 
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': SEED,
    'min_child_weight': 89,
    "max_leaves": 4,
    "reg_alpha": 3.2,
    "reg_lambda": 5,
    "eta": 0.1,
}

oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    
    dtrain_fold = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid_fold = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain_fold,
        num_boost_round=20000,
        evals=[(dvalid_fold, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False 
    )
    
    oof_preds = model.predict(dvalid_fold)
    oof_xgb[val_idx] = oof_preds
    
    dtest_fold = xgb.DMatrix(X_test, enable_categorical=True) 
    test_xgb += model.predict(dtest_fold) / N_FOLDS
    
    fold_score = roc_auc_score(y_valid, oof_preds)
    print(f"Fold {fold+1} AUC: {fold_score:.5f} (Best Iter: {model.best_iteration})")

global_auc = roc_auc_score(y, oof_xgb)
print(f"\n-> XGBoost Global OOF AUC: {global_auc:.6f}")

ensemble_results['oof']['xgboost'] = oof_xgb
ensemble_results['test']['xgboost'] = test_xgb


# Extract OOF predictions
oof_df = ensemble_results['oof'].drop(columns=['target'])
corr = oof_df.corr()

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".4f")
plt.title("Model Correlation Matrix")
plt.show()


from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

# Prepare data
oof_df = ensemble_results['oof']
target = oof_df['target']
models = [col for col in oof_df.columns if col != 'target']
X_oof = oof_df[models].values
X_test = ensemble_results['test'][models].values

# Objective Function: Minimize Negative AUC
def auc_func(weights):
    # Normalize weights so they sum to 1 (optional but recommended for stability)
    norm_weights = np.array(weights) / np.sum(weights)
    
    # Calculate weighted prediction
    final_pred = np.tensordot(X_oof, norm_weights, axes=1)
    
    # Return negative AUC (minimize negative = maximize positive)
    return -roc_auc_score(target, final_pred)

# Initial guess: Equal weights
init_weights = [1/len(models)] * len(models)

# Constraints: Weights must be between 0 and 1
bounds = [(0, 1) for _ in models]

# Run Optimization
res = minimize(
    auc_func, 
    init_weights, 
    method='Nelder-Mead', 
    bounds=bounds, 
    tol=1e-6
)

# Extract optimized weights
opt_weights = res.x / np.sum(res.x)

print("Optimized Weights:")
for model, weight in zip(models, opt_weights):
    print(f"{model}: {weight:.4f}")

# Calculate Final Scores
final_oof_pred = np.tensordot(X_oof, opt_weights, axes=1)
final_score = roc_auc_score(target, final_oof_pred)
print(f"\nFinal Ensemble OOF AUC: {final_score:.6f}")


# Apply weights to test predictions
final_test_pred = np.tensordot(X_test, opt_weights, axes=1)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': df_test['id'], # Ensure you have the ID column from your test load
    'loan_paid_back': final_test_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission saved successfully.")

