import warnings
import optuna

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif

import xgboost as xgb
import lightgbm as lgb


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv").drop('id', axis=1)
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv").drop('id', axis=1)

X = train.drop(columns=['y']).copy()
y = train['y']

# Label encoding for categorical columns only
for col in X.select_dtypes(include='object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Calculate mutual information scores between features and target
mi = mutual_info_classif(X, y, discrete_features='auto')
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print("Mutual Information Scores (sorted by importance):")
print(mi_series)


def apply_feature_engineering(df):
   """
   Feature engineering function - apply identically to train and test
   """
   df_fe = df.copy()
   
   # Label encoding for categorical features
   label_encoders = {}
   for col in df_fe.select_dtypes(include='object').columns:
       if col != 'y':  # exclude target
           le = LabelEncoder()
           df_fe[col] = le.fit_transform(df_fe[col])
           label_encoders[col] = le
   
   # Interaction features
   df_fe['poutcome_duration'] = df_fe['poutcome'] * df_fe['duration']
   df_fe['housing_duration'] = df_fe['housing'] * df_fe['duration'] 
   df_fe['poutcome_housing'] = df_fe['poutcome'] * df_fe['housing']
   
   # Duration-based features
   df_fe['duration_per_campaign'] = df_fe['duration'] / (df_fe['campaign'] + 1)
   df_fe['duration_log'] = np.log1p(df_fe['duration'])
   df_fe['is_long_call'] = (df_fe['duration'] > df_fe['duration'].median()).astype(int)
   
   # Job-based statistics
   df_fe['job_duration_mean'] = df_fe.groupby('job')['duration'].transform('mean')
   df_fe['job_age_mean'] = df_fe.groupby('job')['age'].transform('mean')
   df_fe['job_balance_mean'] = df_fe.groupby('job')['balance'].transform('mean')
   
   # Education-based statistics
   df_fe['education_duration_mean'] = df_fe.groupby('education')['duration'].transform('mean')
   df_fe['education_balance_mean'] = df_fe.groupby('education')['balance'].transform('mean')
   
   # Marital-based statistics
   df_fe['marital_age_mean'] = df_fe.groupby('marital')['age'].transform('mean')
   
   # Financial situation features
   df_fe['total_loans'] = df_fe['housing'] + df_fe['loan']  # total loan count
   df_fe['financial_stress'] = (df_fe['default'] + df_fe['housing'] + df_fe['loan']).clip(0, 3)
   
   # Contact pattern features
   df_fe['contact_success_rate'] = df_fe['poutcome'] / (df_fe['previous'] + 1)  # previous success rate
   
   # Age-based features
   df_fe['age_balance_ratio'] = df_fe['balance'] / (df_fe['age'] + 1)
   
   print(f"Feature engineering completed: {df.shape} -> {df_fe.shape}")
   return df_fe


print(f"Original data size - Train: {train.shape}, Test: {test.shape}")
   
print("Applying feature engineering...")
train_fe = apply_feature_engineering(train)
test_fe = apply_feature_engineering(test)
   
print(f"Data size after FE - Train: {train_fe.shape}, Test: {test_fe.shape}")


X = train_fe.drop(columns=['y']).copy()
y = train_fe['y']

# Label encoding for categorical columns only
for col in X.select_dtypes(include='object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Calculate mutual information scores between features and target
mi = mutual_info_classif(X, y, discrete_features='auto')
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print("Mutual Information Scores (sorted by importance):")
print(mi_series)


warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

X = train_fe.drop(columns=["y"]).copy()
y = train_fe["y"]

# Encode categorical columns
for col in X.select_dtypes("object").columns:
   X[col] = LabelEncoder().fit_transform(X[col])

# Set K-fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True)


# -----------------------
# XGBoost GPU tuning
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500), 
        'max_bin': trial.suggest_int('max_bin', 256, 15000),          
        'max_depth': trial.suggest_int('max_depth', 3, 12),         
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),    
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),      
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),                         
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),                 
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),  
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),  
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'verbosity': -1
    }
    
    model = xgb.XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=1).mean()
    return score

# print("Tuning XGBoost with GPU...")
# xgb_study = optuna.create_study(direction='maximize', study_name='xgb_gpu')
# xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
# print(f"Best XGBoost score: {xgb_study.best_value:.5f}")
# print(f"Best XGBoost params: {xgb_study.best_params}")


# -----------------------
# LightGBM GPU tuning
def lgb_objective(trial):
   params = {
       'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
       'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
       'max_depth': trial.suggest_int('max_depth', 4, 12),
       'max_bin': trial.suggest_int('max_bin', 5000, 7000),
       'num_leaves': trial.suggest_int('num_leaves', 30, 150),
       'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
       'subsample': trial.suggest_float('subsample', 0.6, 1.0),
       'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
       'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
       'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
       'device': 'gpu',
       'objective': 'binary',
       'metric': 'auc',
       'verbosity': -1
   }
   
   model = lgb.LGBMClassifier(**params)
   score = cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=1).mean()
   return score

# print("\nTuning LightGBM with GPU...")
# lgb_study = optuna.create_study(direction='maximize', study_name='lgb_gpu')
# lgb_study.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
# print(f"Best LightGBM score: {lgb_study.best_value:.5f}")
# print(f"Best LightGBM params: {lgb_study.best_params}")


def run_te_cvens_blending(train_data, test_data, submission_path, save_path, n_splits=10, xgb_model=None, lgb_model=None):
    
    # Get train dataset
    X = train_data.drop(columns=["y"]).copy()
    y = train_data["y"]
    
    # Get categorical = discrete featurees
    cat_cols = X.select_dtypes("object").columns.tolist()
    
    # CV + Target Encoding + Model training and prediction
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    CV_result = []
    test_preds = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # Get test dataset
        test_fold = test_data.copy()
        
        # Target Encoding
        for col in cat_cols:
            encoding_dict = y_train.groupby(X_train[col]).mean().to_dict()
            global_mean = y_train.mean()
            
            for category in X_train[col].unique():
                n = (X_train[col] == category).sum()
                smooth_mean = (encoding_dict.get(category, global_mean) * n + global_mean * 5) / (n + 5)
                encoding_dict[category] = smooth_mean
                
            X_train[col] = X_train[col].map(encoding_dict).fillna(global_mean)
            X_valid[col] = X_valid[col].map(encoding_dict).fillna(global_mean)
            test_fold[col] = test_fold[col].map(encoding_dict).fillna(global_mean)
            
        # --- Models ---
        xgb_clf = xgb_model if xgb_model is not None else xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss"
        )
        lgb_clf = lgb_model if lgb_model is not None else lgb.LGBMClassifier(
            objective="binary",
            verbose=-1
        )
        
        # Fit models
        xgb_clf.fit(X_train, y_train)
        lgb_clf.fit(X_train, y_train)
        
        # Predict probabilities
        xgb_pred = xgb_clf.predict_proba(X_valid)[:, 1]
        lgb_pred = lgb_clf.predict_proba(X_valid)[:, 1]
        
        # Average prediction (simple blending)
        y_pred_proba = (xgb_pred + lgb_pred) / 2.0
        
        # Performance
        roc_auc = roc_auc_score(y_valid, y_pred_proba)
        CV_result.append({"fold": fold, "roc_auc": roc_auc})
        
        # Test set prediction (average of 2 models)
        test_pred_fold = (
            xgb_clf.predict_proba(test_fold)[:, 1] +
            lgb_clf.predict_proba(test_fold)[:, 1]
        ) / 2.0
        
        # Prediction for test dataset using trained model in each fold 
        test_preds.append(test_pred_fold)
        
    CV_result = pd.DataFrame(CV_result)
    print(CV_result)
    print(f"Mean CV Score: {CV_result['roc_auc'].mean():.5f}")
    
    # Final prediction = Average of predicted probability in each fold 
    y_test_pred_proba = np.mean(test_preds, axis=0)
    
    submission = pd.read_csv(submission_path)
    submission["y"] = y_test_pred_proba
    submission.to_csv(save_path, index=False)
    return 0


xgb_tuned = xgb.XGBClassifier(
   objective="binary:logistic",
   eval_metric="auc",
   n_estimators=1389,
   max_bin=10847,
   max_depth=8,
   learning_rate=0.05834291047382951,
   subsample=0.8745329018447362,
   colsample_bytree=0.6127843950281475,
   reg_alpha=0.003892147563829401,
   reg_lambda=0.003751824095672918,
   min_child_weight=4,
   gamma=0.1947382650293847,
   scale_pos_weight=1.3284756102938475,
   grow_policy="lossguide",
   tree_method="hist"
)

lgb_tuned = lgb.LGBMClassifier(
   objective="binary",
   metric="auc",
   n_estimators=1200,
   learning_rate=0.08247351690283475,
   num_leaves=110,
   max_depth=12,
   min_child_samples=10,
   subsample=0.7489273051647382,
   colsample_bytree=0.3092847561038572,
   reg_alpha=1.4738291047382951,
   reg_lambda=1.8947382650472839,
   max_bin=6000,
   verbosity=-1
)

run_te_cvens_blending(
    train_data=train_fe,
    test_data=test_fe,
    submission_path="/kaggle/input/playground-series-s5e8/sample_submission.csv",
    save_path="submission.csv",
    xgb_model=xgb_tuned,
    lgb_model=lgb_tuned
)

