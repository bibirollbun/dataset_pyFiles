import os
import warnings
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

# Suppress warnings and unnecessary logs
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTGBM_VERBOSE"] = "0"
os.environ["LGBM_VERBOSITY"] = "0"

logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)

# Pandas display settings
pd.set_option("display.max_columns", None)

# Remove all files in /kaggle/working (use with caution!)
!rm -rf /kaggle/working/*



train_df = pd.read_csv("/kaggle/input/wine-quality-prediction-challenge/train.csv")
test_df = pd.read_csv("/kaggle/input/wine-quality-prediction-challenge/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/wine-quality-prediction-challenge/sample_submission.csv")

train_df = train_df.drop('id', axis=1)
test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)

def create_features(df):
    df_copy = df.copy()
    df_copy['total_acidity'] = df_copy['fixed acidity'] + df_copy['volatile acidity']
    df_copy['acid_balance'] = df_copy['total_acidity'] / (df_copy['pH'] + 1e-6)
    df_copy['free_sulfur_ratio'] = df_copy['free sulfur dioxide'] / (df_copy['total sulfur dioxide'] + 1e-6)
    return df_copy

train_featured_df = create_features(train_df)
test_featured_df = create_features(test_df)

X = train_featured_df.drop('quality', axis=1)
y = train_featured_df['quality']
X_test = test_featured_df

params = {
    'objective': 'multiclass',
    'num_class': len(y.unique()),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 3000, #3000
    'learning_rate': 0.05,
    'num_leaves': 20,
    'max_depth': 5,  #     'max_depth': 5,learning_rate 0.05 ls==>0.37582
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'device': 'gpu',               # Enable GPU
    'gpu_platform_id': 0,          # Platform index (usually 0)
    'gpu_device_id': 0           
}

N_SPLITS = 20
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,eval_set=[(X_val, y_val)],callbacks=[lgb.early_stopping(300, verbose=False)])

    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test) / N_SPLITS

kappa_score = cohen_kappa_score(y, oof_preds, weights='quadratic')
print(f"Kappa Score: {kappa_score}")

final_predictions = np.round(test_preds).astype(int)
min_quality = y.min()
max_quality = y.max()
final_predictions = np.clip(final_predictions, min_quality, max_quality)

submission_df = pd.DataFrame({'id': test_ids, 'quality': final_predictions})
submission_df.to_csv('cv_submission.csv', index=False)
print("submission SAVED")




