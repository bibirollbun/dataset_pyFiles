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


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.calibration import CalibratedClassifierCV

import os
import gc



print("--- Starting Data Loading ---")

DATA_DIR = '/kaggle/input/playground-series-s5e7/'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

train_df, test_df = pd.DataFrame(), pd.DataFrame()

try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    if 'Personality' in test_df.columns:
        test_df.drop(columns='Personality', inplace=True)

except:
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        if 'Personality' in test_df.columns:
            test_df.drop(columns='Personality', inplace=True)
    except Exception as e:
        print(f"Data load error: {e}")
        exit()

if train_df.empty or test_df.empty:
    print("DataFrames are empty. Exiting.")
    exit()



# Save IDs
test_id = test_df['id']
train_id = train_df['id']

# Drop IDs
train_df.drop(columns='id', inplace=True)
test_df.drop(columns='id', inplace=True)

# Add constant feature
train_df['constant_zero_feature'] = 0
test_df['constant_zero_feature'] = 0



TARGET = 'Personality'

if TARGET not in train_df.columns:
    raise KeyError("Target not found in training data.")

y = train_df[TARGET]
X = train_df.drop(columns=[TARGET])
X_test = test_df.copy()

# Label Encoding
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print(f"Classes: {le_target.classes_}")



best_lr = 0.006358
best_params = {
    'max_depth': 8, 'subsample': 0.8854, 'colsample_bytree': 0.6000,
    'reg_lambda': 0.8295, 'reg_alpha': 5.5149, 'gamma': 0.0395, 'min_child_weight': 2
}
FIXED_N_ESTIMATORS = 5000

final_xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor',
    'random_state': 42, 'n_estimators': FIXED_N_ESTIMATORS,
    'learning_rate': best_lr, **best_params,
    'enable_categorical': False
}



N_SPLITS_FINAL_CV = 10
kf_final = StratifiedKFold(n_splits=N_SPLITS_FINAL_CV, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y_encoded))
test_preds_folds = []
fold_accuracies = []
feature_importance_df = None

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()



for fold, (train_idx, val_idx) in enumerate(kf_final.split(X, y_encoded)):
    print(f"--- Fold {fold+1} ---")
    
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
    X_test_fold = X_test.copy()
    
    # Impute numerical features
    imputer = IterativeImputer(max_iter=10, random_state=42, initial_strategy='median')
    X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = imputer.transform(X_val[numerical_cols])
    X_test_fold[numerical_cols] = imputer.transform(X_test_fold[numerical_cols])
    
    # Fill and encode categoricals
    for col in categorical_cols:
        for df in [X_train, X_val, X_test_fold]:
            df[col] = df[col].fillna('Missing')
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
    X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=False)
    X_test_fold = pd.get_dummies(X_test_fold, columns=categorical_cols, drop_first=False)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test_fold = X_test_fold.reindex(columns=X_train.columns, fill_value=0)
    
    # Calibration wrapper
    base_model = xgb.XGBClassifier(**final_xgb_params)
    calibrator_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=101)
    model_fold = CalibratedClassifierCV(base_model, method='isotonic', cv=calibrator_cv)
    
    model_fold.fit(X_train, y_train)
    
    fold_preds = model_fold.predict(X_val)
    oof_preds[val_idx] = fold_preds
    acc = accuracy_score(y_val, fold_preds)
    fold_accuracies.append(acc)
    print(f"Fold Accuracy: {acc:.4f}")
    
    test_preds_folds.append(model_fold.predict_proba(X_test_fold)[:, 1])
    
    if hasattr(model_fold, 'calibrated_classifiers_'):
        base_estimator = model_fold.calibrated_classifiers_[0].estimator
        fi_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': base_estimator.feature_importances_})
        if feature_importance_df is None:
            feature_importance_df = fi_df
        else:
            feature_importance_df = feature_importance_df.merge(fi_df, on='Feature', how='outer', suffixes=('', f'_fold{fold+1}'))
    
    del X_train, X_val, y_train, y_val, X_test_fold, model_fold, base_model, imputer
    gc.collect()



final_test_preds_proba = np.mean(test_preds_folds, axis=0)
final_test_preds_int = (final_test_preds_proba > 0.5).astype(int)

final_cv_acc = accuracy_score(y_encoded, oof_preds)
print(f"Final CV Accuracy: {final_cv_acc:.4f}")

# Save OOF
oof_df = pd.DataFrame({'id': train_id, 'oof_preds_class': oof_preds, 'target': y_encoded})
oof_df.to_csv('oof_predictions.csv', index=False)

# Prepare submission
submission_df = pd.DataFrame({
    'id': test_id,
    'Personality': le_target.inverse_transform(final_test_preds_int)
})
submission_df.to_csv('submission.csv', index=False)
submission_df.head()



feature_importance_df.fillna(0, inplace=True)
importance_cols = [col for col in feature_importance_df.columns if 'Importance' in col]
feature_importance_df['Mean_Importance'] = feature_importance_df[importance_cols].mean(axis=1)
feature_importance_df = feature_importance_df[['Feature', 'Mean_Importance']].sort_values(by='Mean_Importance', ascending=False)

print("Top 20 Important Features:")
print(feature_importance_df.head(20))


