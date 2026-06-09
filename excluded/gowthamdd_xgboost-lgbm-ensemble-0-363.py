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


import os
import gc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation


# MAP@3 Scoring Functions
def apk(actual, predicted, k=3):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Load Data
df_train_original = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_train_additional = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

TARGET = 'Fertilizer Name'
ID_COL = 'id'

X_original = df_train_original.drop([ID_COL, TARGET], axis=1).copy()
y_original = df_train_original[TARGET].copy()

X_additional = df_train_additional.drop(TARGET, axis=1).copy()
y_additional = df_train_additional[TARGET].copy()

X_test = df_test.drop(ID_COL, axis=1).copy()
test_ids = df_test[ID_COL].copy()

# Feature Engineering
def apply_feature_engineering(df):
    df = df.copy()
    df['Temp_Humidity_Interaction'] = df['Temparature'] * df['Humidity']
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'].replace(0, 1e-6))
    df['K_P_Ratio'] = df['Potassium'] / (df['Phosphorous'].replace(0, 1e-6))
    df['Soil_Crop_Combination'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
    for col in ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']:
        df[f'{col}_Binned'] = df[col].astype(str)
    return df

X_original = apply_feature_engineering(X_original)
X_additional = apply_feature_engineering(X_additional)
X_test = apply_feature_engineering(X_test)

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

X_original_poly = poly.fit_transform(X_original[num_cols])
X_additional_poly = poly.transform(X_additional[num_cols])
X_test_poly = poly.transform(X_test[num_cols])

poly_cols = poly.get_feature_names_out(num_cols)
X_original = pd.concat([X_original.drop(columns=num_cols).reset_index(drop=True), pd.DataFrame(X_original_poly, columns=poly_cols)], axis=1)
X_additional = pd.concat([X_additional.drop(columns=num_cols).reset_index(drop=True), pd.DataFrame(X_additional_poly, columns=poly_cols)], axis=1)
X_test = pd.concat([X_test.drop(columns=num_cols).reset_index(drop=True), pd.DataFrame(X_test_poly, columns=poly_cols)], axis=1)

# Categorical Encoding
categorical_cols = ['Soil Type', 'Crop Type', 'Soil_Crop_Combination'] + [f'{col}_Binned' for col in num_cols]

for col in categorical_cols:
    full_col = pd.concat([X_original[col], X_additional[col], X_test[col]], axis=0).astype(str)
    le = LabelEncoder().fit(full_col)
    X_original[col] = le.transform(X_original[col].astype(str))
    X_additional[col] = le.transform(X_additional[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Encode target
target_le = LabelEncoder()
y_all = pd.concat([y_original, y_additional])
target_le.fit(y_all)
y_original_encoded = target_le.transform(y_original)
y_additional_encoded = target_le.transform(y_additional)
classes = target_le.classes_

# Train Both Models
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds_xgb = np.zeros((len(X_original), len(classes)))
oof_preds_lgbm = np.zeros((len(X_original), len(classes)))
test_preds_xgb = []
test_preds_lgbm = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_original, y_original_encoded)):
    print(f"Fold {fold + 1}")
    
    X_train = pd.concat([X_original.iloc[train_idx], X_additional], ignore_index=True)
    y_train = np.concatenate([y_original_encoded[train_idx], y_additional_encoded])
    X_val = X_original.iloc[val_idx]
    y_val = y_original_encoded[val_idx]

    # XGBoost
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(classes),
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)
    test_preds_xgb.append(xgb_model.predict_proba(X_test))

    # LightGBM
    lgbm_model = LGBMClassifier(
        objective='multiclass',
        n_estimators=1000,
        learning_rate=0.05,
        num_class=len(classes),
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42
    )
    lgbm_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(
            stopping_rounds=50,
            first_metric_only=True,
            verbose=True,
            min_delta=0.0
        ),
        log_evaluation(period=100)
    ]
)

    oof_preds_lgbm[val_idx] = lgbm_model.predict_proba(X_val)
    test_preds_lgbm.append(lgbm_model.predict_proba(X_test))

# MAP@3
avg_preds = 0.8 * oof_preds_xgb + 0.2 * oof_preds_lgbm
y_true_map = [[label] for label in y_original]
oof_pred_labels = [[classes[i] for i in np.argsort(row)[-3:][::-1]] for row in avg_oof_preds]
print(f"\nOOF MAP@3: {mapk(y_true_map, oof_pred_labels):.5f}")

# Final Predictions
avg_test_preds = (np.mean(test_preds_xgb, axis=0) + np.mean(test_preds_lgbm, axis=0)) / 2
submission_labels = [" ".join([classes[i] for i in np.argsort(row)[-3:][::-1]]) for row in avg_test_preds]

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': submission_labels
})

submission_df.to_csv('submission_combined_xgb_lgbm.csv', index=False)
print("\nSubmission file generated.")




# Train XGBoost Only
oof_preds_xgb = np.zeros((len(X_original), len(classes)))
test_preds_xgb = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_original, y_original_encoded)):
    print(f"Fold {fold + 1}")
    
    X_train = pd.concat([X_original.iloc[train_idx], X_additional], ignore_index=True)
    y_train = np.concatenate([y_original_encoded[train_idx], y_additional_encoded])
    X_val = X_original.iloc[val_idx]
    y_val = y_original_encoded[val_idx]

    # XGBoost
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(classes),
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)
    test_preds_xgb.append(xgb_model.predict_proba(X_test))

# MAP@3
y_true_map = [[label] for label in y_original]
oof_pred_labels = [[classes[i] for i in np.argsort(row)[-3:][::-1]] for row in oof_preds_xgb]
print(f"\nOOF MAP@3: {mapk(y_true_map, oof_pred_labels):.5f}")

# Final Predictions
avg_test_preds = np.mean(test_preds_xgb, axis=0)
submission_labels = [" ".join([classes[i] for i in np.argsort(row)[-3:][::-1]]) for row in avg_test_preds]

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': submission_labels
})

submission_df.to_csv('submission_xgb_only.csv', index=False)
print("\nSubmission file (XGBoost only) generated.")



%%time

import os
import gc
import warnings
import random as python_random
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  # Optional ensemble

warnings.filterwarnings("ignore")
np.random.seed(42)
python_random.seed(42)

# Load Data
df_train_original = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_train_additional = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print("Original Train:", df_train_original.shape)
print("Additional Train:", df_train_additional.shape)
print("Test:", df_test.shape)

TARGET = 'Fertilizer Name'
ID_COL = 'id'
original_numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Preprocessing
X_original = df_train_original.drop([ID_COL, TARGET], axis=1)
y_original = df_train_original[TARGET]

X_additional = df_train_additional.drop(TARGET, axis=1)
y_additional = df_train_additional[TARGET]

X_test = df_test.drop(ID_COL, axis=1)
test_ids = df_test[ID_COL]

def apply_feature_engineering(df):
    df = df.copy()
    df['Temp_Humidity_Interaction'] = df['Temparature'] * df['Humidity']
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'].replace(0, 1e-6))
    df['K_P_Ratio'] = df['Potassium'] / (df['Phosphorous'].replace(0, 1e-6))
    df['Soil_Crop_Combination'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
    for col in original_numerical_cols:
        df[f'{col}_Binned'] = df[col].astype(str)
    return df

X_original_fe = apply_feature_engineering(X_original)
X_additional_fe = apply_feature_engineering(X_additional)
X_test_fe = apply_feature_engineering(X_test)

numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
                      'Temp_Humidity_Interaction', 'N_P_Ratio', 'K_P_Ratio']
categorical_features = ['Soil Type', 'Crop Type', 'Soil_Crop_Combination'] + [f'{col}_Binned' for col in original_numerical_cols]

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_original_poly = poly.fit_transform(X_original_fe[original_numerical_cols])
X_additional_poly = poly.transform(X_additional_fe[original_numerical_cols])
X_test_poly = poly.transform(X_test_fe[original_numerical_cols])
poly_feature_names = poly.get_feature_names_out(original_numerical_cols)

for df, poly_data in zip([X_original_fe, X_additional_fe, X_test_fe], [X_original_poly, X_additional_poly, X_test_poly]):
    df.drop(columns=original_numerical_cols, inplace=True)
    df[poly_feature_names] = poly_data

numerical_features += list(poly_feature_names)
numerical_features = list(dict.fromkeys([f for f in numerical_features if f not in categorical_features]))
all_features_ordered = numerical_features + categorical_features

X_original_fe = X_original_fe[all_features_ordered]
X_additional_fe = X_additional_fe[all_features_ordered]
X_test_fe = X_test_fe[all_features_ordered]

# Convert categoricals to 'category'
all_categories_union = {col: pd.concat([X_original_fe[col], X_additional_fe[col], X_test_fe[col]], axis=0).astype(str).unique()
                        for col in categorical_features}
for df in [X_original_fe, X_additional_fe, X_test_fe]:
    for col in categorical_features:
        df[col] = pd.Categorical(df[col], categories=all_categories_union[col])

# Encode target
le = LabelEncoder()
y_encoded_all = le.fit_transform(pd.concat([y_original, y_additional]))
y_original_encoded = le.transform(y_original)
y_additional_encoded = le.transform(y_additional)
classes = le.classes_

# CV Setup
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_original_fe), len(classes)))
test_preds_xgb = []
test_preds_lgb = []

# XGBoost Parameters
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': len(classes),
    'eval_metric': 'mlogloss',
    'eta': 0.01,
    'max_depth': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'n_estimators': 4000,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'n_jobs': -1,
    'enable_categorical': True,
    'early_stopping_rounds': 50,
    'verbose': 0
}

# LightGBM Parameters (optional)
use_lgb = True
lgb_params = {
    'objective': 'multiclass',
    'num_class': len(classes),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'max_depth': 10,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'n_estimators': 2000,
    'random_state': 42,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_original_fe, y_original_encoded)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    X_train = pd.concat([X_original_fe.iloc[train_idx], X_additional_fe])
    y_train = np.concatenate([y_original_encoded[train_idx], y_additional_encoded])
    X_val = X_original_fe.iloc[val_idx]
    y_val = y_original_encoded[val_idx]

    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds[val_idx] = model_xgb.predict_proba(X_val)
    test_preds_xgb.append(model_xgb.predict_proba(X_test_fe))

    if use_lgb:
        model_lgb = LGBMClassifier(**lgb_params)
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
        test_preds_lgb.append(model_lgb.predict_proba(X_test_fe))

    del X_train, y_train, X_val, y_val, model_xgb
    if use_lgb:
        del model_lgb
    gc.collect()

# MAP@3
def apk(actual, predicted, k=3):
    if not actual: return 0.0
    predicted = predicted[:k]
    score = sum([1.0 / (i + 1.0) for i, p in enumerate(predicted) if p in actual])
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

y_true_map = [[label] for label in y_original]
oof_pred_labels = [[classes[i] for i in np.argsort(row)[-3:][::-1]] for row in oof_preds]
map3_score = mapk(y_true_map, oof_pred_labels, k=3)
print(f"\nOOF MAP@3: {map3_score:.5f}")

# Final Ensemble Prediction
avg_test_preds = np.mean(test_preds_xgb, axis=0)
if use_lgb:
    avg_test_preds = (avg_test_preds + np.mean(test_preds_lgb, axis=0)) / 2

submission_labels = [" ".join([classes[i] for i in np.argsort(row)[-3:][::-1]]) for row in avg_test_preds]
submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': submission_labels
})

submission_df.to_csv('submission_ensemble.csv', index=False)
print("\nSubmission file saved: submission_ensemble.csv")
print(submission_df.head())



# Final Ensemble Prediction - 80:20 XGB:LGB
avg_test_preds_xgb = np.mean(test_preds_xgb, axis=0)
avg_test_preds_lgb = np.mean(test_preds_lgb, axis=0) if use_lgb else 0

if use_lgb:
    avg_test_preds = 0.99 * avg_test_preds_xgb + 0.01 * avg_test_preds_lgb
else:
    avg_test_preds = avg_test_preds_xgb



submission_labels = [" ".join([classes[i] for i in np.argsort(row)[-3:][::-1]]) for row in avg_test_preds]
submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': submission_labels
})

submission_df.to_csv('submission_ensemble.csv', index=False)
print("\nSubmission file saved: submission_ensemble1.csv")
print(submission_df.head())



%%time

import os
import gc
import warnings
import random as python_random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)
python_random.seed(42)

# --- Load Data ---
df_train_original = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_train_additional = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print("Original Train:", df_train_original.shape)
print("Additional Train:", df_train_additional.shape)
print("Test:", df_test.shape)

TARGET = 'Fertilizer Name'
ID_COL = 'id'
original_numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# --- Feature Engineering (From 1st Approach) ---
def apply_feature_engineering(df):
    df = df.copy()
    df['Temp_Humidity_Interaction'] = df['Temparature'] * df['Humidity']
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'].replace(0, 1e-6))
    df['K_P_Ratio'] = df['Potassium'] / (df['Phosphorous'].replace(0, 1e-6))
    df['Soil_Crop_Combination'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
    for col in original_numerical_cols:
        df[f'{col}_Binned'] = df[col].astype(str)
    return df

X_original = df_train_original.drop([ID_COL, TARGET], axis=1)
y_original = df_train_original[TARGET]

X_additional = df_train_additional.drop(TARGET, axis=1)
y_additional = df_train_additional[TARGET]

X_test = df_test.drop(ID_COL, axis=1)
test_ids = df_test[ID_COL]

# Apply FE to all datasets
X_original_fe = apply_feature_engineering(X_original)
X_additional_fe = apply_feature_engineering(X_additional)
X_test_fe = apply_feature_engineering(X_test)

# Polynomial Features (From 1st Approach)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_original_poly = poly.fit_transform(X_original_fe[original_numerical_cols])
X_additional_poly = poly.transform(X_additional_fe[original_numerical_cols])
X_test_poly = poly.transform(X_test_fe[original_numerical_cols])
poly_feature_names = poly.get_feature_names_out(original_numerical_cols)

# Drop original numerical columns and add polynomial features
for df, poly_data in zip([X_original_fe, X_additional_fe, X_test_fe], [X_original_poly, X_additional_poly, X_test_poly]):
    df.drop(columns=original_numerical_cols, inplace=True)
    df[poly_feature_names] = poly_data

# Define final feature lists
numerical_features = list(poly_feature_names) + ['Temp_Humidity_Interaction', 'N_P_Ratio', 'K_P_Ratio']
categorical_features = ['Soil Type', 'Crop Type', 'Soil_Crop_Combination'] + [f'{col}_Binned' for col in original_numerical_cols]
all_features_ordered = numerical_features + categorical_features

X_original_fe = X_original_fe[all_features_ordered]
X_additional_fe = X_additional_fe[all_features_ordered]
X_test_fe = X_test_fe[all_features_ordered]

# --- Categorical Encoding (From 1st Approach) ---
all_categories_union = {col: pd.concat([X_original_fe[col], X_additional_fe[col], X_test_fe[col]], axis=0).astype(str).unique()
                        for col in categorical_features}
for df in [X_original_fe, X_additional_fe, X_test_fe]:
    for col in categorical_features:
        df[col] = pd.Categorical(df[col], categories=all_categories_union[col])

# --- Target Encoding ---
le = LabelEncoder()
y_encoded_all = le.fit_transform(pd.concat([y_original, y_additional]))
y_original_encoded = le.transform(y_original)
y_additional_encoded = le.transform(y_additional)
classes = le.classes_

# --- Cross-Validation & Ensemble ---
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_preds_xgb = np.zeros((len(X_original_fe), len(classes)))
oof_preds_lgb = np.zeros((len(X_original_fe), len(classes)))
test_preds_xgb = []
test_preds_lgb = []

# XGBoost Parameters (From 1st Approach)
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': len(classes),
    'eval_metric': 'mlogloss',
    'eta': 0.01,
    'max_depth': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'n_estimators': 10000,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'n_jobs': -1,
    'enable_categorical': True,
    'early_stopping_rounds': 50,
    'verbose': 0
}

# LightGBM Parameters (Tuned)
lgb_params = {
    'objective': 'multiclass',
    'num_class': len(classes),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'max_depth': 10,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'n_estimators': 2000,
    'random_state': 42,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_original_fe, y_original_encoded)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    
    # Combine original + additional data
    X_train = pd.concat([X_original_fe.iloc[train_idx], X_additional_fe])
    y_train = np.concatenate([y_original_encoded[train_idx], y_additional_encoded])
    X_val = X_original_fe.iloc[val_idx]
    y_val = y_original_encoded[val_idx]

    # --- XGBoost ---
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_val)
    test_preds_xgb.append(model_xgb.predict_proba(X_test_fe))

    # --- LightGBM ---
    model_lgb = LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_val)
    test_preds_lgb.append(model_lgb.predict_proba(X_test_fe))

    del X_train, y_train, X_val, y_val, model_xgb, model_lgb
    gc.collect()



y_true_map = [[label] for label in y_original]

# --- Save Predictions ---
print("\nSaving predictions to disk...")
np.save("oof_preds_xgb.npy", oof_preds_xgb)
np.save("oof_preds_lgb.npy", oof_preds_lgb)
np.save("test_preds_xgb.npy", test_preds_xgb)
np.save("test_preds_lgb.npy", test_preds_lgb)
np.save("y_true_map.npy", y_true_map)
np.save("classes.npy", classes)
with open('test_ids.pkl', 'wb') as f:
    pickle.dump(test_ids, f)
print("Predictions saved successfully!")

# --- MAP@3 Calculation ---
def apk(actual, predicted, k=3):
    if not actual: return 0.0
    predicted = predicted[:k]
    score = sum([1.0 / (i + 1.0) for i, p in enumerate(predicted) if p in actual])
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# XGB-only OOF MAP@3
oof_pred_labels_xgb = [[classes[i] for i in np.argsort(row)[-3:][::-1]] for row in oof_preds_xgb]
map3_score_xgb = mapk(y_true_map, oof_pred_labels_xgb, k=3)
print(f"\nXGB-only OOF MAP@3: {map3_score_xgb:.5f}")

# LGB-only OOF MAP@3
oof_pred_labels_lgb = [[classes[i] for i in np.argsort(row)[-3:][::-1]] for row in oof_preds_lgb]
map3_score_lgb = mapk(y_true_map, oof_pred_labels_lgb, k=3)
print(f"LGB-only OOF MAP@3: {map3_score_lgb:.5f}")

# Ensemble OOF MAP@3 (80% XGB + 20% LGB)
oof_preds_ensemble = 0.8 * oof_preds_xgb + 0.2 * oof_preds_lgb
oof_pred_labels_ensemble = [[classes[i] for i in np.argsort(row)[-3:][::-1]] for row in oof_preds_ensemble]
map3_score_ensemble = mapk(y_true_map, oof_pred_labels_ensemble, k=3)
print(f"Ensemble OOF MAP@3: {map3_score_ensemble:.5f}")

# # Generate submission with initial ratio (90% XGB + 10% LGB)
# avg_test_preds_xgb = np.mean(test_preds_xgb, axis=0)
# avg_test_preds_lgb = np.mean(test_preds_lgb, axis=0)
# avg_test_preds = 0.9 * avg_test_preds_xgb + 0.1 * avg_test_preds_lgb

submission_labels = [" ".join([classes[i] for i in np.argsort(row)[-3:][::-1]]) for row in avg_test_preds]
submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': submission_labels
})

submission_df.to_csv('submission_ensemble_final.csv', index=False)
print("\nInitial submission saved (90% XGB + 10% LGB)!")
print(submission_df.head())




# ==============================================
# SEPARATE RECALL BLOCK FOR TESTING RATIOS
# ==============================================

# --- Load Saved Predictions ---
import pickle
oof_preds_xgb = np.load("oof_preds_xgb.npy")
oof_preds_lgb = np.load("oof_preds_lgb.npy")
test_preds_xgb = np.load("test_preds_xgb.npy", allow_pickle=True)
test_preds_lgb = np.load("test_preds_lgb.npy", allow_pickle=True)
y_true_map = np.load("y_true_map.npy", allow_pickle=True)
classes = np.load("classes.npy", allow_pickle=True)
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

# --- Test Different Ensemble Ratios ---
best_score = 0
best_ratio = None

for xgb_weight in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    lgb_weight = 1 - xgb_weight
    oof_preds_ensemble = xgb_weight * oof_preds_xgb + lgb_weight * oof_preds_lgb
    score = mapk(y_true_map, [[classes[i] for i in np.argsort(row)[-3:][::-1]] for row in oof_preds_ensemble], k=3)
    
    if score > best_score:
        best_score = score
        best_ratio = (xgb_weight, lgb_weight)
    print(f"Ratio {xgb_weight:.2f}-{lgb_weight:.2f} | MAP@3: {score:.5f}")

print(f"\nBest Ratio: XGB={best_ratio[0]}, LGB={best_ratio[1]} (MAP@3={best_score:.5f})")

# --- Generate Submission with Best Ratio ---
avg_test_preds_xgb = np.mean(test_preds_xgb, axis=0)
avg_test_preds_lgb = np.mean(test_preds_lgb, axis=0)
avg_test_preds = best_ratio[0] * avg_test_preds_xgb + best_ratio[1] * avg_test_preds_lgb

submission_labels = [" ".join([classes[i] for i in np.argsort(row)[-3:][::-1]]) for row in avg_test_preds]
submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': submission_labels
})

submission_df.to_csv(f'submission_best_ensemble_{best_ratio[0]}_{best_ratio[1]}.csv', index=False)
print("\nBest submission saved!")
print(submission_df.head())


