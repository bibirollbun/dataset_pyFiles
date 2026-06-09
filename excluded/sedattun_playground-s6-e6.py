from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import optuna
import shutil
import glob
import json
import gc
import lightgbm as lgb
import os
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
org = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


org_copy = org.copy()
for _ in range(7):
    org = pd.concat([org, org_copy], axis=0)


print(train.info())
print(test.info())


# Feature engineering
numerical_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns 
                      if col != 'id']
for df in [train, test, org]:
    for col in numerical_features:
        df[f'{col}_Binned'] = df[col].astype(str).astype('category')
    df = df.rename(columns={'Temparature': 'Temperature'})
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int8')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float16')


cat_cols = [col for col in train.select_dtypes(include=['object', 'category']).columns 
            if col != "Fertilizer Name"]
for col in cat_cols:
    label_enc = LabelEncoder()
    train[col] = label_enc.fit_transform(train[col])
    org[col] = label_enc.transform(org[col])
    test[col] = label_enc.transform(test[col])

target_enc = LabelEncoder()
train["Fertilizer Name"] = target_enc.fit_transform(train["Fertilizer Name"])
org["Fertilizer Name"] = target_enc.transform(org["Fertilizer Name"])

for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")
    org[col] = org[col].astype("category")


#label_target = LabelEncoder()

#label_target.fit(pd.concat([train["Fertilizer Name"],org["Fertilizer Name"]]))

#train["Fertilizer Name"]= label_target.transform(train["Fertilizer Name"])

#org["Fertilizer Name"] = label_target.transform(org["Fertilizer Name"])


print(train.info())
print(test.info())


X = train.drop(columns=["id", "Fertilizer Name"])
y = train["Fertilizer Name"]
X_test = test.drop(columns=["id"])
X_original = org.drop(columns=["Fertilizer Name"])
y_original = org["Fertilizer Name"]


#map3
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# Model configurations
model_configs = {
    'xgb': {
        'model': XGBClassifier,
        'params': {
            'objective': 'multi:softprob',  
            'num_class': len(np.unique(y)), 
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'max_bin': 128,
            'colsample_bytree': 0.3, 
            'colsample_bylevel': 1,  
            'colsample_bynode': 1,  
            'tree_method': 'hist',  
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'device': "cuda",
            'enable_categorical':True,
            'n_estimators':10000,
            'early_stopping_rounds':50,
        }  

    },
    'lgb_goss': {
        'model': LGBMClassifier,
        'params': {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'boosting_type': 'goss',
            'device': 'gpu',
            'colsample_bytree': 0.3275,
            'learning_rate': 0.02670,
            'max_depth': 9,
            'min_child_samples': 84,
            'n_estimators': 10000,
            'n_jobs': -1,
            'num_leaves': 229,
            'random_state': 42,
            'reg_alpha': 6.87997,
            'reg_lambda': 4.7391,
            'subsample': 0.5411,
            'categorical_feature': cat_cols,
            'verbose': -1
        }
    },
    'lgb': {
        'model': LGBMClassifier,
        'params': {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            "device": "gpu",
            "colsample_bytree": 0.4366,
            "learning_rate": 0.02617,
            "max_depth": 11,
            "min_child_samples": 67,
            "n_estimators": 10000,
            "n_jobs": -1,
            "num_leaves": 243,
            "random_state": 42,
            "reg_alpha": 6.38283,
            "reg_lambda": 9.39295,
            "subsample": 0.79898,
            'categorical_feature': cat_cols,
            "verbose": -1

        }
    }
}


skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
oof_preds = {name: np.zeros((len(X), y.nunique())) for name in model_configs}
test_preds = {name: np.zeros((len(X_test), y.nunique())) for name in model_configs}
map3_scores = {name: [] for name in model_configs}

for name, config in model_configs.items():
    print(f"\nTraining {name.upper()}...")
    model = config['model'](**config['params'])
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/7")
        x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        x_train = pd.concat([x_train, X_original], axis=0, ignore_index=True)
        y_train = pd.concat([y_train, y_original], axis=0, ignore_index=True)
        
        if name == 'xgb':
            model.fit(
                x_train, y_train,
                eval_set=[(x_train, y_train), (x_valid, y_valid)],
                verbose=0
            )
        elif name in ['lgb', 'lgb_goss']:
            model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=100)]
            )
        
        oof_preds[name][valid_idx] = model.predict_proba(x_valid)
        test_preds[name] += model.predict_proba(X_test) / 7
        
        top_3_preds = np.argsort(oof_preds[name][valid_idx], axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y_valid]
        map3_score = mapk(actual, top_3_preds)
        map3_scores[name].append(map3_score)
        print(f" {name.upper()} Fold {fold + 1}: MAP@3 Score: {map3_score:.5f}")
    
    print(f" Average {name.upper()} MAP@3 Score: {np.mean(map3_scores[name]):.5f}")

# Stacking ensemble
stacking_train = np.hstack([oof_preds[name] for name in oof_preds])
stacking_test = np.hstack([test_preds[name] for name in test_preds])

meta_model = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y)),
    learning_rate=0.03,
    n_estimators=10000,
    random_state=42,
    verbose=-1
)

print("\nTraining Stacking Ensemble...")
final_oof = np.zeros((len(y), len(np.unique(y))))
final_test = np.zeros((len(X_test), len(np.unique(y))))
ensemble_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(stacking_train, y)):
    x_train, x_valid = stacking_train[train_idx], stacking_train[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    meta_model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )
    
    final_oof[valid_idx] = meta_model.predict_proba(x_valid)
    final_test += meta_model.predict_proba(stacking_test) / 7
    
    top_3_preds = np.argsort(final_oof[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    ensemble_scores.append(map3_score)
    print(f" Ensemble Fold {fold + 1}: MAP@3 Score: {map3_score:.5f}")

print(f" Average Ensemble MAP@3 Score: {np.mean(ensemble_scores):.5f}")

# Save results
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

np.save(f'{output_dir}/stacking_oof.npy', final_oof)
np.save(f'{output_dir}/stacking_test.npy', final_test)
for name in oof_preds:
    np.save(f'{output_dir}/{name}_oof.npy', oof_preds[name])
    np.save(f'{output_dir}/{name}_test.npy', test_preds[name])

# Prepare submission
top_3_preds = np.argsort(final_test, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
submission = pd.DataFrame({
    'id': submission['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)

# Save scores
with open(f'{output_dir}/scores.txt', 'w') as f:
    for name, scores in map3_scores.items():
        f.write(f"{name.upper()} MAP@3 Scores: {scores}\n")
        f.write(f"{name.upper()} Average MAP@3: {np.mean(scores):.5f}\n")
    f.write(f"Ensemble MAP@3 Scores: {ensemble_scores}\n")
    f.write(f"Ensemble Average MAP@3: {np.mean(ensemble_scores):.5f}\n")


print(submission.head())




