import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
pd.options.mode.copy_on_write = True
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from typing import List
import warnings
warnings.simplefilter('ignore')


# Load data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_orginal = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer-Prediction.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)

# Convert categorical columns to 'category' dtype
for col in df_test.select_dtypes(include=['object']).columns:
    df_train[col] = df_train[col].astype('category')
    df_orginal[col] = df_orginal[col].astype('category')
    df_test[col] = df_test[col].astype('category')

df_train['const'] = 0
df_orginal['const'] = 0
df_test['const'] = 0

# Get Target column
target = df_train.pop('Fertilizer Name')
target_org = df_orginal.pop('Fertilizer Name')

# Encode target labels
le = LabelEncoder()
target = le.fit_transform(target)
target_org = le.transform(target_org)

df_train.shape, df_test.shape


def fast_map_k(actual: List[List], predicted: List[List], k: int = 3) -> float:
    total_score = 0.0
    
    for true_items, pred_items in zip(actual, predicted):
        if not true_items:
            continue
            
        pred_items = pred_items[:k]
        true_set = set(true_items)
        
        # Create boolean mask for hits
        hits = np.array([item in true_set for item in pred_items])
        
        if not hits.any():
            continue
        
        # Calculate cumulative hits and positions
        cumulative_hits = np.cumsum(hits)
        positions = np.arange(1, len(pred_items) + 1)
        
        # Calculate precision at each hit position
        precisions = cumulative_hits[hits] / positions[hits]
        score = np.sum(precisions) / min(len(true_items), k)
        total_score += score
    
    return total_score / len(actual)


FOLDS = 5
sk_fold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros((len(df_train), np.unique(target).shape[0]))
pred_test = np.zeros((len(df_test), np.unique(target).shape[0]))
final_score = 0

params = {
        'objective': 'multi:softprob',
        'num_class': 7,
        'max_depth': 16,
        'learning_rate': 0.01,
        'n_estimators': 100_000,
        'reg_alpha': 3,
        'reg_lambda': 1.4,
        'gamma': 0.26,
        'max_delta_step': 5,
        'subsample': 0.86,
        'colsample_bytree': 0.4,
        'min_child_weight': 5,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss',
        'enable_categorical': True,
        'device': "cuda"   
}

for i, (indx_train, indx_valid) in enumerate(sk_fold.split(df_train, target)):
    print(f"Fold {i+1}")

    X_train, y_train = df_train.iloc[indx_train], target[indx_train]
    X_valid, y_valid = df_train.iloc[indx_valid], target[indx_valid]
    X_test = df_test.copy()

    X_train = pd.concat([X_train, df_orginal], axis=0)
    y_train = np.concatenate([y_train, target_org], axis=0)

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=100_000, 
        evals=[(dtrain, 'train'), (dval, 'validation')], 
        early_stopping_rounds=30, 
        verbose_eval=2000
    )

    oof[indx_valid] = model.predict(dval)
    pred_test += model.predict(dtest)

    top_preds = np.argsort(oof[indx_valid], axis=1)[:, -3:][:, ::-1]  
    score = fast_map_k([[label] for label in y_valid], top_preds)
    final_score += score
    print(f"Score: {score:.5f}\n")

final_score /= FOLDS
print(f"Overall Score: {final_score:.5f}")

top_preds = np.argsort(pred_test, axis=1)[:, -3:][:, ::-1]
top_labels = le.inverse_transform(top_preds.ravel()).reshape(top_preds.shape)

df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
df_sub['Fertilizer Name'] = [' '.join(row) for row in top_labels]

df_sub.to_csv('submission.csv', index=False)

