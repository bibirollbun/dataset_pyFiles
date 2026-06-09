import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
sample_submission = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv")


train.drop_duplicates(keep='first', inplace=True)
test.drop_duplicates(keep='first', inplace=True)


print("Train data overview:")
print(train.head())
print("Test data overview:")
print(test.head())
print("Missing values:")
print(train.isnull().sum())
print(test.isnull().sum())


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

for df in [train, test]:
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['Day_of_Week'] = df['date'].dt.weekday
    df['Quarter'] = df['date'].dt.quarter


train.drop(columns=['date'], inplace=True)
test.drop(columns=['date'], inplace=True)


MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)


torch.set_grad_enabled(False)

def encode_plate_number(plate_numbers):
    tokens = tokenizer(plate_numbers.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=10)
    embeddings = bert_model(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

batch_size = 64
train_embeddings = np.vstack([encode_plate_number(train['plate'][i:i+batch_size]) for i in range(0, len(train), batch_size)])
test_embeddings = np.vstack([encode_plate_number(test['plate'][i:i+batch_size]) for i in range(0, len(test), batch_size)])


train_embed_df = pd.DataFrame(train_embeddings, columns=[f'plate_emb_{i}' for i in range(train_embeddings.shape[1])])
test_embed_df = pd.DataFrame(test_embeddings, columns=[f'plate_emb_{i}' for i in range(test_embeddings.shape[1])])

train = pd.concat([train.reset_index(drop=True), train_embed_df.reset_index(drop=True)], axis=1)
test = pd.concat([test.reset_index(drop=True), test_embed_df.reset_index(drop=True)], axis=1)


train.drop(columns=['plate'], inplace=True)
test.drop(columns=['plate'], inplace=True)


X = train.drop(columns=['id', 'price'])
y = np.log1p(train['price'])  
X_test = test.drop(columns=['id'])


missing_cols = set(X.columns) - set(X_test.columns)
extra_cols = set(X_test.columns) - set(X.columns)


missing_cols = set(X.columns) - set(X_test.columns)
extra_cols = set(X_test.columns) - set(X.columns)

for col in missing_cols:
    X_test[col] = 0  
X_test = X_test[X.columns] 


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))


for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        reg_lambda=1.5,
        reg_alpha=0.5,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', early_stopping_rounds=50, verbose=False)
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits


smape = np.mean(2 * np.abs(y - oof_preds) / (np.abs(y) + np.abs(oof_preds))) * 100
print(f"Validation SMAPE: {smape:.2f}%")


submission = pd.DataFrame({
    'id': test['id'],
    'price': np.round(np.expm1(test_preds))  # Convert back from log scale
})
submission.to_csv("submission.csv", index=False)
print("Submission file created!")




