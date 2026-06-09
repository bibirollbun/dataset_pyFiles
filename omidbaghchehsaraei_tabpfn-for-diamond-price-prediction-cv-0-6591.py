pip_dir = "/kaggle/input/tabpfn-2-0-1-offline/"
!pip install --no-index --find-links=$pip_dir tabpfn 


import os
import torch
import warnings
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set a seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Determine the device to use (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

OFFLINE_MODEL_PATH = "/kaggle/input/tabpfn-2-0-1-offline/tabpfn-v2-regressor.ckpt"


train_file = "/kaggle/input/predicting-the-price-of-diamond/train.csv"
test_file = "/kaggle/input/predicting-the-price-of-diamond/test.csv"

train_df_original = pd.read_csv(train_file)
test_df_original = pd.read_csv(test_file)
submission_df = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/submission.csv")

combined_df = pd.concat([train_df_original.drop(columns='price'), test_df_original], ignore_index=True)

object_cols = combined_df.select_dtypes(include=['object']).columns

for col in object_cols:
    combined_df[col] = combined_df[col].astype('category').astype('str')
    
cat_cols = combined_df.select_dtypes(include=['string']).columns

encoder = OrdinalEncoder()
combined_df[cat_cols] = encoder.fit_transform(combined_df[cat_cols])

train_df = combined_df.iloc[:len(train_df_original)].copy()
test_df = combined_df.iloc[len(train_df_original):].copy()

numerical_cols = train_df.select_dtypes(include=['float64']).columns
for col in numerical_cols:
    train_df[col] = train_df[col].astype('float16')
    test_df[col] = test_df[col].astype('float16') 

train_df[cat_cols] = train_df[cat_cols].astype('int16')
test_df[cat_cols] = test_df[cat_cols].astype('int16') 

train_df['price'] = train_df_original['price'] 

features = [col for col in train_df.columns if col != 'price' and col != 'id']
target = 'price'

X = train_df[features].values
y = train_df[target].values
X_test = test_df[features].values 


%%time

test_predictions = []
oof_predictions = np.zeros(len(X))

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"Starting {n_splits}-fold cross-validation...")

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    
    print(f"\n--- Fold {fold+1}/{n_splits} ---")
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = TabPFNRegressor(device=device, ignore_pretraining_limits=True, model_path=OFFLINE_MODEL_PATH)
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds
    r2 = r2_score(y_val, val_preds)
    print(f"RÂ² score on validation set for Fold {fold+1}: {r2:.5f}")
    
    test_preds = model.predict(X_test)
    test_predictions.append(test_preds)

oof_r2 = r2_score(y, oof_predictions)

print("-----------------------") 
print(f"Overall Out-of-Fold (OOF) RÂ² Score: {oof_r2:.5f}") 


oof_df = pd.DataFrame({'price': oof_predictions, 'id': train_df['id']})
oof_df.to_csv('oof.csv', index=False)

submission_df['price'] = np.mean(test_predictions, axis=0) 
submission_df.to_csv('submission.csv', index=False)
submission_df.head() 

