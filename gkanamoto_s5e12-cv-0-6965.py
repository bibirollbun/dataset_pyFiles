import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import gc, time

import warnings
warnings.filterwarnings('ignore')


# --- Configuration
SEED = 42
FOLDS = 5

sns.set_style("whitegrid")


# --- Load dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
print("train", train_df.shape, "test", test_df.shape)


#--- Check missing values
print('TRAIN: ', train_df.isnull().sum())
print('TEST : ', test_df.isnull().sum())


#--- Check data types
print("======== TRAIN ========")
print(train_df.dtypes, "\n")
print("======== TEST ========")
print(test_df.dtypes)


#--- Distribution of diagnosed_diabetes
plt.figure(figsize=(10, 5))
bar = sns.countplot(data=train_df, x='diagnosed_diabetes', palette='Set1')

counts = train_df['diagnosed_diabetes'].value_counts().sort_index()
ratios = counts / counts.sum() * 100

for p, (cls, ratio) in zip(bar.patches, ratios.items()):
    height = p.get_height()
    bar.annotate(
        f"{int(height):,}\n({ratio:.1f}%)",
        (p.get_x() + p.get_width() / 2., height),
        ha='center', va='bottom',
        fontsize=12, fontweight='bold',
        color='black', xytext=(0, 4),
        textcoords='offset points'
    )

plt.xlabel('diagnosed_diabetes')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


#--- Histograms of numerical features
cols = train_df.select_dtypes(include='number').columns

plt.figure(figsize=(18, 10))
for i, col in enumerate(cols):
    plt.subplot(5, 5, i+1)

    sns.histplot(train_df[col], kde=True, palette='Set1')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()   


#--- Categorical features by diagnosed_diabetes
cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

plt.figure(figsize=(15, 15))
for i, col in enumerate(cols):
    plt.subplot(3, 2, i+1)

    bar = sns.countplot(x=col, hue='diagnosed_diabetes', data=train_df, palette='Set1')
    for i in bar.containers:
        bar.bar_label(i)

    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()       



cols = [col for col in train_df.columns if col not in ['id', 'diagnosed_diabetes']]
new_cols = []

for col in cols:
    # mean
    mean_map = orig.groupby(col)['diagnosed_diabetes'].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name

    train_df = train_df.merge(mean_map, on=col, how='left')
    test_df = test_df.merge(mean_map, on=col, how='left')
    new_cols.append(new_mean_col_name)

    # count
    new_cnt_col_name = f"orig_cnt_{col}"
    cnt_map = orig.groupby(col).size().reset_index(name=new_cnt_col_name)

    train_df = train_df.merge(cnt_map, on=col, how='left')
    test_df = test_df.merge(cnt_map, on=col, how='left')
    new_cols.append(new_cnt_col_name)

for col in new_cols:
    if 'mean' in col:
        train_df[col] = train_df[col].fillna(orig['diagnosed_diabetes'].mean())
        test_df[col] = test_df[col].fillna(orig['diagnosed_diabetes'].mean())
    else:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)


#--- handling categorical features
from sklearn.preprocessing import OneHotEncoder

cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
encoder = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')

# train data
encoded_train = encoder.fit_transform(train_df[cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cols))
train_df = pd.concat([train_df.drop(columns=cols), encoded_train_df], axis=1)

# test data
encoded_test = encoder.transform(test_df[cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cols))
test_df = pd.concat([test_df.drop(columns=cols), encoded_test_df], axis=1)    


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score


#--- Configurations
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 1024
LR = 1e-4
FOLDS = 5
NUM_WORKERS = 2
VAL_CHECK = False

def seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_availabel():
        torch.cuda.manual_seed_all(seed)


# Scaling
num_cols = ['alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
            'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
            'triglycerides', 'family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
orig_cols = [col for col in train_df.columns if 'orig_' in col]
num_cols += orig_cols

scaler = StandardScaler()
scaler.fit(train_df[num_cols])
train_df[num_cols] = scaler.transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])


#--- Define PyTorch Datasets
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


#--- Define MLPClassifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


#--- Loop
if VAL_CHECK is True:
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    auc_scores = []
    X = train_df.drop(columns=['id'])
    # X = train_df.drop(columns=['id', 'diagnosed_diabetes'])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n======= Fold {fold+1} =======")
    
        X_train = X.iloc[train_idx].drop("diagnosed_diabetes", axis=1)
        y_train = X.iloc[train_idx]["diagnosed_diabetes"]
    
        X_val = X.iloc[val_idx].drop("diagnosed_diabetes", axis=1)
        y_val = X.iloc[val_idx]["diagnosed_diabetes"]
        
        train_ds = TabularDataset(X_train, y_train)
        val_ds = TabularDataset(X_val, y_val)
    
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    
        model = MLPClassifier(input_dim=X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
        #--- training
        for epoch in range(EPOCHS):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(X_batch).squeeze()
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
    
        #--- validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).squeeze().numpy()
                val_preds.extend(preds)
                val_targets.extend(y_batch.numpy())
    
        auc = roc_auc_score(val_targets, val_preds)
        auc_scores.append(auc)
        print(f"Fold {fold+1} AUC: {auc:.4f}")
    
    print("\nAvg AUC:", round(sum(auc_scores) / len(auc_scores), 5))


#--- 100% Training
from tqdm import tqdm
import time

for i in tqdm(range(100)):
    
    X_full = train_df.drop(columns=["id", "diagnosed_diabetes"])
    y_full = train_df["diagnosed_diabetes"]
    full_ds = TabularDataset(X_full, y_full)
    full_loader = DataLoader(full_ds, batch_size=256, shuffle=True)
    
    model = MLPClassifier(input_dim=X_full.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in full_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

    time.sleep(0.1)


#--- Inference
test_id = test_df.id
test = test_df.drop("id", axis=1)
test_ds = TabularDataset(test)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

model.eval()
test_preds = []
with torch.no_grad():
    for X_batch in test_loader:
        preds = model(X_batch).squeeze().numpy()
        test_preds.extend(preds)

submit = pd.DataFrame({
    "id": test_id,
    "diagnosed_diabetes": test_preds
})

submit.to_csv("submission.csv", index=False)
submit
print("Complete!!")

