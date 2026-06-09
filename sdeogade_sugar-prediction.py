#IMPORTS
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

target = "diagnosed_diabetes"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


# Cell 2: Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')

print(f"Train: {train.shape} | Test: {test.shape} | Orig: {orig.shape}")


# Cell 3: Outlier Removal
def remove_top50_outliers_quantile(df, low_q=0.01, high_q=0.99):
    df_clean = df.copy()
    cols = df_clean.select_dtypes(include=["int", "float"]).columns
    for col in cols:
        q_low = df_clean[col].quantile(low_q)
        q_high = df_clean[col].quantile(high_q)
        outliers_below = df_clean[df_clean[col] < q_low].index
        outliers_above = df_clean[df_clean[col] > q_high].index
        outliers = list(outliers_below) + list(outliers_above)
        outliers_to_remove = outliers[:50]
        df_clean.drop(outliers_to_remove, inplace=True)
    return df_clean

train = remove_top50_outliers_quantile(train)
print(f"Train after outliers: {train.shape}")


# Cell 4: ORIG Features
BASE = [col for col in train.columns if col not in ['id', target]]
CATS = train.select_dtypes('object').columns.to_list()
NUMS = [col for col in BASE if col not in CATS]

ORIG = []

for col in BASE:
    mean_map = orig.groupby(col)[target].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(f'{len(ORIG)} ORIG Features Created.')


# Cell 5: Fill NaN in ORIG
for col in ORIG:
    if 'mean' in col:
        train[col] = train[col].fillna(orig[target].mean())
        test[col] = test[col].fillna(orig[target].mean())
    else:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)


# Cell 6: Medical Features
def create_medical_features(df):
    df = df.copy()
    
    df['lipid_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    df['map_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
    
    df['bmi_waist_interaction'] = df['bmi'] * df['waist_to_hip_ratio']

    df['lifestyle_risk_score'] = (
        (df['alcohol_consumption_per_week'] / 9.0) + 
        (df['screen_time_hours_per_day'] / 16.0) -  
        (df['physical_activity_minutes_per_week'] / 750.0) - 
        (df['sleep_hours_per_day'] / 10.0)
    )
    
    df['high_bp_flag'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)).astype(int)
    
    df['high_tg_flag'] = (df['triglycerides'] >= 150).astype(int)
    
    df['low_hdl_flag'] = (df['hdl_cholesterol'] < 45).astype(int)
    
    df['obesity_flag'] = (df['bmi'] >= 30).astype(int)
    
    df['metabolic_risk_count'] = (
        df['high_bp_flag'] + 
        df['high_tg_flag'] + 
        df['low_hdl_flag'] + 
        df['obesity_flag'] + 
        df['hypertension_history']
    )
    
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 35, 50, 65, 100], 
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    return df

train = create_medical_features(train)
test = create_medical_features(test)

BASE = [col for col in train.columns if col not in ['id', target]]
CATS = train.select_dtypes('object').columns.to_list()
NUMS = [col for col in BASE if col not in CATS]


# Cell 7: Target Encoder OOF
class TargetEncoderOOF:
    def __init__(self, n_splits=5, smooth=10, random_state=42):
        self.n_splits = n_splits
        self.smooth = smooth
        self.random_state = random_state
        self.map_dict = {}

    def fit_transform(self, X, y, cat_cols):
        X_encoded = X.copy()
        
        if y.nunique() > 20: 
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in cat_cols:
            global_mean = y.mean()
            agg = X.groupby(col)[y.name].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            smooth_mean = (means * counts + global_mean * self.smooth) / (counts + self.smooth)
            self.map_dict[col] = smooth_mean

            X_encoded[f"TE_{col}"] = np.nan 
            
            for train_idx, val_idx in kf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr = y.iloc[train_idx]
                
                fold_agg = X_tr.groupby(col)[y.name].agg(['count', 'mean'])
                fold_counts = fold_agg['count']
                fold_means = fold_agg['mean']
                
                fold_smooth = (fold_means * fold_counts + global_mean * self.smooth) / (fold_counts + self.smooth)
                
                X_encoded.loc[val_idx, f"TE_{col}"] = X_val[col].map(fold_smooth)
            
            X_encoded[f"TE_{col}"] = X_encoded[f"TE_{col}"].fillna(global_mean)
            
        return X_encoded

    def transform(self, X_test, cat_cols):
        X_test_encoded = X_test.copy()
        for col in cat_cols:
            if col in self.map_dict:
                X_test_encoded[f"TE_{col}"] = X_test_encoded[col].map(self.map_dict[col])
                X_test_encoded[f"TE_{col}"] = X_test_encoded[f"TE_{col}"].fillna(self.map_dict[col].mean())
        return X_test_encoded

encoder = TargetEncoderOOF(n_splits=5, smooth=10, random_state=42)
train = encoder.fit_transform(train, train[target], CATS)
test = encoder.transform(test, CATS)


from sklearn.preprocessing import LabelEncoder

# Cell 8: Label Encoding
categorical_vars = CATS

label_encoders = {}
for column in categorical_vars:
    le = LabelEncoder()
    train[column] = le.fit_transform(train[column].astype(str))
    label_encoders[column] = le

for column in categorical_vars:
    le = LabelEncoder()
    test[column] = le.fit_transform(test[column].astype(str))
    label_encoders[column] = le


# Cell 9: One-Hot Encoding
cat_cols = CATS

for column in cat_cols:
    dummies = pd.get_dummies(train[column], prefix=column, drop_first=True)
    train = pd.concat([train, dummies], axis=1)
    train.drop(column, axis=1, inplace=True)

for column in train.select_dtypes(include=['bool']).columns:
    train[column] = train[column].astype(int)
    
for column in cat_cols:
    dummies = pd.get_dummies(test[column], prefix=column, drop_first=True)
    test = pd.concat([test, dummies], axis=1)
    test.drop(column, axis=1, inplace=True)

for column in test.select_dtypes(include=['bool']).columns:
    test[column] = test[column].astype(int)


import gc

# Cell 10: Memory Reduction
def memory_mb(df):
    return df.memory_usage(deep=True).sum() / 1024**2

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
    return df

print("Train memory BEFORE:", f"{memory_mb(train):.2f} MB")
print("Test memory BEFORE:", f"{memory_mb(test):.2f} MB")

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

print("Train memory AFTER:", f"{memory_mb(train):.2f} MB")
print("Test memory AFTER:", f"{memory_mb(test):.2f} MB")

gc.collect()


# Cell 11: Prepare X, y
y = train[target]
X = train.drop(columns=[target, 'id'])
X_test = test.drop(columns=['id'])


X.shape


X_np = X.values
X_test_np = X_test.values
y_np = y.values

print(f"Final feature matrix: {X_np.shape}, Test: {X_test_np.shape}")


# DEFINE MLP

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# TRAIN MLP OOF FUNCTION

def train_mlp_oof(
    X_np, y_np, X_test_np,
    n_splits=10, epochs=10, batch_size=1024, lr=1e-4
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(X_np))
    test_preds = np.zeros(len(X_test_np))

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np.reshape(-1,1), dtype=torch.float32)

    input_dim = X_np.shape[1]

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y_np), 1):
        print(f"\n===== Training MLP Fold {fold}/{n_splits} =====")

        model = MLP(input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        train_ds = TensorDataset(X_tensor[tr_idx], y_tensor[tr_idx])
        val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"  Epoch {epoch+1}/{epochs} Loss: {running_loss:.4f}")

        # Validation preds
        model.eval()
        with torch.no_grad():
            val_pred = model(X_tensor[val_idx].to(device)).cpu().numpy().flatten()
            oof[val_idx] = val_pred

            test_pred = model(X_test_tensor.to(device)).cpu().numpy().flatten()
            test_preds += test_pred / n_splits

        fold_auc = roc_auc_score(y_np[val_idx], val_pred)
        print(f"  Fold {fold} AUC: {fold_auc:.6f}")

    full_auc = roc_auc_score(y_np, oof)
    print("\n===== FINAL MLP OOF AUC:", round(full_auc, 6), "=====")

    return oof, test_preds


# RUN TRAINING
mlp_oof, mlp_test = train_mlp_oof(
    X_np, y_np, X_test_np,
    n_splits=10,
    epochs=10,
    batch_size=1024,
    lr=1e-4
)


# SAVE RESULTS

np.save("mlp_oof.npy", mlp_oof)
np.save("mlp_test.npy", mlp_test)

pd.DataFrame({
    "id": np.arange(len(mlp_oof)),
    "oof": mlp_oof,
    "target": y_np
}).to_csv("mlp_oof.csv", index=False)

print("MLP OOF & TEST predictions saved!")


# SUBMISSION FILE

submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission[target] = mlp_test
submission.to_csv("submission_MLP_only.csv", index=False)

print("submission_MLP_only.csv saved!")




