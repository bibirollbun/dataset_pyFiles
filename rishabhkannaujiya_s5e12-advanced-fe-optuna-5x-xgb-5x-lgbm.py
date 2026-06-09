import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer

from catboost import CatBoostClassifier

import lightgbm as lgb

warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


TARGET = 'diagnosed_diabetes'


print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


train_df.head()


print("Target Distribution:")
print(train_df[TARGET].value_counts(normalize=True))


# Combine
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df[TARGET] = np.nan

all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)


train_viz = all_data[all_data['is_train'] == 1].copy()

plt.figure(figsize=(6, 4))
sns.countplot(data=train_viz, x=TARGET, palette='viridis')
plt.title('Distribution of Target: Diagnosed Diabetes')
plt.xlabel('0 = Negative, 1 = Positive')
plt.ylabel('Count')

counts = train_viz[TARGET].value_counts(normalize=True)
print(f"Negative (0): {counts[0]:.2%}")
print(f"Positive (1): {counts[1]:.2%}")
plt.show()


# Correlation Heatmap
numeric_cols = train_viz.select_dtypes(include=[np.number]).columns.tolist()

numeric_cols = [c for c in numeric_cols if c not in ['id', 'is_train']]

plt.figure(figsize=(14, 12))

corr = train_viz[numeric_cols].corr()

sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()


print("Top 10 Features correlated with Diabetes:")
print(corr[TARGET].abs().sort_values(ascending=False).head(11))


# Numerical Distributions
# orange (Diabetes) and blue (No Diabetes) box
features_to_plot = ['age', 'bmi', 'systolic_bp', 'Pulse_Pressure', 'MAP']

plt.figure(figsize=(15, 10))
for i, col in enumerate(features_to_plot):
    if col in train_viz.columns:
        plt.subplot(2, 3, i + 1)
        sns.boxplot(data=train_viz, x=TARGET, y=col, palette='Set2')
        plt.title(f'{col} Distribution by Diabetes Status')
        plt.xlabel('Diagnosed Diabetes')
plt.tight_layout()
plt.show()


# Fixed: ValueError: num must be an integer with 1 <= num <= 3, not 4
import math

cat_features = ['family_history_diabetes', 'hypertension_history', 'smoking_status', 'gender', 'ethnicity']
n = len(cat_features)

cols = 3
rows = math.ceil(n / cols)

plt.figure(figsize=(cols * 6, rows * 4))

for i, col in enumerate(cat_features):
    if col in train_viz.columns:
        plt.subplot(rows, cols, i + 1)

        # We will calculate the mean of the target which will gives us the probability percentage of diabetes
        risk_per_group = train_viz.groupby(col)[TARGET].mean().sort_values()
        sns.barplot(x=risk_per_group.index, y=risk_per_group.values, palette='Reds')
        plt.title(f'Diabetes Risk by {col}')
        plt.ylabel('Probability of Diabetes')
        plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# Feature Engineering
# Blood Pressure Metrics
all_data['Pulse_Pressure'] = all_data['systolic_bp'] - all_data['diastolic_bp']
all_data['MAP'] = all_data['diastolic_bp'] + (all_data['Pulse_Pressure'] / 3) # Mean Arterial Pressure

all_data['BMI_Age_Interact'] = all_data['bmi'] * all_data['age']

risk_cols = ['hypertension_history', 'cardiovascular_history', 'family_history_diabetes']
all_data['Risk_Count'] = all_data[risk_cols].sum(axis=1)

# High screen time + Low activity = Higher risk
all_data['Sedentary_Factor'] = (all_data['screen_time_hours_per_day'] * 2) - (all_data['physical_activity_minutes_per_week'] / 60)


# Split back to Train/Test
train_clean = all_data[all_data['is_train'] == 1].drop(columns=['id', 'is_train'])
test_clean = all_data[all_data['is_train'] == 0].drop(columns=['id', 'is_train', TARGET])

X = train_clean.drop(columns=[TARGET])
y = train_clean[TARGET]
X_test = test_clean.copy()


cat_cols = ['family_history_diabetes', 'hypertension_history', 'smoking_status', 
            'gender', 'ethnicity', 'education_level', 'income_level', 
            'employment_status', 'alcohol_consumption', 'stress_level', 
            'sleep_quality_category']

cat_cols = [c for c in cat_cols if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"Categorical Cols: {len(cat_cols)}")
print(f"Numerical Cols: {len(num_cols)}")


# We need to ensure integers are contiguous for Embedding layers
for col in cat_cols:
    le = LabelEncoder()
    full_col = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(full_col)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# 3. RankGauss Scaling for Numericals (Crucial for NNs!)
# QuantileTransformer forces data into a Normal Distribution
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 4. define Embedding Dimensions
cat_dims = []
for col in cat_cols:
    cardinality = pd.concat([X[col], X_test[col]]).nunique()
    embedding_dim = min(50, (cardinality + 1) // 2)
    cat_dims.append((cardinality, embedding_dim))
    
print("Data Preparation Complete.")


FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)


NN_PARAMS = {
    'batch_size': 1024,
    'epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 7
}


class TabularDataset(Dataset):
    def __init__(self, X_df, y=None, cat_cols=None, num_cols=None):
        self.X_cat = X_df[cat_cols].values.astype(np.int64)
        self.X_num = X_df[num_cols].values.astype(np.float32)
        self.y = y.values.astype(np.float32) if y is not None else None
        
    def __len__(self):
        return len(self.X_num)
        
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_num[idx], self.y[idx]
        return self.X_cat[idx], self.X_num[idx]

class EmbeddingMLP(nn.Module):
    def __init__(self, cat_dims, num_in_features):
        super(EmbeddingMLP, self).__init__()
        
        # Embedding Layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, dim) for num_classes, dim in cat_dims
        ])
        
        # Calculate total dimension after concatenation
        total_emb_dim = sum([dim for _, dim in cat_dims])
        input_dim = total_emb_dim + num_in_features
        
        self.bn_in = nn.BatchNorm1d(num_in_features) # BN for numericals
        
        # Main Architecture (Wider is often better for tabular)
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(256, 1)
        
    def forward(self, x_cat, x_num):
        # Process Embeddings
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(x_cat[:, i]))
        x_cat = torch.cat(embeddings, 1)
        
        # Process Numericals
        x_num = self.bn_in(x_num)
        
        # Concatenate
        x = torch.cat([x_cat, x_num], 1)
        
        # Forward pass
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


nn_oof = np.zeros(len(X))
nn_pred_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}...")

    # Create Datasets using the column names lists
    train_ds = TabularDataset(X.iloc[train_idx], y.iloc[train_idx], cat_cols, num_cols)
    val_ds = TabularDataset(X.iloc[val_idx], y.iloc[val_idx], cat_cols, num_cols)

    train_loader = DataLoader(train_ds, batch_size=NN_PARAMS['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=NN_PARAMS['batch_size'], shuffle=False)

    model = EmbeddingMLP(cat_dims, len(num_cols)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=NN_PARAMS['learning_rate'], weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_score = 0
    patience_cnt = 0
    best_weights = None

    for epoch in range(NN_PARAMS['epochs']):
        model.train()
        for x_cat, x_num, batch_y in train_loader:
            x_cat, x_num, batch_y = x_cat.to(device), x_num.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(x_cat, x_num).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_cat, x_num, batch_y in val_loader:
                x_cat, x_num = x_cat.to(device), x_num.to(device)
                logits = model(x_cat, x_num).squeeze()
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(batch_y.numpy())

        val_auc = roc_auc_score(np.concatenate(val_targets), np.concatenate(val_preds))
        scheduler.step(val_auc)

        print(f"  Epoch {epoch+1}/{NN_PARAMS['epochs']} - Val AUC: {val_auc:.5f} (Best: {best_score:.5f})")

        if val_auc > best_score:
            best_score = val_auc
            best_weights = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= NN_PARAMS['patience']: break

    # Save OOF
    model.load_state_dict(best_weights)
    model.eval()

    # Predict validation (in chunks to be safe)
    with torch.no_grad():
        val_p = []
        for x_cat, x_num, _ in val_loader:
             x_cat, x_num = x_cat.to(device), x_num.to(device)
             val_p.append(torch.sigmoid(model(x_cat, x_num).squeeze()).cpu().numpy())
        nn_oof[val_idx] = np.concatenate(val_p)

        # Predict Test
        test_ds = TabularDataset(X_test, None, cat_cols, num_cols)
        test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
        test_p = []
        for x_cat, x_num in test_loader:
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            test_p.append(torch.sigmoid(model(x_cat, x_num).squeeze()).cpu().numpy())
        nn_pred_test += np.concatenate(test_p) / FOLDS

    print(f"Fold {fold+1} AUC: {best_score:.5f}")

print(f"Final NN Score: {roc_auc_score(y, nn_oof):.5f}")


xgb_params = {
    'learning_rate': 0.015,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.92536955554124,
    'reg_alpha': 4.647096041905858,
    'reg_lambda': 1.964190052774896,
    'min_child_weight': 3,
    'n_estimators': 3000,
    'tree_method': 'gpu_hist',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'early_stopping_rounds': 100
}


print("Creating NumPy arrays for GPU acceleration...")
X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

xgb_seeds = [42, 43, 44, 45, 46]
xgb_avg_oof = np.zeros(len(X))
xgb_avg_test = np.zeros(len(X_test))


print(f"Starting Optimized XGBoost Seed Averaging ({len(xgb_seeds)} seeds)...")

for seed in xgb_seeds:
    xgb_params['random_state'] = seed
    
    seed_oof = np.zeros(len(X))
    seed_test = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]
        
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_probs = model.predict_proba(X_val)[:, 1]
        seed_oof[val_idx] = val_probs
        seed_test += model.predict_proba(X_test_np)[:, 1] / FOLDS
        
    xgb_avg_oof += seed_oof / len(xgb_seeds)
    xgb_avg_test += seed_test / len(xgb_seeds)
    
    score = roc_auc_score(y, seed_oof)
    print(f"Seed {seed} AUC: {score:.5f}")

xgb_final_score = roc_auc_score(y, xgb_avg_oof)
print(f"\nFinal Optimized XGBoost Score: {xgb_final_score:.5f}")


lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.015,
    'num_leaves': 64,
    'max_depth': 5,
    'min_child_samples': 16,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 9.623757827094243,
    'reg_lambda': 9.658209424638022,
    'n_estimators': 3000,
    'device': 'gpu',
    'verbose': -1,
    'bagging_freq': 1
}


SEEDS = [42, 43, 44, 45, 46]
lgb_avg_oof = np.zeros(len(X))
lgb_avg_test = np.zeros(len(X_test))


X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

lgb_avg_oof = np.zeros(len(X))
lgb_avg_test = np.zeros(len(X_test))

print(f"Starting LightGBM Seed Averaging (Running {len(SEEDS)} seeds) on GPU...")

for seed in SEEDS:
    lgb_params['random_state'] = seed
    
    seed_oof = np.zeros(len(X))
    seed_test = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        val_probs = model.predict_proba(X_val)[:, 1]
        seed_oof[val_idx] = val_probs
        
        seed_test += model.predict_proba(X_test_np)[:, 1] / FOLDS
        
    lgb_avg_oof += seed_oof / len(SEEDS)
    lgb_avg_test += seed_test / len(SEEDS)
    
    seed_score = roc_auc_score(y, seed_oof)
    print(f"Seed {seed} Completed | AUC: {seed_score:.5f}")

lgb_final_score = roc_auc_score(y, lgb_avg_oof)
print(f"\nFinal Seed-Averaged LightGBM Score: {lgb_final_score:.5f}")


print("Finding Best Blend Weights (XGB + LGBM + NN)...")

best_score = 0
best_weights = (0, 0, 0)

for w_xgb in np.arange(0, 1.01, 0.1):
    for w_lgb in np.arange(0, 1.01 - w_xgb, 0.1):
        w_nn = 1.0 - w_xgb - w_lgb
        if w_nn < 0: continue
        
        blend = (xgb_avg_oof * w_xgb) + (lgb_avg_oof * w_lgb) + (nn_oof * w_nn)
        score = roc_auc_score(y, blend)
        
        if score > best_score:
            best_score = score
            best_weights = (w_xgb, w_lgb, w_nn)

print(f"Best Weights -> XGB: {best_weights[0]:.2f}, LGB: {best_weights[1]:.2f}, NN: {best_weights[2]:.2f}")
print(f"Final Ensemble Score: {best_score:.5f}")


final_blend_preds = (xgb_avg_test * best_weights[0]) + \
                    (lgb_avg_test * best_weights[1]) + \
                    (nn_pred_test * best_weights[2])

submission = pd.DataFrame({'id': test_df['id'], 'diagnosed_diabetes': final_blend_preds})
submission.to_csv('submission.csv', index=False)




