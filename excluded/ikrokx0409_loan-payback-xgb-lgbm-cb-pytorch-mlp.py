import pandas as pd
import numpy as np

# models: xgb + lgbm + catboost
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from catboost import CatBoostClassifier, Pool

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer

import optuna
import os
from tqdm.auto import tqdm
import random

SEED = 37
N_SPLITS = 5 # 5-Fold
TARGET = 'loan_paid_back'
INPUT_DIR = '/kaggle/input/playground-series-s5e11/'
ORIG_DATA_PATH = '/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv'

# fixed seed
def set_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print("Libraries loaded")


# load data (competition + original)
train_df = pd.read_csv(f"{INPUT_DIR}train.csv")
test_df = pd.read_csv(f"{INPUT_DIR}test.csv")
submission_df = pd.read_csv(f"{INPUT_DIR}sample_submission.csv")
test_ids = test_df['id']

print(f"Competition train shape: {train_df.shape}")

# add original data
orig_df = pd.read_csv(ORIG_DATA_PATH)
print(f"Original data shape: {orig_df.shape}")

# combine datas
comp_cols_no_id = train_df.columns.drop('id')
orig_df_aligned = orig_df[comp_cols_no_id]
train_df = pd.concat([train_df.drop('id', axis=1), orig_df_aligned], ignore_index=True)

print(f"New combined train shape (comp + orig): {train_df.shape}")

# Disrupt data
train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Final train shape for FE: {train_df.shape}")
print(f"Final test shape for FE: {test_df.shape}")


def feature_engineer(df):
    df_ = df.copy()

    # --- grade & subgrade
    def extract_grade_subgrade(s):
        if pd.isna(s): return (np.nan, np.nan)
        s = str(s).strip()
        if len(s)==0: return (np.nan, np.nan)
        return (s[0], s[1:])

    g = df_['grade_subgrade'].fillna("").astype(str).apply(extract_grade_subgrade)
    df_['grade'] = g.apply(lambda x: x[0]).replace("", np.nan)
    df_['subgrade'] = pd.to_numeric(g.apply(lambda x: x[1]), errors='coerce')

    # numeric
    num_cols = ['annual_income','loan_amount','debt_to_income_ratio','interest_rate','credit_score']
    for c in num_cols:
        if c in df_.columns:
            df_[c] = pd.to_numeric(df_[c], errors='coerce')

    # interactions
    df_['loan_to_income'] = df_['loan_amount'] / (df_['annual_income'] + 1e-6)
    df_['income_to_loan'] = df_['annual_income'] / (df_['loan_amount'] + 1e-6)
    df_['log_annual_income'] = np.log1p(df_['annual_income'].clip(lower=0))
    df_['log_loan_amount'] = np.log1p(df_['loan_amount'].clip(lower=0))
    df_['interest_x_loan'] = df_['interest_rate'] * df_['loan_amount']
    df_['interest_x_credit'] = df_['interest_rate'] * df_['credit_score']
    df_['dti_x_interest'] = df_['debt_to_income_ratio'] * df_['interest_rate']
    
    # Debt-to-Income related
    df_['remaining_income'] = df_['annual_income'] * (1 - df_['debt_to_income_ratio'])
    df_['dti_x_loan'] = df_['debt_to_income_ratio'] * df_['loan_amount']

    # Credit related
    df_['credit_x_income'] = df_['credit_score'] * df_['annual_income']
    
    # credit bucket
    if 'credit_score' in df_.columns:
        df_['credit_score_bucket'] = pd.cut(
            df_['credit_score'],
            bins=[0, 580, 670, 740, 800, 900],
            labels=['poor', 'fair', 'good', 'very_good', 'excellent']
        ).astype(object)

    # freqency encoding
    cat_cols_orig = ['loan_purpose','employment_status','education_level','marital_status','gender']
    for c in cat_cols_orig:
        if c in df_.columns:
            freq = df_[c].fillna('NA').value_counts(normalize=True)
            df_[f'{c}_freq'] = df_[c].fillna('NA').map(freq).astype(float)

    df_['missing_count'] = df_.isna().sum(axis=1)

    # grade one-hot
    if 'grade' in df_.columns:
        df_['grade'] = df_['grade'].astype(object)
        grade_dummies = pd.get_dummies(df_['grade'], prefix='grade', dummy_na=True)
        df_ = pd.concat([df_, grade_dummies], axis=1)

    return df_

train_fe = feature_engineer(train_df)
test_fe = feature_engineer(test_df.drop('id', axis=1)) # ä¿�æŒ� test_id å�•ç‹¬
test_ids = test_df['id']

train_y = train_fe[TARGET]

FEATURES = [c for c in train_fe.columns if c not in ['id', TARGET, 'grade_subgrade']]

test_fe = test_fe.reindex(columns=FEATURES)

print("Feature Engineering complete.")
print(f"Total features: {len(FEATURES)}")


print("Preparing data for Models...")
train_X_tree = train_fe[FEATURES].copy()
test_X_tree = test_fe[FEATURES].copy()

# CatBoost/LGBM/XGB's category dtype
cat_features_tree = []
num_features_tree = []

for c in FEATURES:
    if train_X_tree[c].dtype == 'object' or train_X_tree[c].dtype.name == 'category':
        cat_features_tree.append(c)
    else:
        num_features_tree.append(c)

# filling
for c in num_features_tree:
    med = train_X_tree[c].median()
    train_X_tree[c] = train_X_tree[c].fillna(med)
    test_X_tree[c] = test_X_tree[c].fillna(med)

for c in cat_features_tree:
    train_X_tree[c] = train_X_tree[c].fillna('NA').astype('category')
    test_X_tree[c] = test_X_tree[c].fillna('NA').astype('category')

# LGBM/XGB: cat.codes
train_X_lgbm_xgb = train_X_tree.copy()
test_X_lgbm_xgb = test_X_tree.copy()

for c in cat_features_tree:
    train_X_lgbm_xgb[c] = train_X_lgbm_xgb[c].cat.codes
    test_X_lgbm_xgb[c] = test_X_lgbm_xgb[c].cat.codes

# prepareï¼šPyTorch
print("Preparing data for PyTorch...")
train_X_nn = train_fe[FEATURES].copy()
test_X_nn = test_fe[FEATURES].copy()

cat_cols_nn = cat_features_tree 
num_cols_nn = num_features_tree

# LabelEncoder
cat_dims = []
for col in cat_cols_nn:
    # fill NA
    train_X_nn[col] = train_X_nn[col].fillna('NA')
    test_X_nn[col] = test_X_nn[col].fillna('NA')
    
    le = LabelEncoder()
    combined_series = pd.concat([train_X_nn[col], test_X_nn[col]]).astype(str)
    le.fit(combined_series)
    
    train_X_nn[col] = le.transform(train_X_nn[col].astype(str))
    test_X_nn[col] = le.transform(test_X_nn[col].astype(str))

    cardinality = len(le.classes_)
    cat_dims.append(cardinality)

scaler = StandardScaler()

for col in num_cols_nn:
    med = train_X_nn[col].median()
    train_X_nn[col] = train_X_nn[col].fillna(med)
    test_X_nn[col] = test_X_nn[col].fillna(med)

train_X_nn[num_cols_nn] = scaler.fit_transform(train_X_nn[num_cols_nn])
test_X_nn[num_cols_nn] = scaler.transform(test_X_nn[num_cols_nn])

# numpy (PyTorch Dataset
train_cat_nn_np = train_X_nn[cat_cols_nn].values
train_num_nn_np = train_X_nn[num_cols_nn].values
train_y_nn_np = train_y.values.astype(np.float32).reshape(-1, 1)

test_cat_nn_np = test_X_nn[cat_cols_nn].values
test_num_nn_np = test_X_nn[num_cols_nn].values

print("Preparing data for CatBoost...")

train_X_cb = train_fe[FEATURES].copy()
test_X_cb = test_fe[FEATURES].copy()

for c in cat_features_tree:
    train_X_cb[c] = train_X_cb[c].astype(str).replace('nan', 'Missing')
    test_X_cb[c] = test_X_cb[c].astype(str).replace('nan', 'Missing')

print("Data preparations complete.")


print("\n--- Starting Model: PyTorch MLP ---")

# define PyTorch Dataset
class LoanDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        # Distinguish between test data & train data
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_num[idx], self.y[idx]
        else:
            return self.X_cat[idx], self.X_num[idx]

# define PyTorch Model
class MLP(nn.Module):
    def __init__(self, cat_dims, n_num_features, emb_dim=16, layers=[200, 100, 50], dropout=0.3):
        super().__init__()
        
        # Embedding Layer
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) for cardinality in cat_dims
        ])
        
        # Embedding Dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # input dim
        n_emb_total = sum([emb_dim for _ in cat_dims])
        n_mlp_input = n_emb_total + n_num_features
        
        # MLP Layers
        mlp_layers = []
        in_dim = n_mlp_input
        for out_dim in layers:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.BatchNorm1d(out_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
            
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output Layer
        self.output_layer = nn.Linear(layers[-1], 1)

    def forward(self, x_cat, x_num):
        # Embedding
        x_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(x_emb, 1)
        x_emb = self.emb_dropout(x_emb)
        
        x = torch.cat([x_emb, x_num], 1)
        x = self.mlp(x)
        x = self.output_layer(x)
        return x

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {DEVICE}")

NN_EPOCHS = 50
NN_BATCH_SIZE = 1024
NN_LEARNING_RATE = 1e-3
NN_EMB_DIM = 20 # Embedding
NN_LAYERS = [256, 128, 64] # MLP
NN_DROPOUT = 0.4
NN_PATIENCE = 5 # Early Stop

def train_nn_model(model, train_loader, optimizer, criterion):
    model.train()
    for x_cat_batch, x_num_batch, y_batch in train_loader:
        x_cat_batch, x_num_batch, y_batch = x_cat_batch.to(DEVICE), x_num_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x_cat_batch, x_num_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

def validate_nn_model(model, val_loader, criterion):
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for x_cat_batch, x_num_batch, y_batch in val_loader:
            x_cat_batch, x_num_batch = x_cat_batch.to(DEVICE), x_num_batch.to(DEVICE)
            
            logits = model(x_cat_batch, x_num_batch)
            probs = torch.sigmoid(logits)
            
            val_preds.append(probs.cpu().numpy())
            val_targets.append(y_batch.numpy())
            
    val_preds = np.concatenate(val_preds).ravel()
    val_targets = np.concatenate(val_targets).ravel()
    auc = roc_auc_score(val_targets, val_preds)
    return auc

def predict_nn_model(model, test_loader):
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch_data in test_loader: 

            x_cat_batch = batch_data[0].to(DEVICE)
            x_num_batch = batch_data[1].to(DEVICE)
            logits = model(x_cat_batch, x_num_batch)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().numpy())
            
    test_preds = np.concatenate(test_preds).ravel()
    return test_preds

# K-Fold training
oof_preds_nn = np.zeros(len(train_fe))
test_preds_nn = np.zeros(len(test_fe))
test_dataset = LoanDataset(test_cat_nn_np, test_num_nn_np, y=None)
test_loader = DataLoader(test_dataset, batch_size=NN_BATCH_SIZE * 2, shuffle=False)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print("Training PyTorch model with K-Fold...")
pbar = tqdm(skf.split(train_num_nn_np, train_y), total=N_SPLITS, desc="NN Folds")
for fold, (tr_idx, val_idx) in enumerate(pbar):
    # print(f"  NN Fold {fold+1}/{N_SPLITS}")
    
    # create Dataset & DataLoader
    train_dataset = LoanDataset(train_cat_nn_np[tr_idx], train_num_nn_np[tr_idx], train_y_nn_np[tr_idx])
    val_dataset = LoanDataset(train_cat_nn_np[val_idx], train_num_nn_np[val_idx], train_y_nn_np[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=NN_BATCH_SIZE * 2, shuffle=False)
    
    # init model
    model = MLP(cat_dims, len(num_cols_nn), emb_dim=NN_EMB_DIM, layers=NN_LAYERS, dropout=NN_DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=NN_LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = -1
    patience_counter = 0
    best_model_weights = None

    for epoch in range(NN_EPOCHS):
        train_nn_model(model, train_loader, optimizer, criterion)
        val_auc = validate_nn_model(model, val_loader, criterion)
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_model_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= NN_PATIENCE:
                # print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # print(f"    Best Val AUC: {best_auc:.5f}")
    
    model.load_state_dict(best_model_weights)
    
    val_preds = predict_nn_model(model, val_loader)
    oof_preds_nn[val_idx] = val_preds
    
    fold_test_preds = predict_nn_model(model, test_loader)
    test_preds_nn += fold_test_preds / N_SPLITS

oof_auc = roc_auc_score(train_y, oof_preds_nn)
np.save('oof_preds_nn.npy', oof_preds_nn)
np.save('test_preds_nn.npy', test_preds_nn)
print(f"PyTorch MLP OOF AUC: {oof_auc:.5f}")


print("\n--- Starting Model 2: LightGBM ---")
best_params_lgbm = {'learning_rate': 0.0679, 'num_leaves': 141, 'max_depth': 4, 
                    'min_child_samples': 127, 'subsample': 0.9939, 
                    'colsample_bytree': 0.711, 'reg_alpha': 9.789, 'reg_lambda': 2.45}
best_params_lgbm.update({
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'verbosity': -1, 'seed': SEED, 'n_jobs': -1
})
print(f"Using pre-tuned LGBM Params: {best_params_lgbm}")


# Train LGBM (K-Fold)
def train_final_lgbm(params):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds_lgbm = np.zeros(len(train_X_lgbm_xgb))
    test_preds_lgbm = np.zeros(len(test_X_lgbm_xgb))
    
    print("Training final LGBM model with K-Fold...")
    pbar = tqdm(skf.split(train_X_lgbm_xgb, train_y), total=N_SPLITS, desc="LGBM Folds")
    for fold, (tr_idx, val_idx) in enumerate(pbar):
        # print(f"  LGBM Fold {fold+1}/{N_SPLITS}")
        X_tr, X_val = train_X_lgbm_xgb.iloc[tr_idx], train_X_lgbm_xgb.iloc[val_idx]
        y_tr, y_val = train_y.iloc[tr_idx], train_y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features_tree)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_features_tree)

        bst = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        val_preds = bst.predict(X_val, num_iteration=bst.best_iteration)
        oof_preds_lgbm[val_idx] = val_preds
        
        fold_test_preds = bst.predict(test_X_lgbm_xgb, num_iteration=bst.best_iteration)
        test_preds_lgbm += fold_test_preds / N_SPLITS

    oof_auc = roc_auc_score(train_y, oof_preds_lgbm)
    np.save('oof_preds_lgbm.npy', oof_preds_lgbm)
    np.save('test_preds_lgbm.npy', test_preds_lgbm)
    print(f"LGBM OOF AUC: {oof_auc:.5f}")
    return oof_preds_lgbm, test_preds_lgbm

oof_preds_lgbm, test_preds_lgbm = train_final_lgbm(best_params_lgbm)


pos = train_y.sum()
neg = len(train_y) - pos
scale_pos_weight = float(neg) / float(pos + 1e-6)

print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

def objective_xgb(trial):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx_xgb, val_idx_xgb = next(splitter.split(train_X_tree, train_y))

    params_xgb_search = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 20.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    
    params_xgb_static = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': SEED,
        'n_jobs': -1, 'device': "cuda" if torch.cuda.is_available() else "cpu",
        'tree_method': "hist", 'enable_categorical': True,
        'scale_pos_weight': scale_pos_weight
    }
    params_xgb_static.update(params_xgb_search)

    X_tr, X_val = train_X_tree.iloc[tr_idx_xgb], train_X_tree.iloc[val_idx_xgb]
    y_tr, y_val = train_y.iloc[tr_idx_xgb], train_y.iloc[val_idx_xgb]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    bst = xgb.train(
        params_xgb_static,
        dtrain,
        evals=[(dval, 'val')],
        num_boost_round=5000,
        early_stopping_rounds=150,
        verbose_eval=False
    )
    
    preds = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
    auc = roc_auc_score(y_val, preds)
    return auc

N_TRIALS_XGB = 20
print(f"\n Running Optuna for XGB...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS_XGB, show_progress_bar=True)

best_params_xgb = study_xgb.best_params
print(f"XGBoost best AUC: {study_xgb.best_value:.6f}")
print(f"XGBoost best params: {best_params_xgb}")

best_params_xgb.update({
    'scale_pos_weight': scale_pos_weight,
    'objective': 'binary:logistic', 
    'eval_metric': 'auc', 
    'seed': SEED,
    'n_jobs': -1, 
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'tree_method': "hist"
})

print(f"Using XGB Params: {best_params_xgb}")

# train xgb
def train_final_xgb(params):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds_xgb = np.zeros(len(train_X_tree))
    test_preds_xgb = np.zeros(len(test_X_tree))
    
    dtest = xgb.DMatrix(test_X_tree, enable_categorical=True)

    print("Training final XGB model with K-Fold...")
    pbar = tqdm(skf.split(train_X_lgbm_xgb, train_y), total=N_SPLITS, desc="XGB Folds")
    
    for fold, (tr_idx, val_idx) in enumerate(pbar):
        X_tr, X_val = train_X_tree.iloc[tr_idx], train_X_tree.iloc[val_idx]
        y_tr, y_val = train_y.iloc[tr_idx], train_y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        
        bst = xgb.train(
            params,
            dtrain,
            evals=[(dval, 'val')],
            num_boost_round=10000, 
            early_stopping_rounds=300,
            verbose_eval=False
        )
        
        val_preds = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
        oof_preds_xgb[val_idx] = val_preds
        
        fold_test_preds = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
        test_preds_xgb += fold_test_preds / N_SPLITS

    oof_auc = roc_auc_score(train_y, oof_preds_xgb)
    np.save('oof_preds_xgb.npy', oof_preds_xgb)
    np.save('test_preds_xgb.npy', test_preds_xgb)
    print(f"XGB OOF AUC: {oof_auc:.5f}")
    return oof_preds_xgb, test_preds_xgb

oof_preds_xgb, test_preds_xgb = train_final_xgb(best_params_xgb)


print("\n--- Starting Model 4: CatBoost ---")

# learn from xgb strategy
pos = train_y.sum()
neg = len(train_y) - pos
scale_pos_weight = float(neg) / float(pos + 1e-6)
print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

for c in cat_features_tree:
    train_X_cb[c] = train_X_cb[c].fillna('Missing').astype(str)
    test_X_cb[c] = test_X_cb[c].fillna('Missing').astype(str)

def objective_cat(trial):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx_cb, val_idx_cb = next(splitter.split(train_X_cb, train_y))
    
    # define Hyerparameter Space
    params_cat_search = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'depth': trial.suggest_int('depth', 5, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_categorical('border_count', [128, 254]),
    }
    
    # Static Params
    params_cat_static = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': SEED,
        'task_type': "GPU" if torch.cuda.is_available() else "CPU",
        'verbose': False,
        'allow_writing_files': False,
        'scale_pos_weight': scale_pos_weight,
        'iterations': 5000
    }
    params_cat_static.update(params_cat_search)

    # CatBoost use train_X_cb
    X_tr, X_val = train_X_cb.iloc[tr_idx_cb], train_X_cb.iloc[val_idx_cb]
    y_tr, y_val = train_y.iloc[tr_idx_cb], train_y.iloc[val_idx_cb]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features_tree)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_tree)

    model = CatBoostClassifier(**params_cat_static)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=150,
        use_best_model=True,
        verbose=False
    )
    
    preds = model.predict_proba(val_pool)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

N_TRIALS_CAT = 20
print(f"\n Running Optuna for catboost...")
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=N_TRIALS_CAT, show_progress_bar=True)

# æ‰“å�°æœ€ä½³å�‚æ•°
best_params_cb = study_cat.best_params
print(f" CatBoost best AUC: {study_cat.best_value:.6f}")
print(f" CatBoost best params: {best_params_cb}")

best_params_cb.update({
    'iterations': 10000,
    'scale_pos_weight': scale_pos_weight,
    'loss_function': 'Logloss', 
    'eval_metric': 'AUC', 
    'random_seed': SEED,
    'thread_count': -1, 
    'verbose': 0,
    'allow_writing_files': False,
    'task_type': 'GPU' if torch.cuda.is_available() else 'CPU'
})

# define train
def train_final_cb(params):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    # oof_preds_cb = np.zeros(len(train_X_tree))
    # test_preds_cb = np.zeros(len(test_X_tree))
    oof_preds_cb = np.zeros(len(train_X_cb))
    test_preds_cb = np.zeros(len(test_X_cb))
    
    # test_pool = Pool(X_test_cb, cat_features=cat_features_tree)
    test_pool = Pool(test_X_cb, cat_features=cat_features_tree)

    print("Training final CatBoost model with K-Fold...")
    # pbar = tqdm(skf.split(train_X_lgbm_xgb, train_y), total=N_SPLITS, desc="CatBoost Folds")
    pbar = tqdm(skf.split(train_X_cb, train_y), total=N_SPLITS, desc="CatBoost Folds")
    for fold, (tr_idx, val_idx) in enumerate(pbar):
        # cut data
        # X_tr, X_val = train_X_tree.iloc[tr_idx].copy(), train_X_tree.iloc[val_idx].copy()
        X_tr, X_val = train_X_cb.iloc[tr_idx], train_X_cb.iloc[val_idx]
        y_tr, y_val = train_y.iloc[tr_idx], train_y.iloc[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features_tree)
        val_pool = Pool(X_val, y_val, cat_features=cat_features_tree)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=300, # lower
            use_best_model=True,
            verbose=False
        )
        
        val_preds = model.predict_proba(val_pool)[:, 1]
        oof_preds_cb[val_idx] = val_preds
        
        fold_test_preds = model.predict_proba(test_pool)[:, 1]
        test_preds_cb += fold_test_preds / N_SPLITS

    oof_auc = roc_auc_score(train_y, oof_preds_cb)
    np.save('oof_preds_cb.npy', oof_preds_cb)
    np.save('test_preds_cb.npy', test_preds_cb)
    print(f"CatBoost OOF AUC: {oof_auc:.5f}")
    return oof_preds_cb, test_preds_cb

oof_preds_cb, test_preds_cb = train_final_cb(best_params_cb)


import numpy as np
import os

if os.path.exists('oof_preds_lgbm.npy'):
    oof_preds_lgbm = np.load('oof_preds_lgbm.npy')
    test_preds_lgbm = np.load('test_preds_lgbm.npy')
    
    oof_preds_xgb = np.load('oof_preds_xgb.npy')
    test_preds_xgb = np.load('test_preds_xgb.npy')
    
    oof_preds_cb = np.load('oof_preds_cb.npy')
    test_preds_cb = np.load('test_preds_cb.npy')
    
    oof_preds_nn = np.load('oof_preds_nn.npy')
    test_preds_nn = np.load('test_preds_nn.npy')
    
    test_ids = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')['id']
    TARGET = 'loan_paid_back'
    
    print("All predictions successfully loaded!")
else:
    print("Missing file!")


from scipy.optimize import minimize

print("\n--- Starting Automatic Weight Optimization ---")

oof_matrix = np.column_stack([oof_preds_lgbm, oof_preds_xgb, oof_preds_cb, oof_preds_nn])
test_matrix = np.column_stack([test_preds_lgbm, test_preds_xgb, test_preds_cb, test_preds_nn])

y_true = train_y

# define optimization function
def get_auc_score(weights):
    # Normalized weight
    weights = np.array(weights)
    weights = np.maximum(weights, 0)
    if np.sum(weights) == 0: 
        return 1.0
    weights = weights / np.sum(weights)
    
    final_pred = np.dot(oof_matrix, weights)

    # minimize :- 
    return -roc_auc_score(y_true, final_pred)

# init weights
init_weights = [0.25, 0.25, 0.25, 0.25] 

# Search using Nelder Mead algorithm
result = minimize(get_auc_score, init_weights, method='Nelder-Mead', tol=1e-6)

best_weights = np.maximum(result.x, 0)
best_weights = best_weights / np.sum(best_weights)
best_auc = -result.fun

print(f"\n Optimized Blend OOF AUC: {best_auc:.6f}")
print("-" * 40)
print("Best Weights Distribution:")
model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'PyTorch MLP']
for name, weight in zip(model_names, best_weights):
    print(f"  {name:12s}: {weight:.4f}")
print("-" * 40)

# final optimized submission
final_test_pred = np.dot(test_matrix, best_weights)
# create submission.csv
submission_df = pd.DataFrame({'id': test_ids, TARGET: final_test_pred})
submission_df.to_csv('submission.csv', index=False)
print("submission.csv created successfully!")

