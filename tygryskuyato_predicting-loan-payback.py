import pandas as pd
import numpy as np
import warnings
import gc
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

# Import cÃ¡c model SOTA
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

# Neural Networks
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Optuna cho auto-tuning
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ====================================================
# 1. CONFIGURATION
# ====================================================
class Config:
    SEED = 42
    N_FOLDS = 10
    N_ESTIMATORS = 4000
    EARLY_STOPPING = 250
    LEARNING_RATE = 0.02
    OPTUNA_TRIALS = 20  # Sá»‘ láº§n thá»­ optuna (giáº£m náº¿u cháº¡y lÃ¢u)
    USE_GPU = torch.cuda.is_available()
    DEVICE = 'cuda' if USE_GPU else 'cpu'
    
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

print(f"ğŸ–¥ï¸�  Device: {Config.DEVICE.upper()}")

# ====================================================
# 2. Táº¢I Dá»® LIá»†U
# ====================================================
def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
    sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
    return train, test, sample_sub

train, test, sample_sub = load_data()
print(f"ğŸ“Š Train: {train.shape}, Test: {test.shape}")

# ====================================================
# 3. ADVANCED FEATURE ENGINEERING
# ====================================================
def create_advanced_features(df):
    """60+ features engineering"""
    
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
        df['income_per_month'] = df['annual_income'] / 12
        df['loan_burden'] = df['loan_amount'] / (df['annual_income'] / 12 + 1)
    
    if 'loan_amount' in df.columns and 'loan_term' in df.columns:
        df['monthly_payment_est'] = df['loan_amount'] / (df['loan_term'] + 1)
        df['loan_term_years'] = df['loan_term'] / 12
    
    if 'current_balance' in df.columns and 'total_credit_limit' in df.columns:
        df['credit_utilization'] = df['current_balance'] / (df['total_credit_limit'] + 1)
        df['available_credit'] = df['total_credit_limit'] - df['current_balance']
        df['credit_utilization_sq'] = df['credit_utilization'] ** 2
        df['is_maxed_out'] = (df['credit_utilization'] > 0.9).astype(int)
    
    if 'number_of_credit_accounts' in df.columns:
        if 'total_credit_limit' in df.columns:
            df['credit_per_account'] = df['total_credit_limit'] / (df['number_of_credit_accounts'] + 1)
        if 'current_balance' in df.columns:
            df['balance_per_account'] = df['current_balance'] / (df['number_of_credit_accounts'] + 1)
    
    if 'interest_rate' in df.columns:
        if 'loan_amount' in df.columns and 'loan_term' in df.columns:
            df['total_interest'] = df['loan_amount'] * (df['interest_rate'] / 100) * (df['loan_term'] / 12)
            df['total_payment'] = df['loan_amount'] + df['total_interest']
            df['interest_burden'] = df['total_interest'] / (df['loan_amount'] + 1)
        
        df['is_high_rate'] = (df['interest_rate'] > df['interest_rate'].median()).astype(int)
    
    if 'age' in df.columns:
        df['age_squared'] = df['age'] ** 2
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=[0,1,2,3,4])
        df['age_group'] = df['age_group'].astype(int)
        
        if 'annual_income' in df.columns:
            df['income_per_age'] = df['annual_income'] / (df['age'] + 1)
        
        if 'number_of_credit_accounts' in df.columns:
            df['accounts_per_age'] = df['number_of_credit_accounts'] / (df['age'] + 1)
    
    if 'employment_length' in df.columns:
        if 'annual_income' in df.columns:
            df['income_per_emp_year'] = df['annual_income'] / (df['employment_length'] + 1)
        
        if 'age' in df.columns:
            df['emp_age_ratio'] = df['employment_length'] / (df['age'] + 1)
    
    if 'credit_score' in df.columns:
        df['credit_score_sq'] = df['credit_score'] ** 2
        df['credit_score_log'] = np.log1p(df['credit_score'])
        
        if 'loan_amount' in df.columns:
            df['risk_adjusted_loan'] = df['loan_amount'] / (df['credit_score'] + 1)
    
    if 'loan_amount' in df.columns and 'credit_score' in df.columns:
        df['loan_credit_interaction'] = df['loan_amount'] * df['credit_score']
    
    log_cols = ['loan_amount', 'annual_income', 'total_credit_limit', 'current_balance']
    for col in log_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
            df[f'{col}_sqrt'] = np.sqrt(df[col])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 5:
        df['feature_sum'] = df[numeric_cols].sum(axis=1)
        df['feature_mean'] = df[numeric_cols].mean(axis=1)
        df['feature_std'] = df[numeric_cols].std(axis=1)
    
    return df

print("\nğŸ”§ Feature Engineering...")
train = create_advanced_features(train)
test = create_advanced_features(test)

# Xá»­ lÃ½ categorical
cat_cols = [col for col in train.columns if train[col].dtype == 'object']
for col in cat_cols:
    le = LabelEncoder()
    combined = list(train[col].astype(str)) + list(test[col].astype(str))
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

train = train.fillna(-999)
test = test.fillna(-999)

X = train.drop(['id', 'loan_paid_back'], axis=1)
y = train['loan_paid_back']
X_test = test.drop(['id'], axis=1)

print(f"âœ… Final shape: {X.shape}")

# ====================================================
# 4. NEURAL NETWORK DEFINITION
# ====================================================
class TabularNN(nn.Module):
    """Deep Neural Network cho tabular data"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(TabularNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_neural_network(X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
    """Train Neural Network"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(Config.DEVICE)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(Config.DEVICE)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(Config.DEVICE)
    
    # Model
    model = TabularNN(X_train.shape[1]).to(Config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    
    # Training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).cpu().numpy()
        
        auc = roc_auc_score(y_val, val_pred)
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return model, scaler, best_auc

# ====================================================
# 5. OPTUNA HYPERPARAMETER TUNING
# ====================================================
def optuna_lgb_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective cho LightGBM"""
    params = {
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'num_leaves': trial.suggest_int('num_leaves', 64, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),
        'objective': 'binary',
        'metric': 'auc',
        'random_state': Config.SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    
    pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, pred)

print("\nğŸ”� Ä�ang cháº¡y Optuna Hyperparameter Tuning...")
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.SEED)
train_idx, val_idx = next(kf.split(X, y))
X_train_opt, X_val_opt = X.iloc[train_idx], X.iloc[val_idx]
y_train_opt, y_val_opt = y.iloc[train_idx], y.iloc[val_idx]

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=Config.SEED))
study.optimize(lambda trial: optuna_lgb_objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt), 
               n_trials=Config.OPTUNA_TRIALS, show_progress_bar=False)

best_lgb_params = study.best_params
best_lgb_params.update({
    'n_estimators': Config.N_ESTIMATORS,
    'objective': 'binary',
    'metric': 'auc',
    'random_state': Config.SEED,
    'n_jobs': -1,
    'verbose': -1
})

print(f"âœ… Best LightGBM params found! AUC: {study.best_value:.6f}")

# ====================================================
# 6. Ä�á»ŠNH NGHÄ¨A CÃ�C MODEL
# ====================================================
cat_params = {
    'iterations': Config.N_ESTIMATORS,
    'learning_rate': 0.02,
    'depth': 10,
    'l2_leaf_reg': 8,
    'border_count': 254,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': Config.SEED,
    'verbose': 0,
    'allow_writing_files': False,
    'early_stopping_rounds': Config.EARLY_STOPPING
}

xgb_params = {
    'n_estimators': Config.N_ESTIMATORS,
    'learning_rate': 0.02,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.1,
    'reg_alpha': 0.5,
    'reg_lambda': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': Config.SEED,
    'n_jobs': -1,
    'tree_method': 'hist'
}

et_params = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': Config.SEED,
    'n_jobs': -1
}

hgb_params = {
    'max_iter': 1000,
    'learning_rate': 0.05,
    'max_depth': 15,
    'min_samples_leaf': 20,
    'l2_regularization': 1.0,
    'random_state': Config.SEED
}

# ====================================================
# 7. LAYER 1: BASE MODELS TRAINING
# ====================================================
kf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_et = np.zeros(len(X))
oof_hgb = np.zeros(len(X))
oof_nn = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_et = np.zeros(len(X_test))
test_hgb = np.zeros(len(X_test))
test_nn = np.zeros(len(X_test))

print("\n" + "="*60)
print("ğŸš€ LAYER 1: TRAINING BASE MODELS (6 Models)")
print("="*60)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'='*60}")
    print(f"ğŸ“� FOLD {fold+1}/{Config.N_FOLDS}")
    print(f"{'='*60}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 1. LightGBM (Optuna-tuned)
    print("ğŸ”¹ LightGBM (Optuna-tuned)...")
    lgb = LGBMClassifier(**best_lgb_params)
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    oof_lgb[val_idx] = lgb.predict_proba(X_val)[:, 1]
    test_lgb += lgb.predict_proba(X_test)[:, 1] / Config.N_FOLDS
    print(f"  â†’ AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.6f}")
    
    # 2. CatBoost
    print("ğŸ”¹ CatBoost...")
    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
    test_cat += cat.predict_proba(X_test)[:, 1] / Config.N_FOLDS
    print(f"  â†’ AUC: {roc_auc_score(y_val, oof_cat[val_idx]):.6f}")
    
    # 3. XGBoost
    print("ğŸ”¹ XGBoost...")
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
    test_xgb += xgb.predict_proba(X_test)[:, 1] / Config.N_FOLDS
    print(f"  â†’ AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.6f}")
    
    # 4. ExtraTrees
    print("ğŸ”¹ ExtraTrees...")
    et = ExtraTreesClassifier(**et_params)
    et.fit(X_train, y_train)
    oof_et[val_idx] = et.predict_proba(X_val)[:, 1]
    test_et += et.predict_proba(X_test)[:, 1] / Config.N_FOLDS
    print(f"  â†’ AUC: {roc_auc_score(y_val, oof_et[val_idx]):.6f}")
    
    # 5. HistGradientBoosting
    print("ğŸ”¹ HistGradientBoosting...")
    hgb = HistGradientBoostingClassifier(**hgb_params)
    hgb.fit(X_train, y_train)
    oof_hgb[val_idx] = hgb.predict_proba(X_val)[:, 1]
    test_hgb += hgb.predict_proba(X_test)[:, 1] / Config.N_FOLDS
    print(f"  â†’ AUC: {roc_auc_score(y_val, oof_hgb[val_idx]):.6f}")
    
    # 6. Neural Network
    print("ğŸ”¹ Neural Network...")
    nn_model, scaler, nn_auc = train_neural_network(X_train, y_train, X_val, y_val, epochs=50)
    
    # Predictions
    nn_model.eval()
    with torch.no_grad():
        X_val_scaled = scaler.transform(X_val)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(Config.DEVICE)
        oof_nn[val_idx] = nn_model(X_val_tensor).cpu().numpy().flatten()
        
        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(Config.DEVICE)
        test_nn += nn_model(X_test_tensor).cpu().numpy().flatten() / Config.N_FOLDS
    
    print(f"  â†’ AUC: {nn_auc:.6f}")
    
    gc.collect()
    if Config.USE_GPU:
        torch.cuda.empty_cache()

# ====================================================
# 8. PSEUDO-LABELING
# ====================================================
print("\n" + "="*60)
print("ğŸ”„ PSEUDO-LABELING")
print("="*60)

# Láº¥y predictions trung bÃ¬nh tá»« táº¥t cáº£ models
pseudo_preds = (test_lgb + test_cat + test_xgb + test_et + test_hgb + test_nn) / 6

# Chá»�n samples cÃ³ confidence cao (>0.9 hoáº·c <0.1)
high_conf_mask = (pseudo_preds > 0.9) | (pseudo_preds < 0.1)
pseudo_labels = (pseudo_preds > 0.5).astype(int)

print(f"ğŸ“Š Sá»‘ samples cÃ³ confidence cao: {high_conf_mask.sum()} / {len(X_test)}")

if high_conf_mask.sum() > 100:  # Chá»‰ dÃ¹ng náº¿u cÃ³ Ä‘á»§ samples
    X_pseudo = X_test[high_conf_mask]
    y_pseudo = pseudo_labels[high_conf_mask]
    
    # Káº¿t há»£p vá»›i training data
    X_combined = pd.concat([X, X_pseudo], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y, pd.Series(y_pseudo)], axis=0).reset_index(drop=True)
    
    print(f"âœ… Training data má»›i: {X_combined.shape}")
    
    # Retrain LightGBM vá»›i pseudo-labels
    print("ğŸ”„ Retraining vá»›i pseudo-labels...")
    oof_lgb_pseudo = np.zeros(len(X))
    test_lgb_pseudo = np.zeros(len(X_test))
    
    kf_pseudo = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf_pseudo.split(X, y)):
        # ThÃªm pseudo samples vÃ o training
        train_idx_combined = np.concatenate([train_idx, np.arange(len(X), len(X_combined))])
        
        X_train_p = X_combined.iloc[train_idx_combined]
        y_train_p = y_combined.iloc[train_idx_combined]
        X_val_p = X.iloc[val_idx]
        y_val_p = y.iloc[val_idx]
        
        lgb_p = LGBMClassifier(**best_lgb_params)
        lgb_p.fit(X_train_p, y_train_p, eval_set=[(X_val_p, y_val_p)], callbacks=[])
        
        oof_lgb_pseudo[val_idx] = lgb_p.predict_proba(X_val_p)[:, 1]
        test_lgb_pseudo += lgb_p.predict_proba(X_test)[:, 1] / 5
    
    print(f"  â†’ Pseudo-label AUC: {roc_auc_score(y, oof_lgb_pseudo):.6f}")
else:
    print("âš ï¸�  KhÃ´ng Ä‘á»§ high-confidence samples, bá»� qua pseudo-labeling")
    test_lgb_pseudo = test_lgb.copy()
    oof_lgb_pseudo = oof_lgb.copy()

# ====================================================
# 9. LAYER 2: STACKING META-LEARNER
# ====================================================
print("\n" + "="*60)
print("ğŸ�—ï¸�  LAYER 2: STACKING META-LEARNER")
print("="*60)

# Táº¡o meta-features tá»« OOF predictions
meta_train = np.column_stack([oof_lgb, oof_cat, oof_xgb, oof_et, oof_hgb, oof_nn, oof_lgb_pseudo])
meta_test = np.column_stack([test_lgb, test_cat, test_xgb, test_et, test_hgb, test_nn, test_lgb_pseudo])

print(f"ğŸ“Š Meta features shape: {meta_train.shape}")

# Train meta-learner (Logistic Regression)
meta_oof = np.zeros(len(meta_train))
meta_test_preds = np.zeros(len(meta_test))

kf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

for fold, (train_idx, val_idx) in enumerate(kf_meta.split(meta_train, y)):
    X_train_meta = meta_train[train_idx]
    y_train_meta = y.iloc[train_idx]
    X_val_meta = meta_train[val_idx]
    
    meta_model = LogisticRegression(C=0.1, max_iter=1000, random_state=Config.SEED)
    meta_model.fit(X_train_meta, y_train_meta)
    
    meta_oof[val_idx] = meta_model.predict_proba(X_val_meta)[:, 1]
    meta_test_preds += meta_model.predict_proba(meta_test)[:, 1] / 5

meta_auc = roc_auc_score(y, meta_oof)
print(f"ğŸ�¯ Meta-learner AUC: {meta_auc:.6f}")

# ====================================================
# 10. Káº¾T QUáº¢ Tá»”NG Há»¢P
# ====================================================
print("\n" + "="*60)
print("ğŸ“Š Káº¾T QUáº¢ LAYER 1 (Base Models)")
print("="*60)
print(f"LightGBM (Optuna):    {roc_auc_score(y, oof_lgb):.6f}")
print(f"CatBoost:             {roc_auc_score(y, oof_cat):.6f}")
print(f"XGBoost:              {roc_auc_score(y, oof_xgb):.6f}")
print(f"ExtraTrees:           {roc_auc_score(y, oof_et):.6f}")
print(f"HistGradientBoost:    {roc_auc_score(y, oof_hgb):.6f}")
print(f"Neural Network:       {roc_auc_score(y, oof_nn):.6f}")
print(f"Pseudo-labeled LGB:   {roc_auc_score(y, oof_lgb_pseudo):.6f}")

print(f"\n{'='*60}")
print(f"ğŸ�† LAYER 2 META-LEARNER AUC: {meta_auc:.6f}")
print(f"{'='*60}")

# ====================================================
# 11. LÆ¯U Káº¾T QUáº¢
# ====================================================
submission = pd.DataFrame({
    'id': sample_sub['id'],
    'loan_paid_back': meta_test_preds
})

submission.to_csv('submission_ultimate_stacking.csv', index=False)

print("\nâœ… Ä�Ã£ lÆ°u: submission_ultimate_stacking.csv")
print(f"\nğŸ“Š Thá»‘ng kÃª:")
print(f"   Min:  {meta_test_preds.min():.6f}")
print(f"   Max:  {meta_test_preds.max():.6f}")
print(f"   Mean: {meta_test_preds.mean():.6f}")

