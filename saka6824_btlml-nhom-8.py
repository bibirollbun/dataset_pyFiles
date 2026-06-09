import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight

import torch
import torch.nn as nn
import torch.optim as optim


RANDOM_SEED = 42
NFOLDS = 10
INPUT_DIR = '/kaggle/input/icr-identify-age-related-conditions'

def seed_everything(seed=RANDOM_SEED):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

seed_everything()

# Cấu hình thiết bị 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#HÀM TÍNH ĐIỂM
def balanced_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    nc = np.bincount(y_true)
    logloss = (-1/nc[0]*(np.sum(np.where(y_true==0,1,0) * np.log(1-y_pred))) - 
               1/nc[1]*(np.sum(np.where(y_true!=0,1,0) * np.log(y_pred)))) / 2
    return logloss


#1. CHUẨN BỊ DỮ LIỆU 
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
greeks = pd.read_csv(os.path.join(INPUT_DIR, 'greeks.csv'))

# Xử lý EJ
le = LabelEncoder()
full_ej = pd.concat([train['EJ'], test['EJ']], axis=0)
le.fit(full_ej)
train['EJ'] = le.transform(train['EJ'])
test['EJ'] = le.transform(test['EJ'])

TARGET = 'Class'
ID_COL = 'Id'
features = [c for c in train.columns if c not in [ID_COL, TARGET]]

# Điền dữ liệu thiếu
imputer = SimpleImputer(strategy='median')
train[features] = imputer.fit_transform(train[features])
test[features] = imputer.transform(test[features])

# Scaling
scaler = PowerTransformer(method='yeo-johnson')
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

X = train[features].values
y = train[TARGET].values.astype(int)
X_test = test[features].values

# Tính toán trọng số class
pos_weight = np.sum(y == 0) / np.sum(y == 1)

# Split Strategy
train_greeks = train.merge(greeks[['Id', 'Alpha']], on='Id', how='left')
split_target = train_greeks['Alpha']
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_SEED)

# Mảng lưu kết quả
oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
oof_nn  = np.zeros(len(train))
pred_lgb = np.zeros(len(test))
pred_cat = np.zeros(len(test))
pred_nn  = np.zeros(len(test))


#ĐỊNH NGHĨA MODEL PYTORCH NN
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(), 
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))




#2. VÒNG LẶP TRAINING 
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, split_target)):
    print(f"\n========== FOLD {fold+1}/{NFOLDS} =========")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    #a) LightGBM
    print("Training LightGBM...")
    model_lgb = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', boosting_type='gbdt',
        learning_rate=0.02, num_leaves=20, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5, scale_pos_weight=pos_weight,
        verbosity=-1, seed=RANDOM_SEED, n_estimators=1000
    )
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    pred_lgb += model_lgb.predict_proba(X_test)[:, 1] / NFOLDS

    #b) CatBoost
    print("Training CatBoost...")
    model_cat = CatBoostClassifier(
        loss_function='Logloss', iterations=1000, learning_rate=0.03,
        depth=5, l2_leaf_reg=3, scale_pos_weight=pos_weight,
        random_seed=RANDOM_SEED, allow_writing_files=False, verbose=0
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    pred_cat += model_cat.predict_proba(X_test)[:, 1] / NFOLDS

    #c) PyTorch Neural Network
    print("Training PyTorch NN...")
    
    # Chuyển dữ liệu sang Tensor
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test.copy()).to(device)

    # Khởi tạo model
    model = SimpleNN(X_tr.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop thủ công
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(100): 
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_tr_t)
        
        # Weighted Loss thủ công
        weight = torch.ones_like(y_tr_t)
        weight[y_tr_t == 1] = pos_weight 
        loss_fn = nn.BCELoss(weight=weight)
        loss = loss_fn(y_pred, y_tr_t)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            # Tính balanced log loss trên numpy để so sánh chuẩn xác
            val_loss = balanced_log_loss(y_val, val_pred.cpu().numpy().flatten())
            
        # Early Stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break 
    
    # Load model tốt nhất để dự đoán
    model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
    model.eval()
    with torch.no_grad():
        oof_nn[val_idx] = model(X_val_t).cpu().numpy().flatten()
        pred_nn += model(X_test_t).cpu().numpy().flatten() / NFOLDS



#3. ĐÁNH GIÁ & SUBMIT 
print("\n========== EVALUATION ==========")
print(f"LGBM Score:      {balanced_log_loss(y, oof_lgb):.5f}")
print(f"CatBoost Score:  {balanced_log_loss(y, oof_cat):.5f}")
print(f"PyTorch NN Score:{balanced_log_loss(y, oof_nn):.5f}")

# Trọng số Ensemble
w_lgb, w_cat, w_nn = 0.35, 0.45, 0.20 
final_oof = (w_lgb * oof_lgb) + (w_cat * oof_cat) + (w_nn * oof_nn)
print(f"=> FINAL ENSEMBLE SCORE: {balanced_log_loss(y, final_oof):.5f}")

final_test_pred = (w_lgb * pred_lgb) + (w_cat * pred_cat) + (w_nn * pred_nn)
final_test_pred = np.clip(final_test_pred, 1e-15, 1 - 1e-15)

submission = pd.DataFrame({ID_COL: test[ID_COL], 'class_0': 1 - final_test_pred, 'class_1': final_test_pred})
submission.to_csv('submission.csv', index=False)

