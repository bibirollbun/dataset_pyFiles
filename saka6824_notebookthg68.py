import os
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.utils import class_weight

RANDOM_SEED = 42
NFOLDS = 10
INPUT_DIR = '/kaggle/input/icr-identify-age-related-conditions'

def seed_everything(seed=RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()

#HÀM ĐO ĐIỂM (BALANCED LOG LOSS)
def balanced_log_loss(y_true, y_pred):
    # Giới hạn giá trị dự đoán để tránh log(0)
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    nc = np.bincount(y_true)
    # Công thức Balanced Log Loss của cuộc thi
    logloss = (-1/nc[0]*(np.sum(np.where(y_true==0,1,0) * np.log(1-y_pred))) - 
               1/nc[1]*(np.sum(np.where(y_true!=0,1,0) * np.log(y_pred)))) / 2
    return logloss

#1. Đọc dữ liệu input
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
greeks = pd.read_csv(os.path.join(INPUT_DIR, 'greeks.csv'))

# Xử lý biến phân loại 'EJ'
le = LabelEncoder()
# Fit trên tập hợp cả train và test để tránh lỗi unseen label
full_ej = pd.concat([train['EJ'], test['EJ']], axis=0)
le.fit(full_ej)
train['EJ'] = le.transform(train['EJ'])
test['EJ'] = le.transform(test['EJ'])

# Xác định Features và Target
TARGET = 'Class'
ID_COL = 'Id'
features = [c for c in train.columns if c not in [ID_COL, TARGET]]

# Điền dữ liệu thiếu (Imputation)
imputer = SimpleImputer(strategy='median')
train[features] = imputer.fit_transform(train[features])
test[features] = imputer.transform(test[features])

# Scaling - Sử dụng PowerTransformer cho Neural Net
# Yeo-Johnson hỗ trợ cả số âm và dương, giúp dữ liệu phân phối chuẩn hơn
scaler = PowerTransformer(method='yeo-johnson')
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

X = train[features].values
y = train[TARGET].values.astype(int)
X_test = test[features].values

#Class Weights (cho Neural Net)
class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y), y=y
)
class_weights_dict = dict(enumerate(class_weights_vals))
print(f"Class Weights calculated: {class_weights_dict}")

#Tỷ lệ scale_pos_weight cho GBDT
pos_weight = np.sum(y == 0) / np.sum(y == 1)

#SPLIT (DỰA TRÊN ALPHA)
# Merge Alpha vào train tạm thời để split, sau đó bỏ đi
train_greeks = train.merge(greeks[['Id', 'Alpha']], on='Id', how='left')
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_SEED)
#split dựa trên Alpha thay vì Class để đảm bảo cân bằng tốt hơn
split_target = train_greeks['Alpha']

#Mảng lưu kq
oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
oof_nn  = np.zeros(len(train))

pred_lgb = np.zeros(len(test))
pred_cat = np.zeros(len(test))
pred_nn  = np.zeros(len(test))

#2. Training
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, split_target)):
    print(f"\n========== FOLD {fold+1}/{NFOLDS} =========")
    
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
 
    # a) LightBGM
    print("Training LightGBM...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 20,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': pos_weight,
        'verbosity': -1,
        'seed': RANDOM_SEED
    }
    
    model_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=1000)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    pred_lgb += model_lgb.predict_proba(X_test)[:, 1] / NFOLDS

    # b) CatBoost
    print("Training CatBoost...")
    cat_params = {
        'loss_function': 'Logloss',
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 5,
        'l2_leaf_reg': 3,
        'scale_pos_weight': pos_weight, # Xử lý mất cân bằng
        'random_seed': RANDOM_SEED,
        'allow_writing_files': False,
        'verbose': 0
    }
    
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )
    
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    pred_cat += model_cat.predict_proba(X_test)[:, 1] / NFOLDS

    # c) Neural Network
    print("Training Neural Network...")
    
    def build_model(input_dim):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='swish'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='swish'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['AUC'])
        return model

    model_nn_fold = build_model(X_tr.shape[1])
    
    # Callbacks
    es = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_sched = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    model_nn_fold.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es, lr_sched],
        class_weight=class_weights_dict, 
        verbose=0
    )
    
    oof_nn[val_idx] = model_nn_fold.predict(X_val).flatten()
    pred_nn += model_nn_fold.predict(X_test).flatten() / NFOLDS

#3. Đánh giá kq
print("\n EVALUATION")
score_lgb = balanced_log_loss(y, oof_lgb)
score_cat = balanced_log_loss(y, oof_cat)
score_nn  = balanced_log_loss(y, oof_nn)

print(f"Balanced Log Loss - LightGBM: {score_lgb:.5f}")
print(f"Balanced Log Loss - CatBoost: {score_cat:.5f}")
print(f"Balanced Log Loss - NeuralNet: {score_nn:.5f}")

#4. Kết hợp xong submit
# Trọng số pha trộn: Thường GBDT ổn định hơn nên cho trọng số cao hơn
w_lgb = 0.4
w_cat = 0.4
w_nn  = 0.2

final_oof = (w_lgb * oof_lgb) + (w_cat * oof_cat) + (w_nn * oof_nn)
final_score = balanced_log_loss(y, final_oof)
print(f"==> FINAL ENSEMBLE SCORE: {final_score:.5f}")

final_test_pred = (w_lgb * pred_lgb) + (w_cat * pred_cat) + (w_nn * pred_nn)

# Clip để tránh log_loss 
final_test_pred = np.clip(final_test_pred, 1e-15, 1 - 1e-15)

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    'class_0': 1 - final_test_pred,
    'class_1': final_test_pred
})

submission.to_csv('submission.csv', index=False)


