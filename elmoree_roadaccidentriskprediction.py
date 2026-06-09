import numpy as np
import pandas as pd
import os, warnings, gc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.special import logit, expit  # å…³é”®æ•°å­¦å�˜æ�¢åº“

import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')


SEED = 42
np.random.seed(SEED)
DATA_DIR = "/kaggle/input/playground-series-s5e10" 
OUT_DIR = "/kaggle/working/"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "accident_risk"

CONFIG = {
    "folds": 10,
    "stratified": True,
    "clip_min": 0.0,
    "clip_max": 1.0,
    "early_stopping": 200
}


def process_data(df):
    d = df.copy()
    # åŸºç¡€ç‰©ç�†äº¤äº’
    d['speed_curvature'] = d['speed_limit'] * d['curvature']
    d['speed_lanes'] = d['speed_limit'] * d['num_lanes']
    d['accident_density'] = d['num_reported_accidents'] / (d['num_lanes'] + 0.5)
    d['complexity'] = d['curvature'] / (d['num_lanes'] + 0.5)
    
    # æ–‡æœ¬ç»„å�ˆ
    d['road_lighting'] = d['road_type'] + '_' + d['lighting']
    d['weather_time'] = d['weather'] + '_' + d['time_of_day']
    
    # é£�é™©è¯„åˆ†
    d['heuristic_risk'] = (
        0.35 * d['curvature'] + 
        0.15 * (d['lighting'] == 'night').astype(int) +
        0.15 * (d['weather'].isin(['rainy', 'foggy', 'snowy'])).astype(int) +
        0.20 * (d['speed_limit'] >= 60).astype(int) +
        0.15 * np.log1p(d['num_reported_accidents'])
    )
    return d

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

train['is_train'] = 1
test['is_train'] = 0
test[TARGET] = 0
df = pd.concat([train, test], sort=False).reset_index(drop=True)

df = process_data(df)



print("Generating KNN features...")
knn_cols = ['speed_limit', 'curvature', 'num_lanes', 'num_reported_accidents']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[knn_cols])

# å¯»æ‰¾æœ€è¿‘çš„ 10 ä¸ªé‚»å±…
knn = NearestNeighbors(n_neighbors=10, n_jobs=-1)
knn.fit(X_scaled)
dists, _ = knn.kneighbors(X_scaled)

# ç‰¹å¾�ï¼šä¸�é‚»å±…çš„å¹³å�‡è·�ç¦» (å��æ˜ è¯¥è·¯æ®µæ˜¯å�¦"å�¦ç±»")
df['knn_dist_mean'] = dists.mean(axis=1)
df['knn_dist_max'] = dists.max(axis=1)
df['knn_dist_std'] = dists.std(axis=1)


cat_cols = ['road_type', 'weather', 'lighting', 'time_of_day', 'road_lighting', 'weather_time']
# Label Encoding
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))

# Split back
train_df = df[df['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
test_df  = df[df['is_train'] == 0].drop(columns=['is_train', TARGET]).reset_index(drop=True)

X = train_df.drop(columns=['id', TARGET])
y = train_df[TARGET]
X_test = test_df.drop(columns=['id'])

# CatBoost éœ€è¦�çŸ¥é�“å“ªäº›åˆ—æ˜¯ç±»åˆ«çš„ç´¢å¼•
cat_features_indices = [list(X.columns).index(c) for c in cat_cols if c in X.columns]


# å‡†å¤‡ Stratified KFold
num_bins = 15
y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates='drop')
kf = StratifiedKFold(n_splits=CONFIG['folds'], shuffle=True, random_state=SEED)

# åˆ�å§‹åŒ– OOF å®¹å™¨
oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

oof_cat = np.zeros(len(X))
test_cat = np.zeros(len(X_test))

# æ¨¡å�‹å�‚æ•°
lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.015,
    'n_estimators': 3000, 'num_leaves': 64, 'max_depth': 8,
    'subsample': 0.7, 'colsample_bytree': 0.7, 'n_jobs': -1, 'verbosity': -1
}

xgb_params = {
    'objective': 'reg:squarederror', 'learning_rate': 0.015,
    'n_estimators': 3000, 'max_depth': 6, 'subsample': 0.7,
    'colsample_bytree': 0.7, 'n_jobs': -1, 'enable_categorical': False
}

cat_params = {
    'loss_function': 'RMSE', 'learning_rate': 0.02,
    'iterations': 3000, 'depth': 6, 'subsample': 0.7,
    'verbose': 0, 'allow_writing_files': False,
    'cat_features': cat_features_indices # å�Ÿç”Ÿå¤„ç�†ç±»åˆ«
}

print(f"Starting {CONFIG['folds']}-Fold Training with Logit Transform...")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_bins), 1):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # === æ”¹è¿›: Logit Transform ===
    # å°† y ä»� [0,1] æ˜ å°„åˆ° (-inf, +inf)
    # åŠ ä¸Š eps é˜²æ­¢ log(0)
    y_tr_logit = logit(np.clip(y_tr, 1e-6, 1 - 1e-6))
    y_val_logit = logit(np.clip(y_val, 1e-6, 1 - 1e-6))
    
    # ------------------------- Model 1: LightGBM -------------------------
    model_lgb = LGBMRegressor(**lgb_params, random_state=SEED+fold)
    model_lgb.fit(
        X_tr, y_tr_logit, 
        eval_set=[(X_val, y_val_logit)],
        callbacks=[lgb.early_stopping(CONFIG['early_stopping'], verbose=False)]
    )
    # é¢„æµ‹å¹¶å��å�˜æ�¢ (Sigmoid)
    p_lgb = expit(model_lgb.predict(X_val))
    oof_lgb[val_idx] = p_lgb
    test_lgb += expit(model_lgb.predict(X_test)) / CONFIG['folds']
    
    # ---------------------------- Model 2: XGBoost --------------------------
    model_xgb = XGBRegressor(**xgb_params, random_state=SEED+fold)
    model_xgb.fit(
        X_tr, y_tr_logit,
        eval_set=[(X_val, y_val_logit)],
        early_stopping_rounds=CONFIG['early_stopping'],
        verbose=False
    )
    p_xgb = expit(model_xgb.predict(X_val))
    oof_xgb[val_idx] = p_xgb
    test_xgb += expit(model_xgb.predict(X_test)) / CONFIG['folds']
    
    # ------------------------------- Model 3: CatBoost ------------------------------------
    model_cat = CatBoostRegressor(**cat_params, random_seed=SEED+fold)
    model_cat.fit(
        X_tr, y_tr_logit,
        eval_set=(X_val, y_val_logit),
        early_stopping_rounds=CONFIG['early_stopping'],
        use_best_model=True
    )
    p_cat = expit(model_cat.predict(X_val))
    oof_cat[val_idx] = p_cat
    test_cat += expit(model_cat.predict(X_test)) / CONFIG['folds']
    
    # Fold Score (ç®€å�•å¹³å�‡çœ‹æ•ˆæ�œ)
    avg_pred = (p_lgb + p_xgb + p_cat) / 3
    rmse = np.sqrt(mean_squared_error(y_val, avg_pred))
    print(f"Fold {fold} | LGB: {np.sqrt(mean_squared_error(y_val, p_lgb)):.5f} | XGB: {np.sqrt(mean_squared_error(y_val, p_xgb)):.5f} | CAT: {np.sqrt(mean_squared_error(y_val, p_cat)):.5f} | Blend: {rmse:.5f}")


print("\nFinding Best Weights...")

# å®šä¹‰å¯»æ‰¾æœ€ä½³æ�ƒé‡�çš„å‡½æ•°
def find_best_weights(oof_dict, y_true):
    best_score = 999
    best_weights = []
    
    # ç²—ç•¥ç½‘æ ¼æ�œç´¢ (æ¯”scipy minimizeæ›´ç›´è§‚)
    # æ�ƒé‡�å’Œä¸º1
    import itertools
    weights_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for w1 in weights_grid:
        for w2 in weights_grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0: continue
            
            pred = w1 * oof_dict['lgb'] + w2 * oof_dict['xgb'] + w3 * oof_dict['cat']
            score = np.sqrt(mean_squared_error(y_true, pred))
            
            if score < best_score:
                best_score = score
                best_weights = [w1, w2, w3]
                
    return best_weights, best_score

oof_dict = {'lgb': oof_lgb, 'xgb': oof_xgb, 'cat': oof_cat}
weights, final_score = find_best_weights(oof_dict, y)

print(f"Best Weights -> LGB: {weights[0]:.2f}, XGB: {weights[1]:.2f}, CAT: {weights[2]:.2f}")
print(f"Final Weighted RMSE: {final_score:.6f}")




final_pred = (weights[0] * test_lgb) + (weights[1] * test_xgb) + (weights[2] * test_cat)
final_pred = np.clip(final_pred, CONFIG['clip_min'], CONFIG['clip_max'])

sub = sample.copy()
sub[TARGET] = final_pred
sub_path = os.path.join(OUT_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path}")


# train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
# test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
# sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

# print(f"Train: {train.shape}, Test: {test.shape}")

# # Merge processing to unify feature engineering.
# train['is_train'] = 1
# test['is_train'] = 0
# test[TARGET] = 0
# df = pd.concat([train, test], sort=False).reset_index(drop=True)


# def process_data(df):
#     d = df.copy()
    
#     # 1. åŸºç¡€ç‰©ç�†äº¤äº’ (Interaction)
#     # é€Ÿåº¦ä¸�è·¯å†µçš„ç»“å�ˆ
#     d['speed_curvature'] = d['speed_limit'] * d['curvature']
#     d['speed_lanes'] = d['speed_limit'] * d['num_lanes']
    
#     # 2. å¯†åº¦ç‰¹å¾� (Density)
#     # äº‹æ•…å¯†åº¦ï¼šå·²æŠ¥å‘Šäº‹æ•…æ•° / è½¦é�“æ•° (é˜²æ­¢é™¤0åŠ æ��å°�å€¼)
#     d['accident_density'] = d['num_reported_accidents'] / (d['num_lanes'] + 0.5)
#     # æ‹¥æŒ¤åº¦ä»£ç�†ï¼šè½¦é�“å°‘ä¸”å¼¯é�“å¤§å�¯èƒ½æ›´å�±é™©
#     d['complexity'] = d['curvature'] / (d['num_lanes'] + 0.5)

#     # 3. æ–‡æœ¬/ç±»åˆ«ç»„å�ˆ (Cross Features)
#     # è¿™ç§�ç»„å�ˆç‰¹å¾�å¾€å¾€åŒ…å�«å·¨å¤§çš„é��çº¿æ€§ä¿¡æ�¯
#     d['road_lighting'] = d['road_type'] + '_' + d['lighting']
#     d['weather_time'] = d['weather'] + '_' + d['time_of_day']
#     d['road_weather'] = d['road_type'] + '_' + d['weather']
    
#     # 4. å�¯å�‘å¼�é£�é™©è¯„åˆ† (Heuristic Score) - ç•¥å¾®ä¼˜åŒ–ç³»æ•°
#     d['heuristic_risk'] = (
#         0.35 * d['curvature'] + 
#         0.15 * (d['lighting'] == 'night').astype(int) +
#         0.15 * (d['weather'].isin(['rainy', 'foggy', 'snowy'])).astype(int) +
#         0.20 * (d['speed_limit'] >= 60).astype(int) +
#         0.15 * np.log1p(d['num_reported_accidents'])
#     )
    
#     return d

# df = process_data(df)

# # Encoding
# # Label Encoding for Categoricals
# cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in ['id']]
# print("Categorical Columns:", cat_cols)

# for c in cat_cols:
#     le = LabelEncoder()
#     df[c] = le.fit_transform(df[c].astype(str))

# # Split Train/Test
# train_df = df[df['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
# test_df  = df[df['is_train'] == 0].drop(columns=['is_train', TARGET]).reset_index(drop=True)

# X = train_df.drop(columns=['id', TARGET])
# y = train_df[TARGET]
# X_test = test_df.drop(columns=['id'])


# # ä¸�ç›´æ�¥ç”¨å…¨å±€å�‡å€¼ï¼Œè€Œæ˜¯åœ¨CVå†…éƒ¨å�šï¼Œé˜²æ­¢æ•°æ�®ç©¿è¶Š

# def get_kfold_encoding(train_x, train_y, test_x, cat_features, n_folds=5):
#     # åˆ›å»ºä¸´æ—¶å®¹å™¨
#     out_train = pd.DataFrame()
#     out_test = pd.DataFrame()
    
#     # ç®€å�•èµ·è§�ï¼Œæˆ‘ä»¬å¯¹Testä½¿ç”¨å…¨é‡�Trainçš„å�‡å€¼
#     # å¯¹Trainä½¿ç”¨OOFå�‡å€¼
#     kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
#     for col in cat_features:
#         # åˆ�å§‹åŒ–
#         out_train[f'te_{col}'] = np.zeros(len(train_x))
#         out_test[f'te_{col}'] = np.zeros(len(test_x))
        
#         # è®¡ç®—å…¨å±€å�‡å€¼ä½œä¸ºç¼ºå¤±å¡«å……
#         global_mean = train_y.mean()
        
#         # 1. è®­ç»ƒé›† OOF Encoding
#         for tr_idx, val_idx in kf.split(train_x):
#             X_tr, y_tr = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
#             X_val = train_x.iloc[val_idx]
            
#             # è®¡ç®—å�‡å€¼
#             means = X_tr.join(y_tr).groupby(col)[TARGET].mean()
#             out_train.loc[val_idx, f'te_{col}'] = X_val[col].map(means).fillna(global_mean)
        
#         # 2. æµ‹è¯•é›† Encoding (ç”¨å…¨é‡�Train)
#         means_all = train_x.join(train_y).groupby(col)[TARGET].mean()
#         out_test[f'te_{col}'] = test_x[col].map(means_all).fillna(global_mean)
        
#     return out_train, out_test

# # é€‰å�–é«˜åŸºæ•°æˆ–é‡�è¦�ç±»åˆ«ç‰¹å¾�è¿›è¡Œ Target Encoding
# te_cols = ['road_type', 'weather', 'road_lighting', 'weather_time']
# X_te_tr, X_te_te = get_kfold_encoding(train_df.drop(columns=['id', TARGET]), y, test_df.drop(columns=['id']), te_cols)

# X = pd.concat([X, X_te_tr], axis=1)
# X_test = pd.concat([X_test, X_te_te], axis=1)

# print("New features added via Target Encoding:", X_te_tr.columns.tolist())


# # Stratified KFold Strategy
# # å› ä¸ºTargetæ˜¯è¿�ç»­å€¼ï¼Œæˆ‘ä»¬éœ€è¦�å…ˆå°†å…¶åˆ†ç®±(binning)æ�¥å�šStratified
# num_bins = 10
# y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates='drop')
# kf = StratifiedKFold(n_splits=CONFIG['folds'], shuffle=True, random_state=SEED)

# # Placeholders
# oof_base = np.zeros(len(X))
# test_base_pred = np.zeros(len(X_test))

# oof_resid = np.zeros(len(X))
# test_resid_pred = np.zeros(len(X_test))

# # Hyperparameters (Tuned for Low Learning Rate)
# lgb_params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'learning_rate': 0.02,
#     'n_estimators': 5000,
#     'num_leaves': 128,
#     'max_depth': 12,
#     'subsample': 0.8,
#     'colsample_bytree': 0.7,
#     'reg_alpha': 0.5,
#     'reg_lambda': 0.5,
#     'random_state': SEED,
#     'n_jobs': -1,
#     'verbosity': -1
# }

# xgb_resid_params = {
#     'objective': 'reg:squarederror',
#     'learning_rate': 0.03,
#     'n_estimators': 3000,
#     'max_depth': 6,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'reg_lambda': 1.0,
#     'random_state': SEED + 99,
#     'n_jobs': -1,
#     'enable_categorical': False # XGBå¤„ç�†OneHotæˆ–LabelEncæ¯”å�Ÿç”ŸCatæ”¯æŒ�å¼±ä¸€ç‚¹ï¼Œä½†è¿™é‡Œç”¨æ�¥ä¿®æ®‹å·®è¶³å¤Ÿ
# }

# print(f"Starting {CONFIG['folds']}-Fold Training...")

# for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_bins), 1):
#     X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
#     X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
#     # --- Stage 1: Base Model (LightGBM) ---
#     model_lgb = LGBMRegressor(**lgb_params)
#     model_lgb.fit(
#         X_tr, y_tr, 
#         eval_set=[(X_val, y_val)],
#         callbacks=[lgb.early_stopping(CONFIG['early_stopping'], verbose=False)]
#     )
    
#     # Base Predictions
#     val_pred_base = model_lgb.predict(X_val)
#     oof_base[val_idx] = val_pred_base
#     test_base_pred += model_lgb.predict(X_test) / CONFIG['folds']
    
#     rmse_base_fold = np.sqrt(mean_squared_error(y_val, val_pred_base))
    
#     # --- Stage 2: Residual Model (XGBoost) ---
#     # è®¡ç®—æ®‹å·® (Residuals)
#     residuals_tr = y_tr - model_lgb.predict(X_tr)
#     residuals_val = y_val - val_pred_base
    
#     # XGBoost è®­ç»ƒæ®‹å·®
#     model_xgb = XGBRegressor(**xgb_resid_params)
#     model_xgb.fit(
#         X_tr, residuals_tr,
#         eval_set=[(X_val, residuals_val)],
#         early_stopping_rounds=100,
#         verbose=False
#     )
    
#     val_pred_resid = model_xgb.predict(X_val)
#     oof_resid[val_idx] = val_pred_resid
#     test_resid_pred += model_xgb.predict(X_test) / CONFIG['folds']
    
#     # --- Combined Performance ---
#     final_fold_pred = val_pred_base + val_pred_resid
#     rmse_final_fold = np.sqrt(mean_squared_error(y_val, final_fold_pred))
    
#     print(f"Fold {fold} | Base RMSE: {rmse_base_fold:.5f} | Resid Corrected RMSE: {rmse_final_fold:.5f} (Imp: {rmse_base_fold - rmse_final_fold:.6f})")

# print("Training Done.")



# # Stratified KFold Strategy
# # å› ä¸ºTargetæ˜¯è¿�ç»­å€¼ï¼Œæˆ‘ä»¬éœ€è¦�å…ˆå°†å…¶åˆ†ç®±(binning)æ�¥å�šStratified
# num_bins = 10
# y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates='drop')
# kf = StratifiedKFold(n_splits=CONFIG['folds'], shuffle=True, random_state=SEED)

# # Placeholders
# oof_base = np.zeros(len(X))
# test_base_pred = np.zeros(len(X_test))

# oof_resid = np.zeros(len(X))
# test_resid_pred = np.zeros(len(X_test))

# # Hyperparameters (Tuned for Low Learning Rate)
# lgb_params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'learning_rate': 0.02,
#     'n_estimators': 5000,
#     'num_leaves': 128,
#     'max_depth': 12,
#     'subsample': 0.8,
#     'colsample_bytree': 0.7,
#     'reg_alpha': 0.5,
#     'reg_lambda': 0.5,
#     'random_state': SEED,
#     'n_jobs': -1,
#     'verbosity': -1
# }

# xgb_resid_params = {
#     'objective': 'reg:squarederror',
#     'learning_rate': 0.03,
#     'n_estimators': 3000,
#     'max_depth': 6,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'reg_lambda': 1.0,
#     'random_state': SEED + 99,
#     'n_jobs': -1,
#     'enable_categorical': False # XGBå¤„ç�†OneHotæˆ–LabelEncæ¯”å�Ÿç”ŸCatæ”¯æŒ�å¼±ä¸€ç‚¹ï¼Œä½†è¿™é‡Œç”¨æ�¥ä¿®æ®‹å·®è¶³å¤Ÿ
# }

# print(f"Starting {CONFIG['folds']}-Fold Training...")

# for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_bins), 1):
#     X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
#     X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
#     # --- Stage 1: Base Model (LightGBM) ---
#     model_lgb = LGBMRegressor(**lgb_params)
#     model_lgb.fit(
#         X_tr, y_tr, 
#         eval_set=[(X_val, y_val)],
#         callbacks=[lgb.early_stopping(CONFIG['early_stopping'], verbose=False)]
#     )
    
#     # Base Predictions
#     val_pred_base = model_lgb.predict(X_val)
#     oof_base[val_idx] = val_pred_base
#     test_base_pred += model_lgb.predict(X_test) / CONFIG['folds']
    
#     rmse_base_fold = np.sqrt(mean_squared_error(y_val, val_pred_base))
    
#     # --- Stage 2: Residual Model (XGBoost) ---
#     # è®¡ç®—æ®‹å·® (Residuals)
#     residuals_tr = y_tr - model_lgb.predict(X_tr)
#     residuals_val = y_val - val_pred_base
    
#     # XGBoost è®­ç»ƒæ®‹å·®
#     model_xgb = XGBRegressor(**xgb_resid_params)
#     model_xgb.fit(
#         X_tr, residuals_tr,
#         eval_set=[(X_val, residuals_val)],
#         early_stopping_rounds=100,
#         verbose=False
#     )
    
#     val_pred_resid = model_xgb.predict(X_val)
#     oof_resid[val_idx] = val_pred_resid
#     test_resid_pred += model_xgb.predict(X_test) / CONFIG['folds']
    
#     # --- Combined Performance ---
#     final_fold_pred = val_pred_base + val_pred_resid
#     rmse_final_fold = np.sqrt(mean_squared_error(y_val, final_fold_pred))
    
#     print(f"Fold {fold} | Base RMSE: {rmse_base_fold:.5f} | Resid Corrected RMSE: {rmse_final_fold:.5f} (Imp: {rmse_base_fold - rmse_final_fold:.6f})")

# print("Training Done.")


# # æœ€ç»ˆå�ˆæˆ�
# oof_final = oof_base + oof_resid
# # Clip to valid range (0-1)
# oof_final = np.clip(oof_final, CONFIG['clip_min'], CONFIG['clip_max'])

# score_base = np.sqrt(mean_squared_error(y, oof_base))
# score_final = np.sqrt(mean_squared_error(y, oof_final))

# print(f"\nOverall Base OOF RMSE: {score_base:.6f}")
# print(f"Overall Final OOF RMSE: {score_final:.6f}")
# print(f"Improvement: {score_base - score_final:.6f}")

# # Predict Test
# final_pred = test_base_pred + test_resid_pred
# final_pred = np.clip(final_pred, CONFIG['clip_min'], CONFIG['clip_max'])

# sub = sample.copy()
# sub[TARGET] = final_pred
# sub_path = os.path.join(OUT_DIR, "submission.csv")
# sub.to_csv(sub_path, index=False)
# print(f"Submission saved to {sub_path}")
# sub.head()

