import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
import pickle
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
print('Libraries imported successfully')


# ãƒ‡ãƒ¼ã‚¿ãƒ‘ã‚¹è¨­å®š
DATA_DIR = Path('/kaggle/input/cmi-detect-behavior-with-sensor-data')

# train.csvã�®èª­ã�¿è¾¼ã�¿
print('Loading real training data...')
train_data = pd.read_csv(DATA_DIR / 'train.csv')
print(f'  Loaded train.csv: {len(train_data)} rows')

# sequence_idã�§ã‚°ãƒ«ãƒ¼ãƒ—åŒ–ã�—ã�¦ãƒ¡ã‚¿ãƒ‡ãƒ¼ã‚¿ä½œæˆ�
train_metadata = train_data.groupby('sequence_id').agg({
    'subject': 'first',
    'gesture': 'first',
    'sequence_counter': 'count'
}).reset_index()
train_metadata.columns = ['sequence_id', 'subject', 'gesture', 'sequence_length']

# ã‚¸ã‚§ã‚¹ãƒ�ãƒ£ãƒ¼ã�®ãƒ©ãƒ™ãƒ«ãƒ�ãƒƒãƒ”ãƒ³ã‚°
gesture_labels = sorted(train_metadata['gesture'].unique())
gesture_to_id = {label: i for i, label in enumerate(gesture_labels)}
id_to_gesture = {i: label for label, i in gesture_to_id.items()}

print(f'\nData Summary:')
print(f'  Total sequences: {len(train_metadata)}')
print(f'  Gesture labels: {len(gesture_labels)} types')
print(f'  Average sequence length: {train_metadata["sequence_length"].mean():.1f}')
print(f'\nGesture distribution:')
print(train_metadata['gesture'].value_counts())


def extract_advanced_features(df, seq_id, metadata_row):
    """é«˜åº¦ã�ªç‰¹å¾´é‡�æŠ½å‡ºï¼ˆå®Ÿãƒ‡ãƒ¼ã‚¿ç”¨ã�«æœ€é�©åŒ–ï¼‰"""
    features = {}
    
    # ãƒ¡ã‚¿ãƒ‡ãƒ¼ã‚¿ç‰¹å¾´
    features['subject_id'] = hash(metadata_row['subject']) % 1000
    features['seq_length'] = len(df)
    features['seq_length_log'] = np.log1p(len(df))
    
    # åŠ é€Ÿåº¦è¨ˆç‰¹å¾´ï¼ˆæ‹¡å¼µç‰ˆï¼‰
    for axis in ['x', 'y', 'z']:
        col = f'acc_{axis}'
        if col in df.columns:
            values = df[col].values
            
            # åŸºæœ¬çµ±è¨ˆé‡�
            features[f'acc_{axis}_mean'] = np.mean(values)
            features[f'acc_{axis}_std'] = np.std(values)
            features[f'acc_{axis}_max'] = np.max(values)
            features[f'acc_{axis}_min'] = np.min(values)
            features[f'acc_{axis}_range'] = np.ptp(values)
            features[f'acc_{axis}_skew'] = pd.Series(values).skew()
            features[f'acc_{axis}_kurt'] = pd.Series(values).kurt()
            
            # ãƒ‘ãƒ¼ã‚»ãƒ³ã‚¿ã‚¤ãƒ«
            features[f'acc_{axis}_q25'] = np.percentile(values, 25)
            features[f'acc_{axis}_q50'] = np.percentile(values, 50)
            features[f'acc_{axis}_q75'] = np.percentile(values, 75)
            features[f'acc_{axis}_iqr'] = features[f'acc_{axis}_q75'] - features[f'acc_{axis}_q25']
            
            # å‘¨æ³¢æ•°é ˜åŸŸç‰¹å¾´ï¼ˆæ‹¡å¼µï¼‰
            fft_vals = np.abs(np.fft.fft(values))[:len(values)//2]
            if len(fft_vals) > 0:
                features[f'acc_{axis}_fft_max'] = np.max(fft_vals)
                features[f'acc_{axis}_fft_mean'] = np.mean(fft_vals)
                features[f'acc_{axis}_fft_std'] = np.std(fft_vals)
                features[f'acc_{axis}_fft_energy'] = np.sum(fft_vals**2)
                # ä¸»è¦�å‘¨æ³¢æ•°
                features[f'acc_{axis}_dominant_freq'] = np.argmax(fft_vals)
            else:
                features[f'acc_{axis}_fft_max'] = 0
                features[f'acc_{axis}_fft_mean'] = 0
                features[f'acc_{axis}_fft_std'] = 0
                features[f'acc_{axis}_fft_energy'] = 0
                features[f'acc_{axis}_dominant_freq'] = 0
            
            # ã‚¸ãƒ£ãƒ¼ã‚¯ï¼ˆåŠ é€Ÿåº¦ã�®å¤‰åŒ–ç�‡ï¼‰
            if len(values) > 1:
                jerk = np.diff(values)
                features[f'acc_{axis}_jerk_mean'] = np.mean(np.abs(jerk))
                features[f'acc_{axis}_jerk_std'] = np.std(jerk)
                features[f'acc_{axis}_jerk_max'] = np.max(np.abs(jerk))
            else:
                features[f'acc_{axis}_jerk_mean'] = 0
                features[f'acc_{axis}_jerk_std'] = 0
                features[f'acc_{axis}_jerk_max'] = 0
            
            # ã‚¼ãƒ­äº¤å·®ç�‡
            features[f'acc_{axis}_zero_cross'] = np.sum(np.diff(np.sign(values)) != 0)
    
    # åŠ é€Ÿåº¦ã�®å¤§ã��ã�•ã�¨é‡�åŠ›é™¤å�»
    if all(f'acc_{axis}' in df.columns for axis in ['x', 'y', 'z']):
        # åŠ é€Ÿåº¦ãƒ™ã‚¯ãƒˆãƒ«ã�®å¤§ã��ã�•
        acc_magnitude = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        features['acc_magnitude_mean'] = np.mean(acc_magnitude)
        features['acc_magnitude_std'] = np.std(acc_magnitude)
        features['acc_magnitude_max'] = np.max(acc_magnitude)
        features['acc_magnitude_energy'] = np.sum(acc_magnitude**2)
        
        # é‡�åŠ›æˆ�åˆ†ã�®æ�¨å®šã�¨é™¤å�»
        gravity_x = np.mean(df['acc_x'])
        gravity_y = np.mean(df['acc_y'])
        gravity_z = np.mean(df['acc_z'])
        
        linear_acc_x = df['acc_x'] - gravity_x
        linear_acc_y = df['acc_y'] - gravity_y
        linear_acc_z = df['acc_z'] - gravity_z
        
        linear_magnitude = np.sqrt(linear_acc_x**2 + linear_acc_y**2 + linear_acc_z**2)
        features['linear_acc_mean'] = np.mean(linear_magnitude)
        features['linear_acc_std'] = np.std(linear_magnitude)
        features['linear_acc_max'] = np.max(linear_magnitude)
    
    # ã‚¸ãƒ£ã‚¤ãƒ­ã‚¹ã‚³ãƒ¼ãƒ—ç‰¹å¾´ï¼ˆæ‹¡å¼µç‰ˆï¼‰
    for axis in ['x', 'y', 'z']:
        col = f'gyr_{axis}'
        if col in df.columns:
            values = df[col].values
            features[f'gyr_{axis}_mean'] = np.mean(values)
            features[f'gyr_{axis}_std'] = np.std(values)
            features[f'gyr_{axis}_max'] = np.max(values)
            features[f'gyr_{axis}_min'] = np.min(values)
            features[f'gyr_{axis}_energy'] = np.sum(values**2)
            features[f'gyr_{axis}_rms'] = np.sqrt(np.mean(values**2))
            
            # è§’åŠ é€Ÿåº¦
            if len(values) > 1:
                angular_acc = np.diff(values)
                features[f'gyr_{axis}_angular_acc_mean'] = np.mean(np.abs(angular_acc))
                features[f'gyr_{axis}_angular_acc_max'] = np.max(np.abs(angular_acc))
            else:
                features[f'gyr_{axis}_angular_acc_mean'] = 0
                features[f'gyr_{axis}_angular_acc_max'] = 0
    
    # è§’é€Ÿåº¦ã�®å¤§ã��ã�•
    if all(f'gyr_{axis}' in df.columns for axis in ['x', 'y', 'z']):
        gyr_magnitude = np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2)
        features['gyr_magnitude_mean'] = np.mean(gyr_magnitude)
        features['gyr_magnitude_std'] = np.std(gyr_magnitude)
        features['gyr_magnitude_max'] = np.max(gyr_magnitude)
        features['gyr_magnitude_energy'] = np.sum(gyr_magnitude**2)
    
    # ã‚¯ã‚©ãƒ¼ã‚¿ãƒ‹ã‚ªãƒ³ç‰¹å¾´ï¼ˆå§¿å‹¢æ�¨å®šï¼‰
    for comp in ['w', 'x', 'y', 'z']:
        col = f'quat_{comp}'
        if col in df.columns:
            values = df[col].values
            features[f'quat_{comp}_mean'] = np.mean(values)
            features[f'quat_{comp}_std'] = np.std(values)
            features[f'quat_{comp}_change'] = values[-1] - values[0] if len(values) > 0 else 0
            features[f'quat_{comp}_range'] = np.ptp(values)
    
    # ã‚ªã‚¤ãƒ©ãƒ¼è§’ã�¸ã�®å¤‰æ�›ï¼ˆè¿‘ä¼¼ï¼‰
    if all(f'quat_{comp}' in df.columns for comp in ['w', 'x', 'y', 'z']):
        qw = df['quat_w'].mean()
        qx = df['quat_x'].mean()
        qy = df['quat_y'].mean()
        qz = df['quat_z'].mean()
        
        # Roll
        features['euler_roll'] = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        # Pitch
        features['euler_pitch'] = np.arcsin(2*(qw*qy - qz*qx))
        # Yaw
        features['euler_yaw'] = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    
    # ToFã‚»ãƒ³ã‚µãƒ¼ç‰¹å¾´ï¼ˆè·�é›¢ã‚»ãƒ³ã‚µãƒ¼ï¼‰
    tof_stats = []
    for i in range(8):
        col = f'tof_{i}'
        if col in df.columns:
            values = df[col].values
            tof_stats.extend([
                np.mean(values),
                np.std(values),
                np.median(values),
                np.min(values),
                np.max(values)
            ])
    
    # ToFçµ±è¨ˆé‡�ã‚’ä¸»æˆ�åˆ†ã�¨ã�—ã�¦ä¿�å­˜
    if tof_stats:
        # æœ€åˆ�ã�®10å€‹ã�®çµ±è¨ˆé‡�ã‚’ä½¿ç”¨
        for i in range(min(10, len(tof_stats))):
            features[f'tof_stat_{i}'] = tof_stats[i]
        
        # ToFå…¨ä½“ã�®çµ±è¨ˆ
        features['tof_global_mean'] = np.mean(tof_stats)
        features['tof_global_std'] = np.std(tof_stats)
        features['tof_global_range'] = np.ptp(tof_stats)
    
    # ã‚»ãƒ³ã‚µãƒ¼é–“ã�®ç›¸é–¢ç‰¹å¾´
    if all(f'acc_{axis}' in df.columns for axis in ['x', 'y', 'z']) and \
       all(f'gyr_{axis}' in df.columns for axis in ['x', 'y', 'z']):
        # åŠ é€Ÿåº¦ã�¨ã‚¸ãƒ£ã‚¤ãƒ­ã�®ç›¸é–¢
        for axis in ['x', 'y', 'z']:
            corr = np.corrcoef(df[f'acc_{axis}'], df[f'gyr_{axis}'])[0, 1]
            features[f'acc_gyr_corr_{axis}'] = corr if not np.isnan(corr) else 0
    
    # ãƒ©ãƒ™ãƒ«ï¼ˆæ•´æ•°IDï¼‰
    features['gesture'] = gesture_to_id[metadata_row['gesture']]
    
    return features

print('Advanced feature extractor ready')


# å…¨å®Ÿãƒ‡ãƒ¼ã‚¿ã�‹ã‚‰ç‰¹å¾´é‡�æŠ½å‡º
print(f'Extracting features from {len(train_metadata)} real sequences...')

real_features = []
failed_count = 0

for idx, row in train_metadata.iterrows():
    try:
        # train.csvã�‹ã‚‰è©²å½“ã‚·ãƒ¼ã‚±ãƒ³ã‚¹ã�®ãƒ‡ãƒ¼ã‚¿ã‚’å�–å¾—
        seq_data = train_data[train_data['sequence_id'] == row['sequence_id']]
        if len(seq_data) > 0:
            features = extract_advanced_features(seq_data, row['sequence_id'], row)
            real_features.append(features)
    except Exception as e:
        failed_count += 1
        if failed_count <= 3:
            print(f'  Failed: {row["sequence_id"]}: {e}')
        continue
    
    # é€²æ�—è¡¨ç¤º
    if (idx + 1) % 1000 == 0:
        print(f'  Processed {idx + 1}/{len(train_metadata)} sequences')

print(f'\nFeature extraction completed:')
print(f'  Success: {len(real_features)} sequences')
if failed_count > 0:
    print(f'  Failed: {failed_count} sequences')

# DataFrameã�«å¤‰æ�›
feature_df = pd.DataFrame(real_features)
feature_df = feature_df.fillna(0)

print(f'\nFeature matrix shape: {feature_df.shape}')
print(f'Features: {feature_df.shape[1] - 1} dimensions')


# ç‰¹å¾´é‡�ã�¨ãƒ©ãƒ™ãƒ«ã�®åˆ†é›¢
X = feature_df.drop('gesture', axis=1)
y = feature_df['gesture']

# ã‚¹ã‚±ãƒ¼ãƒªãƒ³ã‚°
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# æ¤œè¨¼ã‚¹ã‚³ã‚¢ä¿�å­˜ç”¨
cv_scores = {
    'lgb': [],
    'xgb': [],
    'cat': [],
    'ensemble': []
}

# å…¨Foldã�®ãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜ï¼ˆæœ€å¾Œã�«ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ï¼‰
all_models = {
    'lgb': [],
    'xgb': [],
    'cat': []
}

print('Starting 5-Fold Cross Validation...')
print('=' * 50)


for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f'\nFold {fold}/5:')
    print('-' * 30)
    
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 18,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 100,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'seed': 42 + fold,
        'n_jobs': -1
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    lgb_pred_class = np.argmax(lgb_pred, axis=1)
    lgb_acc = accuracy_score(y_val, lgb_pred_class)
    cv_scores['lgb'].append(lgb_acc)
    all_models['lgb'].append(lgb_model)
    print(f'  LightGBM: {lgb_acc:.4f}')
    
    # XGBoost
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 18,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'seed': 42 + fold,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    xgb_pred = xgb_model.predict(dval)
    xgb_pred_class = np.argmax(xgb_pred, axis=1)
    xgb_acc = accuracy_score(y_val, xgb_pred_class)
    cv_scores['xgb'].append(xgb_acc)
    all_models['xgb'].append(xgb_model)
    print(f'  XGBoost: {xgb_acc:.4f}')
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        loss_function='MultiClass',
        classes_count=18,
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=100,
        task_type='GPU' if os.path.exists('/proc/driver/nvidia/version') else 'CPU'
    )
    
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    cat_pred_class = cat_model.predict(X_val)
    cat_acc = accuracy_score(y_val, cat_pred_class)
    cv_scores['cat'].append(cat_acc)
    all_models['cat'].append(cat_model)
    print(f'  CatBoost: {cat_acc:.4f}')
    
    # ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ï¼ˆFoldå†…ï¼‰
    lgb_weight = 0.4
    xgb_weight = 0.3
    cat_weight = 0.3
    
    ensemble_proba = (
        lgb_weight * lgb_pred + 
        xgb_weight * xgb_pred + 
        cat_weight * cat_model.predict_proba(X_val)
    )
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_acc = accuracy_score(y_val, ensemble_pred)
    cv_scores['ensemble'].append(ensemble_acc)
    print(f'  Ensemble: {ensemble_acc:.4f}')


print('\n' + '=' * 50)
print('Cross Validation Results:')
print('=' * 50)

for model_name in ['lgb', 'xgb', 'cat', 'ensemble']:
    scores = cv_scores[model_name]
    print(f'\n{model_name.upper()}:')
    print(f'  Folds: {scores}')
    print(f'  Mean: {np.mean(scores):.4f} Â± {np.std(scores):.4f}')

# æœ€çµ‚çš„ã�ªå¹³å�‡ã‚¹ã‚³ã‚¢
final_score = np.mean(cv_scores['ensemble'])
print(f'\n\nFinal CV Score (Ensemble): {final_score:.4f}')


print('Training final models on full dataset...')
print('=' * 50)

# å…¨ãƒ‡ãƒ¼ã‚¿ã�§ãƒˆãƒ¬ãƒ¼ãƒ‹ãƒ³ã‚°
X_full = X_scaled
y_full = y

# LightGBMæœ€çµ‚ãƒ¢ãƒ‡ãƒ«
print('Training final LightGBM...')
lgb_train_full = lgb.Dataset(X_full, label=y_full)
lgb_final = lgb.train(
    lgb_params,
    lgb_train_full,
    num_boost_round=400,  # CVçµ�æ�œã�‹ã‚‰æ±ºå®š
    callbacks=[lgb.log_evaluation(100)]
)

# XGBoostæœ€çµ‚ãƒ¢ãƒ‡ãƒ«
print('Training final XGBoost...')
dtrain_full = xgb.DMatrix(X_full, label=y_full)
xgb_final = xgb.train(
    xgb_params,
    dtrain_full,
    num_boost_round=400,
    verbose_eval=100
)

# CatBoostæœ€çµ‚ãƒ¢ãƒ‡ãƒ«
print('Training final CatBoost...')
cat_final = CatBoostClassifier(
    iterations=400,
    learning_rate=0.05,
    depth=8,
    loss_function='MultiClass',
    classes_count=18,
    random_seed=42,
    verbose=100,
    task_type='GPU' if os.path.exists('/proc/driver/nvidia/version') else 'CPU'
)
cat_final.fit(X_full, y_full)

print('\nAll models trained successfully!')


# ãƒ¢ãƒ‡ãƒ«ä¿�å­˜ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒª
model_dir = Path('/kaggle/working/models_v12')
model_dir.mkdir(exist_ok=True)

# æœ€çµ‚ãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜
lgb_final.save_model(str(model_dir / 'lgb_model_v12_final.txt'))
xgb_final.save_model(str(model_dir / 'xgb_model_v12_final.json'))
cat_final.save_model(str(model_dir / 'cat_model_v12_final.cbm'))

# CV Foldãƒ¢ãƒ‡ãƒ«ã‚‚ä¿�å­˜ï¼ˆã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ç”¨ï¼‰
for fold in range(5):
    all_models['lgb'][fold].save_model(str(model_dir / f'lgb_model_v12_fold{fold}.txt'))
    all_models['xgb'][fold].save_model(str(model_dir / f'xgb_model_v12_fold{fold}.json'))
    all_models['cat'][fold].save_model(str(model_dir / f'cat_model_v12_fold{fold}.cbm'))

# ã‚¹ã‚±ãƒ¼ãƒ©ãƒ¼ã�¨ç‰¹å¾´é‡�å��ã‚’ä¿�å­˜
with open(model_dir / 'scaler_v12.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(model_dir / 'feature_names_v12.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

# ã‚¸ã‚§ã‚¹ãƒ�ãƒ£ãƒ¼ãƒ�ãƒƒãƒ”ãƒ³ã‚°ã‚’ä¿�å­˜
with open(model_dir / 'gesture_mapping_v12.pkl', 'wb') as f:
    pickle.dump({'gesture_to_id': gesture_to_id, 'id_to_gesture': id_to_gesture}, f)

# ãƒ¢ãƒ‡ãƒ«æƒ…å ±
model_info = {
    'version': 'v12_real_data_only',
    'training_data': {
        'real_samples': len(feature_df),
        'synthetic_samples': 0,
        'total_samples': len(feature_df)
    },
    'cv_scores': {
        'lgb': {
            'mean': float(np.mean(cv_scores['lgb'])),
            'std': float(np.std(cv_scores['lgb'])),
            'folds': [float(s) for s in cv_scores['lgb']]
        },
        'xgb': {
            'mean': float(np.mean(cv_scores['xgb'])),
            'std': float(np.std(cv_scores['xgb'])),
            'folds': [float(s) for s in cv_scores['xgb']]
        },
        'catboost': {
            'mean': float(np.mean(cv_scores['cat'])),
            'std': float(np.std(cv_scores['cat'])),
            'folds': [float(s) for s in cv_scores['cat']]
        },
        'ensemble': {
            'mean': float(np.mean(cv_scores['ensemble'])),
            'std': float(np.std(cv_scores['ensemble'])),
            'folds': [float(s) for s in cv_scores['ensemble']]
        }
    },
    'ensemble_weights': {
        'lgb': 0.4,
        'xgb': 0.3,
        'catboost': 0.3
    },
    'feature_count': X.shape[1],
    'n_folds': 5
}

with open(model_dir / 'model_info_v12.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f'\nModels saved to {model_dir}')
print(f'Files created:')
for file in sorted(model_dir.glob('*')):
    print(f'  - {file.name}')


print('=' * 60)
print('V12 å®Ÿãƒ‡ãƒ¼ã‚¿å°‚ç”¨ãƒ¢ãƒ‡ãƒ« - ãƒˆãƒ¬ãƒ¼ãƒ‹ãƒ³ã‚°å®Œäº†')
print('=' * 60)
print(f'\nğŸ“Š ãƒ‡ãƒ¼ã‚¿:')
print(f'  - å®Ÿãƒ‡ãƒ¼ã‚¿: {len(feature_df)} samples (100%)')
print(f'  - ç‰¹å¾´é‡�æ¬¡å…ƒ: {X.shape[1]}')
print(f'\nğŸ�¯ Cross Validationçµ�æ�œ:')
print(f'  - LightGBM: {np.mean(cv_scores["lgb"]):.2%} Â± {np.std(cv_scores["lgb"]):.2%}')
print(f'  - XGBoost: {np.mean(cv_scores["xgb"]):.2%} Â± {np.std(cv_scores["xgb"]):.2%}')
print(f'  - CatBoost: {np.mean(cv_scores["cat"]):.2%} Â± {np.std(cv_scores["cat"]):.2%}')
print(f'  - Ensemble: {np.mean(cv_scores["ensemble"]):.2%} Â± {np.std(cv_scores["ensemble"]):.2%}')
print(f'\nğŸ’¾ ä¿�å­˜æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«:')
print(f'  - æœ€çµ‚ãƒ¢ãƒ‡ãƒ«: 3å€‹ï¼ˆLGB, XGB, CatBoostï¼‰')
print(f'  - Foldãƒ¢ãƒ‡ãƒ«: 15å€‹ï¼ˆ5-Fold Ã— 3ã‚¢ãƒ«ã‚´ãƒªã‚ºãƒ ï¼‰')
print(f'\nğŸš€ æ¬¡ã�®ã‚¹ãƒ†ãƒƒãƒ—:')
print(f'  1. ãƒ¢ãƒ‡ãƒ«ã‚’Kaggle Datasetã�¨ã�—ã�¦ã‚¢ãƒƒãƒ—ãƒ­ãƒ¼ãƒ‰')
print(f'  2. V11ï¼ˆå�ˆæˆ�ãƒ‡ãƒ¼ã‚¿ï¼‰+ V12ï¼ˆå®Ÿãƒ‡ãƒ¼ã‚¿ï¼‰ã�®ãƒ�ã‚¤ãƒ–ãƒªãƒƒãƒ‰ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«')
print(f'  3. Test Time Augmentation (TTA)ã�®å®Ÿè£…')
print(f'  4. æœ€çµ‚æ��å‡º')

