import numpy as np
import pandas as pd
import pickle
import gc
import warnings
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

print('Loading data...')
train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')

gesture_labels = sorted(train_df['gesture'].unique())
num_classes = len(gesture_labels)

print(f'Classes: {num_classes}')
print(f'Total samples: {len(train_df)}')
print(f'Unique sequences: {train_df["sequence_id"].nunique()}')


def remove_gravity_from_acc(df_seq):
    """é‡�åŠ›é™¤å�»åŠ é€Ÿåº¦ã�®è¨ˆç®—ï¼ˆãƒˆãƒƒãƒ—ã‚½ãƒªãƒ¥ãƒ¼ã‚·ãƒ§ãƒ³ã�‹ã‚‰ï¼‰"""
    acc_values = df_seq[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = df_seq[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    
    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    
    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]
    
    return linear_accel

def calculate_angular_velocity_from_quat(df_seq, time_delta=1/200):
    """ã‚¯ã‚©ãƒ¼ã‚¿ãƒ‹ã‚ªãƒ³ã�‹ã‚‰è§’é€Ÿåº¦ã‚’è¨ˆç®—"""
    quat_values = df_seq[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))
    
    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]
        
        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue
        
        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
    
    return angular_vel

def calculate_angular_distance(df_seq):
    """è§’è·�é›¢ã�®è¨ˆç®—"""
    quat_values = df_seq[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)
    
    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]
        
        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
    
    return angular_dist


def extract_advanced_features(df_seq):
    """ãƒˆãƒƒãƒ—ã‚½ãƒªãƒ¥ãƒ¼ã‚·ãƒ§ãƒ³ãƒ™ãƒ¼ã‚¹ã�®ç‰¹å¾´é‡�æŠ½å‡º"""
    features = {}
    
    # åŸºæœ¬çµ±è¨ˆé‡�
    for col in ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']:
        if col in df_seq.columns:
            values = df_seq[col].values
            features[f'{col}_mean'] = np.mean(values)
            features[f'{col}_std'] = np.std(values)
            features[f'{col}_max'] = np.max(values)
            features[f'{col}_min'] = np.min(values)
            features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
            features[f'{col}_skew'] = stats.skew(values)
            features[f'{col}_kurtosis'] = stats.kurtosis(values)
            
            # å¤‰åŒ–é‡�çµ±è¨ˆ
            diff = np.diff(values)
            if len(diff) > 0:
                features[f'{col}_diff_mean'] = np.mean(diff)
                features[f'{col}_diff_std'] = np.std(diff)
    
    # é‡�åŠ›é™¤å�»ç·šå½¢åŠ é€Ÿåº¦
    linear_accel = remove_gravity_from_acc(df_seq)
    df_seq['linear_acc_x'] = linear_accel[:, 0]
    df_seq['linear_acc_y'] = linear_accel[:, 1]
    df_seq['linear_acc_z'] = linear_accel[:, 2]
    
    # ç·šå½¢åŠ é€Ÿåº¦ãƒ�ã‚°ãƒ‹ãƒ�ãƒ¥ãƒ¼ãƒ‰
    df_seq['linear_acc_mag'] = np.sqrt(
        df_seq['linear_acc_x']**2 + 
        df_seq['linear_acc_y']**2 + 
        df_seq['linear_acc_z']**2
    )
    
    # ã‚¸ãƒ£ãƒ¼ã‚¯ï¼ˆç·šå½¢åŠ é€Ÿåº¦ãƒ�ã‚°ãƒ‹ãƒ�ãƒ¥ãƒ¼ãƒ‰ã�®å¤‰åŒ–ç�‡ï¼‰
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
    
    # è§’é€Ÿåº¦
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq['angular_vel_x'] = angular_vel[:, 0]
    df_seq['angular_vel_y'] = angular_vel[:, 1]
    df_seq['angular_vel_z'] = angular_vel[:, 2]
    
    # è§’è·�é›¢
    df_seq['angular_distance'] = calculate_angular_distance(df_seq)
    
    # é«˜åº¦ã�ªç‰¹å¾´é‡�ã�®çµ±è¨ˆ
    advanced_cols = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag',
                     'linear_acc_mag_jerk', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                     'angular_distance']
    
    for col in advanced_cols:
        if col in df_seq.columns:
            values = df_seq[col].values
            features[f'{col}_mean'] = np.mean(values)
            features[f'{col}_std'] = np.std(values)
            features[f'{col}_max'] = np.max(np.abs(values))
            features[f'{col}_energy'] = np.sum(values**2)
    
    # ToFã‚»ãƒ³ã‚µãƒ¼è¦�ç´„ï¼ˆãƒˆãƒƒãƒ—ã‚½ãƒªãƒ¥ãƒ¼ã‚·ãƒ§ãƒ³ã�‹ã‚‰ï¼‰
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        if all(col in df_seq.columns for col in pixel_cols[:10]):  # Check if exists
            tof_data = df_seq[pixel_cols].replace(-1, np.nan)
            features[f'tof_{i}_mean'] = tof_data.mean().mean()
            features[f'tof_{i}_std'] = tof_data.std().mean()
            features[f'tof_{i}_min'] = tof_data.min().min()
            features[f'tof_{i}_max'] = tof_data.max().max()
    
    # æ¸©åº¦ã‚»ãƒ³ã‚µãƒ¼
    temp_cols = [f'thm_{i}' for i in range(1, 6)]
    temp_cols = [col for col in temp_cols if col in df_seq.columns]
    if temp_cols:
        temp_data = df_seq[temp_cols]
        features['temp_mean'] = temp_data.mean().mean()
        features['temp_std'] = temp_data.std().mean()
        features['temp_range'] = temp_data.max().max() - temp_data.min().min()
    
    # ã‚·ãƒ¼ã‚±ãƒ³ã‚¹é•·
    features['sequence_length'] = len(df_seq)
    
    return features


# ç‰¹å¾´é‡�æŠ½å‡ºï¼ˆå…¨ãƒ‡ãƒ¼ã‚¿ï¼‰
print('Extracting advanced features from all sequences...')
X_features = []
y_labels = []
groups = []

le = LabelEncoder()
le.fit(gesture_labels)

sequence_ids = train_df['sequence_id'].unique()
total_sequences = len(sequence_ids)

# ãƒ�ãƒƒãƒ�å‡¦ç�†ã�§åŠ¹ç�‡åŒ–
batch_size = 500
for batch_start in range(0, total_sequences, batch_size):
    batch_end = min(batch_start + batch_size, total_sequences)
    batch_seq_ids = sequence_ids[batch_start:batch_end]
    
    if batch_start % 1000 == 0:
        print(f'Processing {batch_start}/{total_sequences}...')
    
    for seq_id in batch_seq_ids:
        seq_data = train_df[train_df['sequence_id'] == seq_id]
        
        # é«˜åº¦ã�ªç‰¹å¾´é‡�æŠ½å‡º
        features = extract_advanced_features(seq_data)
        X_features.append(features)
        
        # ãƒ©ãƒ™ãƒ«ã�¨ã‚°ãƒ«ãƒ¼ãƒ—
        gesture = seq_data['gesture'].iloc[0]
        label = le.transform([gesture])[0]
        y_labels.append(label)
        
        subject = seq_data['subject'].iloc[0]
        groups.append(subject)
    
    # ãƒ¡ãƒ¢ãƒªè§£æ”¾
    if batch_start % 2000 == 0:
        gc.collect()

X_df = pd.DataFrame(X_features)
y_array = np.array(y_labels)
groups_array = np.array(groups)

print(f'Feature shape: {X_df.shape}')
print(f'Number of features: {len(X_df.columns)}')

# æ¬ æ��å€¤å‡¦ç�†
X_df = X_df.fillna(0)

# ç„¡é™�å€¤ã�®å‡¦ç�†
X_df = X_df.replace([np.inf, -np.inf], 0)


# StratifiedGroupKFoldã�§äº¤å·®æ¤œè¨¼
print('Training models with StratifiedGroupKFold...')
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

val_scores = []
models_lgb = []
models_xgb = []
models_cat = []

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_df, y_array, groups_array)):
    print(f'\n=== Fold {fold + 1}/5 ===')
    
    X_train = X_df.iloc[train_idx]
    X_val = X_df.iloc[val_idx]
    y_train = y_array[train_idx]
    y_val = y_array[val_idx]
    
    # ã‚¯ãƒ©ã‚¹ã‚¦ã‚§ã‚¤ãƒˆè¨ˆç®—
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # LightGBMï¼ˆæœ€é�©åŒ–ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ï¼‰
    lgb_params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 100,
        'learning_rate': 0.03,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42 + fold
    }
    
    # ã‚µãƒ³ãƒ—ãƒ«ã‚¦ã‚§ã‚¤ãƒˆ
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    valid_data = lgb.Dataset(X_val, label=y_val)
    
    model_lgb = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )
    models_lgb.append(model_lgb)
    
    # XGBoost
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'eval_metric': 'mlogloss',
        'seed': 42 + fold,
        'tree_method': 'hist'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dval, 'val')],
        early_stopping_rounds=150,
        verbose_eval=False
    )
    models_xgb.append(model_xgb)
    
    # CatBoost
    cat_params = {
        'loss_function': 'MultiClass',
        'iterations': 1500,
        'learning_rate': 0.03,
        'depth': 7,
        'l2_leaf_reg': 3,
        'random_seed': 42 + fold,
        'verbose': False,
        'early_stopping_rounds': 150,
        'class_weights': class_weight_dict
    }
    
    train_pool = cb.Pool(X_train, y_train)
    val_pool = cb.Pool(X_val, y_val)
    
    model_cat = cb.CatBoostClassifier(**cat_params)
    model_cat.fit(train_pool, eval_set=val_pool, verbose=False)
    models_cat.append(model_cat)
    
    # æ¤œè¨¼ç²¾åº¦è¨ˆç®—
    pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    pred_xgb = model_xgb.predict(dval)
    pred_cat = model_cat.predict_proba(X_val)
    
    # ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ï¼ˆãƒˆãƒƒãƒ—ã‚½ãƒªãƒ¥ãƒ¼ã‚·ãƒ§ãƒ³ã�®é‡�ã�¿ï¼‰
    pred_ensemble = 0.5 * pred_lgb + 0.2 * pred_xgb + 0.3 * pred_cat
    pred_class = np.argmax(pred_ensemble, axis=1)
    accuracy = np.mean(pred_class == y_val)
    
    print(f'Fold {fold + 1} Ensemble Accuracy: {accuracy:.4f}')
    val_scores.append(accuracy)

# æœ€çµ‚çµ�æ�œ
mean_accuracy = np.mean(val_scores)
std_accuracy = np.std(val_scores)
print(f'\n=== Final Results ===')
print(f'Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})')
print(f'Target 87%: {"âœ… ACHIEVED!!!" if mean_accuracy >= 0.87 else "â�Œ NOT YET"}')


# ãƒ¢ãƒ‡ãƒ«ä¿�å­˜
print('\nSaving models...')

# å�„foldã�®ãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜
for i, model in enumerate(models_lgb):
    with open(f'lgb_model_v9_fold{i}.pkl', 'wb') as f:
        pickle.dump(model, f)

for i, model in enumerate(models_xgb):
    with open(f'xgb_model_v9_fold{i}.pkl', 'wb') as f:
        pickle.dump(model, f)

for i, model in enumerate(models_cat):
    model.save_model(f'cat_model_v9_fold{i}.cbm')

# Label encoder
with open('label_encoder_v9.pkl', 'wb') as f:
    pickle.dump(le, f)

print('Models saved successfully')
print(f'\nğŸ�¯ Final Performance: {mean_accuracy:.4f}')


# CMIæ�¨è«–ã‚µãƒ¼ãƒ�ãƒ¼
import sys
sys.path.append('/kaggle/input/cmi-detect-behavior-with-sensor-data')
from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

def predict(sequence, demographics):
    """CMIæ�¨è«–ã‚µãƒ¼ãƒ�ãƒ¼ç”¨äºˆæ¸¬é–¢æ•°ï¼ˆãƒˆãƒƒãƒ—ã‚½ãƒªãƒ¥ãƒ¼ã‚·ãƒ§ãƒ³ãƒ™ãƒ¼ã‚¹ï¼‰"""
    try:
        # DataFrameã�«å¤‰æ�›
        df_seq = pd.DataFrame(sequence)
        
        # é«˜åº¦ã�ªç‰¹å¾´é‡�æŠ½å‡º
        features = extract_advanced_features(df_seq)
        X_feat = pd.DataFrame([features])
        X_feat = X_feat.fillna(0)
        X_feat = X_feat.replace([np.inf, -np.inf], 0)
        
        # å…¨foldã�®äºˆæ¸¬ã‚’å¹³å�‡
        all_preds = []
        
        # LightGBMäºˆæ¸¬
        for model in models_lgb:
            pred = model.predict(X_feat, num_iteration=model.best_iteration)
            all_preds.append(pred[0])
        
        # XGBoostäºˆæ¸¬
        dtest = xgb.DMatrix(X_feat)
        for model in models_xgb:
            pred = model.predict(dtest)
            all_preds.append(pred[0])
        
        # CatBoostäºˆæ¸¬
        for model in models_cat:
            pred = model.predict_proba(X_feat)
            all_preds.append(pred[0])
        
        # é‡�ã�¿ä»˜ã��ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ï¼ˆãƒˆãƒƒãƒ—ã‚½ãƒªãƒ¥ãƒ¼ã‚·ãƒ§ãƒ³ã�®é‡�ã�¿ï¼‰
        # LightGBM: 50%, XGBoost: 20%, CatBoost: 30%
        n_lgb = len(models_lgb)
        n_xgb = len(models_xgb)
        n_cat = len(models_cat)
        
        pred_lgb = np.mean(all_preds[:n_lgb], axis=0)
        pred_xgb = np.mean(all_preds[n_lgb:n_lgb+n_xgb], axis=0)
        pred_cat = np.mean(all_preds[n_lgb+n_xgb:], axis=0)
        
        pred_final = 0.5 * pred_lgb + 0.2 * pred_xgb + 0.3 * pred_cat
        
        # äºˆæ¸¬ã‚¯ãƒ©ã‚¹
        pred_class = np.argmax(pred_final)
        pred_gesture = le.inverse_transform([pred_class])[0]
        
        return pred_gesture
        
    except Exception as e:
        print(f'Prediction error: {e}')
        return 'Text on phone'  # Most common class

print('Starting CMI Inference Server...')
server = CMIInferenceServer(predict)
server.serve()
print('Inference complete')

