# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

# 导入必要的库
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline

from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import polars as pl


# 自定义评估指标
class CompetitionMetric:
    def calculate_hierarchical_f1(self, true_df, pred_df):
        """计算层次化F1分数"""
        # 这里简化实现，实际竞赛中可能有更复杂的层次结构
        return f1_score(true_df['gesture'], pred_df['gesture'], average='macro')

# 设置参数
TRAIN = True                     
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/pretrained-model")  
EXPORT_DIR = Path("./")                                    
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 1e-3
WD = 3e-4
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40
N_OUTER_FOLDS = 5  # 外层交叉验证折数
N_INNER_FOLDS = 3  # 内层交叉验证折数

print("▶ imports ready · tensorflow", tf.__version__)


# Tensor操作函数
def time_sum(x):
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

# Residual CNN Block with SE
def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context


# Normalizes and cleans the time series sequence. 
def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

# MixUp数据增强
class MixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
    
    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx], self.y[idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        return X_mix, y_mix
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# 构建双流模型
def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim+tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMU deep branch
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.3, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.3, wd=wd)

    # TOF/Thermal lighter branch
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.3)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.3)(x2)

    merged = Concatenate()([x1, x2])

    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)



# 构建传统LSTM模型
def build_lstm_model(pad_len, feature_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, feature_dim))
    
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(wd))(inp)
    x = Dropout(0.3)(x)
    x = LSTM(128, kernel_regularizer=l2(wd))(x)
    x = Dropout(0.3)(x)
    
    x = Dense(256, use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)

# 特征提取函数 - 用于传统机器学习模型
def extract_features(X_sequences):
    """从序列数据中提取统计特征"""
    features = []
    for seq in X_sequences:
        # 计算每个特征维度的统计量
        mean = np.mean(seq, axis=0)
        std = np.std(seq, axis=0)
        min_val = np.min(seq, axis=0)
        max_val = np.max(seq, axis=0)
        median = np.median(seq, axis=0)
        skewness = np.mean(((seq - mean) / std) ** 3, axis=0)
        kurtosis = np.mean(((seq - mean) / std) ** 4, axis=0) - 3
        
        # 组合所有特征
        combined = np.hstack([mean, std, min_val, max_val, median, skewness, kurtosis])
        features.append(combined)
    
    return np.array(features)

# 嵌套交叉验证评估模型
def nested_cross_validation(X, y, model_type, param_grid=None):
    """
    执行嵌套交叉验证评估模型性能
    
    参数:
    X: 特征数据
    y: 标签数据
    model_type: 模型类型 ('two_branch', 'lstm', 'xgb')
    param_grid: 超参数网格
    
    返回:
    每个外层折叠的评估结果
    """
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)
    results = []
    
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n=== 外层折叠 {i+1}/{N_OUTER_FOLDS} ===")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 内层交叉验证用于超参数优化
        inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=42)
        
        if model_type == 'two_branch':
            # 为简化示例，这里只使用固定参数
            model = build_two_branch_model(
                X_train.shape[1], 
                len(imu_cols), 
                len(tof_cols), 
                len(np.unique(y)),
                wd=WD
            )
            steps = len(X_train)//BATCH_SIZE
            lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)
            model.compile(
                optimizer=Adam(lr_sched),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            train_gen = MixupGenerator(X_train, to_categorical(y_train), BATCH_SIZE, MIXUP_ALPHA)
            cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
            model.fit(
                train_gen, 
                epochs=EPOCHS, 
                validation_data=(X_test, to_categorical(y_test)),
                callbacks=[cb],
                verbose=0
            )
            y_pred = model.predict(X_test).argmax(1)
            
        elif model_type == 'lstm':
            # 为简化示例，这里只使用固定参数
            model = build_lstm_model(
                X_train.shape[1], 
                X_train.shape[2], 
                len(np.unique(y)),
                wd=WD
            )
            steps = len(X_train)//BATCH_SIZE
            lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)
            model.compile(
                optimizer=Adam(lr_sched),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            train_gen = MixupGenerator(X_train, to_categorical(y_train), BATCH_SIZE, MIXUP_ALPHA)
            cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
            model.fit(
                train_gen, 
                epochs=EPOCHS, 
                validation_data=(X_test, to_categorical(y_test)),
                callbacks=[cb],
                verbose=0
            )
            y_pred = model.predict(X_test).argmax(1)
            
        elif model_type == 'xgb':
            # 提取统计特征用于XGBoost
            X_train_features = extract_features(X_train)
            X_test_features = extract_features(X_test)
            
            if param_grid:
                # 执行网格搜索
                search = GridSearchCV(
                    estimator=xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False, eval_metric='merror'),
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring='f1_macro',
                    n_jobs=-1
                )
                search.fit(X_train_features, y_train)
                model = search.best_estimator_
                print(f"最佳参数: {search.best_params_}")
            else:
                model = xgb.XGBClassifier(
                    objective='multi:softmax',
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric='merror'
                )
                model.fit(X_train_features, y_train)
                
            y_pred = model.predict(X_test_features)
        
        # 评估模型
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"准确率: {acc:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"混淆矩阵:\n{cm}")
        
        results.append({
            'accuracy': acc,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred
        })
    
    return results

    


def build_xgb_model(X_train, y_train, param_grid=None, inner_cv=None):
    """
    封装 XGBoost 模型的训练过程，支持可选的网格搜索
    返回训练好的模型
    """
    # 提取统计特征
    X_train_features = extract_features(X_train)
    
    if param_grid and inner_cv:
        search = GridSearchCV(
            estimator=xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False, eval_metric='merror'),
            param_grid=param_grid,
            cv=inner_cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        search.fit(X_train_features, y_train)
        model = search.best_estimator_
        print(f"XGB 最佳参数: {search.best_params_}")
    else:
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='merror'
        )
        model.fit(X_train_features, y_train)
    
    return model



def nested_cross_validation(X, y, model_type, param_grid=None):
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)
    results = []

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n=== 外层折叠 {i+1}/{N_OUTER_FOLDS} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=42)

        if model_type == 'two_branch':
            model = build_two_branch_model(X_train.shape[1], len(imu_cols), len(tof_cols), len(np.unique(y)), wd=WD)
            steps = len(X_train)//BATCH_SIZE
            lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)
            model.compile(optimizer=Adam(lr_sched), loss='categorical_crossentropy', metrics=['accuracy'])
            train_gen = MixupGenerator(X_train, to_categorical(y_train), BATCH_SIZE, MIXUP_ALPHA)
            cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
            model.fit(train_gen, epochs=EPOCHS, validation_data=(X_test, to_categorical(y_test)), callbacks=[cb], verbose=0)
            y_pred = model.predict(X_test).argmax(1)

        elif model_type == 'lstm':
            model = build_lstm_model(X_train.shape[1], X_train.shape[2], len(np.unique(y)), wd=WD)
            steps = len(X_train)//BATCH_SIZE
            lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)
            model.compile(optimizer=Adam(lr_sched), loss='categorical_crossentropy', metrics=['accuracy'])
            train_gen = MixupGenerator(X_train, to_categorical(y_train), BATCH_SIZE, MIXUP_ALPHA)
            cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
            model.fit(train_gen, epochs=EPOCHS, validation_data=(X_test, to_categorical(y_test)), callbacks=[cb], verbose=0)
            y_pred = model.predict(X_test).argmax(1)

        elif model_type == 'xgb':
            model = build_xgb_model(X_train, y_train, param_grid, inner_cv)
            X_test_features = extract_features(X_test)
            y_pred = model.predict(X_test_features)

        # 评估
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        print(f"准确率: {acc:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")

        results.append({
            'accuracy': acc,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred
        })

    return results



if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")

    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation', 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")

    scaler = StandardScaler().fit(df[feature_cols].ffill().bfill().fillna(0).values)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")

    seq_gp = df.groupby('sequence_id')
    X_list, y_list, lens = [], [], []
    for seq_id, seq in seq_gp:
        mat = preprocess_sequence(seq, feature_cols, scaler)
        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])
        lens.append(len(mat))

    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))

    X = pad_sequences(X_list, maxlen=pad_len, padding='post', truncating='post', dtype='float32')
    y = np.array(y_list)

    cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y)
    class_weight = dict(enumerate(cw_vals))

    models_to_evaluate = {
        'two_branch': '双流CNN+BiLSTM+注意力机制',
        'lstm': '传统LSTM模型',
        'xgb': 'XGBoost'
    }

    xgb_param_grid = {'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}

    all_results = {}
    for model_type, model_name in models_to_evaluate.items():
        print(f"\n===== 评估模型: {model_name} =====")
        results = nested_cross_validation(X, y, model_type, param_grid=xgb_param_grid if model_type == 'xgb' else None)
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1_macro'] for r in results])
        print(f"{model_name} 平均准确率: {avg_acc:.4f}")
        print(f"{model_name} 平均Macro F1: {avg_f1:.4f}")
        all_results[model_type] = {'results': results, 'avg_accuracy': avg_acc, 'avg_f1_macro': avg_f1}

    print("\n===== 模型性能比较 =====")
    for model_type, model_name in models_to_evaluate.items():
        print(f"{model_name}: 准确率={all_results[model_type]['avg_accuracy']:.4f}, Macro F1={all_results[model_type]['avg_f1_macro']:.4f}")

    best_model_type = max(all_results, key=lambda k: all_results[k]['avg_f1_macro'])
    print(f"\n最佳模型: {models_to_evaluate[best_model_type]}")

    if best_model_type == 'two_branch':
        final_model = build_two_branch_model(pad_len, len(imu_cols), len(tof_cols), len(le.classes_), wd=WD)
        steps = len(X)//BATCH_SIZE
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)
        final_model.compile(optimizer=Adam(lr_sched), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
        train_gen = MixupGenerator(X, to_categorical(y), batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
        cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1)
        final_model.fit(train_gen, epochs=EPOCHS, callbacks=[cb], verbose=1)
        final_model.save(EXPORT_DIR / "best_model.h5")
        
    elif best_model_type == 'lstm':
        final_model = build_lstm_model(pad_len, X.shape[2], len(le.classes_), wd=WD)
        steps = len(X)//BATCH_SIZE
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)
        final_model.compile(optimizer=Adam(lr_sched), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
        train_gen = MixupGenerator(X, to_categorical(y), batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
        cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1)
        final_model.fit(train_gen, epochs=EPOCHS, callbacks=[cb], verbose=1)
        final_model.save(EXPORT_DIR / "best_model.h5")
        
    elif best_model_type == 'xgb':
        final_model = build_xgb_model(X, y)  # 直接使用封装好的XGB函数
        joblib.dump(final_model, EXPORT_DIR / "best_model.pkl")

    print("✔ Training done – 最佳模型已保存")



# 主程序
if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")

    # 标签编码
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    # 特征列表
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    imu_cols  = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols  = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")

    # 全局标准化器
    scaler = StandardScaler().fit(df[feature_cols].ffill().bfill().fillna(0).values)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")

    # 构建序列
    seq_gp = df.groupby('sequence_id')
    X_list, y_list, lens = [], [], []
    for seq_id, seq in seq_gp:
        mat = preprocess_sequence(seq, feature_cols, scaler)
        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])
        lens.append(len(mat))
    
    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))

    X = pad_sequences(X_list, maxlen=pad_len, padding='post', truncating='post', dtype='float32')
    y = np.array(y_list)
    
    # 原始代码中的训练/验证分割
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 类别权重
    cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y)
    class_weight = dict(enumerate(cw_vals))

    # 定义要评估的模型
    models_to_evaluate = {
        'two_branch': '双流CNN+BiLSTM+注意力机制',
        'lstm': '传统LSTM模型',
        'xgb': 'XGBoost'
    }
    
    # XGBoost超参数网格
    xgb_param_grid = {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200]
    }
    
    # 存储所有模型的评估结果
    all_results = {}
    
    # 对每个模型进行嵌套交叉验证
    for model_type, model_name in models_to_evaluate.items():
        print(f"\n===== 评估模型: {model_name} =====")
        results = nested_cross_validation(X, y, model_type, 
                                         param_grid=xgb_param_grid if model_type == 'xgb' else None)
        
        # 计算平均性能
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1_macro'] for r in results])
        
        print(f"\n{model_name} 平均准确率: {avg_acc:.4f}")
        print(f"{model_name} 平均Macro F1: {avg_f1:.4f}")
        
        all_results[model_type] = {
            'results': results,
            'avg_accuracy': avg_acc,
            'avg_f1_macro': avg_f1
        }
    
    # 比较模型性能
    print("\n===== 模型性能比较 =====")
    for model_type, model_name in models_to_evaluate.items():
        print(f"{model_name}: 准确率={all_results[model_type]['avg_accuracy']:.4f}, Macro F1={all_results[model_type]['avg_f1_macro']:.4f}")
    
    # 选择最佳模型并使用全部数据训练
    best_model_type = max(all_results, key=lambda k: all_results[k]['avg_f1_macro'])
    print(f"\n最佳模型: {models_to_evaluate[best_model_type]}")
    
    if best_model_type == 'two_branch':
        final_model = build_two_branch_model(pad_len, len(imu_cols), len(tof_cols), len(le.classes_), wd=WD)
        steps = len(X)//BATCH_SIZE
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)
        final_model.compile(
            optimizer=Adam(lr_sched),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        train_gen = MixupGenerator(X, to_categorical(y), batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
        cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1)
        final_model.fit(train_gen, epochs=EPOCHS, callbacks=[cb], verbose=1)
        final_model.save(EXPORT_DIR / "best_model.h5")
        
    elif best_model_type == 'lstm':
        final_model = build_lstm_model(pad_len, X.shape[2], len(le.classes_), wd=WD)
        steps = len(X)//BATCH_SIZE
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)
        final_model.compile(
            optimizer=Adam(lr_sched),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        train_gen = MixupGenerator(X, to_categorical(y), batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
        cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1)
        final_model.fit(train_gen, epochs=EPOCHS, callbacks=[cb], verbose=1)
        final_model.save(EXPORT_DIR / "best_model.h5")
        
    elif best_model_type == 'xgb':
        # 提取特征并训练
        X_features = extract_features(X)
        final_model = xgb.XGBClassifier(
            objective='multi:softmax',
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='merror'
        )
        final_model.fit(X_features, y)
        joblib.dump(final_model, EXPORT_DIR / "best_model.pkl")
    
    print("✔ Training done – 最佳模型已保存")

else:
    # 推理模式保持不变
    print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
    feature_cols   = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]

    # 加载模型
    try:
        # 尝试加载深度学习模型
        custom_objs = {
            'time_sum': time_sum,
            'squeeze_last_axis': squeeze_last_axis,
            'expand_last_axis': expand_last_axis,
            'se_block': se_block,
            'residual_se_cnn_block': residual_se_cnn_block,
            'attention_layer': attention_layer,
        }
        model = load_model(PRETRAINED_DIR / "best_model.h5",
                           compile=False, custom_objects=custom_objs)
        model_type = 'dl'
    except:
        # 尝试加载XGBoost模型
        model = joblib.load(PRETRAINED_DIR / "best_model.pkl")
        model_type = 'xgb'
    
    print("  model, scaler, pads loaded – ready for evaluation")

