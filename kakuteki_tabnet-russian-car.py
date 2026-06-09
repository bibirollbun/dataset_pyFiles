!pip install pytorch_tabnet


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import re
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {train.columns.tolist()}")
print("\nSample data:")
print(train.head())
print("\nTest sample:")
print(test.head())

# ナンバープレートの特徴量エンジニアリング
def extract_plate_features(plate_str):
    """ロシアのナンバープレートから特徴量を抽出"""
    features = {}
    
    # 基本的な情報
    features['plate_length'] = len(str(plate_str))
    
    # 数字の個数
    features['num_digits'] = len(re.findall(r'\d', str(plate_str)))
    
    # 文字の個数
    features['num_letters'] = len(re.findall(r'[A-Za-zА-Яа-я]', str(plate_str)))
    
    # 特殊文字の個数
    features['num_special'] = len(re.findall(r'[^A-Za-zА-Яа-я0-9]', str(plate_str)))
    
    # 最初の文字が数字かどうか
    features['starts_with_digit'] = 1 if str(plate_str)[0].isdigit() else 0
    
    # 最後の文字が数字かどうか
    features['ends_with_digit'] = 1 if str(plate_str)[-1].isdigit() else 0
    
    # 地域コード（最後の数字部分を抽出）
    region_match = re.search(r'(\d+)$', str(plate_str))
    features['region_code'] = int(region_match.group(1)) if region_match else 0
    
    # 連続する数字のパターン
    digit_sequences = re.findall(r'\d+', str(plate_str))
    features['max_digit_sequence'] = max([len(seq) for seq in digit_sequences]) if digit_sequences else 0
    features['num_digit_sequences'] = len(digit_sequences)
    
    # 連続する文字のパターン
    letter_sequences = re.findall(r'[A-Za-zА-Яа-я]+', str(plate_str))
    features['max_letter_sequence'] = max([len(seq) for seq in letter_sequences]) if letter_sequences else 0
    features['num_letter_sequences'] = len(letter_sequences)
    
    # 特定のパターン（ロシアの一般的なナンバープレート形式）
    # X000XX000 形式
    features['is_standard_format'] = 1 if re.match(r'^[A-Za-zА-Яа-я]\d{3}[A-Za-zА-Яа-я]{2}\d{2,3}$', str(plate_str)) else 0
    
    return features

# 日付の特徴量エンジニアリング
def extract_date_features(date_str):
    """日付から特徴量を抽出"""
    features = {}
    
    try:
        # 複数の日付形式を試行
        for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%m/%d/%Y', '%Y/%m/%d']:
            try:
                date_obj = pd.to_datetime(date_str, format=fmt)
                break
            except:
                continue
        else:
            date_obj = pd.to_datetime(date_str)
        
        features['year'] = date_obj.year
        features['month'] = date_obj.month
        features['day'] = date_obj.day
        features['day_of_week'] = date_obj.dayofweek
        features['day_of_year'] = date_obj.dayofyear
        features['quarter'] = date_obj.quarter
        features['is_weekend'] = 1 if date_obj.dayofweek >= 5 else 0
        features['is_month_start'] = 1 if date_obj.day <= 5 else 0
        features['is_month_end'] = 1 if date_obj.day >= 25 else 0
        
    except:
        # デフォルト値
        features['year'] = 2020
        features['month'] = 1
        features['day'] = 1
        features['day_of_week'] = 0
        features['day_of_year'] = 1
        features['quarter'] = 1
        features['is_weekend'] = 0
        features['is_month_start'] = 0
        features['is_month_end'] = 0
    
    return features

# 特徴量エンジニアリング
def feature_engineering(df):
    """特徴量エンジニアリング"""
    df = df.copy()
    
    # ナンバープレートの特徴量抽出
    plate_features = df['plate'].apply(extract_plate_features)
    plate_df = pd.DataFrame(plate_features.tolist())
    
    # 日付の特徴量抽出
    date_features = df['date'].apply(extract_date_features)
    date_df = pd.DataFrame(date_features.tolist())
    
    # 元のデータフレームに結合
    df = pd.concat([df, plate_df, date_df], axis=1)
    
    # 地域コードのカテゴリ化（頻度ベース）
    region_counts = df['region_code'].value_counts()
    # 少数の地域コードは「その他」にまとめる
    rare_regions = region_counts[region_counts < 10].index
    df['region_category'] = df['region_code'].apply(lambda x: 'rare' if x in rare_regions else str(x))
    
    # ナンバープレートの文字列長カテゴリ
    df['plate_length_category'] = pd.cut(df['plate_length'], bins=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    
    return df

# 特徴量エンジニアリングの実行
print("Applying feature engineering...")
train_processed = feature_engineering(train)
test_processed = feature_engineering(test)

print(f"Processed train shape: {train_processed.shape}")
print(f"New features: {[col for col in train_processed.columns if col not in train.columns]}")

# カテゴリカル変数の処理
categorical_columns = ['region_category', 'plate_length_category']

# Label Encodingを安全に適用
label_encoders = {}
for col in categorical_columns:
    if col in train_processed.columns:
        le = LabelEncoder()
        
        # 訓練データで学習
        train_processed[col] = train_processed[col].astype(str).fillna('unknown')
        train_processed[col] = le.fit_transform(train_processed[col])
        label_encoders[col] = le
        
        # テストデータに適用
        if col in test_processed.columns:
            test_processed[col] = test_processed[col].astype(str).fillna('unknown')
            
            # 未知のカテゴリを処理
            unknown_mask = ~test_processed[col].isin(le.classes_)
            if unknown_mask.any():
                # 未知のカテゴリは最頻値で置換
                most_common_encoded = train_processed[col].mode()[0] if len(train_processed[col].mode()) > 0 else 0
                test_processed.loc[unknown_mask, col] = le.classes_[most_common_encoded] if most_common_encoded < len(le.classes_) else le.classes_[0]
            
            test_processed[col] = le.transform(test_processed[col])

# 特徴量とターゲットの分離
feature_columns = [col for col in train_processed.columns if col not in ['id', 'plate', 'date', 'price']]
X = train_processed[feature_columns].values.astype(np.float32)
y = train_processed['price'].values.astype(np.float32).reshape(-1, 1)  # TabNetでは2次元が必要
X_test = test_processed[feature_columns].values.astype(np.float32)

print(f"Number of features: {len(feature_columns)}")
print(f"Features: {feature_columns}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")

# データの正規化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 欠損値チェックと処理
if np.isnan(X_scaled).any():
    print("Warning: NaN values found, filling with 0")
    X_scaled = np.nan_to_num(X_scaled, 0)
if np.isnan(X_test_scaled).any():
    print("Warning: NaN values found in test, filling with 0")
    X_test_scaled = np.nan_to_num(X_test_scaled, 0)

# TabNet モデルの定義と訓練
def train_tabnet_model(X_train, y_train, X_val, y_val, random_state=42):
    """TabNetモデルの訓練"""
    model = TabNetRegressor(
        n_d=32,  # dimension of the prediction layer
        n_a=32,  # dimension of the attention layer
        n_steps=3,  # number of steps in the architecture
        gamma=1.3,  # coefficient for feature reusage in the masks
        n_independent=2,  # number of independent GLU layers
        n_shared=2,  # number of shared GLU layers
        lambda_sparse=1e-3,  # sparsity regularization
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type='entmax',
        scheduler_params=dict(step_size=50, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=random_state,
        verbose=0  # 詳細ログを無効化
    )
    
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        eval_metric=['mae'],
        max_epochs=100,
        patience=15,
        batch_size=256,
        virtual_batch_size=64,
        drop_last=False
    )
    
    return model

# K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X_scaled))
test_predictions = np.zeros((len(X_test_scaled), n_splits))
val_scores = []

print("Starting K-Fold Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    
    X_train_fold = X_scaled[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_scaled[val_idx]
    y_val_fold = y[val_idx]
    
    # モデル訓練
    model = train_tabnet_model(
        X_train_fold, y_train_fold, 
        X_val_fold, y_val_fold, 
        random_state=42 + fold
    )
    
    # 検証データでの予測
    val_pred = model.predict(X_val_fold).reshape(-1)  # 1次元に変換
    oof_predictions[val_idx] = val_pred
    
    # テストデータでの予測
    test_pred = model.predict(X_test_scaled).reshape(-1)  # 1次元に変換
    test_predictions[:, fold] = test_pred
    
    # スコア計算
    val_mae = mean_absolute_error(y_val_fold.reshape(-1), val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_fold.reshape(-1), val_pred))
    val_scores.append(val_mae)
    
    print(f"Fold {fold + 1} - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

# 全体のスコア
overall_mae = mean_absolute_error(y.reshape(-1), oof_predictions)
overall_rmse = np.sqrt(mean_squared_error(y.reshape(-1), oof_predictions))
print(f"\n=== Overall Performance ===")
print(f"CV MAE: {overall_mae:.4f} (+/- {np.std(val_scores):.4f})")
print(f"CV RMSE: {overall_rmse:.4f}")

# アンサンブル予測（平均）
ensemble_preds = np.mean(test_predictions, axis=1)

# 負の値をクリップ
ensemble_preds = np.clip(ensemble_preds, a_min=0, a_max=None)

print(f"\nPrediction statistics:")
print(f"Min: {ensemble_preds.min():.2f}")
print(f"Max: {ensemble_preds.max():.2f}")
print(f"Mean: {ensemble_preds.mean():.2f}")
print(f"Median: {np.median(ensemble_preds):.2f}")

# 提出用ファイルの作成
test_df = test.copy()  # 元のtest DataFrameを保持
submission = pd.DataFrame({
    'id': test_df['id'],  # Ensure 'id' column is taken from the original test_df
    'price': ensemble_preds
})

# Save the submission file to a CSV in the required format.
submission.to_csv('submission1.csv', index=False)
print('\n✅  Submission file "submission.csv" saved successfully.')
print(f"Submission shape: {submission.shape}")
print(submission.head())

# 特徴量重要度の表示（最後のモデルから）
try:
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== Top 10 Feature Importances ===")
        print(feature_importance.head(10))
except:
    print("Feature importance not available")


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import re
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')

# 特徴量抽出

def extract_plate_features(plate_str):
    features = {}
    plate = str(plate_str)
    features['plate_length'] = len(plate)
    features['num_digits'] = len(re.findall(r'\d', plate))
    features['num_letters'] = len(re.findall(r'[A-Za-zА-Яа-я]', plate))
    features['num_special'] = len(re.findall(r'[^A-Za-zА-Яа-я0-9]', plate))
    features['starts_with_digit'] = plate[0].isdigit()
    features['ends_with_digit'] = plate[-1].isdigit()
    match = re.search(r'(\d+)$', plate)
    features['region_code'] = int(match.group(1)) if match else 0
    digit_seq = re.findall(r'\d+', plate)
    letter_seq = re.findall(r'[A-Za-zА-Яа-я]+', plate)
    features['max_digit_sequence'] = max((len(s) for s in digit_seq), default=0)
    features['num_digit_sequences'] = len(digit_seq)
    features['max_letter_sequence'] = max((len(s) for s in letter_seq), default=0)
    features['num_letter_sequences'] = len(letter_seq)
    features['is_standard_format'] = int(bool(re.match(r'^[A-Za-zА-Яа-я]\d{3}[A-Za-zА-Яа-я]{2}\d{2,3}$', plate)))
    return features

def extract_date_features(date_str):
    features = {}
    date_obj = pd.to_datetime(date_str, errors='coerce')
    features['year'] = date_obj.year
    features['month'] = date_obj.month
    features['day'] = date_obj.day
    features['day_of_week'] = date_obj.dayofweek
    features['day_of_year'] = date_obj.dayofyear
    features['quarter'] = date_obj.quarter
    features['is_weekend'] = int(date_obj.dayofweek >= 5)
    features['is_month_start'] = int(date_obj.day <= 5)
    features['is_month_end'] = int(date_obj.day >= 25)
    return features

def feature_engineering(df):
    plate_features = df['plate'].apply(extract_plate_features).apply(pd.Series)
    date_features = df['date'].apply(extract_date_features).apply(pd.Series)
    df = pd.concat([df, plate_features, date_features], axis=1)
    df['region_category'] = pd.qcut(df['region_code'], q=4, labels=False)
    df['plate_length_category'] = pd.cut(df['plate_length'], bins=[0, 7, 9, 15], labels=False)
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# カテゴリ変数をラベルエンコード
categorical_cols = ['region_category', 'plate_length_category']
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 学習データと目的変数
X = train.drop(columns=['id', 'plate', 'date', 'price'])
y = train['price'].values.reshape(-1, 1)
X_test = test.drop(columns=['id', 'plate', 'date', 'price'])

# float32 に変換
X = X.astype('float32')
X_test = X_test.astype('float32')

# TabNet モデル
kf = KFold(n_splits=5, shuffle=True, random_state=42)

preds = np.zeros((X_test.shape[0],))
cv_mae, cv_rmse = [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    print(f"--- Fold {fold + 1}/5 ---")
    X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_train, y_val = y[tr_idx], y[val_idx]

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')

    model = TabNetRegressor(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        lambda_sparse=1e-4, seed=42,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0
    )

    model.fit(
        X_train=X_train.values, y_train=y_train,
        eval_set=[(X_val.values, y_val)],
        eval_metric=['mae'],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    val_preds = model.predict(X_val.values).reshape(-1)
    test_preds = model.predict(X_test.values).reshape(-1)
    preds += test_preds / kf.n_splits

    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold+1} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    cv_mae.append(mae)
    cv_rmse.append(rmse)

print("\n=== Overall Performance ===")
print(f"CV MAE: {np.mean(cv_mae):.4f} (+/- {np.std(cv_mae):.4f})")
print(f"CV RMSE: {np.mean(cv_rmse):.4f}\n")

# 出力
submission = test[['id']].copy()
submission['price'] = preds
submission.to_csv("submission2.csv", index=False)
print("✅ Submission file \"submission2.csv\" saved successfully.")


