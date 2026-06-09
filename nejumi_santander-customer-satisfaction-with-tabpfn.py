
# TabPFN と依存ライブラリのインストール（必要に応じて実行）
!pip install -U tabpfn
# アンサンブルや拡張機能を使う場合（任意）
!pip install -U tabpfn_extensions




import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample

from tabpfn import TabPFNClassifier




# Kaggle の API でダウンロードした train.csv/test.csv を作業ディレクトリに配置してください
train_path = '/kaggle/input/santander-customer-satisfaction/train.csv'
test_path = '/kaggle/input/santander-customer-satisfaction/test.csv'

# データの読み込み
def load_data(train_path: str, test_path: str):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError('train.csv または test.csv が見つかりません。ファイルをアップロードしてください。')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

train_df, test_df = load_data(train_path, test_path)
print('train shape:', train_df.shape)
print('test shape:', test_df.shape)
print('Class distribution in full training data:')
print(train_df['TARGET'].value_counts(normalize=True))
# 最初の数行を確認
a = train_df.head()
a




# 不均衡データに対するアンダーサンプリング関数
# max_total: サンプル後の最大行数
# random_state: サンプリング時のランダムシード

def undersample_training_data(df: pd.DataFrame, target_col: str, max_total: int = 10000, random_state: int = 42):
    class_counts = df[target_col].value_counts()
    # クラスを識別
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    n_minority = class_counts[minority_class]

    # max_total の半分より少ない場合は minority も downsample する
    max_per_class = max_total // 2
    n_minority_sample = min(n_minority, max_per_class)
    n_majority_sample = min(class_counts[majority_class], max_total - n_minority_sample)

    df_minority = df[df[target_col] == minority_class]
    df_majority = df[df[target_col] == majority_class]

    if n_minority_sample < n_minority:
        df_minority = resample(df_minority, replace=False, n_samples=n_minority_sample, random_state=random_state)
    if n_majority_sample < class_counts[majority_class]:
        df_majority = resample(df_majority, replace=False, n_samples=n_majority_sample, random_state=random_state)

    sampled_df = pd.concat([df_minority, df_majority]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df

# サンプリング実施例（乱数固定）
sampled_df = undersample_training_data(train_df, 'TARGET', max_total=10000, random_state=42)
print('Sampled shape:', sampled_df.shape)
print('Class distribution in sample:')
print(sampled_df['TARGET'].value_counts())



# under‑sampling + bagging による TabPFN モデリング（テスト推論をバッチ処理にする）
import torch
import gc

n_runs = 3
max_total = 10000     # 各学習データの最大行数
batch_size = 5000     # テストデータのバッチサイズ（GPUメモリに合わせて調整）

test_predictions = np.zeros(len(test_df))

for i in range(n_runs):
    print(f'\n=== Run {i+1}/{n_runs} ===')
    sampled_df = undersample_training_data(train_df, 'TARGET', max_total=max_total, random_state=42 + i)
    feature_cols = [col for col in sampled_df.columns if col not in ['TARGET', 'ID']]
    X = sampled_df[feature_cols]
    y = sampled_df['TARGET']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = TabPFNClassifier(device='cuda')

    clf.fit(X_train.values, y_train.values)

    val_pred = clf.predict(X_val.values)
    val_proba = clf.predict_proba(X_val.values)[:, 1]
    acc = accuracy_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_proba)
    print(f'Validation accuracy: {acc:.4f}')
    print(f'Validation ROC-AUC: {auc:.4f}')

    clf.fit(X.values, y.values)

    # テストデータをバッチに分割して予測し、累積する
    test_features = test_df[feature_cols].values
    for start in range(0, len(test_features), batch_size):
        end = start + batch_size
        batch = test_features[start:end]
        proba_batch = clf.predict_proba(batch)[:, 1]
        test_predictions[start:end] += proba_batch
        # 各バッチ終了後にキャッシュを解放
        torch.cuda.empty_cache()

    # ループ毎にモデルを削除してガーベジコレクションを実行
    del clf
    torch.cuda.empty_cache()
    gc.collect()

# アンサンブル平均
test_predictions /= n_runs

# サブミッションファイルの作成
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'TARGET': test_predictions
})
submission.to_csv('tabpfn_submission.csv', index=False)
print('\nFinished bagging with batched inference. Submission saved to tabpfn_submission.csv')
submission.head()











