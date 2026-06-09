import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# 1. データの読み込み
train_path = "/kaggle/input/porto-seguro-safe-driver-prediction/train.csv"
test_path = "/kaggle/input/porto-seguro-safe-driver-prediction/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 2. -1 の値を持つカラムのリストアップと前処理
cols_with_minus1 = [col for col in train_df.columns if train_df[col].min() == -1]
print("【-1の値を含むカラム】")
print(cols_with_minus1)

# カテゴリとして扱うカラム：カラム名が '_cat' や '_bin' で終わるもの、またはユニーク数が少ないもの（例：ps_car_11）
cat_cols = [col for col in cols_with_minus1 if col.endswith('_cat') or col.endswith('_bin') or col == 'ps_car_11']

for col in cols_with_minus1:
    if col in cat_cols:
        train_df[col] = train_df[col].replace(-1, np.nan).astype('category')
        test_df[col] = test_df[col].replace(-1, np.nan).astype('category')
    else:
        train_df[col] = train_df[col].replace(-1, np.nan)
        test_df[col] = test_df[col].replace(-1, np.nan)

# 3. 特徴量とターゲットの分離
train_y = train_df['target']
train_x = train_df.drop(['id', 'target'], axis=1)
test_ids = test_df['id']
test_x = test_df.drop(['id'], axis=1)

# 4. Normalized Gini Coefficient の定義
def gini(actual, pred):
    assert len(actual) == len(pred)
    all_data = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float64)
    sort_order = np.lexsort((all_data[:,2], -1 * all_data[:,1]))
    all_data = all_data[sort_order]
    total_losses = all_data[:,0].sum()
    gini_sum = all_data[:,0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.0
    return gini_sum / len(actual)

def normalized_gini(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

# 5. LightGBM を用いた学習（Stratified 5-fold CV）
params = {
    'objective': 'binary',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'metric': 'auc',  # 内部評価は AUC を使用（CV時は Gini を外部評価）
    'verbose': -1,
    'seed': 42,
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(train_x.shape[0])
test_preds = np.zeros(test_x.shape[0])
gini_scores = []

# LightGBM にカテゴリ情報を渡すため、カテゴリ型のカラム名を抽出
categorical_features = [col for col in train_x.columns if str(train_x[col].dtype) == 'category']

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
    print(f"Fold {fold_}")
    trn_data = lgb.Dataset(train_x.iloc[trn_idx],
                           label=train_y.iloc[trn_idx],
                           categorical_feature=categorical_features)
    val_data = lgb.Dataset(train_x.iloc[val_idx],
                           label=train_y.iloc[val_idx],
                           categorical_feature=categorical_features)
    
    # early stopping をコールバック経由で設定
    clf = lgb.train(
        params,
        trn_data,
        num_boost_round=10000,
        valid_sets=[trn_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(200)]
    )
    
    oof_preds[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)
    test_preds += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits
    
    fold_gini = normalized_gini(train_y.iloc[val_idx].values, oof_preds[val_idx])
    gini_scores.append(fold_gini)
    print(f"Fold {fold_} Gini: {fold_gini:.5f}")

print(f"\nOverall OOF Normalized Gini: {normalized_gini(train_y.values, oof_preds):.5f}")
print(f"Mean Gini from CV: {np.mean(gini_scores):.5f}")

# 6. 特徴量重要度の可視化（最後の fold のモデルを使用）
lgb.plot_importance(clf, max_num_features=20, importance_type='gain', figsize=(10, 10))
plt.tight_layout()
plt.show()

# 7. 提出用ファイルの作成
submission = pd.DataFrame({'id': test_ids, 'target': test_preds})
submission.to_csv("submission.csv", index=False)


