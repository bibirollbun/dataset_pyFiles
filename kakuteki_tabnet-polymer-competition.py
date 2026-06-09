pip install /kaggle/input/tabnet/pytorch_tabnet-4.1.0-py3-none-any.whl --no-deps


import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch

# データ読み込み
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# simple length feature
train['smiles_len'] = train['SMILES'].str.len()
test ['smiles_len'] = test ['SMILES'].str.len()
features = ['smiles_len']
target_cols = ['Tg','FFV','Tc','Density','Rg']

# wMAE計算のための重み計算
def calculate_wmae_weights(train_data, test_data, target_cols):
    """wMAE用の重みを計算"""
    weights = {}
    
    for target in target_cols:
        # nt: 特性tに利用可能な値の数（サンプル数）
        nt = train_data[target].notna().sum()
        
        # rt: テストデータに基づく特性tの値域
        # テストデータでは真値が不明なので、訓練データの値域を使用
        train_values = train_data[target].dropna()
        rt = train_values.max() - train_values.min()
        
        # wt = 1 / (sqrt(nt) * rt)
        wt = 1.0 / (np.sqrt(nt) * rt)
        weights[target] = wt
        
        print(f'{target}: nt={nt}, rt={rt:.4f}, wt={wt:.6f}')
    
    return weights

# wMAE重みを計算
wmae_weights = calculate_wmae_weights(train, test, target_cols)

def calculate_wmae(y_true_dict, y_pred_dict, weights):
    """重み付き平均絶対誤差を計算"""
    total_wmae = 0.0
    total_weight = 0.0
    
    for target in y_true_dict.keys():
        if target in weights:
            mask = ~np.isnan(y_true_dict[target])
            if mask.sum() > 0:
                mae = np.mean(np.abs(y_true_dict[target][mask] - y_pred_dict[target][mask]))
                weighted_mae = weights[target] * mae
                total_wmae += weighted_mae
                total_weight += weights[target]
    
    return total_wmae / total_weight if total_weight > 0 else 0.0

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# prepare dataframes
test_preds = pd.DataFrame({'id': test['id']})
oof_preds  = pd.DataFrame(index=train.index, columns=target_cols)

# 各target別の評価結果を保存
target_metrics = {}

for target in target_cols:
    print(f'\n==> Training for target: {target}')
    # 1) only keep rows where this target exists
    mask    = train[target].notnull()
    X_full  = train.loc[mask, features]
    y_full  = train.loc[mask, target].astype(float)

    fold_rmses = []
    fold_maes = []
    test_fold_preds = np.zeros(len(test))

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_full), 1):
        X_tr, y_tr = X_full.iloc[tr_idx], y_full.iloc[tr_idx]
        X_vl, y_vl = X_full.iloc[vl_idx], y_full.iloc[vl_idx]

        # データの標準化（TabNetには推奨）
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_vl_scaled = scaler.transform(X_vl)
        X_test_scaled = scaler.transform(test[features])

        # TabNet Regressorのパラメータ
        model = TabNetRegressor(
            n_d=64,                    # 決定ステップの次元
            n_a=64,                    # 注意機構の次元
            n_steps=5,                 # 決定ステップ数
            gamma=1.5,                 # 特徴量選択の係数
            lambda_sparse=1e-4,        # スパース正則化
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax',        # 'sparsemax' or 'entmax'
            scheduler_params={"step_size":50, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=42,
            verbose=1
        )

        # 学習
        model.fit(
            X_tr_scaled, y_tr.values.reshape(-1, 1),
            eval_set=[(X_vl_scaled, y_vl.values.reshape(-1, 1))],
            eval_name=['valid'],
            eval_metric=['rmse'],
            max_epochs=1000,
            patience=50,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        # predict & score
        vl_pred = model.predict(X_vl_scaled).flatten()
        rmse = np.sqrt(mean_squared_error(y_vl, vl_pred))
        mae = mean_absolute_error(y_vl, vl_pred)
        
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        print(f'  Fold {fold} RMSE: {rmse:.4f}, MAE: {mae:.4f}')

        # store OOF -- map back to original train index
        orig_idx = y_vl.index
        oof_preds.loc[orig_idx, target] = vl_pred

        # accumulate test predictions
        test_fold_preds += model.predict(X_test_scaled).flatten() / kf.n_splits

    test_preds[target] = test_fold_preds
    
    avg_rmse = np.mean(fold_rmses)
    avg_mae = np.mean(fold_maes)
    target_metrics[target] = {'rmse': avg_rmse, 'mae': avg_mae}
    
    print(f'  >>> Avg RMSE for {target}: {avg_rmse:.4f}')
    print(f'  >>> Avg MAE for {target}: {avg_mae:.4f}')

# overall OOF RMSE
valid_targets = []
valid_oof = []
valid_true = []

for target in target_cols:
    mask = train[target].notnull() & oof_preds[target].notnull()
    if mask.sum() > 0:
        valid_targets.append(target)
        valid_oof.append(oof_preds.loc[mask, target].astype(float))
        valid_true.append(train.loc[mask, target].astype(float))

# 全体のRMSE計算
all_oof = np.concatenate([v.values for v in valid_oof])
all_true = np.concatenate([v.values for v in valid_true])
overall_rmse = np.sqrt(mean_squared_error(all_true, all_oof))

print(f'\n=== Overall Evaluation ===')
print(f'Overall OOF RMSE: {overall_rmse:.4f}')

# wMAE計算
y_true_dict = {}
y_pred_dict = {}

for target in target_cols:
    mask = train[target].notnull() & oof_preds[target].notnull()
    if mask.sum() > 0:
        y_true_dict[target] = train.loc[mask, target].values
        y_pred_dict[target] = oof_preds.loc[mask, target].astype(float).values

overall_wmae = calculate_wmae(y_true_dict, y_pred_dict, wmae_weights)
print(f'Overall OOF wMAE: {overall_wmae:.6f}')

# 各targetの詳細メトリクス表示
print(f'\n=== Target-wise Metrics ===')
for target in target_cols:
    if target in target_metrics:
        metrics = target_metrics[target]
        weight = wmae_weights[target]
        weighted_mae = weight * metrics['mae']
        print(f'{target:8s}: RMSE={metrics["rmse"]:.4f}, MAE={metrics["mae"]:.4f}, '
              f'Weight={weight:.6f}, Weighted MAE={weighted_mae:.6f}')

# write submission
submission = test_preds[['id'] + target_cols]
submission.to_csv('submission.csv', index=False)
print(f'\n=== Submission Preview ===')
print(submission.head())

