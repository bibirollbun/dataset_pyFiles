import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# データ読み込み
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# simple length feature
train['smiles_len'] = train['SMILES'].str.len()
test ['smiles_len'] = test ['SMILES'].str.len()
features = ['smiles_len']
target_cols = ['Tg','FFV','Tc','Density','Rg']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# prepare dataframes
test_preds = pd.DataFrame({'id': test['id']})
oof_preds  = pd.DataFrame(index=train.index, columns=target_cols)

for target in target_cols:
    print(f'\n==> Training for target: {target}')
    # 1) only keep rows where this target exists
    mask    = train[target].notnull()
    X_full  = train.loc[mask, features]
    y_full  = train.loc[mask, target].astype(float)

    fold_rmses = []
    test_fold_preds = np.zeros(len(test))

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_full), 1):
        X_tr, y_tr = X_full.iloc[tr_idx], y_full.iloc[tr_idx]
        X_vl, y_vl = X_full.iloc[vl_idx], y_full.iloc[vl_idx]

        # Random Forest Regressorのパラメータ
        model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        # 学習
        model.fit(X_tr, y_tr)

        # predict & score
        vl_pred = model.predict(X_vl)
        rmse    = np.sqrt(mean_squared_error(y_vl, vl_pred))
        fold_rmses.append(rmse)
        print(f'  Fold {fold} RMSE: {rmse:.4f}')

        # store OOF -- map back to original train index
        orig_idx = y_vl.index
        oof_preds.loc[orig_idx, target] = vl_pred

        # accumulate test predictions
        test_fold_preds += model.predict(test[features]) / kf.n_splits

    test_preds[target] = test_fold_preds
    print(f'  >>> Avg RMSE for {target}: {np.mean(fold_rmses):.4f}')

# overall OOF
oof_rmse = np.sqrt(
    ((train[target_cols] - oof_preds[target_cols].astype(float))**2).mean().mean()
)
print(f'\nOverall OOF RMSE: {oof_rmse:.4f}')

# write submission
submission = test_preds[['id'] + target_cols]
submission.to_csv('submission.csv', index=False)
print(submission.head())

