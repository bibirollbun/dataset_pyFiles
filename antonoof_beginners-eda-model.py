import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.info()


train.isna().sum(axis=0)


train1 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')
train2 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
train3 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')


train_res = pd.concat([train, train1, train2, train3])
train_res.shape


train_res.head()


cat_features = []

for x in test.columns:
    if test[f'{x}'].dtype == 'object':
        cat_features.append(x)

cat_features


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

train_res[cat_features] = ordinal_encoder.fit_transform(train_res[cat_features])
test[cat_features] = ordinal_encoder.transform(test[cat_features])


plt.figure(figsize=(12, 10))
corr_matrix = train_res.corr()

sns.heatmap(corr_matrix, 
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5)

plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


train_res.head()


X = train_res.drop(columns=['accident_risk'])
y = train_res['accident_risk']


kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

model_names = ['xgb', 'lgb', 'cb']
oof_dict = {name: np.zeros(len(X)) for name in model_names}
test_dict = {name: np.zeros(len(test)) for name in model_names}


for name in model_names:
    print(f"Training {name}...")
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(test))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if name == 'xgb':
            model = XGBRegressor(
                n_estimators=700,
                max_depth=7,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                random_state=43,
                early_stopping_rounds=50
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
        elif name == 'lgb':
            model = LGBMRegressor(
                n_estimators=600,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=50,
                verbose=-1
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            
        elif name == 'cb':
            model = CatBoostRegressor(
                iterations=700,
                depth=9,
                learning_rate=0.01,
                rsm=0.6,
                subsample=0.8,
                random_state=43,
                early_stopping_rounds=50,
                verbose=False
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        
        oof[val_idx] = model.predict(X_val)
        test_pred += model.predict(test) / kf.n_splits
    
    oof_dict[name] = oof
    test_dict[name] = test_pred


X_aug = X.copy()
X_test_aug = test.copy()

for name in model_names:
    X_aug[f'oof_{name}'] = oof_dict[name]
    X_test_aug[f'oof_{name}'] = test_dict[name]


meta_model = XGBRegressor(
    n_estimators=330,
    learning_rate=0.4,
    max_depth=6,
    random_state=49,
    verbosity=0
)
meta_model.fit(X_aug, y)

final_pred = meta_model.predict(X_test_aug)
final_pred = np.clip(final_pred, 0, 1)


submission['accident_risk'] = final_pred
submission.to_csv('submission.csv', index=False)
submission.head()




