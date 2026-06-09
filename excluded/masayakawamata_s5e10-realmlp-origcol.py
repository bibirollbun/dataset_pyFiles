!pip install -qq pytabkit


import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
orig = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Orig Shape:', orig.shape)

train.head(3)


TARGET = 'accident_risk'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

print(f'{len(BASE)} Base Features:{BASE}')


ORIG = []

for col in BASE:
    tmp = orig.groupby(col)[TARGET].mean()
    new_col_name = f"orig_{col}"
    tmp.name = new_col_name
    train = train.merge(tmp, on=col, how='left')
    test = test.merge(tmp, on=col, how='left')
    ORIG.append(new_col_name)

print(len(ORIG), 'Orig Features Created!!')


META = []

for df in [train, test, orig]:
    base_risk = (
        0.3 * df["curvature"] + 
        0.2 * (df["lighting"] == "night").astype(int) + 
        0.1 * (df["weather"] != "clear").astype(int) + 
        0.2 * (df["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
    )
    df['Meta'] = base_risk

META.append('Meta')


train['orig_curvature'] = train['orig_curvature'].fillna(orig[TARGET].mean())
test['orig_curvature'] = test['orig_curvature'].fillna(orig[TARGET].mean())


FEATURES = BASE + ORIG + META
print(len(FEATURES), 'Features.')


X = train[FEATURES]
y = train[TARGET]
X_test = test[FEATURES]


from sklearn.model_selection import KFold

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


from pytabkit import RealMLP_HPO_Regressor

model = RealMLP_HPO_Regressor(
        device='cuda',
        random_state=42,
        n_cv=1,
        lr=0.001,
        # n_refit=0,
        n_epochs=30, 
        val_metric_name='rmse',
        verbosity=2
    )
    
model.fit(X, y, cat_col_names=CATS)


# from pytabkit import RealMLP_TD_Regressor

# from sklearn.metrics import root_mean_squared_error
# from sklearn.preprocessing import StandardScaler


# params = {'device': 'cuda',
#           'batch_size': 1024,
#           'n_epochs': 64,
#           'random_state': 42,
#           'lr': 0.001,
#           'verbosity': 2, 
#           'hidden_sizes': [256, 256, 256],
#           'p_drop': 0.15,
#           # 'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
#          }


# oof_preds = np.zeros(len(X))
# test_preds = np.zeros(len(test))

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     print(f'--- Fold {fold+1}/{N_SPLITS} ---')
    
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     model = RealMLP_TD_Regressor(**params)
    
#     model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
    
#     oof_preds[val_idx] = model.predict(X_val)
#     test_preds += model.predict(X_test)

#     print(f"Fold {fold+1} RMSE: {root_mean_squared_error(y_val, oof_preds[val_idx]):.5f}")

# test_preds /= N_SPLITS

# print(f"Overall OOF RMSE: {root_mean_squared_error(y, oof_preds):.5f}")


# pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv('oof_realmlp_plus_origcol.csv', index=False)
# pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv('test_realmlp_plus_origcol.csv', index=False)

