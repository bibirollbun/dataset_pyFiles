!pip install -qq pytabkit


import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/train.csv')
test = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/test.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)

train.head(3)


TARGET = 'charges'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = ['sex', 'smoker', 'region']
print(f'{len(BASE)} Base Features:{BASE}')


FEATURES = BASE
print(len(FEATURES), 'Features.')


X = train[FEATURES]
y = np.log1p(train[TARGET])
X_test = test[FEATURES]


# from pytabkit import RealMLP_HPO_Regressor

# model = RealMLP_HPO_Regressor(
#         device='cpu',
#         random_state=42,
#         n_cv=1,
#         # n_refit=0,
#         n_epochs=10, 
#         val_metric_name='rmse',
#         verbosity=2
#     )
    
# model.fit(X, y, cat_col_names=CATS)


from sklearn.model_selection import KFold

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


from pytabkit import RealMLP_TD_Regressor
from sklearn.metrics import root_mean_squared_error


params = {'device': 'cpu',
          'n_epochs': 10,
          'random_state': 42,
          'val_metric_name': 'rmse',
          'verbosity': 2,
          'hidden_sizes': [256, 256, 256],
          'max_one_hot_cat_size': 9,
          'embedding_size': 8, 
          'weight_param': 'ntk',
          'weight_init_mode': 'std',
          'bias_init_mode': 'he+5',
          'bias_lr_factor': 0.1,
          'act': 'selu',
          'use_parametric_act': True,
          'act_lr_factor': 0.1,
          'wd': 0.02, 
          'wd_sched': 'flat_cos',
          'bias_wd_factor': 0.0,
          'block_str': 'w-b-a-d',
          'p_drop': 0.0, 
          'p_drop_sched': 'flat_cos',
          'add_front_scale': True,
          'scale_lr_factor': 6.0,
          'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
          'num_emb_type': 'plr', 
          'plr_sigma': 0.1513700357637058, 
          'plr_hidden_1': 16, 
          'plr_hidden_2': 4,
          'plr_lr_factor': 0.1, 
          'clamp_output': True,
          'normalize_output': True,
          'lr': 0.05846217780681372, 
          'lr_sched': 'coslog4', 
          'opt': 'adam', 
          'sq_mom': 0.95,
         }


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'--- Fold {fold+1}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RealMLP_TD_Regressor(**params)
    
    model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test)

    print(f"Fold {fold+1} RMSE: {root_mean_squared_error(y_val, oof_preds[val_idx]):.5f}")

test_preds /= N_SPLITS

print(f"Overall OOF RMSE: {root_mean_squared_error(y, oof_preds):.5f}")


pd.DataFrame({'id': train.id, TARGET: np.expm1(oof_preds)}).to_csv('oof_realmlp_baseline.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: np.expm1(test_preds)}).to_csv('test_realmlp_baseline.csv', index=False)




