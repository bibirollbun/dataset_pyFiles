import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import  HistGradientBoostingRegressor, BaggingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from torch.utils.data import TensorDataset, DataLoader



df = pd.read_csv(f'/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv(f'/kaggle/input/playground-series-s5e10/test.csv')


missing = df_test.isna().sum()

missing_value = missing[missing > 0]
print(missing)


df.info()


df['time_of_day'].unique()


sns.histplot(df['accident_risk'], kde=True, bins=25)
plt.title('Accident Risk Histogram')
plt.xlabel('Accident Risk')
plt.show()


# relation between object and bool feature with accident_risk
bools = ['road_signs_present','public_road','holiday','school_season']
objects = ['time_of_day', 'road_type','lighting' ,'weather']
plot = bools + objects
fig, axes = plt.subplots(4, 2, figsize=(30, 20))
axes = axes.flatten()

for ax, i in zip(axes, plot):
    sns.boxplot(data = df,x = i, y = 'accident_risk',ax = ax)
    ax.set_title(f'mean of accident risk for: {i}')
    ax.set_xlabel(i) 
    ax.set_ylim(0, df['accident_risk'].max() * 1.1)
    for i in ax.get_xticklabels():
        i.set_rotation(45)
plt.tight_layout()
plt.show()


for i in bools:
    df[i] = df[i].map({
        False: 0,
        True: 1
    })
    df_test[i] = df_test[i].map({
        False: 0,
        True: 1
    })

for j in objects:
    encode = LabelEncoder()
    df[j] = encode.fit_transform(df[j])
    df_test[j] = encode.transform(df_test[j])



X = df.drop(columns = ['id','accident_risk'])
y = df['accident_risk']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 42)


# models = [
#     ('bag',BaggingRegressor()),
#     ('Hist', HistGradientBoostingRegressor())
# ]

# for i, model in models:
#     K = KFold(n_splits=3)
#     result = cross_val_score(model, X, y, cv = K,scoring='neg_root_mean_squared_error')
#     print(f'Model {i}: {result.mean()}')


# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
#         "gamma": trial.suggest_float("gamma", 0.0, 5.0),
#         "subsample": trial.suggest_float("subsample", 0.8, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0),
#         "random_state": 42,
#         "n_jobs": -1,
#         "verbosity": 0
#     }

#     model = XGBRegressor(**params)
#     model.fit(X_train,y_train)
#     pred = model.predict(X_val)
#     result = np.sqrt(mean_squared_error(y_val, pred))

#     return result

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=80)

# print("Best trial:")
# print(study.best_trial.params)
# print("Best RMSE:", study.best_value)

# # {'iterations': 3523, 'learning_rate': 0.03559768597141569, 'depth': 6, 'l2_leaf_reg': 2.088071410610331, 'rsm': 0.7143787401235094, 'min_data_in_leaf': 41, 'subsample': 0.9507629646653237, 'random_strength': 7.340013655670474}
# # Best RMSE: 0.05621941800666226


X_test = df_test.drop(columns = ['id'])


# Base models
lgb_params = {
    'n_estimators': 2421,
    'learning_rate': 0.16301397822315872,
    'num_leaves': 456,
    'max_depth': 7,
    'min_child_samples': 21,
    'subsample': 0.9664813420606216,
    'colsample_bytree': 0.8944189204208853,
    'reg_alpha': 1.1499105212755956,
    'reg_lambda': 5.113001959995742,
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
    'n_jobs': -1,
}

cat_params = {
    'iterations': 3523,
    'learning_rate': 0.03559768597141569,
    'depth': 6,
    'l2_leaf_reg': 2.088071410610331,
    'rsm': 0.7143787401235094,
    'min_data_in_leaf': 41,
    'subsample': 0.9507629646653237,
    'random_strength': 7.340013655670474,
    'loss_function': 'RMSE',
    'verbose': 0,
}

hist_params = {
    'learning_rate': 0.07346700587251265,
    'max_depth': 9,
    'max_leaf_nodes': 237,
    'min_samples_leaf': 50,
    'l2_regularization': 2.9737684041963472,
    'max_bins': 195,
    'early_stopping': False,
    'random_state': 42,
}

xgb_params = {
    'n_estimators': 2571,
    'learning_rate': 0.11692601177203737,
    'max_depth': 10,
    'min_child_weight': 3.6718398588000465,
    'gamma': 0.008022272579628252,
    'subsample': 0.922931612875929,
    'colsample_bytree': 0.9277344241043697,
    'reg_alpha': 0.3859899843963099,
    'reg_lambda': 8.816425335472923,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}
models = [
    ('hist', HistGradientBoostingRegressor(**hist_params)),
    ('lgb', LGBMRegressor(**lgb_params)),
    ('cat', CatBoostRegressor(**cat_params)),
    ('xgb',XGBRegressor(**xgb_params))
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), len(models)))
test_preds = np.zeros((len(X_test), len(models)))
# OOF loop
for i, (name, model) in enumerate(models):
    print(f"{name}")
    oof_pred = np.zeros(len(X))
    test_fold_preds = np.zeros((len(X_test), kf.n_splits))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        oof_pred[valid_idx] = model.predict(X_val)
        test_fold_preds[:, fold] = model.predict(X_test)

        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_pred[valid_idx]))
        print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

    oof_preds[:, i] = oof_pred
    test_preds[:, i] = test_fold_preds.mean(axis=1)


oof_rmse = np.sqrt(mean_squared_error(y, oof_preds.mean(axis=1)))
print(f"Mean RMSE: {oof_rmse}")

meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_preds, y)

meta_oof_pred = meta_model.predict(oof_preds)
rmse_meta = np.sqrt(mean_squared_error(y, meta_oof_pred))
print(f"Meta model OOF RMSE: {rmse_meta:.5f}")


test_prediction = meta_model.predict(test_preds)
submission = pd.DataFrame({
    'id': df_test['id'],
    'accident_risk': test_prediction
})
submission.to_csv('submission.csv', index=False)
print("Saved Successfully")


submission.head()

