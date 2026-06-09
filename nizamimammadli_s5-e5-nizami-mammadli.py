import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV, ElasticNetCV

print("The Libraries imported successfully.")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])
train['Heart_Rate_per_minute'] = train['Heart_Rate'] / train['Duration']
test['Heart_Rate_per_minute'] = test['Heart_Rate'] / test['Duration']
train['HRxDuration'] = train['Heart_Rate'] * train['Duration']
test['HRxDuration'] = test['Heart_Rate'] * test['Duration']
train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
test['BMI'] = test['Weight'] / ((test['Height'] / 100) ** 2)
train['Temp_Deviation'] = train['Body_Temp'] - 37
test['Temp_Deviation'] = test['Body_Temp'] - 37
train['Weight_Height_Ratio'] = train['Weight'] / train['Height']
test['Weight_Height_Ratio'] = test['Weight'] / test['Height']

features = [
    'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex',
    'Heart_Rate_per_minute', 'HRxDuration', 'BMI',
    'Temp_Deviation', 'Weight_Height_Ratio'
]

X = train[features]
X_test = test[features]
y = np.log1p(train['Calories'])


kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ----- XGBoost  -----
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.0115,
    'max_depth': 9,
    'min_child_weight': 6,
    'subsample': 0.72,
    'colsample_bytree': 0.66,
    'reg_alpha': 0.7,
    'reg_lambda': 0.18,
    'n_estimators': 1200,
    'tree_method': 'hist',
    'random_state': None, 
    'verbosity': 0
}

# ----- LightGBM  -----
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.027,
    'max_depth': 8,
    'num_leaves': 192,
    'min_child_samples': 14,
    'subsample': 0.76,
    'colsample_bytree': 0.79,
    'feature_fraction_bynode': 0.72,
    'reg_alpha': 2.0,
    'reg_lambda': 3.2,
    'n_estimators': 1500,
    'random_state': None, 
    'verbose': -1
}

# ----- CatBoost  -----
cat_params = {
    'iterations': 1700,
    'learning_rate': 0.020,
    'depth': 9,
    'rsm': 0.80,
    'l2_leaf_reg': 2.8,
    'bagging_temperature': 0.6,
    'loss_function': 'RMSE',
    'verbose': 0,
    'early_stopping_rounds': 70,
    'random_seed': None 
}


SEEDS = [42]

def run_cv_model(model_func, params, X, y, X_test, kf, seeds):
    oof_preds = []
    test_preds = []
    for seed in seeds:
        # Copy params for each seed (dict mutability fix)
        local_params = params.copy()
        local_params['random_state' if 'random_state' in local_params else 'random_seed'] = seed
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        for tr_idx, val_idx in kf.split(X):
            model = model_func(**local_params)
            if model_func == lgb.LGBMRegressor:
                model.fit(
                    X.iloc[tr_idx], y.iloc[tr_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )
            else:
                model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[val_idx] = model.predict(X.iloc[val_idx])
            test_pred += model.predict(X_test) / kf.get_n_splits()
        oof_preds.append(oof)
        test_preds.append(test_pred)
    return np.mean(oof_preds, axis=0), np.mean(test_preds, axis=0)


val_preds_xgb, test_preds_xgb = run_cv_model(xgb.XGBRegressor, xgb_params, X, y, X_test, kf, SEEDS)

val_preds_lgb, test_preds_lgb = run_cv_model(lgb.LGBMRegressor, lgb_params, X, y, X_test, kf, SEEDS)

def run_cat_cv(params, X, y, X_test, kf, seeds):
    oof_preds = []
    test_preds = []
    for seed in seeds:
        local_params = params.copy()
        local_params['random_seed'] = seed
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        for tr_idx, val_idx in kf.split(X):
            model = CatBoostRegressor(**local_params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
            oof[val_idx] = model.predict(X.iloc[val_idx])
            test_pred += model.predict(X_test) / kf.get_n_splits()
        oof_preds.append(oof)
        test_preds.append(test_pred)
    return np.mean(oof_preds, axis=0), np.mean(test_preds, axis=0)

val_preds_cat, test_preds_cat = run_cat_cv(cat_params, X, y, X_test, kf, SEEDS)


def rmsle_loss(weights):
    w_xgb, w_lgb, w_cat = weights
    blended = w_xgb * val_preds_xgb + w_lgb * val_preds_lgb + w_cat * val_preds_cat
    return np.sqrt(mean_squared_log_error(y, np.clip(blended, 0, None)))

initial_weights = [0.32, 0.28, 0.40]
constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
bounds = [(0, 1), (0, 1), (0, 1)]

result = minimize(rmsle_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = result.x
print(f"✅ Optimize Ensemble Weights: XGB={opt_weights[0]:.4f}, LGB={opt_weights[1]:.4f}, CAT={opt_weights[2]:.4f}")

val_preds_opt = (
    opt_weights[0] * val_preds_xgb +
    opt_weights[1] * val_preds_lgb +
    opt_weights[2] * val_preds_cat
)
test_preds_opt = (
    opt_weights[0] * test_preds_xgb +
    opt_weights[1] * test_preds_lgb +
    opt_weights[2] * test_preds_cat
)


def create_meta_df(xgb_preds, lgb_preds, cat_preds):
    meta = pd.DataFrame({
        'xgb': xgb_preds,
        'lgb': lgb_preds,
        'cat': cat_preds
    })
    meta['mean'] = meta[['xgb', 'lgb', 'cat']].mean(axis=1)
    meta['std'] = meta[['xgb', 'lgb', 'cat']].std(axis=1)
    meta['range'] = meta[['xgb', 'lgb', 'cat']].max(axis=1) - meta[['xgb', 'lgb', 'cat']].min(axis=1)
    meta['min'] = meta[['xgb', 'lgb', 'cat']].min(axis=1)
    meta['max'] = meta[['xgb', 'lgb', 'cat']].max(axis=1)
    meta['sum'] = meta[['xgb', 'lgb', 'cat']].sum(axis=1)
    meta['prod'] = meta[['xgb', 'lgb', 'cat']].prod(axis=1)
    meta['xgb_cat_diff'] = meta['xgb'] - meta['cat']
    meta['lgb_cat_diff'] = meta['lgb'] - meta['cat']
    meta['xgb_lgb_diff'] = meta['xgb'] - meta['lgb']
    meta['xgb_lgb_ratio'] = meta['xgb'] / (meta['lgb'] + 1e-5)
    meta['cat_lgb_ratio'] = meta['cat'] / (meta['lgb'] + 1e-5)
    meta['cat_xgb_ratio'] = meta['cat'] / (meta['xgb'] + 1e-5)
    return meta

stacked_val = create_meta_df(val_preds_xgb, val_preds_lgb, val_preds_cat)
stacked_test = create_meta_df(test_preds_xgb, test_preds_lgb, test_preds_cat)


ridge = RidgeCV(alphas=[0.01, 0.05, 0.1, 0.3, 1.0, 3.0])
ridge.fit(stacked_val, y)
val_preds_ridge = ridge.predict(stacked_val)
test_preds_ridge = ridge.predict(stacked_test)
rmsle_ridge = np.sqrt(mean_squared_log_error(y, np.clip(val_preds_ridge, 0, None)))
print(f"✅ Ridge Stacking CV RMSLE: {rmsle_ridge:.8f}")

enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.8, 0.9, 1.0], alphas=[0.01, 0.05, 0.1, 0.3, 1.0, 3.0])
enet.fit(stacked_val, y)
val_preds_enet = enet.predict(stacked_val)
test_preds_enet = enet.predict(stacked_test)
rmsle_enet = np.sqrt(mean_squared_log_error(y, np.clip(val_preds_enet, 0, None)))
print(f"✅ ElasticNet Stacking CV RMSLE: {rmsle_enet:.8f}")


final_val_preds = 0.40 * val_preds_opt + 0.40 * val_preds_ridge + 0.20 * val_preds_enet
final_test_preds = 0.40 * test_preds_opt + 0.40 * test_preds_ridge + 0.20 * test_preds_enet

rmsle_hybrid = np.sqrt(mean_squared_log_error(y, np.clip(final_val_preds, 0, None)))
print(f"✅ Hybrid Ensemble CV RMSLE: {rmsle_hybrid:.8f}")


submission['Calories'] = np.expm1(np.clip(final_test_preds, 0, 400))
submission.to_csv('submission.csv', index=False)
print("✅ Submission file created.")

