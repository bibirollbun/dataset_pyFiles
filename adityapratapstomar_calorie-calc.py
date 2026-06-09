import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')


print(train.columns)


print(test.columns)


# Drop 'id' column
test_ids = test['id']
test.drop('id', axis=1, inplace=True)


# Separate target
y = train['Calories']
X = train.drop(['id','Calories'], axis=1)
X_test = test.copy()


# One-hot encode all object (categorical) columns
categorical_cols = X.select_dtypes(include='object').columns

# Apply one-hot encoding
X = pd.get_dummies(X, columns=categorical_cols)


# One-hot encode all object (categorical) columns
categorical_cols = X_test.select_dtypes(include='object').columns

# Apply one-hot encoding
X_test = pd.get_dummies(X_test, columns=categorical_cols)


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# --- Feature Engineering ---
def add_features(df):
    df = df.copy()
    df['duration_hr_ratio'] = df['Duration'] / (df['Heart_Rate'] + 1)
    df['age_weight'] = df['Age'] * df['Weight']
    df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2 + 1)  # avoid div zero
    df['log_duration'] = np.log1p(df['Duration'])
    # You can add more features here
    return df

# --- Models ---
def get_models():
    models = {
        'lgb': lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        ),
        'cat': CatBoostRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            depth=6,
            l2_leaf_reg=3,
            verbose=0,
            random_state=42
        )
    }
    return models

# --- Metric ---
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))




# --- Prepare data ---
X = add_features(X)
X_test = add_features(X_test)
y_log = np.log1p(y)  # log-transform target for RMSLE

kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = get_models()

oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}




# --- Train base models ---
for name, model in models.items():
    print(f'Training {name}...')
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

        model.fit(X_train, y_train)
        val_preds_log = model.predict(X_val)
        val_preds = np.expm1(val_preds_log)  # revert log

        oof_preds[name][val_idx] = val_preds
        test_preds[name] += np.expm1(model.predict(X_test)) / kf.n_splits

    score = rmsle(y, oof_preds[name])
    print(f'{name} RMSLE: {score:.5f}')




# --- Create stacked features ---
stacked_train = pd.DataFrame(oof_preds)
stacked_test = pd.DataFrame(test_preds)

# Add meta-features (mean and std of base model predictions)
stacked_train['mean_pred'] = stacked_train.mean(axis=1)
stacked_train['std_pred'] = stacked_train.std(axis=1)
stacked_test['mean_pred'] = stacked_test.mean(axis=1)
stacked_test['std_pred'] = stacked_test.std(axis=1)

# --- Meta-model: GradientBoostingRegressor with scaling ---
meta_model = make_pipeline(StandardScaler(),
                           GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=42))

meta_model.fit(stacked_train, y)
final_preds = meta_model.predict(stacked_test)
final_preds = np.clip(final_preds, 0, None)  # no negatives

# --- Evaluate stacked model ---
stacked_oof = meta_model.predict(stacked_train)
stacked_oof = np.clip(stacked_oof, 0, None)
stacked_score = rmsle(y, stacked_oof)
print(f'Stacked Model RMSLE: {stacked_score:.5f}')


submission['Calories_Burnt'] = final_preds
submission.to_csv('submission.csv', index=False)

