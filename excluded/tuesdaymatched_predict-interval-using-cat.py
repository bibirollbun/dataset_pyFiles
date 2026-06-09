import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn. model_selection import KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgbm
import catboost as cat
from sklearn.impute import KNNImputer
from scipy.optimize import minimize_scalar
# predict sale price which interval do they in


# df_set_imputed.to_csv(f'df_set_full.csv', index =False)
# df_test_imputed.to_csv(f'df_test_full.csv', index = False)


df_set_org = pd.read_csv(r'/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
df_impute = pd.read_csv(r'/kaggle/input/after-handling-missing-value/df_set_full.csv')
df_set = pd.concat([df_impute, df_set_org['sale_price'].reset_index(drop=True)], axis=1)
df_test = pd.read_csv(r'/kaggle/input/after-handling-missing-value/df_test_full.csv')


X = df_set.drop(columns = ['sale_price'])
Y = df_set['sale_price']


# models = [
#     ('RandomForest', RandomForestRegressor()),
#     ('XGB', xgb.XGBRegressor()),
#     ('LGBM', lgbm.LGBMRegressor(verbose=100)),
#     ('Cat', cat.CatBoostRegressor(verbose=100))
# ]

# for name, model in models:
#     cv = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(estimator=model, X=X, y=Y, cv=cv, scoring='r2', n_jobs=-1)
#     print(f"{name}: R² mean = {scores.mean():.4f}")

    


model_cat_lower = cat.CatBoostRegressor(
    loss_function='Quantile:alpha=0.05',
    iterations=5000,
    learning_rate=0.015,
    depth=10,
    l2_leaf_reg=3.0,
    subsample=0.8,
    rsm=0.85,  # feature sampling
    grow_policy='Lossguide',
    min_data_in_leaf=10,
    max_leaves=64,
    random_strength=1.5,
    bootstrap_type='Bernoulli',
    verbose=100,
    random_state=42
)

model_cat_upper = cat.CatBoostRegressor(
    loss_function='Quantile:alpha=0.95',
    iterations=5000,
    learning_rate=0.015,
    depth=10,
    l2_leaf_reg=3.0,
    subsample=0.8,
    rsm=0.85,
    grow_policy='Lossguide',
    min_data_in_leaf=10,
    max_leaves=64,
    random_strength=1.5,
    bootstrap_type='Bernoulli',
    verbose=100,
    random_state=42
)

model_lgbm_lower = lgbm.LGBMRegressor(
    objective='quantile',
    alpha=0.05,
    n_estimators=5000, 
    subsample=0.8,
    colsample_bytree=0.554976,
    learning_rate=0.5,
    max_depth=-1,
    min_child_samples=150,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

model_lgbm_upper = lgbm.LGBMRegressor(
    objective='quantile',
    alpha=0.95,
    n_estimators=5000, 
    subsample=0.8,
    colsample_bytree=0.554976,
    learning_rate=0.5,
    max_depth=-1,
    min_child_samples=150,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)



model_cat_lower.fit(X, Y)
model_cat_upper.fit(X, Y)

y_lower = model_cat_lower.predict(X)
y_upper = model_cat_upper.predict(X)


def winkler_score(y_true, y_lower, y_upper, alpha=0.10):
    width = y_upper - y_lower
    penalty_low  = np.where(y_true < y_lower,
                             (y_lower - y_true) * (2.0 / alpha),
                             0.0)
    penalty_high = np.where(y_true > y_upper,
                             (y_true - y_upper) * (2.0 / alpha),
                             0.0)
    return width + penalty_low + penalty_high

scores = winkler_score(Y.values, y_lower, y_upper, alpha=0.10)
mean_score = np.mean(scores)
print(f"Mean Winkler Score: {mean_score:.4f}")



model_lgbm_lower.fit(X, Y)
model_lgbm_upper.fit(X, Y)

y_lower_lgbm = model_lgbm_lower.predict(X)
y_upper_lgbm = model_lgbm_upper.predict(X)


scores_lgbm = winkler_score(Y.values, y_lower_lgbm, y_upper_lgbm, alpha=0.10)
mean_score_lgbm = np.mean(scores_lgbm)
print(f"Mean Winkler Score: {mean_score_lgbm:.4f}")


pi_lower = model_cat_lower.predict(df_test)
pi_upper = model_cat_upper.predict(df_test)


submission = pd.read_csv(r'/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
submission_df = pd.DataFrame({
    'id': submission['id'],
    'pi_lower': pi_lower,
    'pi_upper': pi_upper
})
submission_df.to_csv("submission.csv", index=False)
submission_df.head(5)



pi_lower_lgbm = model_lgbm_lower.predict(df_test)
pi_upper_lgbm = model_lgbm_upper.predict(df_test)


submission_df_lgbm = pd.DataFrame({
    'id': submission['id'],
    'pi_lower': pi_lower_lgbm,
    'pi_upper': pi_upper_lgbm
})
submission_df_lgbm.to_csv("submission_lgbm.csv", index=False)
submission_df_lgbm.head(5)

