import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, RandomizedSearchCV


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


from sklearn.metrics import mean_squared_error


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


print(train.columns)


X = train.drop(columns = ["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]


X_train_full, X_valid, y_train_full, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 19)


X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size = 0.3, random_state = 19)


xgb_train = XGBRegressor(random_state = 19)


xgb_train.fit(X_train, y_train)


xgb_predictions = xgb_train.predict(X_test)


def rmse(y_obs, y_pred) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between observed and predicted values.

    Args:
        y_obs: Observed values (numpy array or pandas Series).
        y_pred: Predicted values (numpy array or pandas Series).

    Returns:
        float: The root mean squared error.

    Raises:
        ValueError: If inputs have incompatible shapes.
    """
    y_obs = np.asarray(y_obs)  # Convert to numpy array
    y_pred = np.asarray(y_pred)
    
    if y_obs.shape != y_pred.shape:
        raise ValueError(f"Input shapes must match, got {y_obs.shape} and {y_pred.shape}")
    
    return float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))


rmse(y_test, xgb_predictions)


param_distributions = dict(
    n_estimators = [10, 50, 100],
    max_depth = [5, 10],
    learning_rate = [0.01, 0.1]
)


xgb_rand_search = RandomizedSearchCV(XGBRegressor(random_state = 19),
                                     param_distributions, random_state = 19)


xgb_rand_search.fit(X_train, y_train)


print(xgb_rand_search.best_params_)


hyper_xgb = xgb_rand_search.best_estimator_


preds = hyper_xgb.predict(X_test)
rmse(y_test, preds)


lgbm = LGBMRegressor(random_state = 19)


lgbm.fit(X_train, y_train)


lgbm_preds = lgbm.predict(X_test)
rmse(y_test, lgbm_preds)


param_distributions = dict(
    n_estimators = [10, 50, 100],
    max_depth = [5, 10],
    learning_rate = [0.01, 0.1]
)


lgbm_rand_search = RandomizedSearchCV(LGBMRegressor(random_state = 19),
                                     param_distributions, random_state = 19)


lgbm_rand_search.fit(X_train, y_train)


print(lgbm_rand_search.best_params_)


hyper_lgbm = lgbm_rand_search.best_estimator_


preds = hyper_lgbm.predict(X_test)
rmse(y_test, preds)


full_train_xgb = hyper_xgb.fit(X_train_full, y_train_full)
xgb_final_preds = full_train_xgb.predict(X_valid)
rmse(y_valid, xgb_final_preds)


full_train_lgbm = hyper_lgbm.fit(X_train_full, y_train_full)
lgbm_final_preds = full_train_lgbm.predict(X_valid)
rmse(y_valid, lgbm_final_preds)


X_sub = test.drop(columns = ["id"])


xgb_sub = full_train_xgb.predict(X_sub)
print(xgb_sub)


lgbm_sub = full_train_lgbm.predict(X_sub)
print(lgbm_sub)


df_sub_xgb = pd.DataFrame({"id": test["id"], "BeatsPerMinute": xgb_sub})
df_sub_xgb.head()


df_sub_lgbm = pd.DataFrame({"id": test["id"], "BeatsPerMinute": lgbm_sub})
df_sub_lgbm.head()


df_sub_xgb.to_csv("submission2.csv", index = False)
df_sub_lgbm.to_csv("submission3.csv", index = False)




