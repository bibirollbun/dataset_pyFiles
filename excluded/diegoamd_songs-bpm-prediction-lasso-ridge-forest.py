import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, RandomizedSearchCV


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge
#from sklearn.ensemble import RandomForestRegressor
from cuml.ensemble import RandomForestRegressor


from sklearn.metrics import mean_squared_error


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


X = train.drop(columns = ["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]


X_train_full, X_valid, y_train_full, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 19)


X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size = 0.3, random_state = 19)


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


lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_test)
rmse(y_test, lasso_preds)


param_distributions = dict(
    alpha = [0.01, 0.1, 0.5, 1],
    max_iter = [100, 500, 1000]
)


lasso_rand_search = RandomizedSearchCV(Lasso(random_state = 19), param_distributions, random_state = 19)


lasso_rand_search.fit(X_train, y_train)


print(lasso_rand_search.best_params_)


hyper_lasso = lasso_rand_search.best_estimator_


preds = hyper_lasso.predict(X_test)
rmse(y_test, preds)


ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
rmse(y_test, ridge_preds)


param_distributions = dict(
    alpha = [0.01, 0.1, 0.5, 1],
    max_iter = [100, 500, 1000]
)


ridge_rand_search = RandomizedSearchCV(Ridge(random_state = 19), param_distributions, random_state = 19)


ridge_rand_search.fit(X_train, y_train)


print(ridge_rand_search.best_params_)


hyper_ridge = ridge_rand_search.best_estimator_


preds = hyper_ridge.predict(X_test)
rmse(y_test, preds)


forest = RandomForestRegressor(random_state = 19, n_streams = 1)
forest.fit(X_train, y_train)
forest_preds = forest.predict(X_test)
rmse(y_test, forest_preds)


param_distributions = dict(
    n_estimators = [50, 100],
    max_samples = [0.5, 1.0],
    max_depth = [5, 15, 25]
)


forest_rand_search = RandomizedSearchCV(RandomForestRegressor(random_state = 19, n_streams = 1), param_distributions, random_state = 19)


forest_rand_search.fit(X_train, y_train)


print(forest_rand_search.best_params_)


hyper_forest = forest_rand_search.best_estimator_


full_train_lasso = hyper_lasso.fit(X_train_full, y_train_full)
lasso_final_preds = full_train_lasso.predict(X_valid)
rmse(y_valid, lasso_final_preds)


full_train_ridge = hyper_ridge.fit(X_train_full, y_train_full)
ridge_final_preds = full_train_ridge.predict(X_valid)
rmse(y_valid, ridge_final_preds)


full_train_forest = hyper_forest.fit(X_train_full, y_train_full)
forest_final_preds = full_train_forest.predict(X_valid)
rmse(y_valid, forest_final_preds)


X_sub = test.drop(columns = ["id"])


lasso_sub = full_train_lasso.predict(X_sub)
ridge_sub = full_train_ridge.predict(X_sub)


forest_sub = full_train_forest.predict(X_sub)


df_sub_lasso = pd.DataFrame({"id": test["id"], "BeatsPerMinute": lasso_sub})
df_sub_lasso.head()


df_sub_ridge = pd.DataFrame({"id": test["id"], "BeatsPerMinute": ridge_sub})
df_sub_ridge.head()


df_sub_forest = pd.DataFrame({"id": test["id"], "BeatsPerMinute": forest_sub})
df_sub_forest.head()


df_sub_lasso.to_csv("submission4.csv", index = False)
df_sub_ridge.to_csv("submission5.csv", index = False)


df_sub_forest.to_csv("submission6.csv", index = False)




