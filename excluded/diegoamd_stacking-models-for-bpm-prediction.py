import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, KFold


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from cuml.ensemble import RandomForestRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


X = train.drop(columns = ["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]


X_train_full, X_valid, y_train_full, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 19)


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


XGB_model = XGBRegressor(random_state = 19, n_estimators = 10, max_depth = 5, learning_rate = 0.1)


LGBM_model = LGBMRegressor(random_state = 19, n_estimators = 10, max_depth = 10, learning_rate = 0.1)


lasso_model = Lasso(random_state = 19, max_iter = 500, alpha = 0.01)


ridge_model = Ridge(random_state = 19, max_iter = 100, alpha = 1)


RFR_model = RandomForestRegressor(random_state = 19, n_streams = 1, n_estimators = 100, max_samples = 1, max_depth = 5)


XGB_model.fit(X_train_full, y_train_full)
XGB_preds = XGB_model.predict(X_valid)
XGB_rmse = rmse(y_valid, XGB_preds)
print(XGB_rmse)


LGBM_model.fit(X_train_full, y_train_full)
LGBM_preds = LGBM_model.predict(X_valid)
LGBM_rmse = rmse(y_valid, LGBM_preds)
print(LGBM_rmse)


lasso_model.fit(X_train_full, y_train_full)
lasso_preds = lasso_model.predict(X_valid)
lasso_rmse = rmse(y_valid, lasso_preds)
print(lasso_rmse)


ridge_model.fit(X_train_full, y_train_full)
ridge_preds = ridge_model.predict(X_valid)
ridge_rmse = rmse(y_valid, ridge_preds)
print(ridge_rmse)


RFR_model.fit(X_train_full, y_train_full)
RFR_preds = RFR_model.predict(X_valid)
RFR_rmse = rmse(y_valid, RFR_preds)
print(RFR_rmse)


X_test = test.drop(columns = ["id"])


base_models = [XGB_model, LGBM_model, lasso_model, ridge_model, RFR_model]
kf = KFold(n_splits = 5, shuffle = True, random_state = 19)

# Initialize matrices for predictions
X_meta_train = np.zeros((len(X_train_full), 5))
X_meta_val = np.zeros((len(X_valid), 5))
X_meta_test = np.zeros((len(X_test), 5))


X_train_full.reset_index(drop = True, inplace = True)
y_train_full.reset_index(drop = True, inplace = True)
X_valid.reset_index(drop = True, inplace = True)
y_valid.reset_index(drop = True, inplace = True)
X_test.reset_index(drop = True, inplace = True)


# Generate OOF predictions for training meta-model and predictions for validation/test
for i, model in enumerate(base_models):
    for train_idx, val_idx in kf.split(X_train_full):
        # Split training data into train and validation folds
        X_tr, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]
        
        # Train base model on k-1 folds
        model.fit(X_tr, y_tr)
        
        # Predict on held-out fold for OOF predictions
        X_meta_train[val_idx, i] = model.predict(X_val_fold)
    
    # Train base model on full training set for validation and test predictions
    model.fit(X_train_full, y_train_full)
    X_meta_val[:, i] = model.predict(X_valid)   # Validation predictions
    X_meta_test[:, i] = model.predict(X_test)  # Test predictions for submission


# Train meta-model
meta_model = LinearRegression()
meta_model.fit(X_meta_train, y_train_full)


# Validate
val_predictions = meta_model.predict(X_meta_val)
print("Validation RMSE:", rmse(y_valid, val_predictions))


# Generate submission predictions
final_predictions = meta_model.predict(X_meta_test)


# Save submission (ensure test_ids match X_test)
submission = pd.DataFrame({'id': test["id"], 'prediction': final_predictions})
submission.head()


submission.to_csv('submission7.csv', index = False)




