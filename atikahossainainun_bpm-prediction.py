
# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Load Data

train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()




# EDA

print(train.info())
print(train.describe())

# Distribution of target
sns.histplot(train['BeatsPerMinute'], bins=50, kde=True, color="purple")
plt.title("Distribution of Beats Per Minute")
plt.show()




# Data Prep
# Dropping ID column
X = train.drop(columns=['id', 'BeatsPerMinute'])
y = train['BeatsPerMinute']

X_test = test.drop(columns=['id'])

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train set:", X_train.shape, "Validation set:", X_val.shape)




# # 5. Baseline Model

# model = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42,
#     n_jobs=-1
# )

# model.fit(X_train, y_train)

# # Validation performance
# y_val_pred = model.predict(X_val)
# rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
# print("Validation RMSE:", rmse)




# # Predict Test Data

# test_preds = model.predict(X_test)


# submission = pd.DataFrame({
#     "id": test["id"],
#     "BeatsPerMinute": test_preds
# })

# submission.to_csv("submission.csv", index=False)
# submission.head()



import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np



# DMatrix 
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test)

# parameters 
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,    # smaller steps
    "max_depth": 10,          # deeper trees
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,        # L2 regularization
    "reg_alpha": 0.1,         # L1 regularization
    "min_child_weight": 5,    # prevents overfitting
    "seed": 42
}



xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=50
)

#  predictions
y_val_pred = xgb_model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("XGBoost Validation RMSE:", rmse)



# Predict test 
test_preds = xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1))


submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": test_preds
})
submission.to_csv("submission.csv", index=False)
submission.head()


