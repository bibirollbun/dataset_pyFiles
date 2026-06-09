import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


train.head()


train.info()


for col in train.columns:
    if col not in ["id"] and train[col].dtype in ['float64', 'int64']:
            # Plot histogram + KDE
            plt.figure(figsize=(6,4))
            sns.histplot(train[col], bins=50, kde=True, color="skyblue")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()


for col in train.columns:
    if train[col].dtype in ['float64', 'int64']:
        skew_val = train[col].skew()
        if abs(skew_val) > 1:
            print(f"{col}: Skewness = {skew_val:.4f}  <-- Highly Skewed")
        else:
            print(f"{col}: Skewness = {skew_val:.4f}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# X = features (all except id), y = target (you need to set the target column name)
X = train.drop(columns=["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame (optional, keeps column names)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split (80-20 by default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Train-test split (using scaled X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# CatBoost model
cat_model = CatBoostRegressor(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    l2_leaf_reg=3,
    verbose=200
)
cat_model.fit(X_train, y_train)

# LightGBM model
lgbm_model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose = -1
)
lgbm_model.fit(X_train, y_train,eval_metric="rmse")

# Predictions
# cat_pred = cat_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)

# Simple ensemble (average predictions)
# ensemble_pred = (cat_pred + lgbm_pred) / 2

# RMSE
# rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
print(f"Ensemble Test RMSE: {rmse:.4f}")


# Scale test features
test_scaled = scaler.transform(test.drop(columns=["id"]))

# Predict with both models
# cat_test_pred = cat_model.predict(test_scaled)
lgbm_test_pred = lgbm_model.predict(test_scaled)

# Ensemble
# final_pred = (cat_test_pred + lgbm_test_pred) / 2

# Build submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
# submission["BeatsPerMinute"] = final_pred
submission["BeatsPerMinute"] = lgbm_test_pred
submission.to_csv("submission.csv", index=False)
# print("submission.csv generated with ensemble predictions!")


submission.head()




