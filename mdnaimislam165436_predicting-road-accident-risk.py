import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Baseline Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Tree Models
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Ensemble
from sklearn.ensemble import StackingRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")



print("Train shape:", train.shape)
print("Test shape:", test.shape)


# Preview
train.head()


print(train.info())


print(train.describe())


# Missing values
print("Missing values:\n", train.isnull().sum())


plt.figure(figsize=(6,4))
sns.histplot(train['accident_risk'], bins=30, kde=True)
plt.title("Distribution of Accident Risk")
plt.show()



# Automatically detect numerical columns (excluding target column if needed)
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# If 'accident_risk' is your target column, remove it from num_cols
if 'accident_risk' in num_cols:
    num_cols.remove('accident_risk')


plt.figure(figsize=(8,6))
sns.heatmap(train[num_cols + ['accident_risk']].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()


train.drop(["id","accident_risk"], axis=1).hist(figsize=(15,10))
plt.suptitle("Train Feature Distributions")
plt.show()

test.drop("id", axis=1).hist(figsize=(15,10))
plt.suptitle("Test Feature Distributions")
plt.show()


X = train.drop(["id", "accident_risk"], axis=1)
y = train["accident_risk"]
X_test = test.drop("id", axis=1)


for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))


poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
X_test_scaled = scaler.transform(X_test_poly)


X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


lgb_model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


from lightgbm import early_stopping, log_evaluation

lgb_model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train with validation set using callbacks
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[
        early_stopping(stopping_rounds=100),  
        log_evaluation(100)                   
    ]
)



val_preds = lgb_model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, val_preds))
print(f"✅ Validation RMSE: {rmse:.5f}")


lgb_model.fit(X_scaled, y)
final_preds = lgb_model.predict(X_test_scaled)


final_preds = np.clip(final_preds, 0, 1)


submission["accident_risk"] = final_preds
submission.to_csv("submission.csv", index=False)
print("✅ Final submission file saved!")

