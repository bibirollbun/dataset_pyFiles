import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def winkler_score(y_true, lower, upper, alpha=0.1):
    width = upper - lower
    penalty = (2 / alpha) * ((lower - y_true) * (y_true < lower) + (y_true - upper) * (y_true > upper))
    return width + penalty


train_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
test_ids = test_df["id"]


train_df["sale_date"] = pd.to_datetime(train_df["sale_date"])
test_df["sale_date"] = pd.to_datetime(test_df["sale_date"])
train_df["sale_week"] = train_df["sale_date"].dt.isocalendar().week
test_df["sale_week"] = test_df["sale_date"].dt.isocalendar().week

# Drop unneeded columns
train_df.drop(["id", "sale_date"], axis=1, inplace=True)
test_df.drop(["id", "sale_date"], axis=1, inplace=True)


y = train_df["sale_price"]
X = train_df.drop("sale_price", axis=1)

# Identify column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns



num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])


encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
test_df[cat_cols] = encoder.transform(test_df[cat_cols])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_df.copy()


def train_lgb_quantile_model(alpha):
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model

# Train models for 5%, 50%, 95% quantiles
model_05 = train_lgb_quantile_model(0.05)
model_50 = train_lgb_quantile_model(0.50)
model_95 = train_lgb_quantile_model(0.95)


y_val_05 = model_05.predict(X_val)
y_val_50 = model_50.predict(X_val)
y_val_95 = model_95.predict(X_val)

winkler = winkler_score(y_val.values, y_val_05, y_val_95, alpha=0.1)
print("✅ Average Winkler Score (Validation):", np.mean(winkler))

# Plot Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_val_50, alpha=0.3, label="Predicted Median")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label="Ideal")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Median")
plt.title("Validation: Actual vs Predicted Median")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


test_pred_05 = model_05.predict(X_test)
test_pred_50 = model_50.predict(X_test)
test_pred_95 = model_95.predict(X_test)


# ----------------------------
# 10. Create Submission
# ----------------------------
submission = pd.DataFrame({
    "id": test_ids,
    #"Median": test_pred_50,
    "pi_lower": test_pred_05,
    "pi_upper": test_pred_95
})
submission.to_csv("submission.csv", index=False)
submission.head(5)

