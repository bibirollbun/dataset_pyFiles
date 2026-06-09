# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


# âš™ï¸� Winkler Score Function
def winkler_score(y_true, lower, upper, alpha=0.1):
    width = upper - lower
    penalty = (2 / alpha) * ((lower - y_true) * (y_true < lower) + (y_true - upper) * (y_true > upper))
    return width + penalty


# 1. ğŸ“� Load and Prepare Data
train_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
test_ids = test_df["id"]

# ğŸ—“ï¸� Extract features from sale_date
train_df["sale_date"] = pd.to_datetime(train_df["sale_date"])
test_df["sale_date"] = pd.to_datetime(test_df["sale_date"])
train_df["sale_week"] = train_df["sale_date"].dt.isocalendar().week
test_df["sale_week"] = test_df["sale_date"].dt.isocalendar().week
train_df["sale_day"] = train_df["sale_date"].dt.day
test_df["sale_day"] = test_df["sale_date"].dt.day
train_df["sale_month"] = train_df["sale_date"].dt.month
test_df["sale_month"] = test_df["sale_date"].dt.month
train_df["sale_year"] = train_df["sale_date"].dt.year
test_df["sale_year"] = test_df["sale_date"].dt.year

train_df.drop(["id", "sale_date"], axis=1, inplace=True)
test_df.drop(["id", "sale_date"], axis=1, inplace=True)

# 2. ğŸ”„ Impute + Encode
y = train_df["sale_price"]
X = train_df.drop("sale_price", axis=1)

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


# 3. ğŸ”€ Split Data
X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
X_test = test_df.copy()


# 4. ğŸŒ² Train Quantile Models
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

model_05 = train_lgb_quantile_model(0.05)
model_50 = train_lgb_quantile_model(0.50)
model_95 = train_lgb_quantile_model(0.95)


# 5. ğŸ§ª Predict Calibration Set and Compute Î´
cal_pred_05 = model_05.predict(X_cal)
cal_pred_95 = model_95.predict(X_cal)
nonconformity_scores = np.maximum(y_cal - cal_pred_95, cal_pred_05 - y_cal)
alpha = 0.1
delta = np.quantile(nonconformity_scores, 1 - alpha)
print(f"âœ… Conformal adjustment delta (Î´): {delta:.2f}")


# 6. ğŸ§ª Predict Validation
val_pred_05 = model_05.predict(X_val)
val_pred_50 = model_50.predict(X_val)
val_pred_95 = model_95.predict(X_val)
val_pred_05_adj = val_pred_05 - delta
val_pred_95_adj = val_pred_95 + delta


# 7. ğŸ“Š Evaluation
within_uncalibrated = ((y_val >= val_pred_05) & (y_val <= val_pred_95)).mean()
within_calibrated = ((y_val >= val_pred_05_adj) & (y_val <= val_pred_95_adj)).mean()
width_uncalibrated = np.mean(val_pred_95 - val_pred_05)
width_calibrated = np.mean(val_pred_95_adj - val_pred_05_adj)
winkler_uncalibrated = np.mean(winkler_score(y_val, val_pred_05, val_pred_95, alpha=0.1))
winkler_calibrated = np.mean(winkler_score(y_val, val_pred_05_adj, val_pred_95_adj, alpha=0.1))

print("\nğŸ”� Validation Set Coverage & Width Comparison:")
print(f"â†’ Uncalibrated Interval Coverage: {within_uncalibrated:.3f}")
print(f"â†’ Calibrated Interval Coverage:   {within_calibrated:.3f}")
print(f"â†’ Uncalibrated Avg Width:         {width_uncalibrated:.2f}")
print(f"â†’ Calibrated Avg Width:           {width_calibrated:.2f}")
print(f"â†’ Uncalibrated Winkler Score:     {winkler_uncalibrated:.2f}")
print(f"â†’ Calibrated Winkler Score:       {winkler_calibrated:.2f}")


# 8. ğŸ“‰ Plot Interval Width vs Features
plt.figure(figsize=(10, 5))
plt.scatter(X_val["sale_week"], val_pred_95_adj - val_pred_05_adj, alpha=0.5)
plt.title("Prediction Interval Width vs Sale Week")
plt.xlabel("Sale Week")
plt.ylabel("Interval Width")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(X_val["sale_month"], val_pred_95_adj - val_pred_05_adj, alpha=0.5, color="orange")
plt.title("Prediction Interval Width vs Sale Month")
plt.xlabel("Sale Month")
plt.ylabel("Interval Width")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(X_val["sale_year"], val_pred_95_adj - val_pred_05_adj, alpha=0.5, color="green")
plt.title("Prediction Interval Width vs Sale Year")
plt.xlabel("Sale Year")
plt.ylabel("Interval Width")
plt.grid(True)
plt.tight_layout()
plt.show()





# 9. ğŸ§ª Test Set Prediction
test_pred_05 = model_05.predict(X_test) - delta
test_pred_50 = model_50.predict(X_test)
test_pred_95 = model_95.predict(X_test) + delta

# 10. ğŸ’¾ Save Submission
submission = pd.DataFrame({
    "id": test_ids,
    #"Median": test_pred_50,
    "pi_lower": test_pred_05,
    "pi_upper": test_pred_95
})
submission.to_csv("submission.csv", index=False)
submission.head(5)

