import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


#  Winkler Score Function
def winkler_score(y_true, lower, upper, alpha=0.1):
    width = upper - lower
    penalty = (2 / alpha) * ((lower - y_true) * (y_true < lower) + (y_true - upper) * (y_true > upper))
    return width + penalty


# 1. ğŸ“� Load and Prepare Data

train_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
test_ids = test_df["id"]

# Convert dates and extract week number
train_df["sale_date"] = pd.to_datetime(train_df["sale_date"])
test_df["sale_date"] = pd.to_datetime(test_df["sale_date"])
train_df["sale_week"] = train_df["sale_date"].dt.isocalendar().week
test_df["sale_week"] = test_df["sale_date"].dt.isocalendar().week

# Drop unused columns
train_df.drop(["id", "sale_date"], axis=1, inplace=True)
test_df.drop(["id", "sale_date"], axis=1, inplace=True)

# Separate target and features
y = train_df["sale_price"]
X = train_df.drop("sale_price", axis=1)

# Identify column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Imputation
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

# Encoding
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
test_df[cat_cols] = encoder.transform(test_df[cat_cols])


# 2. âœ‚ï¸� Train/Validation Split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_df.copy()


# 3. ğŸŒ² LightGBM Quantile Models
 
def train_lgb_quantile_model(alpha):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        learning_rate=0.05,
        n_estimators=100,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model_lgb_05 = train_lgb_quantile_model(0.05)
model_lgb_50 = train_lgb_quantile_model(0.50)
model_lgb_95 = train_lgb_quantile_model(0.95)


# 4. ğŸ�± CatBoost Quantile Models

def train_catboost_quantile_model(alpha):
    model = CatBoostRegressor(
        loss_function=f'Quantile:alpha={alpha}',
        iterations=100,
        depth=8,
        learning_rate=0.05,
        random_seed=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model

model_cat_05 = train_catboost_quantile_model(0.05)
model_cat_50 = train_catboost_quantile_model(0.50)
model_cat_95 = train_catboost_quantile_model(0.95)


# 5. ğŸ“ˆ Make Predictions

# LightGBM
lgb_val_05 = model_lgb_05.predict(X_val)
lgb_val_50 = model_lgb_50.predict(X_val)
lgb_val_95 = model_lgb_95.predict(X_val)

# CatBoost
cat_val_05 = model_cat_05.predict(X_val)
cat_val_50 = model_cat_50.predict(X_val)
cat_val_95 = model_cat_95.predict(X_val)


# 6. ğŸ§® Compute Winkler Scores

winkler_lgb = winkler_score(y_val.values, lgb_val_05, lgb_val_95)
winkler_cat = winkler_score(y_val.values, cat_val_05, cat_val_95)

print(f"ğŸŒ² LightGBM Winkler Score: {np.mean(winkler_lgb):.2f}")
print(f"ğŸ�± CatBoost Winkler Score: {np.mean(winkler_cat):.2f}")


# 7. ğŸ“Š Compare Average Winkler Scores

plt.figure(figsize=(7, 5))
plt.bar(["LightGBM", "CatBoost"], [np.mean(winkler_lgb), np.mean(winkler_cat)], color=["skyblue", "orange"])
plt.ylabel("Average Winkler Score (Lower is Better)")
plt.title("Model Comparison: Winkler Score")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# 8. ğŸ“� Visualize Interval Width Distribution

interval_lgb = lgb_val_95 - lgb_val_05
interval_cat = cat_val_95 - cat_val_05

plt.figure(figsize=(10, 5))
plt.hist(interval_lgb, bins=50, alpha=0.6, label="LightGBM", color="skyblue")
plt.hist(interval_cat, bins=50, alpha=0.6, label="CatBoost", color="orange")
plt.xlabel("Prediction Interval Width (95% - 5%)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Interval Widths")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 9. ğŸ§ª Predict on Test Set using Catboost
test_pred_05 = model_cat_05.predict(X_test)
test_pred_50 = model_cat_50.predict(X_test)
test_pred_95 = model_cat_95.predict(X_test)


# 10. ğŸ“� Create Submission

submission = pd.DataFrame({
    "id": test_ids,
    #"Median": test_pred_50,
    "pi_lower": test_pred_05,
    "pi_Upper": test_pred_95
})

submission.to_csv("submission.csv", index=False)
submission.head(5)

