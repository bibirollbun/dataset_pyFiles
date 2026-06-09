import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Winkler Score Function
def winkler_score(y_true, lower, upper, alpha=0.1):
    width = upper - lower
    penalty = (2 / alpha) * ((lower - y_true) * (y_true < lower) +
                             (y_true - upper) * (y_true > upper))
    return width + penalty

# Load Data
train_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
test_ids = test_df["id"]

# Extract Week Number
train_df["sale_date"] = pd.to_datetime(train_df["sale_date"])
test_df["sale_date"] = pd.to_datetime(test_df["sale_date"])
train_df["sale_week"] = train_df["sale_date"].dt.isocalendar().week
test_df["sale_week"] = test_df["sale_date"].dt.isocalendar().week

# Drop unused columns
train_df.drop(["id", "sale_date"], axis=1, inplace=True)
test_df.drop(["id", "sale_date"], axis=1, inplace=True)

# Separate Features and Target
y = train_df["sale_price"]
X = train_df.drop("sale_price", axis=1)

# Identify column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Impute Missing Values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

# Encode Categorical Features
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
test_df[cat_cols] = encoder.transform(test_df[cat_cols])


def train_lgb_quantile_model(X_train, y_train, alpha, fold, quantile):
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/model_fold{fold+1}_q{quantile}.pkl")
    return model



# K-Fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
winkler_scores = []
mae_scores = []
rmse_scores = []

X_test = test_df.copy()
ensemble_pred_05 = np.zeros(len(X_test))
ensemble_pred_50 = np.zeros(len(X_test))
ensemble_pred_95 = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nðŸ“‚ Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model_05 = train_lgb_quantile_model(X_train, y_train, 0.05, fold, 5)
    model_50 = train_lgb_quantile_model(X_train, y_train, 0.50, fold, 50)
    model_95 = train_lgb_quantile_model(X_train, y_train, 0.95, fold, 95)

    y_val_05 = model_05.predict(X_val)
    y_val_50 = model_50.predict(X_val)
    y_val_95 = model_95.predict(X_val)

    fold_winkler = np.mean(winkler_score(y_val.values, y_val_05, y_val_95))
    fold_mae = mean_absolute_error(y_val, y_val_50)
    fold_rmse = np.sqrt(mean_squared_error(y_val, y_val_50))

    winkler_scores.append(fold_winkler)
    mae_scores.append(fold_mae)
    rmse_scores.append(fold_rmse)

    print(f"âœ… Fold {fold+1} Winkler: {fold_winkler:.2f} | MAE: {fold_mae:.2f} | RMSE: {fold_rmse:.2f}")

# Plot prediction intervals (actual vs predicted with confidence band)
    plt.figure(figsize=(8, 5))
    sorted_idx = np.argsort(y_val.values)
    plt.plot(y_val.values[sorted_idx], label="Actual", color="black")
    plt.plot(y_val_50[sorted_idx], label="Predicted Median", color="blue")
    plt.fill_between(np.arange(len(y_val)), y_val_05[sorted_idx], y_val_95[sorted_idx],
                     color="orange", alpha=0.3, label="PI (90%)")
    plt.xlabel("Samples (sorted by actuals)")
    plt.ylabel("Sale Price")
    plt.title(f"Prediction Interval Plot (Fold {fold+1})")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"plots/pred_interval_fold{fold+1}.png")
    plt.show()
    # Ensemble test predictions
    ensemble_pred_05 += model_05.predict(X_test)
    ensemble_pred_50 += model_50.predict(X_test)
    ensemble_pred_95 += model_95.predict(X_test)


# Average predictions
ensemble_pred_05 /= kf.get_n_splits()
ensemble_pred_50 /= kf.get_n_splits()
ensemble_pred_95 /= kf.get_n_splits()

# Final Metrics
avg_winkler = np.mean(winkler_scores)
avg_mae = np.mean(mae_scores)
avg_rmse = np.mean(rmse_scores)

print("\nðŸŽ¯ Average Winkler Score:", avg_winkler)
print("ðŸ“‰ Average MAE:", avg_mae)
print("ðŸ“‰ Average RMSE:", avg_rmse)

# Save metrics to log file
with open("evaluation_log.txt", "w") as f:
    for i in range(5):
        f.write(f"Fold {i+1} Winkler: {winkler_scores[i]:.4f}, MAE: {mae_scores[i]:.4f}, RMSE: {rmse_scores[i]:.4f}\n")
    f.write(f"\nAverage Winkler: {avg_winkler:.4f}\nAverage MAE: {avg_mae:.4f}\nAverage RMSE: {avg_rmse:.4f}\n")

# Final Submission
submission = pd.DataFrame({
    "id": test_ids,
    "pi_lower": ensemble_pred_05,
    "pi_upper": ensemble_pred_95
})
submission.to_csv("submission.csv", index=False)
submission

