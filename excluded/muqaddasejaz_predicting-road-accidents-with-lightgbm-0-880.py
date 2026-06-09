# 1. Imports and basic settings

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Print python packages versions (helpful for reproducibility)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("lightgbm:", lgb.__version__)


DATA_DIR = Path("/kaggle/input/playground-series-s5e10")  # change if running elsewhere
if not DATA_DIR.exists():
    # fallback to current dir - user may have files locally
    DATA_DIR = Path(".")

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
sample_sub_path = DATA_DIR / "sample_submission.csv"



# read
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)

print("train shape:", train.shape)
print("test shape:", test.shape)
print("sample_submission shape:", sample_submission.shape)



# show head
display(train.head())



display(test.head())


# 3. Basic info & missing values
display(train.info())


display(train.describe(include='all').T)


print("Missing values (train):")
print(train.isna().sum())


print("\nUnique values per column (train):")
print(train.nunique())


# 4a. Numeric histograms
numeric_cols = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
plt.figure(figsize=(14, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(train[col], kde=False, bins=30)
    plt.title(f"Distribution: {col}")
plt.tight_layout()
plt.show()



# 4b. Target distribution and log transform check
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(train["num_reported_accidents"], bins=30, kde=False)
plt.title("Target: num_reported_accidents (raw)")

plt.subplot(1,2,2)
sns.histplot(np.log1p(train["num_reported_accidents"]), bins=30, kde=False)
plt.title("Target: log1p(num_reported_accidents)")
plt.tight_layout()
plt.show()



# 4c. Categorical countplots
cat_cols = ["road_type", "lighting", "weather", "road_signs_present", "public_road", "time_of_day", "holiday", "school_season"]
plt.figure(figsize=(16, 12))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(4, 2, i)
    sns.countplot(y=col, data=train, order=train[col].value_counts().index)
    plt.title(f"Counts by {col}")
plt.tight_layout()
plt.show()



# 4d. Boxplots for numeric features to inspect outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=train[numeric_cols])
plt.title("Boxplots for numeric columns (outliers check)")
plt.show()




train2 = train.copy()
test2  = test.copy()

# Example engineered features:
def add_features(df):
    df["lanes_over_speed"] = df["num_lanes"] / (df["speed_limit"] + 1e-6)
    df["curvature_over_lanes"] = df["curvature"] / (df["num_lanes"] + 1e-6)
    df["is_high_speed"] = (df["speed_limit"] >= 80).astype(int)  # adjust threshold per dataset
    return df

train2 = add_features(train2)
test2  = add_features(test2)

# List of features to use
feature_cols = [
    "road_type","num_lanes","curvature","speed_limit","lighting","weather",
    "road_signs_present","public_road","time_of_day","holiday","school_season",
    "lanes_over_speed","curvature_over_lanes","is_high_speed"
]

# Label encode categorical columns (LightGBM accepts categorical_feature indices if integers)
categorical_cols = ["road_type","lighting","weather","road_signs_present","public_road","time_of_day","holiday","school_season"]
lbl_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # fit on combined to avoid unseen labels in test
    combined = pd.concat([train2[col], test2[col]], axis=0).astype(str)
    le.fit(combined)
    train2[col] = le.transform(train2[col].astype(str))
    test2[col]  = le.transform(test2[col].astype(str))
    lbl_encoders[col] = le

# Confirm features
print("Features used:", feature_cols)
train2[feature_cols].head()



# Target transform
y = train2["num_reported_accidents"].values
y_log = np.log1p(y)  # model on log scale

X = train2[feature_cols].copy()
X_test = test2[feature_cols].copy()



#  LightGBM training loop (compatible with latest LightGBM)
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importance_df = pd.DataFrame()
rmse_scores = []
mae_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log), 1):
    print(f"\nFold {fold}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    lgb_train = lgb.Dataset(
        X_train, y_train,
        categorical_feature=[X.columns.get_loc(c) for c in categorical_cols if c in X.columns]
    )
    lgb_val = lgb.Dataset(
        X_val, y_val, reference=lgb_train,
        categorical_feature=[X.columns.get_loc(c) for c in categorical_cols if c in X.columns]
    )
    
    #  Use callbacks for early stopping and logging
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Predict
    val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred_log
    test_pred_log = model.predict(X_test, num_iteration=model.best_iteration)
    test_preds += test_pred_log / NFOLD
    
    # Metrics on original scale
    val_pred = np.expm1(val_pred_log)
    val_true = np.expm1(y_val)
    rmse = mean_squared_error(val_true, val_pred, squared=False)
    mae = mean_absolute_error(val_true, val_pred)
    print(f"Fold {fold} RMSE: {rmse:.5f}  MAE: {mae:.5f}")
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    
    # Feature importance
    fold_imp = pd.DataFrame()
    fold_imp["feature"] = X.columns
    fold_imp["importance"] = model.feature_importance(importance_type="gain")
    fold_imp["fold"] = fold
    feature_importance_df = pd.concat([feature_importance_df, fold_imp], axis=0)
    
print("\nCV RMSE mean:", np.mean(rmse_scores), "std:", np.std(rmse_scores))
print("CV MAE mean:", np.mean(mae_scores), "std:", np.std(mae_scores))







oof_preds_orig = np.expm1(oof_preds)
rmse_oof = mean_squared_error(train2["num_reported_accidents"], oof_preds_orig, squared=False)
mae_oof = mean_absolute_error(train2["num_reported_accidents"], oof_preds_orig)
print("OOF RMSE:", rmse_oof)
print("OOF MAE :", mae_oof)



# Aggregate feature importance
imp_mean = feature_importance_df.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=imp_mean)
plt.title("Feature importance (mean gain over folds)")
plt.tight_layout()
plt.show()



plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(oof_preds_orig, bins=30, kde=False)
plt.title("OOF predictions (original scale)")

plt.subplot(1,2,2)
sns.scatterplot(x=train2["num_reported_accidents"], y=oof_preds_orig, alpha=0.4)
plt.xlabel("true")
plt.ylabel("predicted")
plt.title("True vs Predicted (OOF)")
plt.tight_layout()
plt.show()



# Revert test predictions
test_preds_orig = np.expm1(test_preds)

# Construct submission DataFrame
sub = sample_submission.copy()
# Inspect sample_submission columns to find the target column name
print("Sample submission columns:", sub.columns.tolist())

# Choose the likely target column name (if sample uses 'num_reported_accidents', else adjust)
target_col = [c for c in sub.columns if c != "id"][0]  # first non-id col
sub[target_col] = test_preds_orig
# Ensure no negative predictions (just in case) and round if required
sub[target_col] = sub[target_col].clip(lower=0)
sub[target_col] = sub[target_col].round(6)

# Save
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print("Saved submission to", submission_path)
sub.head()


