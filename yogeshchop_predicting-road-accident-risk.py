import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df


df.columns


df.info()


df.describe()


df.isnull().sum()


df['accident_risk'].value_counts(normalize=True)


for col in df.select_dtypes('object'):
    df[col] = df[col].astype('category')
    df[col + '_encoded'] = df[col].cat.codes
df


df.info()


sns.histplot(df['accident_risk'], bins=50)
plt.title('Distribution of Accident Risk')
plt.show()


corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation with Accident Risk")
plt.show()


sns.boxplot(x='road_type', y='accident_risk', data=df)
plt.xticks(rotation=45)
plt.show()


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# Combine for consistent transformations
df = pd.concat([train, test], axis=0, ignore_index=True)


# Basic feature interactions
df["curvature_speed_ratio"] = df["curvature"] / (df["speed_limit"] + 1)
df["lanes_speed_product"] = df["num_lanes"] * df["speed_limit"]
df["curvature_lane_interaction"] = df["curvature"] * df["num_lanes"]

# Safety / Risk indicators
df["is_high_speed"] = (df["speed_limit"] >= 60).astype(int)
df["is_high_curvature"] = (df["curvature"] > 0.5).astype(int)
df["is_risky_combo"] = ((df["is_high_speed"] == 1) & (df["is_high_curvature"] == 1)).astype(int)

# Dim or dark lighting increases risk
df["is_low_light"] = df["lighting"].isin(["dim", "dark"]).astype(int)
df["is_public_dim"] = ((df["public_road"]) & (df["is_low_light"])).astype(int)

# Time & season indicators
df["is_night"] = df["time_of_day"].isin(["night", "evening"]).astype(int)
df["is_peak_hour"] = df["time_of_day"].isin(["morning", "afternoon"]).astype(int)
df["is_school_holiday"] = ((df["school_season"]) & (df["holiday"])).astype(int)
df["is_school_peak"] = ((df["school_season"]) & (df["is_peak_hour"])).astype(int)

# Weather features
df["is_rainy"] = df["weather"].str.contains("rain", case=False, na=False).astype(int)
df["is_foggy"] = df["weather"].str.contains("fog", case=False, na=False).astype(int)
df["is_clear"] = df["weather"].str.contains("clear", case=False, na=False).astype(int)

# Bad weather with poor light = danger
df["bad_weather_flag"] = ((df["is_rainy"] | df["is_foggy"]) & (df["is_low_light"] == 1)).astype(int)

# Road type & lanes
df["is_highway"] = df["road_type"].eq("highway").astype(int)
df["is_urban"] = df["road_type"].eq("urban").astype(int)
df["is_rural"] = df["road_type"].eq("rural").astype(int)
df["lane_density_flag"] = (df["num_lanes"] > 3).astype(int)
df["road_lane_ratio"] = df["num_lanes"] / (df["speed_limit"] + 1)

# Accident history
df["log_num_reported_accidents"] = np.log1p(df["num_reported_accidents"])
df["accidents_per_lane"] = df["num_reported_accidents"] / (df["num_lanes"] + 1)
df["accidents_per_speed"] = df["num_reported_accidents"] / (df["speed_limit"] + 1)

# Composite risk index
df["risky_condition_score"] = (
    df["is_high_speed"] +
    df["is_high_curvature"] +
    df["is_low_light"] +
    df["is_rainy"] +
    df["is_foggy"] +
    (~df["road_signs_present"]).astype(int)
)
df["risk_factor_index"] = df["risky_condition_score"] / 6

# Encode categoricals
cat_cols = ["road_type", "lighting", "weather", "time_of_day"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Split back into train/test
train_fe = df.iloc[:len(train)].copy()
test_fe = df.iloc[len(train):].copy()

# Drop id for model_fiing
train_fe = train_fe.drop(columns=["id"], errors="ignore")
test_fe = test_fe.drop(columns=["id"], errors="ignore")

print(train_fe.shape, test_fe.shape)
train_fe.head()



X = train_fe.drop(columns=["accident_risk"])
y = train_fe["accident_risk"]
X


y


# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train model_fi

import lightgbm as lgb

model_fi = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=-1,
    random_state=42,
    n_jobs=-1
)

model_fi.fit(X_train, y_train)


# FEATURE IMPORTANCE PLOT
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_fi.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance (LightGBM)', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


print(importance_df)


importances = []

for seed in [42, 52, 62]:
    model_fi = lgb.LGBMRegressor(random_state=seed)
    model_fi.fit(X_train, y_train)
    importances.append(model_fi.feature_importances_)

mean_importance = np.mean(importances, axis=0)
importance_df = pd.DataFrame({'Feature': X.columns, 'Mean Importance': mean_importance})
print(importance_df.sort_values(by='Mean Importance', ascending=False))



# DROP FEATURES WITH LOW IMPORTANCE (<10)
low_imp_features = importance_df[importance_df['Mean Importance'] < 10]['Feature'].tolist()
print(f"Dropping {len(low_imp_features)} low-importance features:")
print(low_imp_features)

# Drop them from X
X_reduced = X.drop(columns=low_imp_features)


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

model_fi = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    n_jobs=-1
)

model_fi.fit(X_train, y_train)
y_pred = model_fi.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"\n model_fi retrained with reduced features:")
print(f"Remaining features: {len(X_reduced.columns)}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")



if 'test_X' not in locals():
    test_X = test.drop(columns=['accident_risk'], errors='ignore')
    test_X = test_X.select_dtypes(include=[np.number])

# Ensure all columns from training exist in test before prediction
missing_cols = set(X.columns) - set(test_X.columns)
if missing_cols:
    print(f"⚠️ Adding {len(missing_cols)} missing columns to test_X: {list(missing_cols)}")
    for col in missing_cols:
        test_X[col] = 0  

extra_cols = set(test_X.columns) - set(X.columns)
if extra_cols:
    print(f"⚠️ Dropping {len(extra_cols)} extra columns from test_X: {list(extra_cols)}")
    test_X = test_X.drop(columns=list(extra_cols))

# Align column order to match training
test_X = test_X[X.columns]

print("✅ Train/Test column alignment successful!")
print("Train shape:", X.shape)
print("Test shape:", test_X.shape)



train = df[df["accident_risk"].notna()].copy()
test = df[df["accident_risk"].isna()].copy()

X = train.drop(columns=["accident_risk"])
y = train["accident_risk"]
X = X.select_dtypes(include=[np.number])
test_X = test[X.columns]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=50
)

# Predict and save submission
test["accident_risk"] = model.predict(test_X)
submission = test[["id", "accident_risk"]]

# Save to Kaggle working directory
output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print("✅ submission.csv created successfully.")

