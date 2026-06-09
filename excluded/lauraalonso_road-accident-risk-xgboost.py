## Basic Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

# Statistical Analysis
from scipy.stats import ks_2samp, chi2_contingency

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

# Silence unnecessary warnings for cleaner output
warnings.filterwarnings("ignore")


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Directories
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Load data
train_raw = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_raw = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# Display the first few rows of the training data
train_raw.head()


# Split features and target
X_train = train_raw.drop(columns=["accident_risk", "id"])
y_train = train_raw["accident_risk"]

# Prepare the test set (drop id)
X_test = test_raw.drop(columns=["id"])


# Basic dataset structure
train_raw.info()


# Columns by datatype
num_cols = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
cat_cols = ["road_type", "lighting", "weather", "time_of_day"]
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]

# Target
target = "accident_risk"


# Summary statistics
train_raw[target].describe()


# Target Distribution
sns.histplot(train_raw[target], bins=30, kde=True)
plt.title("Accident_risk (target) distribution")
plt.show()


# Skewness or outliers (boxplot)
plt.figure(figsize=(8, 3))
sns.boxplot(
    x=train_raw[target],
    color="#4C72B0",
    width=0.4,
    fliersize=3
)

# Percentiles
q25 = train_raw[target].quantile(0.25)
q50 = train_raw[target].quantile(0.50)
q75 = train_raw[target].quantile(0.75)

# Add vertical lines
plt.axvline(q25, color="orange", linestyle="--", linewidth=1.5)
plt.axvline(q50, color="red", linestyle="--", linewidth=1.5)
plt.axvline(q75, color="orange", linestyle="--", linewidth=1.5)

# Axis and layout
plt.title("Accident Risk Distribution", fontsize=13, weight="bold")
plt.xlabel("Accident risk", fontsize=11)
plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.10, 0.10)) 
plt.grid(axis="x", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()


# Distribution (histogram)

fig, axes = plt.subplots(2, 2, figsize=(8, 4))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(train_raw[col], bins=30, kde=True, ax=axes[i], color="#4C72B0")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()


# Reassign variable types: num_lanes and speed_limit as categorical
num_cols = [col for col in num_cols if col not in ["num_lanes", "speed_limit"]]
cat_cols.extend(["num_lanes", "speed_limit"])


# Boxplot and outliers

fig, axes = plt.subplots(2, 1, figsize=(8, 4))

# Curvature 
sns.boxplot(
    x=train_raw["curvature"],
    color="#4C72B0",
    width=0.4,
    fliersize=3,
    ax=axes[0]
)
q25 = train_raw["curvature"].quantile(0.25)
q50 = train_raw["curvature"].quantile(0.50)
q75 = train_raw["curvature"].quantile(0.75)

axes[0].axvline(q25, color="orange", linestyle="--", linewidth=1.5)
axes[0].axvline(q50, color="red", linestyle="--", linewidth=1.5)
axes[0].axvline(q75, color="orange", linestyle="--", linewidth=1.5)
axes[0].set_title("Curvature Distribution")
axes[0].set_xlabel("curvature")
axes[0].grid(axis="x", linestyle=":", alpha=0.5)

# Num Reported Accidents
sns.boxplot(
    x=train_raw["num_reported_accidents"],
    color="#4C72B0",
    width=0.4,
    fliersize=3,
    ax=axes[1]
)
q25 = train_raw["num_reported_accidents"].quantile(0.25)
q50 = train_raw["num_reported_accidents"].quantile(0.50)
q75 = train_raw["num_reported_accidents"].quantile(0.75)

axes[1].axvline(q25, color="orange", linestyle="--", linewidth=1.5)
axes[1].axvline(q50, color="red", linestyle="--", linewidth=1.5)
axes[1].axvline(q75, color="orange", linestyle="--", linewidth=1.5)
axes[1].set_title("Num Reported Accidents Distribution")
axes[1].set_xlabel("num_reported_accidents")
axes[1].grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()


# Correlation between numerical features and the target
corr = train_raw[["curvature", "num_reported_accidents", target]].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix (Numerical Variables + Target)", fontsize=13, weight="bold")
plt.show()


# Descriptive statistics
train_raw[num_cols].describe()


# Distribution for each categorical feature

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    # Calculate %
    counts = train_raw[col].value_counts(normalize=True).mul(100).round(2)
    order = counts.index

    # Barplot
    sns.barplot(x=counts.index, y=counts.values, ax=axes[i], palette="Blues_r")
    axes[i].set_title(f"{col} (%)", fontsize=12, weight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("% of total")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")

    # Annotations
    for p, value in zip(axes[i].patches, counts.values):
        axes[i].annotate(f"{value:.1f}%", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=9, color='black', weight='bold')

plt.tight_layout()
plt.show()


# Average accident risk per category
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    order = train_raw.groupby(col)[target].mean().sort_values(ascending=False).index
    
    # Barplot
    sns.barplot(x=col, y=target, data=train_raw, order=order,
                palette="Blues_r", estimator="mean", ax=axes[i])
    
    axes[i].set_title(f"Average accident risk by {col}", fontsize=12, weight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Mean accident_risk")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")
    axes[i].grid(axis="y", linestyle=":", alpha=0.5)

    # Anotations
    for p in axes[i].patches:
        value = p.get_height()
        axes[i].annotate(f"{value:.2f}",
                         (p.get_x() + p.get_width() / 2., value),
                         ha='center', va='bottom', fontsize=9, color='black', weight='bold')

plt.tight_layout()
plt.show()


# Association between categorical

# CramÃ©r's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Compute pairwise CramÃ©râ€™s V for all categorical variables
cat_vars = cat_cols  
n = len(cat_vars)
cramers_results = pd.DataFrame(np.zeros((n, n)), 
                               index=cat_vars, columns=cat_vars)

for col1 in cat_vars:
    for col2 in cat_vars:
        if col1 == col2:
            cramers_results.loc[col1, col2] = 1.0
        else:
            cramers_results.loc[col1, col2] = cramers_v(train_raw[col1], train_raw[col2])

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cramers_results, annot=True, cmap="coolwarm", fmt=".2f", vmin=0, vmax=1)
plt.title("CramÃ©r's V Association Between Categorical Variables", fontsize=13, weight="bold")
plt.tight_layout()
plt.show()


# Distribution of Boolean features

fig, axes = plt.subplots(2, 2, figsize=(8, 4))
axes = axes.flatten()

for i, col in enumerate(bool_cols):
    counts = train_raw[col].value_counts(normalize=True).mul(100).round(2)
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=axes[i], palette="Blues_r")
    axes[i].set_title(f"{col} (%)", fontsize=12, weight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("% of total")
    axes[i].set_ylim(0, 100)

    # Annotations
    for p, value in zip(axes[i].patches, counts.values):
        axes[i].annotate(f"{value:.1f}%", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=9, color='black', weight='bold')

plt.tight_layout()
plt.show()


# Relationship between Boolean features and target variable

fig, axes = plt.subplots(2, 2, figsize=(8, 4))
axes = axes.flatten()

for i, col in enumerate(bool_cols):
    order = [False, True]
    sns.barplot(x=col, y=target, data=train_raw, order=order, palette="Blues_r", ax=axes[i])
    axes[i].set_title(f"Average accident risk by {col}", fontsize=12, weight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Mean accident_risk")
    axes[i].set_ylim(0, 1)
    axes[i].grid(axis="y", linestyle=":", alpha=0.5)

    # Compute mean risk per boolean category
    means = train_raw.groupby(col)[target].mean().to_dict()

    # Anotations
    for j, label in enumerate(order):
        val = means.get(label, np.nan)
        if not np.isnan(val):
            axes[i].annotate(f"{val:.2f}", 
                             (j, val + 0.01), 
                             ha='center', va='bottom', 
                             fontsize=9, weight='bold', color='black')

plt.tight_layout()
plt.show()


# Feature Interaction Creation
train_raw["curvature_speed"] = train_raw["curvature"] * train_raw["speed_limit"]
train_raw["curvature_night"] = train_raw["curvature"] * (train_raw["lighting"] == "night").astype(int)
train_raw["speed_rain"] = train_raw["speed_limit"] * (train_raw["weather"] == "rainy").astype(int)


# Correlation with target variable
corrs = train_raw[["curvature_speed", "curvature_night", "speed_rain", "accident_risk"]].corr()[target].sort_values(ascending=False)
print(corrs)


# Apply the same transformations to the test set
test_raw["curvature_speed"] = test_raw["curvature"] * test_raw["speed_limit"]
test_raw["curvature_night"] = test_raw["curvature"] * (test_raw["lighting"] == "night").astype(int)


# Preprare data
train_drift = train_raw
test_drift = test_raw

train_drift["is_train"] = 1
test_drift["is_train"] = 0

combined = pd.concat([train_drift, test_drift], ignore_index=True)


# Numerical values (Kolmogorovâ€“Smirnov test)

drift_results_num = []

for col in num_cols:
    stat, pval = ks_2samp(train_raw[col], test_raw[col])
    drift_results_num.append({
        "feature": col,
        "p_value": pval,
        "drift_detected": pval < 0.05
    })

drift_num_df = pd.DataFrame(drift_results_num)
display(drift_num_df)


# Categorical and boolean features (Chi-squared test)
drift_results_cat = []

for col in cat_cols + bool_cols:
    contingency = pd.crosstab(combined[col], combined["is_train"])
    stat, pval, _, _ = chi2_contingency(contingency)
    drift_results_cat.append({
        "feature": col,
        "p_value": pval,
        "drift_detected": pval < 0.05
    })

drift_cat_df = pd.DataFrame(drift_results_cat)
display(drift_cat_df)


# Copy datasets
train = train_raw.copy()
test = test_raw.copy()


# Log transformation for "num_reported_accidents"
train["num_reported_accidents_log"] = np.log1p(train["num_reported_accidents"])
test["num_reported_accidents_log"] = np.log1p(test["num_reported_accidents"])


# Standardize numerical features
scaler_log = StandardScaler()
scaler_cs  = StandardScaler()
scaler_cn  = StandardScaler()

train["num_reported_accidents_log_scaled"] = scaler_log.fit_transform(train[["num_reported_accidents_log"]])
test["num_reported_accidents_log_scaled"]  = scaler_log.transform(test[["num_reported_accidents_log"]])

train["curvature_speed_scaled"] = scaler_cs.fit_transform(train[["curvature_speed"]])
test["curvature_speed_scaled"]  = scaler_cs.transform(test[["curvature_speed"]])

train["curvature_night_scaled"] = scaler_cn.fit_transform(train[["curvature_night"]])
test["curvature_night_scaled"]  = scaler_cn.transform(test[["curvature_night"]])


nominal_cols = ["road_type", "lighting", "weather", "time_of_day"]

# One-hot encoding
train_ohe = pd.get_dummies(train[nominal_cols], drop_first=False, prefix=nominal_cols)
test_ohe  = pd.get_dummies(test[nominal_cols], drop_first=False, prefix=nominal_cols)

# Align columns between train and test
train_ohe, test_ohe = train_ohe.align(test_ohe, join="left", axis=1, fill_value=0)

# Concatenate encoded columns back into main datasets
train = pd.concat([train, train_ohe], axis=1)
test  = pd.concat([test, test_ohe], axis=1)

# Convert nominal features to categorical dtype (for XGBoost)
for col in nominal_cols:
    if col in train.columns:
        train[col] = train[col].astype("category")
    if col in test.columns:
        test[col] = test[col].astype("category")  


# Ordinal encoding for num_lanes
lanes_order = {1: 1, 2: 2, 3: 3, 4: 4}  
train["num_lanes_enc"] = train["num_lanes"].map(lanes_order)
test["num_lanes_enc"] = test["num_lanes"].map(lanes_order)

# Ordinal encoding for speed_limit
speed_order = {25: 1, 35: 2, 45: 3, 55: 4, 60: 5, 65: 6, 70: 7}  
train["speed_limit_enc"] = train["speed_limit"].map(speed_order)
test["speed_limit_enc"] = test["speed_limit"].map(speed_order)


# Standardize encoded ordinal variables
scaler = StandardScaler()

# Scale encoded num_lanes
train["num_lanes_enc_scaled"] = scaler.fit_transform(train[["num_lanes_enc"]])
test["num_lanes_enc_scaled"]  = scaler.transform(test[["num_lanes_enc"]])

# Scale encoded speed_limit
train["speed_limit_enc_scaled"] = scaler.fit_transform(train[["speed_limit_enc"]])
test["speed_limit_enc_scaled"]  = scaler.transform(test[["speed_limit_enc"]])


X = train.drop(columns=[target])
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42   # 42 make the results reproducible
)


# Save results
results = pd.DataFrame(columns=["Model", "Interaction", "RMSE"])


# Linear Regression
fea_lin_reg = [
"curvature","num_reported_accidents_log_scaled","num_lanes_enc_scaled","speed_limit_enc_scaled","holiday","public_road",
"road_signs_present","school_season","lighting_daylight","lighting_dim","lighting_night","road_type_highway","road_type_rural","road_type_urban",
"weather_clear","weather_foggy","weather_rainy","time_of_day_afternoon","time_of_day_evening", "time_of_day_morning"]

fea_lin_reg_int = [
    "num_reported_accidents_log_scaled","num_lanes_enc_scaled","speed_limit_enc_scaled", "holiday", "public_road", "road_signs_present", 
    "school_season","road_type_highway", "road_type_rural", "road_type_urban","weather_clear", "weather_foggy", "weather_rainy",
    "time_of_day_afternoon", "time_of_day_evening", "time_of_day_morning","curvature_speed_scaled", "curvature_night_scaled"
]

# Random Forest
fea_rf= [
"curvature","num_reported_accidents","num_lanes_enc","speed_limit_enc","holiday","public_road","road_signs_present",
"school_season","lighting_daylight","lighting_dim","lighting_night","road_type_highway","road_type_rural","road_type_urban",
"weather_clear","weather_foggy","weather_rainy","time_of_day_afternoon","time_of_day_evening","time_of_day_morning"]

fea_rf_int = fea_rf + ["curvature_speed","curvature_night"]

# XGBoost
fea_xgb = [
"curvature","num_reported_accidents","num_lanes_enc","speed_limit_enc","holiday","public_road","road_signs_present",
"school_season","lighting","road_type","weather","time_of_day"]

fea_xgb_int = fea_xgb + ["curvature_speed","curvature_night"]


# Data
X_train_lin = X_train[fea_lin_reg].copy()
X_val_lin   = X_val[fea_lin_reg].copy()

# Training
lin_reg = LinearRegression()
lin_reg.fit(X_train_lin, y_train)

# Validation
y_pred_lin = lin_reg.predict(X_val_lin)
rmse_lin = np.sqrt(mean_squared_error(y_val, y_pred_lin))

# Save results
results.loc[len(results)] = ["Linear Regression", "No interactions", rmse_lin]


# Data
X_train_lin_int = X_train[fea_lin_reg_int].copy()
X_val_lin_int   = X_val[fea_lin_reg_int].copy()

# Training
lin_reg_int = LinearRegression()
lin_reg_int.fit(X_train_lin_int, y_train)

# Validation
y_pred_lin = lin_reg_int.predict(X_val_lin_int)
rmse_lin_int = np.sqrt(mean_squared_error(y_val, y_pred_lin))

# Save results
results.loc[len(results)] = ["Linear Regression", "With interactions", rmse_lin_int]


# Data
X_train_rf = X_train[fea_rf].copy()
X_val_rf   = X_val[fea_rf].copy()

# Train model
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

# Train
rf.fit(X_train_rf, y_train)

# Validation
y_val_pred_rf = rf.predict(X_val_rf)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))

# Save
results.loc[len(results)] = ["Random Forest", "No interactions", rmse_rf]


# Data
X_train_rf_int = X_train[fea_rf_int].copy()
X_val_rf_int   = X_val[fea_rf_int].copy()

# Train model
rf_int = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

# Train
rf_int.fit(X_train_rf_int, y_train)

# Validation
y_val_pred_rf_int = rf_int.predict(X_val_rf_int)
rmse_rf_int = np.sqrt(mean_squared_error(y_val, y_val_pred_rf_int))

# Save
results.loc[len(results)] = ["Random Forest", "With interactions", rmse_rf_int]


# Data 
X_train_xgb = X_train[fea_xgb].copy()
X_val_xgb   = X_val[fea_xgb].copy()

# Model
xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
    eval_metric="rmse",
    enable_categorical=True,
    n_jobs=-1
)

# Train
xgb.fit(
    X_train_xgb,
    y_train,
    eval_set=[(X_val_xgb, y_val)],
    verbose=False
)

# Validation
y_val_pred_xgb = xgb.predict(X_val_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb))

# Save
results.loc[len(results)] = ["XGBoost", "No interactions", rmse_xgb]


# Data 
X_train_xgb_int = X_train[fea_xgb_int].copy()
X_val_xgb_int   = X_val[fea_xgb_int].copy()

# Model
xgb_int = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
    eval_metric="rmse",
    enable_categorical=True,
    n_jobs=-1
)

# Train
xgb_int.fit(
    X_train_xgb_int,
    y_train,
    eval_set=[(X_val_xgb_int, y_val)],
    verbose=False
)

# Validation
y_val_pred_xgb_int = xgb_int.predict(X_val_xgb_int)
rmse_xgb_int = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb_int))

# Save
results.loc[len(results)] = ["XGBoost", "With interactions", rmse_xgb_int]


print(results)


# Final model using the best parameters

# Model
xgb_final = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
    eval_metric="rmse",
    enable_categorical=True,
    n_jobs=-1
)

# Train using all available training data
xgb_final.fit(train[fea_xgb_int], train[target])


# Predictions for the Test Set
test_preds = xgb_final.predict(test[fea_xgb_int])


# Verify if the order of rows is identical between the original and processed test set
same_order = (test["id"].values == test_raw["id"].values).all()

print(same_order)


# Submission File
submission = pd.DataFrame({
    "id": test_raw["id"],
    "accident_risk": test_preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)


# Preview the output
submission.head()

