import kagglehub
import os

import pandas as pd
import numpy as np


from scipy.stats import chi2_contingency
from scipy.stats import ks_2samp

from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression , Ridge

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier

from sklearn.metrics import mean_squared_error, r2_score


import matplotlib.pyplot as plt
import seaborn as sns



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


# Read file in path
data_path = "/kaggle/input/playground-series-s5e10/train.csv"

# df = pd.read_csv(data_path)
# df = df.reset_index(drop=True)
# df.index = df.index + 1
df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_t = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

print(f"âœ… Dataset loaded successfully!")
print(f"ğŸ“� Shape of dataset: {df.shape}")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df.head(10)



#Split feature and target :
X_train = df.drop(columns=["accident_risk", "id"])
Y_train = df["accident_risk"]

#Prepare the test set(drop id) : 
X_test = df.drop(columns=["id"])



#Basic dataset structure
#Primary Data Exploration and Visualization
df.info()



#Columns and target by data type : 
numeric_col = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
categories_col = ["road_type", "lighting", "weather", "time_of_day"]
boolians_col = ["road_signs_present", "public_road", "holiday", "school_season"]

Target = "accident_risk"



#Discribe target and to next step show distributions
df[Target].describe()



# Target Distribution
sns.histplot(df[Target], bins=30, kde=True)
plt.title("Accident_risk (target) distribution")
plt.show()



# Skewness or outliers (boxplot)
plt.figure(figsize=(8, 3))
sns.boxplot(
    x=df[Target],
    color="#4C72B0",
    width=0.4,
    fliersize=3
)

# Percentiles(0.25 , 0.50 , 0.75)
q25 = df[Target].quantile(0.25)
q50 = df[Target].quantile(0.50)
q75 = df[Target].quantile(0.75)

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

for i, col in enumerate(numeric_col[:len(axes)]):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i], color="#4C72B0")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()


# Reassign variable types: num_lanes and speed_limit as categorical
num_cols = [col for col in numeric_col if col not in ["num_lanes", "speed_limit"]]
categories_col.extend(["num_lanes", "speed_limit"])



# Boxplot and outliers

fig, axes = plt.subplots(2, 1, figsize=(8, 4))

# Curvature 
sns.boxplot(
    x=df["curvature"],
    color="#4C72B0",
    width=0.4,
    fliersize=3,
    ax=axes[0]
)
q25 = df["curvature"].quantile(0.25)
q50 = df["curvature"].quantile(0.50)
q75 = df["curvature"].quantile(0.75)

axes[0].axvline(q25, color="orange", linestyle="--", linewidth=1.5)
axes[0].axvline(q50, color="red", linestyle="--", linewidth=1.5)
axes[0].axvline(q75, color="orange", linestyle="--", linewidth=1.5)
axes[0].set_title("Curvature Distribution")
axes[0].set_xlabel("curvature")
axes[0].grid(axis="x", linestyle=":", alpha=0.5)

# Num Reported Accidents
sns.boxplot(
    x=df["num_reported_accidents"],
    color="#4C72B0",
    width=0.4,
    fliersize=3,
    ax=axes[1]
)
q25 = df["num_reported_accidents"].quantile(0.25)
q50 = df["num_reported_accidents"].quantile(0.50)
q75 = df["num_reported_accidents"].quantile(0.75)

axes[1].axvline(q25, color="orange", linestyle="--", linewidth=1.5)
axes[1].axvline(q50, color="red", linestyle="--", linewidth=1.5)
axes[1].axvline(q75, color="orange", linestyle="--", linewidth=1.5)
axes[1].set_title("Num Reported Accidents Distribution")
axes[1].set_xlabel("num_reported_accidents")
axes[1].grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()



# Correlation between numerical features and the target
corr = df[["curvature", "num_reported_accidents", Target]].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix (Numerical Variables + Target)", fontsize=13, weight="bold")
plt.show()



# Descriptive statistics
df[numeric_col].describe()



# Distribution for each categorical feature

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(categories_col[:len(axes)]):
    # Calculate %
    counts = df[col].value_counts(normalize=True).mul(100).round(2)
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

for i, col in enumerate(categories_col[:len(axes)]):
    order = df.groupby(col)[Target].mean().sort_values(ascending=False).index
    
    # Barplot
    sns.barplot(x=col, y=Target, data=df, order=order,
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
#chi2-score between two features .

def chi2score(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Compute pairwise Chi2-score V for all categorical variables
cat_vars = categories_col  
n = len(cat_vars)
cramers_results = pd.DataFrame(np.zeros((n, n)), 
                               index=cat_vars, columns=cat_vars)

for col1 in cat_vars:
    for col2 in cat_vars:
        if col1 == col2:
            cramers_results.loc[col1, col2] = 1.0
        else:
            cramers_results.loc[col1, col2] = chi2score(df[col1], df[col2])

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cramers_results, annot=True, cmap="coolwarm", fmt=".2f", vmin=0, vmax=1)
plt.title("Chi2-score Association Between Categorical Variables", fontsize=13, weight="bold")
plt.tight_layout()
plt.show()



# Distribution of Boolean features

fig, axes = plt.subplots(2, 2, figsize=(8, 4))
axes = axes.flatten()

for i, col in enumerate(boolians_col):
    counts = df[col].value_counts(normalize=True).mul(100).round(2)
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

for i, col in enumerate(boolians_col):
    order = [False, True]
    sns.barplot(x=col, y=Target, data=df, order=order, palette="Blues_r", ax=axes[i])
    axes[i].set_title(f"Average accident risk by {col}", fontsize=12, weight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Mean accident_risk")
    axes[i].set_ylim(0, 1)
    axes[i].grid(axis="y", linestyle=":", alpha=0.5)

    # Compute mean risk per boolean category
    means = df.groupby(col)[Target].mean().to_dict()

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
df["curvature_speed"] = df["curvature"] * df["speed_limit"]
df["curvature_night"] = df["curvature"] * (df["lighting"] == "night").astype(int)
df["speed_rain"] = df["speed_limit"] * (df["weather"] == "rainy").astype(int)




# Correlation with target variable
corrs = df[["curvature_speed", "curvature_night", "speed_rain", "accident_risk"]].corr()[Target].sort_values(ascending=False)
print(corrs)




X_test["curvature_speed"] = X_test["curvature"] * X_test["speed_limit"]
X_test["curvature_night"] = X_test["curvature"] * (X_test["lighting"] == "night").astype(int)



train_drift = df
test_drift = df_t

train_drift["is_train"] = 1
test_drift["is_train"] = 0

combined = pd.concat([train_drift, test_drift], ignore_index=True)



# Numerical values (Kolmogorovâ€“Smirnov test)
drift_results_num = []

for col in num_cols:
    stat, pval = ks_2samp(df[col], df_t[col])
    drift_results_num.append({
        "feature": col,
        "p_value": pval,
        "drift_detected": pval < 0.05
    })

drift_num_df = pd.DataFrame(drift_results_num)
display(drift_num_df)



# Categorical and boolean features (Chi-squared test)
drift_results_cat = []

for col in categories_col + boolians_col:
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
train = df.copy()
test = df_t.copy()



#(curvature Ã— speed)
train["curvature_speed"] = train["curvature"] * train["speed_limit"]
test["curvature_speed"]  = test["curvature"]  * test["speed_limit"]

#(curvature Ã— lighting)
train["curvature_night"] = train["curvature"] * (train["lighting"].isin(['dim', 'night']).astype(int))
test["curvature_night"]  = test["curvature"]  * (test["lighting"].isin(['dim', 'night']).astype(int))



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



X = train.drop(columns=[Target])
y = train[Target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42   # 42 make the results reproducible
)



# Random Forest

fea_rf= [
"curvature","num_reported_accidents","num_lanes_enc","speed_limit_enc","holiday","public_road","road_signs_present",
"school_season","lighting_daylight","lighting_dim","lighting_night","road_type_highway","road_type_rural","road_type_urban",
"weather_clear","weather_foggy","weather_rainy","time_of_day_afternoon","time_of_day_evening","time_of_day_morning"]

fea_rf_int = fea_rf + ["curvature_speed","curvature_night"]



# XGBoost

#split-features:
fea_xgb = [
"curvature","num_reported_accidents","num_lanes_enc","speed_limit_enc","holiday","public_road","road_signs_present",
"school_season","lighting","road_type","weather","time_of_day"]

fea_xgb_int = fea_xgb + ["curvature_speed","curvature_night"]

results = pd.DataFrame(columns=["Model", "Type", "RMSE"])



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
xgb_final.fit(train[fea_xgb_int], train[Target])



# Predictions for the Test Set
test_preds = xgb_final.predict(test[fea_xgb_int])



# Verify if the order of rows is identical between the original and processed test set
same_order = (test["id"].values == df_t["id"].values).all()

print(same_order)


# Submission File
submission = pd.DataFrame({
    "id": df_t["id"],
    "accident_risk": test_preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)




# Preview the output
submission.head()


