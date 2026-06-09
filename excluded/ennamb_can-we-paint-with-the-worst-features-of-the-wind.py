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


# %load_ext cudf.pandas


# Load in the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


import matplotlib.pyplot as plt

# Compute 15 day rolling mean for rainfall trends
train_sorted = train.groupby('day', as_index=False)['rainfall'].mean()
train_sorted['rolling_rainfall'] = train_sorted['rainfall'].rolling(window=15, min_periods=1).mean()

# Identify key trend points
highest_avg_rainfall_day = train_sorted.loc[train_sorted['rainfall'].idxmax(), 'day']
lowest_avg_rainfall_day = train_sorted.loc[train_sorted['rainfall'].idxmin(), 'day']
peak_rolling_mean_day = train_sorted.loc[train_sorted['rolling_rainfall'].idxmax(), 'day']
lowest_rolling_mean_day = train_sorted.loc[train_sorted['rolling_rainfall'].idxmin(), 'day']

print(f"Highest Avg Rainfall Day: {highest_avg_rainfall_day} ({train_sorted['rainfall'].max():.2f} mm)")
print(f"Lowest Avg Rainfall Day: {lowest_avg_rainfall_day} ({train_sorted['rainfall'].min():.2f} mm)")
print(f"Rolling Mean Peaks at Day: {peak_rolling_mean_day} ({train_sorted['rolling_rainfall'].max():.2f} mm)")
print(f"Rolling Mean Lowest at Day: {lowest_rolling_mean_day} ({train_sorted['rolling_rainfall'].min():.2f} mm)")

# Plot the trend with more granular labels to visualize shifts in wet/dry days
plt.figure(figsize=(12, 6))
plt.plot(train_sorted['day'], train_sorted['rolling_rainfall'], label="15-Day Rolling Mean", color="blue")
plt.scatter(train_sorted['day'], train_sorted['rainfall'], alpha=0.3, color="gray", label="Daily Rainfall")

# Mark peak and low points
plt.axvline(highest_avg_rainfall_day, color='red', linestyle="--", label=f"Peak Rainfall (Day {highest_avg_rainfall_day})")
plt.axvline(lowest_avg_rainfall_day, color='green', linestyle="--", label=f"Lowest Rainfall (Day {lowest_avg_rainfall_day})")
plt.axvline(peak_rolling_mean_day, color='purple', linestyle="--", label=f"Peak Rolling Mean (Day {peak_rolling_mean_day})")
plt.axvline(lowest_rolling_mean_day, color='orange', linestyle="--", label=f"Lowest Rolling Mean (Day {lowest_rolling_mean_day})")

# Set labels and title
plt.xlabel("Day of the Year")
plt.ylabel("Average Rainfall (mm)")
plt.title("Rainfall Trends Over the Year")

# Adjust x-axis ticks for better visibility
plt.xticks(range(0, 365, 15))  # Show labels at 15 day intervals

plt.legend()
plt.show()


from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

# **MANUALLY DEFINE TREND BOUNDARIES**
manual_trend_boundaries = [1, 7, 15, 21, 30, 37, 43, 62, 79, 93, 139, 156, 180, 200, 219, 253, 263, 277, 308, 316, 325, 344, 365]

# **Initialize Trend Columns**
train_sorted["trend_slope"] = np.nan  # Float column
train_sorted["trend_bucket_id"] = 0   # Integer column

# **Infer 'year' from 'id'**
test["year"] = (test["id"] // 365) + 1  # Assumes data is sequentially ordered

# **Define test_sorted using unique (year, day) values**
test_sorted = test[["year", "day"]].copy()

# **Initialize Trend Columns**
test_sorted["trend_slope"] = np.nan
test_sorted["trend_bucket_id"] = 0

# **Fit Piecewise Linear Regression for Manual Trend Buckets & Assign to Test**
for i in range(len(manual_trend_boundaries) - 1):
    start, end = manual_trend_boundaries[i], manual_trend_boundaries[i + 1]

    # Extract data within the segment
    X_segment = train_sorted["day"].loc[(train_sorted["day"] >= start) & (train_sorted["day"] < end)].values.reshape(-1, 1)
    y_segment = train_sorted["rolling_rainfall"].loc[(train_sorted["day"] >= start) & (train_sorted["day"] < end)].values.reshape(-1, 1)

    # Fit linear regression
    if len(X_segment) > 1:
        model = LinearRegression()
        model.fit(X_segment, y_segment)
        slope = model.coef_[0][0]  # Extract slope

        # Assign trend slope and bucket ID to TRAIN data
        train_sorted.loc[(train_sorted["day"] >= start) & (train_sorted["day"] < end), "trend_slope"] = float(slope)
        train_sorted.loc[(train_sorted["day"] >= start) & (train_sorted["day"] < end), "trend_bucket_id"] = int(i + 1)

        # Assign the same bucket ID and trend slope to TEST data
        test_sorted.loc[(test_sorted["day"] >= start) & (test_sorted["day"] < end), "trend_slope"] = float(slope)
        test_sorted.loc[(test_sorted["day"] >= start) & (test_sorted["day"] < end), "trend_bucket_id"] = int(i + 1)

# **Print Summary of Manual Trend Buckets**
print("\n=== Rainfall Trend Buckets ===")
for i in range(len(manual_trend_boundaries) - 1):
    start, end = manual_trend_boundaries[i], manual_trend_boundaries[i + 1]
    slope = train_sorted.loc[(train_sorted["day"] >= start) & (train_sorted["day"] < end), "trend_slope"].mean()
    trend_desc = "INCREASING" if slope > 0.01 else "DECREASING" if slope < -0.01 else "STABLE"
    print(f"Days {start} - {end}: {trend_desc} (Slope: {slope:.4f})")

# **Plot Rainfall Trends with Manual Buckets**
plt.figure(figsize=(12, 6))
plt.plot(train_sorted["day"], train_sorted["rolling_rainfall"], label="15-Day Rolling Mean", color="blue")
plt.scatter(train_sorted["day"], train_sorted["rainfall"], alpha=0.3, color="gray", label="Daily Rainfall")

# Mark Manual Trend Boundaries
for boundary in manual_trend_boundaries:
    plt.axvline(x=boundary, linestyle="dashed", color="red", alpha=0.6)

plt.xlabel("Day of the Year")
plt.ylabel("Average Rainfall")
plt.title("Manual Rainfall Trend Segmentation with 7-Day Past Slope")
plt.legend()
plt.show()

# **Prepare Trend Features for Model**
train = train.merge(train_sorted[["day", "trend_slope", "trend_bucket_id"]], on="day", how="left")
test = test.merge(train_sorted[["day", "trend_slope", "trend_bucket_id"]], on="day", how="left")


print("\n Manual Trend Features Added!")


train.head(20)


test.head(20)


# **Create pressure buckets in Train & Test**
#train["pressure_bucket"] = np.where(train["pressure"] > 1020, ">1020 hPa", "≤1020 hPa")
#test["pressure_bucket"] = np.where(test["pressure"] > 1020, ">1020 hPa", "≤1020 hPa")

# **One-hot encode pressure_bucket for both train and test**
#train = pd.get_dummies(train, columns=["pressure_bucket"], dtype=int)
#test = pd.get_dummies(test, columns=["pressure_bucket"], dtype=int)

# **Ensure test set has the same columns as train (handles missing categories in test)**
#missing_cols = set(train.columns) - set(test.columns)
#for col in missing_cols:
#    test[col] = 0  # Add missing one-hot categories as zero

# **Reorder test columns to match train**
#test = test[train.columns]

# **Compute rainfall averages for each pressure bucket (Train only)**
#low_pressure_avg_rainfall = train.loc[train["pressure"] <= 1020, "rainfall"].mean()
#high_pressure_avg_rainfall = train.loc[train["pressure"] > 1020, "rainfall"].mean()

# **Compute percentage decrease in rainfall**
#rainfall_drop_percentage = (low_pressure_avg_rainfall - high_pressure_avg_rainfall) / low_pressure_avg_rainfall * 100

# **Print trend insights**
#print("\n**Analysis: Increased Pressure Decreases Rainfall Chances**")
#print(f"- **Rainfall decreases by {rainfall_drop_percentage:.2f}%** from low-pressure (≤1020 hPa) to high-pressure (>1020 hPa).")
#print(f"- **Average Rainfall at Lower Pressure (≤1020 hPa):** {low_pressure_avg_rainfall:.2f} mm.")
#print(f"- **Average Rainfall at Higher Pressure (>1020 hPa):** {high_pressure_avg_rainfall:.2f} mm.")

# **Group by pressure to analyze rainfall trends (Train only)**
#pressure_trend_df = train.groupby("pressure", as_index=False)["rainfall"].mean()

# **Sort by pressure before smoothing**
#pressure_trend_df = pressure_trend_df.sort_values(by="pressure")

# **Apply rolling mean for rainfall trend smoothing**
#pressure_trend_df["rolling_rainfall"] = pressure_trend_df["rainfall"].rolling(window=15, min_periods=5, center=True).mean()

# **Plot: Pressure vs. Rainfall Trends**
#plt.figure(figsize=(12, 6))
#plt.plot(pressure_trend_df["pressure"], pressure_trend_df["rolling_rainfall"], label="15-Day Rolling Mean", color="blue", linewidth=2)
#plt.scatter(train["pressure"], train["rainfall"], alpha=0.3, color="gray", label="Daily Rainfall")
#plt.axvline(x=1020, color="black", linestyle="--", alpha=0.7, label="1020 hPa Pressure Split")
#plt.xlabel("Pressure (hPa)")
#plt.ylabel("Average Rainfall (mm)")
#plt.title("Rainfall Trends Across Pressure Levels")
#plt.legend()
#plt.show()


# Calculate percentage of days with pressure > 1020 in train set
#train_high_pressure_pct = (train[train["pressure"] > 1020].shape[0] / train.shape[0]) * 100

# Calculate percentage of days with pressure > 1020 in test set
#test_high_pressure_pct = (test[test["pressure"] > 1020].shape[0] / test.shape[0]) * 100

# Print results
#print(f"Percentage of high-pressure days in TRAIN set: {train_high_pressure_pct:.2f}%")
#print(f"Percentage of high-pressure days in TEST set: {test_high_pressure_pct:.2f}%")


import pandas as pd
import numpy as np

# **Define function to map wind direction to cardinal directions**
def wind_to_cardinal(degrees):
    if 337.5 <= degrees or degrees < 22.5:
        return "N"
    elif 22.5 <= degrees < 67.5:
        return "NE"
    elif 67.5 <= degrees < 112.5:
        return "E"
    elif 112.5 <= degrees < 157.5:
        return "SE"
    elif 157.5 <= degrees < 202.5:
        return "S"
    elif 202.5 <= degrees < 247.5:
        return "SW"
    elif 247.5 <= degrees < 292.5:
        return "W"
    elif 292.5 <= degrees < 337.5:
        return "NW"

# **Apply mapping to train and test datasets**
train["wind_cardinal"] = train["winddirection"].apply(wind_to_cardinal)
test["wind_cardinal"] = test["winddirection"].apply(wind_to_cardinal)

# **One-hot encode wind_cardinal for both train and test**
train = pd.get_dummies(train, columns=["wind_cardinal"], dtype=int)
test = pd.get_dummies(test, columns=["wind_cardinal"], dtype=int)

# **Identify missing one-hot categories in test set**
one_hot_cols = [col for col in train.columns if "wind_cardinal_" in col]

# **Ensure test set has all one-hot encoded wind cardinal columns**
for col in one_hot_cols:
    if col not in test.columns:
        test[col] = 0  # Add missing one-hot categories as zero

# **Verify distribution of cardinal directions**
print("\n=== Wind Cardinal Distribution in Train ===")
print(train.filter(like="wind_cardinal").sum())  # One-hot encoded counts

print("\n=== Wind Cardinal Distribution in Test ===")
print(test.filter(like="wind_cardinal").sum())  # One-hot encoded counts



# Compute average rainfall and average pressure per wind cardinal direction in the train set
#wind_stats_train = train.groupby("wind_cardinal").agg(
 #   avg_rainfall=("rainfall", "mean"),
 #   avg_pressure=("pressure", "mean")
#).reset_index()

# Compute value counts for each wind cardinal direction
#wind_counts = train["wind_cardinal"].value_counts().reset_index()
#wind_counts.columns = ["wind_cardinal", "count"]

# Merge value counts with rainfall and pressure averages
#wind_stats_train = wind_stats_train.merge(wind_counts, on="wind_cardinal")

# Sort by avg_rainfall in descending order
#wind_stats_train = wind_stats_train.sort_values(by="avg_rainfall", ascending=False)

# Display results
#print(wind_stats_train)





import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# **Ensure 'id' is not used in training but kept for submission**
features = [col for col in train.columns if col not in ["rainfall", "id"]]  # Exclude 'rainfall' but keep 'id' for later

# Drop rows with missing values in train set
train = train.dropna().reset_index(drop=True)

# **Define target variable**
X = train[features]  # Feature columns
y = train["rainfall"]  # Target variable

# **Split into train-validation set for model tuning**
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# **Prepare test data**
test_data = test.copy()

# **Ensure 'id' is preserved for submission**
test_ids = test_data["id"]  # Store original IDs

# **Standardize numerical features**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# **Train Logistic Regression Model (No RFE)**
model = LogisticRegression(C=1, max_iter=10000)
model.fit(X_train_scaled, y_train)

# **Predict on Training and Validation sets**
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)

# **Predict Probabilities for Log Loss and AUC-ROC**
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]  # Probability of class 1
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]

# **Compute Accuracy Scores**
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

# **Compute Log Loss Scores**
train_log_loss = log_loss(y_train, y_train_proba)
val_log_loss = log_loss(y_val, y_val_proba)

# **Compute AUC-ROC Scores**
train_auc = roc_auc_score(y_train, y_train_proba)
val_auc = roc_auc_score(y_val, y_val_proba)

# **Display Evaluation Metrics**
print("\n=== Baseline Model Evaluation ===")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Train Log Loss: {train_log_loss:.4f}")
print(f"Validation Log Loss: {val_log_loss:.4f}")
print(f"Train AUC-ROC: {train_auc:.4f}")
print(f"Validation AUC-ROC: {val_auc:.4f}")

# **Check and handle missing values in test set**
test_data.fillna(test_data.median(), inplace=True)  # Fill with median

# **Apply the same scaling to the test set**
test_features = test_data[features]  # Keep all features
test_features_scaled = scaler.transform(test_features)

# **Make predictions using all features**
test_data["rainfall"] = model.predict_proba(test_features_scaled)[:, 1]  # Probability of rainfall

# **Restore 'id' for submission**
test_data["id"] = test_ids

# **Ensure submission format**
submission = test_data[["id", "rainfall"]]

# **Save submission file**
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("\nSubmission file saved at:", submission_path)
print("\nSubmission Preview:\n", submission.head())



# Extract feature importance from the trained logistic regression model
feature_importance = pd.DataFrame({
    "Feature": features,  # Use the full feature list instead of selected_features
    "Importance": np.abs(model.coef_).flatten()
})

# Sort features by importance (highest first)
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\n=== True Feature Importance (Logistic Regression Coefficients) ===")
print(feature_importance)


#from sklearn.model_selection import KFold, GroupKFold
#from sklearn.metrics import roc_auc_score
#from sklearn.svm import SVC, LinearSVC
#from cuml.svm import SVC, LinearSVC


#RMV = ['rainfall','id']
#FEATURES = [c for c in list( train.columns ) if not c in RMV]
#print(f"We have {len(FEATURES)} basic features:")
#print( FEATURES )


#INTERACT = []
#for i,c1 in enumerate(FEATURES):
#    for j,c2 in enumerate(FEATURES[i+1:]):
#        n = f"{c1}_{c2}"
#        train[n] = train[c1] * train[c2]
#        test[n] = test[c1] * test[c2]
#        INTERACT.append(n)
#print(f"There are {len(INTERACT)} interaction features:")
#print( INTERACT )

