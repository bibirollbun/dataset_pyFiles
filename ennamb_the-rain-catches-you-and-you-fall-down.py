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


# Load in the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


from IPython.display import Image, display

# Adjust path as needed
display(Image("/kaggle/input/rainfall-storm/riddick_storm_quote.png"))


# Fix the day using the known x/6 intermediary entry
fixes = {1210: 116, 1428: 334}

# Apply the fixes safely
for id_value, corrected_day in fixes.items():
    train.loc[train['id'] == id_value, 'day'] = corrected_day

# Verify the fix
corrected_1210 = train.at[train.index[train['id'] == 1210][0], 'day']
corrected_1428 = train.at[train.index[train['id'] == 1428][0], 'day']

print(f"ID 1210 is now assigned to Day {corrected_1210} (Expected: 116)")
print(f"ID 1428 is now assigned to Day {corrected_1428} (Expected: 334)")


# Dictionary mapping each misplaced ID to its correct day for x/5 and x/7 misclassifications
reassignment_map = {
    1132: 38, 1251: 157, 1284: 190, 1290: 196, 1312: 218, 1318: 224, 
    1346: 252, 1352: 258, 1367: 273, 1373: 279, 1380: 286, 1382: 288, 
    1388: 294, 1395: 301, 1400: 306, 1037: 308, 1403: 309, 1404: 310, 
    1406: 312, 1407: 313, 1409: 315, 1414: 320, 1416: 322, 1420: 326, 
    1430: 336, 1438: 344, 1439: 345, 1445: 351, 1452: 358, 1453: 359, 
    1457: 363, 1458: 364, 1459: 365
}

# Apply the reassignments
for misplaced_id, correct_day in reassignment_map.items():
    train.loc[train['id'] == misplaced_id, 'day'] = correct_day

# Verify that all days now have exactly 6 entries
fixed_counts = train.groupby('day').size()
print("Post-Fix Record Counts:", fixed_counts.value_counts())

# Verify that all sequences are correctly assigned
incorrect_sequences = [
    day for day, group in train.groupby('day')
    if set(group['id']) != set([(day - 1) + (365 * i) for i in range(6)])
]

if incorrect_sequences:
    print(f"ERROR: Some days still have incorrect ID sequences: {incorrect_sequences}")
else:
    print("All day ID sequences are correctly aligned.")


import matplotlib.pyplot as plt

# Compute 15-day rolling mean for rainfall trends **(Using groupby aggregation like original)**
train_avg = train.groupby('day', as_index=False)['rainfall'].mean()

# Compute rolling mean **correctly**
train_avg['rolling_rainfall'] = train_avg['rainfall'].rolling(window=15, min_periods=1).mean()

# Identify key trend points
highest_avg_rainfall_day = train_avg.loc[train_avg['rainfall'].idxmax(), 'day']
lowest_avg_rainfall_day = train_avg.loc[train_avg['rainfall'].idxmin(), 'day']
peak_rolling_mean_day = train_avg.loc[train_avg['rolling_rainfall'].idxmax(), 'day']
lowest_rolling_mean_day = train_avg.loc[train_avg['rolling_rainfall'].idxmin(), 'day']

# Print summary
print(f"Highest Avg Rainfall Day: {highest_avg_rainfall_day} ({train_avg['rainfall'].max():.2f} mm)")
print(f"Lowest Avg Rainfall Day: {lowest_avg_rainfall_day} ({train_avg['rainfall'].min():.2f} mm)")
print(f"Rolling Mean Peaks at Day: {peak_rolling_mean_day} ({train_avg['rolling_rainfall'].max():.2f} mm)")
print(f"Rolling Mean Lowest at Day: {lowest_rolling_mean_day} ({train_avg['rolling_rainfall'].min():.2f} mm)")

# Plot the 15-day rolling trend
plt.figure(figsize=(12, 6))
plt.plot(train_avg['day'], train_avg['rolling_rainfall'], label="15-Day Rolling Mean", color="blue")
plt.scatter(train_avg['day'], train_avg['rainfall'], alpha=0.3, color="gray", label="Daily Rainfall")

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
plt.xticks(range(0, 365, 15))  # Show labels at 15-day intervals

plt.legend()
plt.show()


# Fill NaN in winddirection with 220
test["winddirection"] = test["winddirection"].fillna(220)


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Ensure rolling_trend_slope is correctly derived from train_avg
if "rolling_rainfall" not in train_avg.columns:
    raise KeyError("ERROR: 'rolling_rainfall' column not found in train_avg. Ensure it was computed correctly.")

# Compute rolling trend slope using linear regression over rolling windows
def rolling_slope(series):
    x = np.arange(len(series))
    y = series
    if len(y) < 2:
        return np.nan  # Not enough data to compute slope
    slope = np.polyfit(x, y, 1)[0]
    return slope

train_avg["rolling_trend_slope"] = train_avg["rolling_rainfall"].rolling(window=15, min_periods=2).apply(rolling_slope, raw=True)

# Identify days with missing slopes
nan_slope_days = train_avg[train_avg["rolling_trend_slope"].isna()]["day"].tolist()
print(f"Days with missing rolling trend slope in train_avg: {nan_slope_days}")

# Fill missing slopes by backfilling, then forward-filling if necessary
train_avg["rolling_trend_slope"] = train_avg["rolling_trend_slope"].bfill().ffill()

# Standardize the rolling trend slope
scaler = StandardScaler()
train_avg["rolling_trend_slope_scaled"] = scaler.fit_transform(train_avg[["rolling_trend_slope"]])

# Fit KMeans on train_avg data
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
train_avg["dynamic_trend_bucket"] = kmeans.fit_predict(train_avg[["rolling_trend_slope_scaled"]])

# Print cluster distribution
print("Dynamic Trend Bucket Distribution (Train):")
print(train_avg["dynamic_trend_bucket"].value_counts())

# Map train trend buckets to test using the day feature
day_to_bucket_map = train_avg.set_index("day")["dynamic_trend_bucket"].to_dict()
test["dynamic_trend_bucket"] = test["day"].map(day_to_bucket_map)

# Identify test days missing a trend bucket
missing_test_days = test[test["dynamic_trend_bucket"].isna()]["day"].tolist()
print(f"Test days with missing trend bucket: {missing_test_days}")

# Find test days completely missing from train_avg
missing_from_train_avg = [day for day in missing_test_days if day not in train_avg["day"].values]
print(f"Test days completely missing from train_avg: {missing_from_train_avg}")

# Fill missing test trend buckets using nearest available bucket
for day in missing_test_days:
    if day in train_avg["day"].values:
        closest_day = train_avg["day"].iloc[(train_avg["day"] - day).abs().argsort()[0]]
        test.loc[test["day"] == day, "dynamic_trend_bucket"] = day_to_bucket_map[closest_day]
    else:
        print(f"WARNING: No close match found for test day {day}. Manual review needed.")

# Map slopes using the assigned trend bucket
bucket_to_slope_map = train_avg.groupby("dynamic_trend_bucket")["rolling_trend_slope_scaled"].mean().to_dict()
test["rolling_trend_slope_scaled"] = test["dynamic_trend_bucket"].map(bucket_to_slope_map)

# Debugging: Check if any NaNs exist in test
missing_bucket = test["dynamic_trend_bucket"].isna().sum()
missing_slope = test["rolling_trend_slope_scaled"].isna().sum()

if missing_bucket > 0 or missing_slope > 0:
    print(f"ERROR: {missing_bucket} test rows are missing a trend bucket!")
    print(f"ERROR: {missing_slope} test rows are missing a mapped trend slope!")
    print("This should NOT happen. Investigate why test days are not aligning with train_avg.")

# Print final test cluster distribution
print("Dynamic Trend Bucket Distribution (Test):")
print(test["dynamic_trend_bucket"].value_counts())


import matplotlib.pyplot as plt

# Sort data by day for proper visualization
train_avg = train_avg.sort_values(by="day")

# Plot the clusters
plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    train_avg["day"],
    train_avg["rolling_trend_slope_scaled"],
    c=train_avg["dynamic_trend_bucket"],
    cmap="viridis",
    alpha=0.75
)

# Add colorbar and labels
plt.colorbar(scatter, label="Cluster")
plt.xlabel("Day of the Year")
plt.ylabel("Scaled Rolling Trend Slope")
plt.title("KMeans Clustering of Rainfall Trends Over the Year")

# Show the plot
plt.show()


# Create a mapping of day -> dynamic_trend_bucket from train
day_to_bucket_map = train_avg.set_index("day")["dynamic_trend_bucket"].to_dict()

# Map test days to corresponding trend buckets from train
test["dynamic_trend_bucket"] = test["day"].map(day_to_bucket_map)

# Identify test days that did not match any train day
missing_test_days = test.loc[test["dynamic_trend_bucket"].isna(), "day"]

if not missing_test_days.empty:
    print("Warning: The following test days have no matching trend bucket from train:")
    print(missing_test_days.tolist())

# Fix the FutureWarning by assigning the column explicitly
test["dynamic_trend_bucket"] = test["dynamic_trend_bucket"].fillna(-1)

# Print final test cluster distribution
print("Dynamic Trend Bucket Distribution (Test):")
print(test["dynamic_trend_bucket"].value_counts())

# Show all available train days to compare
print("\nDays available in train:", sorted(train_avg["day"].unique().tolist()))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

# Ensure dynamic_trend_bucket is merged into train
train = train.merge(train_avg[["day", "dynamic_trend_bucket", "rolling_trend_slope_scaled"]], on="day", how="left")

# Compute the mean rolling trend slope per bucket from train
bucket_to_slope_map = train.groupby("dynamic_trend_bucket")["rolling_trend_slope_scaled"].mean().to_dict()
# Map test set days to their corresponding trend bucket
test["dynamic_trend_bucket"] = test["day"].map(day_to_bucket_map)

# Assign the trend slope from train to test using the bucket ID
test["rolling_trend_slope_scaled"] = test["dynamic_trend_bucket"].map(bucket_to_slope_map)

# Check if any test rows were assigned NaN
missing_slope = test["rolling_trend_slope_scaled"].isna().sum()
if missing_slope > 0:
    print(f"Warning: {missing_slope} test rows have no matching slope from train. These will be set to 0.")
    test["rolling_trend_slope_scaled"].fillna(0, inplace=True)

# Re-check if both features exist
missing_features = [col for col in ["rolling_trend_slope_scaled", "dynamic_trend_bucket"] if col not in train.columns]
if missing_features:
    raise KeyError(f"Missing features in train: {missing_features}")

# Define updated feature set
features = ["pressure", "rolling_trend_slope_scaled", "dynamic_trend_bucket", 
            "humidity", 
            "cloud", "sunshine", "windspeed"]

# Prepare X and y
X = train[features]
y = (train["rainfall"] > 0).astype(int)  # Convert rainfall to binary classification

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Dynamic Trend Model Accuracy: {accuracy:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"AUC-ROC Score: {auc_roc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

### **Permutation Importance Evaluation**
perm_importance = permutation_importance(model, X_test_scaled, y_test, scoring="roc_auc", n_repeats=10, random_state=42)

# Store results in a DataFrame
importance_df = pd.DataFrame({'Feature': features, 'Importance': perm_importance.importances_mean})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nTop Features by Permutation Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="blue")
plt.xlabel("Permutation Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance via Permutation Importance")
plt.gca().invert_yaxis()  # Invert y-axis for readability
plt.show()


# STEP 1: Build the agent-time panel
df = train.copy()

# Recompute clean structural fields
df['station_id'] = df['id'] % 365
df['year'] = df['id'] // 365
df['time_index'] = df['year'] * 365 + df['day']

# Convert Rainfall into a binary infection flag
df['infected'] = (df['rainfall'] > 0).astype(int)

# Sort for groupwise operations (per-station time series)
df = df.sort_values(by=['station_id', 'time_index']).reset_index(drop=True)

# Keep only the relevant fields for now
df = df[['id', 'station_id', 'day', 'year', 'time_index', 'infected', 'rainfall', 'pressure', 
         'humidity', 'cloud', 'sunshine', 'windspeed', 'dynamic_trend_bucket', 'rolling_trend_slope_scaled']]



# === SEIR configuration parameters ===
SEIR_PARAMS = {
    'exposure_window_days': 1,        # How far back we look for exposure (t-1)
    'removed_cooldown_days': 1,       # How long an agent is considered removed post-infection
    'exposure_threshold': 1,          # Min infected agents in household to trigger exposure
    'household_size': 6,              # Number of stations per day
}



# === STEP 2: SEIR State Labeling ===

# 1. Previous infection
df['prev_infected'] = df.groupby('station_id')['infected'].shift(SEIR_PARAMS['exposure_window_days'])

# 2. Household exposure (excluding self)
df['household_infection_count'] = df.groupby('day')['infected'].transform('sum') - df['infected']
df['exposed'] = (df['household_infection_count'] >= SEIR_PARAMS['exposure_threshold']).astype(int)

# 3. Cooldown model for "Removed"
df['prev_removed'] = df.groupby('station_id')['infected'].shift(SEIR_PARAMS['removed_cooldown_days'])
df['removed'] = df['prev_removed']  # Simplified for now

# 4. Assign SEIR state
def determine_seir_state(row):
    if row['infected'] == 1:
        return 'I'
    elif row['removed'] == 1:
        return 'R'
    elif row['exposed'] == 1:
        return 'E'
    else:
        return 'S'

df['seir_state'] = df.apply(determine_seir_state, axis=1)


# === DIAGNOSTIC: Check where NaNs originate ===

# 1. Check if 'infected' has any NaNs
infected_nan_rows = df[df['infected'].isna()]
print(f"\nRows where 'infected' is NaN: {len(infected_nan_rows)}")
if not infected_nan_rows.empty:
    print(infected_nan_rows[['id', 'station_id', 'day', 'infected']].head())

# 2. Check if groupby sum returns NaNs
grouped_sum = df.groupby('day')['infected'].sum()
grouped_sum_nans = grouped_sum[grouped_sum.isna()]
print(f"\nDays with NaN infection sum: {len(grouped_sum_nans)}")
if not grouped_sum_nans.empty:
    print("Example day(s) with NaN group sums:", grouped_sum_nans.head())

# 3. Check directly for NaNs in household_infection_count
nan_hic_rows = df[df['household_infection_count'].isna()]
print(f"\nRows with NaN in 'household_infection_count': {len(nan_hic_rows)}")
if not nan_hic_rows.empty:
    print(nan_hic_rows[['id', 'station_id', 'day', 'household_infection_count']].head())



# Step 3B: Household Infection Drift (t-1 exposure)
# Aggregate yesterday's household infection counts by 'day'
day_infections_yesterday = df.groupby('day')['infected'].sum().shift(1)

# Map to current day
df['household_infection_yesterday'] = df['day'].map(day_infections_yesterday).fillna(0)

# Binary flag for exposure via previous day household
df['exposed_from_household_t_minus_1'] = (
    df['household_infection_yesterday'] >= SEIR_PARAMS['exposure_threshold']
).astype(int)



# Patch only 'prev_infected' NaNs (expected from t=0 per station)
df['prev_infected'] = df['prev_infected'].fillna(0).astype(int)


# Expanded exposure flag: any exposure from any known mechanism
df['exposed_combined'] = (
    df['prev_infected'] | 
    df['exposed'] | 
    df['exposed_from_household_t_minus_1']
).astype(int)



# === DIAGNOSTIC: Check for NaNs in exposure-related columns ===

exposure_cols = ['prev_infected', 'exposed', 'exposed_from_household_t_minus_1']

for col in exposure_cols:
    nan_count = df[col].isna().sum()
    dtype = df[col].dtype
    print(f"{col}: {nan_count} NaNs — dtype: {dtype}")
    if nan_count > 0:
        print(df[df[col].isna()][['id', 'station_id', 'day', col]].head())



def updated_seir_state(row):
    if row['infected'] == 1:
        return 'I'
    elif row['removed'] == 1:
        return 'R'
    elif row['exposed_combined'] == 1:
        return 'E'
    else:
        return 'S'

df['seir_state'] = df.apply(updated_seir_state, axis=1)


# Label the previous state (t-1) for each station
df['prev_seir_state'] = df.groupby('station_id')['seir_state'].shift(1)


df = df[df['prev_seir_state'].notna()].copy()


# === Build SEIR-style exposure features into the test set ===
test = test.copy()
test['station_id'] = test['id'] % 365
test['year'] = test['id'] // 365
test['time_index'] = test['year'] * 365 + test['day']

# Placeholder 'infected' column to allow temporal feature computation (always 0 in test)
# (It doesn't affect exposure, which is based on other agents)
test['infected'] = 0  # We are predicting this, but it's needed to structure exposure logic

# Step 1: prev_infected (dummy shift)
test['prev_infected'] = test.groupby('station_id')['infected'].shift(1).fillna(0).astype(int)

# Step 2: household_infection_count (set to 0, since test agents are alone on their day)
test['household_infection_count'] = 0
test['exposed'] = 0

# Step 3: simulate yesterday's household infection count (we don't have it — set to 0)
test['household_infection_yesterday'] = 0
test['exposed_from_household_t_minus_1'] = 0

# Step 4: Combine exposure logic
test['exposed_combined'] = (
    test['prev_infected'] | test['exposed'] | test['exposed_from_household_t_minus_1']
).astype(int)

# Step 5: Simulate 'removed' using previous fake infection state
test['prev_removed'] = test.groupby('station_id')['infected'].shift(1).fillna(0).astype(int)
test['removed'] = test['prev_removed']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, log_loss, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === Step 1: Define the SEIR-safe feature set ===
base_features = [
    "pressure", "rolling_trend_slope_scaled", "dynamic_trend_bucket",
    "humidity", "cloud", "sunshine", "windspeed"
]

# These SEIR-driven features are engineered from exposure context—not final state
seir_features = [
    "prev_infected", "exposed", "exposed_from_household_t_minus_1",
    "exposed_combined", "removed"
]

all_features = base_features + seir_features

# === Step 2: Prepare labeled training data ===
X = df[all_features]
y = (df["rainfall"] > 0).astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 3: Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# === Step 4: Train model ===
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# === Step 5: Evaluate ===
y_pred = model.predict(X_valid_scaled)
y_proba = model.predict_proba(X_valid_scaled)[:, 1]

print(f"Validation Accuracy: {accuracy_score(y_valid, y_pred):.4f}")
print(f"AUC-ROC Score: {roc_auc_score(y_valid, y_proba):.4f}")
print(f"Log Loss: {log_loss(y_valid, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# === Step 6: Prepare test set ===
unlabeled_test = test.copy()


# Ensure all SEIR and base features are present
missing_features = [f for f in all_features if f not in unlabeled_test.columns]
if missing_features:
    raise KeyError(f"Missing required features in test set: {missing_features}")

X_test = unlabeled_test[all_features]
X_test_scaled = scaler.transform(X_test)

# === Step 7: Predict and build submission ===
unlabeled_test["RainTomorrow"] = model.predict_proba(X_test_scaled)[:, 1]
submission = unlabeled_test[["id", "RainTomorrow"]]
submission.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created (SEIR-aware, no label leakage).")


import pandas as pd

# Load submission file
submission = pd.read_csv("submission.csv")

# Preview the first 10 rows
print("Top 10 entries in submission.csv:")
print(submission.head(10))

# Check expected columns
expected_cols = ["id", "RainTomorrow"]
if list(submission.columns) != expected_cols:
    raise ValueError(f"Column mismatch! Expected {expected_cols}, found {list(submission.columns)}")

# Check value ranges
if not submission["RainTomorrow"].between(0, 1).all():
    raise ValueError("Some RainTomorrow values are outside the [0, 1] probability range!")

# Summary confirmation
print("\nSubmission structure looks correct. Ready to upload.")


# === Reconstruct SEIR features into train ===
train = train.copy()
train['station_id'] = train['id'] % 365
train['year'] = train['id'] // 365
train['time_index'] = train['year'] * 365 + train['day']

# Create binary infected status from rainfall
train['infected'] = (train['rainfall'] > 0).astype(int)

# 1. Previous infection
train['prev_infected'] = train.groupby('station_id')['infected'].shift(1).fillna(0).astype(int)

# 2. Household exposure (excluding self)
train['household_infection_count'] = train.groupby('day')['infected'].transform('sum') - train['infected']
train['exposed'] = (train['household_infection_count'] >= 1).astype(int)

# 3. Drift from yesterday’s infections
day_infections = train.groupby('day')['infected'].sum().shift(1)
train['household_infection_yesterday'] = train['day'].map(day_infections).fillna(0)
train['exposed_from_household_t_minus_1'] = (train['household_infection_yesterday'] >= 1).astype(int)

# 4. Combined exposure
train['exposed_combined'] = (
    train['prev_infected'] |
    train['exposed'] |
    train['exposed_from_household_t_minus_1']
).astype(int)

# 5. Removed state (cooldown after infection)
train['prev_removed'] = train.groupby('station_id')['infected'].shift(1).fillna(0).astype(int)
train['removed'] = train['prev_removed']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, log_loss, roc_auc_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === Safe check for already-merged trend features ===
if "dynamic_trend_bucket" not in train.columns or "rolling_trend_slope_scaled" not in train.columns:
    print("Merging trend features into train...")
    train = train.merge(train_avg[["day", "dynamic_trend_bucket", "rolling_trend_slope_scaled"]], on="day", how="left")

if "dynamic_trend_bucket" not in test.columns:
    print("Mapping trend features into test...")
    test["dynamic_trend_bucket"] = test["day"].map(day_to_bucket_map)

if "rolling_trend_slope_scaled" not in test.columns:
    bucket_to_slope_map = train.groupby("dynamic_trend_bucket")["rolling_trend_slope_scaled"].mean().to_dict()
    test["rolling_trend_slope_scaled"] = test["dynamic_trend_bucket"].map(bucket_to_slope_map).fillna(0)

# === Define features ===
base_features = [
    "pressure", "rolling_trend_slope_scaled", "dynamic_trend_bucket",
    "humidity", "cloud", "sunshine", "windspeed"
]
seir_features = [
    "exposed_from_household_t_minus_1"
]
all_features = base_features + seir_features

# === Train-test split ===
X = train[all_features]
y = (train["rainfall"] > 0).astype(int)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# === Train model ===
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# === Evaluate ===
y_pred = model.predict(X_valid_scaled)
y_proba = model.predict_proba(X_valid_scaled)[:, 1]

print(f"Validation Accuracy: {accuracy_score(y_valid, y_pred):.4f}")
print(f"AUC-ROC Score: {roc_auc_score(y_valid, y_proba):.4f}")
print(f"Log Loss: {log_loss(y_valid, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

# === Confusion Matrix Plot ===
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# === Permutation Importance ===
perm_importance = permutation_importance(model, X_valid_scaled, y_valid, scoring="roc_auc", n_repeats=10, random_state=42)
importance_df = pd.DataFrame({'Feature': all_features, 'Importance': perm_importance.importances_mean})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nTop Features by Permutation Importance:")
print(importance_df.head(10))

# === Prepare test predictions ===
unlabeled_test = test.copy()
X_test = unlabeled_test[all_features]
X_test_scaled = scaler.transform(X_test)
unlabeled_test["RainTomorrow"] = model.predict_proba(X_test_scaled)[:, 1]

# === Save submission ===
submission = unlabeled_test[["id", "RainTomorrow"]]
submission.to_csv("submission.csv", index=False)
print(" Submission file 'submission.csv' created.")



# === Preview submission ===
print("\nPreviewing submission file:")

# Show top 10 rows
print(submission.head(10))

# Check for missing values
if submission.isnull().values.any():
    print("\nWarning: Submission contains missing values!")
else:
    print("No missing values detected.")

# Check column types and expected structure
expected_cols = ["id", "RainTomorrow"]
if list(submission.columns) != expected_cols:
    print(f"\nColumn mismatch! Expected columns: {expected_cols}")
    print(f"Found columns: {list(submission.columns)}")
else:
    print("Column structure is correct.")

# Check RainTomorrow value range
if not submission["RainTomorrow"].between(0, 1).all():
    print("Some RainTomorrow values are outside the [0, 1] range!")
else:
    print("All RainTomorrow values are valid probabilities (0 to 1).")


