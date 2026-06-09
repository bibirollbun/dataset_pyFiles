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


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# === Convert Day to Month Function ===
def day_to_month(day):
    """Converts day of year (1-365) to month (1-12)."""
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cumulative_days = np.cumsum(months)
    for month, days_in_month in enumerate(cumulative_days, 1):
        if day <= days_in_month:
            return month
    return 12

# === Convert 'day' to 'month' in train & test data ===
train['month'] = train['day'].apply(day_to_month)
test['month'] = test['day'].apply(day_to_month)

# === Compute Monthly Average Rainfall for TRAINING ONLY ===
train_monthly_rainfall = train.groupby("month")["rainfall"].mean().reset_index()

# === Standardize Rainfall Data for Clustering ===
scaler = StandardScaler()
train_monthly_rainfall["rainfall_scaled"] = scaler.fit_transform(train_monthly_rainfall[["rainfall"]])

# === Automatic Cluster Selection with Elbow & Silhouette ===
def find_optimal_clusters(data, max_k=10):
    distortions, silhouette_scores = [], []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, cluster_labels))

    # Find best k: Prioritize silhouette but use elbow if close
    elbow_k = k_range[np.argmin(np.gradient(distortions))]
    silhouette_k = k_range[np.argmax(silhouette_scores)]
    optimal_k = silhouette_k if abs(silhouette_k - elbow_k) <= 1 else elbow_k

    print(f"Optimal K: {optimal_k} (Elbow: {elbow_k}, Silhouette: {silhouette_k})")
    return optimal_k

# === Find Optimal Clusters Using ONLY TRAINING DATA ===
optimal_clusters = find_optimal_clusters(train_monthly_rainfall[["rainfall_scaled"]], max_k=10)

# === Apply K-Means Clustering Using ONLY TRAINING DATA ===
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
train_monthly_rainfall["season_cluster"] = kmeans.fit_predict(train_monthly_rainfall[["rainfall_scaled"]])

# === Assign Seasonal Labels (Sorted by Rainfall) ===
cluster_means = train_monthly_rainfall.groupby("season_cluster")["rainfall"].mean().sort_values()
cluster_mapping = {cluster: f"Season_{i+1}" for i, cluster in enumerate(cluster_means.index, 1)}
train_monthly_rainfall["season"] = train_monthly_rainfall["season_cluster"].map(cluster_mapping)

# === Assign New Season Labels to Train & Test Data WITHOUT Using Test Rainfall ===
season_dict = dict(zip(train_monthly_rainfall["month"], train_monthly_rainfall["season"]))
train["data_season"] = train["month"].map(season_dict)
test["data_season"] = test["month"].map(season_dict)  # Assign based on TRAINING clusters

# === Plot Monthly Rainfall with Clustered Seasons ===
plt.figure(figsize=(10, 5))
sns.barplot(x="month", y="rainfall", hue="season", data=train_monthly_rainfall, palette="coolwarm")
plt.xlabel("Month")
plt.ylabel("Avg Rainfall")
plt.title("Rainfall Trend-Based Seasonal Clustering")
plt.legend(title="Season", loc="upper right")
plt.show()

# === Map Wind Direction to Cardinal Directions ===
def wind_to_cardinal(degrees):
    """Maps wind direction degrees to cardinal directions."""
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

train["wind_cardinal"] = train["winddirection"].apply(wind_to_cardinal)
test["wind_cardinal"] = test["winddirection"].apply(wind_to_cardinal)

# === Group by Data-Driven Season and Month for Wind Analysis ===
wind_by_data_season = train.groupby(['data_season', 'wind_cardinal']).size().unstack(fill_value=0)
wind_by_month = train.groupby(['month', 'wind_cardinal']).size().unstack(fill_value=0)

# === Visualize Wind Direction by Data-Driven Season ===
plt.figure(figsize=(12, 6))
sns.heatmap(wind_by_data_season.T, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Wind Direction Distribution by Data-Driven Season')
plt.ylabel('Wind Cardinal')
plt.xlabel('Season')
plt.show()

# === Visualize Wind Direction by Month ===
plt.figure(figsize=(12, 6))
sns.heatmap(wind_by_month.T, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Wind Direction Distribution by Month')
plt.ylabel('Wind Cardinal')
plt.xlabel('Month')
plt.show()


train.head()


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === Fill Missing Wind Direction Values ===
train = train.assign(winddirection=train["winddirection"].fillna(train["winddirection"].mode()[0]))
test = test.assign(winddirection=test["winddirection"].fillna(test["winddirection"].mode()[0]))

# === Encode Season Labels Before Splitting ===
season_encoder = LabelEncoder()
train["season_encoded"] = season_encoder.fit_transform(train["data_season"])
test["season_encoded"] = season_encoder.transform(test["data_season"])

# === Select Features (Excluding ID, Rainfall, and Non-Numeric Features) ===
features = [col for col in train.columns if col not in ["rainfall", "id", "data_season", "wind_cardinal"]]
X = train[features]
y = train["rainfall"]

# === Identify Continuous Features for Scaling ===
continuous_features = [col for col in X.columns if col not in ["season_encoded"]]

# === Standardize Continuous Features (Using a Copy to Avoid Warnings) ===
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_features] = scaler.fit_transform(X_scaled[continuous_features].astype(float))

# === Cross-Validation Setup ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_auc_scores, train_logloss_scores, train_accuracy_scores = [], [], []
val_auc_scores, val_logloss_scores, val_accuracy_scores = [], [], []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y), 1):
    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train Logistic Regression Model
    log_reg = LogisticRegression(random_state=42, max_iter=500)
    log_reg.fit(X_train, y_train)

    # === Evaluate on Train Set ===
    y_train_proba = log_reg.predict_proba(X_train)[:, 1]
    y_train_pred = log_reg.predict(X_train)

    train_auc = roc_auc_score(y_train, y_train_proba)
    train_logloss = log_loss(y_train, y_train_proba)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    train_auc_scores.append(train_auc)
    train_logloss_scores.append(train_logloss)
    train_accuracy_scores.append(train_accuracy)

    # === Evaluate on Validation Set ===
    y_val_proba = log_reg.predict_proba(X_val)[:, 1]
    y_val_pred = log_reg.predict(X_val)

    val_auc = roc_auc_score(y_val, y_val_proba)
    val_logloss = log_loss(y_val, y_val_proba)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    val_auc_scores.append(val_auc)
    val_logloss_scores.append(val_logloss)
    val_accuracy_scores.append(val_accuracy)

    print(f"Fold {fold} → Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

# === Print Cross-Validation Metrics ===
print(f"\nMean Train AUC: {np.mean(train_auc_scores):.4f} ± {np.std(train_auc_scores):.4f}")
print(f"Mean Train Log Loss: {np.mean(train_logloss_scores):.4f} ± {np.std(train_logloss_scores):.4f}")
print(f"Mean Train Accuracy: {np.mean(train_accuracy_scores):.4f} ± {np.std(train_accuracy_scores):.4f}")

print(f"\nMean Validation AUC: {np.mean(val_auc_scores):.4f} ± {np.std(val_auc_scores):.4f}")
print(f"Mean Validation Log Loss: {np.mean(val_logloss_scores):.4f} ± {np.std(val_logloss_scores):.4f}")
print(f"Mean Validation Accuracy: {np.mean(val_accuracy_scores):.4f} ± {np.std(val_accuracy_scores):.4f}")

# === Train Final Model on Full Data ===
log_reg_final = LogisticRegression(random_state=42, max_iter=500)
log_reg_final.fit(X_scaled, y)

# === Make Predictions on Test Data for Submission ===
test_scaled = test.copy()
test_scaled[continuous_features] = scaler.transform(test_scaled[continuous_features].astype(float))
X_test_submission = test_scaled[features]

# Predict on test set for submission
test_predictions = log_reg_final.predict_proba(X_test_submission)[:, 1]

# === Prepare Submission File ===
submission = pd.DataFrame({
    "id": test["id"],  # Assuming the test set contains an "id" column for submission
    "rainfall": test_predictions
})

# Save the predictions to a CSV file for submission
submission.to_csv("submission.csv", index=False)

print("\nSubmission file created: submission.csv")
submission.head()

