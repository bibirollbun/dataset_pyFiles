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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import warnings

warnings.filterwarnings("ignore")


# Load the datasets
train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

print(train_data.head())
print(test_data.head())


# Feature Engineering
for df in [train_data, test_data]:
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df["humidity_sunshine_ratio"] = df["humidity"] / (df["sunshine"] + 1)
    df["wind_power"] = df["windspeed"] ** 2
    df['dewpoint_diff'] = df['temparature'] - df['dewpoint']
    df["dewpoint_depression"] = df["maxtemp"] - df["dewpoint"]
    df['humidity_pressure_ratio'] = df['humidity'] / df['pressure']
    df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 0.1)  # Avoid division by zero
    df['wind_effect'] = df['windspeed'] * df['cloud']
    df["cloud_wind_factor"] = df["cloud"] * df["windspeed"]

train_df = train_data
test_df = test_data

print(train_df.head())
print(test_df.head())


# Step 1: Handle Missing Values
missing_values = train_df.isnull().sum()
print("Missing Values Per Column:\n", missing_values[missing_values > 0])


# Step 2: Remove Low-Variance Features
selector = VarianceThreshold(threshold=0.01)
X_train_filtered = train_df.drop(columns=["id", "rainfall"])

# Fit the selector
selector.fit(X_train_filtered)

# Get Variances for Each Feature
variances = pd.Series(selector.variances_, index=X_train_filtered.columns)

# Find low-variance features (below threshold)
low_variance_features = variances[variances < 0.01].index.tolist()

print("ğŸš« Features removed due to low variance:", low_variance_features)
print(f"âœ… Features kept after variance filtering: {X_train_filtered.shape[1] - len(low_variance_features)}")



# Step 3: Correlation Analysis & Drop Highly Correlated Features
corr_matrix = train_df.drop(columns=["id", "rainfall"]).corr()

# Heatmap for collinearity
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# Set a correlation threshold (e.g., 0.9)
correlation_threshold = 0.9

# Compute correlation matrix (excluding 'id' and 'rainfall')
corr_matrix = train_df.drop(columns=["id", "rainfall"]).corr()

# Find features that are highly correlated
highly_correlated_features = set()
correlated_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            highly_correlated_features.add(feature1)  # Keep track of features to drop
            correlated_pairs.append((feature1, feature2, corr_matrix.iloc[i, j]))

# Print Features Being Removed
print("ğŸš« Highly Correlated Features to Drop:")
for feature1, feature2, corr_value in correlated_pairs:
    print(f"  ğŸ”¹ {feature1} â†” {feature2} (Correlation: {corr_value:.2f})")

print(f"âœ… Total features removed: {len(highly_correlated_features)}")


# Drop highly correlated features (above 0.9)
corr_threshold = 0.9
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]

train_df = train_df.drop(columns=to_drop)
test_df = test_df.drop(columns=to_drop)

print("âœ… Dropped highly correlated features:", to_drop)


# Step 4: Feature Importance (Random Forest)
X = train_df.drop(columns=["id", "rainfall"])
y = train_df["rainfall"]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Keep top features
top_features = feature_importances["Feature"].tolist()
print("âœ… Top 10 Features by Importance:\n", feature_importances.head(10))


# Filter dataset with selected features
X = train_df[top_features]
X_test = test_df[top_features]


from sklearn.impute import SimpleImputer
import pandas as pd

# Ensure X_test is a DataFrame before applying imputation
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test, columns=top_features)  # Use the correct column names

# Apply median imputation
imputer = SimpleImputer(strategy="median")  # Can also use "mean" or "most_frequent"
X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)


# Step 5: Splitting Data Before Scaling (To Avoid Data Leakage)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"âœ… Data split completed: Training size: {len(X_train)}, Validation size: {len(X_valid)}")


# Step 5.1: Scaling (After Splitting)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

print("âœ… Data scaling completed.")


# ============================
# ğŸ”¹ Step 6: Train Individual Models
# ============================

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_valid)[:, 1]
rf_roc_auc = roc_auc_score(y_valid, rf_probs)

# Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_probs = mlp.predict_proba(X_valid)[:, 1]
mlp_roc_auc = roc_auc_score(y_valid, mlp_probs)

# XGBoost
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_probs = xgb.predict_proba(X_valid)[:, 1]
xgb_roc_auc = roc_auc_score(y_valid, xgb_probs)

# Print Model Performance
print(f"ğŸ”¹ Random Forest - ROC AUC: {rf_roc_auc:.4f}")
print(f"ğŸ”¹ Neural Network - ROC AUC: {mlp_roc_auc:.4f}")
print(f"ğŸ”¹ XGBoost - ROC AUC: {xgb_roc_auc:.4f}")


# ============================
# ğŸ”¹ Step 7: Stacking Classifier
# ============================

stacking_model = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('mlp', mlp),
        ('xgb', xgb)
    ],
    final_estimator=LogisticRegression(),
    passthrough=True
)

stacking_model.fit(X_train, y_train)
stacking_probs = stacking_model.predict_proba(X_valid)[:, 1]
stacking_roc_auc = roc_auc_score(y_valid, stacking_probs)

print(f"ğŸš€ Stacking Model - ROC AUC: {stacking_roc_auc:.4f}")


# ============================
# ğŸ”¹ Step 8: Select Best Model
# ============================

best_model = max(
    [('rf', rf_roc_auc, rf), ('mlp', mlp_roc_auc, mlp), ('xgb', xgb_roc_auc, xgb), ('stacking', stacking_roc_auc, stacking_model)],
    key=lambda x: x[1]
)[2]

print(f"ğŸ�† Best Model Selected: {best_model.__class__.__name__}")


# ============================
# ğŸ”¹ Step 9: Final Predictions
# ============================

final_probs = best_model.predict_proba(X_test)[:, 1]




# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": final_probs})
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file created successfully.")
print(submission.head())

