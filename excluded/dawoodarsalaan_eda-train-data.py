import pandas as pd

# Load the training dataset
train_path = r"/kaggle/input/playground-series-s5e3/train.csv"
train_df = pd.read_csv(train_path)

# Display basic information and first few rows
train_info = train_df.info()
train_head = train_df.head()

train_info, train_head


import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# 1. Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=train_df["rainfall"], palette="viridis")
plt.title("Rainfall Class Distribution")
plt.xlabel("Rainfall (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


# 2. Feature Distributions
train_df.drop(columns=["id", "day", "rainfall"]).hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()


# 3. Correlation Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = train_df.drop(columns=["id", "day"]).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()



# 4. Boxplots for Key Features
key_features = ["humidity", "dewpoint", "cloud", "sunshine", "pressure", "windspeed"]

plt.figure(figsize=(12, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x=train_df["rainfall"], y=train_df[feature], palette="viridis")
    plt.title(f"{feature} vs Rainfall")

plt.tight_layout()
plt.show()


# Check for non-numeric or unexpected data types
selected_features = ["humidity", "dewpoint", "cloud", "sunshine", "rainfall"]
train_df[selected_features].dtypes


# 1. Scatter plots of key features against each other
plt.figure(figsize=(12, 8))

# Scatter plot of humidity vs. dewpoint
plt.subplot(2, 2, 1)
sns.scatterplot(x=train_df["humidity"], y=train_df["dewpoint"], hue=train_df["rainfall"], palette="viridis", alpha=0.7)
plt.title("Humidity vs Dewpoint")

# Scatter plot of cloud vs. sunshine
plt.subplot(2, 2, 2)
sns.scatterplot(x=train_df["cloud"], y=train_df["sunshine"], hue=train_df["rainfall"], palette="viridis", alpha=0.7)
plt.title("Cloud Cover vs Sunshine")

# Scatter plot of humidity vs. cloud
plt.subplot(2, 2, 3)
sns.scatterplot(x=train_df["humidity"], y=train_df["cloud"], hue=train_df["rainfall"], palette="viridis", alpha=0.7)
plt.title("Humidity vs Cloud Cover")

# Scatter plot of pressure vs. windspeed
plt.subplot(2, 2, 4)
sns.scatterplot(x=train_df["pressure"], y=train_df["windspeed"], hue=train_df["rainfall"], palette="viridis", alpha=0.7)
plt.title("Pressure vs Windspeed")

plt.tight_layout()
plt.show()


# 2. Rainfall Probability by Wind Direction
plt.figure(figsize=(10, 5))
sns.histplot(x=train_df["winddirection"], hue=train_df["rainfall"], bins=30, kde=True, palette="viridis", alpha=0.6)
plt.title("Rainfall Probability by Wind Direction")
plt.xlabel("Wind Direction (degrees)")
plt.ylabel("Count")
plt.show()


# 3. Rainfall Probability Over Time (Day Feature)
plt.figure(figsize=(10, 5))
rainfall_by_day = train_df.groupby("day")["rainfall"].mean()  # Calculate mean rainfall occurrence per day
sns.lineplot(x=rainfall_by_day.index, y=rainfall_by_day.values, marker="o", color="purple")
plt.title("Rainfall Probability Over Time (Day Feature)")
plt.xlabel("Day")
plt.ylabel("Probability of Rainfall")
plt.grid(True)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Define features and target
X = train_df.drop(columns=["id", "day", "rainfall"])  # Drop non-informative columns
y = train_df["rainfall"]  # Target variable

# Split into training and validation sets (80% train, 20% test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred = log_reg.predict(X_val_scaled)
y_prob = log_reg.predict_proba(X_val_scaled)[:, 1]  # Probability scores for ROC-AUC

# Evaluate performance
accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_prob)
classification_rep = classification_report(y_val, y_pred)

accuracy, roc_auc, classification_rep


from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_val)
y_prob_rf = rf_model.predict_proba(X_val)[:, 1]

# Evaluate performance
accuracy_rf = accuracy_score(y_val, y_pred_rf)
roc_auc_rf = roc_auc_score(y_val, y_prob_rf)
classification_rep_rf = classification_report(y_val, y_pred_rf)

accuracy_rf, roc_auc_rf, classification_rep_rf


pip install xgboost


from xgboost import XGBClassifier

# Train XGBoost model
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_val)
y_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]

# Evaluate performance
accuracy_xgb = accuracy_score(y_val, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_val, y_prob_xgb)
classification_rep_xgb = classification_report(y_val, y_pred_xgb)

accuracy_xgb, roc_auc_xgb, classification_rep_xgb



from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Initialize Random Forest model
rf_model_tuned = RandomForestClassifier(random_state=42)

# Perform Grid Search with Cross Validation (CV = 3 for faster search)
grid_search = GridSearchCV(rf_model_tuned, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get best model
best_rf_model = grid_search.best_estimator_

# Predictions with best model
y_pred_best_rf = best_rf_model.predict(X_val)
y_prob_best_rf = best_rf_model.predict_proba(X_val)[:, 1]

# Evaluate performance
accuracy_best_rf = accuracy_score(y_val, y_pred_best_rf)
roc_auc_best_rf = roc_auc_score(y_val, y_prob_best_rf)
classification_rep_best_rf = classification_report(y_val, y_pred_best_rf)
best_params_rf = grid_search.best_params_

accuracy_best_rf, roc_auc_best_rf, classification_rep_best_rf, best_params_rf


pip install scikit-learn


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Define parameter grid for Random Forest
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Initialize Random Forest model
rf_model_tuned = RandomForestClassifier(random_state=42)

# Perform Grid Search with Cross Validation
grid_search = GridSearchCV(rf_model_tuned, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get best model
best_rf_model = grid_search.best_estimator_

# Predictions with best model
y_pred_best_rf = best_rf_model.predict(X_val)
y_prob_best_rf = best_rf_model.predict_proba(X_val)[:, 1]

# Evaluate performance
accuracy_best_rf = accuracy_score(y_val, y_pred_best_rf)
roc_auc_best_rf = roc_auc_score(y_val, y_prob_best_rf)
classification_rep_best_rf = classification_report(y_val, y_pred_best_rf)
best_params_rf = grid_search.best_params_

print("Best Parameters:", best_params_rf)
print("Accuracy:", accuracy_best_rf)
print("ROC AUC Score:", roc_auc_best_rf)
print("Classification Report:\n", classification_rep_best_rf)


# Load test dataset
test_path = r"/kaggle/input/playground-series-s5e3/test.csv"
test_df = pd.read_csv(test_path)

# Drop unnecessary columns (id is kept for submission format)
X_test = test_df.drop(columns=["day"])

# Predict probabilities using the optimized Random Forest model
test_predictions = best_rf_model.predict_proba(X_test.drop(columns=["id"]))[:, 1]

# Load sample submission file
submission_path = "/mnt/data/sample_submission.csv"
submission_df = pd.read_csv(submission_path)

# Prepare submission file
submission_df["rainfall"] = test_predictions

# Save final submission file
final_submission_path = "/mnt/data/final_submission.csv"
submission_df.to_csv(final_submission_path, index=False)

# Return the path of the saved file
final_submission_path

