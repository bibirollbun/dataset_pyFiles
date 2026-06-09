import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


df


df.info()


df.describe()


df['rainfall'].value_counts()



# Rename 'temparature' column if needed
df.rename(columns={"temparature": "temperature"}, inplace=True)

# Compute the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()



# Drop highly correlated features
df.drop(columns=["mintemp", "dewpoint", "temperature", "pressure"], inplace=True)


# Encode cyclical features
df["sin_day"] = np.sin(2 * np.pi * df["day"] / 365)
df["cos_day"] = np.cos(2 * np.pi * df["day"] / 365)
df["sin_winddir"] = np.sin(2 * np.pi * df["winddirection"] / 360)
df["cos_winddir"] = np.cos(2 * np.pi * df["winddirection"] / 360)


# Drop original cyclical features after transformation
df.drop(columns=["day", "winddirection"], inplace=True)


# Scale numerical features
scaler = StandardScaler()
scaled_features = ["maxtemp", "humidity", "cloud", "sunshine", "windspeed"]
df[scaled_features] = scaler.fit_transform(df[scaled_features])


# Separate features and target variable
X = df.drop(columns=["rainfall"])
y = df["rainfall"]


# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Convert back to DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["rainfall"] = y_resampled


# Save processed data for further use
df_resampled.to_csv("processed_rainfall_data.csv", index=False)


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the processed dataset
df = pd.read_csv("processed_rainfall_data.csv")

# Split into features (X) and target variable (y)
X = df.drop(columns=["rainfall"])
y = df["rainfall"]

# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


df


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Train Logistic Regression model
log_reg = LogisticRegression(class_weight="balanced", random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# Evaluate using ROC-AUC
log_reg_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Logistic Regression ROC-AUC: {log_reg_auc:.4f}")



from xgboost import XGBClassifier

# Train XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)

# Make predictions
y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]

# Evaluate using ROC-AUC
xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")



import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Define the objective function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold)

        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        auc_scores.append(roc_auc_score(y_val_fold, y_pred_proba))

    return np.mean(auc_scores)

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Best hyperparameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")



best_xgb = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
best_xgb.fit(X_train, y_train)

y_pred_best = best_xgb.predict_proba(X_test)[:, 1]
final_auc = roc_auc_score(y_test, y_pred_best)

print(f"Final Tuned XGBoost ROC-AUC: {final_auc:.4f}")



# Load test set
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Rename 'temparature' to 'temperature' (if needed)
test_df.rename(columns={"temparature": "temperature"}, inplace=True)

# Drop unnecessary columns (same as training data)
test_df.drop(columns=["mintemp", "dewpoint", "temperature", "pressure"], inplace=True)

# Encode cyclical features
test_df["sin_day"] = np.sin(2 * np.pi * test_df["day"] / 365)
test_df["cos_day"] = np.cos(2 * np.pi * test_df["day"] / 365)
test_df["sin_winddir"] = np.sin(2 * np.pi * test_df["winddirection"] / 360)
test_df["cos_winddir"] = np.cos(2 * np.pi * test_df["winddirection"] / 360)

# Drop original cyclical features
test_df.drop(columns=["day", "winddirection"], inplace=True)

# Scale numerical features (using the same scaler from training)
test_df[scaled_features] = scaler.transform(test_df[scaled_features])


# Generate predictions
test_df["rainfall_probability"] = best_xgb.predict_proba(test_df)[:, 1]

# Create submission file
submission = test_df[["id", "rainfall_probability"]]
submission.to_csv("submission.csv", index=False)

