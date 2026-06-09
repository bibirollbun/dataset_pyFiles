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

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.head()


test.head()


print("Train Dataset Info:")
print(train.info())
print(train.describe())
print(test.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

train.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# print("Test Dataset Info:")
# print(test.info())
# print(test.describe())
# print(test.isnull().sum())

# test.hist(figsize=(15, 10))
# plt.tight_layout()
# plt.show()


missing_values_train = pd.DataFrame({'Feature': train.columns,
                              '[TRAIN] No. of Missing Values': train.isnull().sum().values,})


unique_values = pd.DataFrame({'Feature': train.columns,
                              'No. of Unique Values [FROM TRAIN]': train.nunique().values})

feature_types = pd.DataFrame({'Feature': train.columns,
                              'DataType': train.dtypes})

merged_df = missing_values_train.copy()
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df.style.background_gradient(cmap='viridis')


# Compute correlation matrix
corr_matrix = train.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


train_updated = train.copy()
train_updated = train_updated.drop(columns=['id'])
train_updated = train_updated.drop(columns=['rainfall'])
train_updated['difftemp'] = train_updated["maxtemp"] - train_updated["mintemp"]
train_updated = train_updated.drop(columns=['maxtemp'])
train_updated = train_updated.drop(columns=['mintemp'])

train_updated['sunshine_log'] = np.log1p(train_updated['sunshine'])
train_updated['windspeed_log'] = np.log1p(train_updated['windspeed'])
train_updated.drop(columns=['sunshine', 'windspeed'], inplace=True)
train_updated['winddir_rad'] = np.deg2rad(train_updated['winddirection']).fillna(0)
train_updated['winddir_sin'] = np.sin(train_updated['winddir_rad'])
train_updated['winddir_cos'] = np.cos(train_updated['winddir_rad'])
train_updated.drop(columns=['winddirection', 'winddir_rad'], inplace=True)

train_updated['season'] = pd.cut(train_updated['day'], bins=[0, 90, 180, 270, 366], labels=['winter', 'spring', 'summer', 'fall'])
train_updated = pd.get_dummies(train_updated, columns=['season'], drop_first=True)


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 0: Load your data
features = ["difftemp", "temparature", "dewpoint"]
X_temp = train_updated[features]

# Step 1: Standardize
scaler = StandardScaler()
X_temp_scaled = scaler.fit_transform(X_temp)

# Step 2: Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_temp_scaled)

# Step 3: Explained Variance
explained_var = pca.explained_variance_ratio_
print("Explained Variance Ratios:", explained_var)
print("Cumulative Variance:", explained_var.cumsum())

# Step 4: Select number of components (e.g., keep 95% variance)
pca = PCA(n_components=1)  # Keep top 2 components
X_pca_selected = pca.fit_transform(X_temp_scaled)

# Step 5: Add back to dataset
train_updated["temp_pca1"] = X_pca_selected[:, 0]

# Optional: drop original correlated features
train_updated = train_updated.drop(columns=features)


# Compute correlation matrix
corr_matrix = train_updated.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train_PCA = train.drop(columns=['id', 'rainfall'])

# Standardize features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_PCA)

# Apply PCA, keeping all components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio & cumulative variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot individual & cumulative variance explained
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100, alpha=0.6, label="Individual Variance")
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker="o", linestyle="--", color="red", label="Cumulative Variance")
plt.axhline(y=95, color="green", linestyle="--", label="95% Variance Threshold")
plt.xlabel("Number of Principal Components")
plt.ylabel("Variance Explained (%)")
plt.title("PCA: Individual and Cumulative Variance Explained")
plt.legend()
plt.show()

# # Plot feature contributions to PCs
# plt.figure(figsize=(12, 6))
# sns.heatmap(pca.components_[:10], cmap="coolwarm", annot=True, fmt=".2f", xticklabels=df.columns, yticklabels=[f"PC{i+1}" for i in range(10)])
# plt.xlabel("Original Features")
# plt.ylabel("Principal Components")
# plt.title("Feature Contribution to Principal Components")
# plt.show()

# Standardize features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_PCA)

# Apply PCA with 6 components
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(8)])

# Display PCA-transformed data
pca_df



import lightgbm as lgb

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_updated, train['rainfall'], test_size=0.3, random_state=42
)

# Initialize the model
model = lgb.LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict probabilities for the validation set
val_preds = model.predict_proba(X_val)[:, 1]

print(f"Validation Predictions: {val_preds[:5]}")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# Predict probabilities and labels
val_probs = model.predict_proba(X_val)[:, 1]  # Probability of the positive class
val_preds = model.predict(X_val)  # Direct label predictions

# Calculate metrics
accuracy = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds)
recall = recall_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds)
roc_auc = roc_auc_score(y_val, val_probs)
logloss = log_loss(y_val, val_probs)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Log Loss: {logloss:.4f}")


import lightgbm as lgb

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    pca_df, train['rainfall'], test_size=0.2, random_state=42
)

model_fullpca = lgb.LGBMClassifier(
    objective='binary',
    metric='auc'
)

model_fullpca.fit(X_train, y_train)
val_preds = model_fullpca.predict_proba(X_val)[:, 1]


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# Predict probabilities and labels
val_probs = model_fullpca.predict_proba(X_val)[:, 1]  # Probability of the positive class
val_preds = model_fullpca.predict(X_val)  # Direct label predictions

# Calculate metrics
accuracy = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds)
recall = recall_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds)
roc_auc = roc_auc_score(y_val, val_probs)
logloss = log_loss(y_val, val_probs)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Log Loss: {logloss:.4f}")


import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

def objective(trial):
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'n_jobs': -1,
        'class_weight': 'balanced',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5.0)
    }

    model = LGBMClassifier(**params, n_estimators=1000)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(100), log_evaluation(0)]
    )

    preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)
    return loss


# Split your data first
X_train, X_val, y_train, y_val = train_test_split(train_updated, train['rainfall'], stratify=train['rainfall'], test_size=0.3)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, timeout=1800)  # 50 trials or 30 mins


print("Best trial:")
print(study.best_trial)
print("Best params:", study.best_params)


best_model = LGBMClassifier(
    **study.best_params,
    objective='binary',
    class_weight='balanced',
    eval_metric='auc',
    n_estimators=1000
)

best_model.fit(train_updated, train['rainfall'])


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# Predict probabilities and labels
val_probs = best_model.predict_proba(X_val)[:, 1]  # Probability of the positive class
val_preds = best_model.predict(X_val)  # Direct label predictions

# Calculate metrics
accuracy = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds)
recall = recall_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds)
roc_auc = roc_auc_score(y_val, val_probs)
logloss = log_loss(y_val, val_probs)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Log Loss: {logloss:.4f}")





import matplotlib.pyplot as plt
import lightgbm as lgb

lgb.plot_importance(best_model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()


test_updated = test.copy()
test_updated = test_updated.drop(columns=['id'])
test_updated['difftemp'] = test_updated["maxtemp"] - test_updated["mintemp"]
test_updated = test_updated.drop(columns=['maxtemp'])
test_updated = test_updated.drop(columns=['mintemp'])
test_updated['sunshine_log'] = np.log1p(test_updated['sunshine'])
test_updated['windspeed_log'] = np.log1p(test_updated['windspeed'])
test_updated.drop(columns=['sunshine', 'windspeed'], inplace=True)

test_updated['winddir_rad'] = np.deg2rad(test_updated['winddirection']).fillna(0)
test_updated['winddir_sin'] = np.sin(test_updated['winddir_rad'])
test_updated['winddir_cos'] = np.cos(test_updated['winddir_rad'])
test_updated.drop(columns=['winddirection', 'winddir_rad'], inplace=True)

test_updated['season'] = pd.cut(test_updated['day'], bins=[0, 90, 180, 270, 366], labels=['winter', 'spring', 'summer', 'fall'])
test_updated = pd.get_dummies(test_updated, columns=['season'], drop_first=True)

# Step 0: Load your data
features = ["difftemp", "temparature", "dewpoint"]
X_temp = test_updated[features]

# Step 1: Standardize
scaler = StandardScaler()
X_temp_scaled = scaler.fit_transform(X_temp)

# Step 2: Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_temp_scaled)

# Step 3: Explained Variance
explained_var = pca.explained_variance_ratio_
print("Explained Variance Ratios:", explained_var)
print("Cumulative Variance:", explained_var.cumsum())

# Step 4: Select number of components (e.g., keep 95% variance)
pca = PCA(n_components=1)  # Keep top 2 components
X_pca_selected = pca.fit_transform(X_temp_scaled)

# Step 5: Add back to dataset
test_updated["temp_pca1"] = X_pca_selected[:, 0]

# Optional: drop original correlated features
test_updated = test_updated.drop(columns=features)

# Predict on the test dataset
test_preds = best_model.predict(test_updated)

# Create a DataFrame with the required format
output = pd.DataFrame({
    'id': range(2190, 2190 + len(test_preds)),  # Adjust the starting ID as needed
    'rainfall': test_preds
})

# Save the output to a CSV file
output.to_csv('rainfall_predictions.csv', index=False)

print(output.head(10))  # Preview the first 10 rows



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# test_PCA = test.copy()
# test_PCA = test_PCA.drop(columns=['id'])

# # Standardize features before PCA
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(test_PCA)

# # Apply PCA, keeping all components
# pca = PCA()
# X_pca = pca.fit_transform(X_scaled)

# # Explained variance ratio & cumulative variance
# explained_variance = pca.explained_variance_ratio_
# cumulative_variance = np.cumsum(explained_variance)

# # Plot individual & cumulative variance explained
# plt.figure(figsize=(10, 5))
# plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100, alpha=0.6, label="Individual Variance")
# plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker="o", linestyle="--", color="red", label="Cumulative Variance")
# plt.axhline(y=95, color="green", linestyle="--", label="95% Variance Threshold")
# plt.xlabel("Number of Principal Components")
# plt.ylabel("Variance Explained (%)")
# plt.title("PCA: Individual and Cumulative Variance Explained")
# plt.legend()
# plt.show()

# # # Plot feature contributions to PCs
# # plt.figure(figsize=(12, 6))
# # sns.heatmap(pca.components_[:10], cmap="coolwarm", annot=True, fmt=".2f", xticklabels=df.columns, yticklabels=[f"PC{i+1}" for i in range(10)])
# # plt.xlabel("Original Features")
# # plt.ylabel("Principal Components")
# # plt.title("Feature Contribution to Principal Components")
# # plt.show()

# # Standardize features before PCA
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(test_PCA)

# # Apply PCA with 6 components
# pca = PCA(n_components=8)
# X_pca = pca.fit_transform(X_scaled)

# # Create a DataFrame with the PCA results
# pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(8)])

# # Display PCA-transformed data
# pca_df



# import lightgbm as lgb

# from sklearn.model_selection import train_test_split

# # Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(
#     pca_df, test['rainfall'], test_size=0.2, random_state=42
# )

# model_fullpca = lgb.LGBMClassifier(
#     objective='binary',
#     metric='binary_logloss',
#     random_state=42
# )

# model_fullpca.fit(X_train, y_train)
# val_preds = model_fullpca.predict_proba(X_val)[:, 1]


# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# # Predict probabilities and labels
# val_probs = model_fullpca.predict_proba(X_val)[:, 1]  # Probability of the positive class
# val_preds = model_fullpca.predict(X_val)  # Direct label predictions

# # Calculate metrics
# accuracy = accuracy_score(y_val, val_preds)
# precision = precision_score(y_val, val_preds)
# recall = recall_score(y_val, val_preds)
# f1 = f1_score(y_val, val_preds)
# roc_auc = roc_auc_score(y_val, val_probs)
# logloss = log_loss(y_val, val_probs)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")
# print(f"ROC AUC: {roc_auc:.4f}")
# print(f"Log Loss: {logloss:.4f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import early_stopping, log_evaluation

# 1. Load data
df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')  # update path as needed

# 2. Basic feature engineering
df['dewpoint_spread'] = df['temparature'] - df['dewpoint']
df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1)
df['humid_cloud'] = df['humidity'] * df['cloud']

# Cyclical encoding for wind direction
df['winddirection'] = df['winddirection'].fillna(0)
df['winddir_rad'] = np.deg2rad(df['winddirection'])
df['winddir_sin'] = np.sin(df['winddir_rad'])
df['winddir_cos'] = np.cos(df['winddir_rad'])

# Log transforms
df['sunshine_log'] = np.log1p(df['sunshine'])
df['windspeed_log'] = np.log1p(df['windspeed'])

# Drop columns not needed or replaced
drop_cols = ['id', 'winddir_rad', 'sunshine', 'windspeed', 'winddirection', 
             'temparature', 'maxtemp', 'mintemp', 'dewpoint']
df = df.drop(columns=drop_cols)

# Optional: Fill any remaining NaNs
df = df.fillna(0)

# 3. Split data
X = df.drop(columns=['rainfall'])
y = df['rainfall']
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Optuna hyperparameter tuning
def objective(trial):
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'class_weight': 'balanced',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5.0)
    }

    model = lgb.LGBMClassifier(**params, n_estimators=1000)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(50), log_evaluation(0)]
    )

    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

# Run study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 5. Train best model
best_model = lgb.LGBMClassifier(**study.best_params,
                                 objective='binary',
                                 metric='auc',
                                 class_weight='balanced',
                                 n_estimators=1000)

best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[early_stopping(50), log_evaluation(0)]
)

# 6. Evaluate
probs = best_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, probs)
print(f"\nFinal ROC AUC: {auc:.4f}")

# Plot feature importance
lgb.plot_importance(best_model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# === 0. SETUP ===
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import optuna
import warnings
warnings.filterwarnings("ignore")

# === 1. LOAD DATA ===
df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')  # or path to your actual training data
X = df.drop(columns=["rainfall"])
y = df["rainfall"]

# === 2. FEATURE ENGINEERING ===
X["sunshine_log"] = np.log1p(X["sunshine"])
X["windspeed_log"] = np.log1p(X["windspeed"])
X["dewpoint_spread"] = X["temparature"] - X["dewpoint"]
X["cloud_sun_ratio"] = X["cloud"] / (X["sunshine"] + 1)
X["humid_cloud"] = X["humidity"] * X["cloud"]

# Handle wind direction as cyclical
X["winddirection"] = X["winddirection"].fillna(0)
X["winddir_rad"] = np.deg2rad(X["winddirection"])
X["winddir_sin"] = np.sin(X["winddir_rad"])
X["winddir_cos"] = np.cos(X["winddir_rad"])
X.drop(columns=["winddir_rad"], inplace=True)

# Optional: drop raw columns now replaced
X.drop(columns=["sunshine", "windspeed", "winddirection"], inplace=True)

# === 3. TRAIN/VAL SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === 4. OPTUNA FOR AUC ===
def objective(trial):
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'class_weight': 'balanced',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5.0)
    }

    model = LGBMClassifier(**params, n_estimators=1000, random_state=42)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(50), log_evaluation(0)]
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# === 5. FINAL MODEL WITH BEST PARAMS ===
best_model = LGBMClassifier(
    **study.best_params,
    n_estimators=1000,
    objective='binary',
    class_weight='balanced',
    random_state=42
)

best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[early_stopping(50), log_evaluation(0)]
)

# === 6. EVALUATE & PLOT ROC ===
val_probs = best_model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, val_probs)
print(f"Validation ROC AUC: {auc_score:.4f}")

fpr, tpr, _ = roc_curve(y_val, val_probs)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# === 7. FEATURE IMPORTANCE PLOT ===
plot_importance(best_model, max_num_features=15, importance_type='gain')
plt.title("Top Feature Importances (Gain)")
plt.tight_layout()
plt.show()

