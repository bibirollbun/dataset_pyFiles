import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


# Handling missing values (if any)
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)


# Define features and target
X = train_df.drop(columns=['id', 'rainfall'])  # Dropping ID & Target
y = train_df['rainfall']  # Target
X_test = test_df.drop(columns=['id'])


# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# Apply Bayesian Ridge for feature engineering
bayesian_ridge = BayesianRidge()
X_bayesian = bayesian_ridge.fit(X_scaled, y).predict(X_scaled).reshape(-1, 1)
X_test_bayesian = bayesian_ridge.predict(X_test_scaled).reshape(-1, 1)


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_bayesian, y, test_size=0.2, random_state=42, stratify=y)


# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Model Training
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='auc', random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
}


df_auc_scores = {}
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred)
    df_auc_scores[name] = auc_score
    print(f'{name} AUC: {auc_score}')


# Select best model
best_model_name = max(df_auc_scores, key=df_auc_scores.get)
best_model = models[best_model_name]
print(f'Best Model: {best_model_name}')


# Make final predictions on test set
test_predictions = best_model.predict_proba(X_test_bayesian)[:, 1]



# Save submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_predictions})
submission.to_csv("submission_bayrid.csv", index=False)
print("Submission file saved!")




