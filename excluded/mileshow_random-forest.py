import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# --- Suppress Warnings for Cleaner Output ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Data Loading and Preparation ---
print("--- Loading and Preparing Data ---")
# Load competition data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# --- Data Augmentation ---
try:
    original = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
    train = pd.concat([train, original], ignore_index=True)
    train.drop_duplicates(inplace=True)
    print("Successfully loaded and concatenated original dataset.")
except FileNotFoundError:
    print("Original dataset not found. Proceeding with competition data only.")

# --- Feature Engineering ---
print("Creating 'Soil_Crop_Interaction' feature.")
for df in [train, test]:
    df['Soil_Crop_Interaction'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)

# Drop ID columns
if 'id' in train.columns:
    train.drop('id', axis=1, inplace=True)
test_ids = test['id']
if 'id' in test.columns:
    test.drop('id', axis=1, inplace=True)

# --- Data Preparation for Random Forest ---
base_numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ['Soil Type', 'Crop Type', 'Soil_Crop_Interaction']
target_feature = 'Fertilizer Name'

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(train[target_feature])
print(f"Target '{target_feature}' encoded.")

train = pd.get_dummies(train, columns=categorical_features)
test = pd.get_dummies(test, columns=categorical_features)

# Align columns
train_cols = train.columns
test_cols = test.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    if c != target_feature:
        test[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    train[c] = 0
test = test[train.drop(columns=[target_feature]).columns]

X = train.drop(columns=[target_feature])
y = y_encoded

print("Data preparation complete.")
print(f"Total features being used: {len(X.columns)}")

# --- 2. Optuna Hyperparameter Optimization ---
print("\n--- Starting Optuna Optimization for Random Forest ---")
X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def mapk(actual, predicted, k=3):
    actual_wrapped = [[a] for a in actual]
    return np.mean([
        len(set(p[:k]) & set(a)) / min(len(a), k) for a, p in zip(actual_wrapped, predicted)
    ])

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train_opt, y_train_opt)
    
    probs = model.predict_proba(X_val_opt)
    top3_preds = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    return mapk(y_val_opt, top3_preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, timeout=7200) # 20 trials or 2 hours

print("Optimization finished.")
print("Best trial:")
best_params = study.best_params
print(f"  Value (MAP@3): {study.best_value}")
print("  Params: ")
for key, value in best_params.items():
    print(f"    {key}: {value}")


# --- 3. Final Model Training with Best Parameters ---
print("\n--- Training Random Forest Model using 5-Fold CV ---")
final_params = best_params.copy()
final_params.update({'random_state': 42, 'n_jobs': -1})

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
test_predictions = np.zeros((len(test), len(target_encoder.classes_)))
feature_importances = pd.DataFrame(index=X.columns)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"{'='*10} Fold {fold} {'='*10}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = RandomForestClassifier(**final_params)
    model.fit(X_train, y_train)
    
    feature_importances[f'fold_{fold}'] = model.feature_importances_
    test_predictions += model.predict_proba(test) / FOLDS
    
# --- VISUALIZATION 1: FEATURE IMPORTANCE ---
feature_importances['mean'] = feature_importances.mean(axis=1)
feature_importances.sort_values('mean', ascending=False, inplace=True)

plt.figure(figsize=(12, 10))
sns.barplot(x='mean', y=feature_importances.index[:20], data=feature_importances.head(20), palette='plasma')
plt.title('Top 20 Feature Importances (Random Forest, Averaged Across Folds)')
plt.xlabel('Average Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# --- 4. Create Submission File ---
print("\n--- Creating Submission File ---")
top_3_preds_indices = np.argsort(test_predictions, axis=1)[:, -3:][:, ::-1]
top_3_preds_labels = target_encoder.inverse_transform(top_3_preds_indices.ravel()).reshape(top_3_preds_indices.shape)

submission['Fertilizer Name'] = [' '.join(row) for row in top_3_preds_labels]
submission.to_csv('submission_rf.csv', index=False)

print("✅ Submission file 'submission_rf.csv' created successfully!")
display(submission.head())


