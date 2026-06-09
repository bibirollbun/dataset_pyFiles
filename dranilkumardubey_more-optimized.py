import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_log_error, confusion_matrix
from xgboost import XGBClassifier


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


# Label Encoding for categorical columns
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_target = LabelEncoder()

train['Soil Type'] = le_soil.fit_transform(train['Soil Type'])
test['Soil Type'] = le_soil.transform(test['Soil Type'])

train['Crop Type'] = le_crop.fit_transform(train['Crop Type'])
test['Crop Type'] = le_crop.transform(test['Crop Type'])

train['Fertilizer Name'] = le_target.fit_transform(train['Fertilizer Name'])


# Features and Target
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)


# EDA: Visual Target Distribution
plt.figure(figsize=(10, 4))
sns.countplot(x='Fertilizer Name', data=train)
plt.title('Fertilizer Class Distribution')
plt.xticks(rotation=45)
plt.show()


# Optuna optimization
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    
    model = XGBClassifier(**params)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
        
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best parameters:", study.best_params)


# Train final model with best params
best_params = study.best_params
best_params['use_label_encoder'] = False
best_params['eval_metric'] = 'mlogloss'

final_model = XGBClassifier(**best_params)
final_model.fit(X, y)


# Predict on test set
test_preds = final_model.predict(X_test)
submission['Fertilizer Name'] = le_target.inverse_transform(test_preds)
submission.to_csv('submission.csv', index=False)


# Evaluate on validation split for metrics
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
final_model.fit(X_train, y_train)
y_pred_val = final_model.predict(X_val)


# Metrics
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred_val))
precision = precision_score(y_val, y_pred_val, average='weighted')
recall = recall_score(y_val, y_pred_val, average='weighted')
f1 = f1_score(y_val, y_pred_val, average='weighted')
acc = accuracy_score(y_val, y_pred_val)

print("Validation RMSLE:", rmsle)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", acc)


# Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_val, y_pred_val), annot=True, fmt="d", cmap='viridis')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

