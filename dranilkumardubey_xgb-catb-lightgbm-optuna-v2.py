# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error, accuracy_score, f1_score, recall_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import optuna
from skopt import BayesSearchCV

import warnings
warnings.filterwarnings("ignore")


# Load data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_data = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# Encode categorical features
label_encoders = {}
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le


# Encode target label
target_encoder = LabelEncoder()
train_data['Fertilizer Name'] = target_encoder.fit_transform(train_data['Fertilizer Name'])


# Prepare features and target
X = train_data.drop(['id', 'Fertilizer Name'], axis=1)
y = train_data['Fertilizer Name']
X_test = test_data.drop(['id'], axis=1)


# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# ======== OPTUNA for XGBoost ==========
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
    }
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return rmsle(y_val, preds)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
best_xgb_params = study.best_params


# ======== Final XGBoost Model ==========
xgb_model = XGBClassifier(**best_xgb_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X, y)
xgb_preds = xgb_model.predict(X_test)


# ======== BayesianSearchCV for CatBoost ==========
cat_model = CatBoostClassifier(verbose=0, random_state=42)
search_space_cat = {
    'depth': (4, 10),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'iterations': (100, 1000)
}
opt_cat = BayesSearchCV(cat_model, search_space_cat, n_iter=15, cv=3, scoring='accuracy', random_state=42)
opt_cat.fit(X, y)
cat_preds = opt_cat.predict(X_test)


# ======== BayesianSearchCV for LightGBM ==========
lgb_model = LGBMClassifier(random_state=42)
search_space_lgb = {
    'num_leaves': (31, 150),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'n_estimators': (100, 1000)
}
opt_lgb = BayesSearchCV(lgb_model, search_space_lgb, n_iter=15, cv=3, scoring='accuracy', random_state=42)
opt_lgb.fit(X, y)
lgb_preds = opt_lgb.predict(X_test)


# ======== Choose best model prediction (e.g., XGB) ==========
final_preds = xgb_preds


# Decode labels
submission_data['Fertilizer Name'] = target_encoder.inverse_transform(final_preds)
submission_data.to_csv("submission.csv", index=False)


# ======== Evaluation on validation ==========
val_preds = xgb_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation F1 Score:", f1_score(y_val, val_preds, average='weighted'))
print("Validation Recall:", recall_score(y_val, val_preds, average='weighted'))
print("Validation RMSLE:", rmsle(y_val, val_preds))


# ======== Visualization of predictions ==========
plt.figure(figsize=(10,5))
sns.countplot(x=submission_data['Fertilizer Name'])
plt.title("Distribution of Predicted Fertilizer Names")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




