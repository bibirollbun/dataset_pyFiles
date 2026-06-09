import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE


train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
# Display dataset info
print("Train Data Information:")
print(train.info())

print("\nTest Data Information:")
print(test.info())


categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()


numerical_cols.remove("loan_status")


for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)

for col in numerical_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le


train["loan_income_ratio"] = train["loan_amnt"] / train["person_income"]
test["loan_income_ratio"] = test["loan_amnt"] / test["person_income"]

train["credit_age"] = train["person_age"] - (train["cb_person_cred_hist_length"] / 12)
test["credit_age"] = test["person_age"] - (test["cb_person_cred_hist_length"] / 12)

train["annual_repayment_ratio"] = train["loan_amnt"] / train["loan_int_rate"]
test["annual_repayment_ratio"] = test["loan_amnt"] / test["loan_int_rate"]


# Removing ID as it does not contribute to prediction
drop_cols = ["id"]  
train.drop(columns=drop_cols, axis=1, inplace=True)
test.drop(columns=drop_cols, axis=1, inplace=True)


X = train.drop(columns=["loan_status"])
y = train["loan_status"]


smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'lambda': trial.suggest_float('lambda', 0.0, 5.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)  # Handling class imbalance
    }
    
    model = xgb.XGBClassifier(**params, random_state=42, eval_metric="auc", use_label_encoder=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc").mean()
    
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)


best_params = study.best_params
print("\nBest Parameters from Optuna:", best_params)


final_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric="auc", use_label_encoder=False)
final_model.fit(X_train, y_train)


# Getting Probability Scores
y_pred_proba = final_model.predict_proba(X_val)[:, 1]  
# Converting to Binary
y_pred = (y_pred_proba > 0.5).astype(int)  


roc_auc = roc_auc_score(y_val, y_pred_proba)
print("\nValidation ROC-AUC Score:", roc_auc)
print("\nClassification Report:\n", classification_report(y_val, y_pred))


plt.figure(figsize=(12,6))
xgb.plot_importance(final_model, importance_type="gain", max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.show()


test_preds = final_model.predict_proba(test)[:, 1]


submission = pd.DataFrame({
    "id": test.index,
    "loan_status": test_preds
})
submission.to_csv("submission.csv", index=False)
print("\nSubmission File Saved Successfully")

