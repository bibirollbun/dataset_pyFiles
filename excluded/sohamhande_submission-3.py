import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb



# Load the data
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)



# Encoding categorical variables
label_encoders = {}
for col in train_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    if col in test_df.columns:
        test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le



# Splitting data
X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']


# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Scaling numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_df)



# Optuna Optimization
def objective(trial):
    model_type = trial.suggest_categorical("model", ["random_forest", "xgboost", "lightgbm"])
    
    if model_type == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, random_state=42)
    
    elif model_type == "xgboost":
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        model = xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                  max_depth=max_depth, use_label_encoder=False,
                                  eval_metric='logloss', random_state=42)
    
    else:  # LightGBM
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        num_leaves = trial.suggest_int("num_leaves", 10, 100)
        model = lgb.LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                   num_leaves=num_leaves, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)



# Running Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best parameters:", study.best_params)


# Train best model on full data
best_params = study.best_params
if best_params["model"] == "random_forest":
    final_model = RandomForestClassifier(n_estimators=best_params["n_estimators"],
                                         max_depth=best_params["max_depth"],
                                         min_samples_split=best_params["min_samples_split"],
                                         min_samples_leaf=best_params["min_samples_leaf"],
                                         random_state=42)
elif best_params["model"] == "xgboost":
    final_model = xgb.XGBClassifier(learning_rate=best_params["learning_rate"],
                                    n_estimators=best_params["n_estimators"],
                                    max_depth=best_params["max_depth"],
                                    use_label_encoder=False,
                                    eval_metric='logloss', random_state=42)
else:
    final_model = lgb.LGBMClassifier(learning_rate=best_params["learning_rate"],
                                     n_estimators=best_params["n_estimators"],
                                     num_leaves=best_params["num_leaves"],
                                     random_state=42)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_val)
print("Final Model Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# Predictions on test set
test_predictions = final_model.predict_proba(X_test)[:, 1]


# Create submission file
submission = pd.read_csv(sample_submission_path)
submission['rainfall'] = test_predictions
submission.to_csv('submission3.csv', index=False)

print("Submission file 'submission3.csv' created successfully!")




