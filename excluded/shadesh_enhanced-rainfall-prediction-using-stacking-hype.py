# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna  # For hyperparameter tuning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import warnings

warnings.filterwarnings("ignore")

# Load datasets
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)


# Display dataset info
print("\nðŸ”¹ Train Data Shape:", train_df.shape)
print("\nðŸ”¹ Test Data Shape:", test_df.shape)
print("\nðŸ”¹ Train Data Preview:\n", train_df.head())

# Check for missing values
print("\nðŸ”¹ Missing Values:\n", train_df.isnull().sum())

# Check data types
print("\nðŸ”¹ Data Types:\n", train_df.dtypes)


# Drop ID column and separate target
X = train_df.drop(columns=["id", "rainfall"])  
y = train_df["rainfall"]  

# Encode categorical variables
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    if col in test_df.columns:
        test_df[col] = le.transform(test_df[col])



# Standardize numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)
test_df[X.columns] = scaler.transform(test_df.drop(columns=["id"]))

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Define models
rf_model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
cat_model = cat.CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)
log_model = LogisticRegression()


# Train models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)


# Validate models
rf_pred = rf_model.predict_proba(X_val)[:, 1]
xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
cat_pred = cat_model.predict_proba(X_val)[:, 1]


# Evaluate models
rf_score = roc_auc_score(y_val, rf_pred)
xgb_score = roc_auc_score(y_val, xgb_pred)
lgb_score = roc_auc_score(y_val, lgb_pred)
cat_score = roc_auc_score(y_val, cat_pred)


print(f"\nðŸ”¹ Random Forest AUC: {rf_score:.4f}")
print(f"\nðŸ”¹ XGBoost AUC: {xgb_score:.4f}")
print(f"\nðŸ”¹ LightGBM AUC: {lgb_score:.4f}")
print(f"\nðŸ”¹ CatBoost AUC: {cat_score:.4f}")


# Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[
        ("rf", rf_model),
        ("xgb", xgb_model),
        ("lgb", lgb_model),
        ("cat", cat_model)
    ],
    final_estimator=LogisticRegression()
)

stacking_model.fit(X_train, y_train)
stacking_pred = stacking_model.predict_proba(X_val)[:, 1]
stacking_score = roc_auc_score(y_val, stacking_pred)
print(f"\nðŸ”¹ Stacking Model AUC: {stacking_score:.4f}")


# Hyperparameter Tuning with Optuna for XGBoost
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0)
    }
    
    xgb_model_tuned = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    score = cross_val_score(xgb_model_tuned, X_train, y_train, cv=3, scoring="roc_auc").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("\nðŸ”¹ Best Parameters for XGBoost:", best_params)


# Train final XGBoost model with tuned hyperparameters
xgb_tuned = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_tuned.fit(X, y)
test_preds = xgb_tuned.predict_proba(test_df.drop(columns=["id"]))[:, 1]



# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_preds})
submission.to_csv("submission.csv", index=False)
print("\nâœ… Submission file saved!")

