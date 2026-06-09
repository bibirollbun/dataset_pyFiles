# ====== Imports ======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import optuna
import joblib

# ====== Load datasets ======
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

target = "loan_paid_back"

# ====== Encode categorical features ======
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ====== Split train data ======
X = train.drop(columns=[target])
y = train[target]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== Define Optuna objective ======
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "use_label_encoder": False,
        "tree_method": "hist",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 1, 5),
        "alpha": trial.suggest_float("alpha", 0, 5),
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "random_state": 42,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    preds = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    return auc

# ====== Run Optuna optimization ======
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best AUC:", study.best_value)
print("Best params:", study.best_params)

# ====== Train final model on full data ======
best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
    "tree_method": "hist",
    "random_state": 42
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X, y)

# ====== Save the model ======
joblib.dump(final_model, "xgb_optuna_model.pkl")

# ====== Predict on test set ======
test_preds = final_model.predict_proba(test)[:, 1]

# ====== Create submission file ======
sample["loan_paid_back"] = test_preds
sample.to_csv("submission.csv", index=False)

print("✅ Model saved as 'xgb_optuna_model.pkl'")
print("✅ Submission saved as 'submission.csv'")





