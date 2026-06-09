import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import optuna


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# Ensure correct columns
expected_columns = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside',
                    'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency', 'Personality']
train = train[expected_columns]
print(train.columns.tolist())


# Encode target
le = LabelEncoder()
train["Personality"] = le.fit_transform(train["Personality"])  # Introvert=1, Extrovert=0

X = train.drop("Personality", axis=1)
y = train["Personality"]
X_test = test.drop(columns=["id"])


# Fix object dtype to numerical for XGBoost compatibility
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].map({'Yes': 1, 'No': 0})

for col in X_test.columns:
    if X_test[col].dtype == 'object':
        X_test[col] = X_test[col].map({'Yes': 1, 'No': 0})


# Optional: Tune XGBoost with Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    return score


# Uncomment to run tuning
#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=60)
#print("Best score:", study.best_value)
#print("Best params:", study.best_params)


best_xgb_params = {
    "n_estimators": 471,
    "max_depth": 6,
    "learning_rate": 0.09140436683702755,
    "subsample": 0.6056023401560551,
    "colsample_bytree": 0.56362882753098,
    "gamma": 0.55931531964274,
    "reg_alpha": 4.757232839625068,
    "reg_lambda": 2.354706492408625,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "logloss"
}


# 0.975708
#best_xgb_params = {
#    "n_estimators": 329,
#    "max_depth": 7,
#    "learning_rate": 0.0639140693373098,
#    "subsample": 0.896861911436881,
#    "colsample_bytree": 0.5032667667913882,
#    "gamma": 1.0422164128337141,
#    "reg_alpha": 1.9439865569192492,
#    "reg_lambda": 0.09548527461694079,
#    "random_state": 42,
#    "use_label_encoder": False,
#    "eval_metric": "logloss"
#}


# Define base models
lgb_model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
cat_model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42)
xgb_model = XGBClassifier(**best_xgb_params)


# Ensemble model
stacking_model = StackingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('xgb', xgb_model)
    ],
    final_estimator=CatBoostClassifier(iterations=150, learning_rate=0.05, depth=4, verbose=0, random_state=42),
    cv=5,
    n_jobs=-1,
    passthrough=True
)


# Train model
stacking_model.fit(X, y)

# Predict
final_preds = stacking_model.predict(X_test)


# Final submission
submission = pd.DataFrame({
    "id": test["id"],
    "Personality": ["Extrovert" if p == 0 else "Introvert" for p in final_preds]
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission saved as 'submission.csv'")




