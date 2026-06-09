!pip install lightgbm category_encoders



!pip install matplotlib



!pip install optuna lightgbm category_encoders



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])

# 2. Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 0 = Introvert, 1 = Extrovert

# 3. Handle Categorical + Missing Data
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# 4. Define Base Models
base_models = [
    RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
    XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42),
    LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
    GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
]

# 5. Generate Meta-Features
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], len(base_models)))
test_preds = np.zeros((X_test.shape[0], len(base_models)))
cv_scores = []

for i, model in enumerate(base_models):
    print(f"Training base model {i+1}: {model.__class__.__name__}")
    test_fold_preds = np.zeros((X_test.shape[0], n_folds))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_encoded)):
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        oof_preds[val_idx, i] = model_clone.predict_proba(X_val)[:, 1]
        test_fold_preds[:, fold] = model_clone.predict_proba(X_test_processed)[:, 1]

        val_pred = model_clone.predict(X_val)
        fold_scores.append(accuracy_score(y_val, val_pred))

    test_preds[:, i] = test_fold_preds.mean(axis=1)
    cv_scores.append(np.mean(fold_scores))

# 6. Plot CV Scores
plt.figure(figsize=(8, 4))
plt.bar(
    [m.__class__.__name__ for m in base_models],
    cv_scores,
    color='skyblue'
)
plt.title("Cross-Validation Accuracy of Base Models")
plt.ylabel("Accuracy")
plt.ylim(0.95, 1.0)
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Optuna: Tune Meta Model (Logistic Regression)
def objective(trial):
    C = trial.suggest_loguniform("C", 1e-3, 10)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)

    scores = []
    for train_idx, val_idx in skf.split(oof_preds, y_encoded):
        model.fit(oof_preds[train_idx], y_encoded[train_idx])
        preds = model.predict(oof_preds[val_idx])
        scores.append(accuracy_score(y_encoded[val_idx], preds))
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("ðŸ”§ Best Optuna Params:", study.best_params)

# 8. Final Training on Full Data with Best Meta Model
best_params = study.best_params
solver = "liblinear" if best_params["penalty"] == "l1" else "lbfgs"
final_meta_model = LogisticRegression(**best_params, solver=solver, max_iter=1000)
final_meta_model.fit(oof_preds, y_encoded)
final_preds = final_meta_model.predict(test_preds)

# 9. Decode + Save Submission
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
print("âœ… Final submission saved as 'submission.csv'")


