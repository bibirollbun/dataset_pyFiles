import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
test = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")


y = train["Cover_Type"]

X = train.drop(columns=["Id", "Cover_Type"])

X_test = test.drop(columns=["Id"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train subset:", X_train.shape, "Validation subset:", X_valid.shape)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

validation_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds_valid = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds_valid)
    validation_scores[name] = acc
    print(f"Accuracy на валидации: {acc:.5f}")

print("\nРезультаты на валидации:")
for name, acc in validation_scores.items():
    print(f"{name}: {acc:.5f}")


best_model_name = max(validation_scores, key=validation_scores.get)
best_model = models[best_model_name]

print(f"\nЛучшая модель по валидации: {best_model_name} (accuracy = {validation_scores[best_model_name]:.5f})")


best_model.fit(X, y)


test_preds = best_model.predict(X_test)

submission = pd.DataFrame({
    "Id": test["Id"],
    "Cover_Type": test_preds.astype(int)
})

submission.to_csv("submission.csv", index=False)


