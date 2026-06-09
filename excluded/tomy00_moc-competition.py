import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from timeit import default_timer



# 0. + 1. PREPROCESSING DATA + BUILDING
depression = pd.read_csv("/kaggle/input/tomy-s-moc-competition/moc-competition-mental-health/cleaned_train.csv")
depression = depression.drop(columns=["id", "Name"], errors="ignore")

y = depression["Depression"].astype(float)
X = depression.drop(columns=["Depression"])
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for col in X_train.columns:
    if X_train[col].astype(str).str.contains("More than 8 hours").any():
        print(f"Spalte mit Problem: {col}")

cat_features = [
    "Gender", "City", "Working Professional or Student", "Profession",
    "Dietary Habits", "Degree", 
    "Have you ever had suicidal thoughts ?", "Family History of Mental Illness",
    "Sleep Duration"]




# 2. FITTING 
common_params = {
    "cat_features": cat_features,
    "depth": 6,
    "random_state": 42,
    "verbose": 100
}

# Lernratenplan: Liste aus (learning_rate, iterations)
schedule = [
    (0.1, 2000),   # Phase 1: schnelles Lernen
    (0.05, 3000),  # Phase 2: verlangsamen
    (0.01, 5000)   # Phase 3: feintunen
]

model = None
prev_model = None
for i, (lr, iters) in enumerate(schedule):
    print(f"\nðŸš€ Phase {i+1}: lr={lr}, iterations={iters}")
    
    model = CatBoostClassifier(
        **common_params,
        learning_rate=lr,
        iterations=iters
    )
    
    model.fit(
        X_train, y_train,
        init_model=prev_model  # beim ersten Lauf ist prev_model None
    )
    
    prev_model = model  # speichere das trainierte Modell fÃ¼r nÃ¤chste Phase



# 3. PREDICTING
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {acc:.4f}")





test = pd.read_csv("/kaggle/input/tomy-s-moc-competition/moc-competition-mental-health/cleaned_test.csv")

# Falls nÃ¶tig: gleiche Vorverarbeitung wie bei train.csv (z.B. NaNs ersetzen)
# test.fillna("unknown", inplace=True)
# Falls du kategorische Features hast, ggf. als Strings lassen (CatBoost erkennt automatisch)

# id speichern fÃ¼r Submission
ids = test["id"]
test_processed = test.drop(columns=["id", "Name"], errors='ignore')


predictions = model.predict(test_processed)
submission = pd.DataFrame({
    "id": ids,
    "Depression": predictions.astype(int)  # sicherheitshalber als int (0/1)
})

# Als CSV speichern
submission.to_csv("submission.csv", index=False)

print("Submission-Datei wurde gespeichert.")


