from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import torch


"""PREDICTING OPTIMAL FERTILIZER"""
# 0. PREPROCESSING
duenger = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")

le = LabelEncoder()
y = le.fit_transform(duenger["Fertilizer Name"])
X = duenger.drop(["id", "Fertilizer Name"], axis=1)

cat_features = ["Soil Type", "Crop Type"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# 1. FITTING
# Parameter
initial_lr = 0.1
lr_decay = 0.5
n_phases = 8
iterations_per_phase = 500
depth = 7

current_lr = initial_lr
previous_model = None

for phase in range(n_phases):
    print(f"--- Training Phase {phase+1} | Learning Rate: {current_lr} ---")
    
    model = CatBoostClassifier(
        iterations=iterations_per_phase,
        learning_rate=current_lr,
        depth=depth,
        eval_metric='MultiClass',
        cat_features=cat_features,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        init_model=previous_model  # Nur beim ersten Phase ist das None
    )
    
    previous_model = model  # speichert das Modell für nächste Phase
    current_lr *= lr_decay  # reduziere LR für nächste Phase

print("Training abgeschlossen.")



# 2. PREDICTING
preds_proba = model.predict_proba(X_val)
top3_preds_idx = np.argsort(preds_proba, axis=1)[:, -3:][:, ::-1]
top3_preds_labels = le.inverse_transform(top3_preds_idx.flatten()).reshape(top3_preds_idx.shape)

print(top3_preds_labels[:5]) 



# 3. MAP@3 SCORE
def apk(actual, predicted, k=3):
    """
    Average Precision at k.
    actual: int (true class)
    predicted: list of predicted classes (ordered)
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            break  # since true label found, stop

    return score

def mapk(actuals, predicted_lists, k=3):
    """
    Mean Average Precision at k.
    actuals: list or array of true labels
    predicted_lists: list of lists of predicted labels
    """
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicted_lists)])

# Beispiel wie du es mit deinen Daten aufrufst:
# y_val: true labels (encoded integer)
# top3_preds_idx: vorhergesagte Klassenindices, shape (n_samples, 3)

top3_preds_labels = le.inverse_transform(top3_preds_idx.flatten()).reshape(top3_preds_idx.shape)
y_val_labels = le.inverse_transform(y_val)

score = mapk(y_val_labels, top3_preds_labels, k=3)
print(f"MAP@3 Score: {score:.4f}")



# 4. SAVING AND UPLOADING
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test_ids = test_df["id"]

X_test_final = test_df.drop("id", axis=1)
preds_proba_test = model.predict_proba(X_test_final)

top3_idx = np.argsort(preds_proba_test, axis=1)[:, -3:][:, ::-1]
top3_labels = le.inverse_transform(top3_idx.flatten()).reshape(top3_idx.shape)

submission = pd.DataFrame({
    "id": test_ids,
    "Fertilizer Name": [' '.join(row) for row in top3_labels]
})

# 6. Speichern
submission.to_csv("submission.csv", index=False)
print("✅ Fertig: submission.csv gespeichert.")


