!pip -q install /kaggle/input/pytorchtabnet/pytorch_tabnet-4.1.0-py3-none-any.whl


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# ğŸ“¥ Lecture des donnÃ©es
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv").drop(columns=["id"])
df_train_1 = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction1.csv")
df_train_2 = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction2.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

df_train = pd.concat([df_train, df_train_1, df_train_2], ignore_index=True)

CAT_COLS = ["Soil Type", "Crop Type"]
NUM_COLS = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
FEATURES = NUM_COLS + CAT_COLS
LABEL = "Fertilizer Name"

# Imputation numÃ©rique
imputer = SimpleImputer(strategy="median")
df_train[NUM_COLS] = imputer.fit_transform(df_train[NUM_COLS])
df_test[NUM_COLS]  = imputer.transform(df_test[NUM_COLS])

# Encodage catÃ©goriel
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col]  = le.transform(df_test[col])
    label_encoders[col] = le

# Encodage label cible
label_enc = LabelEncoder()
y = label_enc.fit_transform(df_train[LABEL])
X = df_train[FEATURES].values
X_test = df_test[FEATURES].values

# Fonction d'Ã©valuation MAP@3
def fast_mapk(y_true, y_pred, k=3):
    score = 0
    for actual, pred in zip(y_true, y_pred):
        if actual in pred:
            score += 1 / (pred.tolist().index(actual) + 1)
    return score / len(y_true)

# Stratified Kâ€‘Fold + TabNet
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”� Fold {fold + 1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = TabNetClassifier(
        n_d=8, n_a=8, n_steps=3,
        gamma=1.5, lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax"
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["logloss"],
        max_epochs=10,
        patience=5,
        batch_size=256,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
    )

    oof_preds[val_idx] = clf.predict_proba(X_val)
    test_preds       += clf.predict_proba(X_test) / skf.n_splits

    top3_val = np.argsort(oof_preds[val_idx], axis=1)[:, -3:][:, ::-1]
    score = fast_mapk(y_val, top3_val)
    fold_scores.append(score)
    print(f"ğŸ“Š Fold {fold + 1} MAP@3 score: {score:.5f}")

# ğŸŒŸ RÃ©sumÃ©
top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
map3_score = fast_mapk(y, top3_oof)

print("\nğŸ“ˆ Scores par fold:")
for i, score in enumerate(fold_scores):
    print(f" - Fold {i+1}: {score:.5f}")
print(f"\nğŸ“Š OOF MAP@3 (final): {map3_score:.5f}")

# GÃ©nÃ©ration du fichier de soumission
top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
label_map = dict(enumerate(label_enc.classes_))
top3_test_labels = np.vectorize(label_map.get)(top3_test)

sub[LABEL] = [' '.join(row) for row in top3_test_labels]
sub.to_csv("submission_tabnet.csv", index=False)
print("âœ… Submission (TabNet) saved as submission_tabnet.csv")


