# !pip -q install /kaggle/input/pytorchtabnet/pytorch_tabnet-4.1.0-py3-none-any.whl


# !pip install category_encoders


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
import category_encoders as ce  # TargetEncoder

# Chargement des donnÃ©es
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

df_1 = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction1.csv")
df_2 = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction2.csv")

df_train = pd.concat([df_train, df_1, df_2], ignore_index=True)

# DÃ©finir les colonnes
FEATURES = [c for c in df_train.columns if c not in ['Fertilizer Name', 'id']]
CAT_COLS = FEATURES
LABEL = "Fertilizer Name"

# Encodage de la cible
label_enc = LabelEncoder()
y = label_enc.fit_transform(df_train[LABEL])

X = df_train[CAT_COLS]
X_test = df_test[CAT_COLS]

# Fonction MAP@3
def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    return int(actual in predicted) / (predicted.index(actual)+1) if actual in predicted else 0

def map3(actuals, preds, k=3):
    return np.mean([apk(a, list(p), k) for a, p in zip(actuals, preds)])

# ğŸ“ˆ Fonction de data augmentation simple (duplication + bruit lÃ©ger)
def augment_data(X, y, n_copies=1, noise_level=0.01):
    X_aug = X.copy()
    y_aug = y.copy()
    for _ in range(n_copies):
        X_dup = X.copy()
        for col in X_dup.select_dtypes(include=[np.number]).columns:
            noise = np.random.normal(0, noise_level, size=X_dup.shape[0])
            X_dup[col] = X_dup[col] + noise
        X_aug = pd.concat([X_aug, X_dup], ignore_index=True)
        y_aug = np.concatenate([y_aug, y], axis=0)
    return X_aug, y_aug

# Cross-validation Stratified K-Fold avec Target Encoding
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”� Fold {fold+1}/{N_FOLDS}")

    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y[train_idx], y[val_idx]

    # âš¡ï¸� Data augmentation uniquement sur les donnÃ©es d'entraÃ®nement
    X_train, y_train = augment_data(X_train, y_train, n_copies=1, noise_level=0.01)

    # Target Encoding
    te = ce.TargetEncoder(cols=CAT_COLS, smoothing=0.5)
    X_train[CAT_COLS] = te.fit_transform(X_train[CAT_COLS], y_train)
    X_val[CAT_COLS] = te.transform(X_val[CAT_COLS])
    X_test_enc = X_test.copy()
    X_test_enc[CAT_COLS] = te.transform(X_test[CAT_COLS])

    # ModÃ¨le
    clf = XGBClassifier(
        random_state=42,
        tree_method='gpu_hist',
        n_estimators=4000,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=3,
        gamma=0.2,
        colsample_bytree=0.8,
        subsample=0.85,
        reg_alpha=1,
        reg_lambda=1.5,
        objective='multi:softprob',
        num_class=len(label_enc.classes_),
        eval_metric='mlogloss',
        n_jobs=-1,
        verbosity=1
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=150,
        verbose=200
    )

    oof_preds[val_idx] = clf.predict_proba(X_val)
    test_preds += clf.predict_proba(X_test_enc) / N_FOLDS

    top3_val = np.argsort(oof_preds[val_idx], axis=1)[:, -3:][:, ::-1]
    fold_map3 = map3(y_val, top3_val)
    fold_scores.append(fold_map3)
    print(f"ğŸ“Š Fold {fold+1} MAP@3 score: {fold_map3:.5f}")

# RÃ©sultat global OOF
top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
map3_score = map3(y, top3_oof)

print("\nğŸ“ˆ RÃ©sumÃ© des scores par fold:")
for i, score in enumerate(fold_scores):
    print(f" - Fold {i+1}: {score:.5f}")
print(f"\nğŸ“Š Global OOF MAP@3 score (Target Encoding + Augmentation): {map3_score:.5f}")

# Soumission
top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top3_test_labels = label_enc.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)
sub[LABEL] = [' '.join(row) for row in top3_test_labels]
sub.to_csv("submission.csv", index=False)

print("âœ… Submission file saved as submission.csv")



print(sub.head())

