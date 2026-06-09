import pandas as pd
import numpy as np

DATA_DIR = "/kaggle/input/playground-series-s3e24"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

ID_COL = "id"
TARGET = "smoking"

print("Shapes:", train.shape, test.shape)
print("\nDtypes:\n", train.dtypes)

if TARGET in train.columns:
    print("\nClass counts:\n", train[TARGET].value_counts().sort_index())
    print("\nClass ratio:\n", train[TARGET].value_counts(normalize=True).sort_index())

na_train = train.isna().sum().sort_values(ascending=False)
na_test  = test.isna().sum().sort_values(ascending=False)
print("\nTop NA (train):\n", na_train.head(10))
print("\nTop NA (test):\n", na_test.head(10))

display(train.describe().T.loc[["age","height(cm)","weight(kg)","waist(cm)","Cholesterol","HDL","LDL","triglyceride","Gtp"]])



def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI"] = df["weight(kg)"] / (df["height(cm)"]/100.0)**2

    if {"LDL","HDL"}.issubset(df.columns):
        df["LDL_to_HDL"] = df["LDL"] / df["HDL"].replace(0, np.nan)
    if {"Cholesterol","HDL"}.issubset(df.columns):
        df["Chol_to_HDL"] = df["Cholesterol"] / df["HDL"].replace(0, np.nan)
    if {"waist(cm)","height(cm)"}.issubset(df.columns):
        df["Waist_to_Height"] = df["waist(cm)"] / df["height(cm)"]
    if {"BMI","age"}.issubset(df.columns):
        df["BMI_x_Age"] = df["BMI"] * df["age"]

    clip_cols = [
        "triglyceride","Gtp","ALT","AST","LDL","HDL","Cholesterol",
        "serum creatinine","systolic","relaxation","waist(cm)",
        "hemoglobin","BMI"
    ]
    for c in clip_cols:
        if c in df.columns:
            lo, hi = df[c].quantile([0.005, 0.995])
            df[c] = df[c].clip(lo, hi)

    for c in ["triglyceride","Gtp","ALT","AST"]:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    return df

train_p = add_basic_features(train)
test_p  = add_basic_features(test)

FEATURES = [c for c in train_p.columns if c not in [ID_COL, TARGET]]
X = train_p[FEATURES]
y = train_p[TARGET].astype(int)
X_test = test_p[FEATURES]

print("Prepared shapes:", X.shape, X_test.shape)
print("Any NA in X?", X.isna().sum().sum(), " | Any NA in X_test?", X_test.isna().sum().sum())



import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

preprocess = ColumnTransformer(
    transformers=[("num", SimpleImputer(strategy="median"), FEATURES)],
    remainder="drop"
)

hgb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=700,
    max_depth=None,
    l2_regularization=0.1,
    max_bins=255,
    early_stopping=True,
    random_state=42
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", hgb)
])

classes = np.sort(y.unique())
n_classes = len(classes)
print("Detected classes:", classes)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X), n_classes)) if n_classes > 2 else np.zeros(len(X))
test_pred = np.zeros((len(X_test), n_classes)) if n_classes > 2 else np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    pipe.fit(X_tr, y_tr)
    proba_va = pipe.predict_proba(X_va)
    proba_te = pipe.predict_proba(X_test)

    if n_classes == 2:
        oof[va_idx]   = proba_va[:, 1]               
        test_pred    += proba_te[:, 1] / skf.n_splits
    else:
        oof[va_idx,:] = proba_va
        test_pred     += proba_te / skf.n_splits


if n_classes == 2:
    cv_auc = roc_auc_score(y, oof)
    print(f"[CV] ROC-AUC: {cv_auc:.5f}")
else:
    cv_f1 = f1_score(y, oof.argmax(axis=1), average="macro")
    print(f"[CV] Macro-F1: {cv_f1:.5f}")



from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc as sk_auc
import matplotlib.pyplot as plt

if n_classes == 2:
    oof_prob = oof if oof.ndim == 1 else oof.astype(float)
    preds = (oof_prob >= 0.5).astype(int)
else:
    preds = oof.argmax(axis=1)

cm = confusion_matrix(y, preds, labels=classes)
print("Confusion Matrix:\n", cm)
print(classification_report(y, preds, labels=classes, digits=4))

if n_classes == 2:
    fpr, tpr, _ = roc_curve(y, oof_prob)
    print("OOF AUC:", sk_auc(fpr, tpr))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC (OOF)"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.show()



if n_classes == 2:
    submission = pd.DataFrame({
        "id": test[ID_COL].values,
        "smoking": test_pred  
    })
else:
    test_labels = test_pred.argmax(axis=1)
    submission = pd.DataFrame({
        "id": test[ID_COL].values,
        "smoking": test_labels
    })

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Saved to:", submission_path)
display(submission.head())


