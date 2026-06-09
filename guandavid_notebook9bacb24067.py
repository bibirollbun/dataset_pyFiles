import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay, log_loss
from xgboost import XGBClassifier


sample_submission = pd.read_csv("/kaggle/input/fall25-ece460j-kaggle-competition/sample_submission.csv")
train = pd.read_csv("/kaggle/input/fall25-ece460j-kaggle-competition/train.csv")
test = pd.read_csv("/kaggle/input/fall25-ece460j-kaggle-competition/test.csv")

train.head()


y = train["smoking"].astype(bool)
X = train.drop(columns=["smoking", "id"], )#errors="ignore")
X_test = test.drop(columns=["id"], )#errors="ignore")

y.head()


# feature engineering
def add_features(df):
    df = df.copy()
    if "height(cm)" in df.columns and "weight(kg)" in df.columns:
        h_m = df["height(cm)"] / 100.0 # height in meters for bmi
        df["BMI"] = df["weight(kg)"] / (h_m**2 + 0.000001)
        
    if "systolic" in df.columns and "relaxation" in df.columns:
        df["pulse_pressure"] = df["systolic"] - df["relaxation"]
        
    if set(["LDL", "HDL"]).issubset(df.columns):
        df["ldl_hdl_ratio"] = df["LDL"] / (df["HDL"] + 0.000001)
        
    if set(["Cholesterol", "HDL"]).issubset(df.columns):
        df["chol_hdl_ratio"] = df["Cholesterol"] / (df["HDL"] + 0.000001)
        
    if set(["AST", "ALT"]).issubset(df.columns): # AST/ALT ratio
        df["AST_ALT_ratio"] = df["AST"] / (df["ALT"] + 0.000001)

    if set(["eyesight(left)", "eyesight(right)"]).issubset(df.columns):
        df["eyesight_mean"] = df[["eyesight(left)", "eyesight(right)"]].mean(axis=1)

    if set(["hearing(left)", "hearing(right)"]).issubset(df.columns):
        df["hearing_sum"] = df[["hearing(left)", "hearing(right)"]].sum(axis=1)
        df["hearing_any_loss"] = (df["hearing_sum"] > 0).astype(int)

    for c in ["fasting blood sugar", "triglyceride", "Gtp", "ALT", "AST"]:
        if c in df.columns:
            df[f"log1p_{c.replace(' ', '_')}"] = np.log1p(df[c].clip(lower=0))

    if set(["waist(cm)", "height(cm)"]).issubset(df.columns):
        df["waist_height_ratio"] = df["waist(cm)"] / (df["height(cm)"] + 0.000001)

    if "age" in df.columns:
        df["age_BMI"] = df["age"] * df["BMI"]
        df["age_systolic"] = df["age"] * df["systolic"]
        df["age_chol"] = df["age"] * df["Cholesterol"]

    if set(["pulse_pressure", "relaxation"]).issubset(df.columns):
        df["pulse_relax_ratio"] = df["pulse_pressure"] / (df["relaxation"] + 1e-6)

    return df

X = add_features(X)
X_test = add_features(X_test)


num_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()

cat_cols = [c for c in num_cols_all if set(X[c].dropna().unique()).issubset({0, 1}) or set(X[c].dropna().unique()).issubset({1, 2})]

num_cols = [c for c in num_cols_all if c not in cat_cols]

print(num_cols)
print(cat_cols)


num_tf = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    #("scale", StandardScaler())
])
cat_tf = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocess = ColumnTransformer([
    ("num", num_tf, num_cols),
    ("cat", cat_tf, cat_cols)
])


xgb_params = dict(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    reg_alpha=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)
xgb = XGBClassifier(**xgb_params)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X))
aucs, accs, best_rounds = [], [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr = preprocess.fit_transform(X.iloc[tr_idx], y.iloc[tr_idx])
    X_va = preprocess.transform(X.iloc[va_idx])
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    model = XGBClassifier(**xgb_params, early_stopping_rounds = 100)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    
    proba = model.predict_proba(X_va)[:, 1]
    oof_pred[va_idx] = proba

    aucs.append(roc_auc_score(y_va, proba))
    accs.append(accuracy_score(y_va, (proba >= 0.5).astype(int)))
    best_rounds.append(model.best_iteration + 1)

print(f"OOF AUC     : {np.mean(aucs):.4f}")
print(f"OOF LogLoss : {log_loss(y, oof_pred):.5f}")
print(f"OOF Accuracy: {np.mean(accs):.4f}")
print(f"Avg best rounds: {int(np.mean(best_rounds))}")


final_n = int(np.mean(best_rounds))

X_all = preprocess.fit_transform(X, y)
X_te  = preprocess.transform(X_test)

final_params = {**xgb_params, "n_estimators": final_n}

final_model = XGBClassifier(**final_params)
final_model.fit(X_all, y, verbose=False)

test_proba = final_model.predict_proba(X_te)[:, 1]

sub = sample_submission

sub["smoking"] = test_proba
sub.to_csv("submission.csv", index=False)
print("finished")

