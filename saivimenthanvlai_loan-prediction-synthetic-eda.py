import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


ID_COL = sub.columns[0]
TARGET = sub.columns[1] if sub.columns[1] in train.columns else (
    "loan_paid_back" if "loan_paid_back" in train.columns else
    "Loan_Status" if "Loan_Status" in train.columns else
    [c for c in train.columns if c not in test.columns][-1]
)


print({"train_shape": train.shape, "test_shape": test.shape, "id": ID_COL, "target": TARGET})


display(train.head(10))


display(test.head(10))


# Missing values
mv_train = train.isna().sum().sort_values(ascending=False)
mv_test  = test.isna().sum().sort_values(ascending=False)


display(mv_train[mv_train>0].head(25))


display(mv_test[mv_test>0].head(25))


# Target distribution
train[TARGET].value_counts(dropna=False).plot(kind="bar")
plt.title("Target distribution"); plt.tight_layout(); plt.show()


# Numeric overview
num_cols = train.drop(columns=[TARGET]).select_dtypes(include=[np.number]).columns.tolist()
desc = train[num_cols].describe().T.sort_values("std", ascending=False)
display(desc.head(25))


# Correlation (cap to 30 numeric cols for readability)
if len(num_cols) > 1:
    corr_cols = num_cols[:30]
    corr = train[corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(); plt.xticks(range(len(corr_cols)), corr_cols, rotation=90)
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Correlation heatmap (first 30 numeric)");
    plt.tight_layout();
    plt.show()


def feature_engineering(df):
    df = df.copy()
    df["MissingCount"] = df.isna().sum(axis=1)

    # Loan-style features (created only if columns exist)
    if {"ApplicantIncome","CoapplicantIncome"}.issubset(df.columns):
        df["TotalIncome"] = df["ApplicantIncome"].fillna(0) + df["CoapplicantIncome"].fillna(0)
        df["IncomeDiff"]  = df["ApplicantIncome"].fillna(0) - df["CoapplicantIncome"].fillna(0)
        denom = df["CoapplicantIncome"].replace(0, np.nan)
        df["IncomeRatio"] = (df["ApplicantIncome"] / denom).replace([np.inf, -np.inf], np.nan)

    if {"LoanAmount","Loan_Amount_Term"}.issubset(df.columns):
        denom_t = df["Loan_Amount_Term"].replace(0, np.nan)
        df["AmountPerTerm"] = (df["LoanAmount"] / denom_t).replace([np.inf, -np.inf], np.nan)

    if {"annual_income","loan_amount"}.issubset(df.columns):
        denom_l = df["annual_income"].replace(0, np.nan)
        df["LoanToIncome"] = (df["loan_amount"] / denom_l).replace([np.inf, -np.inf], np.nan)

    return df


X = train.drop(columns=[TARGET])
y = train[TARGET]
X = feature_engineering(X)
test_fe = feature_engineering(test)


# Encode target if strings
if y.dtype == object or str(y.dtype).startswith("category"):
    classes = sorted(y.dropna().unique())
    if set(classes) == {"N","Y"}: enc_map = {"N":0,"Y":1}
    else: enc_map = {c:i for i,c in enumerate(classes)}
    y = y.map(enc_map)
    inv_map = {v:k for k,v in enc_map.items()}
else:
    enc_map = None


import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score


# Columns
if ID_COL in X.columns:
    X = X.drop(columns=[ID_COL])
if ID_COL in test_fe.columns:
    test_fe = test_fe.drop(columns=[ID_COL])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

numeric = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler(with_mean=False))
])
categorical = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
pre = ColumnTransformer([
    ("num", numeric, num_cols),
    ("cat", categorical, cat_cols)
], remainder="drop", sparse_threshold=0.3)


# Define model candidates
ModelDefs = [
    ("LightGBM", lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )),
    ("XGBoost", XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )),
    ("HistGB", HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=None,
        max_leaf_nodes=31,
        random_state=42
    ))
]
is_binary = (y.nunique() == 2)
scoring = "roc_auc" if is_binary else "accuracy"
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in ModelDefs:
    pipe = Pipeline([("pre", pre), ("clf", model)])
    
    # Sample large datasets to 200k rows for faster CV
    if len(X) > 200_000:
        from sklearn.model_selection import train_test_split
        Xs, _, ys, _ = train_test_split(X, y, train_size=200_000, stratify=y, random_state=42)
    else:
        Xs, ys = X, y

    scores = cross_val_score(pipe, Xs, ys, cv=cv, scoring=scoring, n_jobs=-1)
    results.append((name, scores.mean(), scores.std(), scores))
    print(f"{name}: {scoring} {scores.mean():.5f} ± {scores.std():.5f} | {np.round(scores,5)}")


best = max(results, key=lambda t: t[1])
print({"selected_model": best[0], "cv_mean": best[1], "cv_std": best[2]})

best_name = best[0]
best_model = [m for m in ModelDefs if m[0] == best_name][0][1]

# Final fit on full data
final_pipe = Pipeline([("pre", pre), ("clf", best_model)])
final_pipe.fit(X, y)


# Predict class labels for Kaggle (most Playground tasks expect labels, not probabilities)
y_pred = final_pipe.predict(test_fe)

# Map back to original labels if target was strings
if enc_map is not None:
    y_pred = pd.Series(y_pred).map(inv_map).values

submission = pd.DataFrame({
    sub.columns[0]: test[ID_COL].values,
    sub.columns[1]: y_pred
})
submission.to_csv("submission.csv", index=False)
submission.head()


# Optional; skip if time is tight
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe = Pipeline([("pre", pre), ("clf", best_model)])
pipe.fit(X_tr, y_tr)

metric = roc_auc_score(y_val, pipe.predict_proba(X_val)[:,1]) if is_binary and hasattr(best_model,"predict_proba") else accuracy_score(y_val, pipe.predict(X_val))
print({"holdout_metric": metric, "metric_name": scoring})

# Permutation importances over preprocessed features
# Build feature name list matching ColumnTransformer order
feat_names = [f"(num){c}" for c in num_cols] + [f"(cat){c}" for c in cat_cols]
perm = permutation_importance(pipe, X_val, y_val, n_repeats=5, random_state=42, scoring=scoring, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)
imp.head(25)




