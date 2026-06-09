import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)

import warnings
warnings.filterwarnings('ignore')


SAMPLE_SUBMISSION = '/kaggle/input/playground-series-s5e12/sample_submission.csv'
sample_df = pd.read_csv(SAMPLE_SUBMISSION)
sample_df


sns.set()
plt.rcParams["figure.figsize"] = (8, 5)
RANDOM_STATE = 42


DATA_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TARGET_COL = "diagnosed_diabetes"


df = pd.read_csv(DATA_PATH, index_col='id')


print("Shape:", df.shape)
df.head()


df.info()


df.describe(include="all").T


print("Target unique values:", df[TARGET_COL].unique())
print(df[TARGET_COL].value_counts(normalize=True))

sns.countplot(x=TARGET_COL, data=df)
plt.title("Diagnosed diabetes distribution")
plt.show()


missing = df.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]

if not missing.empty:
    print(missing)
    missing.plot(kind="bar")
    plt.title("Fraction of missing values per column")
    plt.ylabel("Missing fraction")
    plt.show()
else:
    print("âœ… No missing values in the dataset.")


numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

if TARGET_COL in numeric_features:
    numeric_features.remove(TARGET_COL)
if TARGET_COL in categorical_features:
    categorical_features.remove(TARGET_COL)

print(f"({len(numeric_features)}) Numeric features:")
print("-"*22)
print(numeric_features)
print("")
print(f"({len(categorical_features)}) Categorical features:")
print("-"*22)
print(categorical_features)


n_num = len(numeric_features)
n_cols = 4
n_rows = int(np.ceil(n_num / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for ax, col in zip(axes, numeric_features):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"{col}")
    ax.set_xlabel("")
    ax.set_ylabel("Density")

for i in range(n_num, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


n_cat = len(categorical_features)
n_cols = 3
n_rows = int(np.ceil(n_cat / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for ax, col in zip(axes, categorical_features):
    df[col].value_counts(normalize=True).head(20).plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution of {col}")
    ax.set_ylabel("Frequency")

for i in range(len(categorical_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()



corr_matrix = df[numeric_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Correlation between numeric features")
plt.show()


target_corr = df[numeric_features + [TARGET_COL]].corr()[TARGET_COL].sort_values(ascending=False)
target_corr


plt.figure(figsize=(6, 10))
target_corr.drop(TARGET_COL).plot(kind="barh")
plt.title("Correlation of numeric features with diabetes")
plt.xlabel("Correlation")
plt.gca().invert_yaxis()
plt.show()


X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


log_reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE))
])


rf_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])



xgb_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss"
    ))
])



models = {
    "LogisticRegression": log_reg,
    "RandomForest": rf_clf,
    "XGBoost": xgb_clf
}


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    
    """Train model, compute metrics, and plot ROC curve."""
    
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)
    print("ROC AUC  :", auc)
    
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC curve: {name}")
    plt.show()
    
    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }


results = []
for name, model in models.items():
    res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results.append(res)

results_df = pd.DataFrame(results).set_index("model")
results_df


voting_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", VotingClassifier(
        estimators=[
            ("log_reg", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
            ("xgb", XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss"
            ))
        ],
        voting="soft"
    ))
])


voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)
y_proba = voting_clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("=== VotingClassifier Performance ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("ROC AUC  :", auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix: VotingClassifier")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve: VotingClassifier")
plt.show()


TEST_PATH = '/kaggle/input/playground-series-s5e12/test.csv'
test_df = pd.read_csv(TEST_PATH)


predictions = voting_clf.predict_proba(test_df)[:, 1]

output = pd.DataFrame({'id': test_df['id'], 'diagnosed_diabetes': predictions})
output.to_csv('submission.csv', index=False)
print("âœ… Submission file saved.")


print("----- Short summary of submission file -----")

total = output.shape[0]

threshold = 0.5
predicted_positive = (output["diagnosed_diabetes"] >= threshold)

total = len(output)
positive_count = predicted_positive.sum()
positive_pct = positive_count / total * 100

print(f"Total predictions: {total}")
print(f"Count above threshold {threshold}: {positive_count}")
print(f"Percentage above threshold: {positive_pct:.2f}%")
print()

