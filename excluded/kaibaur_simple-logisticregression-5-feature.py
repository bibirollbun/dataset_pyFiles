# ============================================
# load library
# ============================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================
# load data
# ============================================

# load train data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


df_train.head()


# ============================================
# EDA
# ============================================

from ydata_profiling import ProfileReport
report = ProfileReport(df_train, title='diabetespredict')
report.to_notebook_iframe()


# ============================================
# feature: bmi, ldl_cholesterol, physical_activity_minutes_per_week, systolic_bp
# label  : diagnosed_diabetes（0/1）
# ============================================
X = df_train[["age", "bmi", "ldl_cholesterol", "physical_activity_minutes_per_week", "systolic_bp"]]
y = df_train["diagnosed_diabetes"]

# drop
data = pd.concat([X, y], axis=1).dropna()


# ============================================
# split data
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================
# LogisticRegression
# Pipeline  : diagnosed_diabetes（0/1）
# ============================================
pipe = Pipeline([
    ("standerd_scaler", StandardScaler()),
    ("Classifier", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)


# ============================================
# predict and evaluate
# ============================================
y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print()
print("Classification report:")
print(classification_report(y_test, y_pred))


# load test data
df_test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X_sub    = df_test[["age", "bmi", "ldl_cholesterol", "physical_activity_minutes_per_week", "systolic_bp"]]
y_sub_proba = pipe.predict_proba(X_sub)[:, 1]

# make submission.csv
submission = pd.DataFrame({
    "id": df_test["id"],
    "loan_paid_back": y_sub_proba
})

submission.to_csv("submission.csv", index=False)
submission.head()

