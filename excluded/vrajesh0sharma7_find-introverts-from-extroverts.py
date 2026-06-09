import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DATA_DIR = Path("/kaggle/input/playground-series-s5e7")

train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

print("Train shape :", train.shape)
print("Test shape  :", test.shape)
train.head()


train.info()


# Target distribution
ax = sns.countplot(x="Personality", data=train, palette="pastel")
ax.set_title("Target Distribution")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", (p.get_x()+0.25, p.get_height()+150))
plt.show()


# Missing-value heatmap
sns.heatmap(train.isna(), cbar=False, yticklabels=False)
plt.title("Missing-Value Pattern"); plt.show()


# Numeric distributions
train.select_dtypes("number").hist(bins=25, figsize=(12,8)); plt.tight_layout()


target = "Personality"
y = train[target].map({"Introvert":0, "Extrovert":1})

X_full   = train.drop(columns=[target])
test_ids = test["id"]
X_test   = test.copy()

num_cols = X_full.select_dtypes(include=["int64", "float64"]).columns.drop("id")
cat_cols = ["Stage_fear", "Drained_after_socializing"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler" , StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe"    , OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])


X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42
)

print("Train:", X_train.shape, " Valid:", X_valid.shape)


logreg_clf = Pipeline([
    ("prep", preprocess),
    ("clf" , LogisticRegression(max_iter=300, class_weight="balanced"))
])

logreg_clf.fit(X_train, y_train)
log_pred = logreg_clf.predict(X_valid)
log_proba = logreg_clf.predict_proba(X_valid)[:,1]

print("Logistic Regression Metrics")
print(classification_report(y_valid, log_pred))
print("ROC-AUC:", roc_auc_score(y_valid, log_proba).round(4))


rf_clf = Pipeline([
    ("prep", preprocess),
    ("clf" , RandomForestClassifier(
        n_estimators=400,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    ))
])

rf_clf.fit(X_train, y_train)
rf_pred  = rf_clf.predict(X_valid)
rf_proba = rf_clf.predict_proba(X_valid)[:,1]

print("Random Forest Metrics")
print(classification_report(y_valid, rf_pred))
print("ROC-AUC:", roc_auc_score(y_valid, rf_proba).round(4))


best_model = rf_clf
best_model.fit(X_full, y)

test_proba = best_model.predict_proba(X_test)[:,1]
test_pred  = (test_proba >= 0.5).astype(int)        # threshold = 0.5
submission = pd.DataFrame({
    "id"         : test_ids,
    "Personality": np.where(test_pred==0, "Introvert", "Extrovert")
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv saved.")


importances = best_model.named_steps["clf"].feature_importances_
feature_names = (list(num_cols) +
                 list(best_model.named_steps["prep"]
                      .named_transformers_["cat"]
                      .named_steps["ohe"]
                      .get_feature_names_out(cat_cols)))
imp_df = (pd.DataFrame({"feature":feature_names, "importance":importances})
          .sort_values("importance", ascending=False).head(10))
sns.barplot(y="feature", x="importance", data=imp_df, palette="viridis")
plt.title("Top-10 Important Features"); plt.show()


manual_record = {
    "Time_spent_Alone"          : 6.5,       # hours per day (float)
    "Stage_fear"                : "Yes",     # "Yes" / "No"
    "Social_event_attendance"   : 2.0,       # events per week (float)
    "Going_outside"             : 2.0,       # times per week (float)
    "Drained_after_socializing" : "Yes",     # "Yes" / "No"
    "Friends_circle_size"       : 6.0,       # number of close friends (float / int)
    "Post_frequency"            : 10.0       # social-media posts per month (float)
}


manual_df = pd.DataFrame([manual_record])
manual_df


pred_code  = best_model.predict(manual_df)[0]              # 0 = Introvert, 1 = Extrovert
pred_prob  = best_model.predict_proba(manual_df)[0, 1]     # probability of class "1" (Extrovert)


pred_label = "Extrovert" if pred_code == 1 else "Introvert"
print(f"Predicted Personality : {pred_label}")
print(f"P(Extrovert)          : {pred_prob:.2%}")




