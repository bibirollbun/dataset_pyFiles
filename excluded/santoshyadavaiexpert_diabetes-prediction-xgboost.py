import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


TARGET = "diagnosed_diabetes"
y = train[TARGET]
X = train.drop(columns=[TARGET])
X_test = test.copy()
print(X.shape, y.shape)


print(train.info())
print(train.isna().sum())
print(y.value_counts(normalize=True))


numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
if 'id' in numeric_features: numeric_features.remove('id')
print(numeric_features)
print(categorical_features)


preprocessor = ColumnTransformer([
    ("num","passthrough",numeric_features),
    ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)
])

model = XGBClassifier(
    n_estimators=500,max_depth=5,learning_rate=0.03,
    subsample=0.9,colsample_bytree=0.8,eval_metric="auc",
    tree_method="hist",random_state=42
)

clf = Pipeline([("preprocessor",preprocessor),("model",model)])


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_valid)[:,1]
print("Validation ROC-AUC:", roc_auc_score(y_valid, pred))


clf.fit(X, y)


clf_imp = Pipeline([("preprocessor",preprocessor),("model",model)])
clf_imp.fit(X_train, y_train)

result = permutation_importance(
    clf_imp, X_valid, y_valid, n_repeats=3, random_state=42, n_jobs=-1
)

importances = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

importances.head(20)


test_pred = clf.predict_proba(X_test)[:,1]
submission = pd.DataFrame({
    "id": X_test["id"],
    "diagnosed_diabetes": test_pred
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
submission.head()

