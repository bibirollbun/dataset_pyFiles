import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

train.head()


train.isna().sum()


sns.countplot(x=train['loan_paid_back'])
plt.title("Баланс классов")
plt.show()

print(train['loan_paid_back'].value_counts(normalize=True))


num_columns = train.select_dtypes(include=['int64','float64']).columns.drop(["loan_paid_back","id"])

train[num_columns].hist(bins=30, figsize=(14,10))
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(train[num_columns].corr(), cmap="coolwarm")
plt.title("Корреляции числовых признаков")
plt.show()


low_var = [col for col in num_columns if train[col].nunique() < 2]
print("Удаляем:", low_var)
train = train.drop(columns=low_var)
test = test.drop(columns=low_var)


cat_cols = train.select_dtypes(include=["object"]).columns

for col in cat_cols:
    freqs = train[col].value_counts(normalize=True)
    rare = freqs[freqs < 0.01].index
    train[col] = train[col].replace(rare, "Other")
    test[col] = test[col].replace(rare, "Other")


X = train.drop(columns=["loan_paid_back", "id"])
y = train["loan_paid_back"]
test_ids = test["id"]
X_test = test.drop(columns=["id"])


num_features = X.select_dtypes(include=["int64","float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])


model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[("preprocess", preprocessor),
                      ("model", model)])


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)


preds = pipe.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, preds)
auc


pipe.fit(X, y)


test_pred = pipe.predict_proba(X_test)[:,1]


sub = pd.DataFrame({
    "id": test_ids,
    "loan_paid_back": test_pred
})

sub.to_csv("submission.csv", index=False)
sub.head()

