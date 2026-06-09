# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


df_train.head()


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)


df_train.isna().sum()


target = "loan_paid_back"
id_col = "id"



plt.figure()
sns.countplot(x=target, data=df_train)
plt.title("Target Distribution: loan_paid_back")
plt.show()


print(df_train[target].value_counts(normalize=True))


numeric_cols = ["annual_income", "debt_to_income_ratio", "credit_score",
                "loan_amount", "interest_rate"]
numeric_cols = [col for col in numeric_cols if col in df_train.columns]


for col in numeric_cols:
    plt.figure()
    sns.histplot(data=df_train, x=col, hue=target, kde=True, element="step")
    plt.title(f'{col} distribution by loan_paid_back')
    plt.show()


num_features_all = df_train.select_dtypes(include=['int64','float64']).drop(columns=[id_col], errors="ignore")
plt.figure(figsize=(10,6))
sns.heatmap(num_features_all.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()


X = df_train.drop(columns=[target, id_col])
y = df_train[target]


X_test_final = df_test.drop(columns=[id_col])


numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Numerical features" , numeric_features)
print("Categorical features: ", categorical_features)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Train size:", X_train.shape, "Valid size:", X_valid.shape)



# one - hot encode categorical columns
X_train = pd.get_dummies(X_train, drop_first=True)
X_valid = pd.get_dummies(X_valid, drop_first=True)



X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)



model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_valid_proba = model.predict_proba(X_valid)[:,1]


from sklearn.metrics import roc_auc_score, roc_curve



valid_auc = roc_auc_score(y_valid, y_valid_proba)
print(f"Validation ROC-AUC (Logistic Regression): {valid_auc:.4f}")


fpr, tpr, thresholds = roc_curve(y_valid, y_valid_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"LogReg (AUC = {valid_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Validation Set")
plt.legend()
plt.show()


# ===============================
# 5. Preprocessing + Logistic Regression Model
# ===============================

# Pipelines for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Logistic Regression model
log_reg = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    # class_weight="balanced"   # you can try this on/off
)

# Final pipeline = preprocessing + model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", log_reg)
])

# ===============================
# 6. Train / Validation Split
# ===============================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape, "Valid size:", X_valid.shape)



model.fit(X, y)



test_proba = model.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame({
    id_col: df_test[id_col],
    target: test_proba
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
submission.head()




