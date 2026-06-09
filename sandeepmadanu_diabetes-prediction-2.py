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


!pip install lightgbm --quiet


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb


sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print("Train:", train.shape)
print("Test:", test.shape)
train.head()


target = "diagnosed_diabetes"
sns.countplot(data=train, x=target)
plt.title("Target Distribution")
plt.show()

train[target].value_counts(normalize=True)



num_cols = train.select_dtypes(include=["int64","float64"]).columns.tolist()
num_cols.remove("id")
num_cols.remove(target)


for col in num_cols:
    plt.figure()
    sns.histplot(train, x=col, hue=target, kde=True, stat="density", common_norm=False)
    plt.title(f"Distribution of {col}")
    plt.show()


corr = train[num_cols+[target]].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()


# %% [code]
X = train.drop(columns=["id", target])
y = train[target]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

lgbm_model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=1200,
    learning_rate=0.02,
    num_leaves=32,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=0.4,
    min_child_samples=40,
    class_weight="balanced",        # Important for imbalance
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", lgbm_model)
])

pipeline.fit(X_train, y_train)
valid_proba = pipeline.predict_proba(X_valid)[:,1]
print("Validation AUC:", roc_auc_score(y_valid, valid_proba))


fpr, tpr, thresholds = roc_curve(y_valid, valid_proba)
auc_score = roc_auc_score(y_valid, valid_proba)

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


prec, rec, thr = precision_recall_curve(y_valid, valid_proba)

plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()



# Get feature names after encoding
encoder = pipeline.named_steps["prep"].named_transformers_["cat"]
encoded_cat = encoder.get_feature_names_out(categorical_cols)
final_features = list(encoded_cat) + numeric_cols

importance = pipeline.named_steps["model"].feature_importances_

# Create importance dataframe
fi = pd.DataFrame({
    "feature": final_features,
    "importance": importance
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(data=fi.head(25), x="importance", y="feature")
plt.title("Top 25 Feature Importances")
plt.show()

fi.head(50)



pipeline.fit(X, y)
test_proba = pipeline.predict_proba(test.drop(columns=["id"]))[:,1]

submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_proba
})

submission.to_csv("submission_lgbm_optimized.csv", index=False)
print("Saved submission_lgbm_optimized.csv")
submission.head()



submission.to_csv("submission.csv", index=False)


