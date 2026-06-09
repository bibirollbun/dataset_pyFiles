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


!pip install --upgrade scikit-learn


import pandas as pd
df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df.head()


df.duplicated().sum()


df.isna().sum()


df.info()


df["y"].value_counts()


X = df.drop(columns = ["id", "y"])
Y = df["y"]


X.head()


df.drop(columns=["id"]).duplicated().sum()


X.duplicated().sum()


cat_unique = X.select_dtypes(include = "object").nunique()
cat_unique


low_card = cat_unique[cat_unique < 10].index.tolist()
low_card


high_card = cat_unique[cat_unique >= 10].index.tolist()
high_card


df["job"].unique()


df["month"].unique()


num_cols = X.select_dtypes(include="number").columns.tolist()
num_cols


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)


xtrain.shape, ytrain.shape


xtest.shape, ytest.shape


from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer


num_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)


low_cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="unknown"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)


high_cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="unknown"),
    TargetEncoder(target_type="binary", random_state=42),
    StandardScaler()
)


pre = ColumnTransformer(
    [
        ("num", num_pipe, num_cols),
        ("low_card", low_cat_pipe, low_card),
        ("high_card", high_cat_pipe, high_card)
    ]
).set_output(transform="pandas")


pre.fit(xtrain, ytrain)


xtrain_pre = pre.transform(xtrain)


xtrain_pre.head()


xtest_pre = pre.transform(xtest)


import lightgbm as lgb
from sklearn.model_selection import cross_val_score


gbm = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=5000)
gbm.fit(xtrain_pre, ytrain, eval_set=[(xtest_pre, ytest)], eval_metric="auc", callbacks=[lgb.early_stopping(10)])


gbm.score(xtrain_pre, ytrain)


gbm.score(xtest_pre, ytest)


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(gbm, xtest_pre, ytest)


from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(gbm, xtest_pre, ytest)


ypred_test = gbm.predict(xtest_pre, num_iteration=gbm.best_iteration_)
ypred_test[0:5]


ytest.head()


from sklearn.metrics import classification_report

print(classification_report(ytest, ypred_test))


xnew = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
xnew.head()


xnew_pre = pre.transform(xnew)


xnew_pre.head()


preds = gbm.predict(xnew_pre, num_iteration=gbm.best_iteration_)
preds


probs = gbm.predict_proba(xnew_pre, num_iteration=gbm.best_iteration_)[:,1]
probs


res = xnew[["id"]]
res


res["y"] = probs


res


res.to_csv("submission.csv", index=False)

