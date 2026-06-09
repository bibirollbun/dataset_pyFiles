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


df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df.head()


df.info()


df.isna().sum()


df["date"] = pd.to_datetime(df["date"])


a = df.groupby(by=df["date"].dt.year) \
    .agg({"num_sold": "sum"})


a


a.plot(kind="line", grid=True)


df["Year"] = df["date"].dt.year
df["Month"] = df["date"].dt.month
df["Day"] = df["date"].dt.day


df.head()


df2 = df.drop(columns=["id", "date"])


df2


from sklearn.impute import KNNImputer


mdn = df["num_sold"].median()
df2["num_sold"] = df2["num_sold"].fillna(mdn)


df2.head()


X = df2.drop(columns=["num_sold"])


Y = df2[["num_sold"]]


from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


cat = list(X.columns[X.dtypes == "object"])
con = list(X.columns[X.dtypes != "object"])


cat


con


num_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)


cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)


pre = ColumnTransformer(
    [
        ("num", num_pipe, con),
        ("cat", cat_pipe, cat)
    ]
).set_output(transform="pandas")


X_pre = pre.fit_transform(X)
X_pre.head()


X_pre.shape


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X_pre, Y, test_size=0.2, random_state=42)


xtrain.shape


xtest.shape


from xgboost import XGBRegressor


model = XGBRegressor(
    n_estimators=300,
    max_depth=3
)
model.fit(xtrain, ytrain)


model.score(xtrain, ytrain)


model.score(xtest, ytest)


from sklearn.model_selection import cross_val_score


scores1 = cross_val_score(model, xtrain, ytrain, cv=5, scoring="r2")
scores1


scores1.mean()


scores2 = cross_val_score(model, xtrain, ytrain, scoring="neg_root_mean_squared_error")
scores2


-scores2.mean()


from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import make_scorer
scorer = make_scorer(mape)
scores3 = cross_val_score(model, xtrain, ytrain, cv=5, scoring=scorer)


ypred_test = model.predict(xtest)


mape(ytest, ypred_test)


imp = pd.Series(model.feature_importances_, index=xtrain.columns)


imp


imp.sort_values()


imp.sort_values().plot(kind="barh")


xnew = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
xnew.head()


xnew["date"] = pd.to_datetime(xnew["date"])


xnew["Year"] = xnew["date"].dt.year
xnew["Month"] = xnew["date"].dt.month
xnew["Day"] = xnew["date"].dt.day


xnew2 = xnew.drop(columns=["id", "date"])


xnew2


xnew_pre = pre.transform(xnew2)
xnew_pre.head()


preds = model.predict(xnew_pre)
preds


res = xnew[["id"]]


res["num_sold"] = preds


res = res.round(0)


res


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor()
]


scorer


def evaluate_single_model(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)
    mape_train = mape(ytrain, ypred_train)
    mape_test = mape(ytest, ypred_test)
    scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring=scorer)
    return {
        "name": type(model).__name__,
        "model": model,
        "mape_train": mape_train,
        "mape_test": mape_test,
        "mape_cv": scores.mean()
    }


def algo_evaluation(models, xtrain, ytrain, xtest, ytest):
    res = []
    for model in models:
        r = evaluate_single_model(model, xtrain, ytrain, xtest, ytest)
        print(r)
        res.append(r)
    res_df = pd.DataFrame(res)
    df = res_df.sort_values(by="mape_cv")
    best_model = df.loc[0, "model"]
    return df, best_model


res, best_model = algo_evaluation(models, xtrain, ytrain, xtest, ytest)


res


best_model = res.loc[4, "model"]


best_model


best_model.score(xtrain, ytrain)


best_model.score(xtest, ytest)


preds = best_model.predict(xnew_pre)
preds


res2 = xnew[["id"]]
res2["num_sold"] = preds


res2


res2.to_csv("submission.csv", index=False)

