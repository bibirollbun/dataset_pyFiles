# train_path = playground_series_s4e12_path + '/train.csv'
# test_path = playground_series_s4e12_path + '/test.csv'
# submission_path = playground_series_s4e12_path + '/sample_submission.csv'

train_path = "/kaggle/input/playground-series-s4e12/train.csv"
test_path = "/kaggle/input/playground-series-s4e12/test.csv"
submission_path = "/kaggle/input/playground-series-s4e12/sample_submission.csv"


# %%capture
# !pip install category_encoders


import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Models
from sklearn.linear_model import LinearRegression, Ridge
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


df = pd.read_csv(train_path, index_col="id")
df.head()


df.info()


df.describe()


(df.isna().sum() / len(df)).sort_values(ascending=False) * 100


fig = px.histogram(
    df,
    x="Age",
    marginal="box",
    nbins=50,
    title="Distribution of Age")
fig.update_layout(bargap=0.1)
fig.show()


fig = px.histogram(
    df,
    x="Annual Income",
    marginal="box",
    color="Gender",
    nbins=50,
    title="Distribution of Annual Income"
)
fig.update_layout(bargap=0.1)
fig.show()


df["Credit Score"].plot(kind="box", vert=False)


df["Smoking Status"].value_counts()


px.histogram(
    df.sample(10000),
    x="Annual Income",
    color="Smoking Status",
    title=""
)


fig = px.scatter(
    df.sample(2000),
    x="Health Score",
    y="Annual Income"
)
fig.show()


sns.heatmap(df.select_dtypes("number").corr())
plt.title("Correlation Matrix")


df.head(1)


def wrangle(filepath):
  df = pd.read_csv(filepath, index_col="id")
  df.drop(columns=["Policy Start Date", "Previous Claims", "Occupation"], inplace=True)
  return df

df = wrangle(train_path)


target = "Premium Amount"
X = df.drop(columns=target)
y = df[target]


def rmse(targets, predictions):
    return np.sqrt(np.mean((targets - predictions) ** 2))


model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    LinearRegression()
)


model.fit(X, y)


model.score(X, y)


rmse(y, model.predict(X))


test_df = wrangle(test_path)
test_df.info()


test_pred = pd.DataFrame(model.predict(test_df), index=test_df.index, columns=["Premium Amount"])
test_pred.info()


subm = pd.read_csv(submission_path, index_col="id")
subm.info()


rmse(subm["Premium Amount"], test_pred["Premium Amount"])


test_pred.rename({"Premium Amount Pred": "Premium Amount"})[["Premium Amount"]].to_csv("submission.csv")


!kaggle competitions submit -c playground-series-s4e12 -f submission.csv -m "Message"

