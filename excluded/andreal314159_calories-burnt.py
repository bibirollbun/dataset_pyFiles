#


import pandas as pd
import numpy as np
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.simplefilter("ignore")



train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
train = train.drop(columns = ["id"])
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test = test.drop(columns = ["id"])
display(train.head())
display(train.info())
display(train.describe())



%%time
"""
g = sns.pairplot(
    data = train,
    plot_kws = {"size":0.05, "marker": "."}
)

g.fig.suptitle("Distribution of various features")
plt.show()
"""


train[train.Height < 140]


import plotly.express as px

fig = px.scatter(
    train,
    x = "Weight",
    y = "Height",
    title= "Weight and Height of individuals working out"
)

fig.update_layout(template="plotly_white")
fig.show()



f = sns.displot(
    data = train,
    x = "Height",
    hue = "Sex"
    )
f.set(title="Height distribution")
sns.despine()
plt.show()


train.loc[(train.Sex == "male") & ((train.Height > 205) | (train.Height < 150)), "Height"] = train[train.Sex == "male"].Height.median()
train.loc[(train.Sex == "female") & ((train.Height > 195) | (train.Height < 133)), "Height"] = train[train.Sex == "female"].Height.median()
train.head()

# test data
test.loc[(test.Sex == "male") & ((test.Height > 205) | (train.Height < 150)), "Height"] = train[train.Sex == "male"].Height.median()
test.loc[(test.Sex == "female") & ((test.Height > 195) | (test.Height < 133)), "Height"] = train[train.Sex == "female"].Height.median()
test.head()



# let's make a BMI index
train["BMI"] = train.Weight / train.Height**2 * 100**2
test["BMI"] = test.Weight / test.Height**2 * 100**2


# plotting
g = sns.displot(data = train, x="BMI", hue = "Sex")
plt.title("Distribution of BMI")
sns.despine()
plt.show()


h = sns.boxplot(
    data = train,
    x = "BMI",
    hue = "Sex")
h.set(title="Distribution and outliers in BMI")
sns.despine()
plt.show()


train[train.BMI < 18.5*0.9]
train[train.BMI > 40]



train_encoded = pd.get_dummies(train, drop_first=True, dtype=int)
test_encoded = pd.get_dummies(test, drop_first=True, dtype=int)

train_encoded.head()


test_encoded


# Correlations.

g = sns.heatmap(train_encoded.corr(), 
                annot=True,
                fmt = ".2f",
                linewidth = 1,
                cmap = "Blues")
g.set(title = "Correlation of features")
plt.show()


from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
X = train_encoded.drop(columns = ["Calories", "Height"])
y = train_encoded["Calories"]
model = RandomForestRegressor()
model = Ridge()
rmsle = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_log_error(y, np.clip(y_pred, 0.001, 1000))), greater_is_better=False)
scores = cross_val_score(model, X, y, cv = 5, scoring = rmsle, verbose=4)

result = np.mean(scores) + np.std(scores)
result


# Submission

model.fit(X, y)

predictions = np.clip(model.predict(test_encoded), 0.0001, 1000)
predictions.min()


subm = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission_df = pd.DataFrame({"Calories": predictions, "id": subm.id})

submission_df.to_csv("submission.csv", index=False)


test




