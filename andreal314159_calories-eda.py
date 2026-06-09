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
test = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = test.drop(columns = ["id"])
display(train.head())
display(train.info())
display(train.describe())



%%time

g = sns.pairplot(
    data = train,
    plot_kws = {"size":0.05, "marker": "."}
)

g.fig.suptitle("Distribution of various features")
plt.show()


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



train.loc[train.Height <133, "Height"] = train.Height.median()
test.loc[test.Height <133, "Height"] = train.Height.median()

# let's make a BMI index
train["BMI"] = train.Weight / train.Height**2 * 100**2
test["BMI"] = test.Weight / test.Height**2 * 100**2



train.loc[train.BMI < 16]






