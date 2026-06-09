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


df_train=pd.read_csv("/kaggle/input/playground-series-s3e20/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s3e20/test.csv")
df_train.head(5)


# import matplotlib.pyplot as plt
# import seaborn as sns
# numeric_df = df_train.select_dtypes(include=['number'])

# sns.heatmap(data=numeric_df, cmap="coolwarm", annot=True)
# plt.title("Heatmap of Numeric Features")
# plt.show()


# numeric_df = df_train.select_dtypes(include=['number'])
# numeric_df.corr()


x_train=df_train.drop("ID_LAT_LON_YEAR_WEEK",axis=1)


x_train.info()


x_train=x_train.dropna()


x_train.info()


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

x = x_train.drop("emission", axis=1)
y = x_train["emission"]  # Make sure "emission" is spelled correctly here

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR()
}

for name, model in models.items(): 
    scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"{name}:")
    print(f"  RMSE scores: {rmse_scores}")
    print(f"  Mean RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")



df_test.head(5)





from sklearn.linear_model import Ridge

# Define and fit the model
pridective_model = Ridge()
pridective_model.fit(x, y)
test = df_test.dropna()
# Prepare test data
test = test.drop("ID_LAT_LON_YEAR_WEEK", axis=1)

# Predict and assign
test["CO2 Emissions"] = pridective_model.predict(test)



test["ID"] = range(1, len(test) + 1)
Submission_df=test[["ID","CO2 Emissions"]]



Submission_df




