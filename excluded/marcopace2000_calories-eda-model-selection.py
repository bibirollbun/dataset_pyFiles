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
import matplotlib.pylab as plt
import seaborn as sns

plt.style.use("ggplot")


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df_input = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv",index_col=0)
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv",index_col=0)


df_input.columns


df_input.isna().sum()


df_input.head(3)


df_input.describe()


num_cols = df_input.shape[1]
fig, axs = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, 4 * num_cols))

for i, column in enumerate(df_input.columns):
    ax = axs[i]
    if pd.api.types.is_numeric_dtype(df_input[column]):
        sns.histplot(df_input[column], bins=30, kde=True, color='skyblue', ax=ax)
    else:
        sns.countplot(x=df_input[column], palette="pastel", ax=ax)
    ax.set_title(f'Distribution of {column}', fontsize=14)

plt.tight_layout()
plt.show()


sns.pairplot(df_input.sample(1000), hue="Sex")
plt.tight_layout()
plt.show()


sns.heatmap(df_input.corr(numeric_only=True), annot = True, cmap='coolwarm')
plt.title("Correlations")
plt.show()


df_input["Sex"] = df_input["Sex"].map({"male":0,"female":1})
df_input["Duration**2"] = df_input["Duration"] * df_input["Duration"]
df_input["Body_temp**2"] = df_input["Body_Temp"] * df_input["Body_Temp"]
df_input["Body_temp*Duration"] = df_input["Body_Temp"] * df_input["Duration"]

df_test["Sex"] = df_test["Sex"].map({"male":0,"female":1})
df_test["Duration**2"] = df_test["Duration"] * df_test["Duration"]
df_test["Body_temp**2"] = df_test["Body_Temp"] * df_test["Body_Temp"]
df_test["Body_temp*Duration"] = df_test["Body_Temp"] * df_test["Duration"]


df_input.head(3)


from sklearn.model_selection import train_test_split

x = df_input.drop(columns="Calories")
y = df_input["Calories"]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)


from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


model_ada = AdaBoostRegressor(n_estimators = 500, random_state = 42)
model_ada.fit(x_train,y_train)
print(model_ada.score(x_val,y_val))


model_cat = CatBoostRegressor(n_estimators = 500,verbose=0, random_state = 42)
model_cat.fit(x_train,y_train)
print(model_cat.score(x_val,y_val))


model_lgbm = LGBMRegressor(n_estimators = 500,verbose = 0, random_state = 42)
model_lgbm.fit(x_train,y_train)
print(model_lgbm.score(x_val,y_val))


from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor

estimators = [('ada', AdaBoostRegressor(n_estimators = 500, random_state = 42)),
              ('cat', CatBoostRegressor(n_estimators = 500,verbose=0, random_state = 42)),
              ('lgbm', LGBMRegressor(n_estimators = 500,verbose = 0, random_state = 42))]

reg_rf = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)
)
reg_rf.fit(x_train, y_train).score(x_val, y_val)


from sklearn.linear_model import RidgeCV

estimators = [('ada', AdaBoostRegressor(n_estimators = 500, random_state = 42)),
              ('cat', CatBoostRegressor(n_estimators = 500,verbose=0, random_state = 42)),
              ('lgbm', LGBMRegressor(n_estimators = 500,verbose = 0, random_state = 42))]

reg_rdg = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV()
)
reg_rdg.fit(x_train, y_train).score(x_val, y_val)


model = CatBoostRegressor(n_estimators = 500,verbose=0, random_state = 42)
model.fit(x,y)

prediction = model.predict(df_test)


prediction = np.maximum(0,prediction)
out = pd.DataFrame(prediction,index=df_test.index,columns=["Calories"])
out.describe().T


out.to_csv("submission.csv")

