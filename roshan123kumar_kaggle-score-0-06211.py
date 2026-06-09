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


train_data=pd.read_csv(r"/kaggle/input/playground-series-s5e5/train.csv")
train_data.head()


test_data=pd.read_csv(r"/kaggle/input/playground-series-s5e5/test.csv")
test_data.head()


print(f"Null value in train data: \n{train_data.isnull().sum()}")
print("#" * 20)
print(f"Null value in test data: \n{test_data.isnull().sum()}")


print(f"Duplicate value in train data: \n{train_data.duplicated().sum()}")
print("*" * 20)
print(f"Duplicate value in test data: \n{test_data.duplicated().sum()}")


train_data.info()


test_data.info()


print(f"Train data shape:\n {train_data.shape}")
print(f"Test data shape:\n {test_data.shape}")


train_data.drop(columns=["id"],inplace=True)


train_data.sample()


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



train_data=pd.get_dummies(train_data,drop_first=True)


train_data.head()


sns.pairplot(data=train_data[:2000],hue="Calories")
plt.show()


plt.title(" Outlier present in Train Data")
sns.boxplot(train_data)
plt.xticks(rotation=90)
plt.show()


plt.title(" Outlier present in test Data")
sns.boxplot(test_data)
plt.xticks(rotation=90)
plt.show()


def remove_outliers_iqr_columns(df, columns):
    """
    Remove rows from the DataFrame where any value in the specified columns is an outlier using the IQR method.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to check for outliers.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    mask = pd.Series(True, index=df.index)
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask &= df[column].between(lower_bound, upper_bound)
    return df[mask].reset_index(drop=True)

# train_data = remove_outliers_iqr_columns(train_data, ["Height", "Weight", "Calories"])
train_data = remove_outliers_iqr_columns(train_data, ["Calories"])


sns.boxplot(train_data)
plt.xticks(rotation=90)
plt.show()


sns.heatmap(train_data.corr(),annot=True)


x=train_data.drop(columns=["Calories"],axis=1)
y=train_data["Calories"]


from sklearn.model_selection import StratifiedKFold,train_test_split
sf=StratifiedKFold(n_splits=10)
for fold, (train_index,test_index) in enumerate(sf.split(x,y)):
    x_train,x_test=x.iloc[train_index],x.iloc[test_index]
    y_train,y_test=y[train_index],y[test_index]
    print(f"Fold {fold}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")



print(f"x_train shape : {x_train.shape}  & y_train shape :{y_train.shape}")
print(f"x_test shape : {x_test.shape}  & y_test shape :{y_test.shape}")#750000


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from  sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from  sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


pipe=Pipeline([
    ("Model",LGBMRegressor(n_estimators=500,learning_rate=0.1))
    
])
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse=mean_squared_error(y_test,y_pred)
print("R-squared:", r2)
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))


models=pd.DataFrame({"Models":["LogisticRegression","DecisionTreeRegressor","RandomForestRegressor","KNeighborsRegressor","LGBMRegressor"],
                    "R-square":[ 0.9642637128045057,0.9930638128349746, 0.9962447090887733,0.9948725084767275, 0.9966669963806372],
                    "Mse":[139.13165333333333,27.00457332962963,14.62042853999922,19.962800533333333,12.950485210429264],
                    "RMSE":[11.795408146110644,5.196592472922004,3.82366689710273,4.467974992469556,3.5986782588096515]})
models.head()


test_ids = test_data["id"]
x_real_test = test_data.drop(columns=["id"])
x_real_test = pd.get_dummies(x_real_test, drop_first=True)

# Align test columns to training columns
x_real_test = x_real_test.reindex(columns=x.columns, fill_value=0)

# === PREDICT & SAVE SUBMISSION FILE === #
y_real_pred = pipe.predict(x_real_test)

submission = pd.DataFrame({
    "id": test_ids,
    "Calories": y_real_pred
})
submission.to_csv("submissionf.csv", index=False)
print(" Submission file saved as 'submissionf.csv'")

