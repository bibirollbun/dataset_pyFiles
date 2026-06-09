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


!pip install lazypredict
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import lazypredict 
from lazypredict import Supervised 
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor



from sklearn.metrics import mean_squared_log_error

def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

df_train = df_train.drop("id",axis = 1)
df_id = df_test["id"]
df_test = df_test.drop("id",axis = 1)


df_train.head()


df_train.size


df_train.describe().T 


df_train.info()


df_train.isnull().sum()



categorical_cols = ['Sex']


numerical_cols = [col for col in df_train.columns if col not in categorical_cols]





encoder = OneHotEncoder(sparse=False, drop='first') 
cat_encoded = encoder.fit_transform(df_train[categorical_cols])
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols))

df_train = pd.concat([df_train[numerical_cols].reset_index(drop=True), cat_encoded_df], axis=1)


df_train.head()


correlation_matrix = df_train.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


X = df_train.drop(["Calories"],axis=1)
y = df_train["Calories"]



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.7,random_state =42)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


for i in range(42):
    print(i+1, lazypredict.Supervised.REGRESSORS[i][0])


reg = LazyRegressor(
    verbose=1,
    ignore_warnings=False,
    custom_metric=None,
    regressors=[
        XGBRegressor,
        GradientBoostingRegressor,
        AdaBoostRegressor
    ]
)

models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)


categorical_cols = ['Sex']


numerical_cols = [col for col in df_test.columns if col not in categorical_cols]


encoder = OneHotEncoder(sparse=False, drop='first') 
cat_encoded = encoder.fit_transform(df_test[categorical_cols])
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols))

df_test = pd.concat([df_test[numerical_cols].reset_index(drop=True), cat_encoded_df], axis=1)
df_test.head()


best_model_name = models.index[0]
print("Best Model:", best_model_name)


model = XGBRegressor()
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
score = model.score(X_test,y_test)
print(score)


test_preds = model.predict(df_test)


submission = pd.DataFrame({
    'id': df_id,
    'Calories': test_preds
})
submission.to_csv('submission.csv', index=False)

