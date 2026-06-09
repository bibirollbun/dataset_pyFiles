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
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor



df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df1 = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df.head()


df.iloc[:, 1:13]


df.sample(2)


df.shape


df.isna().sum()


df.duplicated().sum()


df.info()


df.nunique()


df.describe()


df.head()


num_cols = df.select_dtypes(include=np.number).columns[:13]

# Creating a figure with 13 rows and 2 columns (for histograms & box plots)
fig, axes = plt.subplots(13, 2, figsize=(15, 50)) 

# Loop through each numerical column and plot histogram & box plot
for i, column in enumerate(num_cols):
    # Histogram (Left)
    sns.histplot(df[column], bins=10, kde=True, ax=axes[i, 0], color=np.random.choice(['blue', 'orange', 'green', 'red', 'purple']))
    axes[i, 0].set_title(f'Histogram of {column}')
    axes[i, 0].set_xlabel(column)
    axes[i, 0].set_ylabel('Frequency')

    # Box Plot (Right)
    sns.boxplot(y=df[column], ax=axes[i, 1], color=np.random.choice(['blue', 'orange', 'green', 'red', 'purple']))
    axes[i, 1].set_title(f'Box Plot of {column}')
    axes[i, 1].set_ylabel(column)

# Adjust layout for better readability
plt.tight_layout()
plt.show()


df


X = df.iloc[:,1:-1]


X


y = df['rainfall']


y


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)


# # Define regression models
# models = {
#     "LinearRegression": Pipeline([
#         ("model", LinearRegression())
#     ]),
#     "RidgeRegression": Pipeline([
#         ("model", Ridge(alpha=1.0))
#     ]),
#     "LassoRegression": Pipeline([
#         ("model", Lasso(alpha=0.1))
#     ]),
#     "RandomForest": Pipeline([
#         ("model", RandomForestRegressor(n_estimators=100, random_state=42))
#     ]),
#     "DecisionTree": Pipeline([
#         ("model", DecisionTreeRegressor(random_state=42))
#     ]),
#     "KNeighbors": Pipeline([
#         ("model", KNeighborsRegressor(n_neighbors=5))
#     ]),
#     "SVR": Pipeline([
#         ("model", SVR(kernel='rbf'))
#     ]),
#     "AdaBoost": Pipeline([
#         ("model", AdaBoostRegressor(n_estimators=50, random_state=42))
#     ]),
#     "XGBoost": Pipeline([
#         ("model", XGBRegressor(n_estimators=100, random_state=42))
#     ]),
#     "CatBoost": Pipeline([
#         ("model", CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, verbose=0))
#     ])
# }

# # Dictionary to store results
# results = {}

# # Train and evaluate models
# for name, pipeline in models.items():
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
    
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     results[name] = {"MSE": mse, "MAE": mae, "RÂ² Score": r2}
    
#     print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, RÂ² Score: {r2:.4f}")

# # # Find the best model based on RÂ² Score
# best_model = max(results, key=lambda x: results[x]["RÂ² Score"])
# print(f"\nğŸ�† Best Model: {best_model} - RÂ² Score: {results[best_model]['RÂ² Score']:.4f}")


from sklearn.pipeline import Pipeline
models = {
    "CatBoost": Pipeline([
        ("model", CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, verbose=0))
    ])
}

# Train and evaluate models
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  # Correct metric for regression
    
    print(f"{name} - MSE: {mse:.4f}, RÂ² Score: {r2:.4f}")



prediction = pipeline.predict(pd.DataFrame([[143,1008.1,33.1,30.1,28.0,26.1,78.0,29.0,11.4,220.0,14.5]],columns=['day','pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']))
if prediction >0.5:
    print("RainFall")
else:
    print("NO RainFall")

