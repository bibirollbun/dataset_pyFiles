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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.head()


train.shape


train.info()


train.isnull().sum()


test.isnull().sum()


train.describe().T


train.duplicated().sum()


train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male':1, 'female': 0})



train.head()


train=train.drop(columns=['id'],axis=1)
test=test.drop(columns=['id'],axis=1)


sns.pairplot(train[['Sex','Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']], diag_kind='kde')
plt.show()


cols = ['Sex','Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
for col in cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# Distribution plots
for col in cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# Calories vs Duration/Heart Rate
sns.scatterplot(x='Duration', y='Calories', hue='Sex', data=train)
plt.title('Calories vs Duration')
plt.show()


sns.scatterplot(x='Heart_Rate', y='Calories', hue='Sex', data=train)
plt.title('Calories vs Heart Rate')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


X = train.drop(columns=['Calories'])
y = np.log1p(train['Calories']) 


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)



import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


# RMSLE function to use after inverse transform
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Define models
models = [
    ("LinearRegression", LinearRegression()),
    ("Ridge", Ridge()),
    ("Huber", HuberRegressor()),
    ("ElasticNetCV", ElasticNetCV()),
    ("DecisionTree", DecisionTreeRegressor()),
    ("RandomForest", RandomForestRegressor()),
    ("ExtraTrees", ExtraTreesRegressor()),
    ("GradientBoosting", GradientBoostingRegressor()),
    ("XGBoost", XGBRegressor(verbosity=0)),
    ("CatBoost", CatBoostRegressor(verbose=0)),
    ("LightGBM", LGBMRegressor())
]


# Cross-validation
results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    try:
        scores = cross_val_score(model, X, y, cv=kf, scoring=rmsle_scorer)
        mean_score = -scores.mean()
        results.append((name, mean_score))
    except Exception as e:
        results.append((name, f"Failed: {e}")) 

# Print results
for name, score in results:
    print(f"{name}: RMSLE = {score}")



# Step 1: Find the best model's name and score
valid_results = [(name, score) for name, score in results if isinstance(score, (float, int)) and np.isfinite(score)]
best_name, best_score = sorted(valid_results, key=lambda x: x[1])[0]

# Step 2: Retrieve the corresponding model instance from the models list
best_model = dict(models)[best_name]

print(f"\nBest Model: {best_name} with RMSLE = {best_score:.4f}")



# Predict on test data
best_model.fit(X,y)
y_test_pred = best_model.predict(test)
y_test_pred = np.maximum(0, y_test_pred)



submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories'] = y_test_pred
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")
submission.head()

