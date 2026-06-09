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



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestClassifier
import hdbscan
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import itertools
from sklearn.linear_model import LassoCV
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import QuantileTransformer


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.head()


train.describe()


train.info()


train.isnull().sum()


test.isnull().sum()


train.columns


sns.histplot(train['Calories'],color='forestgreen', kde=True)


features = [ 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp']

target = train['Calories']


for feature in features:
    price_corr = train[[feature, 'Calories']].corr().iloc[0, 1]

    sns.regplot(x=feature,y=target   ,data=train ,scatter_kws={'alpha':0.5})
    plt.show()

    print(price_corr)


scaler = LabelEncoder()

train['Sex'] = scaler.fit_transform(train['Sex'])
test['Sex'] = scaler.fit_transform(test['Sex'])





plt.figure(figsize=(12,8))
sns.heatmap(train.corr(),annot=True,cmap='Greens')


X_train_const = sm.add_constant(train[features])
X_train_const


 
model_fitted = sm.OLS(target,X_train_const).fit()

print(model_fitted.summary())


X_test_const = sm.add_constant(test[features])

#make predictions on the test set 
test_predictions = model_fitted.predict(X_test_const)
test_predictions = np.clip(test_predictions, 0, None)
test_predictions


submission = pd.DataFrame({
    'id': test['id'], 
    'Calories':test_predictions
})

submission.to_csv('submission_calorie_prediction.csv', index=False)
print('CSV saved')







