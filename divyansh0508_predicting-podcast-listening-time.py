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
import warnings
warnings.filterwarnings('ignore')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate


# Load the Datasets -->
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


# Checking the Dataset-->
test_data.head()
train_data.head()


print(f"Test Data Size--> {test_data.shape}\nTrain Data Size--> {train_data.shape}")


# Is there any NUll value
train_data.isnull().sum()
train_data.isnull().sum()


# To Check the datatypes of all cols
train_data.info()
test_data.info()


# Describing it more
train_data.describe()
test_data.describe()


%matplotlib inline
train_data.hist(bins = 50, figsize = (20,15))
plt.show()


# As we have Null val in ['Number_of_Ads', 'Episode_Length_minutes', 'Guest_Popularity_percentage']
train_data['Number_of_Ads'].fillna(train_data['Number_of_Ads'].mean(),inplace = True)
train_data['Episode_Length_minutes'].fillna(train_data['Episode_Length_minutes'].mean(),inplace = True)
train_data['Guest_Popularity_percentage'].fillna(train_data['Guest_Popularity_percentage'].median(),inplace = True)


# Similarly in test set
test_data['Episode_Length_minutes'].fillna(test_data['Episode_Length_minutes'].mean(),inplace = True)
test_data['Guest_Popularity_percentage'].fillna(test_data['Guest_Popularity_percentage'].median(),inplace = True)


# Handling Object Dtype values -->
cat_objects = [var for var in train_data.columns if train_data[var].dtypes == "object"]
print(cat_objects)


# Checking each col                                What step we`ll take    unique vals
train_data['Genre'].value_counts()                # OHE                    10
train_data['Episode_Title'].value_counts()        # Drop
train_data['Podcast_Name'].nunique()              # Drop
train_data['Publication_Day'].value_counts()      # OHE                     7
train_data['Publication_Time'].value_counts()     # OHE                     4
train_data['Episode_Sentiment'].value_counts()    # OrdinalEncoding         3


train_1 = train_data.drop(columns= ["Episode_Title","Podcast_Name", "id"])
test_1 = test_data.drop(columns= ["Episode_Title","Podcast_Name", "id"])


# Now updated data is -->
train_1.head()                   # shape ----> (750000,9)
test_1.head()                    # shape ----> (250000,8)


train_2 = pd.get_dummies(train_1,columns =['Genre', 'Publication_Day', 'Publication_Time'], drop_first= True,dtype = int)
train_2.head()


test_2 = pd.get_dummies(test_1,columns =['Genre', 'Publication_Day', 'Publication_Time'], drop_first= True,dtype = int)
test_2.head()


label = LabelEncoder()
train_2['Episode_Sentiment'] = label.fit_transform(train_2['Episode_Sentiment'])
test_2['Episode_Sentiment'] = label.fit_transform(test_2['Episode_Sentiment'])


x = train_2.drop(columns = ['Listening_Time_minutes'])
y = train_2['Listening_Time_minutes']


# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print(f"{x_train.shape}, {x_test.shape}\n{y_train.shape}, {y_test.shape}")


lr = LinearRegression()
lr.fit(x_train, y_train)


x_train_p = lr.predict(x_train)
x_test_p = lr.predict(x_test)


rmse = np.sqrt(mean_squared_error(y_test, x_test_p))
print(" Linear Regression RMSE:", rmse)

r2 = r2_score(y_test, x_test_p)
print(" Linear Regression R2 Score :", r2*100,'%')


# Training Ridge Regression
model = Ridge(alpha=1.0)  # alpha is the regularization strength
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)


# Gradient Boost Regressor

gb = GradientBoostingRegressor(random_state=42)
gb.fit(x_train, y_train)


y_pred = gb.predict(x_test)

print("GBR R2:", r2_score(y_test, y_pred))
print("GBR RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# hence final result -->
pred_lgbm = gb.predict(test_2)


# So we found GradientBoost is performing for better 
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission_lgbm = pd.DataFrame({'id': sample_submission.id, 'Listening_Time_minutes' : pred_lgbm})
submission_lgbm.to_csv('submission.csv', index=False)


print("completed ---")

