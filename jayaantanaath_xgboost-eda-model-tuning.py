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

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
import xgboost as xgb


train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
train_data = pd.DataFrame(train_data)


train_data.isna().sum()


train_data.head(5)


print(len(train_data['Podcast_Name'].unique()))
train_data['Podcast_Name'].value_counts().sort_values(ascending=True)


train_data['Episode_Title'].value_counts().sort_values(ascending=True)


cat_col = train_data.select_dtypes(include='object').columns
num_col = train_data.select_dtypes(include=['int64','float64']).columns
cat_col = [col for col in cat_col]
num_col = [num for num in num_col]
num_col, cat_col


train_data[cat_col].describe()


train_data[num_col].describe()


train_data[cat_col].isna().sum()


train_data[num_col].isna().sum()


train_data[(train_data['Episode_Length_minutes'] == 0) & (train_data['Listening_Time_minutes'] == 0)]


# Linear Interpolation for Episode_Length_minutes
train_data['Episode_Length_minutes'] = train_data['Episode_Length_minutes'].interpolate(method='slinear')

# Linear Interpolation for Guest_Popularity_percentage
train_data['Guest_Popularity_percentage'] = train_data['Guest_Popularity_percentage'].interpolate(method='slinear')

# There's only one row is missing for Number_of_Ads
train_data['Number_of_Ads'] = train_data['Number_of_Ads'].fillna(train_data['Number_of_Ads'].median())

print(train_data.isna().sum())
train_data[num_col].hist(figsize=(10, 8), bins=50)
plt.show()


# Random Sampling Imputation for Episode_Length_minutes
train_data['Episode_Length_minutes'] = train_data['Episode_Length_minutes'].apply(
    lambda x: x if pd.notna(x) else train_data['Episode_Length_minutes'].dropna().sample(1).values[0]
)

# Random Sampling Imputation for Guest_Popularity_percentage
train_data['Guest_Popularity_percentage'] = train_data['Guest_Popularity_percentage'].apply(
    lambda x: x if pd.notna(x) else train_data['Guest_Popularity_percentage'].dropna().sample(1).values[0]
)

# There's only one row is missing for Number_of_Ads
train_data['Number_of_Ads'] = train_data['Number_of_Ads'].fillna(train_data['Number_of_Ads'].median())

print(train_data.isna().sum())
train_data[num_col].hist(figsize=(10, 8), bins=50)


train_data.head()


train_data[num_col].describe()


train_data[cat_col].describe()


# Create a 3x2 subplot layout
fig, axes = plt.subplots(3, 2, figsize=(10, 12))  # 3 rows, 2 columns

# Flatten the 2D array of axes into a 1D list
axes = axes.flatten()

# Plot boxplots for each numerical column
for i, col in enumerate(num_col):
    sns.boxplot(y=train_data[col], ax=axes[i], color='lightblue')
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()


train_data[train_data['Listening_Time_minutes'] > train_data['Episode_Length_minutes']]


sns.histplot(train_data[train_data['Listening_Time_minutes'] > train_data['Episode_Length_minutes']]['Listening_Time_minutes'], bins=50, kde=True)
plt.title("Distribution of Listening Time for Listeners Who Exceed Episode Length")
plt.show()


train_data['Number_of_Ads'].quantile(.99)


train_data[train_data['Number_of_Ads'] > 3]


train_data[train_data['Episode_Length_minutes'] > 150]


train_data[(train_data['Guest_Popularity_percentage'] > 100)]['Guest_Popularity_percentage'].count()


# Outlier remove

train_data = train_data[train_data['Episode_Length_minutes'] <= 150]
train_data = train_data[train_data['Guest_Popularity_percentage'] <= 100]
train_data = train_data[train_data['Number_of_Ads'] <= 3]


# Create a 3x2 subplot layout
fig, axes = plt.subplots(3, 2, figsize=(8, 10))  # 3 rows, 2 columns

# Flatten the 2D array of axes into a 1D list
axes = axes.flatten()

# Plot boxplots for each numerical column
for i, col in enumerate(num_col):
    sns.boxplot(y=train_data[col], ax=axes[i], color='lightblue')
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()


train_data[cat_col]


print(len(train_data['Podcast_Name'].unique()), 'Podcast(unique)')
print(len(train_data['Episode_Title'].unique()), 'Episode 01_100(unique)')
for col in ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'] :
    print(f"Length: {len(train_data[col].unique())} Col: {train_data[col].unique()}")


cat_col


ohe_col = ['Podcast_Name',
 'Episode_Title',
 'Genre',
 'Publication_Day',
 'Publication_Time']


dummy = pd.get_dummies(train_data[ohe_col])
dummy.tail()


train_data['Episode_Sentiment'].unique()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(['Negative', 'Neutral', 'Positive'])
train_data['Episode_Sentiment_encoded'] = le.transform(train_data['Episode_Sentiment'])
train_data_encoded = pd.concat([train_data, dummy], axis=1)
train_data_encoded.drop(['id','Podcast_Name',
 'Episode_Title',
 'Genre',
 'Publication_Day',
 'Publication_Time','Episode_Sentiment'],axis=1, inplace=True)
train_data_encoded = train_data_encoded.reset_index(drop=True)
train_data_encoded.head()


X = train_data_encoded.drop('Listening_Time_minutes', axis=1)
y = train_data_encoded['Listening_Time_minutes']
X.shape, y.shape


num_col.remove('Listening_Time_minutes')
num_col


from sklearn.preprocessing import LabelEncoder

test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_data.head()

print(test_data.isna().sum())
le = LabelEncoder()
le.fit(['Negative', 'Neutral', 'Positive'])
test_data['Episode_Sentiment_encoded'] = le.transform(test_data['Episode_Sentiment'])
dummy = pd.get_dummies(test_data[ohe_col])
test_data_encoded = pd.concat([test_data, dummy], axis=1)
test_data_encoded.drop(['Podcast_Name',
 'Episode_Title',
 'Genre',
 'Publication_Day',
 'Publication_Time','Episode_Sentiment'],axis=1, inplace=True)
test_data_encoded = test_data_encoded.reset_index(drop=True)
final_test_df = test_data_encoded.drop('id',axis=1)
final_test_df.head()


X_test = final_test_df
X_train = X
y_train = y
X_test.shape, X_train.shape, y_train.shape


import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Convert data to DMatrix (optimized data structure for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)

# Define XGBoost regression parameters
params ={
    "objective": "reg:squarederror",  # Regression objective
    "eval_metric": "rmse",  # Root Mean Squared Error
    "max_depth": 10,  # Tree depth (higher = more complex model)
    "eta": 0.1,  # Learning rate
    "subsample": 0.8,  # Row sampling (to reduce overfitting)
}


# Train the model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)


X_test.shape, X_train.shape, y_train.shape


# Make predictions
# Convert X_test to DMatrix before predicting
dtest = xgb.DMatrix(X_test)
y_pred = xgb_model.predict(dtest) 


y_pred.reshape(-1,1)


test_data_encoded['Listening_Time_minutes'] = y_pred


test_data_encoded
submission = test_data_encoded[['id', 'Listening_Time_minutes']]
submission.tail()


submission.to_csv('/kaggle/working/submission.csv', index=False)




