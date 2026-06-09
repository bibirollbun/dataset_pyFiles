import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


# show train data sample
train_df.head(5) 


# check dataset structure and missing values
train_df.info()


# show test data sample
test_df.head(5) 


# check dataset structure and missing values
test_df.info()


def preprocess(df):
    # fill missing values
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].mean())
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].mean())
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].median()).astype('int')

    # one-hot encoding to categorical columns
    df = pd.get_dummies(df, columns=['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment'])

    return df


# preprocessing
train_df = preprocess(train_df)
test_df = preprocess(test_df)

# prepare X data and Y data
X_features = train_df.drop(['id','Listening_Time_minutes'],axis=1)
y_target = train_df['Listening_Time_minutes']

# split train and test set
X_train, X_test, y_train, y_test= train_test_split(X_features, y_target, test_size=0.2, random_state=0)


# prepare stacking model
estimators = [
    ('cat', CatBoostRegressor(verbose=0)),
    ('xgb', XGBRegressor(verbosity=0)),
    ('lgb', LGBMRegressor(verbose=-1))
]

final_estimator = Ridge()

stack = StackingRegressor(estimators=estimators, final_estimator=final_estimator)

# train
stack.fit(X_train, y_train)


# evaluate model
stack_pred = stack.predict(X_test)
stack_rmse = np.sqrt(mean_squared_error(y_test, stack_pred))
print("model RMSE:", stack_rmse)


# predict data
X_result = test_df.drop(['id'], axis=1, inplace=False)
test_pred = stack.predict(X_result)

# make submission file
test_pd = pd.DataFrame({'id':test_df['id'],'Listening_Time_minutes':test_pred})
test_pd.to_csv("submission.csv",index=False)
test_pd.head(10)

