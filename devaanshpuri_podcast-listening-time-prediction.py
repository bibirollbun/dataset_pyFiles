import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_id = test_df['id']
train_df.head()


train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median(), inplace=True)

test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median(), inplace=True)
test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Number_of_Ads'].fillna(test_df['Number_of_Ads'].median(), inplace=True)


#feature engineering 
#splitting ep number as int
train_df['Episode_Title_num'] = train_df['Episode_Title'].astype(str).str.replace('Episode ', '').astype(int)
test_df['Episode_Title_num'] = test_df['Episode_Title'].astype(str).str.replace('Episode ', '').astype(int)

#calculating ad density
train_df['Ad_Density']= train_df['Number_of_Ads'] / train_df['Episode_Length_minutes']
test_df['Ad_Density']= test_df['Number_of_Ads'] / test_df['Episode_Length_minutes']

#average popularity
train_df['Popularity_Constant'] = (train_df['Guest_Popularity_percentage'] + train_df['Host_Popularity_percentage'])/2
test_df['Popularity_Constant'] = (test_df['Guest_Popularity_percentage'] + test_df['Host_Popularity_percentage'])/2

#train_df.drop(['Episode_Title','Number_of_Ads','Guest_Popularity_percentage','Host_Popularity_percentage'] , axis=1 , inplace=True)
#test_df.drop(['Episode_Title','Number_of_Ads','Guest_Popularity_percentage','Host_Popularity_percentage'] , axis=1 , inplace=True)


train_df.info()


train_df.isnull().sum()


train_df = train_df.drop_duplicates()
train_df = train_df.dropna()


train_df.isnull().sum()


train_df.info()


#encoding various features time , day, sentiment
from sklearn.preprocessing import LabelEncoder

columns_to_encode = ['Publication_Day', 'Genre', 'Publication_Time', 'Episode_Sentiment']
encoders = {}

for col in columns_to_encode:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])  # Assuming test_df has same labels
    encoders[col] = le  # Store encoder in case you need inverse_transform later



train_df.head()


train_df.drop(['id', 'Podcast_Name','Episode_Title'], axis=1, inplace=True)
test_df.drop(['id', 'Podcast_Name','Episode_Title'], axis=1, inplace=True)


train_df.head()


train_df.info()


test_df.info()



X = train_df.drop(['Listening_Time_minutes'],axis=1)
y = train_df['Listening_Time_minutes']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)



import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error 
model = XGBRegressor( objective='reg:squarederror',
    random_state=42,
    n_estimators=2000,
    n_splits = 5,
    learning_rate=0.01,
    #early_stopping_rounds = 100,
    max_depth=20,
    subsample = 0.88,
    colsample_bytree = 0.7,
    #reg_alpha = 0.66,
    #reg_lambda=1.257,
    #tree_method='hist',
    eval_metric='rmse',
    #gamma = 0.08,
    min_child_weight = 5,                                  
    seed=42,
    enable_categorical=True)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Validation Mean Absolute Error (MAE):", mae)


test_predictions = model.predict(test_df)

# Create a submission file
submission = pd.DataFrame({'id':test_id, 'Listening_Time_minutes': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")

