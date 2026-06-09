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


sample_sub = pd.read_csv(r'/kaggle/input/playground-series-s5e4/sample_submission.csv')
train = pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")
train.head()


#approach
#convert episodes into numbers, classify genre into numbers, publication_day into numbers,convert episode_sentiment into ternary classification -1,0,1,same goes for publication_time as well.
#preprocessing: replace NaN into the mean of the particular genre.


train['Episode_number'] = 0
for i in range(len(train)):
    #print(train.at[i,'Episode_Title'].split(" ")[-1])
    train.at[i,'Episode_number'] = int(train.at[i,'Episode_Title'].split(" ")[-1])
print("Loop completed lol!")
train['Episode_number'].describe()


days_index = {'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6}
publication_index = {"Morning":0,"Afternoon":1,"Evening":2,'Night':3}
sentiment_index = {"Neutral":0,"Positive":1,"Negative":-1}
genre_index = {"True Crime":0,"Comedy":1,"Education":2,"Technology":3,"Health":4,"News":5,"Music":6,"Sports":7,"Business":8,"Lifestyle":9}
train[['Day_num','Public_num','Sentiment_num','Genre_num']] = 0


for i in range(len(train)):
    #print(train.at[i,'Episode_Title'].split(" ")[-1])
    train.at[i,'Day_num'] = days_index[train.at[i,'Publication_Day']]
    train.at[i,'Public_num'] = publication_index[train.at[i,'Publication_Time']]
    train.at[i,'Sentiment_num'] = sentiment_index[train.at[i,'Episode_Sentiment']]
    train.at[i,'Genre_num'] = genre_index[train.at[i,'Genre']]
print("Loop completed lol!")


train[['id','Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads','Day_num','Public_num','Sentiment_num','Genre_num','Episode_number','Listening_Time_minutes']]


train.to_csv('my_modified_data.csv', index=False)
#Saved in /kaggle/working/


train.isna().sum()


train['Episode_Length_minutes'] = train.groupby('Genre')['Episode_Length_minutes']\
                                 .transform(lambda x: x.fillna(x.mean()))


# Fill missing Guest_Popularity_percentage with column mean
guest_mean = train['Guest_Popularity_percentage'].mean()
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(guest_mean)



train['Number_of_Ads']=train['Number_of_Ads'].fillna(0)


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



features_required = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads','Day_num','Public_num','Sentiment_num','Genre_num','Episode_number']
X_data = train[features_required]
Y_data = train['Listening_Time_minutes']
# X_train, X_test, y_train, y_test = train_test_split(
#     X_data, Y_data, test_size=0.9, random_state=99
# )


model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=1,
    colsample_bytree=1,
    tree_method='hist',
    device='cuda'# Fastest option for CPUs
)

model.fit(X_data, Y_data)



y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.4f}')



test['Episode_number'] = 0
for i in range(len(test)):
    #print(train.at[i,'Episode_Title'].split(" ")[-1])
    test.at[i,'Episode_number'] = int(test.at[i,'Episode_Title'].split(" ")[-1])
print("Loop completed lol!")
days_index = {'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6}
publication_index = {"Morning":0,"Afternoon":1,"Evening":2,'Night':3}
sentiment_index = {"Neutral":0,"Positive":1,"Negative":-1}
genre_index = {"True Crime":0,"Comedy":1,"Education":2,"Technology":3,"Health":4,"News":5,"Music":6,"Sports":7,"Business":8,"Lifestyle":9}
test[['Day_num','Public_num','Sentiment_num','Genre_num']] = 0
for i in range(len(test)):
    #print(train.at[i,'Episode_Title'].split(" ")[-1])
    test.at[i,'Day_num'] = days_index[test.at[i,'Publication_Day']]
    test.at[i,'Public_num'] = publication_index[test.at[i,'Publication_Time']]
    test.at[i,'Sentiment_num'] = sentiment_index[test.at[i,'Episode_Sentiment']]
    test.at[i,'Genre_num'] = genre_index[test.at[i,'Genre']]
print("Loop completed lol!")


X_test = test[features_required]
#Y_test = test['Listening_Time_minutes']
Y_pred = model.predict(X_test)
print(Y_pred)


sample_sub.head()


sub_required_stuff = ['id','Listening_Time_minutes']
test['Listening_Time_minutes'] = Y_pred
submittable_stuff = test[sub_required_stuff]
submittable_stuff


submittable_stuff.to_csv('submit_karo.csv', index=False)


print(os.getcwd()) 




