import pandas as pd
pd.set_option('display.max_columns',None)


df= pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')


df.head(5)


col_list = df.columns.tolist()
print(col_list)


for i in col_list:
    print(i,df[i].isnull().sum())


df.info()


for i in col_list:
    print(i,df[i].nunique())


dummy_columns = ['Genre','Publication_Day','Publication_Time','Episode_Sentiment']
df_new = pd.get_dummies(df,columns=dummy_columns)


df_new.head(10)


df_new.Genre_Comedy.nunique()


df_new.columns.tolist()


import numpy as np
dummy_cols = ['Genre_Business',
 'Genre_Comedy',
 'Genre_Education',
 'Genre_Health',
 'Genre_Lifestyle',
 'Genre_Music',
 'Genre_News',
 'Genre_Sports',
 'Genre_Technology',
 'Genre_True Crime',
 'Publication_Day_Friday',
 'Publication_Day_Monday',
 'Publication_Day_Saturday',
 'Publication_Day_Sunday',
 'Publication_Day_Thursday',
 'Publication_Day_Tuesday',
 'Publication_Day_Wednesday',
 'Publication_Time_Afternoon',
 'Publication_Time_Evening',
 'Publication_Time_Morning',
 'Publication_Time_Night',
 'Episode_Sentiment_Negative',
 'Episode_Sentiment_Neutral',
 'Episode_Sentiment_Positive']

for i in dummy_cols:
    df_new[i] = np.where(df_new[i].astype('str')=='False',0,1)


df_new.head(5)


df_new.Publication_Time_Afternoon.nunique()
df_new.Publication_Time_Afternoon.value_counts()


df_new.info()


ep_no=[]
import re
for i in df_new.Episode_Title:
    no = re.split(r" |(?<![0-9])[.,](?![0-9])",i)[1]
    ep_no.append(no)

df_new['Episode_no'] = pd.Series(ep_no,index=None,dtype='float')



df_new.info()


df_new.drop(['Podcast_Name','Episode_Title'],inplace=True,axis=1)


df_new.info()


(df_new['Listening_Time_minutes']/df_new['Episode_Length_minutes']).median()





df_new.fillna({'Episode_Length_minutes':df_new['Listening_Time_minutes']/0.6983870967741935},inplace=True)


df_new.info()


df_new.Guest_Popularity_percentage.median()


df_new.fillna({'Guest_Popularity_percentage':df_new.Guest_Popularity_percentage.median()},inplace=True)
df_new.fillna({'Number_of_Ads':df_new.Number_of_Ads.median()},inplace=True)


df_new.info()


test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')


test_df.info()


dummy_columns = ['Genre','Publication_Day','Publication_Time','Episode_Sentiment']
test_df = pd.get_dummies(test_df,columns=dummy_columns)



test_df.columns.tolist()


test_dummy_cols = ['Genre_Business',
 'Genre_Comedy',
 'Genre_Education',
 'Genre_Health',
 'Genre_Lifestyle',
 'Genre_Music',
 'Genre_News',
 'Genre_Sports',
 'Genre_Technology',
 'Genre_True Crime',
 'Publication_Day_Friday',
 'Publication_Day_Monday',
 'Publication_Day_Saturday',
 'Publication_Day_Sunday',
 'Publication_Day_Thursday',
 'Publication_Day_Tuesday',
 'Publication_Day_Wednesday',
 'Publication_Time_Afternoon',
 'Publication_Time_Evening',
 'Publication_Time_Morning',
 'Publication_Time_Night',
 'Episode_Sentiment_Negative',
 'Episode_Sentiment_Neutral',
 'Episode_Sentiment_Positive']
for i in test_dummy_cols:
    test_df[i] = np.where(test_df[i].astype('str')=='False',0,1)


test_df.fillna({'Guest_Popularity_percentage':test_df.Guest_Popularity_percentage.median()},inplace=True)
test_df.fillna({'Number_of_Ads':test_df.Number_of_Ads.median()},inplace=True)



ep_no=[]
import re
for i in test_df.Episode_Title:
    no = re.split(r" |(?<![0-9])[.,](?![0-9])",i)[1]
    ep_no.append(no)

test_df['Episode_no'] = pd.Series(ep_no,index=None,dtype='float')


test_df.fillna({'Episode_Length_minutes':test_df['Episode_Length_minutes'].mean()*0.6983870967741935},inplace=True)


test_df[test_df.Episode_Length_minutes.isnull()]
# df_new['Listening_Time_minutes']


df_new.shape


test_df.shape


test_df.drop(['Podcast_Name','Episode_Title'],inplace=True,axis=1)


test_df.info()


X,y=df_new.drop(['id','Listening_Time_minutes'],axis=1),df_new['Listening_Time_minutes']
X_test=test_df.drop(['id'],axis=1)




from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X,y)
test_df['Listening_Time_minutes'] = model.predict(X_test)


model.score(X,y)


X_test.columns


y_pred = test_df['Listening_Time_minutes']


X_test.shape


test_df[['id','Listening_Time_minutes']]


model.score(X_test,y_pred)




