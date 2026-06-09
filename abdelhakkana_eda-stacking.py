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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OrdinalEncoder,OneHotEncoder


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


print(train.shape)
print(test.shape)


train.head()


train.info()


train.isnull().sum()


test.isnull().sum()


train['Podcast_Name'].describe()


train['Podcast_Name'].value_counts()


train['Episode_Title'].describe()


train['Episode_Title'].value_counts()


train['Episode_Length_minutes'].describe()


plt.scatter(train['Episode_Length_minutes'],train['Listening_Time_minutes'])


#Remove Outliers(short and long podcasts)
df=train[((train['Episode_Length_minutes'])>15 & (train['Episode_Length_minutes']<125)) ]


df


train['Genre'].describe()


train['Genre'].value_counts()


#Remove Outliers(near 0 and large popularity hosts)
train['Host_Popularity_percentage'].describe()


df1=df[df['Host_Popularity_percentage'] <= 100 ]


df1


train['Publication_Day'].describe()


train['Publication_Day'].value_counts()


train['Publication_Time'].describe()


train['Publication_Time'].value_counts()


train['Guest_Popularity_percentage'].describe()


#Remove Outliers(near 0 and large popularity guests)
df2=df1[df1['Guest_Popularity_percentage']<=100]


df2


train['Number_of_Ads'].describe()


#Remove Outliers(large amount of ads)
df3=df2[df2['Number_of_Ads']<3]


df3


train['Episode_Sentiment'].describe()


train['Episode_Sentiment'].value_counts()


train['Listening_Time_minutes'].describe()


df4=df3[df3['Listening_Time_minutes']>3]


df4


tra=df4


tra['Host_popularity_over_guest']=tra['Host_Popularity_percentage']/tra['Guest_Popularity_percentage']
tra['Guest_popularity_over_host']=tra['Guest_Popularity_percentage']/tra['Host_Popularity_percentage']
tra['Average_Podcasters_popularity']=(tra['Guest_Popularity_percentage']+tra['Host_Popularity_percentage'])/2
tra['Ads']=tra['Number_of_Ads'].apply(lambda x:1 if (x > 0) else 0)
tra['Ads_per_Podcasters_popularity']=tra['Ads']*tra['Average_Podcasters_popularity']
tra['Ads_per_minute']=tra['Number_of_Ads']/tra['Episode_Length_minutes']

tra['Is_Weekend']=tra['Publication_Day'].apply(lambda x:1 if ((x=='Saturday')&(x=='Sunday'))  else 0)
tra['Day_Night']=tra['Publication_Time'].apply(lambda x:1 if ((x=='Morning') & (x=='Afternoon'))  else 0)


X=tra.drop(['id','Podcast_Name','Episode_Title','Listening_Time_minutes'],axis=1)
y=tra['Listening_Time_minutes']
df5=X


X


categorical=[i for i in X if X[i].dtype == 'object']
numerical=[j for j in X if X[j].dtype in ['float64']]


categorical_transformer=OrdinalEncoder()
numerical_transformer=SimpleImputer(strategy='mean')
preprocessor=ColumnTransformer(transformers=[('num',numerical_transformer,numerical),('cat',categorical_transformer,categorical)])


X.replace([np.inf, -np.inf], np.nan, inplace=True)
X=preprocessor.fit_transform(X)


sc=StandardScaler()
st=MinMaxScaler()


X1=sc.fit_transform(X)
X2=st.fit_transform(X)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01,random_state=11)
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y,test_size=0.01,random_state=11)
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y,test_size=0.01,random_state=11)


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor,VotingRegressor,StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


r1=LinearRegression()
r1.fit(X_train,y_train)
y1_pred=r1.predict(X_test)
np.sqrt(mean_squared_error(y_test,y1_pred))


r2=HistGradientBoostingRegressor(random_state=33)
r2.fit(X_train,y_train)
y2_pred=r2.predict(X_test)
np.sqrt(mean_squared_error(y_test,y2_pred))


r3=CatBoostRegressor(random_state=33,verbose=0)
r3.fit(X_train,y_train)
y3_pred=r2.predict(X_test)
np.sqrt(mean_squared_error(y_test,y3_pred))


r4=LGBMRegressor(random_state=33,verbose=0)
r4.fit(X_train,y_train)
y4_pred=r4.predict(X_test)
np.sqrt(mean_squared_error(y_test,y4_pred))


r5=XGBRegressor(random_state=33,n_jobs=-1)
r5.fit(X_train,y_train)
y5_pred=r5.predict(X_test)
np.sqrt(mean_squared_error(y_test,y5_pred))


r=VotingRegressor(estimators=[('hist',r2),('cb',r3),('lgb',r4),('xgb',r5)])
r.fit(X_train,y_train)
y_pred=r.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


rr=StackingRegressor(estimators=[('cb',r3),('xgb',r5)],final_estimator=r4)
rr.fit(X_train,y_train)
y_pred=rr.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


test


test['Host_popularity_over_guest']=test['Host_Popularity_percentage']/test['Guest_Popularity_percentage']
test['Guest_popularity_over_host']=test['Guest_Popularity_percentage']/test['Host_Popularity_percentage']
test['Average_Podcasters_popularity']=(test['Guest_Popularity_percentage']+test['Host_Popularity_percentage'])/2
test['Ads']=test['Number_of_Ads'].apply(lambda x:1 if x>0  else 0)
test['Ads_per_Podcasters_popularity']=test['Ads']*test['Average_Podcasters_popularity']
test['Ads_per_minute']=test['Number_of_Ads']/test['Episode_Length_minutes']

test['Is_Weekend']=test['Publication_Day'].apply(lambda x:1 if ((x=='Saturday')&(x=='Sunday'))  else 0)
test['Day_Night']=test['Publication_Time'].apply(lambda x:1 if ((x=='Morning') & (x=='Afternoon'))  else 0)


testt=test[df5.columns]


categorical=[i for i in testt if testt[i].dtype == 'object']
numerical=[j for j in testt if testt[j].dtype in ['float64']]


testt.replace([np.inf, -np.inf], np.nan, inplace=True)

testt=preprocessor.transform(testt)


testt1=sc.transform(testt)
testt2=st.transform(testt)


predictions=rr.predict(testt)
predictions


output=pd.DataFrame({'id':test.id,'Listening_Time_minutes':predictions})
output


submission=output.to_csv('submission.csv',index=False)




