# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_raw = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
le = LabelEncoder()
scaler = MinMaxScaler()
not_features = ['id','listening_time_minutes','episode_title','publication_day','publication_time','number_of_ads']


def pre_process(df):
    
    df= df.map(lambda s:s.lower() if type(s) == str else s) # convert all string column values to lower
    df.columns = [c.replace(' ', '_').lower() for c in df] # convert all column names to lower and replace space with _

    encode_columns = ['genre','publication_day','publication_time']
    
    for col in encode_columns:   
        df[col] = le.fit_transform(df[col])
    
    df['episode_sentiment'] = df['episode_sentiment'].map({'positive':1,
                                                                       'neutral' : 0, 
                                                                       'negative':-1
                                                                      })
    
    
    df['podcast_name'] = df['podcast_name'].astype('category').cat.codes
    
    df['episode_title'] = df['episode_title'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
    df['episode_title'] = pd.to_numeric(df['episode_title'], errors='coerce')

    df = df.fillna(df.mean())

    scaled = scaler.fit_transform(df)

    df = pd.DataFrame(scaled, columns=df.columns)

    return df



train_df = pre_process(train_raw)

X = train_df.drop(not_features,axis = 1)
y = train_df[['listening_time_minutes']]


X.info()


for col in X.columns:
    unique = X[col].unique()
    if len(unique) >= 2:
        print(col,end = ': ')
        print(unique)



X_auto_train, X_auto_test, y_train, y_test = train_test_split(X, y, test_size=0.50,random_state=16)

y_auto_train = np.array(y_train)
y_auto_test = np.array(y_test)


# print(automl.model.estimator)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=0.95)  # keep 95% of variance
# X_auto_train = pca.fit_transform(X_auto_train)
# X_auto_test = pca.fit_transform(X_auto_test)



from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor


r1 = LinearRegression()
r2 = RandomForestRegressor(random_state=1 , bootstrap = False)
r3 = KNeighborsRegressor()
r4 = SGDRegressor(random_state=11)
r5 = AdaBoostRegressor(random_state=11)
vr = VotingRegressor(
    [
        ('lr', r1), 
        ('rf', r2), 
        ('r3', r3),
        ('r4', r4)
        ,('r5', r5)
    ])

vr.fit(X_auto_train, y_train.values.ravel())
print(vr.score(X_auto_train, y_train.values.ravel())) 


y_pred = vr.predict(X_auto_test)
print("Validation :", np.sqrt(mean_squared_error(y_test, y_pred)))



test_raw = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_id = test_raw[['id']]


test_df = pre_process(test_raw)

X_test = test_df.drop(not_features,axis = 1,errors='ignore')
y_test = vr.predict(X_test)



op=pd.DataFrame()
op['id']=test_id
op['Listening_Time_minutes']=y_test

op.to_csv("submission.csv",index=False)
print('File created')

