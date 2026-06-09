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

sns.set_style('whitegrid')

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)


train_df


train_df.info()


train_df.isnull().sum()


train_df.duplicated().sum()


test_df.info()


train_df.isnull().sum()


for col in train_df.columns:
    print(f'{col}---> ',train_df[col].nunique())


train_df.describe()


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.histplot(data = train_df, x = "Listening_Time_minutes",kde = True)
plt.title('Listening Time minutes Distribution')
plt.subplot(1,2,2)
sns.boxplot(x=train_df["Listening_Time_minutes"])
plt.title('Listening Time minutes Distribution')
plt.tight_layout()
plt.show() 


categorical_col = ['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment','Number_of_Ads']

for col in categorical_col:
    plt.figure(figsize=(18,8))
    sns.boxplot(y=train_df["Listening_Time_minutes"], x = train_df[col], hue = train_df[col] )
    plt.title(f'Listening Time minutes Distribution according to {col}')
    plt.show()


numerical_col = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage']

for col in numerical_col:
    plt.figure(figsize=(15,10))
    sns.scatterplot(y=train_df["Listening_Time_minutes"], x = train_df[col])
    plt.title(f'Listening Time minutes Distribution according to {col}')
    plt.show()


for col in numerical_col:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[col], kde=True)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[col])
    plt.tight_layout()
    plt.show()


for col in categorical_col:
    plt.figure(figsize=(18,8))
    sns.boxplot(y=train_df["Episode_Length_minutes"], x = train_df[col], hue = train_df[col] )
    plt.title(f'Episode_Length_minutes Distribution according to {col}')
    plt.show()


for col in categorical_col:
    plt.figure(figsize=(18,8))
    sns.boxplot(y=train_df["Guest_Popularity_percentage"], x = train_df[col], hue = train_df[col] )
    plt.title(f'Guest_Popularity_percentage Distribution according to {col}')
    plt.show()


train_df.fillna(train_df.select_dtypes(include=['number']).median(), inplace=True)
test_df.fillna(test_df.select_dtypes(include=['number']).median(), inplace=True)


def remove_out(dff):
    mask = (dff['Episode_Length_minutes'] <= 150) & (dff['Number_of_Ads'] <= 20)
    return mask

df_cleaned = train_df[remove_out(train_df)]


df_cleaned.shape


df_cleaned.isnull().sum()


df_encoded = pd.get_dummies(df_cleaned, columns=['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment'])
df_encoded.columns


df_encoded_test = pd.get_dummies(test_df, columns=['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment'])
df_encoded_test.columns


df_encoded.head()


df_encoded = df_encoded.drop(columns= ['id'],axis=1)
df_encoded_test = df_encoded_test.drop(columns= ['id'],axis=1)


x = df_encoded.drop(columns = 'Listening_Time_minutes', axis = 1)
y = df_encoded['Listening_Time_minutes']


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size = 0.3,random_state = 42)


corr_matrix = df_encoded.corr()


print(corr_matrix['Listening_Time_minutes'].sort_values(ascending = False).to_string())


High_correlate = corr_matrix[corr_matrix >= 0.75]
print(High_correlate[High_correlate<1.0].stack().to_string())


from sklearn.preprocessing import  StandardScaler
sd = StandardScaler()

x_train_scaled = sd.fit_transform(x_train)
x_test_scaled = sd.transform(x_val)
test_scaled = sd.transform(df_encoded_test)


def model_acc(model):
    model.fit(x_train_scaled,y_train)
    acc = model.score(x_test_scaled,y_val)
    print(str(model)+'-->'+str(acc))


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc(lasso)


from sklearn.metrics import mean_squared_error

def rmse(model):
    y_pred = model.predict(x_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")


rmse(lr)


rmse(lasso)


import xgboost as xgb

train_data = xgb.DMatrix(x_train_scaled, label=y_train)
test_data = xgb.DMatrix(x_test_scaled, label=y_val)

params = {'bootstrap': True,
                   'criterion': 'gini', 
                   'max_depth': None, 
                   'max_features': 'auto', 
                   'min_samples_leaf': 1, 
                   'min_samples_split': 2, 
                   'n_estimators': 100}

xgb_model = xgb.train(params, train_data, num_boost_round=100)

y_pred = xgb_model.predict(test_data)

mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


test_dmatrix = xgb.DMatrix(test_scaled)
y_pred_test = xgb_model.predict(test_dmatrix)


submission_df = pd.DataFrame({ 'id': test_df['id'], 'Listening_Time_minutes': y_pred_test })


submission_df.head()

