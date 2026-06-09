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


train_path = "/kaggle/input/playground-series-s5e1/train.csv"
test_path = "/kaggle/input/playground-series-s5e1/test.csv"


train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


train_df= pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)


train_df.head()


test_df.head()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('whitegrid')

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import warnings
warnings.filterwarnings('ignore')


train_df.info()


test_df.info()


train_df.isnull().sum()


test_df.isnull().sum()


train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


train_df.groupby('country')['num_sold'].mean()


train_df.groupby('store')['num_sold'].mean()


train_df.groupby('product')['num_sold'].mean()


train_df['num_sold'] = train_df.groupby('country')['num_sold'].transform(lambda x: x.fillna(x.mean()))


train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.dayofweek


test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.dayofweek


train_df.head()


train_df["holiday"] = 0
test_df["holiday"] = 0


train_df["country"].unique()
test_df["country"].unique()


import holidays

ca_holidays = holidays.country_holidays('CA') # Canada
fi_holidays = holidays.country_holidays('FI') # Finland
it_holidays = holidays.country_holidays('IT') # Italy
ke_holidays = holidays.country_holidays('KE') # Kenya
no_holidays = holidays.country_holidays('NO') # Norway
sg_holidays = holidays.country_holidays('SG') # Singapore


def set_holiday(row):
    VAL_HOLIDAY = 1
    if row["country"] == "Canada" and row["date"] in ca_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Finland" and row["date"] in fi_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Italy" and row["date"] in it_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Kenya" and row["date"] in ke_holidays:
        row["holiday"] = VAL_HOLIDAY
    
    elif row["country"] == "Norway" and row["date"] in no_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Singapore" and row["date"] in sg_holidays:
        row["holiday"] = VAL_HOLIDAY

    return row


df_train = train_df.apply(set_holiday, axis=1)
df_test = test_df.apply(set_holiday, axis=1)


df_train


df_train_encoded = pd.get_dummies(df_train, columns=['country','store','product'])
df_test_encoded = pd.get_dummies(df_test, columns=['country','store','product'])


def periodic_transform(dff,variable):
    dff[f"{variable}_SIN"] = np.sin(dff[variable] / dff[variable].max()*2*np.pi)
    dff[f"{variable}_COS"] = np.cos(dff[variable] / dff[variable].max()*2*np.pi)
    return dff


cyclic_col = ['month','day','day_of_week']

for col in cyclic_col:
    df_train_final = periodic_transform(df_train_encoded, col)
    df_test_final = periodic_transform(df_test_encoded, col)


df_train_final.columns


df_test_final.columns


df_train_final = df_train_final.drop(['month', 'day', 'day_of_week', 'date', 'id'], axis = 1)
df_test_final = df_test_final.drop(['month', 'day', 'day_of_week', 'date', 'id'], axis = 1)


numeric_df = df_train_final.select_dtypes(include = ['number'])
corr_matrix = numeric_df.corr()


print(corr_matrix['num_sold'].sort_values(ascending = False).to_string())


plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,annot=True,cmap = 'coolwarm', fmt = ".2f")
plt.show()


x = df_train_final.drop(['num_sold'],axis =1)
y = df_train_final['num_sold']


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=42)


from sklearn.preprocessing import MinMaxScaler


mm = MinMaxScaler()
x_train_scaled = mm.fit_transform(x_train)
x_test_scaled = mm.transform(x_test)


df_test_scaled_final = mm.transform(df_test_final)


import lightgbm as lgb


train_data = lgb.Dataset(x_train_scaled, label=y_train)
test_data = lgb.Dataset(x_test_scaled, label=y_test, reference=train_data)


params = {
    'objective': 'regression',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9
}


callbacks = [lgb.early_stopping(stopping_rounds=10)]
num_round = 100
model = lgb.train(
    params,
    train_data,
    num_boost_round=num_round,
    valid_sets=[test_data],
    callbacks=callbacks
)


y_pred = model.predict(x_test_scaled, num_iteration=model.best_iteration)


y_test = y_test.values.flatten()
y_pred = y_pred.flatten()
final_df1 = pd.DataFrame(np.hstack((y_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error


print(f'MAE: {mean_absolute_error(final_df1["Prediction"],final_df1["Real"])}')
print(f'MSE: {mean_squared_error(final_df1["Prediction"],final_df1["Real"])}')
print(f'RMSE: {np.sqrt(mean_squared_error(final_df1["Prediction"],final_df1["Real"]))}')
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred)}')


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_df1['Real'])) ,y=final_df1['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_df1['Prediction'])),y=final_df1['Prediction'],color='red',label='Prediction')
ax.set_xlim([3000,3100])
plt.title('Real vs. Predictions for Decision Tree')
plt.show()


y_pred = model.predict(df_test_scaled_final, num_iteration=model.best_iteration)


submission_df = pd.DataFrame({
    'id': df_test_encoded['id'],
    'Premium Amount': y_pred
})
submission_df

