import pandas as pd

#data_path
data_path = '/kaggle/input/bike-sharing-demand/'

train = pd.read_csv(data_path + 'train.csv')
test  = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sampleSubmission.csv')


# Extract bike demand data that it's not Heavy snow, Rain and Thunder 
train = train[train['weather'] != 4]


all_data_temp = pd.concat([train, test])
all_data_temp   # 17378 rows  * 12 features


all_data = pd.concat([train, test], ignore_index=True)  # ignore_index=True means Ignoring the index of the original data
all_data


from datetime import datetime
#date
all_data['date'] = all_data['datetime'].apply(lambda x : x.split()[0])

#year feature
all_data['year'] = all_data['datetime'].apply(lambda x : x.split()[0].split('-')[0])

#month feature
all_data['month'] = all_data['datetime'].apply(lambda x : x.split()[0].split('-')[1])

# hour
all_data['hour'] = all_data['datetime'].apply(lambda x : x.split()[1].split(':')[0])

# weekday
all_data['weekday'] = all_data['date'].apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())


all_data['datetime'] = pd.to_datetime(all_data['datetime'])  # change into datetime type

all_data['month'] = all_data['datetime'].dt.year  #year
all_data['year']  = all_data['datetime'].dt.month #month
all_data['hour']  = all_data['datetime'].dt.hour  #hour
all_data['weekday'] = all_data['datetime'].dt.weekday  #요일 


#Drop casual, registered feature
drop_features = ['casual', 'registered','datetime', 'date', 'month', 'windspeed']


all_data = all_data.drop(columns = drop_features, axis = 1)


X_train = all_data[~pd.isnull(all_data['count'])]
X_test  = all_data[pd.isnull(all_data['count'])]

#Drop target value of count features
X_train = X_train.drop(['count'], axis = 1)
X_test  = X_test.drop(['count'], axis = 1)

y = train['count']   #Target values 


X_train.head()


import numpy as np

def rmsle(y_true, y_pred, convertExp=True):
    #지수변환
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)

    #After logarithm converting replace missing values into 0
    log_true = np.nan_to_num(np.log(y_true + 1))  # replace NaN into 0
    log_pred = np.nan_to_num(np.log(y_pred + 1))

    #RMSLE Computation
    output = np.sqrt(np.mean((log_true - log_pred)**2))
    return output


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()


log_y = np.log(y)
linear_model.fit(X_train, log_y)


preds = linear_model.predict(X_train)


print(f'RMSLE Value : {rmsle(log_y, preds,True): .4f}')


linearreg_preds = linear_model.predict(X_test)  # Predict with Test data

submission['count'] = np.exp(linearreg_preds)  #exponential convert cuase current predicted values `log(count)`
submission.to_csv('submissioin.csv', index=False)

