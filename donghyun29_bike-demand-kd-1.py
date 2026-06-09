import pandas as pd
# 데이터 경로
data_path ='/kaggle/input/bike-sharing-demand/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sampleSubmission.csv')


train = train[train['weather'] !=4]


all_data = pd.concat([train, test], ignore_index=True)
all_data


from datetime import datetime

all_data['date'] = all_data['datetime'].apply(lambda x: x.split()[0])
all_data['year'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[0])
all_data['month'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[1])
all_data['hour'] = all_data['datetime'].apply(lambda x: x.split()[1].split(':')[0])
all_data['weekday'] = all_data['date'].apply(lambda dateString: datetime.strptime(dateString, "%Y-%m-%d").weekday())


drop_features = ['casual', 'registered', 'datetime', 'date', 'month', 'windspeed']
all_data = all_data.drop(drop_features, axis=1)


x_train = all_data[~pd.isnull(all_data['count'])]
x_test = all_data[pd.isnull(all_data['count'])]

x_train = x_train.drop(['count'], axis=1)
x_test = x_test.drop(['count'], axis=1)

y = train['count']
x_train.head()


import numpy as np
def rmsle(y_true, y_pred, convertExp=True):
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)

    log_true = np.nan_to_num(np.log(y_true+1))
    log_pred = np.nan_to_num(np.log(y_pred+1))

    output = np.sqrt(np.mean((log_true - log_pred)**2))
    return output


from sklearn.linear_model import LinearRegression
import numpy as np

linear_reg_model = LinearRegression()

log_y = np.log(y)
linear_reg_model.fit(x_train, log_y)


preds = linear_reg_model.predict(x_train)
print (f'선형 회귀의 RMSLE 값 : {rmsle(log_y, preds, True):.4f}')


linearreg_preds = linear_reg_model.predict(x_test) #using test data to predict

submission['count'] = np.exp(linearreg_preds)
submission.to_csv('submission.csv', index=False)

