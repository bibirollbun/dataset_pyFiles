

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.metrics import mean_absolute_percentage_error
from keras.optimizers import Adam



import os
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e1'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train.drop(columns = 'id', inplace = True)
test.drop(columns = 'id', inplace = True)


train.head()


train.isna().sum()


train.dropna(inplace = True)


train.duplicated().sum()


train.dtypes


train.date= pd.to_datetime(train.date)
test.date = pd.to_datetime(test.date)


train['year']= train['date'].dt.year
train['month']= train.date.dt.month
train['day']= train.date.dt.day
test['year']= test.date.dt.year
test['month']= test.date.dt.month
test['day']= test.date.dt.day


train.drop(columns = 'date', inplace = True)
test.drop(columns = 'date', inplace = True)


cats = ['country', 'store', 'product']
label_encoders = {}
for i in cats:
    le = LabelEncoder()
    train[i] = le.fit_transform(train[i])
    label_encoders[i] = le
    test[i]= label_encoders[i].transform(test[i])



y= train['num_sold']
train = train.drop(columns = ['num_sold'])


train.head()


trainx,testx,trainy,testy= train_test_split(train,y,test_size=0.2, random_state=42)
scaler = MinMaxScaler()
trainx_scaled= scaler.fit_transform(trainx)
testx_scaled= scaler.transform(testx)
std = StandardScaler()
trainy_scaled = std.fit_transform(trainy.values.reshape(-1,1))
testy_scaled = std.transform(testy.values.reshape(-1,1))


trainx_reshaped = trainx_scaled.reshape(trainx_scaled.shape[0],1,trainx_scaled.shape[1])
testx_reshaped = testx_scaled.reshape(testx_scaled.shape[0],1,testx_scaled.shape[1])


gru_model = Sequential()
gru_model.add(GRU(64, activation='relu',return_sequences = True,  input_shape=(1, trainx_reshaped.shape[2])))
gru_model.add(GRU(8, activation='relu', return_sequences=True))
gru_model.add(GRU(8, activation='relu', return_sequences=False))
gru_model.add(Dense(1))




gru_model.compile(optimizer='adam', loss='mse')



gru_model.fit(trainx_reshaped, trainy_scaled, epochs=50, batch_size=32, validation_data=(testx_reshaped, testy_scaled))



gru_preds = gru_model.predict(testx_reshaped)
gru_mape = mean_absolute_percentage_error(testy_scaled, gru_preds)



test_scaled = scaler.transform(test)
test_reshaped= test_scaled.reshape(test_scaled.shape[0],1,test_scaled.shape[1])
gru_test_pred = gru_model.predict(test_reshaped)
# Inverse transform the predictions to get back to the original scale
gru_test_pred_original = std.inverse_transform(gru_test_pred)



sub=pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sub['num_sold'] = gru_test_pred_original
sub.to_csv('submission.csv', index=False)


