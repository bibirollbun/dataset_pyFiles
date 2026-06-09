import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

%matplotlib inline


df1=pd.read_excel("/kaggle/input/rainfall-prediction-challenge-2025/Data Aktual.xlsx")
df2=pd.read_excel("/kaggle/input/rainfall-prediction-challenge-2025/Data Input Hybrid.xlsx")


df1.head()


df1.shape


df1.isnull().sum()


df1.info()


df2.head()


df2.shape


df2.info()


df2.head()


df2['Tanggal'] = pd.to_datetime(df2['Tanggal'], format='%d-%b-%Y')
df2['Day'] = df2['Tanggal'].dt.day
df2['Month'] = df2['Tanggal'].dt.month
df2['Year'] = df2['Tanggal'].dt.year


df2.drop(columns=["Tanggal"],axis=1,inplace=True)


df2.head()


input_cols = ['w1', 'w2', 'w3', 'w4', 'ehat1', 'ehat2', 'ehat3', 'ehat4']
target_cols = ['eresid1', 'eresid2', 'eresid3', 'eresid4']

scaler=MinMaxScaler()
df2[input_cols]=scaler.fit_transform(df2[input_cols])


df2.head()


X = df2[input_cols]
y = df2[target_cols]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,random_state=42)


base_model = CatBoostRegressor(iterations=1000,learning_rate=0.005,depth=6,verbose=500)

multi_model = MultiOutputRegressor(base_model)  

multi_model.fit(X_train, y_train)


y_true_all = y_test[target_cols].values.flatten()
y_pred = multi_model.predict(X_test)

y_pred_all = y_pred.flatten()

rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
mae_all = mean_absolute_error(y_true_all, y_pred_all)
r2_all = r2_score(y_true_all, y_pred_all)
mape_all = (np.abs(y_true_all - y_pred_all) / np.maximum(y_true_all, 1e-6)).mean() * 100


print(f"RMSE : {rmse_all:.4f}")
print(f"MAE  : {mae_all:.4f}")
print(f"R2   : {r2_all:.4f}")
print(f"MAPE : {mape_all:.2f}%")


# Convert DataFrame to NumPy array before reshaping
X_train_np = X_train.values
X_test_np  = X_test.values

# Reshape for LSTM: (samples, timesteps, features)
# Here timesteps = 1
X_train_3d = X_train_np.reshape((X_train_np.shape[0], 1, X_train_np.shape[1]))
X_test_3d  = X_test_np.reshape((X_test_np.shape[0], 1, X_test_np.shape[1]))

print(f"X_train shape: {X_train_3d.shape}")
print(f"X_test shape:  {X_test_3d.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")



from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

n_features = X_train_3d.shape[2]
n_outputs  = y_train.shape[1]

model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(X_train_3d.shape[1], n_features)),
    Dropout(0.2),
    BatchNormalization(),
    
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dense(n_outputs)  
])


optimizer = Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mean_squared_error'])

model.summary()



from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=10,min_lr=1e-6,verbose=1)


history = model.fit(X_train_3d, y_train,epochs=150,batch_size=64,validation_data=(X_test_3d, y_test),verbose=2, callbacks=[early_stop, reduce_lr])


y_pred = model.predict(X_test_3d)
rmse_overall = np.sqrt(mean_squared_error(y_test, y_pred))
mae_overall  = mean_absolute_error(y_test, y_pred)
r2_overall   = r2_score(y_test, y_pred)
mape_overall = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("Overall Metrics:")
print(f"RMSE : {rmse_overall:.4f}")
print(f"MAE  : {mae_overall:.4f}")
print(f"R2   : {r2_overall:.4f}")
print(f"MAPE : {mape_overall:.2f}%")


X_submission = df2[input_cols].iloc[-100:].values
X_submission_3d = X_submission.reshape((X_submission.shape[0], 1, X_submission.shape[1]))

y_pred_submission = model.predict(X_submission_3d)

submission_df = pd.DataFrame({
    'id': range(len(y_pred_submission)),
    'Y1': y_pred_submission[:, 0],
    'Y2': y_pred_submission[:, 1],
    'Y3': y_pred_submission[:, 2],
    'Y4': y_pred_submission[:, 3]
})

submission_df.to_csv("lstm_submission.csv", index=False)



df=pd.read_csv("/kaggle/working/lstm_submission.csv")


df.head()




