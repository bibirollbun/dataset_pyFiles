import pandas as pd 
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')


df.describe(include ='all')


df.info()


df = df.drop(['Id','Date'],axis=1)
test = test.drop(['Date','ForecastId'],axis=1)


def encode_columns(df, cols_encoder):
    le = LabelEncoder()
    for col in cols_encoder:
        df[col] = le.fit_transform(df[col])
    return df

cols_encoder = ['County', 'Province_State','Country_Region']

df = encode_columns(df, cols_encoder)
test = encode_columns(test, cols_encoder)


def encode_categorical_with_dummies(df):
    categorical_cols = df.select_dtypes(exclude='number').columns
    df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_dummies

df = encode_categorical_with_dummies(df)
test = encode_categorical_with_dummies(test)


def remove_outliers(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            df[col] = df[col].apply(
                lambda x: x if pd.isnull(x) or (lower_limit <= x <= upper_limit) else None
            )
    
    return df

df = remove_outliers(df)
test = remove_outliers(test)


def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':  
            df[column] = df[column].fillna(df[column].mode()[0])  # Fill with mode
        else:
            df[column] = df[column].fillna(df[column].mean())     # Fill with mean
    return df

df = fill_missing_values(df)
test = fill_missing_values(test)


X = df.drop('TargetValue', axis=1)
y = df['TargetValue']

y_test = test


model = Sequential([
    Dense(6, input_dim=6, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1) 
])

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.000001),
    loss='mse',
    metrics=['mae']
)

history = model.fit(X, y, epochs=15, batch_size=32, validation_split=0.3, verbose=1)



start_epoch = 1

mae = history.history['mae'][start_epoch:]
val_mae = history.history['val_mae'][start_epoch:]
loss = history.history['loss'][start_epoch:]
val_loss = history.history['val_loss'][start_epoch:]
epochs_range = range(start_epoch + 1, len(loss) + start_epoch + 1)

# Criando os gráficos
plt.figure(figsize=(18, 6))

# MAE
plt.subplot(1, 2, 1)
plt.plot(epochs_range, mae, label='Training MAE')
plt.plot(epochs_range, val_mae, label='Validation MAE')
plt.legend(loc='upper right')
plt.title('Training and Validation MAE')

# (loss)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()


y_test = test
y_test = model.predict(test).flatten()

submission = pd.DataFrame({
    'Target_Fatalities': test['Target_Fatalities'],
    'TargetValue': y_test
})
submission.to_csv('submission', index=False)

