import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit



train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")


# Information about the DataFrame 'train_data'
train_data.info()


train_data["num_sold"] = train_data["num_sold"].fillna(0)
train_data.info()


# convert 'date' to datetime format for easier feature extraction¶
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])


# a special function for preprocessing date
def f_transf(df0):
    df = df0
    encoder = LabelEncoder()

    # Encode categorical features
    df['country_Encoded'] = encoder.fit_transform(df['country'])
    df['store_Encoded'] = encoder.fit_transform(df['store'])
    df['product_Encoded'] = encoder.fit_transform(df['product'])
    
    df['date'] = pd.to_datetime(df['date'])

    # Extract year, month, and day from the date
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day

    df['DayOfWeek'] = df['date'].dt.dayofweek
    df['IsWeekend'] = df['date'].dt.dayofweek > 4

    df['Hour'] = df['date'].dt.hour
    df['Minute'] = df['date'].dt.minute

    df['Quarter'] = df['date'].dt.quarter
    df['WeekOfYear'] = df['date'].dt.isocalendar().week

    df['Month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 7)

    reference_date = pd.Timestamp('2024-12-12') # it is may be any day
    df['DaysSinceReference'] = (df['date'] - reference_date).dt.days

    df['Season'] = df['date'].dt.month.map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})

    return df


# Feature engineering
train_data = f_transf(train_data)
test_data = f_transf(test_data)


features = ['country_Encoded', 'store_Encoded', 'product_Encoded', 'Year','Month','Day','DayOfWeek','IsWeekend','Hour','Minute','Quarter','WeekOfYear','Month_sin','Month_cos','DayOfWeek_sin','DayOfWeek_cos','DaysSinceReference','Season']
X = train_data[features].values
y = train_data['num_sold'].values


scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))


# Create TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)


# Initialize lists to store results
histories = []
predictions_list = []

# Iterate through the splits
for train_index, val_index in tscv.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y_scaled[train_index], y_scaled[val_index]
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    
    # Define input shape
    input_shape = (1, X_train.shape[2])
    
    # Create the LSTM model
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    histories.append(history)
    
    # Prepare test data for prediction
    X_test = test_data[features].values
    X_test_scaled = scaler_X.transform(X_test)
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Make predictions on the test set
    predictions_scaled = model.predict(X_test_reshaped)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    predictions_list.append(predictions)

# Average predictions from all folds
final_predictions = np.mean(predictions_list, axis=0)

# Add predictions to the test dataframe
test_data['num_sold'] = final_predictions

# Save the results to a CSV file
test_data[['id', 'num_sold']].to_csv('submission.csv', index=False)

print(final_predictions)




