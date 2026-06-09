import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# Load datasets
df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
print('Datasets loaded!')


# Show info about the train dataset
df_train = df_train.dropna()
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['date_numeric'] = df_train['date'].map(pd.Timestamp.toordinal)
df_train['dayofweek'] = df_train['date'].dt.dayofweek
df_train['month'] = df_train['date'].dt.month
df_train['dayofyear'] = df_train['date'].dt.dayofyear
print(df_train.info())
print(df_train.describe())

print(df_train.head())


# Show different product names and numbers
print(df_train.groupby('product')['id'].count())

# Let's convert the products into numbers
df_train = df_train.replace('Kaggle', '0')
df_train = df_train.replace('Kaggle Tiers', '1')
df_train = df_train.replace('Kerneler', '2')
df_train = df_train.replace('Kerneler Dark Mode', '3')
df_train = df_train.replace('Holographic Goose', '4')
df_train['product'] = df_train['product'].astype(int)

print('\n..after transformation..')
# Show different product names and numbers
print(df_train.groupby('product')['id'].count())
print(df_train.info())


# Let's do the same for df_test too!
print('\n\n-------------')
df_test['date'] = pd.to_datetime(df_test['date'])
df_test['date_numeric'] = df_test['date'].map(pd.Timestamp.toordinal)
df_test['dayofweek'] = df_test['date'].dt.dayofweek
df_test['month'] = df_test['date'].dt.month
df_test['dayofyear'] = df_test['date'].dt.dayofyear
df_test = df_test.replace('Kaggle', '0')
df_test = df_test.replace('Kaggle Tiers', '1')
df_test = df_test.replace('Kerneler', '2')
df_test = df_test.replace('Kerneler Dark Mode', '3')
df_test = df_test.replace('Holographic Goose', '4')
df_test['product'] = df_test['product'].astype(int)

print('\n..after transformation..')
print(df_test.info())


# Filtering the df_train for only using sticker 0
df_train_0 = df_train

# Define X and y for both train and test
X = df_train_0[['date_numeric', 'dayofweek', 'month', 'dayofyear', 'product']]
y = df_train_0['num_sold']

# Split data for validation (80% train, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize X and y
scaler_X = StandardScaler() 
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
scaler_y = StandardScaler() 
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

print(f'X shape: {X_train.shape}')
print(f'y shape: {y_train.shape}')


# Create the model
model = tf.keras.Sequential([

    # Define the input shape
    tf.keras.Input(shape=(X_train.shape[1],)),

    # Flatten the input
    tf.keras.layers.Flatten(),

    # Add Hidden, Dense layers
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),

    # Add the output layer
    tf.keras.layers.Dense(1)
    ])


model.summary()


# Compile the model and use accuracy as metrics
model.compile(loss='mae', optimizer=Adam(learning_rate=0.001), metrics=['mse'])

# Define EarlyStopping to prevent to continue after it's not improving anymore (so less chance of OverFitting)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X, y, epochs=10, batch_size=64 ,validation_data=(X_test, y_test), callbacks=[early_stopping])


plt.figure(figsize=(10, 6))
plt.scatter(df_train_0['date_numeric'], df_train_0['num_sold'], alpha=0.5)
plt.xlabel('Date Numeric')
plt.ylabel('Num Sold')
plt.title('Num Sold vs Date Numeric')
plt.show()


# Create predictions
predictions = model.predict(df_test[['date_numeric', 'dayofweek', 'month', 'dayofyear', 'product']])
predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Submission
submission = pd.DataFrame({
    'id': df_test['id'], 
    'num_sold': predictions  
})

submission.to_csv('submission.csv', index = False)

