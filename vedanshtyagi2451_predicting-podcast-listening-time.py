import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df2 = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


print(df.dtypes,"\n\n\n")
print(df2.dtypes)


# label encode Podcast_Name, Episode_Title, Genre, Publication_Day, Publication_Time, Episode_Sentiment from df and df2 both

from sklearn.preprocessing import LabelEncoder

# Categorical columns to encode
cols_to_encode = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Dictionary to store label encoders
encoders = {}

# Loop through each column and encode
for col in cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))      # Fit & transform on training data
    df2[col] = le.transform(df2[col].astype(str))         # Only transform on test data
    encoders[col] = le                                    # Save encoder for future use


from sklearn.preprocessing import StandardScaler

# Define features to scale (all except target column)
features_to_scale = df.columns.drop('Listening_Time_minutes')

# Initialize the scaler
scaler = StandardScaler()

# Fit on training data, transform both train and test
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
df2[features_to_scale] = scaler.transform(df2[features_to_scale])


print(df.dtypes,"\n\n\n")
print(df2.dtypes)


from sklearn.model_selection import train_test_split

df_clean = df.dropna()
X = df_clean.drop('Listening_Time_minutes', axis=1)
y = df_clean['Listening_Time_minutes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


df = df.dropna()


df.shape


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

# Split input features and target
X = df.drop('Listening_Time_minutes', axis=1)
y = df['Listening_Time_minutes']

# Optional: split train into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
from tensorflow.keras import regularizers

# Build the model with L2 regularization
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu',
          kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(32, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(16, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)),
    Dense(1)
])

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Compile with RMSE as metric
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', rmse])
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


# Ensure the df2 has no NaNs and all features are scaled
# Also assume df2 includes the 'id' column (from your dataset)

# Extract features from df2 (excluding target)
X_test = df2

# Predict using the trained model
predictions = model.predict(X_test).flatten()

# Create submission DataFrame
submission = pd.DataFrame({
    'id': df2['id'].values,
    'Listening_Time_minutes': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv saved successfully.")




