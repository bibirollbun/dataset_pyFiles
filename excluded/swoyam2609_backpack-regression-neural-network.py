import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


df_train.shape, df_test.shape


df_train.head()


df_train.isnull().sum()


df_test.isnull().sum()


df_train['Color'].unique()


df_train['Color'].fillna("White", inplace=True)
df_test['Color'].fillna("White", inplace=True)


df_train['Brand'].unique()


df_train['Brand'].fillna("Zudio", inplace=True)
df_test['Brand'].fillna("Zudio", inplace=True)


# Group by Brand and fill null values with mode of Material within each brand
df_train['Material'] = df_train.groupby('Brand')['Material'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['Material'] = df_test.groupby('Brand')['Material'].transform(lambda x: x.fillna(x.mode()[0]))


# Group by Brand and fill null values with mode of Size within each brand
df_train['Size'] = df_train.groupby('Brand')['Size'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['Size'] = df_test.groupby('Brand')['Size'].transform(lambda x: x.fillna(x.mode()[0]))


# Group by Brand and fill null values with mode of Laptop Compartment within each brand
df_train['Laptop Compartment'] = df_train.groupby('Brand')['Laptop Compartment'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['Laptop Compartment'] = df_test.groupby('Brand')['Laptop Compartment'].transform(lambda x: x.fillna(x.mode()[0]))


# Group by Brand and fill null values with mode of Waterproof within each brand
df_train['Waterproof'] = df_train.groupby('Brand')['Waterproof'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['Waterproof'] = df_test.groupby('Brand')['Waterproof'].transform(lambda x: x.fillna(x.mode()[0]))


# Group by Brand and fill null values with mode of Style within each brand
df_train['Style'] = df_train.groupby('Brand')['Style'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['Style'] = df_test.groupby('Brand')['Style'].transform(lambda x: x.fillna(x.mode()[0]))


# Group by Brand and Style, then fill null values with mean of Weight Capacity (kg)
df_train['Weight Capacity (kg)'] = df_train.groupby(['Brand', 'Style'])['Weight Capacity (kg)'].transform(lambda x: x.fillna(x.mean()))
df_test['Weight Capacity (kg)'] = df_test.groupby(['Brand', 'Style'])['Weight Capacity (kg)'].transform(lambda x: x.fillna(x.mean()))


df_train.head()


# Create bins with 5kg intervals
bins = np.arange(df_train['Weight Capacity (kg)'].min(), 
                 df_train['Weight Capacity (kg)'].max() + 5, 
                 5)

# Create a new column with the binned values
df_train['Weight_Bins'] = pd.cut(df_train['Weight Capacity (kg)'], bins=bins)

# Calculate mean price for each bin
weight_price_analysis = df_train.groupby('Weight_Bins')['Price'].mean().round(2)
print(weight_price_analysis)


categorical_columns = df_train.columns


categorical_columns = categorical_columns[1:-3]


# One-hot encode categorical columns for both train and test datasets
df_train_encoded = pd.get_dummies(df_train[categorical_columns], drop_first=True)
df_test_encoded = pd.get_dummies(df_test[categorical_columns], drop_first=True)

# Keep only the common columns between train and test
common_columns = df_train_encoded.columns.intersection(df_test_encoded.columns)

# Align the encoded columns for both datasets
df_train_encoded = df_train_encoded[common_columns]
df_test_encoded = df_test_encoded[common_columns]

# Add back the non-categorical columns to train dataset
df_train_final = pd.concat([
    df_train[['Weight Capacity (kg)', 'Price']], 
    df_train_encoded
], axis=1)

# Add back the non-categorical columns to test dataset
df_test_final = pd.concat([
    df_test[['id','Weight Capacity (kg)']], 
    df_test_encoded
], axis=1)


df_train_final.head()


df_test_final.head()


from sklearn.model_selection import train_test_split


x_train, x_val, y_train, y_val = train_test_split(
    df_train_final.drop('Price', axis=1), 
    df_train_final['Price'], 
    test_size=0.2, 
    random_state=42
)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU


model = Sequential([
    Dense(4096, input_shape=(x_train.shape[1],)),  
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.4),

    Dense(2048),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.35),

    Dense(1024),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),

    Dense(512),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.25),

    Dense(256),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.2),

    Dense(128),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.15),

    Dense(32),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.1),

    Dense(16),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.1),

    Dense(8),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.05),

    Dense(4),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),

    # Output layer for regression
    Dense(1, activation='linear')
])


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Compile the model with RMSE
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=[rmse])


model.fit(x_train, y_train, epochs=5, batch_size=16, validation_data=(x_val, y_val))


finally_predictions = model.predict(df_test_final.drop('id', axis=1))


finally_predictions = pd.DataFrame(finally_predictions, columns=['price'])  # Ensure 'price' column exists
finally_predictions['id'] = df_test_final['id'].values  # Add 'id' column
finally_predictions = finally_predictions[['id', 'price']] 


finally_predictions.head()


finally_predictions.to_csv('submission.csv', index=False)




