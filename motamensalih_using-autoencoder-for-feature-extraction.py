import kagglehub
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Download orignal dataset
path = kagglehub.dataset_download("ruchikakumbhar/calories-burnt-prediction")


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


main_dir = '/kaggle/input/playground-series-s5e5/'
train_url ='train.csv'
test_url = 'test.csv'

train_path = os.path.join(main_dir, train_url)
test_path = os.path.join(main_dir, test_url)

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
original_data = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')


train_data.shape, test_data.shape, original_data.shape


train_data.head()


original_data.head()


print(train_data.isnull().sum(axis = 0))


print(test_data.isnull().sum(axis = 0))


print(original_data.isnull().sum(axis = 0))


train_data.columns, test_data.columns, original_data.columns


train_data.drop(columns=['id'], inplace = True)
original_data.drop(columns=['User_ID'], inplace = True)
train_data.columns, original_data.columns


original_data.rename(columns={'Gender': 'Sex'}, inplace=True)


full_data = pd.concat([train_data, original_data], ignore_index=True)
full_data.head()


le = LabelEncoder()
full_data['Sex'] = le.fit_transform(full_data['Sex'].astype(str))
full_data.shape


plt.figure(figsize=(8, 8))
sns.heatmap(full_data.corr(),
           annot=True,
           cbar=False)
plt.show()


X = full_data.drop(columns=['Calories'], axis=1)
Y = full_data['Calories']

X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)


X_train.shape, X_test.shape


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = XGBRegressor()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Compute RMSLE
y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
y_test_clipped = np.clip(y_test, a_min=0, a_max=None)
rmsle = np.sqrt(mean_squared_log_error(y_test_clipped, y_pred_clipped))
print(f'XGB Regressor RMSLE: {rmsle:.4f}')


input_dim = X_train_scaled.shape[1]
encoding_dim = 128  # size of the bottleneck layer

# Encoder
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(32, activation='linear')(input_layer)
encoded = layers.Dense(64, activation='linear')(encoded)
encoded = layers.Dense(encoding_dim, activation='linear')(encoded)

# Decoder
decoded = layers.Dense(64, activation='linear')(encoded)
decoded = layers.Dense(32, activation='linear')(decoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

# Autoencoder model
autoencoder = models.Model(inputs=input_layer, outputs=decoded)

# Separate encoder model for feature extraction
encoder = models.Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                    loss='mse')

autoencoder.summary()



# 3. Train Autoencoder
autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=10,
    batch_size=32,
    shuffle=True,
    verbose=1
)
# 4. Extract features using encoder
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)


model = XGBRegressor()
model.fit(X_train_encoded, y_train)

# Predict
y_pred = model.predict(X_test_encoded)

# Compute RMSLE
y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
y_test_clipped = np.clip(y_test, a_min=0, a_max=None)
rmsle = np.sqrt(mean_squared_log_error(y_test_clipped, y_pred_clipped))
print(f'XGB Regressor RMSLE: {rmsle:.4f}')


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
    
def swish(x):
    return x * tf.keras.backend.sigmoid(x)
def build_swish_mlp(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(32, activation=swish)(inputs)
    x = Dense(64, activation=swish)(x)
    x = Dense(32, activation=swish)(x)
    x = Dense(1)(x)
    model = Model(inputs, x)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=[root_mean_squared_error])
    return model


y_train_log = np.log1p(y_train)
model = build_swish_mlp(input_shape=(X_train_encoded.shape[1],))
model.fit(X_train_encoded, y_train_log, epochs=10, batch_size=64, verbose=1)


test_data.drop(columns=['id'], inplace = True)


le = LabelEncoder()
test_data['Sex'] = le.fit_transform(test_data['Sex'].astype(str))
test_data.shape


X_final = scaler.transform(test_data)
X_final_encoded = encoder.predict(X_final)


y_test_pred = model.predict(X_final_encoded)
final_pred = np.expm1(y_test_pred).flatten()
final_pred = np.clip(final_pred, y_train.min(), y_train.max())  


submission_data = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission_data['Calories'] = final_pred


submission_data.to_csv('/kaggle/working/submission.csv', index=False)

