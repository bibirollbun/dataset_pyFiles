# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# Define custom metrics and activation functions
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def swish(x):
    return x * tf.keras.backend.sigmoid(x)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


# Load data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


# Drop ID columns
df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


# Load external data
df_extra = pd.read_csv('/kaggle/input/calories-burning-dataset/exercise.csv')
df_cal = pd.read_csv('/kaggle/input/calories-burning-dataset/calories.csv')



# Prepare external data
df_extra.drop(columns=['User_ID'], inplace=True)
df_extra.rename(columns={'Gender': 'Sex'}, inplace=True)
df_extra['Calories'] = df_cal['Calories'].values


# Combine with training data
df_extra = df_extra[df_train.columns]
df_train = pd.concat([df_train, df_extra], ignore_index=True)


# Feature engineering
def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new


def create_advanced_features(df):
    epsilon = 1e-5
    # Basic physiological features
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    # Energy expenditure related features
    df['Exercise_Intensity'] = df['Heart_Rate'] / (220 - df['Age'])
    df['Metabolic_Equivalent'] = df['Heart_Rate'] * df['Duration'] / df['Weight']
    
    # Thermal effect features
    df['Heat_Index'] = df['Body_Temp'] * df['Duration']
    df['Thermal_Strain'] = df['Body_Temp'] * df['Heart_Rate'] / 100
    
    # Polynomial features
    df['Duration_squared'] = df['Duration'] ** 2
    df['HR_squared'] = df['Heart_Rate'] ** 2
    df['Duration_cubed'] = df['Duration'] ** 3
    
    # Interaction features
    df['HR_Weight_Interaction'] = df['Heart_Rate'] * df['Weight'] / 100
    df['Temp_Duration_Weight'] = df['Body_Temp'] * df['Duration'] / df['Weight']
    
    # Logarithmic transformations
    df['Log_Duration'] = np.log1p(df['Duration'])
    df['Log_HR'] = np.log1p(df['Heart_Rate'])
    
    # Ratio features
    df['HR_to_Temp_Ratio'] = df['Heart_Rate'] / df['Body_Temp']
    df['Duration_to_Weight'] = df['Duration'] / df['Weight']
    
    return df


# Apply feature engineering
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
df_train = add_feature_cross_terms(df_train, numerical_features)
df_test = add_feature_cross_terms(df_test, numerical_features)

# Encode categorical features
le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_test['Sex'] = le.transform(df_test['Sex'])

# Prepare data for modeling
X_train = df_train.drop('Calories', axis=1).values.astype(np.float32)
y_train = df_train['Calories'].values.astype(np.float32)
X_test = df_test.values.astype(np.float32)

# Log transform target for better model performance
y_train_log = np.log1p(y_train)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Build model with swish activation
def build_swish_mlp(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(32, activation=swish)(inputs)
    x = Dense(64, activation=swish)(x)
    x = Dense(32, activation=swish)(x)
    x = Dense(1)(x)
    model = Model(inputs, x)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=[root_mean_squared_error])
    return model



# Build advanced model
def build_advanced_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # First branch - deep network
    x1 = Dense(64, kernel_regularizer=l2(1e-5))(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation(mish)(x1)
    x1 = Dropout(0.2)(x1)
    
    x1 = Dense(128, kernel_regularizer=l2(1e-5))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(mish)(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Dense(64, kernel_regularizer=l2(1e-5))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(mish)(x1)
    
    # Second branch - shallow network for direct relationships
    x2 = Dense(32, kernel_regularizer=l2(1e-5))(inputs)
    x2 = Activation(mish)(x2)
    
    # Combine branches
    combined = Concatenate()([x1, x2])
    
    # Output layer
    output = Dense(1)(combined)
    
    model = Model(inputs, output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=[root_mean_squared_error]
    )
    return model


# Train model
model = build_swish_mlp(input_shape=(X_train_scaled.shape[1],))
model.fit(X_train_scaled, y_train_log, epochs=20, batch_size=64, verbose=1)


# Predict and transform back to original scale
y_test_pred = model.predict(X_test_scaled)
final_pred = np.expm1(y_test_pred).flatten()


# Clip predictions to valid range
final_pred = np.clip(final_pred, y_train.min(), y_train.max())


# Create submission
df_sub['Calories'] = final_pred
df_sub.to_csv('submission.csv', index=False)

