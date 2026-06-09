import numpy as np 
import pandas as pd
import seaborn as sb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as Split 

from sklearn.preprocessing import OrdinalEncoder
from keras.utils import to_categorical 
from warnings import filterwarnings
filterwarnings("ignore")
tf.config.list_physical_devices()


df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df


sb.heatmap(df[df.select_dtypes(include=np.number).columns])
plt.title("Heatmap")
plt.show()


df.select_dtypes(include=np.number).columns


df.info()


df.isna().sum()


(df.isna().sum()* 100 / len(df)).round()


def fillNA(df,name):
    df[name] = df[name].fillna(df[name].mean())
    return df 
df =  fillNA(df,'Episode_Length_minutes')
df = fillNA(df,"Guest_Popularity_percentage")
df = df.dropna()
df


df.isna().sum()


df.info()


str_cols = df.select_dtypes(exclude=['int','float'])
str_cols.info()


output = df['Listening_Time_minutes'].values

X = df.drop(['id',"Listening_Time_minutes"],axis=1)
X


ohe = OrdinalEncoder()
X [str_cols.columns] = ohe.fit_transform(X [str_cols.columns])
X


X.info()


x_train,x_test,y_train,y_test = Split(X,output,test_size=0.2,random_state=42)


# Define the model
def create_complex_regression_model(input_shape):
    model = Sequential()

    # Input layer
    model.add(Dense(512, input_shape=(input_shape,), kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # Hidden layers
    model.add(Dense(256, kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(128, kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Dense(64, kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Output layer for regression
    model.add(Dense(1, activation='linear'))  # Predicting a continuous value

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    return model

# Example usage:
# Suppose your feature vector has 20 features
input_shape  = X.shape[1]
model = create_complex_regression_model(input_shape)
model.summary()


#%% Execution
model.fit(x_train,y_train)


model.evaluate(x_test,y_test)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
df_test


df_test.info()


df_test.isna().sum()


df_test = fillNA(df_test,'Episode_Length_minutes')
df_test = fillNA(df_test,'Guest_Popularity_percentage')
df_test


df_test.info()


df_obj =  df_test.select_dtypes(exclude=['int','float'])
df_test[df_obj.columns] = ohe.fit_transform(df_test[df_obj.columns])
df_test


df_test.shape


def submission_file(test):
    ids = test['id'].values  # Extract IDs
    x_test = test.drop('id', axis=1).values
    predict = model.predict(x_test)

    # Ensure predictions are 1-dimensional
    if predict.ndim > 1:
        predict = predict.ravel()

    submission = pd.DataFrame({
        'id': ids,
        'Listening_Time_minutes': predict
    })
    return submission


submission = submission_file(df_test)
submission.to_csv("submission.csv",index=False)

