import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


_ = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv", index_col="id")
_


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col="id")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv", index_col="id")
train_df.head()


train_df.info()


test_df.describe()


print(train_df.shape)
print(test_df.shape)


train_df.isnull().sum()


test_df.isnull().sum()


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Sequential

model = Sequential(
    [
        Dense(500, activation="relu", kernel_regularizer = L2(0.01)),
        Dense(1000, activation="relu", kernel_regularizer=L2(0.01)),
        Dense(1000, activation="relu", kernel_regularizer=L2(0.01)),
        Dense(1, activation="linear")
    ]
)

model.compile(
    loss = tf.keras.losses.mse,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
)


from sklearn.model_selection import train_test_split


train_df.dropna(axis=0, subset=['Listening_Time_minutes'], inplace=True)
y = train_df.Listening_Time_minutes
train_df.drop(['Listening_Time_minutes'], axis=1, inplace=True)

numerical_data = [i for i in train_df.columns if train_df[i].dtype in ["int64", "float64"]]
categorical_data = [i for i in train_df.columns if train_df[i].dtype == "object" and train_df[i].nunique() < 10]

X = train_df[numerical_data + categorical_data].copy()
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,
                                                      test_size = 0.2, random_state=0)
X_test = test_df[numerical_data + categorical_data].copy()


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

numerical_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_data),
    ("cat", categorical_transformer, categorical_data)
])

my_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


my_pipeline.fit(X, y)

preds = my_pipeline.predict(X_valid)


preds = my_pipeline.predict(X_test)


preds.flatten



output = pd.DataFrame({"id":X_test.index, "Listening_Time_minutes":preds[:,0]})
output


output.to_csv("submission.csv", index=False)




