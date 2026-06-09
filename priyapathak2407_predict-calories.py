import numpy as np
import pandas as pd
import os
# from sklearn.ensemble import RandomForestClassifier # random forest classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_log_error
# import xgboost as xgb # xgboost this is also ensemble model this adds decision trees to solve the error of the previous one
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler # For normalizing


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_data = train_data.drop_duplicates()


x = train_data.drop('Calories', axis=1)
y = train_data['Calories']


# sanity check
print(x.columns)
print(len(x)==len(y))
print(len(x.columns))


norm_cols = x.columns[2:] # They are numerical columns
print(norm_cols)


# Normalizing columns
x[norm_cols] = (x[norm_cols] - x[norm_cols].min()) / (x[norm_cols].max() - x[norm_cols].min())
test_data[norm_cols] = (test_data[norm_cols] - test_data[norm_cols].min()) / (test_data[norm_cols].max() - test_data[norm_cols].min())


# Splitting the train data into train and validation
train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=44)


print('Length of training data: ',len(train_data))
print('Length of testing data: ',len(test_data))


print('Length of train data: ',len(train_x))
print('Length of val data: ', len(val_x))


columns = train_x.columns


columns


train_x.describe()


# converting the sex column to one hot-encoding
train_x = pd.get_dummies(train_x, columns=['Sex'])
val_x = pd.get_dummies(val_x, columns=['Sex'])
test_data = pd.get_dummies(test_data, columns=['Sex'])


# dropping id as it is just sequence and not contributing anything to result
train_x = train_x.drop('id', axis=1)
val_x = val_x.drop('id', axis=1)
test_data = test_data.drop('id', axis=1)


# how would you call the mean_squared_log_error and root it example
# rmsle_value = np.sqrt(mean_squared_log_error(actual, predicted))


def rmlse_loss(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return K.sqrt(msle(y_true, y_pred))


input_dim = len(train_x.columns)
print(input_dim)


model = Sequential([
    Dense(500, activation='relu', input_shape=(input_dim,)),
    Dropout(0.5),
    Dense(400, activation='relu'),
    Dropout(0.5),
    Dense(200, activation='relu'),
    Dense(150, activation='relu'),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])


model.compile(
    optimizer='adam',
    loss=rmlse_loss,
    metrics=['mae']  # You can add other metrics as needed
)


model.summary()


model.fit(train_x, train_y, epochs=50, batch_size=32, validation_data=(val_x, val_y))


# for layer in model.layers:
    # print(layer.name, layer.get_config())


# help(Dense)


from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, rankdir='TB')


val_y_pred = model.predict(val_x)


# accuracy = accuracy_score(val_y, val_y_pred)
# print("Accuracy:", accuracy)


y_preds = model.predict(test_data)


# sanity check for y_preds
print(type(y_preds))
print(len(y_preds))
print(y_preds[0])
print(type(y_preds[0]))
print(y_preds[1])


# Save submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission["Calories"] = y_preds
submission.to_csv("submission.csv", index=False)
print('Sumission done!')
submission.head()




