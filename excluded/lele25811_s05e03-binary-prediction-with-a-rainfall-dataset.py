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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_df.head(5)


test_df.head(5)


train_df.isnull().sum()


test_df.isnull().sum()


test_df.dtypes


test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].median())
test_df.isnull().sum()


test_df.dtypes


test_ids = test_df['id']
test_df = test_df.drop(['id', 'day'], axis=1)


X = train_df[['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']]
#X = train_df[["cloud", "sunshine", "humidity", "dewpoint"]]
y = train_df['rainfall']


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
len(train_X), len(test_X), len(train_y), len(test_y)


correlation = X.assign(target=y).corr()['target'].drop('target')
print(correlation.sort_values(ascending=False))


from sklearn.metrics import accuracy_score
def acc_score(predictions, test_y):
    return accuracy_score(test_y, predictions)


from sklearn.linear_model import LogisticRegression

model_1 = LogisticRegression(max_iter = 1000)
model_1.fit(train_X, train_y)
predictions_1 = model_1.predict(test_X)


acc_score(predictions_1, test_y)


from sklearn.ensemble import RandomForestClassifier

model_3 = RandomForestClassifier()
model_3.fit(train_X, train_y)
predictions_3 = model_3.predict(test_X)


acc_score(predictions_3, test_y)


from xgboost import XGBClassifier

model_2 = XGBClassifier(objective="binary:logistic", max_depth=3, learning_rate=0.1, n_estimators=1500,  early_stopping_rounds=10)
model_2.fit(train_X, train_y, eval_set=[(test_X, test_y)], verbose=False)
predictions_2 = model_2.predict(test_X)


acc_score(predictions_2, test_y)


X_val, X_test, y_val, y_test = train_test_split(test_X, test_y, test_size=0.3)


from tensorflow import keras
from tensorflow.keras import layers

input_shape = [train_X.shape[1]]

nn_model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])


nn_model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)


early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)

history = nn_model.fit(
    train_X, train_y,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=200,
    callbacks=[early_stopping]
)


import pandas as pd

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")


nn_predictions = (nn_model.predict(X_test) > 0.5).astype("int32")


acc_score(nn_predictions, y_test)


submission_predictions = (nn_model.predict(test_df) > 0.5).astype("int32").flatten()


submission = pd.DataFrame({'id': test_ids,
                          'rainfall': submission_predictions 
                          })
submission.head(5)


submission.to_csv('/kaggle/working/rain_submission.csv', index=False)

