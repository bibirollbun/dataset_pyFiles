import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')


# New features

train['temp_diff'] = train['maxtemp'] - train['mintemp']
train['cloud_to_sunshine'] = train['cloud'] * train['sunshine']
train['cloud_humidity'] = train['cloud'] + train['humidity']
train['humidity_sunshine'] = train['humidity'] * train['sunshine']

# Adding more features
# Dew point depression
train['dew_point_depression'] = train['temparature'] - train['dewpoint']

test['temp_diff'] = test['maxtemp'] - test['mintemp']
test['cloud_to_sunshine'] = test['cloud'] * test['sunshine']
test['cloud_humidity'] = test['cloud'] + test['humidity']
test['humidity_sunshine'] = test['humidity'] * test['sunshine']
# Dew point depression
test['dew_point_depression'] = test['temparature'] - test['dewpoint']


train.drop(columns=['temparature', 'winddirection'], inplace=True)

test.drop(columns=['temparature', 'winddirection'], inplace=True)


def remove_outliers_iqr(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

# Columns to check for outliers
columns_to_check = ['humidity', 'dewpoint', 'cloud']

# Remove outliers
train = remove_outliers_iqr(train, columns_to_check)


# Balance the dataset
from imblearn.over_sampling import SMOTE

X, y = train.drop(columns=['rainfall']), train['rainfall']

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled,
                                                  test_size=.25,
                                                  random_state=777
                                                  # stratify=y
                                                 )


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_val_scaled = sc.transform(X_val)

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)


X_train_scaled.shape


input_shape=int(X_train_scaled.shape[-1])


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(777)

model = keras.Sequential([
    layers.Input((input_shape,)),
    layers.Dense(64, activation='relu',),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu',),
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu',),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid'),
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)


early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)


history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    batch_size=128,
    epochs=50,
    callbacks=[early_stopping],
)


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")


test_scaled = pd.DataFrame(sc.transform(test), index=test.index, columns=test.columns)
predictions = model.predict(test_scaled)


predictions[:5]


submission = pd.DataFrame({'id': test.index, 'rainfall': predictions[:, 0]})
submission.to_csv('submission.csv', index=False)







