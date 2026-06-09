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


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')



test.bfill(inplace = True)
test.bfill(inplace = True)


def create_new_features(df):
    df["temp_range"] = df["maxtemp"] - df["mintemp"]
    df["temp_deviation"] = df["maxtemp"] - df["temparature"]
    df["humidity_dew_diff"] = df["humidity"] - df["dewpoint"]
    df["winddir_rad"] = np.radians(df["winddirection"])
    df["wind_x"] = df["windspeed"] * np.cos(df["winddir_rad"])
    df["wind_y"] = df["windspeed"] * np.sin(df["winddir_rad"])
    features = [ 'pressure', 'maxtemp', 'temparature', 'mintemp',
           'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
           'windspeed', 'temp_range','temp_deviation',
               'humidity_dew_diff','winddir_rad','wind_x','wind_y']
    #shifted features
    shifts = [1,2,4,5]
    for f in features:
        for s in shifts:
            df[f'{f}_shift_{s}'] = df[f].shift(s)
    df['months'] = (df['id']%365) // 30 +1
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 365)
    df = df.fillna(method = 'bfill', axis = 0)
    return df


train, test = create_new_features(train),create_new_features(test)


train.shape, test.shape


X = train[[col for col in train.columns if col not in ['rainfall','id','day']]].copy()
y = train['rainfall'].copy()


X.head(1)


test_data = test[[col for col in test.columns if col not in ['id','day']]]


assert (X.columns == test_data.columns).all()
assert X.shape[0] == len(y)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_data)


X_train,X_valid,y_train, y_valid = train_test_split(X_scaled, y, test_size=0.33, random_state=42,stratify = y)


model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1,activation = 'sigmoid')
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC'],
)


early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=100000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)


history_df = pd.DataFrame(history.history)


history_df.head(2)


history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['AUC', 'val_AUC']].plot()


submission = pd.DataFrame({
    'id': test['id'],         
    'rainfall': model.predict(test_scaled).ravel()
})


assert submission.shape[0] == test.shape[0]


# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
print("Submission created")


