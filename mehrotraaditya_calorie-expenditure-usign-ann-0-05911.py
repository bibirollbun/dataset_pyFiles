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


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns


def rmsle(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0, tf.reduce_max(y_pred))  # Ensure predictions are non-negative
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


df_train.head()


df_train.info()


# Target variable
target = "Calories"

# Handle categorical variables
categorical_cols = ['Sex']
df = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)


df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True) 


df_test.head()


df.isna().sum()


# Prepare features and target
X = df.drop(columns=['id', target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Advanced ANN architecture
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU


# def build_model(input_dim):
#     inputs = Input(shape=(input_dim,))
#     x = Dense(1024)(inputs)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#     x = Dropout(0.2)(x)

#     x = Dense(512)(inputs)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#     x = Dropout(0.2)(x)

#     x = Dense(256)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#     x = Dropout(0.2)(x)

#     x = Dense(128)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#     x = Dropout(0.2)(x)

#     x = Dense(32)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)

#     outputs = Dense(1)(x)

#     model = Model(inputs, outputs)
#     return model



# model = build_model(X_train_scaled.shape[1])


# # Compile model
# optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)
# model.compile(optimizer=optimizer, loss=rmsle)

# # Callbacks
# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)


# # Train model
# history = model.fit(
#     X_train_scaled, y_train,
#     validation_split=0.2,
#     epochs=100,
#     batch_size=512,
#     callbacks=[early_stop, reduce_lr],
#     verbose=1
# )


# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])


# Save the model to disk
# model.save("ann_model.h5")


from tensorflow.keras.models import load_model
model = load_model("/kaggle/input/calori_expenditure_v1_ann_model/keras/default/1/ann_model.h5", custom_objects={'rmsle': rmsle})


ids = df_test['id']
df_test.drop(columns=['id'],inplace=True)


df_test_scaled = scaler.transform(df_test)


predictions = model.predict(df_test_scaled)


submission = pd.DataFrame({
    'id': ids,  # or whatever your ID column is
    'target': predictions.flatten()
})
submission.to_csv('submission.csv', index=False)


submission




