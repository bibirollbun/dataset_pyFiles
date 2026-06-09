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


import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test =pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_train['temp_range'] = df_train['maxtemp'] - df_train['mintemp']
df_test['temp_range'] = df_test['maxtemp'] - df_test['mintemp']


df_train['temp_from_dewpoint'] = df_train['temparature'] - df_train['dewpoint']
df_test['temp_from_dewpoint'] = df_test['temparature'] - df_test['dewpoint']


df_train.head(3)


df_test.head(3)


df_train = df_train.drop(['mintemp','maxtemp','dewpoint'],axis="columns")
df_test = df_test.drop(['mintemp','maxtemp','dewpoint'],axis="columns")


df_train.info(),df_test.info()


df_train.isnull().sum(),df_test.isnull().sum()


df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].mean())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X=df_train.drop('rainfall',axis=1)
y=df_train['rainfall']


scaler = StandardScaler()

# Fit and transform the data (scale the features)
X_scaled = scaler.fit_transform(X)

# If you want to check the scaled data:
print(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


X_scaled.shape


import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(9, input_shape=(11,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['auc'])

model.fit(X_train, y_train, epochs = 100, batch_size = 4, validation_split =0.25)


val_preds = model.predict(X_test)


from sklearn import metrics
from sklearn.metrics import auc
metrics.roc_auc_score(y_test, val_preds)


df_test.head(3)


df_test_scaled = scaler.fit_transform(df_test)


test_pred_proba = model.predict(df_test_scaled)


print(test_pred_proba.shape)


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission_df['rainfall'] = test_pred_proba
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

