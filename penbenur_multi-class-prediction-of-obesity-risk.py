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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import joblib


train_df=pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')


train_df.head(20)


train_df.info()


train_df.shape


train_df.corr(numeric_only=True)


train_df.describe()


train_df.isnull().sum()


train_df['CAEC'].value_counts()


train_df['CALC'].value_counts()


train_df['MTRANS'].value_counts()


train_df['NObeyesdad'].value_counts()


plt.figure(figsize=(10, 6))
sns.countplot(x='NObeyesdad', data=train_df)
plt.title('Distribution of NObeyesdad')
plt.xlabel('NObeyesdad')
plt.ylabel('Count')
plt.show()


train_df["NObeyesdad"] = train_df["NObeyesdad"].map({"Insufficient_Weight": 0, "Normal_Weight": 1, "Overweight_Level_I": 2,
                                                    "Overweight_Level_II":3, "Obesity_Type_I":4, "Obesity_Type_II":5, "Obesity_Type_III":6})


# Encoding categorical variables
label_encoder = LabelEncoder()
train_df['Gender'] = label_encoder.fit_transform(train_df['Gender'])
train_df['MTRANS'] = label_encoder.fit_transform(train_df['MTRANS'])
train_df['CAEC'] = label_encoder.fit_transform(train_df['CAEC'])
train_df['CALC'] = label_encoder.fit_transform(train_df['CALC'])


# Save the encoder
joblib.dump(label_encoder, 'label_encoder.pkl')


train_df["family_history_with_overweight"] = train_df["family_history_with_overweight"].map({"yes":1, "no":0})
train_df["FAVC"] = train_df["FAVC"].map({"yes":1, "no":0})
train_df["SMOKE"] = train_df["SMOKE"].map({"yes":1, "no":0})
train_df["SCC"] = train_df["SCC"].map({"yes":1, "no":0})


# Splitting features and target
X = train_df.drop(columns=['id','NObeyesdad'])
y = train_df['NObeyesdad']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Save the scaler
joblib.dump(scaler, 'scaler.pkl')


# Define the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)


# Save the model
model.save('model.h5')




