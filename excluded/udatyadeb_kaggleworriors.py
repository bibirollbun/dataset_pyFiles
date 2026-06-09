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


df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
df.head()


df.head()


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



df[df['phase']=='Transition'].head()
# orientation	behavior	phase	gesture


df_filtered = df[['subject',	'behavior',	'phase',	'gesture',	'acc_x',	'acc_y',	'acc_z',	'rot_w',	'rot_x',	'rot_y',	'rot_z'	,'thm_1',	'thm_2'	,'thm_3'	,'thm_4'	,'thm_5']]
df_filtered.head()

# sequence_type orientation  





df_filtered['behavior'].value_counts()


df_filtered[df_filtered['gesture']=='Text on phone'].head()


df_test = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
df_test.head()


df['gesture'].value_counts()


df_filtered.head()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load your dataset
df = df_filtered

# Separate features and labels
X = df.drop(columns=["gesture"])
y = df["gesture"]

# Encode categorical input features
X = pd.get_dummies(X, columns=["subject", "behavior", "phase"])

# Normalize numerical features
numeric_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 
                'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Encode gesture (label)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)





y_encoded


X_train.head()


# Build the neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_categorical.shape[1], activation='softmax')  # Softmax for multi-class
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)




# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

