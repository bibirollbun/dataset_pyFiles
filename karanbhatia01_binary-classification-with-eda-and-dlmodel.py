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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train_df.head()


test_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


test_df.duplicated().value_counts()


train_df.info()


import matplotlib.pyplot as plt
import seaborn as sns 


df = train_df.drop(columns=['id'])


for col in df.columns:
    if df[col].dtype == 'O':   
        plt.figure(figsize=(3,3))
        sns.countplot(x=df[col])
        plt.title(f"Count plot of {col}")
        plt.xticks(rotation=45)
        plt.show()


train_df.drop(columns=['id','default','day','month','poutcome','contact'],inplace=True)
test_df.drop(columns=['id','default','day','month','poutcome','contact'],inplace=True)


train_df.head()


train_df['job'].unique()


from sklearn.preprocessing import OneHotEncoder
# Specify columns to encode
columns_to_encode = ['job', 'marital','education','housing','loan']

# Create encoder
ohe = OneHotEncoder(sparse=False, drop=None)

# Transform and combine with the rest of the dataframe
encoded = pd.DataFrame(
    ohe.fit_transform(train_df[columns_to_encode]),
    columns=ohe.get_feature_names_out(columns_to_encode)
)

df_encoded = pd.concat([train_df.drop(columns_to_encode, axis=1), encoded], axis=1)


X = df_encoded.drop(columns=['y'])
y = df_encoded['y']


print(X.shape,y.shape)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


model = Sequential()

model.add(Dense(128,activation='relu',input_dim=29))
model.add(Dropout(0.3))  # 30% dropout
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))  # 30% dropout
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.summary()


early_stop = EarlyStopping(
    monitor='val_loss',     # Watch validation loss
    patience=3,             # Stop after 3 epochs with no improvement
    restore_best_weights=True # Roll back to best weights
)


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


history = model.fit(
    X_train_trf, y_train,
    batch_size=110,
    epochs=50,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stop]
)


y_pred = model.predict(X_test_trf)


y_pred = y_pred.argmax(axis=-1)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load test set
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_df.drop(columns=['default','day','month','poutcome','contact'],inplace=True)
X_ids = test_df['id']

# Apply same preprocessing as training
columns_to_encode = ['job', 'marital', 'education', 'housing', 'loan']

# Use already fitted encoder (ohe) and scaler from training
encoded = pd.DataFrame(
    ohe.transform(test_df[columns_to_encode]),
    columns=ohe.get_feature_names_out(columns_to_encode)
)

df_encoded = pd.concat([test_df.drop(columns_to_encode, axis=1), encoded], axis=1)

X_testing_trf = scaler.fit_transform(df_encoded.drop(columns=['id']))

# Predict
y_pred = model.predict(X_testing_trf)
y_pred_class = (y_pred > 0.5).astype(int).flatten()

# Create submission
submission = pd.DataFrame({
    'id': X_ids,
    'y': y_pred_class
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved.")

# Plot losses
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()



sub = pd.read_csv('/kaggle/working/submission.csv')


sub




