import numpy as np
import pandas as pd
import os

from scipy.stats import ks_2samp #nonparametric test

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns

import warnings



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


pd.set_option('display.max_columns', None)


train


train.info()


for col in train:
    if train[col].dtype == 'object':
        print(col, train[col].unique())


train.isna().sum().sum()


test


test.isna().sum().sum()


for col in test:
    if test[col].dtype == 'object':
        print(col, test[col].unique())


submission


del_cols = []

for col in test:
    stat, pv = ks_2samp(train[col], test[col])
    if pv < 0.05:
        del_cols.append(col)

print(del_cols)

train = train.drop(del_cols, axis = 1)
test = test.drop(del_cols, axis = 1)

train.shape, test.shape


#analyse target
plt.bar(train['y'].unique(), train['y'].value_counts(), width=0.1)
plt.show()


#analyse age
plt.hist(train['age'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.show()


#analyse job
plt.barh(train['job'].unique(), train['job'].value_counts())
plt.show()


#analyse marital status
plt.pie(train['marital'].value_counts(), labels=train['marital'].unique(), autopct='%1.1f%%', startangle=140)
plt.show()


#analyse education
plt.pie(train['education'].value_counts(), labels=train['education'].unique(), autopct='%1.1f%%', startangle=140)
plt.show()


#analyse default
plt.bar(train['default'].unique(), train['default'].value_counts(), width=0.1)
plt.show()


#analyse housing
plt.bar(train['housing'].unique(), train['housing'].value_counts(), width=0.1)
plt.show()


#analyse loan
plt.bar(train['loan'].unique(), train['loan'].value_counts(), width=0.1)
plt.show()


#analyse contact
plt.pie(train['contact'].value_counts(), labels=train['contact'].unique(), autopct='%1.1f%%', startangle=140)
plt.show()


#analyse day
plt.hist(train['day'], bins=31, color='blue', edgecolor='black', alpha=0.7)
plt.show()


#analyse month
plt.barh(train['month'].unique(), train['month'].value_counts())
plt.show()


#analyse poutcome
plt.pie(train['poutcome'].value_counts(), labels=train['poutcome'].unique(), autopct='%1.1f%%', startangle=140)
plt.show()


train_num = train.select_dtypes(exclude = ['object'])
corr = train_num.corr()
sns.heatmap(corr,cmap='crest')


enc = OrdinalEncoder()

for col in test:
    if test[col].dtype =='object':
        train[col] = enc.fit_transform(train[col].values.reshape(-1,1))
        test[col] = enc.transform(test[col].values.reshape(-1,1))


train.info()


test.info()


y = train.pop('y')
X = train
X_test = test


scaler = StandardScaler()

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# Compute class weights which might come in handy for our model to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

class_weight_dict = dict(enumerate(class_weights))
class_weight_dict # Display computed class weights


# Define our neural network model. The model is sequential meaning that 
# our inputs pass trough the network in a sequential manner, layer after layer.
# The model starts out very wide with a lot of neurons (256) in a single dense layer.
# With each dense layer, the model becomes narrower and narrower and eventually 
# outputs a single value between 0 and 1, which will be the prediction given one row of input features.

model = Sequential([
    Dense(256, activation='relu', input_dim=X.shape[1]), 
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2), 

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

# Compile our neural network
model.compile(optimizer=Adam(learning_rate=1e-2),
              loss='binary_crossentropy',
              metrics=['AUC'],
              )

# Define criteria for early stopping during training. 
# This will prevent the model from overfitting to the training data.
# The training will stop when the validation loss has not improved for 10 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0,
                                                  patience=10,
                                                  verbose=1,
                                                  mode='auto',
                                                  restore_best_weights=True, # Restores the weights of the best epoch
                                                  start_from_epoch=20
                                                 )

# Fit our model on the data. Using a validation split of 0.2, the model will use 
# 80% of the data for training and the leftover for validation during the fitting process.
# The number of epochs determines the number of times the model will see our training data 
# during the fitting process. The batch size determines the size of each batch that gets 
# passed to our model in one training instance. Lastly, we pass our class_weights_dict so the model
# can take the class imbalance into account.
model.fit(X,
          y,
          validation_split=0.2,
          epochs=100,
          batch_size=256,
          verbose=1,
          callbacks=[early_stopping],
          class_weight=class_weight_dict
         )


pred = model.predict(X_test)
pred


submission['y'] = pred
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

