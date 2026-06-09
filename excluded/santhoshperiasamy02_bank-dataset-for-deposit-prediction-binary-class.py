pip install tensorflow


import pandas as pd # for data frame
import numpy as np  # for numerical computation

import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization

from sklearn.preprocessing import StandardScaler # for scaling the data
from sklearn.model_selection import train_test_split # for splitting the data into train and test

from tensorflow.keras.models import Sequential # for building the neural network model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout , BatchNormalization # for defining the layers of the neural network
from tensorflow.keras.optimizers import Adam , AdamW# for optimizing the neural network
from tensorflow.keras.activations import elu
from tensorflow.keras.losses import BinaryCrossentropy # for calculating the loss
from tensorflow.keras.metrics import BinaryAccuracy , AUC # for evaluating the model

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")



df.head()


test.head()


df.shape


df.info()


test.info()


data = pd.concat([df,test])


data.head()


data.tail()


data.drop(columns=['id'],inplace=True)
# remove the Id columns , that doesnot contributes to model.


data.age.value_counts()


plt.hist(data.age, bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()


data.columns


data['job'].value_counts()


data[(data['job'] == 'unknown') & (data['education'] == 'unknown')&(data['poutcome'] == 'unknown')&(data['contact'] == 'unknown')]


temp = data[(data['job'] == 'unknown') & (data['education'] == 'unknown')&(data['poutcome'] == 'unknown')&(data['contact'] == 'unknown')]
temp['default'].value_counts()


data[(data['job'] == 'unknown') & (data['education'] == 'unknown')&(data['poutcome'] == 'unknown')&(data['contact'] == 'unknown')&(data['y'].isna())]


categorical_cols = data.select_dtypes(include='object').columns
print(categorical_cols)


for i in categorical_cols:
  print(i)
  print('\n')
  print(data[i].value_counts())


numerical_cols = data.select_dtypes(exclude='object').columns
print(numerical_cols)


for col in numerical_cols:
  print(col)
  print('\n')
  plt.boxplot(data[col])
  plt.title(f'Box plot of {col}')
  plt.ylabel('Value')
  plt.show()


data.balance.value_counts()

# need to take action





data.previous.value_counts()
# need to take action


correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Training Data')
plt.show()


data.head()


categorical_cols


numerical_cols


data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
display(data_encoded.head())


train_encoded = data_encoded[data_encoded['y'].notna()]
test_encoded = data_encoded[data_encoded['y'].isna()]


scale_columns = numerical_cols.drop('y')


scale = StandardScaler()
train_encoded[scale_columns] = scale.fit_transform(train_encoded[scale_columns])
test_encoded[scale_columns] = scale.transform(test_encoded[scale_columns])


train_encoded.head()


test_encoded.drop(columns='y')


correlation_matrix_encoded = train_encoded.corr(numeric_only=True)
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix_encoded, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Scaled and Encoded Training Data')
plt.show()


train_encoded.y.value_counts(normalize=True)


X = train_encoded.drop(columns=['y'])
y = train_encoded['y']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = X_val
y_test = y_val


# Step 1: Define the Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer matches feature count
    Dropout(0.3),  # 30% dropout for regularization
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])


# Step 2: Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(),
              metrics=[AUC(name='roc_auc')])


# Step 3: Train the Model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=256,
                    verbose=1)


# Step 4: Evaluate the Model
# Get predictions for ROC AUC calculation
y_pred_proba = model.predict(X_test)
from sklearn.metrics import roc_auc_score
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test ROC AUC: {test_roc_auc:.4f}')


y_final_predict = model.predict(test_encoded.drop(columns=['y']))


y_final_predict


y_pred_binary = (y_final_predict >= 0.5).astype(int)


y_pred_binary.flatten()


submission = pd.DataFrame({'id': test['id'], 'y': y_pred_binary.flatten()})


submission.y.value_counts()


train_encoded.y.value_counts()


submission.to_csv('submission.csv', index=False)


model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],)),  # First hidden layer (increased size)
    BatchNormalization(),  # Normalization
    Dropout(0.3),  # Dropout for regularization
    Dense(128, activation=elu),  # Second hidden layer with ELU
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation=elu),  # Third hidden layer with ELU
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation=elu),  # Fourth hidden layer (added) with ELU
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer
])


model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(),
              metrics=[AUC(name='roc_auc')])


history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=256,
                    verbose=1)


y_pred_proba = model.predict(X_test)
from sklearn.metrics import roc_auc_score
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test ROC AUC: {test_roc_auc:.4f}')


y_pred_proba = model.predict(X_test)
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test ROC AUC: {test_roc_auc:.4f}')


y_final_predict = model.predict(test_encoded.drop(columns=['y']))


y_pred_binary = (y_final_predict >= 0.5).astype(int)


submission = pd.DataFrame({'id': test['id'], 'y': y_pred_binary.flatten()})


submission.y.value_counts()


submission.to_csv('submission_1.csv', index=False)


model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],)),  # First hidden layer (increased size)
    BatchNormalization(),  # Normalization
    Dense(128, activation=elu),  # Second hidden layer with ELU
    BatchNormalization(),
    Dense(64, activation=elu),  # Third hidden layer with ELU
    BatchNormalization(),
    Dense(32, activation=elu),  # Fourth hidden layer (added) with ELU
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer
])


model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(),
              metrics=[AUC(name='roc_auc')])


history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=256,
                    verbose=1)


y_pred_proba = model.predict(X_test)
from sklearn.metrics import roc_auc_score
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test ROC AUC: {test_roc_auc:.4f}')


y_final_predict = model.predict(test_encoded.drop(columns=['y']))
y_pred_binary = (y_final_predict >= 0.5).astype(int)


submission = pd.DataFrame({'id': test['id'], 'y': y_pred_binary.flatten()})


submission.to_csv('submission_2.csv', index=False)


model = Sequential([
    Dense(128,activation= 'relu' , input_shape=(X_train.shape[1],)),  # First hidden layer (increased size)
    BatchNormalization(),  # Normalization
    Dense(64, activation= 'relu'),  # Third hidden layer
    BatchNormalization(),
    Dense(32, activation= 'relu'),  # Fourth hidden layer (added)
    BatchNormalization(),
    Dense(1, activation='sigmoid')  # Output layer
])


model.compile(optimizer=AdamW(learning_rate=0.0005, weight_decay=0.01),
              loss=BinaryCrossentropy(),
              metrics=[AUC(name='roc_auc')])


history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=64,
                    verbose=1)


y_pred_proba = model.predict(X_test)
from sklearn.metrics import roc_auc_score
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test ROC AUC: {test_roc_auc:.4f}')


y_final_predict = model.predict(test_encoded.drop(columns=['y']))
y_pred_binary = (y_final_predict >= 0.5).astype(int)


submission = pd.DataFrame({'id': test['id'], 'y': y_pred_binary.flatten()})


submission.to_csv('submission_4.csv', index=False)




