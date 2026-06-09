# to handle data
import pandas as pd
import numpy as np

# to visualize data
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# to preprocess data

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer

# deep learning tasks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV

#metrics
from sklearn.metrics import accuracy_score,precision_score, f1_score ,recall_score ,confusion_matrix, classification_report, mean_absolute_error,mean_squared_error,r2_score

# ignore warnings

import warnings
warnings.filterwarnings('ignore')


# train data

df_train = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/train.csv')
df_train.head()


# test data

df_test = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/test.csv')
df_test.head()


# sample_submission file

df_sample = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/sample_submission.csv')
df_sample.head()


print('In Train dataset')
print(f'Number of rows: {df_train.shape[0]}')
print(f'Number of columns: {df_train.shape[1]}')
print('------------------------')
print('In Test dataset')
print(f'Number of rows: {df_test.shape[0]}')
print(f'Number of columns: {df_test.shape[1]}')


# columns in train

df_train.columns


# columns in test

df_test.columns


# Information about train dataset

df_train.info()


# Information about test dataset

df_test.info()


# Summary of train dataset in transpose

df_train.describe().T


# Unique values in train datset

print('Unique values in Train dataset\n')
print(df_train.nunique())


# Check null values

print(df_train.isnull().sum().sort_values(ascending=False)/len(df_train)*100)


# plot it using seaborn

fig = plt.figure(figsize=(12, 6))
sns.heatmap(df_train.isnull(), cmap='magma', annot=False, fmt='.2f', linewidths=.5)
plt.title('Missing Values Heatmap', fontsize=16)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Rows', fontsize=12)

plt.show()


df_train.info()


# split data into numerical & categorical columns

num_cols = [col for col in df_train.columns if df_train[col].dtype!='O']
cat_cols = [col for col in df_train.columns if col not in num_cols]



# numerical columns

num_cols


# make Boxplot of numeric columns using for loop
plt.figure(figsize=(22, 32))

# Extend the colors list to have at least as many colors as num_cols
colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'brown', 'cyan', 'magenta','pink','lightblue']

# Calculate the number of rows needed based on the number of columns
num_rows = (len(num_cols) + 1) // 2  # Divide by 2 and round up

for i, col in enumerate(num_cols):
    plt.subplot(num_rows, 4, i+1)  # Adjusted to dynamic rows, 2 columns
    sns.boxplot(x=df_train[col], color=colors[i % len(colors)]) # Use modulo operator to cycle through colors
    plt.title(col)
plt.show()


df = df_train.copy()


# Explore age column

# histplot of age using seaborn

fig = plt.figure(figsize=(12,6))
sns.histplot(df['Age'], kde=True)
plt.axvline(df['Age'].mean(),color='red')
plt.axvline(df['Age'].median(),color='green')
plt.axvline(df['Age'].mode()[0],color='blue')
plt.title('Age Distribution')
plt.show()

# print the values of mean, median & mode
print("-----------------------")
print('Mean',df['Age'].mean())
print('Median',df['Age'].median())
print('Mode',df['Age'].mode())



# Explore Balance column

# histplot of age using seaborn

fig = plt.figure(figsize=(12,6))
sns.histplot(df['Balance'], kde=True)
plt.axvline(df['Balance'].mean(),color='red')
plt.axvline(df['Balance'].median(),color='green')
plt.axvline(df['Balance'].mode()[0],color='blue')
plt.title('Balance Distribution')
plt.show()

# print the values of mean, median & mode
print("-----------------------")
print('Mean',df['Balance'].mean())
print('Median',df['Balance'].median())
print('Mode',df['Balance'].mode())



# Explore EstimatedSalary column

# histplot of EstimatedSalary using seaborn

fig = plt.figure(figsize=(36,10))
sns.histplot(df['EstimatedSalary'], kde=True)
plt.axvline(df['EstimatedSalary'].mean(),color='red')
plt.axvline(df['EstimatedSalary'].median(),color='green')
plt.axvline(df['EstimatedSalary'].mode()[0],color='blue')
plt.title('EstimatedSalary Distribution')
plt.show()

# print the values of mean, median & mode
print("-----------------------")
print('Mean',df['EstimatedSalary'].mean())
print('Median',df['EstimatedSalary'].median())
print('Mode',df['EstimatedSalary'].mode())



# Explore Credit Score column

# histplot of Credit Score using seaborn

fig = plt.figure(figsize=(12,6))
sns.histplot(df['CreditScore'], kde=True)
plt.axvline(df['CreditScore'].mean(),color='red')
plt.axvline(df['CreditScore'].median(),color='green')
plt.axvline(df['CreditScore'].mode()[0],color='blue')
plt.title('Credit Score Distribution')
plt.show()

# print the values of mean, median & mode
print("-----------------------")
print('Mean',df['CreditScore'].mean())
print('Median',df['CreditScore'].median())
print('Mode',df['CreditScore'].mode())



cat_cols


# countplot of Geography

fig = plt.figure(figsize=(12, 6))
sns.countplot(df, x='Geography', palette='viridis')  # Use a color palette
plt.title('Countplot of Geography', fontsize=16, fontweight='medium')  # Enhance title
plt.xlabel('Geography', fontsize=14)  # Enhance x-axis label
plt.ylabel('Count', fontsize=14)  # Enhance y-axis label
plt.xticks(fontsize=10)  # Enhance x-axis tick labels
plt.yticks(fontsize=10)  # Enhance y-axis tick labels
sns.despine()  # Remove top and right spines for a cleaner look

plt.show()


# countplot of Gender

fig = plt.figure(figsize=(12,6))
sns.countplot(df,x ='Gender',palette='viridis')
plt.title('Countplot of Gender', fontsize=16, fontweight='medium')
plt.ylabel('Count', fontsize=14)  # Enhance y-axis label
plt.xticks(fontsize=10)  # Enhance x-axis tick labels
plt.yticks(fontsize=10)  # Enhance y-axis tick labels
sns.despine()  # Remove top and right spines for a cleaner look
plt.show()


# countplot of Geography based on Gender

fig = plt.figure(figsize=(12, 6))
sns.countplot(df, x='Geography', palette='viridis', hue='Gender')  # Use a color palette
plt.title('Countplot of Geography', fontsize=16, fontweight='medium')  # Enhance title
plt.xlabel('Geography', fontsize=14)  # Enhance x-axis label
plt.ylabel('Count', fontsize=14)  # Enhance y-axis label
plt.xticks(fontsize=10)  # Enhance x-axis tick labels
plt.yticks(fontsize=10)  # Enhance y-axis tick labels
sns.despine()  # Remove top and right spines for a cleaner look


plt.show()


df.describe()


# Scale creditscore, balance, estimatedsalary in train data using standard scalar

df['CreditScore'] = StandardScaler().fit_transform(df[['CreditScore']])
df['Balance'] = StandardScaler().fit_transform(df[['Balance']])
df['EstimatedSalary'] = StandardScaler().fit_transform(df[['EstimatedSalary']])



# encode categorical columns in train data separately using label encoder

df['Geography'] = LabelEncoder().fit_transform(df['Geography'])
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])



df.head()


# Scale creditscore, balance, estimatedsalary in test data using standard scalar

df_test['CreditScore'] = StandardScaler().fit_transform(df_test[['CreditScore']])
df_test['Balance'] = StandardScaler().fit_transform(df_test[['Balance']])
df_test['EstimatedSalary'] = StandardScaler().fit_transform(df_test[['EstimatedSalary']])



# encode categorical columns in test data separately using label encoder

df_test['Geography'] = LabelEncoder().fit_transform(df_test['Geography'])
df_test['Gender'] = LabelEncoder().fit_transform(df_test['Gender'])



df_test.head()


# Check Columns

df.columns


# Define features and target

X = df.drop(['id', 'CustomerId', 'Surname','Exited'], axis=1)
y = df['Exited']

df_test = df_test.drop(['id', 'CustomerId', 'Surname'], axis=1)


# Spilit the data into X train and y train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



df_test.head()


# Create neural network for binary classification

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu')) # Input layer

model.add(Dense(32, activation='relu')) # Hidden layer

model.add(Dense(1, activation='sigmoid')) # Output layer


# Compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose = 1)

# Evaluating the model
loss = model.evaluate(X_test, y_test, verbose=0)
loss

print("--------------------------------")
print("Loss: ", loss[0])
print("Accuracy: ", loss[1])

# Plot the training loss and accuracy at each epoch

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history['loss'], label='Training loss')
ax.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
ax.legend()
plt.show()

# Plot the testing loss and accuracy at each epoch

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history['accuracy'], label='Training accuracy')
ax.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
ax.legend()
plt.show()



# Building the model

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # Input layer
    tf.keras.layers.Dense(32, activation='relu'), # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for regression
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the callback function
early_stopping = EarlyStopping(patience=11)

# Train the model with the callback function
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Evaluating the model
loss = model.evaluate(X_test, y_test, verbose=0)
loss

print("--------------------------------")
print("Loss: ", loss[0])
print("Accuracy: ", loss[1])

# Plotting the training and testing loss
plt.subplots(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



# Prediction

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to class labels (0 or 1)



# plot the confusion matrix

fig = plt.figure(figsize=(12, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix', fontsize=16, fontweight='medium')  # Enhance title
plt.xlabel('Predicted', fontsize=12)  # Enhance x-axis label
plt.ylabel('True', fontsize=12)  # Enhance y-axis label

plt.show()


# create a submission file

# Ensure y_pred has the same length as df_sample
y_pred_full = model.predict(df_test) # Predict on the entire test data
y_pred_full = (y_pred_full > 0.5).astype(int)

df_sample['Exited'] = y_pred_full  # Assign the full predictions to the DataFrame
df_sample.to_csv('submission.csv', index=False)

