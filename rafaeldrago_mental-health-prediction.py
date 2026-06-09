
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/depressed-people/train.csv')
test = pd.read_csv('/kaggle/input/depressed-people/test.csv')
sample = pd.read_csv('/kaggle/input/depressed-people/sample_submission.csv')


test_copy = test.copy()


df.info()


df.describe(include='all')


df.head(4)


test.head(4)


sample.head(4)


df = df.drop('index',axis=1)
test = test.drop('index',axis=1)


x_values = df.select_dtypes(include=['number'])



corr_matrix = df.select_dtypes(include='number').corr()
plt.figure(figsize=(16, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.1f',cmap="Blues")
plt.show() 



plt.figure(figsize=(10, 8))
sns.countplot(x='Depression', data=df)
plt.title('label Distribution')
plt.show()



fig, axis = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))

for ax, x_value in zip(axis.flat, x_values):
    sns.kdeplot(data=df, x=x_value, fill=True,hue='Depression', common_norm=False, alpha=0.5, ax=ax)
    ax.set_title(f'{x_value.capitalize()}')
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))

for i, x_value in enumerate(x_values):
    ax = axes.flatten()[i] 
    sns.boxplot(data=df, x='Depression', y=x_value, hue='Depression', ax=ax,palette=["#006992", "#ff7d00"])
    ax.set_title(f'{x_value.capitalize()}')
    ax.set_ylabel(x_value.capitalize())
plt.tight_layout()
plt.show()


df['HoursPerPressureUnit'] = df['Study Hours']/df['Academic Pressure']
df['age_stress_ratio'] = df['Age']/df['Academic Pressure']

test['HoursPerPressureUnit'] = test['Study Hours']/test['Academic Pressure']
test['age_stress_ratio'] = test['Age']/test['Academic Pressure']


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
categorical_columns = categorical_columns.drop('Depression')


def _one_hot_encode_columns(df, categorical_columns):
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded

df = _one_hot_encode_columns(df,categorical_columns)
test = _one_hot_encode_columns(test,categorical_columns)


def scale_numerical_features(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

df = scale_numerical_features(df,numerical_columns)
test = scale_numerical_features(test,numerical_columns)


Encoder = LabelEncoder()
df['Depression'] = Encoder.fit_transform(df['Depression'])


X = df.drop('Depression', axis=1)
y = df['Depression'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X_train.head(3)


model = keras.Sequential([
    layers.Dense(64, input_dim=15, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid') 
])

model.summary()


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=32,validation_data=(X_test, y_test))


y_pred = model.predict(X_test)
y_pred_class = (y_pred >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Precision:", precision_score(y_test, y_pred_class))
print("Recall:", recall_score(y_test, y_pred_class))
print("F1-score:", f1_score(y_test, y_pred_class))



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()



cm = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.grid(False)
plt.show()


pred = model.predict(test)
y_pred_submission = (pred >= 0.5).astype(int)
y_pred_submission = pd.Series(y_pred_submission.ravel()).replace({0: 'No', 1: 'Yes'})


submission = pd.DataFrame({
    'index': test_copy['index'],
    'Depression': y_pred_submission.ravel()
})
submission.to_csv('submission', index=False)

