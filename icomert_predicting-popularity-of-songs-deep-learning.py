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
import tensorflow as tf
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('/kaggle/input/spotify-popularity-prediction-v2/train.csv')


df.head()


df.info()


df.isnull().sum()


plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=30, kde=True, color='purple')
plt.title('Distribution of Popularity', fontsize=18)
plt.xlabel('Popularity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid()
plt.show()


#correlation heatmap
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidths=.5)
plt.title('Feature Correlation Heatmap', fontsize=18)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.show()


#preprocessing the training data
train = pd.read_csv('/kaggle/input/spotify-popularity-prediction-v2/train.csv')
test = pd.read_csv('/kaggle/input/spotify-popularity-prediction-v2/test.csv')

x = train.drop(columns=['id', 'artists','name','release_date','popularity'])
y = train['popularity']

#one-hot encoding categorical features in training data
x = pd.get_dummies(x, drop_first=True)


#splitting the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

#scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


#building the deep learning model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

#compiling the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])


#training the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, batch_size=32,verbose=0)


#evaluating the model on validation set
val_predictions = model.predict(x_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f'Validation RMSE: {val_rmse}')


model.summary()


#RMSE over epochs
rmse_per_epoch = [val_loss ** 0.5 for val_loss in history.history['val_loss']]
plt.figure(figsize=(10, 6))
plt.plot(rmse_per_epoch, label='Validation RMSE', color='blue', lw=2)
plt.title("RMSE Over Epochs", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.grid(alpha=0.3)
plt.legend(fontsize=10, shadow=True)
plt.tight_layout()
plt.show()


#preprocessing test data
submission = pd.DataFrame({'id': test['id']})
test = test.drop(columns=['id'])

#one-hot encoding the test data and align columns
test = pd.get_dummies(test, drop_first=True)
test = test.reindex(columns=x.columns, fill_value=0)

#scaling
test_scaled = scaler.transform(test)


#making predictions on test data
predictions = model.predict(test_scaled)

#adding predicted values to df and create submission file
submission['popularity'] = predictions.flatten()
submission.to_csv('submission.csv', index=False)

