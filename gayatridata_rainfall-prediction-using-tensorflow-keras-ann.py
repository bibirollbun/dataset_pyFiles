import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


train_df= pd.read_csv("/kaggle/input/rainfall-prediction-dataset/train.csv").set_index('id')
test_df= pd.read_csv("/kaggle/input/rainfall-prediction-dataset/test.csv").set_index('id')
submission_df= pd.read_csv("/kaggle/input/rainfall-prediction-dataset/sample_submission.csv")


train_df.info()


train_df.isna().sum()


test_df.isna().sum()


test_df['winddirection'].fillna(test_df['winddirection'].median(), inplace=True)


sns.countplot(data = train_df,x = 'rainfall')


corr = train_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(data=corr, annot=True, linewidths=0.2);


X_train = train_df.drop(columns=['day', 'rainfall'])
y_train = train_df['rainfall']
X_test = test_df.drop(columns=['day'])

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Initialize Neural Network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile Model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2, 
                    callbacks=[early_stopping], verbose=1)


# Make Predictions
y_pred = model.predict(X_test_scaled).flatten()


submission_df['rainfall'] = y_pred
submission_df.to_csv('submission_keras1.csv', index=False)
submission_df




