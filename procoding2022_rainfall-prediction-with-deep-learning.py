import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


df_train= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_subm= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


df_test['winddirection'].fillna(df_test['winddirection'].median(), inplace=True)


X = df_train.drop(['rainfall'], axis=1)
y = df_train['rainfall']


X_train = df_train.drop(columns=['day', 'rainfall'])
y_train = df_train['rainfall']
X_test = df_test.drop(columns=['day'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


model = Sequential([
    Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(16, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='sigmoid') 
])


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)


y_pred_prob = model.predict(X_test_scaled)


df_submission = pd.DataFrame(
                {'id': df_test.index, 'rainfall': y_pred_prob.flatten()}
                )


df_submission


df_submission.to_csv('submission.csv', index=False)

