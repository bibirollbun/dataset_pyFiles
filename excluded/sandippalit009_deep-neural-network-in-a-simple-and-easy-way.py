import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer, LabelEncoder
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


try:
    df_train = df_train.drop(columns='id')
    df_test = df_test.drop(columns='id')
except:
    print("Columns already deleted")

df_train = df_train.drop_duplicates()
df_train.describe().round(2)


columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

plt.figure(figsize=(16, 4), constrained_layout=True)
for i, col in enumerate(columns, 1):
    plt.subplot(1, 7, i)
    sns.violinplot(y=df_train[col], color='lightblue')
    plt.title(f'Violin Plot of {col}')

plt.show()


plt.figure(figsize=(16, 4), constrained_layout=True)
sns.heatmap(df_train.corr(numeric_only=True), annot = True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


def feature_engineering(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HeartRate_per_Weight'] = df['Heart_Rate'] / df['Weight']
    df['BodyTemp_times_Duration'] = df['Body_Temp'] * df['Duration']
    df['Weight_per_Duration'] = df['Weight'] / df['Duration']
    df['Weight_times_Height'] = df['Weight'] * df['Height']
    df['Age_times_HeartRate'] = df['Age'] * df['Heart_Rate']
    df['Duration_times_HeartRate'] = df['Duration'] * df['Heart_Rate']
    df['HeartRate_per_Duration'] = df['Heart_Rate'] / df['Duration']
    df['BMI_per_Age'] = df['BMI'] / df['Age']
    df['log_Duration'] = np.log1p(df['Duration'])
    df['log_Weight'] = np.log1p(df['Weight'])
    df['exp_neg_Age'] = np.exp(-df['Age'])
    df['Temp_adjusted_HR'] = df['Heart_Rate'] * df['Body_Temp']
    return df

le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_test['Sex'] = le.transform(df_test['Sex'])

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)

X = df_train.drop(['Calories'], axis=1)
y = df_train['Calories']

scaler = PowerTransformer()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(df_test)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_test_poly = poly.transform(X_test_scaled)

y_log = np.log1p(y)


def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256), BatchNormalization(), tf.keras.layers.Activation('relu'),
        Dropout(0.4),
        Dense(128), BatchNormalization(), tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        Dense(64), BatchNormalization(), tf.keras.layers.Activation('relu'),
        Dropout(0.2),
        Dense(32), BatchNormalization(), tf.keras.layers.Activation('relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


model = build_model(input_dim=X_poly.shape[1])

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5),
    ModelCheckpoint("final_model.h5", save_best_only=True)
]

history = model.fit(
    X_poly, y_log,
    validation_split=0.2,
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

best_val_rmse = min(history.history['val_root_mean_squared_error'])
print(f"\nValidation RMSE: {best_val_rmse:.4f}")

preds = model.predict(X_test_poly).squeeze()
preds = np.expm1(preds)


plt.figure(figsize=(16, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.ylim(0.003,0.01)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


df_sub['Calories'] = preds
df_sub.to_csv('submission.csv', index=False)


df_sub.head(10)

