import pandas as pd
from sklearn.model_selection import GroupKFold
import numpy as np

# Load data
train = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/test.csv")




import optuna

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from IPython.display import display



train.head()


print(train.describe().T)


train.info()


train.isnull().sum()


train.isnull().sum()


plt.figure(figsize=(8,5))
sns.histplot(train['pressure'], kde=True, bins=30)
plt.title(f"Distribution of {'pressure'}")
plt.show()


train=train.drop(['id'],axis=1)


num_features = train.select_dtypes(include=[np.number]).columns.tolist()

# Plot histograms for all numeric features
train[num_features].hist(bins=30, figsize=(15,12), layout=(4,3))
plt.suptitle("Feature Distributions")
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(train[num_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


train.shape


test.shape


import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- 1. Basic breath-wise cumulative features ---
    df['u_in_cumsum'] = df.groupby('breath_id')['u_in'].cumsum()
    # cumulative mean = cumulative sum / count so far
    df['u_in_cummean'] = df['u_in_cumsum'] / (df.groupby('breath_id').cumcount() + 1)

    # --- 2. Lag features ---
    for lag in [1, 2, 3]:
        df[f'u_in_lag{lag}'] = df.groupby('breath_id')['u_in'].shift(lag).fillna(0)
        df[f'du_in_lag{lag}'] = df['u_in'] - df[f'u_in_lag{lag}']

    # --- 3. Time-related features ---
    df['time_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['u_in_rate'] = df.groupby('breath_id')['u_in'].diff().fillna(0) / df['time_diff'].replace(0, 1)

    # --- 4. Rolling statistics ---
    df['u_in_rolling_mean'] = (
        df.groupby('breath_id')['u_in']
        .rolling(window=5, min_periods=1).mean()
        .reset_index(0, drop=True)
    )
    df['u_in_rolling_std'] = (
        df.groupby('breath_id')['u_in']
        .rolling(window=5, min_periods=1).std()
        .reset_index(0, drop=True)
        .fillna(0)
    )

    # --- 5. Phase features ---
    df['phase_inhale'] = (df['u_out'] == 0).astype(int)
    df['phase_exhale'] = (df['u_out'] == 1).astype(int)

    # step index (0,1,2,...) within each breath
    step_index = df.groupby('breath_id').cumcount()
    step_count = df.groupby('breath_id')['time_step'].transform('count')
    df['step_pos'] = step_index / step_count

    # --- 6. Statistical features per breath (broadcast back to all rows) ---
    for col in ['u_in', 'u_in_cumsum']:
        df[f'{col}_mean'] = df.groupby('breath_id')[col].transform('mean')
        df[f'{col}_max'] = df.groupby('breath_id')[col].transform('max')
        df[f'{col}_min'] = df.groupby('breath_id')[col].transform('min')
        df[f'{col}_std'] = df.groupby('breath_id')[col].transform('std')

    return df

train = create_features(train)
test = create_features(test)


train.head()


print('train_shape :',train.shape)
print('test_shape :',test.shape)


y = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure', 'breath_id'], axis=1, inplace=True)
test = test.drop(['id', 'breath_id'], axis=1)


y.shape


train.shape


RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)


import numpy as np

train = np.array(train)
y = np.array(y)

SEQ_LEN = 80
num_features = train.shape[1]

print("Train shape before reshape:", train.shape)
print("y shape before reshape:", y.shape)



# Reshape train into (breaths, timesteps, features)
X = train.reshape(-1, SEQ_LEN, num_features)

# Reshape y into (breaths, timesteps, 1)
y = y.reshape(-1, SEQ_LEN, 1)

print("X shape after reshape:", X.shape)
print("y shape after reshape:", y.shape)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)




from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense

model = Sequential()
model.add(Input(shape=(80,num_features)))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.summary()




history=model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1)


# Eğitim ve doğrulama kayıplarının grafiklerini çizme
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss for Final Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


test


y_val_pred = model.predict(X_val)


y_val_pred.shape


from sklearn.metrics import mean_absolute_error

mae_val = mean_absolute_error(y_val.reshape(-1), y_val_pred.reshape(-1))
print("Validation MAE:", mae_val)


import matplotlib.pyplot as plt

# Plot true vs predicted pressures
plt.figure(figsize=(12, 6))
plt.plot(y_val.reshape(-1)[:200], label='True Pressure', color='#1f77b4', linewidth=2)
plt.plot(y_val_pred.reshape(-1)[:200], label='LSTM Prediction', color='#d62728', linestyle='-.', linewidth=2)

# Titles and labels
plt.title("LSTM Predictions", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Pressure", fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()



test_reshaped = test.reshape(-1, SEQ_LEN, num_features)

# Make predictions
predictions = model.predict(test_reshaped)
predictions = predictions.reshape(-1)


import pandas as pd

# Load the original test file again (with id column intact)
test_raw = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/test.csv")

# Grab the ids
test_ids = test_raw['id'].values
print("test_ids shape:", test_ids.shape)

# Now create submission
submission = pd.DataFrame({
    "id": test_ids,
    "pressure": predictions
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")


