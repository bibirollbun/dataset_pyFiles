import pandas as pd

df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')


df.head()


print(df.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix=df[df.select_dtypes(include=['number']).columns.tolist()].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Select categorical columns
categorical_cols = df.select_dtypes(include='object').columns

# Plot value counts for each categorical column
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    df[col].value_counts(dropna=False).plot(kind='bar')
    plt.title(f'Value Counts for {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



# copy for processing
df_proc = df.iloc[:100000].copy()

# Drop
df_proc = df_proc.drop(columns=["id", "Podcast_Name", "Episode_Title"])

# Handle missing numeric values
df_proc['Episode_Length_minutes'] = df_proc['Episode_Length_minutes'].fillna(df_proc['Episode_Length_minutes'].mean())
df_proc['Guest_Popularity_percentage'] = df_proc['Guest_Popularity_percentage'].fillna(df_proc['Guest_Popularity_percentage'].mean())
df_proc['Number_of_Ads'] = df_proc['Number_of_Ads'].fillna(df_proc['Number_of_Ads'].mean())


# One-hot
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
df_proc = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=False,dtype=int)

# features and target
X = df_proc.drop(columns=['Listening_Time_minutes'])
y = df_proc['Listening_Time_minutes']



from sklearn.model_selection import train_test_split

# Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


from sklearn.preprocessing import StandardScaler


X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()


import numpy as np
import tensorflow as tf

# Reshape inputs for RNN
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build RNN model for regression
rnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, X_train_scaled.shape[1])),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
tf.keras.layers.Dense(1)
])


optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)

rnn_model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])


rnn_model.fit(X_train_rnn, y_train_scaled,epochs=20, batch_size=32, validation_data=(X_test_rnn, y_test_scaled))
loss, mae = rnn_model.evaluate(X_test_rnn, y_test_scaled, verbose=0)


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Predict
y_pred_scaled = rnn_model.predict(X_test_rnn)
y_pred_scaled_rnn = y_pred_scaled
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))


mae_real = mean_absolute_error(y_true, y_pred)
rmse_real = np.sqrt(mean_squared_error(y_true, y_pred))

print("RNN Mean Absolute Error (real):", mae_real)
print("RNN Root Mean Squared Error (real):", rmse_real)



import tensorflow as tf

fc_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

fc_model.compile(optimizer='adam',
                 loss='mse',
                 metrics=['mae'])

# Train
fc_model.fit(X_train_scaled, y_train_scaled,
             epochs=20,
             batch_size=32,
             validation_data=(X_test_scaled, y_test_scaled))

# Evaluate
loss, mae = fc_model.evaluate(X_test_scaled, y_test_scaled, verbose=0)



y_pred_scaled = fc_model.predict(X_test_scaled)
y_pred_scaled_fcn = y_pred_scaled
# Inverse transform
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

mae_real = mean_absolute_error(y_true, y_pred)
rmse_real = np.sqrt(mean_squared_error(y_true, y_pred))

print("FDD Mean Absolute Error (real):", mae_real)
print("FDD Root Mean Squared Error (real):", rmse_real)



model1 = fc_model

# List of optimizers to test
optimizers = [
    tf.keras.optimizers.Adam(learning_rate=0.001),         # Adam
    tf.keras.optimizers.RMSprop(learning_rate=0.001),      # RMSprop
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)  # SGD with momentum
]

# Dictionary to store results
results = {}

# Loop over each optimizer, compile and train the model
for optimizer in optimizers:
    print(f"Training with optimizer: {optimizer.__class__.__name__}")
    
    # Clone the model to reset weights
    new_model = tf.keras.models.clone_model(model1)
    new_model.set_weights(model1.get_weights())
    
    # Compile the new model with the current optimizer
    new_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Train the model (no validation data used)
    history = new_model.fit(X_train_scaled, y_train_scaled, 
                            epochs=20, batch_size=32, verbose=0)
    
    # Store the final training MAE for comparison
    final_train_mae = history.history['mae'][-1]
    results[optimizer.__class__.__name__] = final_train_mae

# Print the results for comparison
print("Optimization Algorithm Comparison (Training MAE):")
for optimizer_name, train_mae in results.items():
    print(f"{optimizer_name}: Final Training MAE = {train_mae:.4f}")



model2 = rnn_model

# List of optimizers to test
optimizers = [
    tf.keras.optimizers.Adam(learning_rate=0.001),         # Adam
    tf.keras.optimizers.RMSprop(learning_rate=0.001),      # RMSprop
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)  # SGD with momentum
]

# Dictionary to store results
results = {}

# Loop over each optimizer, compile and train the model
for optimizer in optimizers:
    print(f"Training with optimizer: {optimizer.__class__.__name__}")
    
    # Clone the model to reset weights
    new_model = tf.keras.models.clone_model(model1)
    new_model.set_weights(model1.get_weights())
    
    # Compile the new model with the current optimizer
    new_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Train the model (no validation data used)
    history = new_model.fit(X_train_scaled, y_train_scaled, 
                            epochs=20, batch_size=32, verbose=0)
    
    # Store the final training MAE for comparison
    final_train_mae = history.history['mae'][-1]
    results[optimizer.__class__.__name__] = final_train_mae

# Print the results for comparison
print("Optimization Algorithm Comparison (Training MAE):")
for optimizer_name, train_mae in results.items():
    print(f"{optimizer_name}: Final Training MAE = {train_mae:.4f}")



import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error


X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


rnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, X_train_scaled.shape[1])),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)
])


optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)

rnn_model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])


rnn_model.fit(X_train_rnn, y_train_scaled,
              epochs=20,
              batch_size=32,
              validation_data=(X_test_rnn, y_test_scaled))


loss, mae = rnn_model.evaluate(X_test_rnn, y_test_scaled, verbose=0)
print("RNN MAE (scaled):", mae)
print("RNN MSE (scaled):", loss)

# === Real-world Evaluation ===
y_pred_scaled = rnn_model.predict(X_test_rnn)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

mae_real = mean_absolute_error(y_true, y_pred)
rmse_real = np.sqrt(mean_squared_error(y_true, y_pred))

print("RNN MAE (real):", mae_real)
print("RNN RMSE (real):", rmse_real)



!pip install -q tensorflow

import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error

X_tr = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_te = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])
y_tr = y_train_scaled
y_te = y_test_scaled

param_grid = {
    'units1': [32, 64],
    'units2': [16, 32],
    'lr':     [1e-3, 1e-4],
    'batch': [16, 32]
}

best_mae = float('inf')
best_params = None
best_model = None

for units1, units2, lr, batch_size in itertools.product(
        param_grid['units1'],
        param_grid['units2'],
        param_grid['lr'],
        param_grid['batch']):

    model = keras.Sequential([
        keras.layers.Input(shape=(1, X_tr.shape[2])),
        keras.layers.LSTM(units1, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(units2),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss='mse'
    )

    model.fit(X_tr, y_tr, epochs=10, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X_te, batch_size=batch_size, verbose=0)
    y_pred_real = y_scaler.inverse_transform(y_pred)
    y_true_real = y_scaler.inverse_transform(y_te.reshape(-1,1))
    mae_real = mean_absolute_error(y_true_real, y_pred_real)

    print(f"units1={units1}, units2={units2}, lr={lr}, batch={batch_size} -> MAE={mae_real:.4f}")

    if mae_real < best_mae:
        best_mae = mae_real
        best_params = (units1, units2, lr, batch_size)
        best_model = keras.models.clone_model(model)
        best_model.set_weights(model.get_weights())

print("\nBest MAE:", best_mae, "with params:", best_params)

u1, u2, lr, bs = best_params
final_model = keras.Sequential([
    keras.layers.Input(shape=(1, X_tr.shape[2])),
    keras.layers.LSTM(u1, return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(u2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1)
])
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
    loss='mse',
    metrics=['mae']
)
final_model.fit(X_tr, y_tr, epochs=20, batch_size=bs,
                validation_data=(X_te, y_te), verbose=1)

loss, mae = final_model.evaluate(X_te, y_te, verbose=0)
y_pred = final_model.predict(X_te)
y_pred_real = y_scaler.inverse_transform(y_pred)
y_true_real = y_scaler.inverse_transform(y_te.reshape(-1,1))
mae_real = mean_absolute_error(y_true_real, y_pred_real)
rmse_real = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
print("Test MAE (real):", mae_real)
print("Test RMSE (real):", rmse_real)



!pip install -q optuna

import optuna
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from functools import partial

X_tr = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_te = X_test_scaled .reshape(-1, 1, X_test_scaled .shape[1])
y_tr = y_train_scaled
y_te = y_test_scaled

def objective(trial):
    units1 = trial.suggest_categorical('units1', [32, 64, 128])
    units2 = trial.suggest_categorical('units2', [16, 32, 64])
    lr     = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    bs     = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    model = keras.Sequential([
        keras.layers.Input((1, X_tr.shape[2])),
        keras.layers.LSTM(units1, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(units2),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss='mse'
    )

    model.fit(X_tr, y_tr, epochs=5, batch_size=bs, verbose=0)
    
    y_pred = model.predict(X_te, batch_size=bs, verbose=0)
    y_pred_real = y_scaler.inverse_transform(y_pred)
    y_true_real = y_scaler.inverse_transform(y_te.reshape(-1,1))
    mae = mean_absolute_error(y_true_real, y_pred_real)
    return mae


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("Best MAE:", study.best_value)
print("Best params:", study.best_params)


best = study.best_params
model = keras.Sequential([
    keras.layers.Input((1, X_tr.shape[2])),
    keras.layers.LSTM(best['units1'], return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(best['units2']),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1)
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best['lr'], clipnorm=1.0),
    loss='mse',
    metrics=['mae']
)
model.fit(X_tr, y_tr, epochs=20, batch_size=best['batch_size'],
          validation_data=(X_te, y_te), verbose=1)



import numpy as np
from tensorflow import keras
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# Manually set best params
u1 = 128
u2 = 32
lr = 0.0017
bs = 64

X_tr = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_te = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])
y_tr = y_train_scaled
y_te = y_test_scaled

model = keras.Sequential([
    keras.layers.Input(shape=(1, X_tr.shape[2])),
    keras.layers.LSTM(u1, return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(u2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1)
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
    loss='mse',
    metrics=['mae']
)

es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_tr, y_tr,
    epochs=50,
    batch_size=bs,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)

y_pred_scaled = model.predict(X_te, batch_size=bs)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_te.reshape(-1,1))

# Compute metrics
mae   = mean_absolute_error(y_true, y_pred)
rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
mape  = mean_absolute_percentage_error(y_true, y_pred)
r2    = r2_score(y_true, y_pred)

print(f"Test MAE : {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAPE: {mape:.2%}")
print(f"Test R2  : {r2:.4f}")



# For FCN
fc_history = fc_model.fit(
    X_train_scaled, y_train_scaled,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_scaled)
)

# For RNN
rnn_history = rnn_model.fit(
    X_train_rnn, y_train_scaled,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_rnn, y_test_scaled)
)



import matplotlib.pyplot as plt

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fc_history.history['loss'], label='FCN Train Loss')
plt.plot(fc_history.history['val_loss'], label='FCN Val Loss')
plt.plot(rnn_history.history['loss'], label='RNN Train Loss')
plt.plot(rnn_history.history['val_loss'], label='RNN Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(fc_history.history['mae'], label='FCN Train MAE')
plt.plot(fc_history.history['val_mae'], label='FCN Val MAE')
plt.plot(rnn_history.history['mae'], label='RNN Train MAE')
plt.plot(rnn_history.history['val_mae'], label='RNN Val MAE')
plt.title('MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



import seaborn as sns

residuals_rnn = y_true.flatten() - y_pred_scaled_rnn.flatten()
residuals_fcn = y_true.flatten() - y_pred_scaled_fcn.flatten()

plt.figure(figsize=(10, 6))
sns.histplot(residuals_rnn, kde=True, color="blue", label="RNN Residuals", stat="density", bins=30)
sns.histplot(residuals_fcn, kde=True, color="orange", label="FCN Residuals", stat="density", bins=30)
plt.axvline(0, color='red', linestyle='--')
plt.title("Residual Distribution Comparison")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()



import pandas as pd
sns.boxplot(data=pd.DataFrame({
    'RNN Absolute Error': abs(residuals_rnn),
    'FCN Absolute Error': abs(residuals_fcn)
}))





