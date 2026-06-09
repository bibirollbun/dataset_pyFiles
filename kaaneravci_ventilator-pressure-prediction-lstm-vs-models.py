import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout, Input,SimpleRNN,GRU
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import load_model


data_path = "/kaggle/input/ventilator-pressure-prediction/train.csv"
df = pd.read_csv(data_path , delimiter = ",")


df.info()


df.shape


df.isna().sum()


df = df.drop(columns= "id")


df.head(10)


# Create subplots for the graphs
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Histogram showing the distribution of Pressure for different R values
axes[0].hist([df[df['R'] == r]['pressure'] for r in df['R'].unique()], bins=30, label=[f'R={r}' for r in df['R'].unique()], edgecolor='black')
axes[0].set_title('Histogram of Pressure for Different R Values')
axes[0].set_xlabel('Pressure')
axes[0].set_ylabel('Frequency')
axes[0].legend(title="R Values")
axes[0].grid(True)

# Histogram showing the distribution of Pressure for different C values
axes[1].hist([df[df['C'] == c]['pressure'] for c in df['C'].unique()], bins=30, label=[f'C={c}' for c in df['C'].unique()], edgecolor='black')
axes[1].set_title('Histogram of Pressure for Different C Values')
axes[1].set_xlabel('Pressure')
axes[1].set_ylabel('Frequency')
axes[1].legend(title="C Values")
axes[1].grid(True)

# Adjust the layout and show the plots
plt.tight_layout()
plt.show()



rc_combination = df.groupby(['R', 'C']).size()

# Plot the distribution of R and C combinations as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(rc_combination, labels=[f'R={r}, C={c}' for r, c in rc_combination.index], autopct='%1.1f%%', startangle=90)
plt.title('Distribution of R and C Combinations')
plt.axis('equal')  
plt.show()


breath_ids = [1, 2, 3, 4]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
axes = axes.flatten()  # Convert the 2D array to 1D for easy indexing

for i, breath_id in enumerate(breath_ids):
    df_breath = df[df['breath_id'] == breath_id]  # Filter data for the current breath ID
    
    if 'time' in df_breath.columns:
        x = df_breath['time']  # Use 'time' if available
    else:
        x = df_breath.index  # Otherwise, use the index

    y = df_breath['pressure']  # Extract pressure values
    axes[i].plot(x, y, label=f'Breath ID {breath_id}')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Pressure')
    axes[i].set_title(f'Breath ID {breath_id}')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()  # Adjust spacing between subplots for better visualization
plt.show()


correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


data_sample = df.sample(frac=0.01, random_state=42)  


data_sample.shape


X = data_sample.drop(columns=['pressure']).values  
y = data_sample['pressure'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


class KerasLSTMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, units=50, optimizer='adam', batch_size=32, epochs=20):
        self.units = units
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], 1)))  
        model.add(LSTM(self.units, return_sequences=True))
        model.add(Dropout(0.5))  
        model.add(LSTM(self.units))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def fit(self, X, y):
        model = self.build_model()
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1) 
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)


model = KerasLSTMRegressor()


param_grid = {
    'units': [50, 100, 150],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [64,128], 
    'epochs': [10],      
}


grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_result = grid.fit(X_train, y_train)


print("Best Parameters:", grid_result.best_params_)


features = ['R', 'C', 'time_step', 'u_in', 'u_out'] 
scaler = MinMaxScaler(feature_range=(0, 1))
df[features] = scaler.fit_transform(df[features])
scaler_pressure = MinMaxScaler(feature_range=(0, 1))
df['pressure'] = scaler_pressure.fit_transform(df[['pressure']])


def create_dataset(df, time_step=1):
    X, y = [], []
    for i in range(len(df) - time_step):
        X.append(df[i:(i + time_step), :-1])  
        y.append(df[i + time_step, -1])  
    return np.array(X), np.array(y)


time_step = 30 

data = df[features + ['pressure']].values

X, y = create_dataset(data, time_step)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2])


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False,random_state = 43)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False,random_state = 44)


rnn_model = Sequential()
rnn_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  
rnn_model.add(SimpleRNN(units=150, return_sequences=False))
rnn_model.add(Dense(units=1)) 
rnn_model.compile(optimizer='adam', loss='mean_squared_error')


rnn_history = rnn_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1)


rnn_model.save('rnn_pressure_model.keras')
print("Model has been saved as 'rnn_pressure_model.keras")


plt.plot(rnn_history.history['loss'], label='Train Loss')
plt.plot(rnn_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss for Final Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


gru_model = Sequential()
gru_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  
gru_model.add(GRU(units=150, return_sequences=False))
gru_model.add(Dense(units=1)) 
gru_model.compile(optimizer='adam', loss='mean_squared_error')


gru_history = gru_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1)


gru_model.save('gru_pressure_model.keras')
print("Model has been saved as 'gru_pressure_model.keras'")


# Eğitim ve doğrulama kayıplarının grafiklerini çizme
plt.plot(gru_history.history['loss'], label='Train Loss')
plt.plot(gru_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss for Final Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


lstm_model = Sequential()
lstm_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  
lstm_model.add(LSTM(units=150, return_sequences=False))
lstm_model.add(Dense(units=1)) 
lstm_model.compile(optimizer='adam', loss='mean_squared_error')


lstm_history = lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1)


lstm_model.save('lstm_pressure_model.keras')
print("Model has been saved as 'lstm_pressure_model.keras'")


plt.plot(lstm_history.history['loss'], label='Train Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss for Final Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


rnn_model = load_model("rnn_pressure_model.keras")
gru_model = load_model("gru_pressure_model.keras")
lstm_model = load_model("lstm_pressure_model.keras")


# Making Predictions on Test Data and Inverse Scaling the Results
predictions_model1 = scaler_pressure.inverse_transform(rnn_model.predict(X_test))
predictions_model2 = scaler_pressure.inverse_transform(gru_model.predict(X_test))
predictions_model3 = scaler_pressure.inverse_transform(lstm_model.predict(X_test))
y_test_actual = scaler_pressure.inverse_transform(y_test.reshape(-1, 1))


fig, axes = plt.subplots(3, 1, figsize=(10, 18))  
axes = axes.flatten() 

# RNN Model Predictions
axes[0].plot(y_test_actual, label='Actual Pressure', color='#1f77b4', linewidth=2)  
axes[0].plot(predictions_model1, label='RNN Prediction', color='#ff7f0e', linestyle='-', linewidth=2) 
axes[0].set_title("RNN Predictions vs Actual", fontsize=14)
axes[0].set_xlabel("Time", fontsize=12)
axes[0].set_ylabel("Pressure", fontsize=12)
axes[0].legend()
axes[0].grid(True)

# GRU Model Predictions
axes[1].plot(y_test_actual, label='Actual Pressure', color='#1f77b4', linewidth=2)  
axes[1].plot(predictions_model2, label='GRU Prediction', color='#2ca02c', linestyle='--', linewidth=2)  
axes[1].set_title("GRU Predictions vs Actual", fontsize=14)
axes[1].set_xlabel("Time", fontsize=12)
axes[1].set_ylabel("Pressure", fontsize=12)
axes[1].legend()
axes[1].grid(True)

# LSTM Model Predictions
axes[2].plot(y_test_actual, label='Actual Pressure', color='#1f77b4', linewidth=2)  
axes[2].plot(predictions_model3, label='LSTM Prediction', color='#d62728', linestyle='-.', linewidth=2)  
axes[2].set_title("LSTM Predictions vs Actual", fontsize=14)
axes[2].set_xlabel("Time", fontsize=12)
axes[2].set_ylabel("Pressure", fontsize=12)
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()  
plt.show()



df_results = pd.DataFrame({
    "Actual Values": y_test_actual.flatten(),
    "RNN Predictions": predictions_model1.flatten(),
    "GRU Predictions": predictions_model2.flatten(),
    "LSTM Predictions": predictions_model3.flatten()
})

print(df_results.head(10))


def evaluate_model(predictions, y_test_actual, model):
    mse = mean_squared_error(y_test_actual, predictions)
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, predictions)
    total_params = model.count_params()
    memory_usage = total_params * 4 / (1024 ** 2)  # Assuming 4 bytes per parameter
    
    return mse, mae, rmse, r2, total_params, memory_usage


model_results = []

# Model 1 (RNN)
mse1, mae1, rmse1, r21, total_params1, memory_usage1 = evaluate_model(predictions_model1, y_test_actual, rnn_model)
model_results.append(["RNN Model", mse1, mae1, rmse1, r21, total_params1, memory_usage1])

# Model 2 (GRU)
mse2, mae2, rmse2, r22, total_params2, memory_usage2 = evaluate_model(predictions_model2, y_test_actual, gru_model)
model_results.append(["GRU Model", mse2, mae2, rmse2, r22, total_params2, memory_usage2])

# Model 3 (LSTM)
mse3, mae3, rmse3, r23, total_params3, memory_usage3 = evaluate_model(predictions_model3, y_test_actual, lstm_model)
model_results.append(["LSTM Model", mse3, mae3, rmse3, r23, total_params3, memory_usage3])


results_df = pd.DataFrame(model_results, columns=["Model", "MSE", "MAE", "RMSE", "R²", "Total Parameters", "Memory Usage (MB)"])
print(results_df.to_string(index=False))

