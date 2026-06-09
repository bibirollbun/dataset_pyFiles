# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


test_df = pd.read_parquet('/kaggle/input/crypto-market-data/feature_agg')
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')


test_df = test_df.rename(columns={'prediction':'label'})
test_df.head()


import dask.dataframe as dd
from sklearn.preprocessing import MinMaxScaler
import gc


df = pd.concat([train_df, test_df]).reset_index(drop=True)
dask_df = dd.from_pandas(df, npartitions=100)
dask_df.to_parquet("my_data.parquet", compression="snappy")

del test_df, train_df, df,
gc.collect #free up memory




X_features = dd.read_parquet("my_data.parquet").compute()

print(X_features.head())
X_features.drop('label', axis=1, inplace=True)
X_clean = X_features.astype(np.float32)
print(X_clean.shape)
del X_features
gc.collect
clean_dask = dd.from_pandas(X_clean, npartitions=100)
del X_clean
gc.collect




scaler = MinMaxScaler()

sample = clean_dask.sample(frac=0.01).compute()
scaler = MinMaxScaler().fit(sample)

# Define function to scale each partition
def scale_partition(partition):
    return pd.DataFrame(scaler.transform(partition), columns=partition.columns)

X_scaled = clean_dask.map_partitions(scale_partition)




X_scaled = X_scaled.compute()
print(X_scaled.shape)


del clean_dask
gc.collect


import shutil

shutil.rmtree("./pca_chunks", ignore_errors=True)



import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt



print("3. DIMENSIONALITY REDUCTION")
print("="*50)

chunk_size = 1000
ipca = IncrementalPCA(n_components=100)
for i in range(0, X_scaled.shape[0], chunk_size):
    chunk = X_scaled[i:i+chunk_size]
    X_pca = ipca.fit_transform(chunk)
    temp_df = pd.DataFrame(X_pca, columns=[f'pc{i}' for i in range(X_pca.shape[1])])
    dd_chunk = dd.from_pandas(temp_df, npartitions=1)
    
    # Save each chunk into the same directory
    dd_chunk.to_parquet("pca_chunks/", append=True, ignore_divisions=True)

# 6. Explained variance analysis
explained_variance_ratio = ipca.explained_variance_ratio_
cumsum_variance = np.cumsum(explained_variance_ratio)
n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1

print(f"Number of components needed for 90% variance: {n_components_90}")
print(f"Number of components needed for 95% variance: {n_components_95}")

# 7. Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1),
         explained_variance_ratio, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA - Individual Explained Variance')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum_variance) + 1),
         cumsum_variance, 'ro-')
plt.axhline(y=0.90, color='g', linestyle='--', label='90% variance')
plt.axhline(y=0.95, color='b', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




del X_scaled, temp_df, X_pca, dd_chunk, chunk
gc.collect


pca_df = dd.read_parquet('pca_chunks/').compute()


from sklearn.cluster import KMeans


print("\n" + "="*50)
print("5. CLUSTERING ANALYSIS")
print("="*50)


# K-means clustering on PCA-reduced data
print("Performing K-means clustering...")
n_clusters_range = range(2, 11)
inertias = []

for k in n_clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_df)  # Use first 50 PCA components
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('K-means Clustering - Elbow Method')
plt.grid(True)
plt.show()


y_aligned = dd.read_parquet('my_data.parquet', columns=['label']).compute()
y_aligned.to_parquet('Y.parquet')



optimal_k = 4  # You can adjust based on elbow curve
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans_final.fit_predict(pca_df)
# Analyze clusters
cluster_analysis = pd.DataFrame({
    'cluster': clusters,
    'target': y_aligned.squeeze()
})

print(f"\nCluster analysis with {optimal_k} clusters:")
cluster_stats = cluster_analysis.groupby('cluster')['target'].agg(['count', 'mean', 'std'])
print(cluster_stats)

# Visualize clusters using first 2 PCA components
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=clusters, alpha=0.6, cmap='tab10')
plt.colorbar(scatter, label='Cluster')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-means Clusters in PCA Space')

plt.subplot(1, 2, 2)
scatter = plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=y_aligned.squeeze(), alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Target Value')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Target Values in PCA Space')

plt.tight_layout()
plt.show()





del y_aligned
gc.collect


#add distance to cluster centroid as a feature
from scipy.stats import pearsonr

from scipy.spatial.distance import cdist
distances = cdist(pca_df, kmeans_final.cluster_centers_, metric='euclidean')  # shape: (n_samples, n_clusters)
for i in range(distances.shape[1]):
    pca_df[f'distance_to_centroid_{i}'] = distances[:, i]
mean_distances = distances.mean(axis=1)  # shape: (n_samples,)
pca_df['mean_distance_to_centroids'] = mean_distances
pca_df['cluster_label'] = clusters
pca_df = pd.get_dummies(pca_df, columns=['cluster_label'], prefix='cluster')
pca_df.to_parquet('pca_df.parquet')


"""#Import required libraries
import math
import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import save_model
from keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Dropout
from keras.layers import MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D, Embedding, Add
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from keras.models import Sequential, Model
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, space_eval
import tensorflow as tf
from keras.models import load_model
import os
from tensorflow.keras import backend as K
from optuna.integration import TFKerasPruningCallback


# Apply Holt-Winters Exponential Smoothing with multiplicative trend and seasonality
hw_model = ExponentialSmoothing(close_prices, trend='mul', seasonal='mul', seasonal_periods=365)

hw_fit = hw_model.fit()

# View optimized parameters
optimized_params = hw_fit.params

# Extract level, trend, and seasonal components
df['HW_Level'] = hw_fit.level
df['HW_Trend'] = hw_fit.trend
df['HW_Seasonal'] = hw_fit.season

# Deseasonalize the data
df['Deseasonalized'] = df['Close'] / (df['HW_Seasonal'] * df['HW_Level'])

# Length of training data
values = df['Deseasonalized'].values
training_data_len = math.ceil(len(values) * 0.80)

time_step = 30

# Split data into training and test sets before scaling
train_values = values[:training_data_len]

test_values = values[training_data_len - time_step:]

# Normalize using only training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_values.reshape(-1, 1))
scaled_test = scaler.transform(test_values.reshape(-1, 1))

# Build training sequences
x_train, y_train = [], []

for i in range(time_step, len(scaled_train)):
    x_train.append(scaled_train[i - time_step:i, 0])
    y_train.append(scaled_train[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape x_train to have the shape (samples, time_step, 1) for the transformer model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build test sequences
x_test = []

for i in range(time_step, len(scaled_test)):
    x_test.append(scaled_test[i - time_step:i, 0])

x_test = np.array(x_test)

# Reshape x_test to have the shape (samples, time_step, 1)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# y_test is taken from the original close_prices from the start of the test period onward
y_test = close_prices.values[training_data_len:]

# Define the Mish activation function
def mish(x):
    return x * K.tanh(K.softplus(x))

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
    units = trial.suggest_int('units', 20, 50, step=5)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 50, 150, step=10)
    num_blocks = trial.suggest_int('num_blocks', 1, 4)
    num_heads = trial.suggest_int('num_heads', 2, 10, step=2)
    head_size = trial.suggest_int('head_size', 8, 64, step=8)

    # Define the model architecture
    def create_helformer_model(input_shape):
        inputs = Input(shape=input_shape)
        x = inputs

        for _ in range(num_blocks):
            x_norm1 = LayerNormalization(epsilon=1e-6)(x)
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout_rate)(x_norm1, x_norm1)
            x = x + attention_output

            x_norm2 = LayerNormalization(epsilon=1e-6)(x)
            x = x + x_norm2

        # LSTM layer
        x = LSTM(units, activation=mish, return_sequences=False)(x)

        # Output layer
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    # Create and compile the model
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = create_helformer_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=0,
        callbacks=[TFKerasPruningCallback(trial, 'val_loss')]
    )

    # Predictions on validation data
    val_loss = min(history.history['val_loss'])
    return val_loss

# Run the Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Retrain the model using the best hyperparameters
def create_best_helformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(best_params['num_blocks']):
        x_norm1 = LayerNormalization(epsilon=1e-6)(x)
        attention_output = MultiHeadAttention(num_heads=best_params['num_heads'], key_dim=best_params['head_size'], dropout=best_params['dropout_rate'])(x_norm1, x_norm1)
        x = x + attention_output

        x_norm2 = LayerNormalization(epsilon=1e-6)(x)
        x = x + x_norm2

    # LSTM layer
    x = LSTM(best_params['units'], activation=mish, return_sequences=False)(x)

    # Output layer
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model with the best parameters
final_model_Helformer = create_best_helformer_model((x_train.shape[1], x_train.shape[2]))
final_model_Helformer.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')

# Train the model
print("Training final Helformer model")
final_history_Helformer = final_model_Helformer.fit(x_train, y_train, batch_size=best_params['batch_size'], epochs=best_params['epochs'], verbose=0)# Verbose set to 1 for detailed output during training

final_model_Helformer.save('Main_Helformer.h5')

# Predictions on training data
train_predictions = final_model_Helformer.predict(x_train)
train_predictions = scaler.inverse_transform(train_predictions).flatten()

# Restore level and seasonal components for training predictions
train_predictions = train_predictions * df['HW_Seasonal'].values[time_step:training_data_len] * df['HW_Level'].values[time_step:training_data_len]

y_train_rescaled = scaler.inverse_transform([y_train]).flatten()
y_train_rescaled = y_train_rescaled * df['HW_Seasonal'].values[time_step:training_data_len] * df['HW_Level'].values[time_step:training_data_len]

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to calculate KGE (Kling-Gupta Efficiency)
def kge(y_true, y_pred):
    cc = np.corrcoef(y_true, y_pred)[0, 1]  # Correlation coefficient
    alpha = np.std(y_pred) / np.std(y_true)  # Variability ratio
    beta = np.mean(y_pred) / np.mean(y_true)  # Bias ratio
    return 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

# Predictions on testing data
print("Predicting test data")
test_predictions = final_model_Helformer.predict(x_test)
test_predictions = scaler.inverse_transform(test_predictions)
test_predictions = test_predictions.flatten()

# Restore level and seasonal components for test predictions
test_predictions = test_predictions * df['HW_Seasonal'].values[training_data_len:] * df['HW_Level'].values[training_data_len:]

# Calculate metrics for training data
train_mse = mean_squared_error(y_train_rescaled, train_predictions)
train_rmse = np.sqrt(train_mse)
train_mape = mean_absolute_percentage_error(y_train_rescaled, train_predictions)
train_mae = mean_absolute_error(y_train_rescaled, train_predictions)
train_r2 = r2_score(y_train_rescaled, train_predictions)
train_evs = explained_variance_score(y_train_rescaled, train_predictions)
train_kge = kge(y_train_rescaled, train_predictions)

# Calculate metrics for testing data
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_mape = mean_absolute_percentage_error(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)
test_evs = explained_variance_score(y_test, test_predictions)
test_kge = kge(y_test, test_predictions)

# Get current date and time
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
readable_time = now.strftime("%Y-%m-%d %H:%M:%S")

# Define file path and name
filename = f"Performance_metrics_{timestamp}.txt"
filepath = os.path.join("/results/", filename)


# Write metrics to file
with open(filepath, 'w') as f:               # Change filename to filepath for codeocean
    f.write(f"::     ::\n")
    f.write(f"Training Data Evaluation Metrics\n")
    f.write(f"Timestamp: {readable_time}\n\n")
    f.write(f"MSE: {train_mse:.4f}\n")
    f.write(f"RMSE: {train_rmse:.4f}\n")
    f.write(f"MAPE: {train_mape:.4f}%\n")
    f.write(f"MAE: {train_mae:.4f}\n")
    f.write(f"R^2 Score: {train_r2:.4f}\n")
    f.write(f"Explained Variance Score (EVS): {train_evs:.4f}\n")
    f.write(f"KGE: {train_kge:.4f}\n")
    
    f.write(f"-----\n")
    f.write(f"Testing Data Evaluation Metrics\n")
    f.write(f"Timestamp: {readable_time}\n\n")
    f.write(f"MSE: {test_mse:.4f}\n")
    f.write(f"RMSE: {test_rmse:.4f}\n")
    f.write(f"MAPE: {test_mape:.4f}%\n")
    f.write(f"MAE: {test_mae:.4f}\n")
    f.write(f"R^2 Score: {test_r2:.4f}\n")
    f.write(f"Explained Variance Score (EVS): {test_evs:.4f}\n")
    f.write(f"KGE: {test_kge:.4f}\n")

print(f"Metrics saved to: {filename}")


# Plot the data
plt.figure(figsize=(16,8), dpi=300)
plt.title('BTC Prediction using Helformer')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)

# Plot original data
plt.plot(df['Close'], label='True Values')

# Create a list of NaNs to fill in the missing values in the plot
train_predictions_plot = [np.nan] * len(df)
train_predictions_plot[time_step:training_data_len] = train_predictions

# Add training predictions to the plot
plt.plot(df.index, train_predictions_plot, color='red', label='Train Predictions')

# Create a list of NaNs to fill in the missing values in the plot
test_predictions_plot = [np.nan] * len(df)
test_predictions_plot[training_data_len:] = test_predictions

# Add test predictions to the plot
plt.plot(df.index, test_predictions_plot, color='black', label='Test Predictions')

plt.legend()

plt.savefig ("/results/Helformer_Predictions.png")


# Assuming observed_values = y_test and predicted_values = test_predictions
observed_values = y_test
predicted_values = test_predictions

# Initialize an array to store returns with transaction cost
returns_with_cost = []

for i in range(1, len(observed_values)):
    trade_signal = np.sign(predicted_values[i] - observed_values[i-1])

    Rt = np.log(observed_values[i] / observed_values[i-1]) * trade_signal

    # Apply transaction cost
    Rt_after_cost = Rt - 0.01 * abs(Rt)

    returns_with_cost.append(Rt_after_cost)

# Calculate net value (NV)
net_value = np.cumsum(returns_with_cost) + 1

# Calculate total returns
total_return = net_value[-1] - 1
total_return_pcnt = total_return * 100

# Calculate volatility (standard deviation of returns)
volatility = np.std(returns_with_cost)

# Calculate max drawdown
drawdowns = 1 - net_value / np.maximum.accumulate(net_value)
max_drawdown = np.max(drawdowns)
max_drawdown_inv = max_drawdown * -1

# Set annual risk-free rate (Rf) to 1%
risk_free_rate_annual = 0.01
risk_free_rate_daily = risk_free_rate_annual / 365  # Convert to daily risk-free rate

# Calculate Sharpe ratio
expected_return = np.mean(returns_with_cost)
sharpe_ratio = (expected_return - risk_free_rate_daily) / volatility * np.sqrt(365)

# Define file path and name
filename = f"Trading_metrics.txt"
filepath = os.path.join("/results", filename)    #for code ocean

# Write metrics to file
with open(filepath, 'w') as f:                 
    f.write(f"::     ::\n")
    f.write(f"Trading Metrics\n")
    f.write(f"Timestamp: {readable_time}\n")
    f.write(f"Total Return: {total_return_pcnt:.4f}%\n")
    f.write(f"Volatility: {volatility:.4f}\n")
    f.write(f"Max Drawdown: {max_drawdown_inv:.4f}\n")
    f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")

print(f"Trading metrics saved to: {filename}")     # Change filename to filepath for codeocean
"""

