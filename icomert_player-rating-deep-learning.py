import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


train = pd.read_csv('/kaggle/input/scrabble-player-rating/train.csv')
games = pd.read_csv('/kaggle/input/scrabble-player-rating/games.csv')
turns = pd.read_csv('/kaggle/input/scrabble-player-rating/turns.csv')


#merging datasets on 'game_id'
train_merged = train.merge(games, on='game_id', how='left').merge(turns, on='game_id', how='left')


train_merged.head()


train_merged.shape


train_merged.isnull().sum()


train_merged.info()


#correlation heatmap
numeric_columns = train_merged.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features in train_merged", fontsize=16, pad=20)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.show()


#filling missing values for  train_merged
train_merged['turn_type'] = train_merged['turn_type'].fillna("UNKNOWN")


#dropping the columns that are not used in model
train_merged.drop(['game_id', 'rack', 'location', 'move', 'nickname_x', 'nickname_y', 'created_at'], axis=1, inplace=True)


#ensuring reproducibility
np.random.seed(42)

#one-hot encoding
categorical_cols = ['first', 'game_end_reason', 'rating_mode', 'turn_type', 'time_control_name','lexicon']
train_merged = pd.get_dummies(train_merged, columns=categorical_cols)

#stratified sampling 20% of data
sample_size = int(len(train_merged) * 0.2)
train_sampled, _ = train_test_split(train_merged, train_size=sample_size, stratify=train_merged['rating'], random_state=42)

#separating features and target variable for sampled data
train_features = train_sampled.drop(['rating'], axis=1)
train_target = train_sampled['rating']


#train-test split
x_train, x_val, y_train, y_val = train_test_split(train_features, train_target, test_size=0.2, random_state=42)

#scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

#building the model
model = Sequential([
    Dense(128, activation='relu', input_dim=x_train.shape[1]),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

#compiling the model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])


#training the model on the sampled dataset
history = model.fit(x_train, y_train,validation_data=(x_val, y_val),epochs=80, batch_size=64, verbose=0)


model.summary()


#evaluating the model
val_predictions = model.predict(x_val).flatten()
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
r2 = r2_score(y_val, val_predictions)

print(f"RMSE: {rmse:.4f}")
print(f"RÂ²: {r2:.4f}")


#predicted vs actual values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=val_predictions, alpha=0.6, color='dodgerblue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r', label='Ideal Fit')
plt.title("Predicted vs Actual Values", fontsize=16)
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.legend()
plt.show()


#residuals plot
residuals = y_val - val_predictions
plt.figure(figsize=(8, 6))
sns.scatterplot(x=val_predictions, y=residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--', label='Zero Error')
plt.title("Residuals Plot", fontsize=16)
plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.legend()
plt.show()


#residuals distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='green', bins=30, alpha=0.6)
plt.title("Residuals Distribution", fontsize=16)
plt.xlabel("Residuals", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()


#training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Training vs Validation Loss", fontsize=16)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.show()

