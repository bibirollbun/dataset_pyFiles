import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load dfs
df_train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
df_sample = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')

# Describe df_train
print(df_train.info())
print(df_train.describe())


# Since the df is too big we'll use a sample
df_sampletrain = df_train.sample(1000)

# Scatterplot to see distribution
plt.figure(figsize=(3,2))
sns.scatterplot(x='id', y='Premium Amount', data=df_sampletrain)
plt.title('Premium amount for IDs')
plt.show()

# Trying other plots
plt.figure(figsize=(3,2))
sns.regplot(x='Annual Income', y='Premium Amount', data=df_sampletrain)
plt.title('Annual income VS Premium amount')
plt.show()

plt.figure(figsize=(3,2))
sns.regplot(x='Age', y='Premium Amount', data=df_sampletrain)
plt.title('Age VS Premium amount')
plt.show()

# Plot the distribution of Premium Amount
plt.figure(figsize=(3,2))
sns.histplot(df_train['Premium Amount'], kde=True)
plt.show()


# Pre processing data
# Drop missing values
df_train = df_train.dropna()
df_test = df_test.dropna()

# Select relevant numerical columns
numerical_columns = [
    'Age', 'Annual Income', 'Number of Dependents', 'Health Score',
    'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration'
]

# Prepare X_train and X_test
X_train = df_train[numerical_columns]
X_test = df_test[numerical_columns]

# Prepare y_train
y_train = df_train['Premium Amount']


# Starting with Deep Learning model
# Define model Sequential CNN
model = Sequential()

# Create layers
model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
# Output layer for regression
model.add(Dense(1, activation='linear')) 

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Print output of the model
print(model.summary())


# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train_split, y_train_split, validation_data=(X_val_split, y_val_split), batch_size=10000, epochs=50)


# Evaluate the model on the validation set
val_loss, val_mae = model.evaluate(X_val_split, y_val_split)
print(f'Validation Loss: {val_loss}, Validation MAE: {val_mae}')

# Make predictions on the test set
y_pred = model.predict(X_test)


