import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# --- EDA ---
# 1. Descriptive Statistics
print("Train Data Descriptive Statistics:\n", train_data.describe())
print("\nTest Data Descriptive Statistics:\n", test_data.describe())


# 2. Distribution of Target Variable (Price)
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(8, 6))
sns.histplot(train_data['Price'], kde=True)
plt.title('Distribution of Price')
plt.show()


# 3. Missing Value Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Train Data')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(test_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Test Data')
plt.show()


# 4. Explore Categorical Features
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_data, x=col, order=train_data[col].value_counts().index[:10])  # Show top 10
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
    plt.show()


# 5. Explore Numerical Features vs. Target Variable
numerical_cols = ['Compartments', 'Weight Capacity (kg)']


for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=train_data, x=col, y='Price')
    plt.title(f'{col} vs. Price')
    plt.show()


# 6. Correlation Analysis (Numerical Features)
correlation_matrix = train_data[numerical_cols + ['Price']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# --- Data Preprocessing ---

# Separate features and target
X_train = train_data.drop(columns=['Price', 'id'])  # Drop 'id' and 'Price'
y_train = train_data['Price']
X_test = test_data.drop(columns=['id'])  # Drop 'id' from test data


# Preprocessing for numerical data
# Impute missing values with the median
numerical_imputer = SimpleImputer(strategy='median')
X_train[numerical_cols] = numerical_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = numerical_imputer.transform(X_test[numerical_cols])


# Scale numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# Preprocessing for categorical data
# Impute missing values with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = categorical_imputer.transform(X_test[categorical_cols])


# One-hot encode categorical features
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_categorical = onehot_encoder.fit_transform(X_train[categorical_cols])
X_test_categorical = onehot_encoder.transform(X_test[categorical_cols])


# Combine numerical and categorical features
X_train_processed = np.concatenate([X_train[numerical_cols], X_train_categorical], axis=1)
X_test_processed = np.concatenate([X_test[numerical_cols], X_test_categorical], axis=1)


# Train/Validation Split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42
)


# --- ANN Model ---
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train_processed.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer for regression


# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')  # Mean Squared Error for regression


# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# Train the model
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)



# --- Evaluate the Model ---

from sklearn.metrics import mean_squared_error

loss = model.evaluate(X_val_split, y_val_split, verbose=0)
print(f'Validation Loss: {loss}')

# Make predictions on validation set
y_pred_val = model.predict(X_val_split)

# Calculate RMSE (Root Mean Squared Error) on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val_split, y_pred_val))
print(f'Validation RMSE: {rmse_val}')


# --- Evaluate the Model ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Evaluate the Model ---
loss = model.evaluate(X_val_split, y_val_split, verbose=0)
print(f'Validation Loss: {loss}')

# Predictions on validation set
y_pred_val = model.predict(X_val_split).flatten()

# Compute Evaluation Metrics
mse_val = mean_squared_error(y_val_split, y_pred_val)  # Mean Squared Error
rmse_val = np.sqrt(mse_val)                            # Root Mean Squared Error
mae_val = mean_absolute_error(y_val_split, y_pred_val) # Mean Absolute Error
r2_val = r2_score(y_val_split, y_pred_val)             # R-squared Score
mape_val = np.mean(np.abs((y_val_split - y_pred_val) / y_val_split)) * 100  # Mean Absolute Percentage Error

# Print Evaluation Metrics
print(f'Validation RMSE: {rmse_val}')
print(f'Validation MSE: {mse_val}')
print(f'Validation MAE: {mae_val}')
print(f'Validation R2 Score: {r2_val}')
print(f'Validation MAPE: {mape_val:.2f}%')

# --- Predictions on Test Data ---
y_test_pred = model.predict(X_test_processed).flatten()


# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# --- Save the Model, Scaler, and OneHotEncoder ---
model.save('/content/drive/MyDrive/ANN/bag_price_model123.h5')  # Save the entire model
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(onehot_encoder, 'onehot_encoder.joblib')
joblib.dump(numerical_imputer, 'numerical_imputer.joblib')
joblib.dump(categorical_imputer, 'categorical_imputer.joblib')

print("Model, scaler, one-hot encoder, and imputers saved successfully!")


# --- Make Predictions ---
predictions = model.predict(X_test_processed)

# --- Prepare Submission ---
submission = pd.DataFrame({'id': test_data['id'], 'Price': predictions.flatten()})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")


submission.to_csv('submission.csv', index=False)

