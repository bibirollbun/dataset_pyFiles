import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df3 = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
df4 = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


print(df.head(),"\n\n\n")
print(test_df.head(),"\n\n\n")
print(df3.head(),"\n\n\n")
print(df4.head())


df.shape


df4.shape


full_df = pd.concat([df, df4], ignore_index=True)


print("ğŸ”� Total NaNs in full_df:", full_df.isnull().sum().sum())
print("ğŸ”� Total NaNs in test_df:", test_df.isnull().sum().sum())


print("ğŸ“Š Number of NaNs per column:\n")
print(full_df.isnull().sum())


# Fill numeric
full_df['Weight Capacity (kg)'].fillna(full_df['Weight Capacity (kg)'].mean(), inplace=True)

# Fill binary
full_df['Laptop Compartment'].fillna('No', inplace=True)
full_df['Waterproof'].fillna('No', inplace=True)

# Fill categorical with 'Unknown'
for col in ['Brand', 'Material', 'Size', 'Style', 'Color']:
    full_df[col].fillna('Unknown', inplace=True)



# Fill numeric
test_df['Weight Capacity (kg)'].fillna(full_df['Weight Capacity (kg)'].mean(), inplace=True)

# Fill binary
test_df['Laptop Compartment'].fillna('No', inplace=True)
test_df['Waterproof'].fillna('No', inplace=True)

# Fill categorical with 'Unknown'
for col in ['Brand', 'Material', 'Size', 'Style', 'Color']:
    test_df[col].fillna('Unknown', inplace=True)



import pandas as pd

nan_info = pd.DataFrame({
    'Null Count': full_df.isnull().sum(),
    'Dtype': full_df.dtypes
})

# Optional: Only show columns that have nulls
nan_info = nan_info[nan_info['Null Count'] > 0].sort_values(by='Null Count', ascending=False)

print(nan_info)


import pandas as pd

nan_info = pd.DataFrame({
    'Null Count': test_df.isnull().sum(),
    'Dtype': test_df.dtypes
})

# Optional: Only show columns that have nulls
nan_info = nan_info[nan_info['Null Count'] > 0].sort_values(by='Null Count', ascending=False)

print(nan_info)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd

# Separate input and target
X = full_df.drop(['Price', 'id'], axis=1)
y = full_df['Price']

# Apply same column drops to test_df
X_test = test_df.drop(['id'], axis=1)


# Combine for consistent encoding
combined = pd.concat([X, X_test], axis=0)

# One-hot encode
combined_encoded = pd.get_dummies(combined)

# Split back
X_encoded = combined_encoded.iloc[:len(X)]
X_test_encoded = combined_encoded.iloc[len(X):]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_test_scaled = scaler.transform(X_test_encoded)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


import tensorflow.keras.backend as K

# Define custom RMSE metric
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Build the model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dense(1)
])

# Compile the model with RMSE
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', rmse])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))


# Step 1: Preprocess test_df just like training data
# (Assumes you already filled NaNs, encoded, and scaled test_df â†’ X_test_scaled)

# Step 2: Predict on test set
y_test_pred = model.predict(X_test_scaled).flatten()  # flatten to 1D

# Step 3: Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],         # use original 'id' from test set
    'Price': y_test_pred         # predicted values
})

# Step 4: Save to CSV
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file 'submission.csv' created successfully!")




