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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col = "id")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col = "id")
extra_train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col = "id")


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')

# Apply imputer to categorical columns
train_data_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
extra_train_data_imputed = pd.DataFrame(imputer.transform(extra_train_data), columns=extra_train_data.columns)
test_data_imputed = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)


from sklearn.preprocessing import OrdinalEncoder

priority_order = ['Small', 'Medium', 'Large'] 

# Apply Ordinal Encoding
encoder = OrdinalEncoder(categories=[priority_order])
train_data_imputed['Size'] = encoder.fit_transform(train_data_imputed[['Size']])
test_data_imputed['Size'] = encoder.transform(test_data_imputed[['Size']])
extra_train_data_imputed['Size'] = encoder.transform(extra_train_data_imputed[['Size']])

train_data_imputed.head()


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(extra_train_data_imputed[['Color']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Color']))

# Concatenate with original DataFrame (excluding original 'Color' column)
extra_train_data_final = pd.concat([extra_train_data_imputed.drop(columns=['Color']), encoded_data], axis=1)

extra_train_data_final.head()


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(train_data_imputed[['Color']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Color']))

# Concatenate with original DataFrame (excluding original 'Color' column)
train_data_final = pd.concat([train_data_imputed.drop(columns=['Color']), encoded_data], axis=1)


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(test_data_imputed[['Color']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Color']))

# Concatenate with original DataFrame (excluding original 'Color' column)
test_data_final = pd.concat([test_data_imputed.drop(columns=['Color']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(train_data_final[['Brand']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Brand']))

# Concatenate with original DataFrame (excluding original 'Color' column)
train_data_final = pd.concat([train_data_final.drop(columns=['Brand']), encoded_data], axis=1)



encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(test_data_final[['Brand']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Brand']))

# Concatenate with original DataFrame (excluding original 'Color' column)
test_data_final = pd.concat([test_data_final.drop(columns=['Brand']), encoded_data], axis=1)



encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(extra_train_data_final[['Brand']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Brand']))

# Concatenate with original DataFrame (excluding original 'Color' column)
extra_train_data_final = pd.concat([extra_train_data_final.drop(columns=['Brand']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(train_data_final[['Material']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Material']))

# Concatenate with original DataFrame (excluding original 'Color' column)
train_data_final = pd.concat([train_data_final.drop(columns=['Material']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(test_data_final[['Material']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Material']))

# Concatenate with original DataFrame (excluding original 'Color' column)
test_data_final = pd.concat([test_data_final.drop(columns=['Material']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(extra_train_data_final[['Material']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Material']))

# Concatenate with original DataFrame (excluding original 'Color' column)
extra_train_data_final = pd.concat([extra_train_data_final.drop(columns=['Material']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(train_data_final[['Laptop Compartment']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Laptop Compartment']))

# Concatenate with original DataFrame (excluding original 'Color' column)
train_data_final = pd.concat([train_data_final.drop(columns=['Laptop Compartment']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(test_data_final[['Laptop Compartment']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Laptop Compartment']))

# Concatenate with original DataFrame (excluding original 'Color' column)
test_data_final = pd.concat([test_data_final.drop(columns=['Laptop Compartment']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(extra_train_data_final[['Laptop Compartment']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Laptop Compartment']))

# Concatenate with original DataFrame (excluding original 'Color' column)
extra_train_data_final = pd.concat([extra_train_data_final.drop(columns=['Laptop Compartment']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(train_data_final[['Waterproof']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Waterproof']))

# Concatenate with original DataFrame (excluding original 'Color' column)
train_data_final = pd.concat([train_data_final.drop(columns=['Waterproof']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(test_data_final[['Waterproof']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Waterproof']))

# Concatenate with original DataFrame (excluding original 'Color' column)
test_data_final = pd.concat([test_data_final.drop(columns=['Waterproof']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(extra_train_data_final[['Waterproof']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Waterproof']))

# Concatenate with original DataFrame (excluding original 'Color' column)
extra_train_data_final = pd.concat([extra_train_data_final.drop(columns=['Waterproof']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(train_data_final[['Style']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Style']))

# Concatenate with original DataFrame (excluding original 'Color' column)
train_data_final = pd.concat([train_data_final.drop(columns=['Style']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(test_data_final[['Style']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Style']))

# Concatenate with original DataFrame (excluding original 'Color' column)
test_data_final = pd.concat([test_data_final.drop(columns=['Style']), encoded_data], axis=1)


encoder = OneHotEncoder(drop='first', sparse_output=False)  # Set sparse_output=False to get a dense NumPy array

# Encode only the 'Color' column
encoded_array = encoder.fit_transform(extra_train_data_final[['Style']])

# Convert encoded array to DataFrame
encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Style']))

# Concatenate with original DataFrame (excluding original 'Color' column)
extra_train_data_final = pd.concat([extra_train_data_final.drop(columns=['Style']), encoded_data], axis=1)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential([
    layers.Dense(100, activation='relu', input_shape=(19,)),  # First hidden layer
    layers.Dense(100, activation='relu'),  # Second hidden layer
    layers.Dense(100, activation='relu'),  # Third hidden layer
    layers.Dense(50, activation='relu'),  # Fourth hidden layer
    layers.Dense(1, activation='linear')  # Output layer for regression (no activation)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Summary of the model
model.summary()


from sklearn.model_selection import train_test_split

X = train_data_final.drop(columns=['Price']).copy()
y = train_data_final['Price'].copy()

# Convert problematic object columns to numeric
X['Compartments'] = pd.to_numeric(X['Compartments'], errors='coerce')
X['Weight Capacity (kg)'] = pd.to_numeric(X['Weight Capacity (kg)'], errors='coerce')

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import numpy as np

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


import matplotlib.pyplot as plt

# Train the model and store the history
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

# Plot RMSE over epochs
plt.plot(history.history['loss'], label='Train RMSE')
plt.plot(history.history['val_loss'], label='Validation RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('RMSE Over Epochs')
plt.legend()
plt.show()


test_data_final = np.array(test_data_final, dtype=np.float32)
y_data = model.predict(test_data_final)


custom_index = np.arange(300000, 500000)
y_data = y_data.reshape(200000,)

# Convert y_pred into a DataFrame with the custom index
df_predictions = pd.DataFrame({'Price': y_data}, index=custom_index)

# Reset index but keep the custom index intact
df_predictions.reset_index(drop=False, inplace=True)  # Keeps custom index as a column

# Rename the index column (if needed)
df_predictions.rename(columns={'index': 'id'}, inplace=True)


df_predictions.to_csv("Submission11.csv", index = False)




