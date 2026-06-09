import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# !pip install kagglehub

# import kagglehub
# import os

# # Download the dataset (not the specific file)
# path = kagglehub.dataset_download("sushant097/bank-marketing-dataset-full")

# # Show all files inside the downloaded dataset
# print("Dataset files:", os.listdir(path))

# # Path to a specific file like bank-full.csv
# csv_path = os.path.join(path, "bank-full.csv")
# print("Path to CSV:", csv_path)


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=';')


original.head()


train.head()


print("The shape of train.csv is", train.shape)
print("The shape of test.csv is", test.shape)
print("The shape of original.csv is", original.shape)


train.drop(columns='id', inplace=True)
print("Train columns datatypes", train.dtypes)


# Map 'yes' -> 1 and 'no' -> 0
original['y'] = original['y'].map({'yes': 1, 'no': 0}).astype('int64')
print("Original columns datatypes", original.dtypes)


# Get all numeric columns
numericalColumns = train.select_dtypes(include='number').columns.tolist()

# Remove the output column 'y' if present
if 'y' in numericalColumns:
    numericalColumns.remove('y')

# Show the result
print("ğŸ“Œ Numerical Columns (excluding target):")
print(numericalColumns)


# Get all categorical (object or category dtype) columns
categoricalColumns = train.select_dtypes(include=['object', 'category']).columns.tolist()

# Show the result
print("ğŸŸª Categorical Columns:")
print(categoricalColumns)


for col in categoricalColumns:
    unique_vals = train[col].unique()
    print(f"\nğŸŸª {col} ({len(unique_vals)} unique values):")
    print(unique_vals)


import matplotlib.pyplot as plt
import seaborn as sns

# Optional: set a clean style
sns.set(style="whitegrid")

# Loop through each categorical column and plot count
for col in categoricalColumns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=train, x=col, palette="Set2", order=train[col].value_counts().index)
    plt.title(f'ğŸ”¢ Value Counts of "{col}"', fontsize=14, color='darkblue')
    plt.xlabel(col.capitalize(), fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Function to check and display null values in a DataFrame
def check_nulls(name, df):
    print(f"\nğŸ”� Null Values in {name} DataFrame:")
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    if null_columns.empty:
        print("âœ… No null values found.")
    else:
        print(null_columns)

# Check for nulls in each dataset
check_nulls("Original", original)
check_nulls("Train", train)
check_nulls("Test", test)


# Define binary columns
binary_columns = ['default', 'housing', 'loan']

# Safe conversion function
for df in [train, test, original]:
    for col in binary_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})  # step 1: map
        df[col] = df[col].fillna(0)                # step 2: handle NaN
        df[col] = df[col].astype('int64')          # step 3: convert to int


train.head()


original.head()


import pandas as pd

# Make a copy of the dataframe to avoid altering the original
encoded_train = train.copy()
encoded_test = test.copy()

# ğŸŸª job (12 unique values)
encoded_train = pd.get_dummies(encoded_train, columns=['job'], prefix='job')
encoded_test = pd.get_dummies(encoded_test, columns=['job'], prefix='job')

# ğŸŸª marital (3 unique values)
encoded_train = pd.get_dummies(encoded_train, columns=['marital'], prefix='marital')
encoded_test = pd.get_dummies(encoded_test, columns=['marital'], prefix='marital')

# ğŸŸª education (4 unique values)
encoded_train = pd.get_dummies(encoded_train, columns=['education'], prefix='edu')
encoded_test = pd.get_dummies(encoded_test, columns=['education'], prefix='edu')

# ğŸŸª contact (3 unique values)
encoded_train = pd.get_dummies(encoded_train, columns=['contact'], prefix='contact')
encoded_test = pd.get_dummies(encoded_test, columns=['contact'], prefix='contact')

# ğŸŸª month (12 unique values)
encoded_train = pd.get_dummies(encoded_train, columns=['month'], prefix='month')
encoded_test = pd.get_dummies(encoded_test, columns=['month'], prefix='month')

# ğŸŸª poutcome (4 unique values)
encoded_train = pd.get_dummies(encoded_train, columns=['poutcome'], prefix='poutcome')
encoded_test = pd.get_dummies(encoded_test, columns=['poutcome'], prefix='poutcome')

# Result
encoded_train.head()


encoded_test.head()


encoded_train.dtypes


bool_cols = encoded_train.select_dtypes(include=['bool']).columns
encoded_train[bool_cols] = encoded_train[bool_cols].astype('int8')
encoded_test[bool_cols] = encoded_test[bool_cols].astype('int8')


encoded_train.dtypes


import matplotlib.pyplot as plt

# Select only the numeric columns you listed
cols_to_plot = [
    'age', 'default', 'balance', 'housing', 'loan',
    'day', 'duration', 'campaign', 'pdays', 'previous', 'y'
]

# Plot histograms
encoded_train[cols_to_plot].hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.suptitle('Distribution of Numeric Columns', fontsize=16)
plt.show()


import numpy as np

cols_to_convert = ['default', 'housing', 'loan']
encoded_train[cols_to_convert] = encoded_train[cols_to_convert].astype(np.int8)
encoded_test[cols_to_convert] = encoded_test[cols_to_convert].astype(np.int8)


from sklearn.preprocessing import RobustScaler

# Columns to scale
robust_cols = ['balance', 'duration', 'campaign', 'pdays', 'previous']

# Initialize scaler
robust_scaler = RobustScaler()

# Fit & transform
encoded_train[robust_cols] = robust_scaler.fit_transform(encoded_train[robust_cols])
encoded_test[robust_cols] = robust_scaler.transform(encoded_test[robust_cols])


from sklearn.preprocessing import StandardScaler

# Columns to scale
standard_cols = ['age', 'day']

# Initialize scaler
standard_scaler = StandardScaler()

# Fit only on training data
encoded_train[standard_cols] = standard_scaler.fit_transform(encoded_train[standard_cols])

# Apply same scaling to test set
encoded_test[standard_cols] = standard_scaler.transform(encoded_test[standard_cols])


encoded_train.dtypes


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Separate features and target
X_train = encoded_train.drop('y', axis=1)
y_train = encoded_train['y']

X_test = encoded_test

input_dim = X_train.shape[1]


model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),  # Input layer
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


with tf.device('/GPU:0'):
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=1
    )


# # Drop 'id' before prediction
# X_test_for_pred = encoded_test.drop(columns=['id'])

# # Predict probabilities
# y_pred_probs = model.predict(X_test_for_pred)

# # Convert to 0/1 labels
# y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

# # Merge predictions with original IDs
# submission = encoded_test[['id']].copy()
# submission['y'] = y_pred

# # Save submission file
# submission.to_csv('submission.csv', index=False)
# print("âœ… submission.csv created")


encoded_train.columns


encoded_test.columns



# âœ… Or save in HDF5 format (good for portability)
model.save('my_model.h5')

print("âœ… Model saved successfully")



from tensorflow.keras.models import load_model

# Or load .h5 format
model = load_model('my_model.h5')


# Drop 'id' before prediction
X_test_for_pred = encoded_test.drop(columns=['id'])

# Predict probabilities
y_pred_probs = model.predict(X_test_for_pred)

# Convert to 0/1 labels
y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

# Merge predictions with original IDs
submission = encoded_test[['id']].copy()
submission['y'] = y_pred

# Save submission file
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv created")




