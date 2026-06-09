import polars as pl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# Load the train and test CSV files into Polars DataFrames (like tables)
train = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
test = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')

# Convert Polars DataFrames to pandas DataFrames for compatibility with sklearn (the model library)
train_pd = train.to_pandas()
test_pd = test.to_pandas()

# Print the first few rows of train and test data to see what they look like
print(train_pd.head())    # Shows 5 rows by default
print(test_pd.head())


# Pick column names that start with 'acc_' or 'rot_' (all IMU sensor columns)
imu_cols = [col for col in train_pd.columns if col.startswith('acc_') or col.startswith('rot_')]

# Print out the list of IMU columns so we know which features we’re using
print("Selected IMU columns:", imu_cols)


# For each gesture (identified by 'sequence_id'), calculate the average value of each IMU feature
train_agg = train_pd.groupby('sequence_id')[imu_cols].mean().reset_index()

# Print the shape of the new DataFrame (how many rows and columns), and see the first few rows
print("Aggregated train shape:", train_agg.shape)
print(train_agg.head())


# For each sequence, get the 'sequence_type' (our label: target or non-target)
labels = train_pd.groupby('sequence_id')['sequence_type'].first().reset_index()
print(labels.head())   # Show the first few labels

# Add (merge) the labels to our feature DataFrame so each row has both features and its label
train_agg = train_agg.merge(labels, on='sequence_id')
print(train_agg.head())


X = train_agg[imu_cols]   # Features: the IMU columns (acceleration & rotation)
y = train_agg['sequence_type'].map({'Target': 1, 'Non-Target': 0})  # Labels as numbers

# Print the shape of X and how many targets/non-targets we have
print("X shape:", X.shape)
print("y distribution:", y.shape)


print(X.isnull().sum())      # See how many NaNs per feature
print(X.isnull().sum().sum()) # Total NaNs


# After creating X and y
X_clean = X.dropna()
y_clean = y[X_clean.index]   # Make sure labels match the cleaned features

print(X_clean.shape)         # Check how many rows remain
print(y_clean.shape)


print(X_clean.isnull().sum())      # See how many NaNs per feature
print(X_clean.isnull().sum().sum()) # Total NaNs


from sklearn.model_selection import train_test_split    # Tool for splitting data

# Split: 80% for training, 20% for validation; set random_state for repeatability
X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Show the size of each split
print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)



from sklearn.linear_model import LogisticRegression     # Our simple model

# Create the logistic regression model; max_iter helps it finish training if data is tricky
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)    # Teach the model to predict y_train from X_train



from sklearn.metrics import f1_score    # F1 score is a good metric for imbalanced data

y_pred = model.predict(X_val)           # Predict labels for the validation features
print("Validation F1 Score:", f1_score(y_val, y_pred))   # Print F1 score

