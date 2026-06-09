import polars as pl
import pandas as pd

# Load the train and test CSV files into Polars DataFrames (like spreadsheets)
train = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
test = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')

# Convert Polars DataFrames to pandas DataFrames for compatibility with sklearn (the model library)
train_pd = train.to_pandas()
test_pd = test.to_pandas()

# Print the first few rows to check what our raw data looks like
print("First few rows of TRAIN data:")
print(train_pd.head())
print("First few rows of TEST data:")
print(test_pd.head())


# Pick all columns that start with "acc_" or "rot_"
imu_cols = [col for col in train_pd.columns if col.startswith('acc_') or col.startswith('rot_')]
print("Selected IMU columns:", imu_cols)


# Group the data by sequence_id (each gesture) and calculate the mean (average) for each IMU feature
train_agg = train_pd.groupby('sequence_id')[imu_cols].mean().reset_index()

# For each sequence, get its gesture label (first label for that sequence)
gesture_labels = train_pd.groupby('sequence_id')['gesture'].first().reset_index()

# Merge features and gesture labels so each row represents a gesture, with its average sensor data and its name
train_agg = train_agg.merge(gesture_labels, on='sequence_id')

print("First few rows of aggregated train data:")
print(train_agg.head())



X = train_agg[imu_cols]           # Features: sensor values
y = train_agg['gesture']          # Labels: gesture names (what we want to predict)

# Print some info about the labels
print("Unique gesture labels:", y.unique())
print("How many examples per gesture label:\n", y.value_counts())

# Drop rows with missing values (NaN) to keep only clean data
X_clean = X.dropna()
y_clean = y[X_clean.index]

print("Shape of features and labels after cleaning:", X_clean.shape, y_clean.shape)


from sklearn.model_selection import train_test_split

# 80% for training, 20% for validation (random_state=42 gives repeatable results)
X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

print("Train shape (for learning):", X_train.shape)
print("Validation shape (for checking):", X_val.shape)


from sklearn.linear_model import LogisticRegression

# max_iter=1000 gives the model more chances to learn; multinomial is for multiclass problems
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)    # Model "learns" to predict gesture labels from IMU features


from sklearn.metrics import f1_score

# Predict gesture names for validation data
y_pred = model.predict(X_val)

print("Predicted gesture labels on validation set:\n", y_pred)
print("\n\n True gesture labels for validation set:\n", y_val)

# Macro F1 score: average F1 score across all gesture classes
print("\n\nValidation Macro F1 Score:", f1_score(y_val, y_pred, average='macro'))


from sklearn.metrics import f1_score

# 1. List all unique target gestures using the original training data
target_gestures = train_pd.loc[train_pd['sequence_type'] == "Target", 'gesture'].unique()

print("All target gestures in the data:")
print(target_gestures)


# 2. Calculate Binary F1
def gestures_to_binary(y):
    # Map all target gestures to 1, others to 0
    return [1 if g in target_gestures else 0 for g in y]

binary_true = gestures_to_binary(y_val)
binary_pred = gestures_to_binary(y_pred)

binary_f1 = f1_score(binary_true, binary_pred)
print("Binary F1 score:", binary_f1)


# 3. Calculate Macro F1 (collapse all non-targets into 'non_target')
def collapse_gestures(y):
    return [g if g in target_gestures else 'non_target' for g in y]

macro_true = collapse_gestures(y_val)
macro_pred = collapse_gestures(y_pred)

macro_f1 = f1_score(macro_true, macro_pred, average='macro')
print("Macro F1 score:", macro_f1)


# 4. Final score: average the two
final_score = (binary_f1 + macro_f1) / 2
print("Final (competition) score:", final_score)


import joblib

# Suppose you used a scaler and a trained model as in previous steps
joblib.dump(model, 'logreg_model.joblib')


import joblib

model = joblib.load('/kaggle/working/logreg_model.joblib')


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # 1. Make sure IMU feature columns are the same as in training
    imu_cols = [col for col in sequence.columns if col.startswith('acc_') or col.startswith('rot_')]

    # 2. Compute mean values and convert to a pandas DataFrame, with the same column order
    feature_means = [sequence[col].mean() for col in imu_cols]
    feature_vector = pd.DataFrame([feature_means], columns=imu_cols)  # Shape (1, n_features)

    # 3. If using scaler, scale the features (remove this line if not using scaler)
    # feature_vector = scaler.transform(feature_vector)

    # 4. Make sure there are no NaNs
    feature_vector = feature_vector.fillna(0)

    # 5. Predict gesture
    gesture_pred = model.predict(feature_vector)[0]
    return gesture_pred


import kaggle_evaluation.cmi_inference_server

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

import os
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )


