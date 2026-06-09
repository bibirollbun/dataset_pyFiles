!pip install pytorch-tabnet


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import os


df_train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


# Set a random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --- 2. Data Preprocessing ---
# Define the target and features
target = 'BeatsPerMinute'
features = [col for col in df_train.columns if col not in ['id', target]]

# Handle missing values by filling with the mean.
# This is a simple strategy. For a real-world scenario, you might use more advanced techniques.
for col in features:
    if df_train[col].isnull().any():
        df_train[col] = df_train[col].fillna(df_train[col].mean())
    if df_test[col].isnull().any():
        df_test[col] = df_test[col].fillna(df_test[col].mean())

# --- 3. Feature Engineering ---
# Create new features from existing ones
df_train['Log_TrackDurationMs'] = np.log1p(df_train['TrackDurationMs'])
df_test['Log_TrackDurationMs'] = np.log1p(df_test['TrackDurationMs'])

# Create a combined score feature
df_train['Rhythm_Acoustic_Score'] = df_train['RhythmScore'] * df_train['AcousticQuality']
df_test['Rhythm_Acoustic_Score'] = df_test['RhythmScore'] * df_test['AcousticQuality']

# Create a ratio feature, handling potential division by zero
df_train['Vocal_to_Instrumental_Ratio'] = df_train['VocalContent'] / (df_train['InstrumentalScore'] + 1e-6)
df_test['Vocal_to_Instrumental_Ratio'] = df_test['VocalContent'] / (df_test['InstrumentalScore'] + 1e-6)

# Update the feature list to include the new features
features.extend(['Log_TrackDurationMs', 'Rhythm_Acoustic_Score', 'Vocal_to_Instrumental_Ratio'])

# Identify categorical and continuous features.
# all features except 'id' and 'BeatsPerMinute' are continuous.
categorical_features = []
continuous_features = [col for col in features if col not in categorical_features]

# Prepare data for the model
X = df_train[features].values
y = df_train[target].values.reshape(-1, 1) # Reshape for TabNet
X_test_pred = df_test[features].values


# --- 3. Training and Validation Split ---
# Split the training data into a training and a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)


# --- 4. Model Instantiation and Training ---
# Define the TabNet Regressor model
regressor = TabNetRegressor(
    n_d=64, # Width of the decision prediction layer (nodes in each layer)
    n_a=64, # Width of the attention embedding layer
    n_steps=5, # Number of steps in the model
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size":50, "gamma":0.9},
    mask_type='sparsemax', # Gating function
    verbose=1,
    seed=SEED
)

print("Starting model training...")

# Fit the model
regressor.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_name=['train', 'val'],
    eval_metric=['rmse'],
    max_epochs=200,
    patience=5,
    batch_size=1024,
    virtual_batch_size=128
)

print("Model training complete.")


# --- 5. Prediction on the Test Dataset ---
print("Making predictions on the test dataset...")
y_pred = regressor.predict(X_test_pred)

# --- 6. Create the Submission File ---
print("Generating submission.csv file...")
submission_df = pd.DataFrame({
    'id': df_test['id'],
    'BeatsPerMinute': y_pred.flatten() # Flatten the predictions to a 1D array
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")

