import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

print("Baseline Model Development Started (with Correct Submission Format)!")

# --- CONFIGURATION ---
BASE_PATH = '/kaggle/input/ariel-data-challenge-2025'

# --- 1. Ground Truth Data Load Karna ---
print("Step 1: Loading Ground Truth (train.csv) to calculate mean spectrum...")
train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
train_spectra = train_df.drop(columns=['planet_id']).to_numpy()

# --- 2. "Dumb" Prediction Banana (Mean aur Standard Deviation) ---
print("Step 2: Calculating Mean Spectrum and Standard Deviation for our prediction...")

# Mean spectrum (hamara prediction)
mean_spectrum_prediction = np.mean(train_spectra, axis=0)

# Standard deviation (hamari uncertainty)
uncertainty_prediction = np.std(train_spectra, axis=0)

# Agar uncertainty mein koi value 0 hai, to use ek choti value de dein taaki error na aaye
uncertainty_prediction[uncertainty_prediction == 0] = 1e-9 

print(f"Mean prediction shape: {mean_spectrum_prediction.shape}")
print(f"Uncertainty prediction shape: {uncertainty_prediction.shape}")

# Dono predictions ko jodna
full_prediction_values = np.concatenate([mean_spectrum_prediction, uncertainty_prediction])


# --- 3. Submission File Banana (Sahi Tarike se) ---
print("\nStep 3: Creating the submission file using the correct wide format...")

# Sample submission ko load karna taaki humein sahi structure mil jaye
sample_submission_path = os.path.join(BASE_PATH, 'sample_submission.csv')
submission_df = pd.read_csv(sample_submission_path)

# Prediction columns ke naam nikalna (planet_id ko chhodkar)
prediction_columns = submission_df.columns[1:]

for col, val in tqdm(zip(prediction_columns, full_prediction_values), total=len(prediction_columns), desc="Filling Submission"):
    submission_df[col] = val

# Final CSV file save karna
submission_df.to_csv('submission.csv', index=False)

print("\nâœ… Success! `submission.csv` file has been created with the correct format.")
print("You can now submit this file. My apologies for the previous error!")

