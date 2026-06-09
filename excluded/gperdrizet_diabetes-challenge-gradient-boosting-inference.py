# Standard library imports
import subprocess
import sys
import urllib.request
from pathlib import Path

# Third party imports
import joblib
import pandas as pd


# Flag to control environment-specific paths & configurations
KAGGLE = True


# Add path to custom transformers module
if KAGGLE:

    # On Kaggle, the transformers file should be uploaded as part of the dataset
    transformers_path = Path('/kaggle/input/diabetes-challenge-gradient-boosting-assets')

else:
    # For local/GitHub, use the models directory
    transformers_path = Path('../models').resolve()

sys.path.insert(0, str(transformers_path))

# Import custom transformers (needed for model deserialization)
from gradient_boosting_transformers import (
    IDColumnDropper, IQRClipper, DifferenceFeatures, SumFeatures,
    RatioFeatures, ReciprocalFeatures, LogFeatures, SquareRootFeatures,
    KMeansClusterFeatures
)


# Set file paths based on environment
if KAGGLE:

    # Kaggle paths - data is in /kaggle/input/
    test_df_path = '/kaggle/input/playground-series-s5e12/test.csv'
    model_path = '/kaggle/input/diabetes-challenge-gradient-boosting-assets/gradient_boosting.joblib'

else:

    # Otherwise, load from GitHub
    test_df_path = 'https://gperdrizet.github.io/FSA_devops/assets/data/unit3/diabetes_prediction_test.csv'
    model_url = 'https://github.com/gperdrizet/diabetes-prediction/raw/refs/heads/main/models/gradient_boosting.joblib'
    
    # Download model to temporary location
    model_path = Path('gradient_boosting.joblib')
    urllib.request.urlretrieve(model_url, model_path)

# Load the testing dataset
test_df = pd.read_csv(test_df_path)

# Load the model
model = joblib.load(model_path)

# Display first few rows of training data
test_df.head()


predictions_df = pd.DataFrame({
    'id': test_df['id'].astype(int),
    'diagnosed_diabetes': model.predict_proba(test_df)[:, 1]
})

predictions_df.info()


# Set submission file path based on environment
if KAGGLE:

    # Save submission file in current working directory
    submission_path = Path('submission.csv')

else:

    # Create data directory if it doesn't exist
    data_dir = Path('../data')
    data_dir.mkdir(parents=True, exist_ok=True)
    submission_path = data_dir / 'gradient_boosting_submission.csv'

# Save submission file
predictions_df.to_csv(submission_path, index=False)

print(f'Submission saved to: {submission_path}\n')
predictions_df.head()


# Clean up downloaded model file if not on Kaggle
if not KAGGLE and model_path.exists():
    model_path.unlink()
    print(f'Cleaned up temporary model file: {model_path}')

