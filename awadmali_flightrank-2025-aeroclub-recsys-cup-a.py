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


# Setup and Imports
!pip install lightgbm pyarrow fastparquet -q

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
print("âœ… Ready")


# Load Data

try:
    train_df = pd.read_parquet("/kaggle/input/aeroclub-recsys-2025/train.parquet")
    test_df = pd.read_parquet("/kaggle/input/aeroclub-recsys-2025/test.parquet")
    sample_submission_df = pd.read_parquet("/kaggle/input/aeroclub-recsys-2025/sample_submission.parquet")
    print("âœ… Data modified successfully")
except FileNotFoundError:
    print("Error")
    # Create dummy dataframes to prevent further errors
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()



# Preprocessing for Dates and Categories

if not train_df.empty:
    # --- NEW: Handle Date/Time Columns ---
    datetime_features = train_df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if datetime_features:
        print(f"Found datetime features: {datetime_features}")
        for col in datetime_features:
            # Extract useful numerical features from the date
            train_df[f'{col}_year'] = train_df[col].dt.year
            train_df[f'{col}_month'] = train_df[col].dt.month
            train_df[f'{col}_day'] = train_df[col].dt.day
            train_df[f'{col}_dayofweek'] = train_df[col].dt.dayofweek # Monday=0, Sunday=6
            
            test_df[f'{col}_year'] = test_df[col].dt.year
            test_df[f'{col}_month'] = test_df[col].dt.month
            test_df[f'{col}_day'] = test_df[col].dt.day
            test_df[f'{col}_dayofweek'] = test_df[col].dt.dayofweek
            
            # Drop the original datetime column
            train_df = train_df.drop(col, axis=1)
            test_df = test_df.drop(col, axis=1)
        print("âœ… Converted datetime features to numerical format.")
    
    # --- EXISTING: Handle Categorical Columns ---
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

    if categorical_features:
        print(f"Found categorical features: {categorical_features}")
        for col in categorical_features:
            le = LabelEncoder()
            # Combine train and test data to ensure all categories are learned
            combined_data = pd.concat([train_df[col], test_df[col]]).astype(str)
            le.fit(combined_data)
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
        print("âœ… Converted categorical features to numerical format.")
        
    print("\nPreprocessing complete.")


# Model Training

if not train_df.empty:
    # Define features (X) and target (y)
    features = [col for col in train_df.columns if col not in ['ranker_id', 'flight_id', 'selected']]
    X_train = train_df[features]
    y_train = train_df['selected']

    # Initialize and train the LightGBM Classifier
    lgbm_ranker = lgb.LGBMClassifier(
        objective='binary',
        metric='logloss',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    print("ðŸš€ ...Start model training")
    lgbm_ranker.fit(X_train, y_train)
    print("âœ… Model training completed")



# Prediction and Ranking

if not test_df.empty:
    X_test = test_df[features]

    # Predict the probability of being 'selected' (class 1)
    probabilities = lgbm_ranker.predict_proba(X_test)[:, 1]

    # Add the prediction scores to the test dataframe
    test_df['score'] = probabilities

    # Calculate the rank within each group based on the score
    test_df['rank'] = test_df.groupby('ranker_id')['score'].rank(method='first', ascending=False).astype(int)
    
    print("âœ… The results were successfully predicted and arranged")



# Check if the test_df is empty
print(f"Is test_df empty? {test_df.empty}")
print(f"Number of rows in test_df: {len(test_df)}")


import pandas as pd
import numpy as np

# --- Step 1: Load Data using the correct function and path ---
# This is the corrected line for your Kaggle Notebook.
# We use read_parquet because the file ends with .parquet.
try:
    path_to_file = "/kaggle/input/aeroclub-recsys-2025/test.parquet"
    test_df = pd.read_parquet(path_to_file)
    print("Test data loaded successfully from Parquet file!")
    print("Test data shape:", test_df.shape)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please double-check the file path and that the file exists.")


# --- Step 2: Generate your predictions ---
# (Your model's code goes here)
# For demonstration, we will create dummy predictions.
# Ensure the number of predictions matches the number of rows in test_df
if 'test_df' in locals():
    # Make sure to use your actual model's prediction logic here
    model_predictions = np.random.randint(0, 2, size=len(test_df))
    predictions = model_predictions


# --- Step 3: Create the submission file ---
# This part will now work with your actual test data
if 'predictions' in locals():
    print("Creating submission file...")

    submission_df = pd.DataFrame({
        "Id": test_df["Id"], # Make sure your parquet file has a column named 'Id'
        "Predicted": predictions
    })

    # The submission file is usually expected in CSV format
    submission_df.to_csv("submission.csv", index=False)
    
    print("Submission file 'submission.csv' created successfully!")
    print(submission_df.head())

