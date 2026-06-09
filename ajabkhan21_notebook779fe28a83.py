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


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier 
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ===============================================================
# --- 1. DUMMY DATA SETUP (REPLACE WITH YOUR REAL DATA) ---
# This section defines X_train, y_train, and X_test to fix the NameError.
# To compete, you must replace this with code that loads the AIRR-ML-25 files 
# and performs your best feature engineering (k-mer counts, gene frequencies, etc.)

num_samples_train = 1000  # Number of training samples
num_samples_test = 500    # Number of test samples
num_features = 50         # Number of features (columns)

# Create placeholder feature matrices (X_train, X_test)
X_train = pd.DataFrame(np.random.rand(num_samples_train, num_features))
y_train = pd.Series(np.random.randint(0, 2, size=num_samples_train)) # Target: 0 (Healthy) or 1 (Disease)
X_test = pd.DataFrame(np.random.rand(num_samples_test, num_features))

# Create a placeholder submission DataFrame with a correct ID column
# (You will need to load the actual test IDs from the competition file)
submission_df = pd.DataFrame({'ID': [f'sample_{i}' for i in range(num_samples_test)]}) 

print("Dummy data for X_train, y_train, and X_test has been successfully created.")
# --- END DUMMY DATA SETUP ---
# ===============================================================


# --- 2. Define Base Models (The First Layer) ---
# These powerful models learn from your features independently.
# NOTE: The optimal hyperparameters should be found via tuning (e.g., GridSearchCV).
lgbm = LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, n_jobs=-1, verbose=-1)
xgb = XGBClassifier(random_state=42, n_estimators=500, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
cat = CatBoostClassifier(random_state=42, n_estimators=500, learning_rate=0.05, verbose=0, allow_writing_files=False)

estimators = [
    ('lgbm', lgbm),
    ('xgb', xgb),
    ('cat', cat)
]

# --- 3. Define the Stacking Classifier (The Second Layer) ---
# The final_estimator combines the predictions of the three base models.
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(solver='lbfgs', C=0.1, random_state=42),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
    n_jobs=-1,
    verbose=0
)

# --- 4. Train the Stacking Model ---
print("\nStarting Stacking Model Training...")
# This line now works because X_train and y_train are defined above.
stacking_model.fit(X_train, y_train) 
print("Training Complete.")

# --- 5. Generate Predictions ---
# We use predict_proba to get the probability of the positive class (Class 1 / Disease)
# This is required for the AUC metric used in the competition.
predictions_proba = stacking_model.predict_proba(X_test)[:, 1]
print("Predictions generated.")

# --- 6. Create Submission File ---
# Add the predicted probabilities to your submission DataFrame
submission_df['label_positive_probability'] = predictions_proba

# Display the first few rows of the final submission (using dummy data)
print("\nFirst 5 rows of the submission file:")
print(submission_df.head())

# To save the file for submission, uncomment the line below:
# submission_df.to_csv('airr_ml_stacking_submission.csv', index=False)


import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm # Import tqdm for progress bar (helps with large files)

def load_and_engineer_airr_data(data_path='../input/adaptive-immune-profiling-challenge-2025'):
    
    # 1. Load Metadata (Contains the 'filename' and the 'y' labels)
    metadata_df = pd.read_csv(os.path.join(data_path, 'metadata.csv'))
    
    # Prepare the DataFrame to store the new features
    feature_list = []
    
    # Define the folders where the TSV files are located
    # NOTE: You MUST replace these paths with the actual folder names in your Kaggle/local directory
    tsv_folders = ['train_repertoire', 'test_repertoire'] 
    
    # Loop through all files mentioned in the metadata
    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting Features"):
        filename = row['filename']
        
        # Determine if the file is in the train or test folder
        file_path = None
        for folder in tsv_folders:
            check_path = os.path.join(data_path, folder, filename)
            if os.path.exists(check_path):
                file_path = check_path
                break
        
        if file_path is None:
            # Handle files not found (e.g., test files you don't have the labels for yet)
            continue
            
        # --- CORE FEATURE EXTRACTION ---
        try:
            # Read the TSV file using tab separator
            df_repertoire = pd.read_csv(file_path, sep='\t')
            
            # 1. V-Gene Frequency Feature
            # Normalize=True calculates the relative frequency (percentage)
            v_gene_counts = df_repertoire['v_call'].value_counts(normalize=True)
            
            # Find the frequency of the most common V-gene
            max_v_freq = v_gene_counts.max() if not v_gene_counts.empty else 0.0
            
            # 2. Total Clones Feature (Counts are always useful)
            total_clones = len(df_repertoire)
            
            # Create a dictionary for this patient's features
            patient_features = {
                'filename': filename,
                'max_v_gene_freq': max_v_freq,
                'total_clones': total_clones,
                # ... Add more features here (e.g., max_j_gene_freq, unique_clone_count)
            }
            
            feature_list.append(patient_features)
            
        except Exception as e:
            # Handle bad files
            print(f"Error processing {filename}: {e}")
            pass
            
    # 3. Combine Features and Labels
    feature_df = pd.DataFrame(feature_list)
    final_df = metadata_df.merge(feature_df, on='filename', how='left')

    # Prepare final X and y
    X = final_df[['max_v_gene_freq', 'total_clones']] # Your real features
    y = final_df['disease_label'].map({'healthy': 0, 'disease': 1})
    
    # --- IMPORTANT: Handle Missing Data ---
    # Since you are only extracting a few simple features, some files might not have data.
    # Impute NaNs with 0 or the mean.
    X = X.fillna(0) 

    return X, y

