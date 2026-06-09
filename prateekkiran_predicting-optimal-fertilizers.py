# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb # Using the much faster LightGBM model
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


INPUT_DIR = "/kaggle/input/playground-series-s5e6"
OUTPUT_DIR = "/kaggle/working/" # Kaggle's directory for saving files


print("--- Loading Training Data ---")
try:
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    print("train.csv loaded successfully.")
except FileNotFoundError:
    print(f"Error: 'train.csv' not found in {INPUT_DIR}.")
    exit()


print("--- Preprocessing Data ---")
y = train_df['Fertilizer Name']
X = train_df.drop('Fertilizer Name', axis=1)

categorical_features = ['Soil Type', 'Crop Type']
for col in categorical_features:
    X[col] = X[col].astype('category')

le = LabelEncoder()
y_encoded = le.fit_transform(y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)



print("--- Training LightGBM with built-in categorical support ---")
best_model = lgb.LGBMClassifier(random_state=42, n_estimators=150)

# Pass the data directly. LightGBM will handle the 'category' dtype automatically.
best_model.fit(X, y_encoded)

print("Model training is complete.")




print("--- Saving Model and Supporting Files ---")
joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'fertilizer_lgbm_model.joblib'))
# We save the feature names to ensure consistency in the prediction step
with open(os.path.join(OUTPUT_DIR, 'model_features.pkl'), 'wb') as f:
    pickle.dump(list(X.columns), f)
with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print("\nTraining complete. Model and supporting files have been saved.")


print("--- Loading Test Data and Saved Model ---")
try:
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
    print("test.csv loaded successfully.")

    # --- DIAGNOSTIC STEP: Print all column names ---
    print("\nColumns found in test.csv:")
    print(list(test_df.columns))
    print("--------------------------------------\n")
    # ---------------------------------------------

    # --- ACTION REQUIRED: Find your ID column in the list above ---
    # Replace 'id' below with the correct name of your ID column from the printed list.
    ID_COLUMN_NAME = 'id'
    # -------------------------------------------------------------

    test_ids = test_df[ID_COLUMN_NAME]
    test_df_features = test_df.drop(ID_COLUMN_NAME, axis=1)

    model = joblib.load(os.path.join(OUTPUT_DIR, 'fertilizer_lgbm_model.joblib'))
    with open(os.path.join(OUTPUT_DIR, 'model_features.pkl'), 'rb') as f:
        model_features = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

except FileNotFoundError as e:
    print(f"Error: Could not find a required file. {e}")
    exit()
except KeyError:
    print(f"KeyError: The column '{ID_COLUMN_NAME}' was not found in test.csv.")
    print("Please update the ID_COLUMN_NAME variable with the correct name from the list of columns printed above.")
    exit()


print("--- Preprocessing Test Data without One-Hot Encoding ---")
# KEY CHANGE: Apply the same 'category' conversion
categorical_features = ['Soil Type', 'Crop Type']
for col in categorical_features:
    if col in test_df.columns:
        test_df[col] = test_df[col].astype('category')

# Ensure test columns match the order of the trained model
test_final = test_df[model_features]


print("--- Making Predictions ---")
predictions_encoded = model.predict(test_final)
final_predictions = le.inverse_transform(predictions_encoded)


print("--- Creating Submission File ---")
# --- NEW: Create the DataFrame with the 'id' and the prediction ---
submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': final_predictions
})
# --------------------------------------------------------------

submission_df.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)

print("\nPrediction complete!")
print("The predictions have been saved to 'submission.csv'.")
print("\nFirst 5 rows of the submission file:")
print(submission_df.head())




