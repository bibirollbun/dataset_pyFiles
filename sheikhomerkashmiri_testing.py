import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import joblib

# Load test data
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

# Define features (Ensure they match training features)
EXCLUDE_COLS = ["ID", "efs", "efs_time", "y"]
FEATURES = [col for col in test.columns if col not in EXCLUDE_COLS]

# Load trained model
MODEL_PATH = "/kaggle/input/notebook308fc88103/random_forest_model.pkl"  # Updated path
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except FileNotFoundError:
    print(f"❌ Model file not found: {MODEL_PATH}")
    exit()

# Apply necessary feature transformations
def preprocess(df):
    """Feature Engineering: Convert categorical variables and handle missing values."""
    
    # Apply categorical mappings
    mappings = {
        "arrhythmia": {"No": 0, "Not done": 0, "Yes": 1},
        "cardiac": {"No": 0, "Not done": 0, "Yes": 1},
        "diabetes": {"No": 0, "Not done": 0, "Yes": 1},
        "hepatic_mild": {"No": 0, "Not done": 0, "Yes": 1},
        "hepatic_severe": {"No": 0, "Not done": 0, "Yes": 3},
    }
    
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Convert all non-numeric values to NaN and fill with a default
    df.replace({"N/A - non-malignant indication": np.nan}, inplace=True)
    df.fillna(0, inplace=True)  # Replace NaN with 0 or another default
    
    # Ensure all columns are numeric
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]  # Convert categorical text to numeric
    
    # Handle missing features
    missing_features = [col for col in FEATURES if col not in df.columns]
    for col in missing_features:
        df[col] = 0  # Assign default value
    
    return df

# Preprocess test data
test = preprocess(test)

# Ensure features match model training
test_features = test[FEATURES]

# Predict survival probabilities
test["prediction"] = model.predict(test_features)

# Save submission file
submission = test[["ID", "prediction"]]
submission.to_csv("submission.csv", index=False)

print("✅ Submission file saved as submission.csv")





