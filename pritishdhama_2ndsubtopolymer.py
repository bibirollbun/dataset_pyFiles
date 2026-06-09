import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor



train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
sample_submission = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")



def add_smiles_features(df):
    df["smiles_length"] = df["SMILES"].str.len()
    df["num_c"] = df["SMILES"].str.count("C")
    df["num_o"] = df["SMILES"].str.count("O")
    df["num_n"] = df["SMILES"].str.count("N")
    df["num_double"] = df["SMILES"].str.count("=")
    df["num_ring"] = df["SMILES"].str.count("1") + df["SMILES"].str.count("2")
    return df

train = add_smiles_features(train)
test = add_smiles_features(test)

print("Features added successfully!")



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re

# -------------------------------
# Load Data
# -------------------------------
train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

# Targets to predict
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

# -------------------------------
# Feature Engineering
# -------------------------------
def smiles_features(smiles):
    """Extract simple features from SMILES string."""
    return {
        "smiles_length": len(smiles),
        "num_c": smiles.count("C"),
        "num_o": smiles.count("O"),
        "num_n": smiles.count("N"),
        "num_double": smiles.count("="),
        "num_ring": len(re.findall(r"\d", smiles)),
    }

# Apply feature extraction
train_feats = pd.DataFrame([smiles_features(s) for s in train["SMILES"]])
test_feats = pd.DataFrame([smiles_features(s) for s in test["SMILES"]])

# Merge with original data
train = pd.concat([train, train_feats], axis=1)
test = pd.concat([test, test_feats], axis=1)

# -------------------------------
# Data Cleaning
# -------------------------------
# Keep rows where at least one target is present
train = train.dropna(how="all", subset=targets)

# -------------------------------
# Model Training
# -------------------------------
preds = {}

for target in targets:
    target_data = train.dropna(subset=[target])
    
    if target_data.empty:
        print(f"⚠️ Skipping {target}, no training data available.")
        preds[target] = np.zeros(len(test))  # fallback: zero prediction
        continue

    X = target_data.drop(columns=["id", "SMILES"] + targets)
    y = target_data[target]

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RF model
    model = RandomForestRegressor(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)

    # Validation score
    y_val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    print(f"✅ {target}: RMSE = {rmse:.4f}")

    # Predict on test
    X_test = test.drop(columns=["id", "SMILES"])
    preds[target] = model.predict(X_test)

# -------------------------------
# Submission
# -------------------------------
submission = pd.DataFrame({"id": test["id"]})

for target in targets:
    submission[target] = preds[target]

submission.to_csv("submission.csv", index=False)
print("✅ Submission file 'submission.csv' has been created!")


