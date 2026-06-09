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


# ============================================================
# 0) IMPORTS
# ============================================================
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Base path for Kaggle dataset
base_path = "/kaggle/input/joint-crack-prediction-regression/data_examen2/"

# ============================================================
# 1) INSPECT FILES
# ============================================================
print("=== Files found in dataset ===")
for dirname, _, filenames in os.walk(base_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ============================================================
# 2) LOAD CRACK LENGTH DATA
# ============================================================
crack_path = os.path.join(base_path, "crack_length.csv")
crack_length_df = pd.read_csv(crack_path)

print("\n=== Crack Length Data Info ===")
print(crack_length_df.head())
print("\nColumns:", crack_length_df.columns.tolist())
print("\nShape:", crack_length_df.shape)
print("\nDescription:\n", crack_length_df.describe(include="all"))

# Rename columns for convenience
crack_length_df = crack_length_df.rename(columns={
    "Tnumber": "Joint",
    "Number of cycle": "Cycle",
    "Crack length (mm)": "CrackLength"
})

print("\n✅ Renamed Crack Length Columns:", crack_length_df.columns.tolist())

# ============================================================
# 3) SIGNAL FILE INSPECTION
# ============================================================
sample_signal = os.path.join(base_path, "T2/72000/signal_1.csv")
sig_df = pd.read_csv(sample_signal)

print("\n=== Sample Signal File ===")
print("File:", sample_signal)
print("Columns:", sig_df.columns.tolist())
print(sig_df.head())
print(sig_df.describe(include="all"))

# ============================================================
# 4) FEATURE EXTRACTION FUNCTION
# ============================================================
def extract_signal_features(signal_file):
    try:
        df = pd.read_csv(signal_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(f"⚠️ No numeric columns in {signal_file}")
            return None

        features = {}
        for col in numeric_cols:
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_max"] = df[col].max()
            features[f"{col}_min"] = df[col].min()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_range"] = df[col].max() - df[col].min()
        return features
    except Exception as e:
        print(f"Error processing {signal_file}: {e}")
        return None

# ============================================================
# 5) BUILD TRAINING DATA
# ============================================================
X_train = []
y_train = []

train_joints = ['T2', 'T3', 'T4', 'T5']
for joint in train_joints:
    joint_path = os.path.join(base_path, joint)
    cycles = [d for d in os.listdir(joint_path) if os.path.isdir(os.path.join(joint_path, d))]
    
    for cycle in cycles:
        cycle_path = os.path.join(joint_path, cycle)
        signal_1 = os.path.join(cycle_path, "signal_1.csv")
        signal_2 = os.path.join(cycle_path, "signal_2.csv")
        
        features = {}
        if os.path.exists(signal_1):
            feat1 = extract_signal_features(signal_1)
            if feat1: features.update({f"s1_{k}": v for k,v in feat1.items()})
        if os.path.exists(signal_2):
            feat2 = extract_signal_features(signal_2)
            if feat2: features.update({f"s2_{k}": v for k,v in feat2.items()})
        
        features['cycle'] = int(cycle)
        features['Joint'] = joint
        
        # Match against crack_length_df
        crack_row = crack_length_df[
            (crack_length_df["Joint"] == joint) &
            (crack_length_df["Cycle"] == int(cycle))
        ]
        
        if not crack_row.empty:
            y_val = crack_row["CrackLength"].values[0]
            X_train.append(features)
            y_train.append(y_val)

print("\n✅ Training data built")
print("Number of samples:", len(X_train))

# ============================================================
# 6) BUILD TEST DATA (T1)
# ============================================================
test_joint = "T1"
test_cycles = [50000, 60000, 62500, 65500, 69025, 70026, 70766]
X_test = []

for cycle in test_cycles:
    cycle_path = os.path.join(base_path, test_joint, str(cycle))
    signal_1 = os.path.join(cycle_path, "signal_1.csv")
    signal_2 = os.path.join(cycle_path, "signal_2.csv")
    
    features = {}
    if os.path.exists(signal_1):
        feat1 = extract_signal_features(signal_1)
        if feat1: features.update({f"s1_{k}": v for k,v in feat1.items()})
    if os.path.exists(signal_2):
        feat2 = extract_signal_features(signal_2)
        if feat2: features.update({f"s2_{k}": v for k,v in feat2.items()})
    
    features['cycle'] = cycle
    features['Joint'] = test_joint
    X_test.append(features)

# ============================================================
# 7) ALIGN COLUMNS
# ============================================================
X_train_df = pd.DataFrame(X_train).fillna(0)
X_test_df = pd.DataFrame(X_test).fillna(0)

# Align columns
missing_cols = set(X_train_df.columns) - set(X_test_df.columns)
for col in missing_cols:
    X_test_df[col] = 0
X_test_df = X_test_df[X_train_df.columns]

print("\n=== Train DataFrame ===")
print(X_train_df.head())
print("\n=== Test DataFrame ===")
print(X_test_df.head())

# ============================================================
# 8) MODEL TRAINING
# ============================================================
if len(X_train_df) > 0:
    y_train_series = pd.Series(y_train)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_df.drop(columns=["Joint"]), y_train_series)

    predictions = model.predict(X_test_df.drop(columns=["Joint"]))

    # Save submission
    submission = pd.DataFrame({
        "ID": test_cycles,
        "TARGET": predictions
    })
    submission.to_csv("submission.csv", index=False)
    print("\n✅ Submission file created: submission.csv")
    print(submission)
else:
    print("⚠️ No training data available. Check crack_length.csv structure.")

# ============================================================
# 9) EDA - QUICK PLOTS
# ============================================================
plt.figure(figsize=(8,5))
sns.scatterplot(data=crack_length_df, x="Cycle", y="CrackLength", hue="Joint")
plt.title("Crack Length Progression by Joint")
plt.show()

sns.histplot(y_train, bins=10, kde=True)
plt.title("Distribution of Crack Lengths in Training Data")
plt.show()


