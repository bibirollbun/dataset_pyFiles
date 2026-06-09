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


# ğŸ“¦ STEP 1: Import Required Libraries

import pandas as pd                       # For data handling
import numpy as np                        # For numerical operations
import os                                 # For directory/file management
import matplotlib.pyplot as plt           # For plotting
import seaborn as sns                     # For prettier visualizations

from sklearn.model_selection import train_test_split   # For splitting data
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding and scaling
from sklearn.ensemble import RandomForestClassifier     # Optional classifier
from xgboost import XGBClassifier                      # XGBoost for better performance
from sklearn.metrics import classification_report       # For classification metrics

# ğŸ“� Ensure a directory exists to save plots
os.makedirs("plots", exist_ok=True)

# ğŸ“‚ STEP 2: Load Dataset (Simulated here, replace with real dataset path)

np.random.seed(42)  # Set random seed for reproducibility

# Create a mock dataset with sample features
df = pd.DataFrame({
    "id": range(1000),  # Unique ID for each sample
    "Crop Type": np.random.choice(["Wheat", "Rice", "Maize", "Cotton"], 1000),  # Crop types
    "Soil pH": np.random.uniform(4.5, 9.0, 1000),        # Soil pH values
    "Rainfall": np.random.uniform(50, 300, 1000),        # Rainfall in mm
    "Fertilizer Name": np.random.choice(["Urea", "DAP", "NPK", "14-35-14", "10-26-26"], 1000)  # Target labels
})

# ğŸ“Š STEP 3: Exploratory Data Analysis (EDA)

# Plot distribution of fertilizer classes
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Fertilizer Name', order=df['Fertilizer Name'].value_counts().index)
plt.title("Fertilizer Class Distribution")
plt.savefig("plots/class_distribution.png")  # Save plot to disk
plt.close()

# Plot soil pH histogram
plt.figure(figsize=(6,4))
sns.histplot(df['Soil pH'], bins=30, kde=True)
plt.title("Soil pH Distribution")
plt.savefig("plots/soil_ph.png")  # Save plot to disk
plt.close()

# ğŸ§¼ STEP 4: Data Preprocessing & Feature Engineering

# Encode 'Crop Type' from string to numeric values
df['Crop Type'] = LabelEncoder().fit_transform(df['Crop Type'])

# Encode target variable 'Fertilizer Name'
label_encoder = LabelEncoder()
df['Fertilizer_Label'] = label_encoder.fit_transform(df['Fertilizer Name'])

# Define feature columns and target
features = ['Crop Type', 'Soil pH', 'Rainfall']
target = 'Fertilizer_Label'

# âœ‚ï¸� STEP 5: Train-Test Split

# Split data into 80% training and 20% validation sets
X = df[features]        # Input features
y = df[target]          # Target label
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale feature values (mean=0, std=1) to improve model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# âš™ï¸� STEP 6: Train XGBoost Classifier

# Instantiate and train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train_scaled, y_train)

# ğŸ“ˆ STEP 7: Define Custom MAP@3 Evaluation Metric

def mapk(actual, predicted, k=3):
    """
    Calculates the Mean Average Precision at K.
    actual: list of true labels
    predicted: list of top-k predicted label lists
    """
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                score += 1.0 / (i + 1.0)
                break  # Only count the first correct prediction
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Predict probabilities for all classes
probs = model.predict_proba(X_val_scaled)

# Get indices of top 3 predictions for each sample
top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # Reverse order for top predictions

# Compute MAP@3 score
map3_score = mapk(y_val.tolist(), top3_preds.tolist(), k=3)
print(f"Validation MAP@3: {map3_score:.4f}")

# ğŸ§ª STEP 8: Make Predictions on Test Set (Simulated here)

# Simulate test data (you should replace this with actual test set)
test_df = pd.DataFrame({
    "id": range(2000, 2100),  # New sample IDs
    "Crop Type": np.random.choice(["Wheat", "Rice", "Maize", "Cotton"], 100),
    "Soil pH": np.random.uniform(4.5, 9.0, 100),
    "Rainfall": np.random.uniform(50, 300, 100)
})

# Encode 'Crop Type' in test set
test_df['Crop Type'] = LabelEncoder().fit_transform(test_df['Crop Type'])

# Apply the same scaling to test features
X_test_scaled = scaler.transform(test_df[features])

# Predict probabilities for test samples
test_probs = model.predict_proba(X_test_scaled)

# Get top 3 predicted fertilizer labels per sample
test_top3 = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]

# Convert numeric labels back to original fertilizer names
test_top3_labels = [
    " ".join(label_encoder.inverse_transform(row))
    for row in test_top3
]

# ğŸ“� STEP 9: Generate Submission File

# Combine IDs and predicted fertilizers into a submission dataframe
submission = pd.DataFrame({
    "id": test_df["id"],
    "Fertilizer Name": test_top3_labels
})

# Save submission file to disk
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")


