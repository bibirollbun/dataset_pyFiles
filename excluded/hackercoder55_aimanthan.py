!pip install --upgrade scikit-learn imbalanced-learn


import sklearn
import imblearn

print(f"Scikit-Learn Version: {sklearn.__version__}")
print(f"Imbalanced-Learn Version: {imblearn.__version__}")

# If Scikit-Learn is 1.3 or higher, the error will disappear.




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE 
import joblib
import os
print("Files in input folder:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd

# Load the specific Fraud Training file
file_path = '/kaggle/input/manthan-ai-2025-ai-summit/ecommerce_fraud_train.csv'
df = pd.read_csv(file_path)

# Show the column names and the first 5 rows
print("Dataset Loaded Successfully!")
print("Columns in the table:", df.columns.tolist())
display(df.head())


# 1. Prepare the Data
# Drop 'is_fraud' (the answer) and 'user_id' (useless for prediction)
X = df.drop(['is_fraud', 'user_id'], axis=1)
y = df['is_fraud']

# 2. Clean the Data
# Fill missing values (like empty credit scores) with 0
X = X.fillna(0)

# Convert text columns (like 'mobile', 'chrome') into numbers
X = pd.get_dummies(X, drop_first=True)

print("Starting SMOTE (This handles the imbalance)...")
# 3. Apply SMOTE to balance the data (The B.Tech requirement)
from imblearn.over_sampling import SMOTE 
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"Data Balanced! Fraud cases increased from {sum(y==1)} to {sum(y_res==1)}")

# 4. Train the Model
print("Training Random Forest (This might take 1 minute)...")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_res, y_res)

# 5. Save the Model & Column Names (Important for the App!)
import joblib
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')

print("SUCCESS! 'fraud_model.pkl' has been saved.")

