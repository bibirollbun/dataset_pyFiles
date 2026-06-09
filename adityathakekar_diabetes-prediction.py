import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# --- STEP 0: CONFIGURATION (UNCOMMENT THE ONE YOU ARE DOING) ---

# OPTION 1: TITANIC
#TARGET = 'Survived'; problem_type = 'classification'; ID_COL = 'PassengerId'

# OPTION 2: SPACESHIP TITANIC (Uncomment 3 lines below if doing this)
# TARGET = 'Transported'; problem_type = 'classification'; ID_COL = 'PassengerId'

# OPTION 3: DIABETES (Playground S5E12) (Uncomment 3 lines below if doing this)
TARGET = 'diagnosed_diabetes' 
problem_type = 'classification' 
ID_COL = 'id'

# OPTION 4: HOUSE PRICES (Uncomment 3 lines below if doing this)
# TARGET = 'SalePrice'; problem_type = 'regression'; ID_COL = 'Id'

# ---------------------------------------------------------------

# --- STEP 1: ROBUST DATA LOADING ---
train_path = ''
test_path = ''

# Search input directory for the correct files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if 'train' in filename:
            train_path = os.path.join(dirname, filename)
        elif 'test' in filename:
            test_path = os.path.join(dirname, filename)

print(f"Loaded Train: {train_path}")
print(f"Loaded Test: {test_path}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# --- STEP 2: MINIMAL PREPROCESSING ---
print("Preprocessing data...")

# Drop Target from Train to define X and y
X = train.drop([TARGET], axis=1)
y = train[TARGET]

# Align Test columns
X_test = test.copy() 

# Keep only Number and Text columns (Drop complex stuff like dates/objects for now)
X = X.select_dtypes(include=['number', 'object'])
X_test = X_test[X.columns]

# Handle Missing Values
num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Fill numbers with 0
if len(num_cols) > 0:
    imputer_num = SimpleImputer(strategy='constant', fill_value=0)
    X[num_cols] = imputer_num.fit_transform(X[num_cols])
    X_test[num_cols] = imputer_num.transform(X_test[num_cols])

# Fill text with "Unknown"
if len(cat_cols) > 0:
    imputer_cat = SimpleImputer(strategy='constant', fill_value='Unknown')
    X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
    X_test[cat_cols] = imputer_cat.transform(X_test[cat_cols])

# Encode Text to Numbers
for col in cat_cols:
    le = LabelEncoder()
    # Combine to fit all possible categories
    combined = pd.concat([X[col], X_test[col]]).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# --- STEP 3: TRAIN MODEL ---
print(f"Training on {X.shape[1]} features...")

if problem_type == 'classification':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X, y)

# --- STEP 4: PREDICT & SUBMIT ---
predictions = model.predict(X_test)

# Fix for Spaceship Titanic (needs True/False, not 0/1)
if TARGET == 'Transported': 
    predictions = predictions.astype(bool)

# Create submission file
output = pd.DataFrame({ID_COL: test[ID_COL], TARGET: predictions})
output.to_csv('submission.csv', index=False)
print("SUCCESS: submission.csv saved!")




