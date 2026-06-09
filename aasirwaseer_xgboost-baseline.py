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


# Import required libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Load data
train = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')

# Process age ranges
def process_age(age_str):
    if pd.isna(age_str):
        return None
    age_str = str(age_str).lower().strip()
    
    age_mapping = {
        '20-25': 22.5,
        '15-20': 17.5,
        '45 and above': 47.5,
        '22-25': 23.5,
        '50-60': 55,
        '30-35': 32.5,
        '35-44': 39.5,
        '25-30': 27.5,
        '25-25': 25,
        'less than 20': 19,
        'less than 20)': 19,
        'less than 20-25': 19,
        '30-25': 27.5,
        '30-40': 35,
        '30-30': 30,
        'less than 20-25': 22.5,
        '45-49': 47
    }
    
    # First try the mapping
    if age_str in age_mapping:
        return age_mapping[age_str]
    
    # Handle ranges
    if '-' in age_str:
        low, high = map(float, age_str.split('-'))
        return (low + high) / 2
        
    # Try direct conversion
    try:
        return float(age_str)
    except ValueError:
        # Default value if nothing else works
        return 25.0  # median age as fallback

# Define category mappings
category_mappings = {
    'Hormonal_Imbalance': {'No': 0, 'Yes': 1, 'Yes Significantly': 1, 'No, Yes, not diagnosed by a doctor': 0.5},
    'Hyperandrogenism': {'No': 0, 'Yes': 1},
    'Hirsutism': {'No': 0, 'Yes': 1, 'No, Yes, not diagnosed by a doctor': 0.5},
    'Conception_Difficulty': {'No': 0, 'Yes': 1, 'Yes, diagnosed by a doctor': 1, 'No, Yes, not diagnosed by a doctor': 0.5},
    'Insulin_Resistance': {'No': 0, 'Yes': 1, 'No, Yes, not diagnosed by a doctor': 0.5},
    'Exercise_Frequency': {
        'Never': 0, 'Rarely': 1, '1-2 Times a Week': 2, 
        '3-4 Times a Week': 3, '6-8 Times a Week': 4
    },
    'Exercise_Duration': {
        'Not Applicable': 0, 'Less than 30 minutes': 1, '30 minutes': 2,
        '45 minutes': 3, 'More than 30 minutes': 3, '30 minutes to 1 hour': 3
    },
    'Sleep_Hours': {
        '3-4 hours': 3.5, 'Less than 6 hours': 5, '6-8 hours': 7,
        '9-12 hours': 10.5, 'More than 12 hours': 13
    },
    'Exercise_Benefit': {
        'Not at All': 0, 'Not Much': 1, 'Somewhat': 2, 'Yes Significantly': 3
    }
}

# Process Exercise Type
def process_exercise_type(x):
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        if 'No Exercise' in x:
            return 0
        elif ',' in x:
            return 4  # Multiple types
        elif 'Cardio' in x:
            return 1
        elif 'Strength' in x:
            return 2
        elif 'Flexibility' in x:
            return 3
    return 0

# Preprocess data
def preprocess_data(df):
    df = df.copy()
    
    # Process Age
    df['Age'] = df['Age'].apply(process_age)
    
    # Process Exercise Type
    df['Exercise_Type'] = df['Exercise_Type'].apply(process_exercise_type)
    
    # Apply category mappings
    for col, mapping in category_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    return df

# Apply preprocessing
train = preprocess_data(train)
test = preprocess_data(test)

# Define features
numeric_features = ['Age', 'Weight_kg']
categorical_features = [col for col in train.columns if col not in numeric_features + ['ID', 'PCOS']]

# Create preprocessing pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        learning_rate=0.05,
        n_estimators=200,
        max_depth=5,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Prepare data for modeling
X = train[numeric_features + categorical_features]
y = train['PCOS'].map({'Yes': 1, 'No': 0})

# Fit model
model_pipeline.fit(X, y)

# Generate predictions
test_features = test[numeric_features + categorical_features]
predictions = model_pipeline.predict_proba(test_features)[:, 1]

# Create submission
submission = pd.DataFrame({
    'ID': test['ID'],
    'PCOS': predictions
})

# Save results
submission.to_csv('pcos_predictions.csv', index=False)

# Display results
print("Submission Preview:")
print(submission.head())
print("\nPrediction Statistics:")
print(f"Number of predictions: {len(predictions)}")
print(f"Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
print(f"Mean prediction: {predictions.mean():.3f}")


