import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time

import warnings
warnings.filterwarnings('ignore')


# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')

# Combine train and test for preprocessing
all_data = pd.concat([train, test], axis=0, ignore_index=True)

# Preprocess the data
def preprocess_data(df):
    # Handle missing values
    df['Academic Pressure'].fillna(df['Academic Pressure'].mean(), inplace=True)
    df['Work Pressure'].fillna(df['Work Pressure'].mean(), inplace=True)
    df['CGPA'].fillna(df['CGPA'].mean(), inplace=True)
    df['Study Satisfaction'].fillna(df['Study Satisfaction'].mean(), inplace=True)
    df['Job Satisfaction'].fillna(df['Job Satisfaction'].mean(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Gender', 'City', 'Working Professional or Student', 'Profession', 
                        'Sleep Duration', 'Dietary Habits', 'Degree']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Convert binary columns to numeric
    df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
    
    return df

all_data = preprocess_data(all_data)

# Split back into train and test
train = all_data[:len(train)]
test = all_data[len(train):]

# Prepare features and target
features = [col for col in train.columns if col not in ['id', 'Name', 'Depression']]
X = train[features]
y = train['Depression']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Random search code with increased ranges in both directions
param_dist = {
    'n_estimators': [10, 50, 100, 500, 1000, 2500, 5000],
    'max_depth': [1, 3, 5, 8, 10, 15, 20],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
    'subsample': [0.1, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0],  
    'colsample_bytree': [0.1, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0],  
    'min_child_weight': [0, 1, 10, 25, 50, 75, 100]
}

# Initialize XGBoost classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Perform random search
random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=100, 
                                   cv=5, random_state=42, n_jobs=-1, verbose=1)

# Start timing
start_time = time.time()

# Fit the random search model
random_search.fit(X_train, y_train)

# End timing
end_time = time.time()

# Print results
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Evaluate on validation set
y_pred = random_search.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)
print(f"Validation accuracy: {val_accuracy:.4f}")

# Train final model on entire training data
best_model = random_search.best_estimator_
best_model.fit(X, y)

# Make predictions on test set
test_predictions = best_model.predict(test[features])

# Create submission file
submission = pd.DataFrame({'id': test['id'], 'Depression': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created.")





