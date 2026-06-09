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


# Import all libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# Load the data
print("Loading data...")
train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print("\nTraining data info:")
print(train_data.info())


print("First few rows of training data:")
print(train_data.head())

print("\nMissing values in training data:")
print(train_data.isnull().sum())

print("\nMissing values in test data:")
print(test_data.isnull().sum())


# Separate features and target
X_train = train_data.drop(['id', 'Personality'], axis=1)
y_train = train_data['Personality']
X_test = test_data.drop(['id'], axis=1)
test_ids = test_data['id']

print(f"Feature columns: {list(X_train.columns)}")
print(f"\nTarget distribution:")
print(y_train.value_counts())


# Handle categorical variables
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
print(f"Categorical columns: {categorical_cols}")

# Convert categorical columns to numerical
for col in categorical_cols:
    if col in X_train.columns:
        # For training data
        X_train[col] = X_train[col].map({'Yes': 1, 'No': 0})
        # For test data
        if col in X_test.columns:
            X_test[col] = X_test[col].map({'Yes': 1, 'No': 0})

print("\nAfter categorical encoding:")
print("Training data dtypes:", X_train.dtypes)


# Handle missing values
print("Handling missing values...")
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

print("Missing values after imputation:")
print("Training:", X_train_imputed.isnull().sum().sum())
print("Test:", X_test_imputed.isnull().sum().sum())


# Feature scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

print("Feature scaling completed")
print("Training data shape after processing:", X_train_scaled.shape)
print("Test data shape after processing:", X_test_scaled.shape)


# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train split shape: {X_train_split.shape}")
print(f"Validation split shape: {X_val_split.shape}")


# Define models to test
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True),
    'Neural Network': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50))
}

print(f"Models to evaluate: {list(models.keys())}")


# Store results
results = {}
model_objects = {}

print("Training and evaluating models...")
print("-" * 50)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_split, y_train_split)
    
    # Make predictions on validation set
    y_pred = model.predict(X_val_split)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val_split, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Store results
    results[name] = {
        'Validation Accuracy': accuracy,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    }
    
    model_objects[name] = model
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


# Display results summary
print("MODEL COMPARISON RESULTS")
print("="*50)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('CV Mean', ascending=False)
print(results_df)

# Select best model
best_model_name = results_df.index[0]
best_model = model_objects[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Best CV Score: {results_df.loc[best_model_name, 'CV Mean']:.4f}")


# Define parameter grids for each model
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'poly']
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
}

param_grid = param_grids.get(best_model_name, {})
print(f"Parameter grid for {best_model_name}:")
print(param_grid)


# Perform grid search
print(f"Performing hyperparameter tuning for {best_model_name}...")
grid_search = GridSearchCV(
    models[best_model_name], 
    param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")


# Final model
final_model = grid_search.best_estimator_

print("Training final model on complete training data...")
final_model.fit(X_train_scaled, y_train)

print("Final model training completed!")
print(f"Model: {best_model_name}")
print(f"Best parameters: {grid_search.best_params_}")


# Make predictions on test data
test_predictions = final_model.predict(X_test_scaled)

print(f"Generated {len(test_predictions)} predictions")
print(f"Prediction distribution:")
unique, counts = np.unique(test_predictions, return_counts=True)
for pred, count in zip(unique, counts):
    print(f"  {pred}: {count} ({count/len(test_predictions)*100:.1f}%)")


#  submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': test_predictions
})

# Save to CSV
submission.to_csv('personality_predictions.csv', index=False)
print(f"Predictions saved to 'personality_predictions.csv'")
print(f"Submission file shape: {submission.shape}")
print("\nFirst few predictions:")
print(submission.head(10))


print(f"✅ Best Model: {best_model_name}")
print(f"✅ Best CV Score: {grid_search.best_score_:.4f}")
print(f"✅ Predictions saved to: personality_predictions.csv")
print(f"✅ Total predictions: {len(test_predictions)}")




