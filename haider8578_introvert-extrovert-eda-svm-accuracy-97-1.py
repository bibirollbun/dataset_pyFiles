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


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# Display the first few rows of the training data
print("Training Data Head:")
print(train_df.head())

# Check the shape of the datasets
print("\nShape of training data:", train_df.shape)
print("Shape of test data:", test_df.shape)

# Check data types
print("\nData types in training data:")
print(train_df.dtypes)

# Check for missing values
print("\nMissing values in training data:")
print(train_df.isnull().sum())

print("\nMissing values in test data:")
print(test_df.isnull().sum())

# Check the distribution of the target variable
print("\nDistribution of Personality types:")
print(train_df['Personality'].value_counts())

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=train_df)
plt.title('Distribution of Personality Types')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()

# Explore numerical features
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
train_df[numerical_features].describe()

# Visualize distributions of numerical features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train_df[feature].dropna(), kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Explore categorical features
categorical_features = ['Stage_fear', 'Drained_after_socializing']
for feature in categorical_features:
    print(f"\nValue counts for {feature}:")
    print(train_df[feature].value_counts())

# Visualize categorical features
plt.figure(figsize=(12, 5))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, 2, i)
    sns.countplot(x=feature, hue='Personality', data=train_df)
    plt.title(f'{feature} by Personality')
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = train_df[numerical_features + ['Personality']].copy()
correlation_matrix['Personality'] = correlation_matrix['Personality'].apply(lambda x: 1 if x == 'Extrovert' else 0)
sns.heatmap(correlation_matrix.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Box plots for numerical features by personality
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Personality', y=feature, data=train_df)
    plt.title(f'{feature} by Personality')
plt.tight_layout()
plt.show()


# Separate features and target
X = train_df.drop('Personality', axis=1)
y = train_df['Personality']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)

# Define preprocessing steps
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_features = ['Stage_fear', 'Drained_after_socializing']

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

# Evaluate each model using cross-validation
results = {}
for name, model in models.items():
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = cv_scores.mean()
    print(f"{name}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Sort models by performance
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\nModel Ranking:")
for i, (name, score) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {score:.4f}")


# Select the best performing models for hyperparameter tuning
best_models = [model_name for model_name, _ in sorted_results[:3]]
print(f"\nTop models for hyperparameter tuning: {best_models}")

# Define hyperparameter grids for each model
param_grids = {
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.1, 1],
        'classifier__kernel': ['linear', 'rbf']
    }
}

# Perform grid search for each model
best_estimators = {}
for model_name in best_models:
    if model_name in param_grids:
        print(f"\nPerforming grid search for {model_name}...")
        
        # Create a pipeline with preprocessing and the model
        model = models[model_name]
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[model_name], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store the best estimator
        best_estimators[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")


# Evaluate the best models on the validation set
for model_name, estimator in best_estimators.items():
    # Make predictions
    y_pred = estimator.predict(X_val)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n{model_name} Validation Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_val, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Extrovert', 'Introvert'],
                yticklabels=['Extrovert', 'Introvert'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Select the best model based on validation accuracy
best_model_name = max(best_estimators.keys(), key=lambda k: accuracy_score(y_val, best_estimators[k].predict(X_val)))
best_model = best_estimators[best_model_name]
print(f"\nBest Model: {best_model_name}")


# Make predictions on the test data
test_predictions = best_model.predict(test_df)

# Create a submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")

# Display the first few rows of the submission
print("\nSubmission file head:")
print(submission.head())

# Check the distribution of predictions
print("\nDistribution of predictions:")
print(submission['Personality'].value_counts())


# If the best model is a tree-based model, let's analyze feature importance
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    # Get feature names after preprocessing
    feature_names = []
    
    # Add numerical feature names
    feature_names.extend(numerical_features)
    
    # Add categorical feature names after one-hot encoding
    cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    for i, feature in enumerate(categorical_features):
        categories = cat_encoder.categories_[i]
        for category in categories:
            feature_names.append(f"{feature}_{category}")
    
    # Get feature importances
    importances = best_model.named_steps['classifier'].feature_importances_
    
    # Create a dataframe for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance_df.head(10))


# Print a summary of the analysis
print("\n=== SUMMARY ===")
print(f"1. Dataset contains {train_df.shape[0]} training samples and {test_df.shape[0]} test samples.")
print(f"2. Target variable distribution: {train_df['Personality'].value_counts().to_dict()}")
print(f"3. Best performing model: {best_model_name}")
print(f"4. Validation accuracy: {accuracy_score(y_val, best_model.predict(X_val)):.4f}")
print("\n5. Key insights from EDA:")
print("   - Numerical features show varying distributions between extroverts and introverts")
print("   - Some features have missing values that need imputation")
print("   - Feature correlations with the target variable are moderate")
print("\n6. Recommendations:")
print("   - The model can be further improved by feature engineering")
print("   - Ensemble methods might provide better performance")
print("   - More data could help improve model generalization")


import joblib
joblib.dump(best_model, "Introvert_Extrovert.pkl")
print("Done")

