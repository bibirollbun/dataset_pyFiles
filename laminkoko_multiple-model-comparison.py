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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib


df = pd.read_csv("/kaggle/input/mentally-stability-of-the-person/train.csv")
df_test = pd.read_csv("/kaggle/input/mentally-stability-of-the-person/test.csv")
df_sample_submission = pd.read_csv("/kaggle/input/mentally-stability-of-the-person/sample_submission.csv")


df.info()


# Drop unnecessary columns
df = df.drop(['id', 'Name'], axis=1, errors='ignore')
df.dropna()
    
    # Handle missing values
numeric_features = df.select_dtypes(include=['float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns
    
# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
    
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
])
    
    # Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # Drop any columns not specified
)
    


plt.figure(figsize=(15, 10))
    
    # Distribution of target variable
plt.subplot(2, 2, 1)
sns.countplot(data=df, x='Depression')
plt.title('Distribution of Depression Cases')
    
    # Age distribution
plt.subplot(2, 2, 2)
sns.histplot(data=df, x='Age', hue='Depression', bins=30)
plt.title('Age Distribution by Depression Status')
    
    # Sleep Duration vs Depression
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='Depression', y='Sleep Duration')
plt.title('Sleep Duration vs Depression')
    
    # Financial Stress vs Depression
plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='Depression', y='Financial Stress')
plt.title('Financial Stress vs Depression')
    
plt.tight_layout()
plt.show()
    
    # Correlation matrix for numeric features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


def train_and_evaluate_models(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Train model on full training set
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store results
        results[name] = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': model
        }
        
        # Plot ROC curve
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.show()
        
    return results


X = df.drop('Depression', axis=1)
y = df['Depression']


# Transform features
X_transformed = preprocessor.fit_transform(X)
    
    # Train and evaluate models
results = train_and_evaluate_models(X_transformed, y)
    
    # Save best model
best_model_name = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
best_model = results[best_model_name]['model']
    
    # Save model and preprocessor
joblib.dump(best_model, '/kaggle/working/best_model.joblib')
joblib.dump(preprocessor, '/kaggle/working/preprocessor.joblib')
    
    # Print results
for name, result in results.items():
    print(f"\nResults for {name}:")
    print(f"Cross-validation accuracy: {result['cv_mean']:.3f} (+/- {result['cv_std']*2:.3f})")
    print("\nClassification Report:")
    print(result['classification_report'])
    print("\nConfusion Matrix:")
    print(result['confusion_matrix'])


# Preprocess test data
# Apply the same preprocessing steps as in the `load_and_preprocess_data` function
df_test_processed = df_test.copy()
df_test_processed.drop(['id', 'Name'], axis=1, errors='ignore')

# Select numeric and categorical features
numeric_features = df_test_processed.select_dtypes(include=['float64', 'int64']).columns
categorical_features = df_test_processed.select_dtypes(include=['object']).columns

# Impute missing values
for col in numeric_features:
    median_val = df_test_processed[col].median()
    df_test_processed[col].fillna(median_val, inplace=True)

for col in categorical_features:
    df_test_processed[col].fillna('missing', inplace=True)

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    df_test_processed[col] = le.fit_transform(df_test_processed[col])

# Preprocess test data using the fitted preprocessor
X_test = df_test_processed
X_test_transformed = preprocessor.transform(X_test)

# Make predictions
pred_array_baseline = best_model.predict(X_test_transformed)

# Create submission file
df_test["Depression"] = pred_array_baseline
submission = pd.DataFrame({"id": df_test["id"],
                            "Depression": df_test["Depression"]})
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")

