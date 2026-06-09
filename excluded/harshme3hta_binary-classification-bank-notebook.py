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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# Target distribution
plt.figure(figsize=(12, 8))

# Target distribution plot
plt.subplot(2, 3, 1)
train_df['y'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.xlabel('Target')
plt.ylabel('Count')

# Age distribution
plt.subplot(2, 3, 2)
plt.hist(train_df['age'], bins=30, alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Education vs Target
plt.subplot(2, 3, 3)
education_target = train_df.groupby('education')['y'].mean()
education_target.plot(kind='bar')
plt.title('Target Rate by Education')
plt.xticks(rotation=45)

# Job vs Target
plt.subplot(2, 3, 4)
job_target = train_df.groupby('job')['y'].mean()
job_target.plot(kind='bar')
plt.title('Target Rate by Job')
plt.xticks(rotation=45)

# Marital Status vs Target
plt.subplot(2, 3, 5)
marital_target = train_df.groupby('marital')['y'].mean()
marital_target.plot(kind='bar')
plt.title('Target Rate by Marital Status')

# Campaign vs Target
plt.subplot(2, 3, 6)
plt.scatter(train_df['campaign'], train_df['y'], alpha=0.5)
plt.title('Campaign vs Target')
plt.xlabel('Campaign')
plt.ylabel('Target')

plt.tight_layout()
plt.show()


# Combine train and test for preprocessing
all_data = pd.concat([train_df.drop('y', axis=1), test_df], ignore_index=True)

# Identify categorical and numerical columns
categorical_cols = all_data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = all_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('id')  # Remove ID column

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                            labels=['young', 'adult', 'middle', 'senior', 'elderly'])
    
    # Create duration groups
    if 'duration' in df.columns:
        df['duration_group'] = pd.cut(df['duration'], bins=[0, 100, 300, 600, 5000], 
                                     labels=['short', 'medium', 'long', 'very_long'])
    
    # Create campaign groups
    df['campaign_group'] = pd.cut(df['campaign'], bins=[0, 1, 3, 10, 100], 
                                 labels=['low', 'medium', 'high', 'very_high'])
    
    # Previous campaign success rate
    if 'previous' in df.columns and 'poutcome' in df.columns:
        df['prev_success_rate'] = (df['poutcome'] == 'success').astype(int) * df['previous']
    
    return df

# Apply feature engineering
all_data_fe = feature_engineering(all_data)

# Label encoding for categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    all_data_fe[col] = le.fit_transform(all_data_fe[col].astype(str))
    label_encoders[col] = le

# Encode new categorical features
new_cat_cols = ['age_group', 'duration_group', 'campaign_group']
for col in new_cat_cols:
    if col in all_data_fe.columns:
        le = LabelEncoder()
        all_data_fe[col] = le.fit_transform(all_data_fe[col].astype(str))
        label_encoders[col] = le

# Split back to train and test
train_processed = all_data_fe[:len(train_df)].copy()
test_processed = all_data_fe[len(train_df):].copy()

# Add target back to train
train_processed['y'] = train_df['y'].values

print("Processed train shape:", train_processed.shape)
print("Processed test shape:", test_processed.shape)


# Prepare features and target
X = train_processed.drop(['id', 'y'], axis=1)
y = train_processed['y']
X_test = test_processed.drop(['id'], axis=1)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                  random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Train and evaluate models
model_scores = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                               scoring='roc_auc')
    
    print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    # Validation score
    val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    print(f"Validation AUC: {val_auc:.4f}")
    
    model_scores[name] = val_auc
    trained_models[name] = model

# Best model
best_model_name = max(model_scores, key=model_scores.get)
best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name} with AUC: {model_scores[best_model_name]:.4f}")



# Feature importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance.head(15)['feature'], feature_importance.head(15)['importance'])
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))



# Create ensemble predictions
ensemble_predictions = np.zeros(len(X_test_scaled))

for name, model in trained_models.items():
    pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    ensemble_predictions += pred_proba

# Average ensemble
ensemble_predictions /= len(trained_models)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': ensemble_predictions
})

print("Submission shape:", submission.shape)
print("Submission head:")
print(submission.head())

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved as 'submission.csv'")

# Display submission statistics
print(f"\nSubmission Statistics:")
print(f"Min probability: {submission['y'].min():.4f}")
print(f"Max probability: {submission['y'].max():.4f}")
print(f"Mean probability: {submission['y'].mean():.4f}")
print(f"Std probability: {submission['y'].std():.4f}")



# Confusion matrix for best model
val_pred = best_model.predict(X_val_scaled)
cm = confusion_matrix(y_val, val_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_val, val_pred))

# Model performance summary
print(f"\nModel Performance Summary:")
for name, score in model_scores.items():
    print(f"{name}: {score:.4f}")

print(f"\nEnsemble prediction range: [{ensemble_predictions.min():.4f}, {ensemble_predictions.max():.4f}]")


