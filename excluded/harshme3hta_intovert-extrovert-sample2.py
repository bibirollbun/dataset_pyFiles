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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print(f"âœ… Training data shape: {train_df.shape}")
print(f"âœ… Test data shape: {test_df.shape}")


print("\nFirst 5 rows of training data:")
print(train_df.head())

print("\nDataset Info:")
print(train_df.info())

print("\nTarget variable distribution:")
print(train_df['Personality'].value_counts())


print("\nğŸ”� Missing Values Check:")
print("Training data missing values:")
print(train_df.isnull().sum())
print("\nTest data missing values:")
print(test_df.isnull().sum())


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
train_df['Personality'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Distribution of Personality Types')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.xticks(rotation=0)


plt.subplot(1, 3, 2)
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numerical_cols:
    numerical_cols.remove('id')
corr_matrix = train_df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Matrix')



plt.subplot(1, 3, 3)
if len(numerical_cols) > 1:
    feature_to_plot = numerical_cols[1]  # Skip target if it's numerical
    sns.boxplot(data=train_df, x='Personality', y=feature_to_plot)
    plt.title(f'Distribution of {feature_to_plot} by Personality')

plt.tight_layout()
plt.show()


print("\nğŸ“Š Statistical Summary:")
print(train_df.describe())


X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']
X_test = test_df.drop(['id'], axis=1)

print(f"âœ… Features shape: {X.shape}")
print(f"âœ… Target shape: {y.shape}")
print(f"âœ… Test features shape: {X_test.shape}")


print("\nData types:")
print(X.dtypes.value_counts())

# Handle any categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"ğŸ“� Categorical columns: {categorical_cols}")
print(f"ğŸ”¢ Numerical columns: {len(numerical_cols)} columns")


if categorical_cols:
    print("\nğŸ”¤ Encoding categorical variables...")
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        le_dict[col] = le
        print(f"âœ… Encoded {col}")



if X.isnull().sum().sum() > 0:
    print("\nğŸ”§ Handling missing values...")
    # Fill numerical missing values with median
    for col in numerical_cols:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
            print(f"âœ… Filled missing values in {col} with median: {median_val}")

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"\nğŸ�¯ Target classes: {le_target.classes_}")


print("\nâš™ï¸� Starting Feature Engineering...")

# Create statistical features
feature_cols = X.columns.tolist()

# Add sum, mean, std, min, max across all features for each row
X['feature_sum'] = X[feature_cols].sum(axis=1)
X['feature_mean'] = X[feature_cols].mean(axis=1)
X['feature_std'] = X[feature_cols].std(axis=1)
X['feature_min'] = X[feature_cols].min(axis=1)
X['feature_max'] = X[feature_cols].max(axis=1)

# Do the same for test data
X_test['feature_sum'] = X_test[feature_cols].sum(axis=1)
X_test['feature_mean'] = X_test[feature_cols].mean(axis=1)
X_test['feature_std'] = X_test[feature_cols].std(axis=1)
X_test['feature_min'] = X_test[feature_cols].min(axis=1)
X_test['feature_max'] = X_test[feature_cols].max(axis=1)

print("âœ… Created statistical features: sum, mean, std, min, max")


if len(feature_cols) >= 2:
    # Create a few interaction features between highly correlated features
    corr_matrix = X[feature_cols].corr().abs()
    
    # Find top correlated pairs
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i,j]))
    
    # Sort by correlation and take top 3
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = corr_pairs[:3]
    
    for i, (col1, col2, corr_val) in enumerate(top_pairs):
        interaction_name = f'interaction_{i+1}'
        X[interaction_name] = X[col1] * X[col2]
        X_test[interaction_name] = X_test[col1] * X_test[col2]
        print(f"âœ… Created {interaction_name}: {col1} * {col2} (corr: {corr_val:.3f})")

print(f"\nğŸ“Š Final feature set shape: {X.shape}")


print("\nâš–ï¸� Scaling features...")

# Split the data for training
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("âœ… Features scaled using StandardScaler")
print(f"âœ… Training set: {X_train_scaled.shape}")
print(f"âœ… Validation set: {X_val_scaled.shape}")



print("\nğŸ¤– Training Multiple Models...")

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000
    ),
    'SVM': SVC(
        kernel='rbf',
        random_state=42,
        probability=True
    )
}

# Train and evaluate each model
model_scores = {}
trained_models = {}

for name, model in models.items():
    print(f"\nğŸ”„ Training {name}...")
    
    # Train the model
    if name in ['Logistic Regression', 'SVM']:
        # Use scaled features for these models
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    else:
        # Use original features for tree-based models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    model_scores[name] = {
        'Validation Accuracy': accuracy,
        'CV Mean': cv_mean,
        'CV Std': cv_std
    }
    trained_models[name] = model
    
    print(f"âœ… {name} - Validation Accuracy: {accuracy:.4f}")
    print(f"   Cross-validation: {cv_mean:.4f} (+/- {cv_std*2:.4f})")

# Display model comparison
print("\nğŸ“Š Model Comparison:")
scores_df = pd.DataFrame(model_scores).T
print(scores_df.round(4))



best_models = [
    ('rf', trained_models['Random Forest']),
    ('gb', trained_models['Gradient Boosting']),
    ('lr', trained_models['Logistic Regression'])
]

# Create ensemble
ensemble = VotingClassifier(
    estimators=best_models,
    voting='soft'  # Use probability averaging
)

# Train ensemble
print("ğŸ”„ Training ensemble model...")
ensemble.fit(X_train_scaled, y_train)  # Using scaled features for ensemble

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_val_scaled)
ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)

print(f"âœ… Ensemble Accuracy: {ensemble_accuracy:.4f}")


if ensemble_accuracy > max([scores['Validation Accuracy'] for scores in model_scores.values()]):
    print("ğŸ�† Using Ensemble Model for final predictions")
    final_model = ensemble
    final_predictions = final_model.predict(X_test_scaled)
    final_probabilities = final_model.predict_proba(X_test_scaled)
else:
    # Find best single model
    best_model_name = max(model_scores.keys(), 
                         key=lambda x: model_scores[x]['Validation Accuracy'])
    print(f"ğŸ�† Using {best_model_name} for final predictions")
    final_model = trained_models[best_model_name]
    
    if best_model_name in ['Logistic Regression', 'SVM']:
        final_predictions = final_model.predict(X_test_scaled)
        final_probabilities = final_model.predict_proba(X_test_scaled)
    else:
        final_predictions = final_model.predict(X_test)
        final_probabilities = final_model.predict_proba(X_test)

# Convert predictions back to original labels
final_predictions_labels = le_target.inverse_transform(final_predictions)

print(f"âœ… Generated {len(final_predictions)} predictions")
print(f"ğŸ“Š Prediction distribution:")
unique, counts = np.unique(final_predictions_labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"   {label}: {count} ({count/len(final_predictions)*100:.1f}%)")


submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': final_predictions_labels
})

# Save submission file
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file saved as 'submission.csv'")
print(f"âœ… Submission shape: {submission.shape}")
print("\nFirst 10 predictions:")
print(submission.head(10))


if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nğŸ�¯ Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

elif hasattr(final_model, 'coef_'):
    # For logistic regression
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': final_model.coef_[0]
    })
    feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    
    print("\nğŸ�¯ Top 10 Most Important Features (by coefficient magnitude):")
    print(feature_importance.head(10))


print("\n" + "="*60)
print("ğŸ�‰ SOLUTION SUMMARY")
print("="*60)
print(f"ğŸ“Š Dataset: {train_df.shape[0]} training samples, {test_df.shape[0]} test samples")
print(f"ğŸ”§ Features: {X.shape[1]} features after engineering")
print(f"ğŸ¤– Models trained: {len(models)} individual models + 1 ensemble")
print(f"ğŸ�† Best model accuracy: {max([scores['Validation Accuracy'] for scores in model_scores.values()]):.4f}")
print(f"ğŸ�­ Ensemble accuracy: {ensemble_accuracy:.4f}")
print(f"ğŸ“� Submission file: 'submission.csv' with {len(final_predictions)} predictions")
print("\nğŸ’¡ Key techniques used:")
print("   â€¢ Feature engineering (statistical features, interactions)")
print("   â€¢ Multiple model comparison")
print("   â€¢ Cross-validation")
print("   â€¢ Ensemble methods")
print("   â€¢ Feature scaling")
print("\nğŸš€ Ready for submission to Kaggle!")
print("="*60)




