!pip install -q lightautoml[all]

import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import torch
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import os

# Set consistent styling for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Store test IDs for final submission
test_ids = test['id'].copy()

# Drop 'id' as it is not predictive
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# Shuffle training data for better generalization
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)
train.head()


plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=train, palette=['#FF6B6B', '#4ECDC4'])
plt.title('Distribution of Personality Types', fontsize=16, fontweight='bold')
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

class_distribution = train['Personality'].value_counts(normalize=True) * 100
print("Class Distribution:")
print(f"Extrovert: {class_distribution['Extrovert']:.2f}%")
print(f"Introvert: {class_distribution['Introvert']:.2f}%")


numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 
                     'Going_outside', 'Friends_circle_size', 'Post_frequency']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(numerical_features):
    sns.boxplot(x='Personality', y=feature, data=train, ax=axes[i], 
                palette=['#FF6B6B', '#4ECDC4'])
    axes[i].set_title(f'{feature} by Personality', fontweight='bold')

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='Stage_fear', hue='Personality', data=train, 
              palette=['#FF6B6B', '#4ECDC4'])
plt.title('Stage Fear Distribution by Personality', fontsize=16, fontweight='bold')
plt.xlabel('Stage Fear', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Personality')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Drained_after_socializing', hue='Personality', data=train,
              palette=['#FF6B6B', '#4ECDC4'])
plt.title('Drained After Socializing by Personality', fontsize=16, fontweight='bold')
plt.xlabel('Drained After Socializing', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Personality')
plt.show()


# Identify columns
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
categorical_mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}

# Fill numerical missing values
for col in numerical_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

# Fill and encode categorical features
for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)
    
    train[col] = train[col].map(categorical_mapping)
    test[col] = test[col].map(categorical_mapping)
    
    train[col] = pd.to_numeric(train[col], errors='coerce').fillna(0)
    test[col] = pd.to_numeric(test[col], errors='coerce').fillna(0)

print("\nMissing values after preprocessing:")
print("Train:", train.isnull().sum().sum())
print("Test:", test.isnull().sum().sum())


# Encode target for correlation
le = LabelEncoder()
train_numeric = train.copy()
train_numeric['Personality_encoded'] = le.fit_transform(train_numeric['Personality'])

correlation_matrix = train_numeric.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Correlations with target
target_corr = correlation_matrix['Personality_encoded'].drop('Personality_encoded').sort_values(key=abs, ascending=True)
plt.figure(figsize=(10, 6))
target_corr.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Feature Correlations with Personality', fontsize=16, fontweight='bold')
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.show()


# Set random seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.set_num_threads(os.cpu_count())

# Define task
task = Task('binary')

# Initialize LightAutoML
automl = TabularAutoML(
    task=task,
    timeout=3600,  # 1 hour
    cpu_limit=os.cpu_count(),
    reader_params={
        'n_jobs': os.cpu_count(),
        'cv': 5,
        'random_state': RANDOM_STATE,
        'advanced_roles': True
    },
    general_params={
        "use_algos": [['lgb']]
    }
)

# Encode target
y_train_encoded = le.fit_transform(train['Personality'])
X_train = train.drop('Personality', axis=1).copy()
X_train['target'] = y_train_encoded

# Fit model
oof_predictions = automl.fit_predict(
    X_train, 
    roles={'target': 'target'},
    verbose=3
)


oof_pred_classes = (oof_predictions.data[:, 0] > 0.5).astype(int)

accuracy = accuracy_score(y_train_encoded, oof_pred_classes)
print(f"Out-of-Fold Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_train_encoded, oof_pred_classes, target_names=le.classes_))


test_predictions_proba = automl.predict(test).data
test_pred_classes = (test_predictions_proba[:, 0] > 0.5).astype(int)
test_predictions = le.inverse_transform(test_pred_classes)

submission = pd.DataFrame({
    'id': test_ids,
    'Personality': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
print(f"Predicted {sum(test_pred_classes==0)} Introverts and {sum(test_pred_classes==1)} Extroverts")
submission.head(10)

