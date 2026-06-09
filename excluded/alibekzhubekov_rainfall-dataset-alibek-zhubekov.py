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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as ex
from scipy import stats


plt.style.use('seaborn-v0_8')
sns.set_theme()


base_dir = '/kaggle/input/playground-series-s5e3/'
train_data = pd.read_csv(base_dir + 'train.csv')
test_data = pd.read_csv(base_dir + 'train.csv')


# Basic information about the dataset
print("\nTraining Dataset Info:")
print("=====================")
print(train_data.info())
print("\nShape:", train_data.shape)
print("\nFirst few rows:")
print(train_data.head())

print("\nMissing values in training data:")
print(train_data.isnull().sum())

print("Basic statistics of numerical columns:")
print(train_data.describe())

# Analysis of target variable
print("\nTarget Variable Distribution:")
print(train_data['rainfall'].value_counts())
print("\nPercentage:")
print(train_data['rainfall'].value_counts(normalize=True) * 100)



# Visualizations
# 1. Target Variable
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='rainfall', data=train_data)
plt.title('Distribution of Rainfall (Target Variable)')

# Use bar_label to add labels to each bar in the container
ax.bar_label(ax.containers[0], padding=3, fontsize=12)
plt.show()


#2. Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = train_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()



# 3. Distribution of numerical features
numerical_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                     'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

plt.figure(figsize=(15, 20))
for i, feature in enumerate(numerical_features):
    plt.subplot(5, 2, i+1)
    sns.histplot(train_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()



# 4. Feature distributions by rainfall
plt.figure(figsize=(15, 20))
for i, feature in enumerate(numerical_features):
    plt.subplot(5, 2, i+1)
    sns.boxplot(x='rainfall', y=feature, data=train_data)
    plt.title(f'{feature} by Rainfall')
plt.tight_layout()
plt.show()



# 5. Scatter plot of temperature vs. pressure colored by rainfall
plt.figure(figsize=(10, 8))
sns.scatterplot(x='temparature', y='pressure', hue='rainfall', data=train_data)
plt.title('Temperature vs. Pressure by Rainfall')
plt.show()



# 6. Day of year vs rainfall
plt.figure(figsize=(12, 6))
sns.lineplot(x='day', y='rainfall', data=train_data, estimator='mean', ci=None)
plt.title('Daily Rainfall Probability Throughout the Year')
plt.xlabel('Day of Year')
plt.ylabel('Rainfall Probability')
plt.show()


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')



# Set random seed for reproducibility
np.random.seed(42)

# Load the data
train_data = pd.read_csv(base_dir + 'train.csv')
test_data = pd.read_csv(base_dir + 'test.csv')

# Separate features and target
X = train_data.drop(['id', 'rainfall'], axis=1)
y = train_data['rainfall']
test_features = test_data.drop(['id'], axis=1)

# Print class distribution
print("\nClass distribution in training data:")
print(y.value_counts(normalize=True) * 100)


# ==============================================
# Feature engineering
# ==============================================

# Create cyclical features for day (to capture seasonality)
def create_cyclical_features(df, col, period):
    """Create sin and cos features to capture cyclical nature of time variables"""
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/period)
    return df

# Apply to both train and test
X = create_cyclical_features(X, 'day', 365)
test_features = create_cyclical_features(test_features, 'day', 365)

# Create some intercation features based on EDA insights
# Temperature and humidity interaction (hypothetical relationship)
X['temp_humidity'] = X['temparature'] * X['humidity']
test_features['temp_humidity'] = test_features['temparature'] * test_features['humidity']

# Cloud cover and sunshine intercation
X['cloud_sunshine'] = X['cloud'] * X['sunshine']
test_features['cloud_sunshine'] = test_features['cloud'] * test_features['sunshine']



# ==============================================
# Train-Test Split and Stratification
# ==============================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================================
# Feature selection
# ==============================================

# Get correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print(f'Dropping highly correlated features: {to_drop}')

# Drop features
X_train = X_train.drop(to_drop, axis=1)
X_val = X_val.drop(to_drop, axis=1)
test_features = test_features.drop(to_drop, axis=1)



# ==============================================
# Model Training and Evaluation
# ==============================================

# Create a list of models to try
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each model
best_model = None
best_score = 0

results = {}

for name, model in models.items():
    print(f'\nTraining {name}...')

    # Create a Pipeline with scaling
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_val)
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    # Calculate PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)

    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'model': pipeline
    }

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.show()

    # Check if this is the best model based on PR AUC
    if pr_auc > best_score:
        best_score = pr_auc
        best_model = name


# Print summary of all models
print("\n===== Model Performance Summary =====")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  PR AUC:    {metrics['pr_auc']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")


print(f"\nBest model based on PR AUC: {best_model} with score {best_score:.4f}")


# =============================================================================
# Generate Predictions on Test Set
# =============================================================================
best_pipeline = results[best_model]['model']
test_predictions = best_pipeline.predict_proba(test_features)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_predictions
})

submission.to_csv('submission.csv', index=False)





