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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the data 
train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')

# Display basic info
print("Train Data Shape:", train.shape)
print("Test Data Shape:", test.shape)
print("\nTrain Columns:", train.columns.tolist())
print("\nTrain Target Distribution:\n", train['NObeyesdad'].value_counts())


# Separate features and target
X = train.drop(['id', 'NObeyesdad'], axis=1)
y = train['NObeyesdad']
test_ids = test['id']
test_features = test.drop('id', axis=1)

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Define preprocessing functions
def preprocess_data(df):
    # Copy to avoid modifying original data
    df = df.copy()

    # Binary encoding
    binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    for col in binary_cols[1:]:
        df[col] = df[col].map({'no': 0, 'yes': 1})

    # Ordinal encoding
    ordinal_mappings = {
        'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    }
    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping)

    # One-hot encoding for nominal features
    nominal_cols = ['MTRANS']
    df = pd.get_dummies(df, columns=nominal_cols)

    return df

# Apply preprocessing
X_processed = preprocess_data(X)
test_processed = preprocess_data(test_features)

# Align test columns with training data
test_processed = test_processed.reindex(columns=X_processed.columns, fill_value=0)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42
)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_val)
print("Decision Tree Accuracy:", accuracy_score(y_val, y_pred_dt))
print(classification_report(y_val, y_pred_dt, target_names=le.classes_))


bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42
)
bag.fit(X_train, y_train)
y_pred_bag = bag.predict(X_val)
print("Bagging Accuracy:", accuracy_score(y_val, y_pred_bag))
print(classification_report(y_val, y_pred_bag, target_names=le.classes_))


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf, target_names=le.classes_))


xgb_clf = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    random_state=42
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_val)
print("XGBoost Accuracy:", accuracy_score(y_val, y_pred_xgb))
print(classification_report(y_val, y_pred_xgb, target_names=le.classes_))


from sklearn.metrics import precision_recall_fscore_support, accuracy_score
def compare_models(models, model_names, X_test, y_test):
    """
    Compare model metrics and create visualizations
    Returns DataFrame with comparison results
    """
    metrics_dict = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    for name, model in zip(model_names, models):
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Store results
        metrics_dict['Model'].append(name)
        metrics_dict['Accuracy'].append(accuracy)
        metrics_dict['Precision'].append(precision)
        metrics_dict['Recall'].append(recall)
        metrics_dict['F1-Score'].append(f1)

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_dict)
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    metrics_df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
    
    sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df_melted)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    
    # Add value labels
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.show()
    
    return metrics_df

# Assuming you have trained models stored in these variables
models = [dt, bag, rf, xgb_clf]
model_names = ['Decision Tree', 'Bagging', 'Random Forest', 'XGBoost']

# Run comparison (using validation set)
metrics_comparison = compare_models(models, model_names, X_val, y_val)

# Display formatted table
print("\nModel Metrics Comparison:")
print(metrics_comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def plot_feature_importance(model, title, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title(title)
    sns.barplot(x=importances[indices[:10]], y=features[indices[:10]])
    plt.show()

# Decision Tree
plot_feature_importance(dt, 'Decision Tree Feature Importance', X_processed.columns)

# Random Forest
plot_feature_importance(rf, 'Random Forest Feature Importance', X_processed.columns)

# XGBoost
plot_feature_importance(xgb_clf, 'XGBoost Feature Importance', X_processed.columns)


plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_val, y_pred_xgb), annot=True, fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


plt.figure(figsize=(12, 6))  # Increase figure size
ax = sns.countplot(x=y)
plt.title('Class Distribution')
plt.xlabel('Obesity Category')  # Proper x-axis label
plt.ylabel('Count')  # Add y-axis label

# Rotate x-axis labels and adjust alignment
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    ha='right',
    fontsize=10
)


plt.figure(figsize=(20, 15))
sns.heatmap(X_processed.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


from sklearn.metrics import accuracy_score

# Get best model (replace X_val, y_val with your validation data)
best_model = max([dt, bag, rf, xgb_clf], key=lambda m: accuracy_score(y_val, m.predict(X_val)))

# Create submission
pd.DataFrame({'id': test_ids, 'NObeyesdad': le.inverse_transform(best_model.predict(test_processed))}).to_csv('submission.csv', index=False)

