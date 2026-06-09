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


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# CatBoost 
from catboost import CatBoostClassifier, Pool

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Example: Adjust file paths to your dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Preview
train.head()


train.columns


train.isnull().sum()


test.isnull().sum()


# Numerical columns to impute with MEDIAN
numerical_cols_to_impute = [
    'Time_spent_Alone', 
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

# Categorical columns to impute with MODE (Yes/No type)
categorical_cols_to_impute = [
    'Stage_fear', 
    'Drained_after_socializing'
]



# Impute numerical features with median (robust to outliers)
for col in numerical_cols_to_impute:
    median_value = train[col].median()
    train[col].fillna(median_value, inplace=True)
    test[col].fillna(median_value, inplace=True)



# Impute categorical features with mode (most frequent value)
for col in categorical_cols_to_impute:
    mode_value = train[col].mode()[0]
    train[col].fillna(mode_value, inplace=True)
    test[col].fillna(mode_value, inplace=True)


# Define target variable
target = 'Personality'

# Features (drop target and id)
X = train.drop(columns=[target, 'id'])
y = train[target]

# Test features (drop id)
X_test = test.drop(columns=['id'])


# Split train data into training and validation sets (80% train, 20% validation)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Identify columns with object (string) data type as categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print("Categorical features:", categorical_features)


# Initialize CatBoost Classifier
model = CatBoostClassifier(
    iterations=1000,           # Number of trees
    learning_rate=0.05,        # Step size shrinkage
    depth=6,                    # Tree depth
    eval_metric='Accuracy',     # Use accuracy for evaluation
    cat_features=categorical_features, # Specify categorical columns
    verbose=100,                # Print progress every 100 iterations
    random_seed=42,             # For reproducibility
    early_stopping_rounds=50    # Stop if no improvement after 50 rounds
)

# Train the model
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))



# Predict on validation data
y_pred = model.predict(X_valid)


# Calculate accuracy
acc = accuracy_score(y_valid, y_pred)
print("Validation Accuracy:", round(acc, 4))


# Create confusion matrix
cm = confusion_matrix(y_valid, y_pred, labels=['Introvert', 'Extrovert'])

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Introvert', 'Extrovert'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


# Print precision, recall, F1-score
print("Classification Report:\n")
print(classification_report(y_valid, y_pred))


# Predict on test set
test_pred = model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_pred
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


# Get feature importances
feature_importances = model.get_feature_importance()
features = X.columns

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance (CatBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()



import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def objective(trial):
    # Suggest hyperparameters to tune
    param = {
        'iterations': trial.suggest_int('iterations', 200, 1000),           # Number of boosting rounds
        'depth': trial.suggest_int('depth', 4, 10),                        # Tree depth
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),  # Step size shrinkage
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),          # L2 regularization
        'random_seed': 42,
        'eval_metric': 'Accuracy',
        'early_stopping_rounds': 50,
        'verbose': 0,
    }

    # Convert categorical feature names to indices (CatBoost requirement)
    cat_features_idx = [X_train.columns.get_loc(c) for c in categorical_features]

    # Use Stratified K-Fold to maintain class distribution across folds
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accuracies = []

    # Cross-validation loop
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Initialize CatBoost with current trial parameters
        model = CatBoostClassifier(**param, cat_features=cat_features_idx)
        
        # Train model with early stopping
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0, use_best_model=True)
        
        # Predict and evaluate accuracy
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        accuracies.append(acc)

    # Return average accuracy across folds (maximize this)
    return np.mean(accuracies)



import warnings
warnings.filterwarnings('ignore')


# Create a study object that maximizes accuracy
study = optuna.create_study(direction='maximize')

# Run the optimization over a set number of trials (e.g., 30)
study.optimize(objective, n_trials=30)


print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value:.4f}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


# Update best params with fixed parameters
best_params = trial.params
best_params.update({
    'random_seed': 42,
    'eval_metric': 'Accuracy',
    'early_stopping_rounds': 50,
    'verbose': 100,
})

# Convert categorical features to indices again
cat_features_idx = [X_train.columns.get_loc(c) for c in categorical_features]

# Initialize and train the final model
final_model = CatBoostClassifier(**best_params, cat_features=cat_features_idx)
final_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))



y_pred = final_model.predict(X_valid)
print("Validation Accuracy (final model):", accuracy_score(y_valid, y_pred))

cm = confusion_matrix(y_valid, y_pred, labels=['Introvert', 'Extrovert'])
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Introvert', 'Extrovert']).plot()
plt.show()

print(classification_report(y_valid, y_pred))



test_pred = final_model.predict(X_test)
submission = pd.DataFrame({'id': test['id'], 'Personality': test_pred})
submission.to_csv('submission.csv', index=False)
print("Final submission saved!")


