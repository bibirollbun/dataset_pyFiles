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

train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


print("Shape of train_df:", train_df.shape)
print("\nData types of train_df:")
train_df.info()
print("\nDescriptive statistics of train_df:")
train_df.describe()


import matplotlib.pyplot as plt
import seaborn as sns

# List of numerical features to visualize
numerical_features = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'sleep_hours_per_day',
    'bmi',
    'systolic_bp',
    'cholesterol_total'
]
for feature in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Create a figure with two subplots

    # Histogram
    sns.histplot(train_df[feature], kde=True, ax=axes[0])
    axes[0].set_title(f'Distribution of {feature.replace("_", " ").title()}')
    axes[0].set_xlabel(feature.replace("_", " ").title())
    axes[0].set_ylabel('Frequency')

    # Box plot
    sns.boxplot(y=train_df[feature], ax=axes[1])
    axes[1].set_title(f'Box Plot of {feature.replace("_", " ").title()}')
    axes[1].set_ylabel(feature.replace("_", " ").title())

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
excluded_cols = ['id']
numerical_features_for_corr = [col for col in numerical_cols if col not in excluded_cols]
correlation_matrix = train_df[numerical_features_for_corr].corr()
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features', fontsize=16)

# Display the plot
plt.show()



import pandas as pd

test_file_path = "/kaggle/input/playground-series-s5e12/test.csv"
test_df = pd.read_csv(test_file_path)

print("First 5 rows of test_df:")
test_df.head()


import pandas as pd

# Identify categorical columns
categorical_cols = train_df.select_dtypes(include='object').columns.tolist()

# Apply One-Hot Encoding to train_df
train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=False)

# Apply One-Hot Encoding to test_df
test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=False)

print("Shape of train_encoded:", train_encoded.shape)
print("Shape of test_encoded:", test_encoded.shape)
print("First 5 rows of train_encoded:")
print(train_encoded.head())
print("\nFirst 5 rows of test_encoded:")
print(test_encoded.head())


import numpy as np

train_cols = set(train_encoded.columns)
test_cols = set(test_encoded.columns)
missing_in_test = list(train_cols - test_cols)
missing_in_train = list(test_cols - train_cols)

# Add missing columns to test_encoded and fill with zeros
for col in missing_in_test:
    test_encoded[col] = 0

for col in missing_in_train:
    test_encoded = test_encoded.drop(columns=[col])

# Ensure the order of columns is the same
test_encoded = test_encoded[train_encoded.columns.drop('diagnosed_diabetes')]

# Define features (X) and target (y) for the training set
X_train = train_encoded.drop(columns=['id', 'diagnosed_diabetes'])
y_train = train_encoded['diagnosed_diabetes']

X_test = test_encoded.drop(columns=['id'])


print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Columns in X_train:\n", X_train.columns.tolist())
print("Columns in X_test:\n", X_test.columns.tolist())
print("Are columns of X_train and X_test identical:", (X_train.columns == X_test.columns).all())



!pip install optuna


import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# Define the objective function for Optuna
def objective(trial):
    # 2a. Suggest values for hyperparameters
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Instantiate an LGBMClassifier with the suggested hyperparameters
    model = LGBMClassifier(**param)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Calculate the cross-validation roc_auc_score
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')

    # Return the mean of the cross-validation scores
    return np.mean(scores)

# Create an Optuna study object
print("Creating Optuna study...")
study = optuna.create_study(direction='maximize')

# Run the optimization
print("Running Optuna optimization...")
study.optimize(objective, n_trials=50, show_progress_bar=True)
print("Optuna optimization complete.")

# Print best trial results
print("\nBest trial:")
print(f"  Value: {study.best_value:.4f}")
print("  Params: ")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")



study.best_params

