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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Set plot style
sns.set_style('whitegrid')

# Load the data
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
except FileNotFoundError:
    print("Ensure train.csv, test.csv, and sample_submission.csv are in the same directory.")
    # Dummy data for demonstration if files are not found
    train_df = pd.DataFrame() 
    test_df = pd.DataFrame()

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

# Display the first few rows of the training data
train_df.head()


plt.figure(figsize=(8, 5))
sns.countplot(x='y', data=train_df, palette='viridis')
plt.title('Distribution of Term Deposit Subscriptions', fontsize=16)
plt.xlabel('Subscribed (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
# Calculating percentage
total = len(train_df)
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 5, f'{100 * height/total:.1f}%', ha="center")
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Distribution of Key Numerical Features', fontsize=16)

sns.histplot(train_df['age'], bins=30, ax=axes[0], color='skyblue', kde=True)
axes[0].set_title('Age Distribution')

sns.histplot(train_df['balance'], bins=30, ax=axes[1], color='salmon', kde=True)
axes[1].set_title('Balance Distribution')
axes[1].set_xlim(-2000, 10000) # Limiting for better visualization

sns.histplot(train_df['duration'], bins=30, ax=axes[2], color='lightgreen', kde=True)
axes[2].set_title('Call Duration Distribution')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


fig, axes = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Distribution of Key Categorical Features', fontsize=16)

sns.countplot(y='job', data=train_df, order=train_df['job'].value_counts().index, ax=axes[0], palette='plasma')
axes[0].set_title('Job Types')

sns.countplot(x='marital', data=train_df, order=train_df['marital'].value_counts().index, ax=axes[1], palette='magma')
axes[1].set_title('Marital Status')

sns.countplot(y='education', data=train_df, order=train_df['education'].value_counts().index, ax=axes[2], palette='cividis')
axes[2].set_title('Education Level')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Separate target variable
X = train_df.drop("y", axis=1)
y = train_df["y"]

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines
numerical_transformer = StandardScaler()
# handle_unknown='ignore' prevents errors if the test set has categories not seen in the training set
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model - LightGBM
lgbm = lgb.LGBMClassifier(random_state=42)

# Create the full model pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', lgbm)])

# Train the model on the full training data
print("Training model...")
model_pipeline.fit(X, y)
print("Model training complete!")


# Make predictions on the test set
# We use predict_proba to get the probability of the positive class (1) for ROC AUC
print("Making predictions on the test set...")
test_predictions_proba = model_pipeline.predict_proba(test_df)[:, 1]

# Create the submission file
submission_df = pd.DataFrame({'id': test_df['id'], 'y': test_predictions_proba})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
submission_df.head()

