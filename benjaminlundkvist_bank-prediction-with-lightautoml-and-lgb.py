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
from sklearn.metrics import roc_auc_score
import torch
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import os


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# Drop 'id' as it's not useful for modeling
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# Shuffle train data for randomness
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

# Preview training data
train.head()


sns.countplot(x='y', data=train)
plt.title("Distribution of Target Variable 'y'")
plt.show()


plt.hist(train['age'], bins=20, color='skyblue', edgecolor='black')
plt.title("Age Distribution of Clients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(12,5))
sns.countplot(x='job', data=train, order=train['job'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Distribution of Job Categories")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x='education', data=train, order=train['education'].value_counts().index)
plt.title("Distribution of Education Levels")
plt.show()


# Encode binary columns
binary_map = {'yes': 1, 'no': 0}
for col in ['default', 'housing', 'loan']:
    train[col] = train[col].map(binary_map)
    test[col] = test[col].map(binary_map)

# Encode month as numeric
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
train['month'] = train['month'].map(month_map)
test['month'] = test['month'].map(month_map)


# Set random seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.set_num_threads(os.cpu_count())

# Define task
task = Task('binary')

# Initialize LightAutoML with multiple algorithms
automl = TabularAutoML(
    task=task,
    timeout=2*3600,  # 2 hours
    cpu_limit=os.cpu_count(),
    reader_params={'n_jobs': os.cpu_count(), 'cv': 10, 'random_state': RANDOM_STATE, 'advanced_roles': True},
    general_params={"use_algos": [['lgb']]}
)


oof_predictions = automl.fit_predict(
    train,
    roles={'target': 'y'},
    verbose=3
)


roc_auc = roc_auc_score(train['y'], oof_predictions.data)
print(f"Out-of-Fold ROC AUC Score: {roc_auc:.4f}")


# Predict probabilities for the positive class
test_predictions = automl.predict(test).data[:, 0]

# Prepare submission dataframe
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission['y'] = test_predictions

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
submission.head()

