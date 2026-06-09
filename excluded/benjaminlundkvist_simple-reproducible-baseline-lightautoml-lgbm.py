!pip install -q lightautoml

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.metrics import mean_squared_error
import torch
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import os


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# Save IDs for submission
test_ids = test['id']

# Drop 'id' column
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# Shuffle train data for randomness
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

train.head()


# Target distribution
plt.figure(figsize=(8,4))
sns.histplot(train['BeatsPerMinute'], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Beats Per Minute (Target)")
plt.xlabel("Beats Per Minute")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
corr = train.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap of Features")
plt.show()


# Check for missing values
print("Missing values per column:\n", train.isnull().sum())

# Ensure correct dtypes
train = train.astype(float)
test = test.astype(float)


# Set random seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.set_num_threads(os.cpu_count())

# Define regression task
task = Task('reg')

# Initialize LightAutoML with LGBM
automl = TabularAutoML(
    task=task,
    timeout=2*3600,  # 2 hours
    cpu_limit=os.cpu_count(),
    reader_params={'n_jobs': os.cpu_count(), 'cv': 5, 'random_state': RANDOM_STATE, 'advanced_roles': True},
    general_params={"use_algos": [['lgb']]}
)

# Fit model
oof_predictions = automl.fit_predict(
    train,
    roles={'target': 'BeatsPerMinute'},
    verbose=3
)


# Evaluate model
rmse = mean_squared_error(train['BeatsPerMinute'], oof_predictions.data, squared=False)
print(f"Out-of-Fold RMSE Score: {rmse:.4f}")


# Predict BPM for test data
test_predictions = automl.predict(test).data[:, 0]

# Prepare submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'BeatsPerMinute': test_predictions
})

# save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
submission.head()

