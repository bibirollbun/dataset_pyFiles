# Install necessary packages
!uv pip install -q --system scikit-learn==1.5.2  # Quiet install of scikit-learn v1.5.2
!pip install autogluon                           # Install AutoGluon for AutoML
!pip install -U ipywidgets                       # Upgrade ipywidgets for notebook interactivity


import pandas as pd

# Load train and test datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Quick data overview
train.head()
test.head()

# Data structure and types
train.info()

# Unique values per column
train.nunique()


# Drop the 'id' column as it's not useful for training
train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

# Descriptive statistics
train.describe()

# Null value checks
train.isnull().sum()
round(train.isnull().sum() * 100 / len(train), 2)

# Preview rows with missing values
train[train.isna().any(axis=1)].head()


# Check for mismatched categories between train and test datasets
counter = 0
for i in test.select_dtypes(include=['object']).columns.tolist():
    if (len(list(set(train[i].unique().tolist()) ^ set(test[i].unique().tolist()))) != 0):
        print(i, 'need to be worked on')
        counter += 1
if counter == 0:
    print('No work needed')


# Convert 'Yes'/'No' to 1/0 for 'Stage_fear'
train['Stage_fear'] = train['Stage_fear'].replace({'Yes': 1, 'No': 0})
test['Stage_fear'] = test['Stage_fear'].replace({'Yes': 1, 'No': 0})

# Convert 'Yes'/'No' to 1/0 for 'Drained_after_socializing'
train['Drained_after_socializing'] = train['Drained_after_socializing'].replace({'Yes': 1, 'No': 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].replace({'Yes': 1, 'No': 0})


# Check for duplicates
train.duplicated().value_counts()

# Examine target label distribution
round(train['Personality'].value_counts() * 100 / len(train), 2)
train['Personality'].value_counts()


from autogluon.tabular import TabularDataset, TabularPredictor

# Define label
label = 'Personality'

# Fit AutoGluon predictor
predictor = TabularPredictor(label=label,
                             eval_metric='accuracy',
                             problem_type="binary"
                            ).fit(train,
                                  presets='medium_quality',
                                  time_limit=3600*9,  # Max 9 hours
                                  verbosity=3,
                                  ag_args_fit={'num_gpus': 1}
                                 )

# Summarize training results
results = predictor.fit_summary()


# Show leaderboard of model performances
predictor.leaderboard()

# Predict on test dataset
df = predictor.predict(test).to_frame(name=label)
df.head()


# Load sample submission file
sol = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# Assign predictions to submission DataFrame
sol[label] = df[label]

# Save to CSV for submission
sol.to_csv('./Autogluon_medium_quality_gpu.csv', index=False)

