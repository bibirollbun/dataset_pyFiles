# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import libraries
import ydf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# create test and train dataframe variable
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


# drop the id column from the train and test df
df_test = df_test.drop('id', axis=1)
df_train = df_train.drop('id', axis=1)

# the label field is categorical.  Need to cast as int
df_train['diagnosed_diabetes'] = df_train['diagnosed_diabetes'].astype(int)

df_train.head()


# encode the dataframes
encoded_df_test = pd.get_dummies(df_test, dtype= int, drop_first=True)
encoded_df_train = pd.get_dummies(df_train, dtype= int, drop_first=True)


# Identify categorical columns
#categorical_cols = df_train.select_dtypes(include='object').columns

# Apply one-hot encoding to categorical columns
#df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)

# Calculate the correlation matrix for df_train_encoded
correlation_matrix = encoded_df_train.corr()

# Get correlations with 'diagnosed_diabetes'
diabetes_correlations = correlation_matrix['diagnosed_diabetes'].sort_values(ascending=False)

# Exclude self-correlation if it exists after encoding
if 'diagnosed_diabetes' in diabetes_correlations.index:
    diabetes_correlations = diabetes_correlations.drop('diagnosed_diabetes')

# Plotting the bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x=diabetes_correlations.values, y=diabetes_correlations.index, hue=diabetes_correlations.index, palette='viridis', legend=False)
plt.title('Correlation with Diagnosed Diabetes in df_train (Encoded)')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Sort by absolute correlation to find those closest to zero
diabetes_correlations_abs_sorted = diabetes_correlations.reindex(diabetes_correlations.abs().sort_values().index)

# Get the top 3 closest to zero
closest_to_zero = diabetes_correlations_abs_sorted.head(10)
display(closest_to_zero)


print("Encoded df_test info pre drop: ", encoded_df_test.info())
cols_to_drop_attempt = ['ethnicity_Black', 'employment_status_Student', 'employment_status_Unemployed']

# Filter the list to only include columns that actually exist in the DataFrame
existing_cols_train = [col for col in cols_to_drop_attempt if col in encoded_df_train.columns]
existing_cols_test = [col for col in cols_to_drop_attempt if col in encoded_df_test.columns]

if existing_cols_train: # Only drop if there are columns to drop
    encoded_df_train = encoded_df_train.drop(existing_cols_train, axis=1)
if existing_cols_test: # Only drop if there are columns to drop
    encoded_df_test = encoded_df_test.drop(existing_cols_test, axis=1)

print("Encoded df_test info: ", encoded_df_test.info())
print("Encoded df_train info: ", encoded_df_train.info())


# split the encoded training df into two sets
# Randomly split the dataset into a training (70%) and testing (30%) dataset
encoded_df_train = encoded_df_train.sample(frac=1)
split_idx = len(encoded_df_train) * 7 // 10
train_ds = encoded_df_train.iloc[:split_idx]
test_ds = encoded_df_train.iloc[split_idx:]


'''tuner = ydf.RandomSearchTuner(num_trials=10)
tuner.choice("validation_ratio", [0.1, 0.2, 0.3])
tuner.choice("max_depth", [5, 6, 7])
tuner.choice("subsample", [0.8, 0.9, 1.0])
tuner.choice("shrinkage", [0.1, 0.2, 0.3])
tuner.choice("num_trees", [100, 200, 300])'''


learner = ydf.GradientBoostedTreesLearner(num_trees=300,
                                          max_depth=5,
                                          validation_ratio=0.2,
                                          subsample=1.0,
                                          shrinkage=0.3,
                                          early_stopping="MIN_LOSS_FINAL",
                                          #tuner = tuner,
                                          label='diagnosed_diabetes')

model = learner.train(train_ds)
#model = ydf.RandomForestLearner(num_trees=100, max_depth=5, label='diagnosed_diabetes').train(train_ds)


model.describe()


evaluate = model.evaluate(test_ds)
evaluate



# cross validate the model
#cross_validate = learner.cross_validation(encoded_df_train, folds=10)
#cross_validate


prediction = model.predict(encoded_df_test)
prediction


submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission['diagnosed_diabetes'] = prediction
submission.to_csv('submission.csv', index=False)

