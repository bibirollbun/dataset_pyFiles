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


# Data Handling
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Model Selection
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LinearRegression, Lasso

# Evaluation
from sklearn.metrics import mean_squared_error



train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')



train_df.shape


test_df.shape


train_df.head(5)


# Feature engineering: Add title length and word count
train_df['Episode_Title_Length'] = train_df['Episode_Title'].apply(lambda x: len(str(x)))
train_df['Episode_Title_WordCount'] = train_df['Episode_Title'].apply(lambda x: len(str(x).split()))



# Prepare features and target
X = train_df.drop(columns=['id', 'Listening_Time_minutes', 'Episode_Title'])
y = train_df['Listening_Time_minutes']


# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()


#Define Preprocessing and Model Pipeline


# Numerical preprocessing pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

# Categorical preprocessing pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


# Combine preprocessor
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Full modeling pipeline with Lasso
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=0.1))
])


import time

# Start timing before training
start = time.time()


# Model Training

# Fit the model

model_pipeline.fit(X, y)

# Predict and calculate RMSE on validation set
y_train_pred= model_pipeline.predict(X)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
print(f"âœ… Train RMSE: {train_rmse:.4f}")



# Your code here
end = time.time()

print(f"Computation Time: {end - start:.2f} seconds")
minutes = int((end - start) // 60)
seconds = (end - start) % 60
print(f"Total Computation Time: {minutes} min {seconds:.2f} sec")



# Show predicted vs actual values for training data
output_df = X.copy()
output_df['Actual'] = y
output_df['Predicted'] = y_train_pred

# Display first few rows
print(output_df[['Actual', 'Predicted']].head(10))



# Preprocess Test Data


# Feature engineering (same as train)
test_df['Episode_Title_Length'] =test_df['Episode_Title'].apply(lambda x: len(str(x)))
test_df['Episode_Title_WordCount'] =test_df['Episode_Title'].apply(lambda x: len(str(x).split()))

# Prepare final test features
X_test_final = test_df.drop(columns=['id', 'Episode_Title'])

# Predict listening time
y_test_pred = model_pipeline.predict(X_test_final)



y_test_pred



# Create Submission File
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': y_test_pred
})
submission.to_csv("submission.csv", index=False)
print("\nğŸ“� Submission file 'submission.csv' created successfully!")


