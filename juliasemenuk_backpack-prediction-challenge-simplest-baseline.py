
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression



# Directory where files are stored
data_dir = Path('/kaggle/input/playground-series-s5e2')


# Load train – for training
# Test – we need to make predictions for it; it has no target
# Submission – a sample of what the prediction should look like
train = pd.read_csv(data_dir/'train.csv')
test = pd.read_csv(data_dir/'test.csv')
submission = pd.read_csv(data_dir/'sample_submission.csv')


# Let’s take a look at the train dataset.
# We see that there are missing values.
train.info()
train.sample(5)


# Let’s check the values in the Size column
train['Size'].value_counts(dropna=False)


# Size can be converted to ordinal.
# Let’s prepare a function to replace sizes with ordinal values.
# This will be added to the pipeline as a FunctionTransformer.
# Missing values will be filled using an imputer in the pipeline.
def map_size_to_number(cols):
    np.place(cols, cols=='Small', [0])
    np.place(cols, cols=='Medium', [1])
    np.place(cols, cols=='Large', [2])
    return cols


# Let’s create lists of features by type
num_features = ['Compartments', 'Weight Capacity (kg)']
cat_features = ['Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']
ord_features = ['Size']


# Let’s build pipelines for different data types
num_process = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler()),
])

cat_process = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore')),
])

ord_process = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ordinal', FunctionTransformer(map_size_to_number)),
])

# For each type of processing, we will specify which columns to use
preprocess = ColumnTransformer([
    ('num_process', num_process, num_features),
    ('cat_process', cat_process, cat_features),
    ('ord_process', ord_process, ord_features),
], remainder='drop')

# Final pipeline
pipe = Pipeline([
    ('preproces', preprocess),
    ('model', LinearRegression())
])


X = train[num_features + cat_features + ord_features]
y = train['Price']

pipe.fit(X, y)


# In the prediction sample, replace the Price column with the predicted values and save it.
# This prediction will be used to calculate the leaderboard metric.
submission['Price'] = pipe.predict(test)
submission.to_csv('submission.csv', index=False)

