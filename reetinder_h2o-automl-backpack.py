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


import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Initialize H2O
h2o.init()

# Load data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Convert categorical columns to categorical dtype
cat_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
for col in cat_cols:
    df_train[col] = df_train[col].astype("category")
    df_test[col] = df_test[col].astype("category")

# Drop "id" column
df_train.drop(columns=["id"], inplace=True, errors="ignore")
df_test.drop(columns=["id"], inplace=True, errors="ignore")

# Handle missing values
df_train.fillna(df_train.mean(numeric_only=True), inplace=True)  # Fill numeric missing values with median
df_train.fillna(df_train.mode().iloc[0], inplace=True)  # Fill categorical missing values with mode

df_test.fillna(df_test.mean(numeric_only=True), inplace=True)
df_test.fillna(df_test.mode().iloc[0], inplace=True)

# Convert Pandas DataFrame to H2OFrame
df_train_h2o = h2o.H2OFrame(df_train)
df_test_h2o = h2o.H2OFrame(df_test)

# Define target and features
target = "Price"
features = df_train.columns.difference([target]).tolist()

# Train H2O AutoML model
aml = H2OAutoML(max_models=20, seed=42)
aml.train(x=features, y=target, training_frame=df_train_h2o)

# Make predictions on test data
predictions = aml.predict(df_test_h2o)




predictions_df=pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
predictions1 = predictions.as_data_frame()
predictions_df['Price'] =predictions1
predictions_df.to_csv('x.csv',index=False)

