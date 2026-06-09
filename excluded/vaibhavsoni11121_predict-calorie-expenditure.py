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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df_train=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


df_train.head()


df_train.isna().sum()


df_train.describe()


df_train.columns


df_train.info()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# Drop 'id'
df_train = df_train.drop('id', axis=1)

# Define features and target
X = df_train.drop('Calories', axis=1)
y = df_train['Calories']

# Column types
categorical = ['Sex']
numerical = [col for col in X.columns if col not in categorical]


# Preprocessor pipeline (encoding + standardization)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(drop='first'), categorical)
])

# Model pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor())
])


pipeline.fit(X, y)



train_score = pipeline.score(X, y)
print(f"Train Score (R²): {train_score:.4f}")
    



ids = df_test['id']

# Transform test data (use your preprocessor from earlier)
X_test = df_test.drop('id', axis=1)

# Predict using your trained model (example: model could be RandomForestRegressor)
predictions = pipeline.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': ids,
    'Calories': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")


