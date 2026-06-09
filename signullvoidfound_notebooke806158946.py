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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# File paths
train_path = '../input/recruitment-task-for-gdsc-ml/MiNDAT.csv'
test_path = '../input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv'

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Specify target and ID column
target_col = 'CORRUCYSTIC_DENSITY'  # Replace with actual target column
id_col = 'LOCAL_IDENTIFIER'          # Replace with actual ID column in test set

# Drop rows with missing target values
train_df = train_df.dropna(subset=[target_col])

# Split features and target
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing for numerical features: KNN imputation + scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features: most frequent imputation + one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Full pipeline with Random Forest Regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Validation performance
y_val_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print("Validation MSE:", mse)
print("Validation R2 Score:", r2)

# Predict on test data
test_predictions = best_model.predict(test_df)

# Prepare submission
submission = pd.DataFrame({
    id_col: test_df[id_col],
    'CORRUCYSTIC_DENSITY': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file created as 'submission.csv'.")



specimen = pd.read_csv("../input/recruitment-task-for-gdsc-ml/SPECIMEN.csv")

target_col = "CORRUCYSTIC_DENSITY"  
specimen[target_col] = test_predictions
specimen.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")


