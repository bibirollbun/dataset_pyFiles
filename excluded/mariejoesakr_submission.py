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

# Correct folder path
folder = '/kaggle/input/competition'

# Load train and test files
train = pd.read_csv(f'{folder}/train.csv')
test = pd.read_csv(f'{folder}/test.csv')

# Prepare features and target
X = train.drop(columns=['id', 'loss'])
y = train['loss']
X_test = test.drop(columns=['id'])

# Check the data
train.head()




# Step 1: Check the columns inside train.csv
import pandas as pd

folder = '/kaggle/input/competition'

train = pd.read_csv(f'{folder}/train.csv')
test = pd.read_csv(f'{folder}/test.csv')

# Print the columns
print(train.columns)



import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Define the XGBoost regressor with better parameters
model = xgb.XGBRegressor(
    n_estimators=200,        # more trees than RF for better performance
    max_depth=6,             # not too deep, avoids overfitting
    learning_rate=0.1,       # standard learning rate
    subsample=0.8,           # sample 80% of rows
    colsample_bytree=0.8,    # sample 80% of features
    n_jobs=-1,
    random_state=42
)

# Evaluate model using 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"ðŸ“‰ Cross-validated RMSE: {-scores.mean():.5f}")

# Train the model on full data
model.fit(X, y)
print("âœ… XGBoost model trained.")




# Predict on test set
predictions = model.predict(X_test)

# Reverse the log transformation
import numpy as np
predictions = np.expm1(predictions)  # Applies inverse of log1p

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'loss': predictions
})

# Save submission
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("ðŸš€ Submission file created with log-transformed target reversed!")



print("âœ… Model trained!")  # Simple success message


# Prepare test features
X_test = test.drop(columns=['id'])

# Predict using the trained model
predictions = model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'loss': predictions
})

# Save submission
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("âœ… Submission file created!")

