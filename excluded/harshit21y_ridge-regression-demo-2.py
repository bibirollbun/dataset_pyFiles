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
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
print('done')


# Drop columns with all NaNs
train = train.dropna(axis=1, how='all')

# Drop rows with any NaNs
train = train.dropna(axis=0, how='any')

# Drop same columns in test as in train
test = test[train.drop(columns=["label"]).columns]


X_all = train.drop(columns=["label"])
y = train["label"]
# Compute correlation of each column with the target (faster than .corr())
correlations = X_all.corrwith(y).abs().sort_values(ascending=False)

# Select top 30 features
top_features = correlations.head(500).index

# Final training and test sets
X = train[top_features]
y = train["label"]
X_test = test[top_features]

print('done')


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)
)

model.fit(X_train, y_train)


# Evaluate
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
best_alpha = model.named_steps['ridgecv'].alpha_
print(f"âœ… Validation MSE: {mse:.4f}")
print(f"ğŸ”� Best Alpha: {best_alpha}")




from scipy.stats import pearsonr

# y_val: actual target values from validation set
# y_pred: predicted target values from Ridge model

pearson_corr, _ = pearsonr(y_val, y_pred)
print(f"ğŸ“ˆ Pearson Correlation (Predictions vs Actual): {pearson_corr:.4f}")



submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
submission["prediction"] = model.predict(X_test)
submission.to_csv("submission.csv", index=False)
print("ğŸ“� Submission file saved as 'submission.csv'")




