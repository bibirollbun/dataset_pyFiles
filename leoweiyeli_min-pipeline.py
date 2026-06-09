# 1. Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# 2. Load datasets using Parquet format
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


# 3. Select anonymized feature columns (X_1 to X_780)
feature_cols = [col for col in train.columns if col.startswith('X')]



# 4. Prepare training and testing datasets
X_train = train[feature_cols]
y_train = train['label']
X_test = test[feature_cols]


# 5. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# 6. Make predictions
preds = model.predict(X_test)


# 7. Prepare submission file
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission['predictions'] = preds
sample_submission.to_csv('submission.csv', index=False)

