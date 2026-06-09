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


# ğŸ“¦ Step 1: Import libraries
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# ğŸ“� Step 2: Load datasets
train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')

# ğŸ�¯ Step 3: Extract target and drop from features
y = train['sale_price']
X = train.drop(columns=['sale_price', 'id'])
X_test = test.drop(columns=['id'])

# ğŸ§¼ Step 4: Label encode categorical columns
# Combine for consistent encoding
combined = pd.concat([X, X_test], axis=0)
cat_cols = combined.select_dtypes(include='object').columns

for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# Split back
X = combined.iloc[:len(X)]
X_test = combined.iloc[len(X):]

# Fill missing values
X = X.fillna(-1)
X_test = X_test.fillna(-1)

# âš™ï¸� Step 5: Define Quantile LightGBM model
def get_lgb_model(alpha):
    return lgb.LGBMRegressor(
        objective='quantile',
        alpha=alpha,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

# ğŸ�‹ï¸� Step 6: Train 5th and 95th percentile models
model_lower = get_lgb_model(0.05)
model_upper = get_lgb_model(0.95)

model_lower.fit(X, y)
model_upper.fit(X, y)

# ğŸ”® Step 7: Predict lower and upper bounds
pred_lower = model_lower.predict(X_test)
pred_upper = model_upper.predict(X_test)

# ğŸ“� Step 8: Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'pi_lower': pred_lower,
    'pi_upper': pred_upper
})

# ğŸ’¾ Step 9: Save submission file (required path for Kaggle)
submission.to_csv('/kaggle/working/submission.csv', index=False)

# ğŸ‘� Step 10: Show preview
submission.head()

