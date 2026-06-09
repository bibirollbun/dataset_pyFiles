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

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')



from sklearn.preprocessing import LabelEncoder

# Encode categorical features
soil_enc = LabelEncoder()
crop_enc = LabelEncoder()

train['Soil Type'] = soil_enc.fit_transform(train['Soil Type'])
test['Soil Type'] = soil_enc.transform(test['Soil Type'])

train['Crop Type'] = crop_enc.fit_transform(train['Crop Type'])
test['Crop Type'] = crop_enc.transform(test['Crop Type'])

# Select features (no 'id' or target column)
features = [
    'Temparature', 'Humidity', 'Moisture',
    'Soil Type', 'Crop Type',
    'Nitrogen', 'Potassium', 'Phosphorous'
]

X = train[features]
y = train['Fertilizer Name']
X_test = test[features]



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode target labels
target_enc = LabelEncoder()
y_enc = target_enc.fit_transform(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42)



import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
params = {
    'objective': 'multi:softprob',
    'num_class': len(target_enc.classes_),
    'eval_metric': 'mlogloss',
    'eta': 0.3,                # Faster learning
    'max_depth': 3,            # Shallower tree = faster
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42
}

# Train model with fewer rounds
model = xgb.train(
    params,
    dtrain,
    num_boost_round=50,        # Try 50 rounds first
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10,
    verbose_eval=10
)


import numpy as np
import pandas as pd

# Convert test features into DMatrix for prediction
dtest = xgb.DMatrix(X_test)

# Predict probabilities for each fertilizer class
pred_probs = model.predict(dtest)

# Choose the class with the highest predicted probability
best_pred_idx = np.argmax(pred_probs, axis=1)

# Convert the predicted class indices back to actual fertilizer names
fertilizer_pred = target_enc.inverse_transform(best_pred_idx)

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': fertilizer_pred  # ğŸ”� Must match Kaggle's required column name
})

# Save to CSV file for Kaggle submission
submission.to_csv('submission.csv', index=False)

# Preview
submission.head()





