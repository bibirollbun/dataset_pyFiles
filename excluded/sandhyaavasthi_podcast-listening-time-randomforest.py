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


#loading train and test data through pandas read function
train = pd.read_csv("/kaggle/input/predict-podcast-listening-time/train.csv")
test = pd.read_csv("/kaggle/input/predict-podcast-listening-time/test.csv")


train.info()


#top 10 rows in train data
train.head(10)


# listening minutes is target variable or predictor variable, remove it from train data

y = train.pop('Listening_Time_minutes')
y.head()


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(train)
plt.show()


# explore train data attributes
train.columns


X=train


X.head()


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Preprocess categorical columns
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))



#Impute Missing Values

from sklearn.impute import SimpleImputer

# Combine train and test to apply consistent preprocessing
combined = pd.concat([X, test], axis=0)

# Encode categorical features
for col in combined.select_dtypes(include='object').columns:
    combined[col] = combined[col].astype(str)
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Impute missing values with mean (you can choose median or others too)
imputer = SimpleImputer(strategy='mean')
combined_imputed = pd.DataFrame(imputer.fit_transform(combined), columns=combined.columns)

# Split back to train and test
X_clean = combined_imputed.iloc[:len(train)]
test_clean = combined_imputed.iloc[len(train):]


from sklearn.model_selection import train_test_split
# Train/Test split for evaluation
X_train, X_val, y_train, y_val = train_test_split(X_clean, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



# Evaluate
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")

# Predict on test
test_preds = model.predict(test_clean)




# code to create submission csv file
# Submission

# Separate features and target
TARGET = 'listening_time'
ID_COL = 'id'
submission = pd.DataFrame({
    ID_COL: test[ID_COL] if ID_COL in test.columns else range(len(test)),
    TARGET: test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file created.")


from xgboost import XGBRegressor
# Train XGBoost model
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    missing=np.nan,  # Let XGBoost handle missing values
    random_state=42,
    tree_method='hist',  # Fast and efficient
)
xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)


# Evaluate on validation set
val_preds = xgb_model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"XGBoost Validation RMSE: {rmse:.4f}")

# Predict on test set
test_preds = xgb_model.predict(test_clean)

# Prepare submission
submission = pd.DataFrame({
    ID_COL if ID_COL else 'id': test[ID_COL] if ID_COL else range(len(test)),
    TARGET: test_preds
})
submission.to_csv("xgb_submission.csv", index=False)
print("XGBoost submission file created.")

