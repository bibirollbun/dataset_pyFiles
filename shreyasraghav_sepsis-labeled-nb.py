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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from lightgbm import LGBMRegressor
import os


# Define paths to the training and testing data
training_data_path = "/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/"
testing_data_path = "/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/"


# Load key datasets
sepsis_label = pd.read_csv(os.path.join(training_data_path, 'SepsisLabel_train.csv'))
devices = pd.read_csv(os.path.join(training_data_path, 'devices_train.csv'))
drugs = pd.read_csv(os.path.join(training_data_path, 'drugsexposure_train.csv'))
demographics = pd.read_csv(os.path.join(training_data_path, 'person_demographics_episode_train.csv'))


# Step 1: Data Preprocessing and Merging
# Merge key files on `person_id` and time-based columns to create a unified training dataset
sepsis_label['measurement_datetime'] = pd.to_datetime(sepsis_label['measurement_datetime'])
devices['device_datetime_hourly'] = pd.to_datetime(devices['device_datetime_hourly'])
drugs['drug_datetime_hourly'] = pd.to_datetime(drugs['drug_datetime_hourly'])


# Merge demographics with sepsis labels
data = sepsis_label.merge(demographics, on="person_id", how="left")


# Merge devices and drugs data
merged_devices = devices.groupby('person_id').size().reset_index(name='device_count')
data = data.merge(merged_devices, on='person_id', how='left')


merged_drugs = drugs.groupby('person_id').size().reset_index(name='drug_count')
data = data.merge(merged_drugs, on='person_id', how='left')


# Fill missing values
data.fillna(0, inplace=True)


# Encode categorical variables (e.g., gender)
data = pd.get_dummies(data, columns=['gender'], drop_first=True)


# Step 2: Feature Selection and Target Variable
features = ['age_in_months', 'device_count', 'drug_count'] + [col for col in data.columns if 'gender_' in col]
target = 'SepsisLabel'

X = data[features]
y = data[target]
test = test_data.drop(['id'],axis=1)


# Step 3: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)


# Step 4: Train Model
model = LGBMRegressor(n_estimators=1000, random_state=42,force_col_wise=True )
model.fit(X_train, y_train)


test = pd.read_csv(r"/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/SepsisLabel_test.csv")


parameters = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 10,
    'num_leaves': 31,
    'min_child_samples': 25,
    'subsample': 0.6,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.1,
    'reg_lambda': 6,
    'verbosity': -1
}



from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error





# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)*100

# Cross-validation for LGBMRegressor
def cross_val_lgbm_mape(X, y, test, n_splits=8, **parameters):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_val = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_val = X[train_index], X[valid_index]
            y_train, y_val = y[train_index], y[valid_index]

        # Initialize and train the model
        model = LGBMRegressor(random_state=42, **parameters)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_val)
        score = mape(y_val, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean


average_mape, lgb_preds = cross_val_lgbm_mape(X, y, test, n_splits=12, **parameters)

print(f"Average MAPE across folds: {average_mape:.4f}")



# Step 5: Validate Model
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))


# Step 6: Prepare Test Data for Predictions
test_sepsis_label = pd.read_csv(os.path.join(testing_data_path, 'SepsisLabel_test.csv'))

test_sepsis_label['measurement_datetime'] = pd.to_datetime(test_sepsis_label['measurement_datetime'])
test_data = test_sepsis_label.merge(demographics, on="person_id", how="left")

test_data = test_data.merge(merged_devices, on='person_id', how='left')
test_data = test_data.merge(merged_drugs, on='person_id', how='left')
test_data.fillna(0, inplace=True)

test_data = pd.get_dummies(test_data, columns=['gender'], drop_first=True)

X_test = test_data[features]


# Step 7: Predict on Test Data
test_predictions = model.predict(X_test)
test_data['SepsisLabel'] = test_predictions


# Step 8: Create Submission File
test_data['person_id_datetime'] = test_data['person_id'].astype(str) + "_" + test_data['measurement_datetime'].astype(str)
submission = test_data[['person_id_datetime', 'SepsisLabel']]
submission.to_csv('SepsisLabel_submission9.csv', index=False)

print("Submission file created: SepsisLabel_submission8.csv")


!rm -rf /kaggle/working/*




