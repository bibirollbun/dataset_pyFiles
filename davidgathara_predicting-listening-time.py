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


"""Your Goal: Your task it to predict listening time of a podcast episode."""
# I will be using xgboost since we are comparing Listening_Time_minutes and Episode_Length_minutes


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load data
train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# Handle missing data


#Finding the missing data in percentages for training data
Missing_data = train_data.isna().mean()*100
print(Missing_data)


#Finding the missing data in percentage for test data
Missing_data = test_data.isna().mean()*100
print(Missing_data)


# Fill columns with <5% missing
Be_Filled = ['Number_of_Ads']
train_data[Be_Filled] = train_data[Be_Filled].fillna(train_data[Be_Filled].mean())# For training data
test_data[Be_Filled] = test_data[Be_Filled].fillna(test_data[Be_Filled].mean()) # For test data


# Fill columns with 5-20% missing
Columns_to_impute = ['Episode_Length_minutes']
imputer = SimpleImputer(strategy="mean")
train_data[Columns_to_impute] = imputer.fit_transform(train_data[Columns_to_impute])#For training  data
test_data[Columns_to_impute] = imputer.transform(test_data[Columns_to_impute])#For test data


# Encode categorical data


# Initialize encoders for each categorical column
encoders = {
    'Podcast_Name': LabelEncoder(),
    'Episode_Title': LabelEncoder(),
    'Genre': LabelEncoder(),
    'Publication_Time': LabelEncoder(),
    'Publication_Day': LabelEncoder()
}


# Fit and transform on training data
for col, encoder in encoders.items():
    train_data[f"{col}_Encoded"] = encoder.fit_transform(train_data[col])


# Transform test data using the same encoders
for col, encoder in encoders.items():
    # Handle unseen labels by assigning a default value (e.g., -1)
    test_data[f"{col}_Encoded"] = test_data[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
    # Update encoder classes to include 'Unknown' if not already present
    if 'Unknown' not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, 'Unknown')
    test_data[f"{col}_Encoded"] = encoder.transform(test_data[col])


# One-hot encode Episode_Sentiment
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
onehot_train = onehot_encoder.fit_transform(train_data[["Episode_Sentiment"]])
onehot_test = onehot_encoder.transform(test_data[["Episode_Sentiment"]])


# Create DataFrames for one-hot encoded features
onehot_cols = onehot_encoder.get_feature_names_out(["Episode_Sentiment"])
onehot_train_df = pd.DataFrame(onehot_train, columns=onehot_cols, index=train_data.index)
onehot_test_df = pd.DataFrame(onehot_test, columns=onehot_cols, index=test_data.index)


# Drop original categorical columns
columns_to_drop = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Time', 'Publication_Day', 'Episode_Sentiment']


final_train_data = pd.concat([train_data.drop(columns=columns_to_drop), onehot_train_df], axis=1)
final_test_data = pd.concat([test_data.drop(columns=columns_to_drop), onehot_test_df], axis=1)


# Prepare features and target
X = final_train_data.drop("Listening_Time_minutes", axis=1)
y = final_train_data["Listening_Time_minutes"]


# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)


# Validate
y_pred = model.predict(X_val)
print(f"MAE: {mean_absolute_error(y_val, y_pred):.2f}")
print(f"R²: {r2_score(y_val, y_pred):.2f}")


# Predict on test data
test_pred = model.predict(final_test_data)


# Create submission file
submission = pd.DataFrame({"id": test_data["id"], "Listening_Time_minutes": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission shape:", submission.shape)

