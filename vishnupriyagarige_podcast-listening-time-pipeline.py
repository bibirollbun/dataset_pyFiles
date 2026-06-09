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
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


!pip install h2o


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")
original_data = pd.read_csv(r"/kaggle/input/podcast-data/podcast_dataset.csv")
sample_submission = pd.read_csv(r"/kaggle/input/playground-series-s5e4/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("original_data shape :",original_data.shape)
print("sample_submission shape :",sample_submission.shape)


train_data.head()


original_data.dropna(subset=['Listening_Time_minutes'],inplace=True)
original_data.shape


test_data.head()


train_data = train_data.drop("id", axis=1)
train_data = pd.concat([train_data, original_data], ignore_index=True)
print("shape of the data :",train_data.shape)


set(train_data.columns) - set(test_data.columns)


#train_data = train_data.drop('id', axis = 1)
num_cols = list(train_data.select_dtypes(exclude=['object']).columns.difference(['Listening_Time_minutes']))
cat_cols = list(train_data.select_dtypes(include=['object']).columns)

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns.difference(['id']))
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)


# Fill missing values
train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
train_data[train_data.select_dtypes(include=['object', 'category']).columns] = train_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))

# Fill missing values
test_data[test_data.select_dtypes(include=['number']).columns] = test_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
test_data[test_data.select_dtypes(include=['object', 'category']).columns] = test_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))


def remove_outliers(df, method='iqr', threshold=1.5):
    """
    Removes outliers from all numerical columns using the specified method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        method (str): 'iqr' for Interquartile Range or 'zscore' for Z-score method
        threshold (float): The threshold for defining an outlier (default 1.5 for IQR)
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['number']).columns  # Select only numeric columns

    if method == 'iqr':
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    elif method == 'zscore':
        from scipy.stats import zscore
        df_clean = df_clean[(df_clean[numeric_cols].apply(zscore).abs() < threshold).all(axis=1)]

    return df_clean

# Usage Example
train_data = remove_outliers(train_data, method='iqr')  # Remove outliers from training data


from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
label_encoders = {col: LabelEncoder() for col in cat_cols}

# Apply LabelEncoder to each categorical column
for col in cat_cols:
    train_data[col] = label_encoders[col].fit_transform(train_data[col])
    test_data[col] = label_encoders[col].transform(test_data[col])


# Episode-Based Features
train_data["Ads_per_minute"] = train_data["Number_of_Ads"] / train_data["Episode_Length_minutes"]
test_data["Ads_per_minute"] = test_data["Number_of_Ads"] / test_data["Episode_Length_minutes"]

train_data["Guest_Impact"] = train_data["Guest_Popularity_percentage"] * train_data["Episode_Sentiment"]
test_data["Guest_Impact"] = test_data["Guest_Popularity_percentage"] * test_data["Episode_Sentiment"]

train_data["Host_Impact"] = train_data["Host_Popularity_percentage"] * train_data["Episode_Sentiment"]
test_data["Host_Impact"] = test_data["Host_Popularity_percentage"] * test_data["Episode_Sentiment"]

# Time-Based Features
train_data["Is_Weekend"] = train_data["Publication_Day"].isin(["Saturday", "Sunday"]).astype(int)
test_data["Is_Weekend"] = test_data["Publication_Day"].isin(["Saturday", "Sunday"]).astype(int)

#train_data["Episode_Title_Length"] = train_data["Episode_Title"].apply(len)
#test_data["Episode_Title_Length"] = test_data["Episode_Title"].apply(len)

# Aggregated Features
genre_avg_length = train_data.groupby("Genre")["Episode_Length_minutes"].mean().to_dict()
train_data["Avg_Episode_Length_for_Genre"] = train_data["Genre"].map(genre_avg_length)
test_data["Avg_Episode_Length_for_Genre"] = test_data["Genre"].map(genre_avg_length)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols_test] = scaler.transform(test_data[num_cols_test])


X = train_data.drop(['Listening_Time_minutes'], axis=1)
y = train_data['Listening_Time_minutes']
test = test_data.drop('id', axis=1)


import h2o
from h2o.automl import H2OAutoML

# Initialize H2O cluster
h2o.init()

# Convert training data to H2OFrame
train_h2o = h2o.H2OFrame(pd.concat([X, y], axis=1))

# Define the feature columns and target column
x = X.columns.tolist()  # Feature names
y_col = 'Listening_Time_minutes'  # Target column name

# Initialize H2O AutoML
aml = H2OAutoML(
    max_runtime_secs=600,
    seed=42,
    max_models=20,
    sort_metric="RMSE",
    include_algos=["GBM", "XGBoost", "GLM"],
    nfolds=5,
    stopping_metric="RMSE",
    stopping_rounds=3,
    stopping_tolerance=1e-4
)

# Train AutoML on the training set
aml.train(x=x, y=y_col, training_frame=train_h2o)

# Prepare the test data (make sure it has the same features as the training data)
X_test = test  # Assuming the test data does not include 'Listening_Time_minutes'
test_h2o = h2o.H2OFrame(X_test)

# Make predictions on the separate test set using the trained model
predictions = aml.leader.predict(test_h2o)

# Convert predictions from H2OFrame to Pandas DataFrame
preds_df = h2o.as_list(predictions)

# Shut down H2O cluster
h2o.shutdown()


submission = pd.DataFrame({'id': sample_submission.id, 'Listening_Time_minutes': preds_df['predict']})
print(submission.head())
submission.to_csv('submission.csv', index=False)

