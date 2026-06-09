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


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)


df_train=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
df_test=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
sample_submission=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")


df_train.head(2)


df_test.head(5)


df_train.info()


df_test.info()


df_train.isna().sum().sort_values()


df_test.isna().sum().sort_values()


len(df_train["ID"]),len(df_test["ID"])



(((df_train.isnull().sum())/(len(df_train["ID"])))*100).sort_values()



train_cat_columns=[]
for i in df_train.columns:
    if df_train[i].dtypes=='O':
        train_cat_columns.append(i)
train_cat_columns


test_cat_columns=[]
for i in df_test.columns:
    if df_test[i].dtypes=='O':
        test_cat_columns.append(i)
test_cat_columns


df_train[train_cat_columns].nunique()



df_test[test_cat_columns].nunique()



df_train1=df_train.copy()
df_test1=df_test.copy()


for column in df_train.columns:
    if df_train[column].dtype == 'object':
        # Fill with mode for object columns
        mode_value = df_train[column].mode()[0]  # Get the mode and take the first one if there are multiple
        df_train[column].fillna(mode_value, inplace=True)
    elif df_train[column].dtype in ['int64', 'float64']:
        # Fill with mean for numeric columns
        mean_value = df_train[column].mean()
        df_train[column].fillna(mean_value, inplace=True)


df_train.isnull().sum()



for column in df_test.columns:
    if df_test[column].dtype == 'object':
        # Fill with mode for object columns
        mode_value = df_test[column].mode()[0]  # Get the mode and take the first one if there are multiple
        df_test[column].fillna(mode_value, inplace=True)
    elif df_test[column].dtype in ['int64', 'float64']:
        # Fill with mean for numeric columns
        mean_value = df_test[column].mean()
        df_test[column].fillna(mean_value, inplace=True)


df_test.isnull().sum()



from lifelines import KaplanMeierFitter
kmf=KaplanMeierFitter()
kmf.fit(df_train["efs_time"],df_train["efs"])
y = kmf.survival_function_at_times(df_train["efs_time"]).values


y


y.shape,df_train.shape


import matplotlib.pyplot as plt
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.show()


from sklearn.preprocessing import LabelEncoder

# Loop through all columns in df_train
for column in df_train.columns:
    if df_train[column].dtype == 'object':  # Check if the column is categorical
        le = LabelEncoder()  # Create a LabelEncoder object
        df_train[column] = le.fit_transform(df_train[column].astype(str))  # Fit and transform the column


# Loop through all columns in df_test
for column in df_test.columns:
    if df_test[column].dtype == 'object':  # Check if the column is categorical
        le = LabelEncoder()  # Create a LabelEncoder object
        df_test[column] = le.fit_transform(df_test[column].astype(str))  # Fit and transform the column


df_train.head(3)


df_test.head(3)


# df_train.drop(columns="ID", axis=1, inplace=True)
# df_test.drop(columns="ID", axis=1, inplace=True)


# Define survival outcome variables
time_col = "efs_time"  # Survival time
event_col = "efs"       # Event indicator (1 = event, 0 = censored)


y_time = df_train1[time_col].values
y_event = df_train1[event_col].values


y_time,y_event


train_cat_columns==test_cat_columns


cat_columns=train_cat_columns


cat_columns


for col in cat_columns:
    df_train[col] = df_train[col].astype("category")
    df_test[col] = df_test[col].astype("category")


import xgboost as xgb

# Prepare training data (with labels)
X_train = df_train.drop(columns=["efs", "efs_time"])  
y_train = df_train["efs_time"]  

# Prepare test data (only features, no labels)
X_test = df_test  # No need to drop target variables as they don't exist

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train.values,enable_categorical=True)  # Labels required for training
dtest = xgb.DMatrix(X_test,enable_categorical=True)  # No labels in test data

# Define model parameters
params = {
    "objective": "survival:cox",  
    "eval_metric": "cox-nloglik",  
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# Train model
model_XGB = xgb.train(params=params, dtrain=dtrain, num_boost_round=500, evals=[(dtrain, "train")], early_stopping_rounds=10)


from lifelines import CoxPHFitter

# Prepare Data
df_train["efs_time"] = df_train["efs_time"].astype(float)
df_train["efs"] = df_train["efs"].astype(int)

# Fit Cox Model
cph = CoxPHFitter()
cph.fit(df_train, duration_col="efs_time", event_col="efs")

# Predict Risk Score
COX_PH_Risk_Score_Test = cph.predict_partial_hazard(df_test)
COX_PH_Risk_Score_Train = cph.predict_partial_hazard(df_train)




COX_PH_Risk_Score_Test


COX_PH_Risk_Score_Train[0:5]


import matplotlib.pyplot as plt

# Get survival function predictions
survival_curves = cph.predict_survival_function(df_test)

# Plot survival curves
plt.figure(figsize=(10, 6))
for i in range(min(10, len(df_test))):  # Plot only 10 patients for clarity
    plt.plot(survival_curves.index, survival_curves.iloc[:, i], label=f'Patient {i+1}')

plt.xlabel("Time (EFS Time)")
plt.ylabel("Survival Probability")
plt.title("Predicted Survival Curves")
plt.legend()
plt.show()



# from sksurv.ensemble import RandomSurvivalForest
# from sksurv.util import Surv

# # Convert data to structured format
# y_train_rsf = Surv.from_arrays(df_train["efs"].values.astype(bool), df_train["efs_time"].values)

# # Train RSF Model
# rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10)
# rsf.fit(df_train.drop(columns=["efs", "efs_time"]), y_train_rsf)

# # Predict Survival Function
# df_test["rsf_risk"] = rsf.predict(df_test)




import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


# Prepare y_true (actual survival labels)
y_true = df_train[["ID", "efs", "efs_time", "race_group"]].copy()

# Predict survival times on training data
dtrain_pred = xgb.DMatrix(X_train,enable_categorical=True)  # Convert training features to XGBoost DMatrix
predicted_survival_times = model_XGB.predict(dtrain_pred)  # Predict using trained AFT model

# Prepare y_pred (predictions)
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = predicted_survival_times  # Use predicted survival times

# Evaluate using custom metric function
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost AFT =", m)



# Prepare y_true (actual survival labels)
y_true = df_train[["ID", "efs", "efs_time", "race_group"]].copy()

# Predict survival times on training data

predicted_survival_times = cph.predict_partial_hazard(df_train)  # Predict using trained AFT model

# Prepare y_pred (predictions)
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = predicted_survival_times  # Use predicted survival times

# Evaluate using custom metric function
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for COX PH =", m)



# # Load the sample submission file
# sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")


# # Predict survival times using XGBoost AFT model
# predicted_survival_times_test = model_XGB.predict(dtest)

# # Assign predictions to submission file
# sub["prediction"] = predicted_survival_times_test

# # Save final submission file
# sub.to_csv("submission.csv", index=False)

# # Print summary
# print("Submission shape:", sub.shape)
# sub.head()



# Load the sample submission file
sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")


# Predict survival times using XGBoost AFT model
predicted_survival_times_test = cph.predict_partial_hazard(df_test)

# Assign predictions to submission file
sub["prediction"] = predicted_survival_times_test

# Save final submission file
sub.to_csv("submission.csv", index=False)

# Print summary
print("Submission shape:", sub.shape)
sub.head()





