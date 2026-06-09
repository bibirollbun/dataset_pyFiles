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


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import power_transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import classification_report
from lightgbm import LGBMRegressor

import warnings
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


# Load dataset

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# Check everything

for df in [train, test, submission]:
    print(df.info())


print(train.shape, test.shape, submission.shape)


train.describe()


# Check Missing Data

for df in [train, test, submission]:
    print(df.isnull().sum())


train.head()


# Check Distribution
fig, axes = plt.subplots(4, 2, figsize=(15, 15))

for i, col in enumerate(train.select_dtypes(include=["number"]).columns):
    ax = axes.flat[i]
    sns.kdeplot(data=train, x=col, ax=ax)  # KDE plot (no histogram)
    ax.set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


# Check Outliers
fig, axes = plt.subplots(4, 2, figsize=(15, 15))

for i, col in enumerate(train.select_dtypes(include=["number"]).columns):
    ax = axes.flat[i]
    sns.boxplot(data=train, x=col, ax=ax)  # KDE plot (no histogram)
    ax.set_title(f'Outlier of {col}')

plt.tight_layout()
plt.show() # Show each plot one at a time


# Check correlation
# Drop id for train, test and keep for submission
train = train.drop(columns=["id"], axis=1)
test = test.drop(columns=["id"], axis=1)

# Sex is categorical variable, need to convert numerical
# Manual One-Hot Encoding Female to 0 Male to 1
train = pd.get_dummies(train, columns=["Sex"], drop_first=True)
train = train.rename(columns={"Sex_male": "Gender"})
train = train.astype({"Gender": int})

test = pd.get_dummies(test, columns=["Sex"], drop_first=True)
test = test.rename(columns={"Sex_male": "Gender"})
test = test.astype({"Gender": int})

submission = pd.get_dummies(submission, columns=["Sex"], drop_first=True)
submission = submission.rename(columns={"Sex_male": "Gender"})
submission = submission.astype({"Gender": int})

plt.figure(figsize=(15,15))
c= train.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
plt.show()


# IQR
def remove_outlier(data, column):
    # Calculate the upper and lower limits
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.0*IQR
    upper = Q3 + 1.0*IQR
    
    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(data[column] >= upper)[0]
    lower_array = np.where(data[column] <= lower)[0]

    # Removing the outliers
    data.drop(index=upper_array, inplace=True)
    data.drop(index=lower_array, inplace=True)
    data = data.reset_index(drop=True)

    return data

# Check Outliers


train = remove_outlier(train, "Calories")

print("New Shape: ", train.shape, "Calories")

sns.boxplot(data=train, x="Calories")  # KDE plot (no histogram)
plt.title(f'Outlier of Calories')
plt.figure(figsize=(15,15))
plt.tight_layout()
plt.show()


# Since high correlation, I believe we can come out with new features.
# Duration has the highest correlation with calories which is 0.96, 
# I am using Duration as the main factor to generate body_temp per Duration & heart_rate per Duration

# Height and Weight correlates with each other. So I think it is possible to combine them become BMI
# BMI Formula = Weight (kgs) / (Height (m) )^2

def add_features(df):
    df["heart_rate/duration"] = (df["Heart_Rate"] / df["Duration"])
    df["body_temp/duration"] = (df["Body_Temp"] / df["Duration"])
    df["BMI"] = (df["Weight"] / (df["Height"] / 100) ** 2)

    return df

train=add_features(train)
test=add_features(test)
submission=add_features(submission)


# Regression Model (LightGBM)
X = train.drop(columns=["Calories"], axis=1)
y = train[["Calories"]]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search_params = {
        "boosting_type": ["gbdt"],
        "num_leaves": [31, 64, 128],
        "max_depth": [-1, 8, 16],
        "feature_fraction": [0.8],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [3000],
        "min_child_samples": [20],
        "reg_alpha": [0, 0.1, 0.3],
        "reg_lambda": [0, 0.1, 0.3],
        "random_state": [42],
        "verbose": [0]
}
# Initialize RandomForestClassifier
lgbm_model = LGBMRegressor()

RSCV_model = RandomizedSearchCV(lgbm_model, search_params)

# Fit the classifier to the training data
RSCV_model.fit(X_train, np.ravel(y_train), eval_set=(X_test, np.ravel(y_test)), callbacks=[lgb.early_stopping(stopping_rounds=50)])

best_params = RSCV_model.best_params_
LGBM_model = LGBMRegressor(**best_params)
LGBM_model.fit(X_train, np.ravel(y_train), eval_set=(X_test, np.ravel(y_test)), callbacks=[lgb.early_stopping(stopping_rounds=50)])


# # Regression Model (Random Forest)

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import classification_report


# X = train.drop(columns=["Calories"], axis=1)
# y = train[["Calories"]]

# # Split train-test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize RandomForestClassifier
# RF_model = RandomForestRegressor(n_estimators=100,
#                                  random_state=42,
#                                  verbose=1)

# # Fit the classifier to the training data
# RF_model.fit(X_train, np.ravel(y_train))


# # Regression Model (LightGBM)
# X = train.drop(columns=["Calories"], axis=1)
# y = train[["Calories"]]

# # Split train-test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# hyper_params = { 
#         "boosting_type": "gbdt",
#         "num_leaves": 64,
#         "max_depth": 12,
#         "learning_rate": 0.05,
#         "n_estimators": 1500,
#         "min_child_samples": 20,
#         "reg_alpha": 0.1,
#         "reg_lambda": 0.1,
#         "random_state": 42
# }

# # Initialize RandomForestClassifier
# RF_model = LGBMRegressor(**hyper_params)

# # Fit the classifier to the training data
# RF_model.fit(X_train, np.ravel(y_train), eval_set=(X_test, np.ravel(y_test)), callbacks=[lgb.early_stopping(stopping_rounds=50)])


# Save model

# import joblib
# joblib.dump(RF_model, "/kaggle/working/calories_random_forest_v1.pkl")


# Make predictions
# y_pred = RF_model.predict(X_test, num_iteration=RF_model.best_iteration_)
# rmse = np.sqrt(mean_squared_log_error(y_test, y_pred))
# print(f"Root Mean Square Error (RMSE): {rmse}")

y_pred = LGBM_model.predict(X_test, num_iteration=LGBM_model.best_iteration_)
rmse = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f"Root Mean Square Error (RMSE): {rmse}")


# Submission

final_submission = submission[[col for col in submission.columns if col != 'id']]
submission_pred = LGBM_model.predict(final_submission)
submission_df = pd.DataFrame(data={"id":submission["id"],"Calories":submission_pred})
submission_df.to_csv("submission.csv", index=False)


check_negative = (submission_df.values < 0).any()
check_negative

