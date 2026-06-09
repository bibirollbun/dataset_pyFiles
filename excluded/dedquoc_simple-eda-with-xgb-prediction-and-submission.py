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


#!pip install lifetime
!pip install lifelines==0.27.4  # Example version, adjust as needed
'''
!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl
'''


## and restart kernel
# import os
# os._exit(00)


from lifelines import KaplanMeierFitter


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# Load the datasets
path = '/kaggle/input/equity-post-HCT-survival-predictions/'

train_df = pd.read_csv(path + 'train.csv', nrows=500)
test_df = pd.read_csv(path + 'test.csv',  nrows=500)
sample_submission_df = pd.read_csv(path + 'sample_submission.csv', nrows=500)
data_dict = pd.read_csv(path + 'data_dictionary.csv',  nrows=500)


### Display the first few rows of the training data
train_df.head()
# data_dict.columns


# Check for missing values in the training dataset
print("Missing values in train dataset:\n", train_df.isnull().sum())

# Check for missing values in the test dataset
print("Missing values in test dataset:\n", test_df.isnull().sum())


# Distribution of the target variable 'efs'
plt.figure(figsize=(8, 6))
sns.countplot(x='efs', data=train_df)
plt.title('Distribution of Event-Free Survival (efs)')
plt.show()

# Distribution of numerical features
numerical_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(train_df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()


# Analyze the distribution and impact of categorical features..
# Identify categorical features
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

# Distribution of categorical features
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, data=train_df)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)
    plt.show()

# Impact of categorical features on the target variable
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, hue='efs', data=train_df)
    plt.title(f'Impact of {feature} on Event-Free Survival (efs)')
    plt.xticks(rotation=45)
    plt.show()


# Analyze the distribution and impact of categorical features
# Identify categorical features
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

# Distribution of categorical features
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, data=train_df)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)
    plt.show()

# Impact of categorical features on the target variable
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, hue='efs', data=train_df)
    plt.title(f'Impact of {feature} on Event-Free Survival (efs)')
    plt.xticks(rotation=45)
    plt.show()


numeric_columns_train = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns_test= test_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns_train = train_df.select_dtypes(include="object").columns.tolist()
categorical_columns_test = test_df.select_dtypes(include="object").columns.tolist()


from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Initialize the KaplanMeierFitter
kmf = KaplanMeierFitter()

# Extract the time and event data
T = train_df["efs_time"]
E = train_df["efs"]

# Fit the Kaplan-Meier model
kmf.fit(T, event_observed=E, label="Kaplan-Meier Estimate")

# Plot the survival function
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.show()


from lifelines import KaplanMeierFitter
import pandas as pd

# Initialize the KaplanMeierFitter
kmf = KaplanMeierFitter()

# Extract the time and event data
T = train_df["efs_time"]
E = train_df["efs"]

# Fit the Kaplan-Meier model
kmf.fit(T, event_observed=E, label="Kaplan-Meier Estimate")

# Calculate the survival function at the event times
train_df['km_label'] = kmf.survival_function_at_times(train_df['efs_time']).values

# Adjust the survival probability for censored data
train_df.loc[train_df['efs'] == 0, 'km_label'] -= 0.1

# Display the first few rows of the DataFrame to verify
print(train_df.head())


train_df[categorical_columns_train] = train_df[categorical_columns_train].astype('category')
test_df[categorical_columns_test] = test_df[categorical_columns_test].astype('category')


X = train_df.iloc[:,:-3]
y = train_df.iloc[:,-1]


from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from bayes_opt import BayesianOptimization

# Initialize XGBoost regressor
xgb_reg = XGBRegressor(
    objective='reg:squarederror', 
    eval_metric='rmse',
    enable_categorical=True  # Enable categorical data support (experimental)
)

# Five-fold cross-validation (regression task using KFold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and generate predictions
y_pred_original = cross_val_predict(xgb_reg, X, y, cv=kf)

# Output regression report (using R² and mean squared error)
print("Original Dataset Regression Report")
print(f"R2 Score: {r2_score(y, y_pred_original):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred_original):.4f}")

def xgb_eval(n_estimators, learning_rate, max_depth, colsample_bytree):
    """
    Evaluate XGBoost model using cross-validation and return the mean R² score.
    
    Parameters:
    - n_estimators: Number of trees to fit.
    - learning_rate: Step size shrinkage used in update to prevents overfitting.
    - max_depth: Maximum depth of a tree.
    - colsample_bytree: Subsample ratio of columns when constructing each tree.
    
    Returns:
    - Mean R² score from cross-validation.
    """
    params = {
        "n_estimators": int(round(n_estimators)),
        "learning_rate": learning_rate,
        "max_depth": int(round(max_depth)),
        "colsample_bytree": colsample_bytree,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "enable_categorical": True  # Enable categorical data support (experimental)
    }
    # Perform cross-validation using KFold and R² as the scoring metric
    cv_result = cross_val_score(
        XGBRegressor(**params),
        X,
        y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2"
    ).mean()
    return cv_result

# Bayesian optimization
xgb_bo = BayesianOptimization(
    f=xgb_eval,
    pbounds={
        "n_estimators": (50, 300),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "colsample_bytree": (0.5, 1.0)
    },
    random_state=42
)

# Perform optimization: initialize with 5 random points and run 30 iterations
xgb_bo.maximize(init_points=5, n_iter=30)

# Output the best parameters
best_params = xgb_bo.max["params"]

print("Best XGBoost Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value:.4f}")


from xgboost import XGBRegressor

# Define the best parameters obtained from the Bayesian optimization
best_params = {
    "n_estimators": int(round(168.0585)),
    "learning_rate": 0.1900,
    "max_depth": int(round(3.0753)),
    "colsample_bytree": 0.8761,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "enable_categorical": True
}

# Initialize the tuned XGBoost regressor with the best parameters
xgb_tuned = XGBRegressor(**best_params)

# Fit the model to the data
xgb_tuned.fit(X, y)


from lifelines.utils import concordance_index
from xgboost import XGBRegressor

# Assuming xgb_tuned is already trained and X, y are your feature and target data
# If not, you need to train the model first
# xgb_tuned.fit(X, y)

# Predict the scores (risk scores or survival times)
y_pred = xgb_tuned.predict(X)

# Calculate the C-index
c_index = concordance_index(y, y_pred)
print("C-index:", c_index)


import matplotlib.pyplot as plt
from xgboost import plot_importance

# Create a figure and axis with specified size
fig, ax = plt.subplots(1, 1, figsize=(24, 22))

# Plot feature importance
ax = plot_importance(
    xgb_tuned,
    show_values=False,  # Do not display numerical values on the plot
    title="Feature Importance | XGBoost Model",
    ax=ax,
    xlabel="",  # Remove x-axis label
    height=0.7,  # Bar height
    color="#1f77b4"  # Blue color for bars
)

# Add numerical values on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="{:,.1f}", fontsize=8)

# Remove grid lines
ax.grid(False)

# Show the plot
plt.show()



import pandas as pd

# Assuming 'prediction' is the array of predictions from your model
# and 'test_df' is your test DataFrame containing the 'ID' column

prediction = xgb_tuned.predict(test_df)
# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_df['ID'],
    'prediction': prediction
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

# Print a success message
print("Submission was successfully saved!")

