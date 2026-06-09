import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib
import os
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns


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


# load the dataset for training , testing , and submission csv files
prefix = '/kaggle/input/playground-series-s5e10/'

df_train = pd.read_csv(prefix +'train.csv')
df_test = pd.read_csv(prefix + 'test.csv')
df_submission = pd.read_csv(prefix + 'sample_submission.csv')


# global variables
NEED_HANDLE_BOOL_VALUES = False
TARGET = 'accident_risk'
ID = 'id'
RANDOM_STATE = 42
HOW_TO_HANDLE_CATEGORICAL = 'onehot'  # 'onehot' or 'labelencoding'
NEED_TO_SCALE_NUMERICAL = False
NEED_TO_REMOVE_USLESSARY_FEATURES = False
USLESSARY_FEATURES = ['road_type', 'time_of_day']


if NEED_TO_REMOVE_USLESSARY_FEATURES:
    df_train = df_train.drop(columns=USLESSARY_FEATURES, errors='ignore')
    df_test = df_test.drop(columns=USLESSARY_FEATURES, errors='ignore')


# check all columns in the df_train, found out the columns names and types
print(df_train.info())


columns_name_list = df_train.columns.tolist()
print(columns_name_list)


# check the df_train,  df_test to see if there is any missing values
print(df_train.isnull().sum())
print(df_test.isnull().sum())

# from the result we can see that there is no missing values in the dataset
# which is good for us


# 3) if you want nullable integers for missing bools:
# df[bool_cols] = df[bool_cols].astype('Int64')
# 3) identify categorical columns (object / category)
cat_cols = [c for c in df_train.columns
            if df_train[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df_train[c])]

# avoid encoding id-like columns (if present)
id_cols = [c for c in df_train.columns if c.lower().startswith('id')]
cat_cols = [c for c in cat_cols if c not in id_cols]


print(f"Categorical columns list is : {cat_cols}")

# fill missing categorical values with a sentinel
for df in (df_train, df_test):
    df[cat_cols] = df[cat_cols].fillna('MISSING')


# get all the columns types, we have 4 boolean columns, 4 categorical columns, and 4 numerical columns
bool_cols = df_train.select_dtypes(include=['bool']).columns.tolist()
print(f"boolean columns list is : {bool_cols}")
number_cols = df_train.select_dtypes(include=['number']).columns.tolist()
number_cols.remove(TARGET)  # exclude target from features
number_cols.remove(ID)
print(f"number columns list is : {number_cols}")
print(f"Categorical columns list is : {cat_cols}")




print("Generating visualizations for categorical columns vs. target...")

# --- 1. Box Plots ---
print("\n--- Box Plots: Distribution of target by category ---")
for col in cat_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y=TARGET, data=df_train, palette='Set3')
    plt.title(f'Distribution of {TARGET} by {col}', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel(TARGET, fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

# --- 2. Violin Plots ---
print("\n--- Violin Plots: Density and distribution of target by category ---")
for col in cat_cols:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=col, y=TARGET, data=df_train, inner='quartile') # 'inner' shows quartiles inside
    plt.title(f'Density and Distribution of {TARGET} by {col}', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel(TARGET, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# --- 3. Bar Plots ---
print("\n--- Bar Plots: Mean target by category with confidence intervals ---")
for col in cat_cols:
    plt.figure(figsize=(10, 6))
    # ci='sd' shows standard deviation, ci='None' removes them
    sns.barplot(x=col, y=TARGET, data=df_train, ci='sd')
    plt.title(f'Mean {TARGET} by {col} (with Std Dev)', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel(f'Mean {TARGET}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# --- 4. Point Plots ---
# Point plots can be great if you want to emphasize changes or compare groups across another variable
# For just one categorical vs target, it's similar to bar plot but with points and lines.
print("\n--- Point Plots: Mean target by category with confidence intervals ---")
for col in cat_cols:
    plt.figure(figsize=(10, 6))
    sns.pointplot(x=col, y=TARGET, data=df_train, ci='sd', linestyles="--", markers="o")
    plt.title(f'Mean {TARGET} by {col} (with Std Dev)', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel(f'Mean {TARGET}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

print("\nVisualizations completed!")


# --- 2. Code to Generate Box Plots ---
print("--- Visualizing Target Distribution using Box Plots ---")
for column in bool_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=column, y=TARGET, data=df_train, palette="Set2")
    plt.title(f'Distribution of {TARGET} by {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel(TARGET, fontsize=12)
    plt.tight_layout()
    # Save the plot to a file (optional)
    # plt.savefig(f'boxplot_{column}.png')
    plt.show()


# --- 3. Code to Generate Bar Plots ---
print("\n--- Visualizing Mean Target using Bar Plots ---")
for column in bool_cols:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=column, y=TARGET, data=df_train, palette="Set3")
    plt.title(f'Mean {TARGET} by {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel(f'Mean {TARGET}', fontsize=12)
    plt.tight_layout()
    # Save the plot to a file (optional)
    # plt.savefig(f'barplot_{column}.png')
    plt.show()

print("\nCode execution finished.")


# use seaborn and matplotlib to visualize the relationship between categorical columns and the target variable. use distrubution plots, box plots, bar plots, and point plots

for column in cat_cols:
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_train, x=TARGET, hue=column, element="step", stat="density", common_norm=False)
    plt.title(f'Distribution of {TARGET} by {column}', fontsize=16)
    plt.xlabel(TARGET, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.show()



# check the number columns vs target using scatter plots
for columns in number_cols:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_train, x=columns, y=TARGET, alpha=0.5)
    plt.title(f'Scatter Plot of {TARGET} vs {columns}', fontsize=16)
    plt.xlabel(columns, fontsize=12)
    plt.ylabel(TARGET, fontsize=12)
    plt.tight_layout()
    plt.show()


if HOW_TO_HANDLE_CATEGORICAL == 'onehot':
    # 4) one-hot encode with pandas.get_dummies on concatenated data to ensure same columns
    concat = pd.concat([df_train.drop(columns=id_cols, errors='ignore'),
                        df_test.drop(columns=id_cols, errors='ignore')],
                    keys=['_train','_test'])
    concat = pd.get_dummies(concat, columns=cat_cols, drop_first=False)

    # split back into train/test and restore id columns if any
    df_train_ohe = concat.xs('_train')
    df_test_ohe = concat.xs('_test')

    # if there were id columns, reattach them (preserve original index)
    for c in id_cols:
        if c in df_train.columns:
            df_train_ohe[c] = df_train[c].values
        if c in df_test.columns:
            df_test_ohe[c] = df_test[c].values

    # overwrite variables so subsequent cells use encoded data
    df_train = df_train_ohe.reset_index(drop=True)
    df_test = df_test_ohe.reset_index(drop=True)
elif HOW_TO_HANDLE_CATEGORICAL == 'labelencoding':
    # 5) label encode categorical columns (ordinal encoding)
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(pd.concat([df_train[cat_cols], df_test[cat_cols]], axis=0))
    df_train[cat_cols] = encoder.transform(df_train[cat_cols])
    df_test[cat_cols] = encoder.transform(df_test[cat_cols])
else:
    print(f"Unknown HOW_TO_HANDLE_CATEGORICAL: {HOW_TO_HANDLE_CATEGORICAL} , do nothing")


df_train.info()


# convert 4 boolean columns to int
# normalize boolean-like columns to 0/1
# native bools -> int
if NEED_HANDLE_BOOL_VALUES:
    bool_cols = df_train.select_dtypes(include=['bool']).columns.tolist()
    print(f"boolean columns list is : {bool_cols}")

    # 
    # normalize boolean-like columns to 0/1
    # 1) native bools -> int
    bool_cols = df_train.select_dtypes(include=['bool']).columns.tolist()
    for df in (df_train, df_test):
        if bool_cols:
            print(f"Converting boolean columns to int: {bool_cols}")
            df[bool_cols] = df[bool_cols].astype(int)

    # 2) string 'True'/'False' -> int
    for df in (df_train, df_test):
        for c in bool_cols:
            print(f"Converting string boolean columns True/False to int 1/0 : {c}")
            df[c] = df[c].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})


# loop over numerical columns in Train and Test datasets, find the null fill with 0
for columns in number_cols:
    for df in (df_train, df_test):
        if df[columns].isnull().sum() > 0:
            print(f"Filling missing values in numerical column {columns} with 0")
            df[columns] = df[columns].fillna(0)


y_train = df_train[TARGET]
X_train = df_train.drop(columns=[TARGET, ID], errors='ignore')
X_test = df_test.drop(columns=[ID , TARGET], errors='ignore')


# use split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)

# scale the numerical columns
if NEED_TO_SCALE_NUMERICAL:
    scaler = StandardScaler()
    scaler.fit(X_train[number_cols])
    X_train[number_cols] = scaler.transform(X_train[number_cols])
    X_val[number_cols] = scaler.transform(X_val[number_cols])
    X_test[number_cols] = scaler.transform(X_test[number_cols])
    print("Numerical columns scaled using StandardScaler.")

# use XGBoost to train the model and validate it with the validation set, use k-fold cross validation to find the best hyperparameters
# use XGBoost to train the model and validate it with the validation set, use k-fold cross validation to find the best hyperparameters
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
# evaluate the model with the validation set
y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)
print(f"Validation RMSE: {rmse}")



# use the model to predict the test set
X_test = df_test.drop(columns=[ID , TARGET], errors='ignore')
y_test_pred = model.predict(X_test)
print("Test set predictions completed.", y_test_pred)



# print the shape of the test set predictions
print("Test set predictions shape:", y_test_pred.shape)


df_submission[TARGET] = y_test_pred
df_submission.to_csv('submission_xgboost.csv', index=False)

