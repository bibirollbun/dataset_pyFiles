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
from sklearn.model_selection import train_test_split
import optuna
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error


df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


df_train.head()


df_test.head()


df_sample.head()


print("Shape of the Training Data:- ", df_train.shape)
print("Shape of the Testing Data:- ", df_test.shape)
print("Shape of the Sample_Submission Data:- ", df_sample.shape)


print("Duplicate present in Taining Data:- ", df_train.duplicated().sum())
print("Duplicate present in Testing Data:- ", df_train.duplicated().sum())


print("Missing Values in Training Data:-\n", df_train.isnull().sum())
print("Missing Values in Testing Data:-\n", df_test.isnull().sum())


df_train.info()


df_train.info()


df_train = df_train.drop("id", axis=1)
df_train.head()


columns_to_plot = ['country', 'store', 'product']

for column in columns_to_plot:
    plt.figure(figsize=(8, 6), dpi=100) 
    value_counts = df_train[column].value_counts()  
    sns.barplot(x=value_counts.index, y=value_counts.values)  
    plt.title(f'Value Counts for {column}')  
    plt.xlabel(f'{column} Categories')  
    plt.ylabel('Counts')  
    plt.xticks(rotation=45) 
    plt.show()  


df_train.describe()


plt.figure(figsize=(8, 6), dpi=100)
plt.hist(df_train['num_sold'], bins=40, alpha=0.7, edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('num_sold')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8, 6), dpi=100)
sns.boxplot(x = df_train["num_sold"])
plt.title("Boxplot of Sticker Sales")
plt.xlabel("Num_Sold")
plt.show()


# Skewness and Kurtosis
skewness = df_train['num_sold'].skew()
kurtosis = df_train['num_sold'].kurt()

print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")



features = ['country', 'store', 'product']
label = 'num_sold'

# Plot each feature against the label
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_train, x=feature, y=label)
    plt.title(f'Relationship between {feature} and {label}')
    plt.show()



# Fill with a constant value
df_train['num_sold'] = df_train['num_sold'].fillna(df_train['num_sold'].mean())


df_train.head()


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columns_to_encode = ['country', 'store', 'product']

for column in columns_to_encode:
    df_train[column] = encoder.fit_transform(df_train[column])


for column in columns_to_encode:
    df_test[column] = encoder.fit_transform(df_test[column])


df_test.head()


df_train.head()


id_col = df_test.pop('id')


id_col.head()


df_train = df_train.drop("date", axis=1)


df_train.head()


x = df_train.drop('num_sold', axis=1)
y = df_train['num_sold']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


def objective(trial):
    # Define hyperparameters to tune
    params = {
        "learning_rate": 0.1,
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 20, 30),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "reg_lambda": trial.suggest_float("reg_lambda", 20, 25),
        "max_bin": trial.suggest_int("max_bin", 260000, 270000),
    }

    # Initialize the model within a pipeline
    model = make_pipeline(
        StandardScaler(),  # Feature scaling
        XGBRegressor(**params)  # Using XGBRegressor for regression tasks
    )

    # Fit the model on training data
    model.fit(x_train, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(x_test)
    
    # Calculate the Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    return mape


# Create an Optuna study
study = optuna.create_study(study_name='XGBoost_MAPE', direction='minimize')  # MAPE is minimized
study.optimize(objective, n_trials=35)  # Run the optimization


best_params = study.best_params
print(best_params)


model =  XGBRegressor(**best_params)


model.fit(x_train, y_train)


df_test = df_test.drop("date", axis=1)


# Predictions on Test data
y_pred = model.predict(df_test)


pred = model.predict(x_test)


df_test['num_sold'] = y_pred


result = pd.DataFrame({'id' : id_col, 'num_sold' : df_test['num_sold']})
# Save the df as csv
result.to_csv("/kaggle/working/submission.csv",index=False)

