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


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
original = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')


print(train.head())
print(test.head())




print(original.head())


print(train.shape)
print(test.shape)
print(original.shape)


train_df = pd.concat([train, original])


train_df['Price'].isnull().sum()


train_df.describe(include ='all').T


train_df.info()


test.info()


train_df.isnull().sum()


test.isnull().sum()


print(train_df.duplicated().sum())
print(test.duplicated().sum())


train_df = train_df.drop("id", axis=1)
test = test.drop("id", axis=1)


train_df.shape


train.columns


test.shape


import matplotlib.pyplot as plt
import seaborn as sns

# Separate numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns

# Plotting for numerical columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    
    # Histogram for distribution
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
    
    # Boxplot for spread and outliers
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Plotting for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))

# Count plot for distribution of categories
    sns.countplot(x=train[col])
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.show()


cat_cols = train_df.select_dtypes(include ="object").columns
print(cat_cols)

num_cols = train_df.select_dtypes(exclude = 'object').columns
print(num_cols)


cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof','Style', 'Color']
num_cols = ['Compartments', 'Weight Capacity (kg)'] 


mean_value = train_df['Price'].mean()
train_df['Price'].fillna(mean_value, inplace=True)


import pandas as pd

def fill_missing_values(train, test, cat_cols, num_cols):
    """
    Fills missing values in categorical columns with mode (from train) 
    and numerical columns with mean (from train).

    Parameters:
    train (pd.DataFrame): Training dataset
    test (pd.DataFrame): Testing dataset
    cat_cols (list): List of categorical column names
    num_cols (list): List of numerical column names

    Returns:
    pd.DataFrame, pd.DataFrame: Updated train and test DataFrames
    """
    # Fill categorical columns with mode from train set
    for col in cat_cols:
        mode_value = train_df[col].mode()[0]  # Mode from train set
        train_df[col].fillna(mode_value, inplace=True)
        test[col].fillna(mode_value, inplace=True)  # Apply same mode to test set

    # Fill numerical columns with mean from train set
    for col in num_cols:
        mean_value = train_df[col].mean()  # Mean from train set
        train_df[col].fillna(mean_value, inplace=True)
        test[col].fillna(mean_value, inplace=True)  # Apply same mean to test set

    return train, test


train_df, test = fill_missing_values(train_df, test, cat_cols, num_cols)



train_df.isnull().sum()


test.isnull().sum()


test.shape


import pandas as pd

def encode_categorical_features(train, test, cat_cols):
    """
    Performs one-hot encoding on categorical columns and ensures train-test consistency.

    Parameters:
    train (pd.DataFrame): Training dataset
    test (pd.DataFrame): Testing dataset
    cat_cols (list): List of categorical column names to encode

    Returns:
    pd.DataFrame, pd.DataFrame: Encoded train and test DataFrames
    """
    # Apply one-hot encoding to categorical columns
    train_encoded = pd.get_dummies(train, columns=cat_cols, drop_first=True)
    test_encoded = pd.get_dummies(test, columns=cat_cols, drop_first=True)
    
    # Align train and test to have the same columns
    train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)
    
    return train_encoded, test_encoded


train_df, test = encode_categorical_features(train_df, test, cat_cols)



train_df.head()


test.head()


X = train_df.drop(["Price"], axis=1)
y = train_df["Price"]
test = test.drop(["Price"], axis=1)


X.head()


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)


print("X_train columns:", X_train.shape)
print("X_test columns:", test.shape)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# Fit only on training data
X_train = sc.fit_transform(X_train)

# Transform validation & test sets using the same scaler
X_val = sc.transform(X_val)
X_test = sc.transform(test) 



import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

# Define the Optuna objective function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "objective": "reg:squarederror",
        "tree_method": "hist"  # Use GPU if available
    }

    model = xgb.XGBRegressor(**params)

    # Use 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
    
    return scores.mean()

# Run Optuna optimization
study = optuna.create_study(direction="maximize")  # Maximize negative RMSE (minimize RMSE)
study.optimize(objective, n_trials=50)

# Best parameters
print("Best params:", study.best_params)



best_params = study.best_params
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)



y_val_pred =best_model.predict(X_val)


print(y_val_pred)


from sklearn.metrics import mean_squared_error

rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(rmse_val)


prediction = best_model.predict(X_test)


submission =pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
submission["Price"] = prediction


submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")

