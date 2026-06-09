import warnings
warnings.filterwarnings('ignore')


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


df_train=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


df_train.head(5)


df_train.drop(columns=["id"], axis=1,inplace=True)
df_test.drop(columns=["id"], axis=1,inplace=True)


import seaborn as sns
import matplotlib.pyplot as plt
target_variable="Listening_Time_minutes"
def eda_pipeline(df_train, df_test):
    
    # Display first few rows
    print("\n--- First few rows of train data ---")
    display(df_train.head())
    
    print("\n--- First few rows of test data ---")
    display(df_test.head())
    
    # Dataset info
    print("\n--- Train Data Info ---")
    print(df_train.info())
    
    print("\n--- Test Data Info ---")
    print(df_test.info())
    
    # Missing values
    print("\n--- Missing Values in Train Data ---")
    print(df_train.isnull().sum())
    
    print("\n--- Missing Values in Test Data ---")
    print(df_test.isnull().sum())
    
    print("\n--- Percentage of Missing Values in Train Data ---")
    print((df_train.isnull().sum() / len(df_train)) * 100)
    
    print("\n--- Percentage of Missing Values in Test Data ---")
    print((df_test.isnull().sum() / len(df_test)) * 100)
    
    # Summary statistics
    print("\n--- Train Data Summary Statistics ---")
    print(df_train.describe())
    
    print("\n--- Test Data Summary Statistics ---")
    print(df_test.describe())
    
    # Identify categorical columns
    train_cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'O']
    test_cat_columns = [col for col in df_test.columns if df_test[col].dtype == 'O']
    
    print("\n--- Categorical Columns in Train Data ---")
    print(train_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Train) ---")
    print(df_train[train_cat_columns].nunique())
    
    print("\n--- Categorical Columns in Test Data ---")
    print(test_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Test) ---")
    print(df_test[test_cat_columns].nunique())
    
    # Identify numerical columns
    train_num_columns = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64']]
    test_num_columns = [col for col in df_test.columns if df_test[col].dtype in ['int64', 'float64']]
    
    print("\n--- Numerical Columns in Train Data ---")
    print(train_num_columns)
    
    print("\n--- Numerical Columns in Test Data ---")
    print(test_num_columns)
    
    # Check for duplicate rows
    print("\n--- Duplicate Rows in Train Data ---")
    print(df_train.duplicated().sum())
    
    print("\n--- Duplicate Rows in Test Data ---")
    print(df_test.duplicated().sum())
    
    # Correlation matrix (excluding non-numeric columns)
    print("\n--- Correlation Matrix ---")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_train[train_num_columns].corr(), annot=True, cmap='coolwarm')
    plt.show()
    
    # Correlation with Target Variable
    print("\n--- Correlation with Target Variable ---")
    target_corr = df_train[train_num_columns].corr()[target_variable].sort_values(ascending=False)
    print(target_corr)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=target_corr.index, y=target_corr.values, palette='coolwarm')
    plt.xticks(rotation=90)
    plt.title(f'Feature Correlation with {target_variable}')
    plt.show()   
    
    # Distribution plots for numerical features
    print("\n--- Distribution of Numerical Features ---")
    df_train[train_num_columns].hist(figsize=(12, 10), bins=30)
    plt.show()
    
    # Box plots for outlier detection
    print("\n--- Box Plots for Outlier Detection ---")
    for col in train_num_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df_train[col])
        plt.title(f'Box plot of {col}')
        plt.show()
    
    # Value counts for categorical features
    print("\n--- Value Counts for Categorical Columns ---")
    for col in train_cat_columns:
        print(f"\nValue counts for {col}:")
        print(df_train[col].value_counts())


eda_pipeline(df_train, df_test)


# # Drop Episode_Title as it has 100 unique values
# df_train.drop(columns=['Episode_Title'],axis=1, inplace=True)
# df_test.drop(columns=['Episode_Title'],axis=1, inplace=True)


# df_train.head(2)


from sklearn.preprocessing import LabelEncoder

def data_preprocessing_pipeline(df_train, df_test):
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.
    """
    # Fill missing values
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            mode_value = df_train[column].mode()[0]  # Fill categorical with mode
            df_train[column].fillna(mode_value, inplace=True)
        elif df_train[column].dtype in ['int64', 'float64']:
            mean_value = df_train[column].mean()  # Fill numerical with mean
            df_train[column].fillna(mean_value, inplace=True)
    
    for column in df_test.columns:
        if df_test[column].dtype == 'object':
            mode_value = df_test[column].mode()[0]
            df_test[column].fillna(mode_value, inplace=True)
        elif df_test[column].dtype in ['int64', 'float64']:
            mean_value = df_test[column].mean()
            df_test[column].fillna(mean_value, inplace=True)
    
    # Encode categorical features
    label_encoders = {}
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            le = LabelEncoder()
            df_train[column] = le.fit_transform(df_train[column].astype(str))
            label_encoders[column] = le  # Store encoder for consistency
    
    for column in df_test.columns:
        if df_test[column].dtype == 'object':
            if column in label_encoders:
                le = label_encoders[column]
                # Handle unseen labels by assigning -1
                df_test[column] = df_test[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                df_test[column] = -1  # Assign -1 if encoder was not created in df_train
    
    return df_train, df_test


# df_train,df_test = data_preprocessing_pipeline(df_train, df_test)


# eda_pipeline(df_train, df_test)


from sklearn.preprocessing import StandardScaler

def standardize_data(df_train, df_test):
    """
    Standardize all numerical features using StandardScaler,
    ensuring both train and test have the same columns, while preserving the target variable.
    """
    # Separate target column from train data
    target_values = df_train[target_variable]
    df_train = df_train.drop(columns=[target_variable])
    
    # Ensure both datasets have the same feature columns
    common_columns = df_train.columns.intersection(df_test.columns)
    df_train = df_train[common_columns]
    df_test = df_test[common_columns]
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit on train data and transform both train and test data
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=common_columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=common_columns)
    
    # Reattach the target column to the scaled train data
    df_train_scaled[target_variable] = target_values.reset_index(drop=True)
    
    return df_train_scaled, df_test_scaled


# df_train_scaled, df_test_scaled = standardize_data(df_train, df_test)


# df_train.head(2)


# df_train_scaled.head(2)


X = df_train.drop(columns=[target_variable])
y = df_train[target_variable]


# X1 = df_train_scaled.drop(columns=[target_variable])
# y1 = df_train_scaled[target_variable]


# Identify categorical columns
train_cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'O']
test_cat_columns = [col for col in df_test.columns if df_test[col].dtype == 'O']


train_cat_columns


test_cat_columns


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)


# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=500,
#                          random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)
# rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
# print(f'XGBoost RMSE: {rmse_xgb:.4f}')


# catboost_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.01, 
#                                    loss_function='RMSE', verbose=0)
# catboost_model.fit(X_train, y_train,cat_features=train_cat_columns)
# # Predictions

# y_pred_catboost = catboost_model.predict(X_test)

# # Compute RMSE
# rmse_catboost = np.sqrt(mean_squared_error(y_test, y_pred_catboost))

# # Print RMSE scores
# print(f'CatBoost RMSE: {rmse_catboost:.4f}')


for col in train_cat_columns:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')
    df_test[col] = df_test[col].astype('category')


# import optuna
# import numpy as np
# import pandas as pd
# from lightgbm import LGBMRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # Objective function for Optuna
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'num_leaves': trial.suggest_int('num_leaves', 10, 300),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
#         'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
#         'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
#         'boosting_type': 'gbdt',  # Keep default boosting type
#         'random_state': 42
#     }

#     # Split data into train and validation sets
#     X_train_, X_val_, y_train_, y_val_ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#     # Train the model with suggested parameters
#     model = LGBMRegressor(**params)
#     model.fit(X_train_, y_train_, eval_set=[(X_val_, y_val_)])

#     # Predict and compute RMSE
#     y_pred = model.predict(X_val_)
#     rmse = np.sqrt(mean_squared_error(y_val_, y_pred))
    
#     return rmse  # Minimize RMSE

# # Run Optuna optimization
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# # Best parameters
# print("Best Parameters:", study.best_params)



best_params={'n_estimators': 1741, 'learning_rate': 0.018932176136796892, 
             'max_depth': 11, 'num_leaves': 255, 'min_data_in_leaf': 47, 
             'feature_fraction': 0.6018134320675068, 
             'bagging_fraction': 0.8238644579396707, 
             'bagging_freq': 2, 'lambda_l1': 2.292538745570246, 
             'lambda_l2': 3.323840129415645}


lgbm_model = LGBMRegressor(**best_params, verbosity=-1,random_state=42)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
print(f'LightGBM RMSE: {rmse_lgbm:.4f}')



# # Define K-Fold Cross Validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Store RMSE for each fold
# rmse_scores = []

# for train_index, val_index in kf.split(X_train):
#     # Split into training and validation sets
#     X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
#     y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
#     # Initialize and train the model
#     lgbm_model_KFold = LGBMRegressor(n_estimators=1000, random_state=42)
#     lgbm_model_KFold.fit(X_train_fold, y_train_fold)
    
#     # Predict on validation set
#     y_val_pred = lgbm_model_KFold.predict(X_val_fold)
    
#     # Compute RMSE
#     rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
#     rmse_scores.append(rmse)

# # Print results
# print(f'RMSE scores for each fold: {rmse_scores}')
# print(f'Average RMSE: {np.mean(rmse_scores):.4f}')


# hgb_model = HistGradientBoostingRegressor(
#     categorical_features=categorical_feature_indices,
#     random_state=42
# )

# hgb_model.fit(X_train, y_train)
# y_pred_hgb = hgb_model.predict(X_test)

# rmse_hgb = np.sqrt(mean_squared_error(y_test, y_pred_hgb))
# print(f'HGB RMSE: {rmse_hgb:.4f}')


# result1=hgb_model.predict(df_test)
# result2=lgbm_model.predict(df_test)
# result3=xgb_model.predict(df_test)
# result4=catboost_model.predict(df_test)


final_prediction=lgbm_model.predict(df_test)


# final_prediction=(result1+result2+result3+result4)/4


sample_submission.head(3)


sample_submission["Listening_Time_minutes"]=final_prediction
sample_submission.to_csv("submission.csv",index=False)


sample_submission.head(3)




