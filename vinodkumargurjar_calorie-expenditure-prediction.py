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


from sklearn.metrics import mean_squared_error, r2_score


import warnings
warnings.filterwarnings('ignore')


df_train=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


df_train.head(5)


df_test.head(5)


# Feature Engineering
# Body Mass Index
df_train['BMI'] = df_train['Weight'] / (df_train['Height'] / 100) ** 2
df_test['BMI'] = df_test['Weight'] / (df_test['Height'] / 100) ** 2


df_train.head(5)


df_test.head(5)


df_train.drop(columns=["id"], axis=1,inplace=True)
df_test.drop(columns=["id"], axis=1,inplace=True)


import seaborn as sns
import matplotlib.pyplot as plt
target_variable="Calories"
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
                df_test[column] = label_encoders[column].transform(df_test[column].astype(str))
            else:
                le = LabelEncoder()
                df_test[column] = le.fit_transform(df_test[column].astype(str))
    
    return df_train, df_test


df_train, df_test = data_preprocessing_pipeline(df_train, df_test)


# df_train=pd.get_dummies(df_train, columns=["Sex"], drop_first=True)
# df_test=pd.get_dummies(df_test, columns=["Sex"], drop_first=True)


df_train.head(3)


df_test.head(3)


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


X = df_train.drop(columns=[target_variable])
y = df_train[target_variable]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# import optuna
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import cross_val_score, KFold
# import numpy as np

# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
#         "random_state": 42,
#         "verbosity": 0,
#     }
    
#     model = XGBRegressor(**params)
    
#     # Use 5-fold cross-validation with neg_mean_squared_error as scoring
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(model, X_train, y_train, cv=kf,
#                              scoring='neg_mean_squared_error', n_jobs=-1)
#     # Calculate RMSE from negative MSE scores
#     rmse = np.sqrt(-scores.mean())
#     return rmse

# # Create and optimize study
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50, timeout=600)

# print("Best RMSE (CV):", study.best_value)
# print("Best parameters:", study.best_params)

# # Train final model on full training data with best params
# best_model = XGBRegressor(**study.best_params)
# best_model.fit(X_train, y_train)

# # Predict on test data
# y_pred_best = best_model.predict(X_test)
# final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
# print(f"Final RMSE on Test Set: {final_rmse:.4f}")


# parameters= {'n_estimators': 506, 'learning_rate': 0.12161139412371726, 
#              'max_depth': 9, 'subsample': 0.5011785080755777, 
#              'colsample_bytree': 0.6432061913322253, 'gamma': 4.584312821464269, 
#              'reg_alpha': 0.30573782694787377, 'reg_lambda': 0.28010183008756095}


from sklearn.linear_model import LinearRegression

# ===========================
# 1. LINEAR REGRESSION
# ===========================

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)


print("\n=== Linear Regression ===")
print(f"Liner regression - MSE: {mse_lr:.4f}, RMSE: {rmse_lr:.4f}, RÂ²: {r2_lr:.4f}")



from sklearn.linear_model import ElasticNet

# Initialize Elastic Net
elastic = ElasticNet(alpha=0.001, l1_ratio=0.9, random_state=42)  # l1_ratio: 0 = Ridge, 1 = Lasso

# Fit to training data
elastic.fit(X_train, y_train)

# Predict
y_pred_elastic = elastic.predict(X_test)

# Evaluation
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
rmse_elastic = np.sqrt(mse_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)

# Print results
print("\n=== Elastic Net Regression ===")
print(f"MSE: {mse_elastic:.4f}, RMSE: {rmse_elastic:.4f}, RÂ²: {r2_elastic:.4f}")



from lightgbm import LGBMRegressor

# Initialize the model
lgbm_model = LGBMRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)

# Train
lgbm_model.fit(X_train, y_train)

# Predict
lgbm_preds = lgbm_model.predict(X_test)

# Evaluate
print("ðŸ”¸ LightGBM Regressor ðŸ”¸")
print("R2 Score:", r2_score(y_test, lgbm_preds))
print("RMSE:", mean_squared_error(y_test, lgbm_preds, squared=False))


from xgboost import XGBRegressor
# Initialize and train the XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.01, 
                         max_depth=10, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse) 

print(f"Mean Squared Error: {mse:.4f}")
print(f"RÂ² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# import optuna
# from catboost import CatBoostRegressor, Pool
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split

# # Optional: split  train set again to create a validation set for tuning
# X_tune, X_val, y_tune, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# def objective(trial):
#     params = {
#         "iterations": 1000,
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
#         "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
#         "border_count": trial.suggest_int("border_count", 32, 255),
#         "loss_function": "RMSE",
#         "verbose": 0,
#         "random_seed": 42
#     }

#     model = CatBoostRegressor(**params)
#     model.fit(X_tune, y_tune, eval_set=(X_val, y_val), early_stopping_rounds=50)

#     preds = model.predict(X_val)
#     rmse = mean_squared_error(y_val, preds, squared=False)
#     return rmse


# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)  # Increase trials for better tuning

# print("âœ… Best Trial:")
# print("  RMSE:", study.best_value)
# print("  Params:", study.best_params)


Params={'learning_rate': 0.09386391801815282, 'depth': 10, 
        'l2_leaf_reg': 0.06392249865950517, 'bagging_temperature': 0.005571026289453912,
        'random_strength': 0.0013022072654723985, 'border_count': 253,"iterations": 1000,
    "loss_function": "RMSE","random_seed": 42,"verbose": 100}


from catboost import CatBoostRegressor


# Initialize the model
cat_model = CatBoostRegressor(**Params)

# Train
cat_model.fit(X_train, y_train)

# Predict
cat_preds = cat_model.predict(X_test)

# Evaluate
print("ðŸ”¸ CatBoost Regressor ðŸ”¸")
print("R2 Score:", r2_score(y_test, cat_preds))
print("RMSE:", mean_squared_error(y_test, cat_preds, squared=False))



from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

# Base models
cat_model = CatBoostRegressor(verbose=0, random_seed=42)
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
lgbm_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)

# Stacking model with Ridge as meta-learner
stack_model = StackingRegressor(
    estimators=[
        ('cat', cat_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ],
    final_estimator=RidgeCV(),  # Meta-learner
    cv=5,
    passthrough=True,  # Pass original features + base predictions to meta-learner
    n_jobs=-1
)

# Train the stack
stack_model.fit(X_train, y_train)

# Predict
stack_preds = stack_model.predict(X_test)

# Evaluate
print("ðŸ”¸ Stacked Ensemble ðŸ”¸")
print("R2 Score:", r2_score(y_test, stack_preds))
print("RMSE:", mean_squared_error(y_test, stack_preds, squared=False))




errors = np.abs(y_test - stack_preds)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=stack_preds, hue=errors, palette="coolwarm", legend=False)
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Stacking Model: Actual vs Predicted (colored by error)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.show()



final_result=stack_model.predict(df_test)


sample_submission.head(4)


sample_submission["Calories"]=final_result
sample_submission.to_csv('submission.csv',index=False)


sample_submission.head(4)




