from IPython.display import clear_output
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import csv
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression


# Loading data
data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


# Dimension 
dim_train = data.shape
dim_test = test_data.shape
print(f'Dimension of the training set: {dim_train}')
print(f'Dimension of the test set: {dim_test}')


# Variables 
var_list = list(data)
var_list


# Description of variables 
basic_st = data.describe()
basic_st


# Types of variables (training set)
data.dtypes


# Types of variables (test set)
test_data.dtypes


# Transforming dates to datetime 
data['date'] = pd.to_datetime(data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])


# Nulls and n/a (Training set)
n_a = data.isna().sum()

# Seaborn style
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

# Barplot representing the n/a values (%)
plt.figure(figsize=(12, 6))
sns.barplot(x=n_a.index, y=n_a / len(data), palette="Reds")  
plt.title('Values n/a  (%)')
plt.xlabel('Variables')
plt.ylabel('% of n/a per variable')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Numeric values
print("** Total n/a variables **")
print(n_a)



# Nulls and n/a (Test set)
n_a = test_data.isna().sum()

# Seaborn style
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

# Barplot representing the n/a values (%)
plt.figure(figsize=(12, 6))
sns.barplot(x=n_a.index, y=n_a / len(data), palette="Greens")  
plt.title('Values n/a  (%)')
plt.xlabel('Variables')
plt.ylabel('% of n/a per variable')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Numeric values
print("** Total n/a variables **")
print(n_a)


# Cleaning n/a values 
data = data.dropna(subset=['num_sold'])


# Nulls and n/a (Training set)
n_a = data.isna().sum()

# Seaborn style
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

# Barplot representing the n/a values (%)
plt.figure(figsize=(12, 6))
sns.barplot(x=n_a.index, y=n_a / len(data), palette="Reds")  # Removed 'hue' parameter
plt.title('Values n/a  (%)')
plt.xlabel('Variables')
plt.ylabel('% of n/a per variable')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Numeric values
print("** Total n/a variables **")
print(n_a)



# Separate numeric and categorical data directly into DataFrames
numeric_data = data.select_dtypes(include=['number'])
categorical_data = data.select_dtypes(exclude=['number','datetime'])


# Unique categories for each categorical variable
category_counts = categorical_data.nunique()

print("Number of categories in each categorical variable:")
print(category_counts)


# Histograms for all variables in the dataset
data['num_sold'].hist(bins=50, figsize=(8, 8), color='skyblue', edgecolor='black')

# Title for the entire figure
plt.suptitle('Histogram of the target variable: num_sold', fontsize=20, fontweight='bold')

# Showing the plot
plt.tight_layout()  
plt.show()


# Boxplot for the target variable 'num_sold'
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['num_sold'], color="skyblue", width=0.6)

# Titles and labels
plt.title('Boxplot of the Target Variable: num_sold', fontsize=16, weight='bold')
plt.xlabel('num_sold', fontsize=12)

# Grid for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()


# Copy of the dataframes
data_1 = data.copy()
test_1 = test_data.copy()


# Extracting year, month, day, and day of the week into separate columns
# Training set
data_1['year'] = data_1['date'].dt.year
data_1['month'] = data_1['date'].dt.month
data_1['day'] = data_1['date'].dt.day
data_1['day_of_week'] = data_1['date'].dt.day_name()

# Test set
test_1['year'] = test_1['date'].dt.year
test_1['month'] = test_1['date'].dt.month
test_1['day'] = test_1['date'].dt.day
test_1['day_of_week'] = test_1['date'].dt.day_name()



#Mapping of days to integers
day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

# Applying the mapping to the 'day_of_week' column
data_1['day_of_week'] = data_1['day_of_week'].map(day_mapping)
test_1['day_of_week'] = test_1['day_of_week'].map(day_mapping)


# Weekend indicator
data_1['is_weekend'] = data_1['day_of_week'].isin([5, 6]).astype(int)
test_1['is_weekend'] = test_1['day_of_week'].isin([5, 6]).astype(int)

# Quarter of the year
data_1['quarter'] = pd.to_datetime(data_1['date']).dt.quarter
test_1['quarter'] = pd.to_datetime(test_1['date']).dt.quarter


# Defining categorical and numeric features
categorical_columns = ['store', 'product']
numeric_features = ['year', 'month', 'day', 'day_of_week', 'quarter']
target_column = 'num_sold'


seed = 52


warnings.filterwarnings("ignore", category=FutureWarning)

# Creating target encodings using K-Fold to reduce data leakage
def target_encode(train_df, test_df, target_col, categorical_cols, n_splits=5):
    encoded_train = train_df.copy()
    encoded_test = test_df.copy()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for col in categorical_cols:
        # Placeholder for encoded values
        train_encoded = pd.Series(0, index=encoded_train.index)
        test_values = []
        
        for train_idx, val_idx in kf.split(train_df):
            fold_train, fold_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
            mean_encoding = fold_train.groupby(col)[target_col].mean()
            train_encoded.iloc[val_idx] = fold_val[col].map(mean_encoding)
        
        # Filling in train_df
        encoded_train[f'{col}_target_encoded'] = train_encoded
        
        # Calculating mean encoding on full training set for test_df
        full_mean_encoding = train_df.groupby(col)[target_col].mean()
        encoded_test[f'{col}_target_encoded'] = encoded_test[col].map(full_mean_encoding)
    
    # Dropping original categorical columns
    for col in categorical_cols:
        encoded_train.drop(columns=[col], inplace=True)
        encoded_test.drop(columns=[col], inplace=True)
    
    return encoded_train, encoded_test



# Applying target encoding
data_1, test_1 = target_encode(data_1, test_1, target_col='num_sold', categorical_cols=categorical_columns)


# Creating an instance of OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fitting the encoder on the 'country' column of data_1
encoder.fit(data_1[['country']])

# Applying the encoding to both data_1 and test_1
data_1_enc= pd.DataFrame(encoder.transform(data_1[['country']]), 
                              columns=encoder.get_feature_names_out(['country']),
                              index=data_1.index)
test_1_enc = pd.DataFrame(encoder.transform(test_1[['country']]), 
                              columns=encoder.get_feature_names_out(['country']),
                              index=test_1.index)

# Dropping the original 'country' column and concatenate the encoded columns
data_1 = pd.concat([data_1.drop(columns=['country']), data_1_enc], axis=1)
test_1 = pd.concat([test_1.drop(columns=['country']), test_1_enc], axis=1)


# Log-transform the target variable
data_1['num_sold'] = np.log1p(data_1['num_sold'])


# Splitting data into features and target
X = data_1.drop(columns=['id', 'date', 'num_sold'])
y = data_1['num_sold']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


# Defining models to test
models = {
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=seed),
    'LightGBM': lgb.LGBMRegressor(objective='regression', random_state=seed),
    'Random Forest': RandomForestRegressor(random_state=seed),
    'Gradient Boosting': GradientBoostingRegressor(random_state=seed),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=seed)
}

# Dictionary to store results
results = {}

# Training and evaluating each model
for model_name, model in models.items():
    print(f"Training and evaluating: {model_name}")
    
    # Training the model
    model.fit(X_train, y_train)
    
    # Predicting on the training set
    y_train_pred_log = model.predict(X_train)  # Log-transformed predictions
    y_train_pred = np.expm1(y_train_pred_log)  # Reverse log-transformation
    y_train_actual = np.expm1(y_train)         # Reverse log-transformation of actual values
    
    # Predicting on the test set
    y_test_pred_log = model.predict(X_test)   # Log-transformed predictions
    y_test_pred = np.expm1(y_test_pred_log)   # Reverse log-transformation
    y_test_actual = np.expm1(y_test)          # Reverse log-transformation of actual values
    
    # Calculating metrics for training set
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    
    # Calculating metrics for test set
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    
    # Storing the results
    results[model_name] = {
        'MAE (Train)': mae_train,
        'RMSE (Train)': rmse_train,
        'MAE (Test)': mae_test,
        'RMSE (Test)': rmse_test}
    

clear_output()


# Displaying the results
clear_output()
print("Model Performance Metrics:")
for model, metrics in results.items():
    print(f"{model}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")
    print()


# Objective functions using RMSE
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 500]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.1),
    }
    model = XGBRegressor(random_state=seed, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 500]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 31, 100),
    }
    model = LGBMRegressor(random_state=seed, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_categorical('iterations', [100, 200, 500]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01),
        'depth': trial.suggest_int('depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.1),
    }
    model = CatBoostRegressor(random_state=seed, silent=True, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

# Running Optuna for each model
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=25)


study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=25)


study_catboost = optuna.create_study(direction='minimize')
study_catboost.optimize(objective_catboost, n_trials=25)


clear_output()


# Best Parameters 
print(f"Best parameters for XGBoost: {study_xgb.best_trial.params}")
print(f"Best parameters for LightGBM: {study_lgbm.best_trial.params}")
print(f"Best parameters for CatBoost: {study_catboost.best_trial.params}")


# Splitting data into features and target
X = data_1.drop(columns=['id', 'date', 'num_sold'])
y = data_1['num_sold']
X_test = test_1.drop(columns=['id', 'date'])


# Best parameters for XGBoost
xgb_best = XGBRegressor(
    learning_rate=0.09, 
    max_depth=8, 
    n_estimators=500, 
    subsample=1.0, 
    random_state=seed)

# Fitting the model on the entire training dataset
xgb_best.fit(X, y)

# Making predictions on the test dataset
y_pred_xgb = xgb_best.predict(X_test)

# Reverting the log transformation 
y_pred_xgb_original = np.expm1(y_pred_xgb)


# Preparing data for submission
test_xgb = test_1.copy()
test_xgb['num_sold'] = y_pred_xgb_original
test_xgb = test_xgb[['id','num_sold']]


test_xgb.to_csv('submission_1.csv', index=False)


# Best parameters for LightGBM
lgbm_best = LGBMRegressor(
    learning_rate=0.1,  
    max_depth=7,        
    n_estimators=500,   
    num_leaves=64,      
    random_state=seed)

# Fitting the model on the entire training dataset
lgbm_best.fit(X, y)

# Making predictions on the test dataset
y_pred_lgbm = lgbm_best.predict(X_test)

# Reverting the log transformation 
y_pred_lgbm_original = np.expm1(y_pred_lgbm)

clear_output()


# Preparing data for submission
test_lgbm = test_1.copy()
test_lgbm['num_sold'] = y_pred_lgbm_original
test_lgbm = test_lgbm[['id','num_sold']]


test_lgbm.to_csv('submission_2.csv', index=False)


test_lgbm


# Best parameters for CatBoost
catboost_best = CatBoostRegressor(
    iterations=200,
    learning_rate=0.09,
    depth=10,
    subsample=0.8,
    random_state=seed,)

# Fitting the model on the entire training dataset
catboost_best.fit(X, y)

# Making predictions on the test dataset
y_pred_catboost = catboost_best.predict(X_test)

# Reverting the log transformation 
y_pred_catboost_original = np.expm1(y_pred_catboost)


clear_output()


# Preparing data for submission
test_cb = test_1.copy()
test_cb['num_sold'] = y_pred_catboost_original
test_cb = test_cb[['id','num_sold']]


test_cb.to_csv('submission_3.csv', index=False)


# Combining predictions into a DataFrame
predictions = pd.DataFrame({
    'XGBoost': y_pred_xgb_original,
    'LightGBM': y_pred_lgbm_original,
    'CatBoost': y_pred_catboost_original
})

# Calculating the correlation matrix
correlation_matrix = predictions.corr()

# Displaying the correlation matrix
print("Correlation between model predictions:")
print(correlation_matrix)


# Defining the base models
base_models = [
    ('xgb', XGBRegressor(
        learning_rate=0.09, 
        max_depth=8, 
        n_estimators=500, 
        subsample=1.0, 
        random_state=seed
    )),
    ('lgbm', LGBMRegressor(
        learning_rate=0.1, 
        max_depth=7, 
        n_estimators=500, 
        num_leaves=64, 
        random_state=seed
    )),]


# Defining the meta-model (for the stacking)
meta_model = LinearRegression()

# Creating the stacking model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Training the stacking model on the entire training dataset
stacking_model.fit(X, y)

# Making predictions on the test dataset
y_test_pred_stack = stacking_model.predict(X_test)

# Reverting the log transformation 
y_test_pred_stack_original = np.expm1(y_test_pred_stack)

clear_output()


# Preparing data for submission
test_stacked = test_1.copy()
test_stacked['num_sold'] = y_test_pred_stack_original
test_stacked = test_stacked[['id','num_sold']]


test_stacked


test_stacked.to_csv('submission_4.csv', index=False)

