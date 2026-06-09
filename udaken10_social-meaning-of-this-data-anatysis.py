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

def analyze_homelessness_correlation(file_path='/kaggle/input/california-homelessness-prediction-challenge/train.csv'):
    """
    Loads the homelessness dataset, calculates the correlation of all features
    with the HOMELESS_RATE, and prints the sorted results.

    """
    try:
        # Load the uploaded CSV file
        df = pd.read_csv(file_path)

        # Ensure HOMELESS_RATE is in the dataframe
        if 'HOMELESS_RATE' not in df.columns:
            print("Error: 'HOMELESS_RATE' column not found in the file.")
            return

        # Calculate the correlation matrix
        correlation_matrix = df.corr(numeric_only=True)

        # Get the correlations with the target variable 'HOMELESS_RATE'
        homeless_rate_correlation = correlation_matrix['HOMELESS_RATE'].sort_values(ascending=False)

        # Print the results
        print("--- Correlation with HOMELESS_RATE ---")
        print(homeless_rate_correlation)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Execute the analysis
analyze_homelessness_correlation()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_and_evaluate_model(file_path='/kaggle/input/california-homelessness-prediction-challenge/train.csv'):
    """
    Loads data, trains a Gradient Boosting Regressor model,
    and evaluates its performance using multiple metrics.

    """
    try:
        # 1. Load Data
        df = pd.read_csv(file_path)

        # 2. Define Features (X) and Target (y)
        # Drop the target variable and the non-numeric ID
        X = df.drop(['HOMELESS_RATE', 'ID'], axis=1)
        y = df['HOMELESS_RATE']

        # 3. Split Data into Training and Testing sets
        # 80% for training, 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Initialize and Train the Model
        # GradientBoostingRegressor is a powerful model for tabular data
        print("Training the Gradient Boosting model...")
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        print("Training complete.")

        # 5. Make Predictions on the Test Set
        y_pred = model.predict(X_test)

        # 6. Evaluate the Model
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n--- Model Evaluation Results ---")
        print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
        print(f"MAE (Mean Absolute Error):     {mae:.6f}")
        print(f"RÂ² Score (Coefficient of Determination): {r2:.6f}")
        print("---------------------------------")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Execute the training and evaluation
train_and_evaluate_model()


import pandas as pd

def analyze_high_homeless_rate_areas(file_path='/kaggle/input/california-homelessness-prediction-challenge/train.csv'):
    """
    Identifies areas with the highest homeless rates and analyzes their
    common characteristics compared to the overall average.

    """
    try:
        # Load the data
        df = pd.read_csv(file_path)

        # Define what constitutes a "high" homeless rate.
        # Let's use the top 10% of areas as our sample group.
        quantile_threshold = df['HOMELESS_RATE'].quantile(0.90)
        high_rate_df = df[df['HOMELESS_RATE'] >= quantile_threshold]

        if high_rate_df.empty:
            print("No areas found in the top 10% quantile. The threshold might be too high or the data is uniform.")
            return

        # Drop non-feature columns for analysis
        features_df = df.drop(columns=['ID', 'HOMELESS_RATE'])
        high_rate_features_df = high_rate_df.drop(columns=['ID', 'HOMELESS_RATE'])

        # Calculate the average characteristics for the high-rate group and the overall dataset
        high_rate_avg = high_rate_features_df.mean()
        overall_avg = features_df.mean()

        # Create a comparison dataframe to see the differences clearly
        comparison_df = pd.DataFrame({
            'High-Rate Areas Average': high_rate_avg,
            'Overall Average': overall_avg
        })
        comparison_df['Difference'] = comparison_df['High-Rate Areas Average'] - comparison_df['Overall Average']
        comparison_df['Ratio (High/Overall)'] = comparison_df['High-Rate Areas Average'] / comparison_df['Overall Average']

        # Sort by the ratio to see the most pronounced differences
        sorted_comparison = comparison_df.sort_values(by='Ratio (High/Overall)', ascending=False)

        print(f"--- Characteristics of Areas with the Highest Homelessness Rates (Top 10%) ---")
        print(f"\nThese areas, when compared to the average, have notably different demographics.")
        print("Below are the most significant differences, sorted by how much higher the feature is compared to the average:\n")
        print(sorted_comparison)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the analysis
analyze_high_homeless_rate_areas()


features = ['HOMELESS_RATE','RACE_BLACK_NH_PCT', 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT', 'AGE_25_34_PCT','FAMILY_HH_CHILD_LT18_PCT', 'VETERAN_POP_PCT', 'FAMILY_HH_TOTAL'] 


train_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')
sub = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def RMSE_GB(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_LR(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_svr(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_RF(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_CatBoost(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = CatBoostRegressor(silent=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_LGBM(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = LGBMRegressor(verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_XGB(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


df = train_df[features]

print('GB ', RMSE_GB(df))
print('LR ', RMSE_LR(df))
print('SVR ',RMSE_svr(df))
print('RF ' ,RMSE_RF(df))
print('CB', RMSE_CatBoost(df))
print('LGB', RMSE_LGBM(df))
print('XGB', RMSE_XGB(df))


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostRegressor

X = train_df[features].drop('HOMELESS_RATE', axis = 1)
y = train_df['HOMELESS_RATE']

model_cat = CatBoostRegressor(silent=True)

param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'depth': [7, 8, 9, 10],
    'l2_leaf_reg': [1, 3, 5]
}

grid_search = GridSearchCV(estimator=model_cat, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(best_params)
print(best_model)

best_model.fit(X, y)

y_pred = best_model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))

rmse


test_features =  ['RACE_BLACK_NH_PCT', 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT', 'AGE_25_34_PCT','FAMILY_HH_CHILD_LT18_PCT', 'VETERAN_POP_PCT', 'FAMILY_HH_TOTAL'] 
test_df[test_features].isnull().sum()


prediction = best_model.predict(test_df[test_features])

pred_positive = []

for i in prediction:
    if i < 0:
        i = -i
    pred_positive.append(i)

sub['HOMELESS_RATE'] = pred_positive
# sub.to_csv("submission.csv", index=False)     
sub_cat = sub.copy()


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error

X = train_df[features].drop('HOMELESS_RATE', axis=1)
y = train_df['HOMELESS_RATE']

model_xgb = XGBRegressor()

param_grid = {
    'n_estimators': [100,150,200,250,300],
    'learning_rate': [0.01,0.05, 0.1,0.2],
    'max_depth': [3,4,5,6,7],
    'min_child_weight': [1, 2, 3, 4, 5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(best_params)
print(best_model)

best_model.fit(X, y)

y_pred = best_model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))

print(rmse)




# test_df[features] = test_df[features].fillna(0)
test_df = test_df[test_features]

predictions = best_model.predict(test_df)


sub['HOMELESS_RATE'] = predictions
# sub.to_csv("submission.csv", index=False)
sub_xgb = sub.copy()
sub_xgb


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()

X = train_df[features].drop('HOMELESS_RATE', axis=1)
y = train_df['HOMELESS_RATE']

param_grid = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'positive': [True, False],
    'n_jobs': [-1, None]
}

grid_search = GridSearchCV(estimator=model_lr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(best_params)
print(best_model)

best_model.fit(X, y)

y_pred = best_model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))

print(rmse)



test_df = test_df[test_features]

predictions = best_model.predict(test_df)
predictions


sub['HOMELESS_RATE'] = predictions
sub.to_csv("submission_lr.csv", index=False)
sub_lr = sub.copy()


y = (sub_cat['HOMELESS_RATE'] + sub_lr['HOMELESS_RATE'] + sub_xgb['HOMELESS_RATE'])/3
sub['HOMELESS_RATE'] = y
# sub.to_csv("submission.csv", index=False)
sub


y = (sub_cat['HOMELESS_RATE']*99 + sub_xgb['HOMELESS_RATE'])/100
sub['HOMELESS_RATE'] = y
sub.to_csv("submission.csv", index=False)


y = sub_cat['HOMELESS_RATE']
sub['HOMELESS_RATE'] = y
# sub.to_csv("submission.csv", index=False)

