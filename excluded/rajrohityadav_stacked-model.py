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


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, SGDRegressor, LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
train.shape, test.shape


train['country'].unique()


def preprocess_data(df, is_train=True):
    df['date'] = pd.to_datetime(df['date'])

    # Extract basic date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['quarter'] = df['date'].dt.quarter
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) 
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7
    df['cos_year'] = np.cos(df['year'] * (2 * np.pi) / 100)
    df['sin_year'] = np.sin(df['year'] * (2 * np.pi) / 100)

    # Add weekend flag
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # --- New Time Features ---
    
    # 2. Month boundaries
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # # 3. Country-specific holidays (Key Addition!)
    # def _get_holiday_flag(row):
    #     try:
    #         country_holidays = holidays.CountryHoliday(row['country'])
    #         return 1 if row['date'] in country_holidays else 0
    #     except:
    #         return 0  # Fallback for unsupported countries
    # df['is_holiday'] = df.apply(_get_holiday_flag, axis=1)

    # Interaction Features
    df['country_product'] = df['country'] + "_" + df['product']
    df['store_product'] = df['store'] + "_" + df['product']

    # One-Hot Encoding for categorical features
    categorical_cols = ['country', 'store', 'product', 'country_product', 'store_product', 'month_name', 'day_of_week']
    onehot_encoder = OneHotEncoder(sparse=False)  # Use drop='first' to avoid multicollinearity

    # Fit and transform the categorical features
    onehot_encoded = pd.DataFrame(
        onehot_encoder.fit_transform(df[categorical_cols]),
        columns=onehot_encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    # Drop original categorical columns and add one-hot encoded columns
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, onehot_encoded], axis=1)

    # Drop rows with missing target in training data
    if is_train:
        df = df.dropna(subset=['num_sold'])

    return df



# Apply preprocessing
train = preprocess_data(train)
test = preprocess_data(test, is_train=False)

# Extract test IDs for submission
test_ids = test['id']
train = train.drop(columns=['id', 'date'])
test = test.drop(columns=['id', 'date'])


train.shape


train.info()


test.info()


# Dictionary to store models and results for each country
country_models = {}
country_results = {}


# Identify the one-hot encoded country columns
country_columns = ['country_Canada', 'country_Finland', 'country_Italy', 'country_Kenya', 'country_Norway', 'country_Singapore']



# best_ridge_model = Ridge(alpha=0.0001)
# best_sgd_model = SGDRegressor(alpha=1e-05, learning_rate='adaptive', max_iter=10000, penalty='elasticnet')
# best_rf_model = RandomForestRegressor(max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=150)
# best_gbr_model = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=150)

best_ridge_model = Ridge(alpha=0.0001)
best_sgd_model = SGDRegressor()
best_rf_model = RandomForestRegressor()
best_gbr_model = GradientBoostingRegressor()


# Iterate over each country column
for country_col in country_columns:
    # Extract the country name from the column
    country_name = country_col.split('_')[1]
    print(f"Training model for {country_name}...")

    # Filter data for the current country
    country_data = train[train[country_col] == 1]
    X_country = country_data.drop(columns=['num_sold'] + country_columns)
    y_country = country_data['num_sold']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_country, y_country, test_size=0.2, random_state=42)

    # Train a Random Forest model
    # model = LinearRegression()
    # model = best_rf_model
    model = StackingRegressor(
        estimators=[
            ('ridge', best_ridge_model),
            ('sgd', best_sgd_model),
            ('rf', best_rf_model),
            ('gbr', best_gbr_model)
        ],
        final_estimator=best_rf_model
    )
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_val_pred)

    # Store the model and results
    country_models[country_name] = model
    country_results[country_name] = {'MAPE': mape}

    print(f"MAPE for {country_name}: {mape:.4f}")


print("\nModel Performance by Country:")
for country_name, result in country_results.items():
    print(f"{country_name}: MAPE = {result['MAPE']:.4f}")


round_predictions = True  # Toggle rounding


# Create submission with preserved 'id'
submission = pd.DataFrame({'id': test_ids, 'num_sold': 0})

# Iterate over each country model to predict the `num_sold`
for country_col in country_columns:
    country_name = country_col.split('_')[1]
    print(f"Predicting for {country_name}...")

    country_test_rows = test[test[country_col] == 1]

    if not country_test_rows.empty:
        X_country_test = country_test_rows.drop(columns=country_columns)
        model = country_models[country_name]
        y_country_pred = model.predict(X_country_test)

        if round_predictions:
            y_country_pred = np.round(y_country_pred)

        submission.loc[country_test_rows.index, 'num_sold'] = y_country_pred

# Ensure predictions are integers
submission['num_sold'] = submission['num_sold']*1.05  #.astype(int)

# Save the submission file
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")



out=pd.read_csv("/kaggle/working/submission.csv")


out.head()

