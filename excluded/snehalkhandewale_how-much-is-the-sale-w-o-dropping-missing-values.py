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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


def fill_missing_values_by_country_product(data, country):
    country_data = data[data['country']== country]
    product_means = country_data.groupby('product')['num_sold'].mean()

    def fill_value(row):
        if pd.isnull(row['num_sold']) and row['country']==country:
            return product_means.get(row['product'], 0)
        return row['num_sold']
    return data.apply(fill_value, axis =1)





# Fill missing values for Canada and Kenya in the training dataset
train_data['num_sold'] = fill_missing_values_by_country_product(train_data, 'Canada')
train_data['num_sold'] = fill_missing_values_by_country_product(train_data, 'Kenya')

# Check remaining missing values
remaining_missing = train_data['num_sold'].isnull().sum()
print(f"Remaining missing values: {remaining_missing}")



train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])


cat_cols = train_data.select_dtypes(include=['object']).columns
print("Categorical Columns:", cat_cols)


def create_individual_lag_features(df, group_cols, target_col, lags):
    """
    Create lag features for the target column grouped by group_cols.
    """
    for lag in range(1, lags + 1):
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    return df


# Add placeholder `num_sold` for test dataset
test_data['num_sold'] = 0



# Specify grouping columns
group_cols = ['country', 'store', 'product']

# Create lag features for train and test datasets
train_data = create_individual_lag_features(train_data, group_cols, 'num_sold', lags=7)
test_data = create_individual_lag_features(test_data, group_cols, 'num_sold', lags=7)

# Drop placeholder column
test_data = test_data.drop(columns=['num_sold'], errors='ignore')



encoded_train = pd.get_dummies(train_data, columns=['country', 'store', 'product'], drop_first=True)
encoded_test = pd.get_dummies(test_data, columns=['country', 'store', 'product'], drop_first=True)



# Define train-validation split dates
train_split_date = '2015-01-01'
val_split_date = '2017-01-01'

# Training set: Data before 2015
train_set = encoded_train[encoded_train['date'] < train_split_date]

# Validation set: Data from 2015 to 2016
val_set = encoded_train[(encoded_train['date'] >= train_split_date) & (encoded_train['date'] < val_split_date)]





# Extract temporal features from `date` AFTER splitting
for df in [train_set, val_set, encoded_test]:
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)




val_set.describe()




# Define features and target for training
X_train = train_set.drop(columns=['num_sold'], errors='ignore')
y_train = train_set['num_sold']

X_val = val_set.drop(columns=['num_sold'], errors='ignore')
y_val = val_set['num_sold']

# Drop the `date` column from the datasets
X_train = X_train.drop(columns=['date'], errors='ignore')
X_val = X_val.drop(columns=['date'], errors='ignore')
encoded_test = encoded_test.drop(columns=['date'], errors='ignore')



from xgboost import XGBRegressor

# Initialize the XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=500,       # Number of trees
    learning_rate=0.05,     # Step size shrinkage
    max_depth=6,            # Maximum depth of trees
    subsample=0.8,          # Subsample ratio of rows
    colsample_bytree=0.8,   # Subsample ratio of columns
    random_state=42         # For reproducibility
)

# Train the model on training data
xgb_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = xgb_model.predict(X_val)



from sklearn.metrics import mean_absolute_percentage_error

# Calculate MAPE for validation set
mape_val = mean_absolute_percentage_error(y_val, y_val_pred)
print(f"Validation MAPE: {mape_val:.2%}")



# Prepare test data
X_test = encoded_test

# Predict on the test dataset
test_predictions = xgb_model.predict(X_test)

# Prepare submission file
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission = pd.DataFrame({'id': test_data['id'], 'num_sold': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")





submission.head()




