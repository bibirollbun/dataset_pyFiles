!pip uninstall -y scikit-learn
!pip install scikit-learn==1.3.1
!pip install category_encoders


import pandas as pd
folder_path='/kaggle/input/playground-series-s5e1/'
try:
  dictionary_df=pd.read_csv(f"{folder_path}data_dictionary.csv")
except:
  dictionary_df=None




train_df=pd.read_csv(f"{folder_path}train.csv")
train_df=train_df.sample(frac=1, random_state=42)
train_df.reset_index(drop=True, inplace=True)
print(train_df.info())
print(train_df.head())
print(train_df.describe())
for column in train_df.columns:
  unique_values = train_df[column].value_counts()
  print(column)

  print(unique_values)
  null_count_train = train_df[column].isnull().sum()
  print(f"Null: {null_count_train}")

  print("-" * 20)





test_df=pd.read_csv(f"{folder_path}test.csv")

print(test_df.info())
print(test_df.head())
print(test_df.describe())
for column in test_df.columns:
  unique_values = test_df[column].value_counts()
  print(unique_values)
  print("-" * 20)


# prompt: show difference in columns between train and test

set(train_df.columns) - set(test_df.columns)


# prompt: list number and object columns from train

import pandas as pd

# Assuming train_df is already loaded as in your previous code
# train_df = pd.read_csv(...)

numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
object_cols = train_df.select_dtypes(include=['object']).columns.tolist()

print("Numeric columns:")
print(numeric_cols)
print("\nObject columns:")
object_cols



target_columns=['num_sold']



analisys_df=train_df.copy()[:3000]




!pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'analisys_df', 'target_columns', and 'dictionary_df' are defined

for col in analisys_df.columns:
    if pd.api.types.is_numeric_dtype(analisys_df[col]) and col not in target_columns:


        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

        # Density plot (kdeplot) on the first subplot
        sns.kdeplot(x=analisys_df[col], y=analisys_df[target_columns[0]], fill=True, cmap="Blues", thresh=0, ax=axes[0])
        axes[0].set_xlabel(col)
        axes[0].set_ylabel(target_columns[0])
        axes[0].set_title(f'{col} vs. {target_columns[0]} (Density)')
        axes[0].grid(True)

        # Simple scatter plot on the second subplot
        axes[1].scatter(analisys_df[col], analisys_df[target_columns[0]], alpha=0.5)
        axes[1].set_xlabel(col)
        axes[1].set_ylabel(target_columns[0])
        axes[1].set_title(f'{col} vs. {target_columns[0]} (Scatter)')
        axes[1].grid(True)

        plt.show()




onehot_columns=[]

caterogical_to_numeric_columns=[]
numeric_columns=[]
date_columns=['date']

drop_columns=[]

score_maps = {

}
catboost_columns=['date', 'country', 'store', 'product']
default_values_map={}



from datetime import date
# prompt: count number of days since 2018-1-1 to Policy Start Date enclose it in function that returns new df

import pandas as pd

def days_since_2018(df):
    # Convert 'Policy Start Date' to datetime objects
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])

    # Calculate the difference in days from 2018-01-01
    reference_date = pd.to_datetime('2018-01-01')
    output_df=pd.DataFrame()
    output_df['Days Since 2018'] = (df['Policy Start Date'] - reference_date).dt.days

    return output_df

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DaysSinceCalculator(BaseEstimator, TransformerMixin):
  def __init__(self, reference_date='2015-01-01'):
      self.reference_date = pd.to_datetime(reference_date)
      self.date_columns = None  # Initialize date_columns to None

  def fit(self, X, y=None):
      # Get the column names from the input DataFrame during fit
      self.date_columns = X.columns.tolist()

      return self

  def transform(self, X):
      # Convert date columns to datetime objects
      X_temp = X.copy()
      for col in self.date_columns:
          X_temp[col] = pd.to_datetime(X_temp[col])  # Handle non-date columns

      # Calculate the difference in days from the reference date
      output_df = pd.DataFrame(index=X.index) # Use original index
      for col in self.date_columns:
          try:  # Handle potential errors during date calculation
              output_df[f'Days Since Reference ({col})'] = (X_temp[col] - self.reference_date).dt.days
          except AttributeError:  # If column is not a datetime
              output_df[f'Days Since Reference ({col})'] = pd.NaT # Assign NaT for non-date columns

      return output_df



# prompt: create function that will assign numbers from score_maps to cateegories and return new df

def assign_scores(df, score_maps):
    """Assigns numerical scores to categorical features based on score_maps."""
    new_df = df.copy()  # Create a copy to avoid modifying the original DataFrame

    for column, score_map in score_maps.items():
        if column in new_df.columns: # Check if column exists in DataFrame
            new_df[column] = new_df[column].map(score_map).fillna(0) # Map and handle missing values

    return new_df




# prompt: using kfolds ccreate folds in irain_df, if there are missing target column value fill it with mean of the fold

from sklearn.model_selection import KFold

def create_folds(df, n_splits=5, target_column='num_sold'):
    """Creates k-folds with mean imputation for missing target values within each fold."""

    df['kfold'] = -1  # Initialize kfold column
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame

    kf = KFold(n_splits=n_splits, shuffle=False) # No need to shuffle again
    for fold, (train_index, val_index) in enumerate(kf.split(X=df)):
        df.loc[val_index, 'kfold'] = fold

        # Impute missing target values with the fold mean
        fold_mean = df.loc[train_index, target_column].mean()
        df.loc[val_index, target_column] = df.loc[val_index, target_column].fillna(fold_mean)

    return df

# Example usage:






from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
def default_value_filler(df, default_values_map=default_values_map):
    df_transformed = df.copy()
    for column, default_value in default_values_map.items():
        if column in df_transformed.columns:
            df_transformed[column] = df_transformed[column].fillna(default_value)
    return df_transformed


import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# Create a CatBoost encoder pipeline
catboost_pipeline = Pipeline([
    ('default_value_filler', FunctionTransformer(default_value_filler)),

    ('catboost', ce.cat_boost.CatBoostEncoder(verbose=1))
])
#works better without filter
catboost_pipeline = Pipeline([

    ('catboost', ce.cat_boost.CatBoostEncoder(verbose=1))
])


def add_is_not_null_column(X):
    """Adds a boolean column 'is_not_null_<column_name>' for each numeric column."""
    output_df = X.copy()
    for col in X.columns:
        output_df[f'is_not_null_{col}'] = X[col].notna().astype(int)
    return output_df


numeric_pipeline = Pipeline([
    ('add_is_not_null', FunctionTransformer(add_is_not_null_column)),
    ('imputer', SimpleImputer(strategy='median')),  # Add imputation step
    ('scaler', MinMaxScaler())
])
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert 'date' column to datetime objects
        datetime = pd.to_datetime(X['date'],format='%Y-%m-%d')
        X['year'] = datetime.dt.year
        X['month'] = datetime.dt.month
        X['day'] = datetime.dt.day
        X['day_of_week'] = datetime.dt.dayofweek
        X['day_int'] = datetime.astype('int')


        return X.drop(columns='date')

# Example usage within a scikit-learn pipeline:
date_pipeline = Pipeline([
    ('date_features', DateFeatureExtractor())
])


full_pipeline = ColumnTransformer([


('catboost',catboost_pipeline, catboost_columns),

    ('date', date_pipeline, date_columns)
])

fold_df=train_df.dropna()
transformed_array= full_pipeline.fit_transform(fold_df.drop(columns=target_columns+drop_columns), fold_df[target_columns])


X_train, X_val, y_train, y_val = train_test_split(transformed_array,fold_df[target_columns], test_size=0.2, random_state=42)


# prompt: generate lgb model. for this data X_train, X_val, y_train, y_val = train_test_split(transformed_array,train_df[target_columns], test_size=0.2, random_state=42)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Define the LightGBM parameters
params = {
    'objective': 'regression',  # Use regression for predicting 'efs'
    'metric': 'rmse',  # Use RMSE as evaluation metric
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the LightGBM model
num_round = 1000
lgb_model = lgb.train(params,
                      train_data,
                      num_boost_round=num_round,
                      valid_sets=[val_data]
                      )


# Make predictions
y_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

# Evaluate the model (example using RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse}")


# prompt: predict target for test df and save it to csv with ID and prediction column, dont inverse transform

# Transform the test data using the same pipeline
transformed_test_array = full_pipeline.transform(test_df.drop(columns=drop_columns))

# Make predictions on the transformed test data
test_predictions = lgb_model.predict(transformed_test_array)

# Create a DataFrame for submission
submission_df = pd.DataFrame({'id': test_df['id'], 'num_sold': test_predictions})

# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)

