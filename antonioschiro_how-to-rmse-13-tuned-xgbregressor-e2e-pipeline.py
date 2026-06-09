# Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Importing sklearn classes
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector

# Models
from xgboost import XGBRegressor
# Score
from sklearn.metrics import mean_squared_error
# Defining Root mean squared error since it's not present in this version of sklearn.
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
# Tuning hyperparameters libraries
import optuna

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


# Take a first look to train dataset:
rows, columns = df.shape
print(f'Train dataset has:\n{rows = }\n{columns = }')
df.head()


# Check columns dtypes:
dtypes_df = pd.DataFrame(df.dtypes)
dtypes_df.rename(columns = {0: 'dtype'})


# Looking at some stats:
df.describe()


# Let's check correlations among numerical features.
# I suggest to use Spearman correlation instead of Pearson correlation since I prefer to catch any monotonic relationship among each pair of variables.
df[[col for col in df if df[col].dtype == 'float64']].corr(method = 'spearman')


# Looking for null values.
print("Null values found: ")
df.isnull().sum()


# Dropping target values ('Listening_Time_minutes') and 'id' from X train dataset.
X = df.drop('Listening_Time_minutes', axis = 1)
y = df['Listening_Time_minutes']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Define a custom transformer to drop 'id' column.
# Defining a custom transformer is useful to preprocess data within a pipeline.
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_drop = 'id'):
        self.column_to_drop = column_to_drop
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y = None) -> None:       
        self.is_fitted = True
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError('You must call fit() method before calling transform()')
        try:
            # Drop unnecessary column
            X.drop(self.column_to_drop, axis = 1, inplace = True)
            return X
        except Exception as e:
            print(f'An exception occured: {e}. More info: {e.args[0]}')
            raise ValueError('Check your input data.')
    
    def fit_transform(self, X: pd.DataFrame, y = None) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


# Create a pipeline to clean data.
# Custom transformer:
# 1. Instanciate an object from CustomTransformer class.
custom_transformer = CustomTransformer()
# Numerical features:
# 1. Instanciate imputer to handle missing values for numerical values.
simple_imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
# 2. Create a Pipeline for these two process.
numerical_pipe = Pipeline([('imputer', simple_imputer),
                          ])
# Categorical features:
# 1. Instanciate target encoder to handle categorical features
te = ce.TargetEncoder() 
# 2. Put it in a pipeline
categorical_pipe = Pipeline([('encoder', te)])

# Manage the last two pipelines within ColumnTransformer
ct = ColumnTransformer([
                    ('numerical_pipe', numerical_pipe, selector(dtype_include = "float64")),
                    ('categorical_pipe',categorical_pipe, selector(dtype_exclude = "float64")),
                    ])
# Now I define and I'll apply it on all features after ColumnTransformer since it will convert categoricals into numericals.
std_scaler = StandardScaler()
# Create a preprocess pipe to manage all transformer together
preprocess_pipe = Pipeline([('custom_transfomer', custom_transformer),
                            ('column_transformer', ct),
                            ('std_scaler', std_scaler)
                           ])
preprocess_pipe


X_train_data = preprocess_pipe.fit_transform(X_train, y_train) # Type: numpy ndarray. You need y_train since you're using TargetEncoder.
print('Train data processed (ndarray): \n')
print(X_train_data,'\n', X_train_data.shape)

X_val_data = preprocess_pipe.transform(X_val) # Numpy ndarray.
print('Validation data processed (ndarray): \n')
print(X_val_data, '\n', X_val_data.shape)


# Retrieve columns names after processing data.
processed_cols = [processed_col.split('__')[1] for processed_col in preprocess_pipe[1].get_feature_names_out()]
# Remove spaces in columns names because they could be problematic for some regressors such as LGBMRegressor.
processed_cols = list(map(lambda x: x.replace(' ', '_'), processed_cols))
# Convert the ndarray data into dataframe format.
X_train_processed = pd.DataFrame(X_train_data, columns = processed_cols)
X_val_processed = pd.DataFrame(X_val_data, columns = processed_cols)


# Take a look to validation dataset:
print('Validationd dataset: \n')
X_val_processed.head()


def objective(trial, X_train = X_train_processed, y_train = y_train, X_test = X_val_processed, y_test = y_val):
    # Defining parameters:
    param = {}
    param['eta'] = trial.suggest_float("eta", 0.1, 1.)
    param['max_depth'] = trial.suggest_int('max_depth', 3, 11)
    param['n_estimators'] = trial.suggest_int('n_estimators', 100, 3000)
    param['alpha'] = trial.suggest_float('alpha', 0., 10.)
    param['lambda'] = trial.suggest_float('lambda', 0., 10.)
    param['min_child_weight'] = trial.suggest_int('min_child_weight', 1,10)
    param['subsample'] = trial.suggest_float('subsample', 0.3, 1.0)
    param['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.3, 1.)
    param['eval_metric'] = 'rmse'
    param['seed'] = 0
    # Initialize and fit XGBRegressor.
    regressor = XGBRegressor(**param)
    regressor.fit(X_train.copy(), y_train.copy(),
                  eval_set=[(X_test.copy(), y_test.copy())],
                  early_stopping_rounds=100)
    # Calculate RMSE
    y_pred = regressor.predict(X_test)
    rmse_test = rmse(y_test, y_pred)
    return rmse_test

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, show_progress_bar= True)


print('Best result found:')
print(study.best_params)
print(study.best_trial.value)


# Retrain regressor on the entire dataset.
# Here's the regressor feed with the best params I found:
tuned_regressor = XGBRegressor(**study.best_params, eval_metric = 'rmse', seed = 0)


# Create an end-to-end pipeline that includes the fine tuned regressor:
e2e_pipe = Pipeline([('preprocess_pipe', preprocess_pipe), # pipeline for preprocess data
                     ('regressor',tuned_regressor)]) # regressor for fit and predict data
# Take a look to how it's structured:
e2e_pipe


# Fit parameters on the entire dataframe.
e2e_pipe.fit(X, y)


# Save this before passing test_df to pipeline since this column
# will be dropped in the pipeline and I need them to identify the target values.
submission_ids = test_df['id']


# Predict Listening time minutes for test dataset.
y_test = e2e_pipe.predict(test_df)


# Finally, I can create the submission file.
# Store results into Dataframe.
submission_df = pd.DataFrame({'id': submission_ids, 'Listening_Time_minutes': y_test})
submission_df.head()


# Output a CSV file.
submission_df.to_csv(path_or_buf = '/kaggle/working/submission.csv', index = False)

