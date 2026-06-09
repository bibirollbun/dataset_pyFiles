import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Importing sklearn classes
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
# Importing models
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import BayesianRidge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# Import Tensorflow and Keras
from tensorflow import keras
from keras import layers, losses, metrics
# Score
from sklearn.metrics import mean_squared_error
# Plots packages
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


rows, columns  = df.shape
print(f'{rows = }, {columns = }')
df.head()


df.dtypes


df.describe()


# Correlations among numerical features.
df[[col for col in df if df[col].dtype == 'float64']].corr(method = 'spearman')


# Looking for null values.
df.isnull().sum()


# Dropping target values ('Listening_Time_minutes') and 'id' from X train dataset.
#X = df.drop(['id','Listening_Time_minutes'], axis = 1)
X = df.drop('Listening_Time_minutes', axis = 1)
y = df['Listening_Time_minutes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Define a custom transformer to handle 'Episode_Title' conversion into a numerical feature and to drop 'id' column.
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_convert = 'Episode_Title', column_to_drop = 'id'):
        self.column_to_convert = column_to_convert
        self.column_to_drop = column_to_drop
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y = None) -> None:       
        self.is_fitted = True
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    
        if not self.is_fitted:
            raise ValueError('You must call fit() method before calling transform()')
        try:
            # Convert column in a numerical feature
            X[self.column_to_convert] = X[self.column_to_convert].apply(lambda cell: cell.strip().split(' ')[1])
            X[self.column_to_convert] = X[self.column_to_convert].astype('float64')
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
simple_imputer = SimpleImputer(missing_values = np.nan , strategy = 'median')
# 2. Instanciate scaler to rescale numerical features.
std_scaler = StandardScaler()
# 3. Create a Pipeline for these two process.
numerical_pipe = Pipeline([('imputer', simple_imputer),
                        ('scaler', std_scaler),
                          ])
# Categorical features:
# 1. Create one-hot encoder to handle categorical features
ohe = OneHotEncoder(drop='first', sparse=False)
# 2. Put it in a pipeline
categorical_pipe = Pipeline([
                      ('encoder', ohe)
                        ])

# Manage the last two pipelines within ColumnTransformer
ct = ColumnTransformer([
                    ('numerical_pipe', numerical_pipe, selector(dtype_include = "float64")),
                    ('categorical_pipe',categorical_pipe, selector(dtype_exclude = "float64")),
                    ])
# Create a preprocess pipe to manage all transformer together
preprocess_pipe = Pipeline([('custom_transfomer', custom_transformer),
                            ('column_transformer', ct)])

preprocess_pipe


X_train_data = preprocess_pipe.fit_transform(X_train) # Type: numpy ndarray
print(X_train_data[:, :10],'\n', X_train_data.shape)

X_val_data = preprocess_pipe.transform(X_val) # Type: pandas DataFrame
print(X_val_data[:, 10], '\n', X_val_data.shape)


# Retrieve columns names after processing data.
processed_cols = [processed_col.split('__')[1] for processed_col in preprocess_pipe[1].get_feature_names_out()]
# Remove spaces in columns names because they conflicts with LGBMRegressor.
processed_cols = list(map(lambda x: x.replace(' ', '_'), processed_cols))
# Convert the ndarray data into dataframe format.
X_train_processed = pd.DataFrame(X_train_data, columns = processed_cols)
X_val_processed = pd.DataFrame(X_val_data, columns = processed_cols)


regressors_result = {'Regressor': [], 'RMSE_train': [], 'RMSE_validation': []}


# List of categorical processed features. This is needed for HistGradientBoostingRegressor.
categorical_features = processed_cols[5:]


estimators = [
    HistGradientBoostingRegressor(
            max_iter= 100,
            max_depth= None,
            learning_rate= 0.01,
            loss= 'squared_error',
            categorical_features = categorical_features
    ),
    RandomForestRegressor(
            n_estimators = 100,
            criterion= 'squared_error',
    ),
    AdaBoostRegressor(
            n_estimators=100,
    ),
    BayesianRidge(),
    CatBoostRegressor(
           loss_function='RMSE' 
    ),
    LGBMRegressor(
            n_estimators = 100,  
    ),
    XGBRegressor(),
]



for regressor in estimators:
    try:
        # Fit regressor on train dataset
        regressor.fit(X_train_processed, y_train)
        y_pred_train = regressor.predict(X_train_processed)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        # Predict values and calculate root mean square error on validation dataset
        y_pred_val = regressor.predict(X_val_processed)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
        # Print out results
        regressor_name = regressor.__class__.__name__
        print(regressor_name)
        print(f'{rmse_train = }, {rmse_val = } \n\n')
        # Store values in dict
        regressors_result['Regressor'].append(regressor_name)
        regressors_result['RMSE_train'].append(rmse_train)
        regressors_result['RMSE_validation'].append(rmse_val)
    except Exception as e:
        print(f"Exception {e} occurred: {e.args}.")
        continue


regressors_df = pd.DataFrame(regressors_result, columns = regressors_result.keys())

regressors_df


input_shape = [X_train_data.shape[1]]
rmse_metric = keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
rmse_loss = keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error", dtype=None)


hidden_neurons = int(input_shape[0]/2)
model = keras.Sequential([
                layers.Input(shape = input_shape),
                layers.Dense(hidden_neurons, activation = 'relu'),
                layers.Dense(1, activation = 'relu')
                ])

model.compile(
        optimizer = 'adam',
        loss = rmse_loss,
        metrics = [rmse_metric]
)

early_stopping = keras.callbacks.EarlyStopping(
                patience = 10,
                min_delta = 0.001,
                restore_best_weights = True
)

model.summary()


history = model.fit(
    X_train_data, y_train,
    validation_data = (X_val_data, y_val),
    batch_size = 512,
    epochs = 100,
    callbacks = [early_stopping],
    verbose = 0
)


history_df = pd.DataFrame(history.history)

# Loss plot
loss_plot = history_df.loc[:, ['loss', 'val_loss']].plot()
loss_plot.set_title('Loss plots for train and validation sets')
loss_plot.set_ylim(170, 200)
loss_plot.legend(['Train dataset', 'Validation dataset'], title = 'Loss (mean squared error):')

# RMSE plot
rmse_plot = history_df.loc[:, ['root_mean_squared_error', 'val_root_mean_squared_error']].plot()
rmse_plot.set_title('RMSE plots for train and validation sets')
rmse_plot.set_ylim(13, 14)
rmse_plot.legend(['Train dataset', 'Validation dataset'], title = 'RMSE:')

rmse_val = history_df['val_root_mean_squared_error'].iloc[-1]
rmse = history_df['root_mean_squared_error'].iloc[-1]

print(f'RMSE for train dataset: {rmse:.2f}')

print(f'RMSE for validation dataset: {rmse_val:.2f}')


# Save this before passing test_df to pipeline
submission_ids = test_df['id']


e2e_pipe = Pipeline([('preprocess_pipe', preprocess_pipe), 
                     ('regressor',CatBoostRegressor(loss_function='RMSE'))])

e2e_pipe


# Fit parameters on the entire dataframe
e2e_pipe.fit(X, y)


# Predict Listening time minutes for test dataset.
y_test = e2e_pipe.predict(test_df)


# Store them into Dataframe
submission_df = pd.DataFrame({'id': submission_ids, 'Listening_Time_minutes': y_test})
submission_df.head()


submission_df.to_csv(path_or_buf = '/kaggle/working/submission.csv', index = False)

