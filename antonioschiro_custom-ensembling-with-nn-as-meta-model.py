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
# Importing base-models
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
# Importing meta-model
from tensorflow import keras
from keras import layers, losses, metrics
# Score
from sklearn.metrics import mean_squared_error
# Defining Root mean squared error since it's not present in this version of sklearn
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
# Tuning hyperparameters libraries
!pip install -q optuna-integration[catboost]
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.integration import CatBoostPruningCallback

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


# Dropping target values ('Listening_Time_minutes') and 'id' from X train dataset.
X = df.drop('Listening_Time_minutes', axis = 1)
y = df['Listening_Time_minutes']
# First split:
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42)
# Second split: I'm going to split the X_val, y_val dataset into two new dataset: (X_val, y_val) and (X_test, y_test).
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.3, random_state = 42)
# Take a look to test_size parameters in the previous two lines:
# With these values, I'll have the following proportion among the datasets:
print(f'Percentage of data for X_train: {X_train.shape[0]/X.shape[0]*100} %')
print(f'Percentage of data for X_val: {X_val.shape[0]/X.shape[0]*100} %')
print(f'Percentage of data for X_test: {X_test.shape[0]/X.shape[0]*100} %')


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
simple_imputer = SimpleImputer(missing_values = np.nan , strategy = 'median')
# 2. Create a Pipeline for these two process.
numerical_pipe = Pipeline([('imputer', simple_imputer),
                          ])
# Categorical features:
# 1. Instanciate target encoder to handle categorical features
te = ce.TargetEncoder() 
# 2. Put it in a pipeline
categorical_pipe = Pipeline([
                      ('encoder', te)
                        ])

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
                            ('std_scaler', std_scaler)])
preprocess_pipe


X_train_data = preprocess_pipe.fit_transform(X_train, y_train) # Type: numpy ndarray. You need y_train since you're using TargetEncoder.
print('Train data processed (ndarray): \n')
print(X_train_data,'\n', X_train_data.shape)

X_val_data = preprocess_pipe.transform(X_val) # Numpy ndarray.
print('Validation data processed (ndarray): \n')
print(X_val_data, '\n', X_val_data.shape)

X_test_data = preprocess_pipe.transform(X_test) # Same type here: numpy ndarray.
print('Test data processed (ndarray): \n')
print(X_test_data, '\n', X_test_data.shape)


# Retrieve columns names after processing data.
processed_cols = [processed_col.split('__')[1] for processed_col in preprocess_pipe[1].get_feature_names_out()]
# Remove spaces in columns names because they could be problematic for some regressors such as LGBMRegressor.
processed_cols = list(map(lambda x: x.replace(' ', '_'), processed_cols))
# Convert the ndarray data into dataframe format.
X_train_processed = pd.DataFrame(X_train_data, columns = processed_cols)
X_val_processed = pd.DataFrame(X_val_data, columns = processed_cols)
X_test_processed = pd.DataFrame(X_test_data, columns = processed_cols)


# Take a look, for example, to validation dataset:
print('Validationd dataset: \n')
X_val_processed.head()


def objective(trial, X_train = X_train_processed, y_train = y_train, X_test = X_val_processed, y_test = y_val):
    # Defining parameters:
    param = {}
    param['learning_rate'] = trial.suggest_float("learning_rate", 0.001, 0.7)
    param['depth'] = trial.suggest_int('depth', 5, 11)
    param['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 1.0, 5.0)
    param['min_child_samples'] = trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32])
    param['grow_policy'] = 'Depthwise'
    param['iterations'] = 2000
    param['use_best_model'] = True
    param['eval_metric'] = 'RMSE'
    param['od_type'] = 'iter'
    param['od_wait'] = 50
    param['random_state'] = 42
    param['logging_level'] = 'Silent'
    # Initialize and fit Catboost regressor.
    regressor = CatBoostRegressor(**param)
    regressor.fit(X_train.copy(), y_train.copy(),
                  eval_set=[(X_test.copy(), y_test.copy())],
                  early_stopping_rounds=100)
    # Calculate RMSE
    y_pred = regressor.predict(X_test)
    rmse_test = rmse(y_test, y_pred)
    return rmse_test

catboost_study = optuna.create_study(direction='minimize')
catboost_study.optimize(objective, n_trials=30, show_progress_bar= True)


print('Best result found for Catboost:')
print(catboost_study.best_params)
print(catboost_study.best_trial.value)


# Here's the regressor feed with the best params I found:
tuned_catboost = CatBoostRegressor(
    learning_rate = catboost_study.best_params['learning_rate'],
    depth = catboost_study.best_params['depth'],
    l2_leaf_reg = catboost_study.best_params['l2_leaf_reg'],
    min_child_samples = catboost_study.best_params['min_child_samples'],
    grow_policy = 'Depthwise',
    iterations = 2000,
    eval_metric = 'RMSE',
    od_type = 'Iter',
    od_wait = 50,
    random_state = 42,
    logging_level = 'Silent',
)


# Predict validation dataset.
tuned_catboost.fit(X_train_processed, y_train)
y_val_pred_catboost =tuned_catboost.predict(X_val_processed)

rmse_val_catboost = rmse(y_val, y_val_pred_catboost)

print(f"RMSE for Catboost on validation dataset is {rmse_val_catboost}")


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
    param['verbosity'] = 0
    # Initialize and fit XGBRegressor.
    regressor = XGBRegressor(**param)
    regressor.fit(X_train.copy(), y_train.copy(),
                  eval_set=[(X_test.copy(), y_test.copy())],
                  early_stopping_rounds=100)
    # Calculate RMSE
    y_pred = regressor.predict(X_test)
    rmse_test = rmse(y_test, y_pred)
    return rmse_test

xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(objective, n_trials=20, show_progress_bar= True)

print('Best result found for XGB:')
print(xgb_study.best_params)
print(xgb_study.best_trial.value)


tuned_xgb = XGBRegressor(**xgb_study.best_params, eval_metric = 'rmse', seed = 0, verbosity = 0)
# Fit parameters on validation dataset.
tuned_xgb.fit(X_train_processed, y_train)
y_val_pred_xgb =tuned_xgb.predict(X_val_processed)
rmse_val_xgb = rmse(y_val, y_val_pred_xgb)
print(f"RMSE for XGB on validation dataset is {rmse_val_xgb}")


def objective(trial, X_train = X_train_processed, y_train = y_train, X_test = X_val_processed, y_test = y_val):
    # Defining parameters:
    param = {}
    param['max_depth'] = trial.suggest_int('max_depth', 3, 11)
    param['n_estimators'] = trial.suggest_int('n_estimators', 100, 200)
    param['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 5)
    param['criterion'] = 'squared_error'
    param['random_state'] = 0
    param['verbose'] = 0
    # Initialize and fit RandomForestRegressor.
    regressor = RandomForestRegressor(**param)
    regressor.fit(X_train, y_train)
    # Calculate RMSE
    y_pred = regressor.predict(X_test)
    rmse_test = rmse(y_test, y_pred)
    return rmse_test

randomforest_study = optuna.create_study(direction='minimize')
randomforest_study.optimize(objective, n_trials=5, show_progress_bar= True)

print('Best result found for RandomForestRegressor:')
print(randomforest_study.best_params)
print(randomforest_study.best_trial.value)


tuned_randomforest = RandomForestRegressor(**randomforest_study.best_params, criterion = 'squared_error', random_state=0, verbose=0)
# Fit parameters on validation dataset.
tuned_randomforest.fit(X_train_processed, y_train)
y_val_pred_randomforest =tuned_randomforest.predict(X_val_processed)
rmse_val_randomforest = rmse(y_val, y_val_pred_randomforest)
print(f"RMSE for XGB on validation dataset is {rmse_val_randomforest}")


# Defining input shape, metrics and loss function for NN.
input_shape = [X_val_meta.shape[1]]
rmse_metric = keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
rmse_loss = keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error", dtype=None)


# Defining the model
# Hidden_neurons variable stores the number of neurons I'm using in the deep layer
hidden_neurons = int(input_shape[0]*2/3)
meta_model = keras.Sequential([
                layers.Input(shape = input_shape),
                layers.Dense(1, activation = 'relu'),
                layers.Dense(1, activation = 'relu')
                ])
meta_model.compile(
        optimizer = 'adam',
        loss = rmse_loss,
        metrics = [rmse_metric]
)
early_stopping = keras.callbacks.EarlyStopping(
                patience = 10,
                min_delta = 0.001,
                restore_best_weights = True
)
# Recap of the model. It's important to see how many parameters I'm dealing with.
meta_model.summary()


# Creating the features for meta-model from y values predicted by base models:
X_val_meta = np.column_stack([y_val_pred_catboost, y_val_pred_xgb, y_val_pred_randomforest])


# Predicting values of test dataset with base models
y_test_pred_catboost = tuned_catboost.predict(X_test_processed)
y_test_pred_xgb = tuned_xgb.predict(X_test_processed)
y_test_pred_randomforest = tuned_randomforest.predict(X_test_processed)
# Now I'm stacking these into the features of meta-model.These will be used to test the meta-model.
X_test_meta = np.column_stack([y_test_pred_catboost, y_test_pred_xgb, y_test_pred_randomforest])


# Fit model and validate it against test dataset.
history = meta_model.fit(
    X_val_meta, y_val,
    validation_data = (X_test_meta, y_test),
    batch_size = 512,
    epochs = 100,
    callbacks = [early_stopping],
    verbose = 0
    )


# Check the results:
history_df = pd.DataFrame(history.history)
# Loss plot
loss_plot = history_df.loc[:, ['loss', 'val_loss']].plot()
loss_plot.set_title('Meta-model: loss plots for validation and test sets')
loss_plot.set_ylim(150, 200)
loss_plot.legend(['Validation dataset', 'Test dataset'], title = 'Loss (mean squared error):')
# RMSE plot
rmse_plot = history_df.loc[:, ['root_mean_squared_error', 'val_root_mean_squared_error']].plot()
rmse_plot.set_title('Meta-model: RMSE plots for validation and test sets')
rmse_plot.set_ylim(12, 14)
rmse_plot.legend(['Validation dataset', 'Test dataset'], title = 'RMSE:')
rmse_val = history_df['val_root_mean_squared_error'].iloc[-1]
rmse = history_df['root_mean_squared_error'].iloc[-1]

print(f'RMSE for validation dataset: {rmse:.2f}')

print(f'RMSE for test dataset: {rmse_val:.2f}')


# Save this before passing test_df to pipeline since this column
# will be dropped in the pipeline and I need them to identify the target values.
submission_ids = submission_df['id']


# Preprocessing data with pipeline:
submission_df_processed = preprocess_pipe.transform(submission_df)


# Predict Listening time minutes for test dataset.
y_pred_catboost = tuned_catboost.predict(submission_df_processed)
y_pred_xgb = tuned_xgb.predict(submission_df_processed)
y_pred_randomforest = tuned_randomforest.predict(submission_df_processed)
X_meta = np.column_stack([y_pred_catboost, y_pred_xgb, y_pred_randomforest])
y_pred_meta = meta_model.predict(X_meta)


y_pred_meta = y_pred_meta.ravel() # I need to ravel the array in order to make it flatten.
# Now, I'm going to replace all negative values with 0 since the formers have no physical meaning (Time can't be negative, of course).
# By the way, this operation doesn't affect the RMSE.
y_pred_meta_test = np.where(y_pred_meta < 0, 0, y_pred_meta)


# Finally, I can create the submission file.
# Store results into Dataframe.
submission_df = pd.DataFrame({'id': submission_ids, 'Listening_Time_minutes': y_pred_meta_test})
submission_df.head()


# Output a CSV file.
submission_df.to_csv(path_or_buf = '/kaggle/working/submission_ensemble2.csv', index = False)

