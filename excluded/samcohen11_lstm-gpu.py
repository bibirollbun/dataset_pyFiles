# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
import warnings
import time
from sklearn.model_selection import train_test_split
import optuna
!pip install optuna-integration[tfkeras]

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load in data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

# Convert date
df_train['date'] = pd.to_datetime(df_train['date'])

# Sort by each category
df_train.sort_values(by=['country', 'store', 'product', 'date'], inplace=True)


# Imputation
# First, let's fill the full missing categories with -1s to drop in the next step
# I am assuming these are sold out or they don't sell this specific type in that specific country
missing_combos = [('Canada', 'Discount Stickers', 'Holographic Goose'),
                 ('Kenya', 'Discount Stickers', 'Holographic Goose')]

for missing_combo in missing_combos:
    df_train.loc[(df_train['country'] == missing_combo[0]) & 
                (df_train['store'] == missing_combo[1]) & 
                (df_train['product'] == missing_combo[2]), 'num_sold'] = 0

# # Lagged value/mean imputation
df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].shift(7)
df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].shift(14)
df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].shift(364)
df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].shift(728)
df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].transform(lambda x: x.fillna(x.mean()))
# # I checked the graphs... good enough!!
# We could add a std multiplied by a random variable but otherwise it's not hugely significant I presume


# Add Test Data
df_testR = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_testR['date'] = pd.to_datetime(df_testR['date'])
df_full = pd.concat([df_train, df_testR])
df_full.sort_values(by=['country', 'store', 'product', 'date'], inplace=True)
# Drop the -1's I created in the step before to get rid of all full missing categories
# df_full = df_full[df_full['num_sold'] != -1]


# Feature Creation

# Days of the week, month, year
df_full['year'] = df_full['date'].dt.year
df_full['day_of_week'] = df_full['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df_full['month'] = df_full['date'].dt.month
df_full['day'] = df_full['date'].dt.day


# Determine whether each date is in a leap year
df_full['is_leap_year'] = (
    (df_full['date'].dt.year % 4 == 0) &
    ((df_full['date'].dt.year % 100 != 0) | (df_full['date'].dt.year % 400 == 0))
)
df_full['is_leap_year'] = df_full['is_leap_year'].astype('category')


# Calculate day of year with leap year adjustment
df_full['days_in_year'] = np.where(df_full['is_leap_year'], 366, 365)
df_full['day_of_year'] = df_full['date'].dt.dayofyear

# Calculate day in 2 year with leap year adjustment
days_in_year_df = df_full[['year', 'days_in_year']].drop_duplicates()
new_row = {'year': 2020, 'days_in_year': 366}

# Add the row using pd.concat
days_in_year_df = pd.concat([days_in_year_df, pd.DataFrame([new_row])], ignore_index=True)

days_in_year_df['two_year_group'] = (days_in_year_df['year'] - days_in_year_df['year'].min()) // 2
# Sum days within each group
two_year_summary = days_in_year_df.groupby('two_year_group')['days_in_year'].sum().reset_index()

# Rename for clarity
two_year_summary.rename(columns={'days_in_year': 'days_in_two_years'}, inplace=True)
days_in_year_df = days_in_year_df.merge(two_year_summary, on='two_year_group', how='left')
days_in_year_df = days_in_year_df.drop(columns=['two_year_group', 'days_in_year'])
df_full = df_full.merge(days_in_year_df, on='year', how='left')
df_full[['year', 'days_in_two_years']]

df_full['day_of_two_year'] = 0
df_full.loc[df_full['year'] % 2 == 1, 'day_of_two_year'] = df_full['days_in_two_years'] - df_full['days_in_year'] + df_full['day_of_year']
df_full.loc[df_full['year'] % 2 == 0, 'day_of_two_year'] = df_full['day_of_year']


# Fourier terms for complex seasonality
# Compute Fourier terms using the adjusted periodicity

# Week Seasonality
for k in range(1, 3):  # Use 2 harmonics as an example
    df_full[f'sin_week_{k}'] = np.sin(2 * np.pi * k * df_full['date'].dt.dayofweek / 7)
    df_full[f'cos_week_{k}'] = np.cos(2 * np.pi * k * df_full['date'].dt.dayofweek / 7)

# Year Seasonality
for k in range(1, 2):  # Use 1 harmonics for the year
    df_full[f'sin_year_{k}'] = np.sin(2 * np.pi * k * df_full['day_of_year'] / df_full['days_in_year'])
    df_full[f'cos_year_{k}'] = np.cos(2 * np.pi * k * df_full['day_of_year'] / df_full['days_in_year'])

# Every other year (bi-annual) Seasonality
for k in range(1, 2):  # Use the first 2 harmonics
    df_full[f'sin_biyear_{k}'] = np.sin(2 * np.pi * k * df_full['day_of_two_year'] / df_full['days_in_two_years'])
    df_full[f'cos_biyear_{k}'] = np.cos(2 * np.pi * k * df_full['day_of_two_year'] / df_full['days_in_two_years'])



# Save the mean/std for later reconversion
mean_std_df = pd.DataFrame({'mean': df_full.groupby(['country', 'store', 'product'])['num_sold'].mean(),
                            'std': df_full.groupby(['country', 'store', 'product'])['num_sold'].std()})
mean_std_df = mean_std_df.reset_index()
mean_std_df


# Standardize num_sold
df_full['num_sold'] = df_full.groupby(['country', 'store', 'product'])['num_sold'].transform(lambda x: (x - x.mean()) / x.std())

# One Hot Encode the categorical vars
df_full = pd.get_dummies(df_full, columns=['country', 'store', 'product', 'day_of_week'], dtype='int')
                                                            
# Drop unnecessary variables
df_full = df_full.drop(['is_leap_year', 'days_in_year', 'day_of_year', 'days_in_two_years', 'day_of_two_year', 'year', 'month', 'day'], axis=1)


df_full


# Train, test, split
train_test_split_date = pd.Timestamp('2017-01-01')

# # Make train and test sets
df_train1 = df_full[(df_full['date'] < train_test_split_date)] 
df_test = df_full[(df_full['date'] >= train_test_split_date)]

# Split Train into train and validation
train_valid_split_date = pd.Timestamp('2016-01-01')
df_valid = df_train1[(df_train1['date'] >= train_valid_split_date)]
df_train2 = df_train1[(df_train1['date'] < train_valid_split_date)]

train_val_df = df_train1.fillna(0)
train_df = df_train2.fillna(0)
val_df = df_valid.fillna(0)

#X and y
# train_val_df = df_train1.drop(['date', 'id'], axis=1)

# train_df = df_train2.drop(['date', 'id'], axis=1)

# val_df = df_valid.drop(['date', 'id'], axis=1)

# test_df = df_test.drop(['date', 'id'], axis=1)
# y_test_id = df_test[['id', 'num_sold']]

# test_df


def df_to_X_y(df, window_size=7, dataset_type='train', pred_start=0):
  # TO DO:
  # 1. If there are some dangling obs at the end of the dataset (i.e. 3), go back in time
  #    the appropriate number of spots so that we can get predictions for the last few obs

  # First, calculate date difference in days
  # This will be our category cutpoint, as each category shows up once a day
  total_days = (df['date'].max() - df['date'].min()).days + 1 # we need to add one because it doesn't include the first date being subtracted
  # print(total_days)
  # Drop unnecessary columns
  df = df.drop(['date', 'id'], axis=1)

  # Convert to numpy
  df_as_np = df.to_numpy()
  X = []
  y = []

  if dataset_type == 'train': # training set makes a new data point every step
      i = 0
      for _ in range(0, len(df_as_np)-(90*window_size)): # 90 categories
        if i+window_size >= len(df_as_np):
            print('end train dataset creation')
            break # terminate
        elif (i+window_size % total_days == 0):
            # need to jump forward to the next category here
            i = i + window_size
        else:
            row = [r for r in df_as_np[i:i+window_size]]
            X.append(row)
    
            # Get labels for window size number of rows
            label = [la for la in df_as_np[i+window_size:i+(2*window_size), 0]]
            y.append(label)

            i = i + 1
    
            # if i == 49:
            #     break

  elif dataset_type == 'val' or dataset_type == 'test': # test and val set make a new data point every window_size step
      i = 0
      for _ in range(0, len(df_as_np)-window_size, window_size):
        if i+window_size >= len(df_as_np):
            print('end test/val dataset creation')
            break # terminate
        elif (i+window_size % total_days == 0):
            # need to jump forward to the next category here
            i = i + window_size
        elif ((i+window_size) % total_days != 0) and ((total_days - ((i+window_size) % total_days)) < window_size ): #needs to be when i+window_size isn't totally divisible by the total number of days, but there is a straggling amount of days
            straggler_days = total_days - ((i+window_size) % total_days)
            # we can't feed something into the model with less that window_size days, so I am just going back in time and getting obs we've seen already
            # I will account for the overlap when graphing later
            row = [r for r in df_as_np[(i-(window_size-straggler_days)):(i+straggler_days)]]
            X.append(row)
        
            # Get labels for window size number of rows
            label = [la for la in df_as_np[(i+straggler_days):(i+straggler_days+window_size), 0]]
            y.append(label)

            i = i + window_size + straggler_days
        else:
            row = [r for r in df_as_np[i:i+window_size]]
            X.append(row)
        
            # Get labels for window size number of rows
            label = [la for la in df_as_np[i+window_size:i+(2*window_size), 0]]
            y.append(label)

            i = i + window_size
            
  elif dataset_type == 'pred':
      # we just need the next 7 obs
      X.append([r for r in df_as_np[pred_start:pred_start+window_size]])
      y.append([0]) # Doesn't do anything, just a placeholder as we don't know the y value

  elif dataset_type == 'last_window_size':
      for i in range(total_days, len(df_as_np)+1, total_days):
          X.append([r for r in df_as_np[i-window_size:i]]) # assuming we use weekly data to train the model here
          y.append([0]) # Doesn't do anything, just a placeholder as we don't know the y value
      print('end last window size dataset creation to get last 7 obs from the training set')

  elif dataset_type == 'chunk':
      for i in range(0, len(df_as_np), total_days):
          X.append([r for r in df_as_np[i:i+total_days]]) # assuming we use weekly data to train the model here
          y.append([0]) # Doesn't do anything, just a placeholder as we don't know the y value
      print('end chunk dataset creation for unknown test set')

  return np.array(X), np.array(y)


X_train_val, y_train_val = df_to_X_y(train_val_df, dataset_type='train')
print(f"x train-val shape: {X_train_val.shape} and y train-val shape: {y_train_val.shape}")
X_train, y_train = df_to_X_y(train_df, dataset_type='train')
print(f"x train shape: {X_train.shape} and y train shape: {y_train.shape}")
X_val, y_val = df_to_X_y(val_df, dataset_type='val')
print(f"x val shape: {X_val.shape} and y val shape: {y_val.shape}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from optuna.integration import TFKerasPruningCallback


gpus = tf.config.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")

# Define Optuna objective function
def objective(trial):
    # Hyperparameters to tune
    lstm_units = trial.suggest_int("lstm_units", 16, 32, step=16)
    dense_units = trial.suggest_int("dense_units", 16, 32, step=16)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Build the LSTM model
    model = Sequential()
    model.add(InputLayer((7, 30)))  # Input shape
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(7, activation='linear'))  # Output for a week's prediction

    print(model.summary())
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse', metrics=[MeanAbsolutePercentageError()])

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )
    # Add the pruning callback
    pruning_callback = TFKerasPruningCallback(trial, monitor="val_loss")

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Allow for sufficient epochs
        batch_size=batch_size,
        verbose=0,
        callbacks=[lr_scheduler, early_stopping, pruning_callback]  # Add both callbacks
    )

    # Get the validation loss for optimization
    val_loss = history.history['val_loss'][-1]
    return val_loss


# Create and run the Optuna study
study = optuna.create_study(direction="minimize")  # Minimize validation loss
study.optimize(objective, n_trials=25)  # Run 25 trials

# Best hyperparameters
print("Best hyperparameters:", study.best_params)

# GPU check
print("GPU available:", tf.config.list_physical_devices('GPU'))


# Train the best model
# Best hyperparameters from Optuna
best_params = study.best_params

# Rebuild the best model
best_model = Sequential()
best_model.add(InputLayer((7, 30)))  # Input shape
best_model.add(LSTM(best_params["lstm_units"], return_sequences=False))
best_model.add(Dropout(best_params["dropout_rate"]))
best_model.add(Dense(best_params["dense_units"], activation='relu'))
best_model.add(Dense(7, activation='linear'))

# Compile the model
best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
    loss='mse',
    metrics=['mape']
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

# Train the best model with callbacks
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  # Allow a sufficient number of epochs
    batch_size=best_params["batch_size"],
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]  # Include callbacks
)



# Some model stats
print(history.history['loss'])      # Training loss over epochs
print(history.history['val_loss'])  # Validation loss over epochs


# View training and validation loss to diagnose overfitting
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs. Validation Loss')
plt.show()


predictions = best_model.predict(X_val)
predictions


# Convert predictions back to original units

# column names
col_names = ['country_Canada', 'country_Finland', 'country_Italy',
       'country_Kenya', 'country_Norway', 'country_Singapore',
       'store_Discount Stickers', 'store_Premium Sticker Mart',
       'store_Stickers for Less', 'product_Holographic Goose',
       'product_Kaggle', 'product_Kaggle Tiers', 'product_Kerneler',
       'product_Kerneler Dark Mode']

# Convert to pandas
back_to_pd = pd.DataFrame(X_val.reshape((-1, 30)))
# Add in column names
back_to_pd = back_to_pd.rename(columns={9:col_names[0],
                                       10:col_names[1],
                                       11:col_names[2],
                                       12:col_names[3],
                                       13:col_names[4],
                                       14:col_names[5],
                                       15:col_names[6],
                                       16:col_names[7],
                                       17:col_names[8],
                                       18:col_names[9],
                                       19:col_names[10],
                                       20:col_names[11],
                                       21:col_names[12],
                                       22:col_names[13]})
# Merge mean and std on
mean_std_df_upd = pd.get_dummies(mean_std_df, columns=['country', 'store', 'product',], dtype='int')
back_to_pd = back_to_pd.merge(mean_std_df_upd, on=col_names,  how='left')

# Convert to original units
back_to_pd['num_sold_actual'] = (back_to_pd[0] * back_to_pd['std']) + back_to_pd['mean']
back_to_pd = pd.concat([back_to_pd, pd.DataFrame({'pred': predictions.flatten()})], axis=1)
back_to_pd['num_sold_preds'] = (back_to_pd['pred'] * back_to_pd['std']) + back_to_pd['mean']

back_to_pd


from sklearn.metrics import mean_absolute_percentage_error as mape

def plot_predictions1(predictions, actual, start=0, end=100000):
  df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':actual})
  plt.plot(df['Predictions'][start:end])
  plt.plot(df['Actuals'][start:end])
  return mape(actual, predictions)
mape1 = plot_predictions1(back_to_pd['num_sold_preds'], back_to_pd['num_sold_actual'])
print(f'mape is {mape1}')


plot_predictions1(back_to_pd['num_sold_preds'], back_to_pd['num_sold_actual'], start=10500, end=10550)


# Train the best model with callbacks
history = best_model.fit(
    X_train_val, y_train_val,
    epochs=50,  # Allow a sufficient number of epochs
    batch_size=best_params["batch_size"],
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]  # Include callbacks
)


def make_preds2(X_test_df, X_train_df, y_test_df, window_size=7, num_groups=90):
    # 1095 for 3 years (non leap years of 2017-2019) of data 365*3
    
    # Get the last 7 obs from each training category
    x_train, _ = df_to_X_y(X_train_df, window_size=window_size, dataset_type='last_window_size')
    
    to_pred_x, to_pred_y = df_to_X_y(X_test_df, dataset_type='chunk')

    print(x_train.shape)
    print(to_pred_x.shape)
    real_start = time.time()
    for i, j, k, l in zip(to_pred_x, to_pred_y, x_train, range(0, num_groups)):
        # can parallelize here!!! on GPU
        start = time.time()
        # basically we just need to do each category independently
        # a. find the last 7 from the x_train in that category (2557 * l - 7)
        # b. make the first prediction and put that prediction into the first pred_x/pred_y
        # c. make all the preds and stop at 2557 interval

        # First prediction from the last 7 obs
        pred1 = best_model.predict(np.array([k]), verbose=0).flatten()

        # Update the array to include those first new predictions
        for pred_num in range(len(pred1)): # there are 7 obs
            i[pred_num, 0] = pred1[pred_num]

        # Loop through the to_pred_x making predictions for all the obs in the category
        for counter in range(0, i.shape[0]-window_size, window_size):

            # This code edits the extra straggler observations at the end
            if ((counter+window_size) % i.shape[0] != 0) and ((i.shape[0] - ((counter+window_size) % i.shape[0])) < window_size ):
                straggler_days = i.shape[0] - ((counter+window_size) % i.shape[0])
                preds = best_model.predict(np.array([i[(counter-(window_size-straggler_days)):(counter+straggler_days)]]), verbose=0).flatten()
                
                for pred_num in range(len(preds)):
                    # update the x_train dataset
                    # I could add an average here
                    # something like:
                    # if the current i[counter+straggler_days+pred_num, 0] != np.nan:
                    #    i[counter+straggler_days+pred_num, 0] = np.average(i[counter+straggler_days+pred_num, 0], preds[pred_num])
                    i[counter+straggler_days+pred_num, 0] = preds[pred_num]

            # This code is the normal step-by-step of the predict and update obs to use for the next prediction
            else:
                preds = best_model.predict(np.array([i[counter:counter+window_size]]), verbose=0).flatten()
             
                for pred_num in range(len(preds)):
                    # update the x_train dataset
                    i[counter+window_size+pred_num, 0] = preds[pred_num]
            # if counter == 49:
            #     break
        end = time.time()
        print(f"one group processed - time of group: {round(end - start, 0)}")
        print(f"total time spent: {round(end-real_start, 0)}")

    return to_pred_x
        


preds_np = make_preds2(df_test, train_val_df, _)


# # Loop of predictions
# def make_preds(X_test_df, X_train_df, y_test_df, window_size=7):
#     # Suppress all warnings globally
#     warnings.filterwarnings("ignore")
    
#     # TO DO:
#     # What is happening is that i is not perfectly lined up. Basically when we need to go back to
#     # the training data, i gets messed up.
#     # I basically need to adjst the i to not increase when 
#     # I think I fixed it

#     # Initialize filter column name variable values
#     filter_col_names = ['country_Canada', 'country_Finland', 'country_Italy',
#        'country_Kenya', 'country_Norway', 'country_Singapore',
#        'store_Discount Stickers', 'store_Premium Sticker Mart',
#        'store_Stickers for Less', 'product_Holographic Goose',
#        'product_Kaggle', 'product_Kaggle Tiers', 'product_Kerneler',
#        'product_Kerneler Dark Mode']
#     prev_variables = {name: 0 for name in filter_col_names}

#     i = 0 # i is the main counter that marks the position in the X_test_df where we are
#     multiplication_factor = 0 # the multiplication factor is used to keep track of updates to the y_test_df, where we update the predictions and then use them for our next calculation
#     while i < (len(X_test_df)-window_size-(window_size*multiplication_factor)): # the multiplication factor keeps track of the number of groups we have
#         # if i > 6000:
#         #      break
    
#         if (X_test_df['country_Canada'].iloc[i] == prev_variables['country_Canada']) & (X_test_df['country_Finland'].iloc[i] == prev_variables['country_Finland']) & (X_test_df['country_Italy'].iloc[i] == prev_variables['country_Italy']) & (X_test_df['country_Kenya'].iloc[i] == prev_variables['country_Kenya']) & (X_test_df['country_Norway'].iloc[i] == prev_variables['country_Norway']) & (X_test_df['country_Singapore'].iloc[i] == prev_variables['country_Singapore']) & (X_test_df['store_Discount Stickers'].iloc[i] == prev_variables['store_Discount Stickers']) & (X_test_df['store_Premium Sticker Mart'].iloc[i] == prev_variables['store_Premium Sticker Mart']) & (X_test_df['store_Stickers for Less'].iloc[i] == prev_variables['store_Stickers for Less']) & (X_test_df['product_Holographic Goose'].iloc[i] == prev_variables['product_Holographic Goose']) & (X_test_df['product_Kaggle'].iloc[i] == prev_variables['product_Kaggle']) & (X_test_df['product_Kaggle Tiers'].iloc[i] == prev_variables['product_Kaggle Tiers']) & (X_test_df['product_Kerneler'].iloc[i] == prev_variables['product_Kerneler']) & (X_test_df['product_Kerneler Dark Mode'].iloc[i] == prev_variables['product_Kerneler Dark Mode']):
            
#             to_pred, _ = df_to_X_y(test_df, window_size=window_size, dataset_type='pred', pred_start=i)
#             i = i + window_size

#         else:
#             # Reset values
#             prev_variables['country_Canada'] = X_test_df['country_Canada'].iloc[i]
#             prev_variables['country_Finland'] = X_test_df['country_Finland'].iloc[i]
#             prev_variables['country_Italy'] = X_test_df['country_Italy'].iloc[i]
#             prev_variables['country_Kenya'] = X_test_df['country_Kenya'].iloc[i]
#             prev_variables['country_Norway'] = X_test_df['country_Norway'].iloc[i]
#             prev_variables['country_Singapore'] = X_test_df['country_Singapore'].iloc[i]
#             prev_variables['store_Discount Stickers'] = X_test_df['store_Discount Stickers'].iloc[i]
#             prev_variables['store_Premium Sticker Mart'] = X_test_df['store_Premium Sticker Mart'].iloc[i]
#             prev_variables['store_Stickers for Less'] = X_test_df['store_Stickers for Less'].iloc[i]
#             prev_variables['product_Holographic Goose'] = X_test_df['product_Holographic Goose'].iloc[i]
#             prev_variables['product_Kaggle'] = X_test_df['product_Kaggle'].iloc[i]
#             prev_variables['product_Kaggle Tiers'] = X_test_df['product_Kaggle Tiers'].iloc[i]
#             prev_variables['product_Kerneler'] = X_test_df['product_Kerneler'].iloc[i]
#             prev_variables['product_Kerneler Dark Mode'] = X_test_df['product_Kerneler Dark Mode'].iloc[i]

#             df_sample = X_train_df[(X_train_df['country_Canada'] == prev_variables['country_Canada']) &
#                                    (X_train_df['country_Finland'] == prev_variables['country_Finland']) &
#                                    (X_train_df['country_Italy'] == prev_variables['country_Italy']) &
#                                    (X_train_df['country_Kenya'] == prev_variables['country_Kenya']) &
#                                    (X_train_df['country_Norway'] == prev_variables['country_Norway']) &
#                                    (X_train_df['country_Singapore'] == prev_variables['country_Singapore']) &
#                                    (X_train_df['store_Discount Stickers'] == prev_variables['store_Discount Stickers']) &
#                                    (X_train_df['store_Premium Sticker Mart'] == prev_variables['store_Premium Sticker Mart']) &
#                                    (X_train_df['store_Stickers for Less'] == prev_variables['store_Stickers for Less']) &
#                                    (X_train_df['product_Holographic Goose'] == prev_variables['product_Holographic Goose']) &
#                                    (X_train_df['product_Kaggle'] == prev_variables['product_Kaggle']) &
#                                    (X_train_df['product_Kaggle Tiers'] == prev_variables['product_Kaggle Tiers']) &
#                                    (X_train_df['product_Kerneler'] == prev_variables['product_Kerneler']) &
#                                    (X_train_df['product_Kerneler Dark Mode'] == prev_variables['product_Kerneler Dark Mode'])]

#             multiplication_factor = multiplication_factor + 1 # this is to ensure there is no overlap in the updating of the x/y test
#             to_pred, _ = df_to_X_y(df_sample.iloc[-window_size:], window_size=window_size, dataset_type='pred', pred_start=0)
            
#         next_preds = model.predict(to_pred, verbose=0).flatten()

#         # Update X test
#         for j in range(len(next_preds)):
#             if (i+j+(window_size*multiplication_factor)) >= len(X_test_df):
#                 break # this will be the end of the predictions
#             elif (i+j+(window_size*multiplication_factor)) % 1095 == 0:
#                 break # this is just the end of prediction for that specific category
#             else:
#                 y_test_df['num_sold'].iloc[(i+j+(window_size*multiplication_factor))] = next_preds[j]
#                 X_test_df['num_sold'].iloc[(i+j+(window_size*multiplication_factor))] = next_preds[j]

#     return X_test_df, y_test_df


# X_pred_df, y_pred_df = make_preds(test_df, train_val_df, y_test_id, window_size=7)
# y_pred_df['num_sold']


# column names
col_names = ['num_sold', 'sin_week_1', 'cos_week_1', 'sin_week_2',
       'cos_week_2', 'sin_year_1', 'cos_year_1', 'sin_biyear_1',
       'cos_biyear_1', 'country_Canada', 'country_Finland', 'country_Italy',
       'country_Kenya', 'country_Norway', 'country_Singapore',
       'store_Discount Stickers', 'store_Premium Sticker Mart',
       'store_Stickers for Less', 'product_Holographic Goose',
       'product_Kaggle', 'product_Kaggle Tiers', 'product_Kerneler',
       'product_Kerneler Dark Mode', 'day_of_week_0', 'day_of_week_1',
       'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5',
       'day_of_week_6']

# Convert to pandas
back_to_pd = pd.DataFrame(preds_np.reshape((-1, 30)))
# Add in column names
back_to_pd = back_to_pd.rename(columns={0:col_names[0],
                                        1:col_names[1],
                                        2:col_names[2],
                                        3:col_names[3],
                                        4:col_names[4],
                                        5:col_names[5],
                                        6:col_names[6],
                                        7:col_names[7],
                                        8:col_names[8],
                                        9:col_names[9],
                                       10:col_names[10],
                                       11:col_names[11],
                                       12:col_names[12],
                                       13:col_names[13],
                                       14:col_names[14],
                                       15:col_names[15],
                                       16:col_names[16],
                                       17:col_names[17],
                                       18:col_names[18],
                                       19:col_names[19],
                                       20:col_names[20],
                                       21:col_names[21],
                                       22:col_names[22],
                                       23:col_names[23],
                                       24:col_names[24],
                                       25:col_names[25],
                                       26:col_names[26],
                                       27:col_names[27],
                                       28:col_names[28],
                                       29:col_names[29]})
# Add on date and id
updated_df = pd.concat([df_test[['date', 'id']].reset_index(drop=True), back_to_pd], axis=1)

# Stack other dataframe in
df_with_preds = pd.concat([train_val_df, updated_df])

# Merge mean and std on
mean_std_df_upd = pd.get_dummies(mean_std_df, columns=['country', 'store', 'product',], dtype='int')
df_with_preds = df_with_preds.merge(mean_std_df_upd, on=col_names[9:23],  how='left')

# Convert to original units
df_with_preds['num_sold_pred'] = round((df_with_preds['num_sold'] * df_with_preds['std']) + df_with_preds['mean'], 0)


# Consolidate the dummy columns for graphing/viz purposes
df_with_preds['country'] = pd.from_dummies(df_with_preds[['country_Canada', 'country_Finland', 'country_Italy', 'country_Kenya', 'country_Norway', 'country_Singapore']], sep='country_')
df_with_preds['store'] = pd.from_dummies(df_with_preds[['store_Discount Stickers', 'store_Premium Sticker Mart', 'store_Stickers for Less']], sep='store_') 
df_with_preds['product'] = pd.from_dummies(df_with_preds[['product_Holographic Goose', 'product_Kaggle', 'product_Kaggle Tiers', 'product_Kerneler', 'product_Kerneler Dark Mode']], sep='product_')


# see = y_pred_df.fillna(0)
# df_test_with_preds = df_testR[['date', 'id', 'country', 'product', 'store']].merge(see, on='id', how='left', suffixes=('_true', '_pred'))
# df_test_with_preds = df_test_with_preds.merge(mean_std_df, on=['country', 'store', 'product'], how='left', suffixes=('_true', '_pred'))
# df_test_with_preds['num_sold'] = (df_test_with_preds['num_sold'] * df_test_with_preds['std']) + df_test_with_preds['mean']
# df_with_preds = pd.concat([df_train[['date', 'id', 'country', 'product', 'store', 'num_sold']], df_test_with_preds])
# df_with_preds.fillna(0, inplace=True)
# df_with_preds['num_sold'] = df_with_preds['num_sold'].apply(lambda x: round(max(x, 0)))


date1 = pd.Timestamp('2017-10-01')
date2 = pd.Timestamp('2017-10-15')
i=0
combinations = list(itertools.product(df_with_preds['country'].unique(), df_with_preds['store'].unique(), df_with_preds['product'].unique()))
for cat1, cat2, cat3 in combinations:
    
    df_sample = df_with_preds[(df_with_preds['country'] == cat1) &
                (df_with_preds['store'] == cat2) &
                (df_with_preds['product'] == cat3)] #&
                # (df_with_preds['date'] >= date1) &
                # (df_with_preds['date'] <= date2)]


    plt.figure(figsize=(8, 6))  # Set figure size
    plt.plot(df_sample['date'],df_sample['num_sold_pred'])  # Create the histogram
    plt.title(f'Num Sold over time for {cat1}, {cat2}, {cat3}')  # Title of the plot
    plt.xlabel('Date')  # X-axis label
    plt.ylabel('Number Sold')  # Y-axis label
    plt.grid(True)  # Show grid
    plt.show()

    i = i+1
    # if i ==10:
    #     break


# # Plot feature importance
# plot_importance(model, importance_type='weight')
# plot_importance(model_final, importance_type='cover')
# plot_importance(model_final, importance_type='gain')



date1 = pd.Timestamp('2017-01-01')

to_keep = df_with_preds[df_with_preds['date'] >= date1]
to_keep = to_keep[['id', 'num_sold_pred']]
to_keep['num_sold'] = to_keep['num_sold_pred']
to_keep.drop('num_sold_pred', axis=1).to_csv('submission.csv', index=False)

