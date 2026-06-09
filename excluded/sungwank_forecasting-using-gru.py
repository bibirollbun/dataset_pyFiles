# Import packages
import random
import holidays
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


# Set parameters
WINDOW_LENGTH = 10
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE


# Import data
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Date variable to pandas datetime type
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Print number of rows for each
print(train.shape, test.shape)

# Display data
display(train.tail())
display(test.head())


# Data description states that this dataset includes real-world effects
# such as weekend and holiday effect, seasonality, and so on.
def add_time_signals(data):
    def is_weekend(date):
        return date.weekday() > 4
    
    # Add a column indicating whether each date is a weekend
    data['is_weekend'] = data['date'].apply(is_weekend).astype(int)

    # Create a set of US holidays for the relevant years
    us_holidays = holidays.US(years=data['date'].dt.year.unique())
    
    # Check if each date is a holiday
    data['is_holiday'] = data['date'].dt.date.isin(us_holidays).astype(int)
    
    # Convert date to seconds
    timestamp_s = data['date'].map(pd.Timestamp.timestamp)
    
    # Use sine and cosine transforms to get "Time of month", "Time of week", and "Time of year" signals
    day   = 24 * 60 * 60
    week  = 7 * day
    month = 30.4167 * day
    year  = 365.2425 * day
    
    data.loc[:, ['week_sin']]  = np.sin(timestamp_s * (2 * np.pi / week))
    data.loc[:, ['week_cos']]  = np.cos(timestamp_s * (2 * np.pi / week))
    data.loc[:, ['month_sin']] = np.sin(timestamp_s * (2 * np.pi / month))
    data.loc[:, ['month_cos']] = np.cos(timestamp_s * (2 * np.pi / month))
    data.loc[:, ['year_sin']]  = np.sin(timestamp_s * (2 * np.pi / year))
    data.loc[:, ['year_cos']]  = np.cos(timestamp_s * (2 * np.pi / year))

    return data

# Add time related variables
train = add_time_signals(train)
test = add_time_signals(test)

# Display data
display(train.tail())
display(test.head())


# We will create a combination variable for each country, store, and product combination
def combine_categories(row):
    return f"{row['country']}_{row['store']}_{row['product']}"
train['combination'] = train.apply(combine_categories, axis=1)
test['combination'] = test.apply(combine_categories, axis=1)

# Encode combination as an integer
encoder = OrdinalEncoder()
train['combination'] = encoder.fit_transform(train[['combination']])
test['combination'] = encoder.transform(test[['combination']])


# Sort values by combination and then date
train = train.sort_values(['combination', 'date'])
test = test.sort_values(['combination', 'date'])

# We want to explicitly give the model the date we are predicting is either weekend or holiday (we know this in advance!)
# Otherwise the model has to predict whether date is weekend or holiday which will be hard
# Weekend might be easy to predict based on other variables, but holiday is difficult to predict
train['next_day_is_weekend'] = train.groupby('combination')['is_weekend'].shift(-1)
train['next_day_is_holiday'] = train.groupby('combination')['is_holiday'].shift(-1)

test['next_day_is_weekend'] = test.groupby('combination')['is_weekend'].shift(-1)
test['next_day_is_holiday'] = test.groupby('combination')['is_holiday'].shift(-1)

# Impute next_day_is_weekend with appropriate value
train.loc[train['next_day_is_weekend'].isna(), 'next_day_is_weekend'] = 1 # 2017-01-01 is Sunday
test.loc[test['next_day_is_weekend'].isna(), 'next_day_is_weekend'] = 1 # 2020-01-01 is Wednesday

# Impute next_day_is_holiday with 1 since next day is New Year
train.loc[train['next_day_is_holiday'].isna(), 'next_day_is_holiday'] = 1 # 2017-01-01
test.loc[test['next_day_is_holiday'].isna(), 'next_day_is_holiday'] = 1 # 2020-01-01

# Display data
display(train.head())
display(test.head())


# Data is definitely not missing at random.
# num_sold is missing if <200 for Canada and <5 for Kenya.
# Please refer to this notebook for more details: https://www.kaggle.com/code/cabaxiom/s5e1-eda-and-linear-regression-baseline/notebook
# For simplicity, we will set num_sold = 200 for Canada and 5 for Kenya if missing for now
# Impute missing values
train.loc[(train['country'] == 'Canada') & (train['num_sold'].isnull()), 'num_sold'] = 200
train.loc[(train['country'] == 'Kenya') & (train['num_sold'].isnull()), 'num_sold'] = 5


# Encode categorical values as numeric
for category in ['country', 'store', 'product']:
    category_encoder = OrdinalEncoder()
    train[category] = encoder.fit_transform(train[[category]])
    test[category] = encoder.transform(test[[category]])

# Display results
display(train.sample(n=5))
display(test.sample(n=5))


# Split the training data into train/validation set
validation_start_date = '2015-01-01'
train_set = train.loc[train['date'] < validation_start_date] # renamed it train_set
validation = train.loc[train['date'] >= validation_start_date]


# Exclude some columns from modeling
columns_to_drop = ['id', 'date', 'combination']

def create_model_data(data):
    data_list = []

    for comb in data['combination'].unique():
        # Subset combination to not mix historic values from different combinations
        subset_data = data.loc[data['combination'] == comb]
        
        # Sort by date
        subset_data = subset_data.sort_values(by='date')
        
        # Drop columns
        subset_data = subset_data.drop(columns=columns_to_drop)
        
        data_list.append(tf.keras.utils.timeseries_dataset_from_array(
                             subset_data.to_numpy(),
                             targets=subset_data['num_sold'][WINDOW_LENGTH:],
                             sequence_length=WINDOW_LENGTH,
                             batch_size=None # don't batch yet
                         )
                        )
    
    return data_list


# Separate out Train/Validation/Test since it might take long
# Train
train_data_list = create_model_data(train_set)

# Combine data randomly
train_data = tf.data.Dataset.sample_from_datasets(train_data_list)

# Shuffle data
train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE)

# Optimize the dataset for training
train_data = train_data.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)


# Validation
validation_data_list = create_model_data(validation)

# Combine data randomly
validation_data = tf.data.Dataset.sample_from_datasets(validation_data_list)

# Shuffle data
validation_data = validation_data.shuffle(SHUFFLE_BUFFER_SIZE)

# Optimize the dataset for training
validation_data = validation_data.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)


# Check our input and output data shape
for example_inputs, example_labels in train_data.take(1):
    NUMBER_OF_FEATURES = example_inputs.shape[2]
    
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

    # Print out an example
    print(example_inputs)
    print(example_labels)


# Add normalization layer
norma_layer = tf.keras.layers.Normalization(axis=-1)
norma_layer.adapt(train_data.map(lambda x, y: x))

# Create model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(WINDOW_LENGTH, NUMBER_OF_FEATURES)),
    norma_layer,
    tf.keras.layers.GRU(16, return_sequences=True),
    tf.keras.layers.GRU(16),
    tf.keras.layers.Dense(1)
])

# Print the model summary
model.summary()


# Get initial weights
init_weights = model.get_weights()

# Set the training parameters
model.compile(loss="mse",
              optimizer="adam",
              metrics=["mape"])

# Train the model
history = model.fit(train_data, epochs=30, validation_data=validation_data)


# Plot loss
# source: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W4/ungraded_labs/C4_W4_Lab_1_LSTM.ipynb
def plot_series(x, y, format="-", start=0, end=None, 
                title=None, xlabel=None, ylabel=None, legend=None ):
    """
    Visualizes time series data

    Args:
      x (array of int) - contains values for the x-axis
      y (array of int or tuple of arrays) - contains the values for the y-axis
      format (string) - line style when plotting the graph
      start (int) - first time step to plot
      end (int) - last time step to plot
      title (string) - title of the plot
      xlabel (string) - label for the x-axis
      ylabel (string) - label for the y-axis
      legend (list of strings) - legend for the plot
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))
    
    # Check if there are more than two series to plot
    if type(y) is tuple:

      # Loop over the y elements
      for y_curr in y:

        # Plot the x and current y values
        plt.plot(x[start:end], y_curr[start:end], format)

    else:
      # Plot the x and y values
      plt.plot(x[start:end], y[start:end], format)

    # Label the x-axis
    plt.xlabel(xlabel)

    # Label the y-axis
    plt.ylabel(ylabel)

    # Set the legend
    if legend:
      plt.legend(legend)

    # Set the title
    plt.title(title)

    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()

# Get mape and loss from history log
mape = history.history['mape']
loss = history.history['loss']
val_mape = history.history['val_mape']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(loss)) 

# Plot Loss
plot_series(
    x=epochs, 
    y=(loss, val_loss), 
    title='Loss', 
    xlabel='Epochs',
    legend=['Train Loss', 'Validation Loss']
    )


# Plot MAPE
plot_series(
    x=epochs, 
    y=(mape, val_mape), 
    title='MAPE', 
    xlabel='Epochs',
    legend=['Train MAPE', 'Validation MAPE']
    )


# Check how a prediction on a random combination looks like
def plot_random_combination():
    # Pick a random combination to make prediction
    random_combination = random.choice(train['combination'].unique())

    # Subset train and validation based on randomly chosen combination
    random_train = train_set.loc[train_set['combination'] == random_combination]
    random_validation = validation.loc[validation['combination'] == random_combination]

    # Need to append training data to create lagged num_sold
    random_train_to_append = random_train.loc[random_train['date'] >= pd.to_datetime(validation_start_date) - pd.Timedelta(days=WINDOW_LENGTH)]
    random_validation = pd.concat([random_train_to_append, random_validation])

    # Create model data
    random_validation_data = create_model_data(random_validation)[0].batch(BATCH_SIZE)

    # Delete appended train data from validation
    random_validation = random_validation.loc[random_validation['date'] >= validation_start_date]
    
    # Make prediction and save it as num_sold_pred column
    random_validation['num_sold_pred'] = model.predict(random_validation_data).flatten()

    # Plot train and test prediction
    plt.figure(figsize=(12, 6))
    plt.plot(random_train['date'], random_train['num_sold'], c='g', label='Train')
    plt.plot(random_validation['date'], random_validation['num_sold'], c='b', label='Validation (Actual)')
    plt.plot(random_validation['date'], random_validation['num_sold_pred'], c='orange', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Number of Items Sold')
    plt.title(f'Number of Items Sold Over Time for Combination {random_combination}')
    plt.legend()
    plt.show()

# See how we did
plot_random_combination()


# Try on different combination
plot_random_combination()


# Try one last time
plot_random_combination()


# # Making prediction takes a very long time to run. I was able to run this code locally on my machine.

# # Create a target variable in test set
# test['num_sold'] = np.nan

# # Save start date before adding training data
# test_start_date = test['date'].min()

# # To create time series data, we need to append training data to create lagged num_sold
# train_to_append = train.loc[train['date'] >= test_start_date - pd.Timedelta(days=WINDOW_LENGTH)]
# print(f"Number of training rows to append: {train_to_append.shape[0]} - should be 900")
# test = pd.concat([train_to_append, test])

# # Iterate through each date
# for d in pd.date_range(start=test_start_date, end=test['date'].max()):
#     # Create a counter, a list of test ids, and a prediciton dataframe
#     counter = 0
#     test_ids = []
#     predictions = pd.DataFrame(columns=['id', 'num_sold'])

#     # Subset data based on date
#     subset_by_date = test.loc[(test['date'] >= d - pd.Timedelta(days=WINDOW_LENGTH)) & (test['date'] <= d)]
    
#     # Iterate through each combination
#     for c in test['combination'].unique():
#         # Subset data based on combination
#         subset_test = subset_by_date.loc[subset_by_date['combination'] == c]
        
#         # Skip redundant work if not missing (in case you restart code block)
#         if subset_test['num_sold'].isnull().any():
#             # Store id before dropping
#             test_ids.append(subset_test.loc[subset_test['num_sold'].isnull(), 'id'].item())

#             # Sort by date and drop columns
#             subset_test = subset_test.sort_values(by='date').drop(columns=columns_to_drop)

#             # Create time series data
#             ts_test = tf.keras.utils.timeseries_dataset_from_array(
#                 subset_test.to_numpy(),
#                 targets=subset_test['num_sold'][WINDOW_LENGTH:],
#                 sequence_length=WINDOW_LENGTH,
#                 batch_size=None)

#             # If loop start of the date
#             if counter == 0:
#                 combined_test = ts_test
#             else:
#                 # Add data to existing data 
#                 combined_test = combined_test.concatenate(ts_test)

#             # Add 1 to counter
#             counter += 1
    
#     # If there is data to predict
#     if counter > 0:
#         # Batch and predict
#         combined_test = combined_test.batch(BATCH_SIZE)
#         preds = model.predict(combined_test)

#         # Assign predictions to the corresponding ids
#         pred_df = pd.DataFrame({'id': test_ids, 'num_sold': preds.flatten()})
#         test.loc[test['id'].isin(test_ids), 'num_sold'] = test['id'].map(pred_df.set_index('id')['num_sold'])

#     # Progress tracking
#     if d.is_month_end:
#         print(f"Done with : {d.month_name()} {d.year}")

# # Drop rows we added from training
# test = test.loc[test['date'] >= test_start_date]

# # Save predictions for submission
# test[['id', 'num_sold']].to_csv('submission.csv', index=False)

