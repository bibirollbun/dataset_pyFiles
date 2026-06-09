import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import ( 
    Input, Conv1D, Multiply, Add, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Activation, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric
from tensorflow.keras import mixed_precision 

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from datetime import datetime
import joblib 
from sklearn.preprocessing import OrdinalEncoder
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression

tf.config.set_visible_devices([], 'GPU')  
physical_devices = tf.config.list_physical_devices('CPU')
assert len(physical_devices) > 0, "No CPU devices found"
tf.config.set_logical_device_configuration(
    physical_devices[0],
    [tf.config.LogicalDeviceConfiguration()]
)

np.random.seed(42)
tf.random.set_seed(42)


train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
submission = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')


# Additional holiday days
czech_holiday = [ 
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]
brno_holiday = [
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]
munich_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]
frankfurt_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]

# Functions
def fill_loss_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

def enrich_calendar(df):
    df = df.sort_values('date').reset_index(drop=True)

    # Number of days until next holiday
    df['next_holiday_date'] = df.loc[df['holiday'] == 1, 'date'].shift(-1)
    # Fill NaT values by using the next valid observation to fill the gap
    df['next_holiday_date'] = df['next_holiday_date'].bfill() 
    df['date_days_to_next_holiday'] = (df['next_holiday_date'] - df['date']).dt.days
    df.drop(columns=['next_holiday_date'], inplace=True)

    # Number of days until shops are closed
    df['next_shops_closed_date'] = df.loc[df['shops_closed'] == 1, 'date'].shift(-1)
    df['next_shops_closed_date'] = df['next_shops_closed_date'].bfill()
    df['date_days_to_shops_closed'] = (df['next_shops_closed_date'] - df['date']).dt.days
    df.drop(columns=['next_shops_closed_date'], inplace=True)

    # Was the shop closed yesterday?
    df['date_day_after_closed_day'] = ((df['shops_closed'] == 0) & (df['shops_closed'].shift(1) == 1)).astype(int)

    # Are shops closed today and were they also closed yesterday (e.g., December 26 in Germany)?
    df['date_second_closed_day'] = ((df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)).astype(int)

    # Was the shop closed the last two days?
    df['date_day_after_two_closed_days'] = ((df['shops_closed'] == 0) & (df['date_second_closed_day'].shift(1) == 1)).astype(int)

    return df

calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=czech_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Brno_1'], holidays=brno_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Munich_1'], holidays=munich_holidays)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Frankfurt_1'], holidays=frankfurt_holidays)

calendar_enriched = pd.DataFrame()

for location in ['Frankfurt_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Prague_3', 'Prague_1', 'Budapest_1']:
    calendar_enriched = pd.concat([
        calendar_enriched,enrich_calendar(calendar.query('date >= "2020-08-01 00:00:00" and warehouse ==@location'))])
calendar_enriched.loc[:,'year'] = calendar_enriched['date'].dt.year
calendar_enriched.sort_values('date')[['date','holiday_name','shops_closed','warehouse','date_days_to_next_holiday']].head(5)

calendar_enriched = calendar_enriched.rename(columns={
    'holiday_name':'date_holiday_name',
    'year':'date_year',
    'holiday':'date_holiday_flag',
    'holiday':'date_holiday_flag',
    'shops_closed':'date_shops_closed_flag',
    'winter_school_holidays':'date_winter_school_holidays_flag',
    'school_holidays':'date_school_holidays_flag',
})

def stack_datasets(df, calendar_extended, inventory, weights):
    """
    Stacks the given DataFrame with additional data from calendar, inventory, and weights.

    Args:
        df: The main DataFrame to be stacked.
        calendar_extended: DataFrame containing calendar-related information.
        inventory: DataFrame containing inventory information.
        weights: DataFrame containing weight information for unique IDs.

    Returns:
        pandas.DataFrame: The stacked DataFrame.
    """
    # Merge with calendar_extended on date and warehouse
    df = df.merge(calendar_extended, on=['date', 'warehouse'], how='left')
    
    # Merge with inventory on unique_id and warehouse
    df = df.merge(inventory, on=['unique_id', 'warehouse'], how='left')
    
    # Perform a VLOOKUP-style merge with weights on unique_id
    df = df.merge(weights, on='unique_id', how='left')
    
    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    return df

df_train = stack_datasets(train, calendar_enriched, inventory, weights)
df_train

# Fill 'date_holiday_name' with 'Working Day' where it's NaN
df_train['date_holiday_name'] = df_train['date_holiday_name'].fillna('Working Day')
df_train


# Calculate the split index for 80% training data
# The remaining 20% will be for the test set
df_split = df_train.copy()
train_split_index = int(len(df_split) * 0.80)

# Split the DataFrame chronologically
df_train = df_split.iloc[:train_split_index].copy()
df_test = df_split.iloc[train_split_index:].copy() 

print("New df_train shape (80%):", df_train.shape)
print("New df_test shape (20%):", df_test.shape)


df_train.isnull().sum()


df_test.isnull().sum()


# Identify numerical columns/ Only numeric columns can have a mean calculated for filling.
numeric_cols = df_train.select_dtypes(include=np.number).columns

# Fill NaN values in numeric columns with the mean of their respective columns
for col in numeric_cols:
    if df_train[col].isnull().any(): 
        col_mean = df_train[col].mean()
        df_train[col] = df_train[col].fillna(col_mean)
        print(f"\nFilled NaNs in '{col}' with mean: {col_mean:.2f}")

print("\ndf_train shape after filling NaNs:", df_train.shape)
print("\nNaN count per column after filling:")
print(df_train.isnull().sum())


df_tr_corr = df_train.copy()


featurex = df_tr_corr.drop(['sales'], axis=1)
featurey = df_tr_corr[['sales']]
print("featurex", featurex.shape)
print("featurey", featurey.shape)
print('-------------------------------------------------------------------------')


# Get a list of column names with string data type
string_columns = df_tr_corr.select_dtypes(include=['object']).columns.tolist() 

print(string_columns) 


encoder = OrdinalEncoder()
featurex[['warehouse', 'date_holiday_name', 'name', 'L1_category_name_en', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']] = encoder.fit_transform(featurex[['warehouse', 'date_holiday_name', 'name', 'L1_category_name_en', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']])
featurex


def encode_datetime(df, datetime_column):
  """
  Encodes datetime features in a pandas DataFrame.

  Args:
    df: The pandas DataFrame containing the datetime column.
    datetime_column: The name of the datetime column in the DataFrame.

  Returns:
    pandas.DataFrame: The DataFrame with encoded datetime features.  
  """

  df[datetime_column] = pd.to_datetime(df[datetime_column]) 

  # Extract features
  df['year'] = df[datetime_column].dt.year
  df['month'] = df[datetime_column].dt.month
  df['day'] = df[datetime_column].dt.day

  # Create cyclical features (optional)
  df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
  df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
  df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
  df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)

  # Drop the original datetime column
  df.drop(datetime_column, axis=1, inplace=True)

  return df

featurex = encode_datetime(featurex, 'date')
featurex


# Calculate mutual information scores for each feature with respect to each target
X, y = featurex, featurey
def calculate_mi_for_target(target_index):
    return mutual_info_regression(X, y.iloc[:, target_index], random_state=42)

# Run parallelized MI calculations on the original y (multi-dimensional)
mi_scores = Parallel(n_jobs=-1)(delayed(calculate_mi_for_target)(i) for i in range(y.shape[1]))

# Convert the list of scores to a DataFrame
mi_scores_df = pd.DataFrame(mi_scores, columns=featurex.columns, index=featurey.columns)
print("\nMutual Information Scores for each feature with respect to each target:")
mi_scores_df

# Aggregate scores across all targets (e.g., by averaging)
aggregated_mi_scores = mi_scores_df.mean(axis=0).sort_values(ascending=False)
print("\nAggregated Mutual Information Scores:")
aggregated_mi_scores


mi_scores_df.T.describe()


aggregated_mi_scores.describe()


mi_scores_df


A = df_train.isnull().sum()
B = df_test.isnull().sum()
print("NaN value in train dataset")
print(A)
print("-"*100)
print("NaN value in test dataset")
print(B)


# Fill NaN values in df_train with the mean of each numerical column in df_train
for col in df_train.select_dtypes(include=np.number).columns:
    if df_train[col].isnull().any():
        mean_val_train = df_train[col].mean()
        df_train[col] = df_train[col].fillna(mean_val_train) 
        print(f"Filled NaN in df_train['{col}'] with mean: {mean_val_train:.2f}")

# Fill NaN values in df_test with the mean of each numerical column in df_test
for col in df_test.select_dtypes(include=np.number).columns:
    if df_test[col].isnull().any(): 
        mean_val_test = df_test[col].mean()
        df_test[col] = df_test[col].fillna(mean_val_test) 
        print(f"Filled NaN in df_test['{col}'] with mean: {mean_val_test:.2f}")

print("\n--- After NaN Imputation ---")
print("df_train info:")
df_train.info()
print("\ndf_test info:")
df_test.info()


def sort_dataframe_by_date(df, date_column):
  """
  Sorts a pandas DataFrame by a specified date column in ascending order.

  Args:
    df: The pandas DataFrame to be sorted.
    date_column: The name of the date column in the DataFrame.

  Returns:
    pandas.DataFrame: The sorted DataFrame.
  """

  # Ensure the date column is in datetime format
  df[date_column] = pd.to_datetime(df[date_column])

  # Sort the DataFrame by the date column in ascending order
  df = df.sort_values(by=date_column) 

  return df

df_train = sort_dataframe_by_date(df_train, 'date')
df_test = sort_dataframe_by_date(df_test, 'date')


print("df_train shape before dropping columns:", df_train.shape)
print("df_test shape before dropping columns:", df_test.shape)


columns_to_drop = [
    'type_2_discount',
    'date_holiday_flag',
    'date_school_holidays_flag',
    'date_shops_closed_flag',
    'date_second_closed_day',
    'date_winter_school_holidays_flag',
    'date_day_after_closed_day',
    'date_day_after_two_closed_days',
    'type_5_discount',
    'type_3_discount',
    'type_1_discount',
    'unique_id',
    "availability" 
]

# Drop columns from df_train
df_train = df_train.drop(columns=columns_to_drop, axis=1, errors='ignore')

# Drop columns from df_test
df_test = df_test.drop(columns=columns_to_drop, axis=1, errors='ignore')

print("df_train shape after dropping columns:", df_train.shape)
print("df_test shape after dropping columns:", df_test.shape)


def preprocess_data(df_train, df_test, target_col):
    """
    Preprocesses df_train (fit and transform) and df_test (transform) for training and testing.

    Args:
        df_train: pandas DataFrame for training.
        df_test: pandas DataFrame for testing.
        target_col: Name of the target column (e.g., 'sales').

    Returns:
        x_train: Training features (float32 NumPy array).
        y_train: Training targets (float32 NumPy array).
        weight_train: Training weights (float32 NumPy array).
        x_test: Test features (float32 NumPy array).
        y_test: Test targets (float32 NumPy array).
        weight_test: Test weights (float32 NumPy array).
    """
    # Extract target and weights
    y_train = df_train[target_col].values.astype(np.float32)
    y_test = df_test[target_col].values.astype(np.float32)
    weight_train = df_train['weight'].values.astype(np.float32)
    weight_test = df_test['weight'].values.astype(np.float32)

    # Extract features
    x_train = df_train.drop([target_col, 'weight'], axis=1).copy()
    x_test = df_test.drop([target_col, 'weight'], axis=1).copy()

    # Handle datetime columns
    datetime_cols = x_train.select_dtypes(include=['datetime']).columns
    if len(datetime_cols) > 0:
        for col in datetime_cols:
            x_train[col + '_month'] = x_train[col].dt.month
            x_train[col + '_day'] = x_train[col].dt.day
            x_test[col + '_month'] = x_test[col].dt.month
            x_test[col + '_day'] = x_test[col].dt.day
        x_train = x_train.drop(datetime_cols, axis=1)
        x_test = x_test.drop(datetime_cols, axis=1)

    # Define categorical and numerical columns
    categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns
    numeric_cols = x_train.select_dtypes(include=['number']).columns

    # Encode categorical features
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if len(categorical_cols) > 0:
        x_train[categorical_cols] = encoder.fit_transform(x_train[categorical_cols])
        x_test[categorical_cols] = encoder.transform(x_test[categorical_cols])

    # Scale numerical features
    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
        x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

    # Save encoder and scaler
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Convert to NumPy arrays
    x_train = x_train.values.astype(np.float32)
    x_test = x_test.values.astype(np.float32)

    return x_train, y_train, weight_train, x_test, y_test, weight_test

# Preprocess data
x_train, y_train, weight_train, x_test, y_test, weight_test = preprocess_data(df_train, df_test, 'sales')

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Hyperparameters
SEQUENCE_LENGTH = 100
BATCH_SIZE = 256

# Preprocessing: Split data and scale targets
val_split = 0.2
val_size = int(len(x_train) * val_split)
x_val, y_val, weights_val = x_train[-val_size:], y_train[-val_size:], weight_train[-val_size:]
x_train, y_train, weights_train = x_train[:-val_size], y_train[:-val_size], weight_train[:-val_size]

# Cast to float32
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)
x_test = x_test.astype(np.float32)
weights_train = weights_train.astype(np.float32)
weights_val = weights_val.astype(np.float32)
weight_test = weight_test.astype(np.float32)

# Scale targets
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)
joblib.dump(scaler_y, 'scaler_y.pkl')

# Function to create time-series dataset with sample weights
def create_dataset(features, targets, weights, sequence_length, batch_size):
    """
    Creates a time-series dataset with features, targets, and sample weights.

    Args:
        features (np.ndarray): Input features (float32).
        targets (np.ndarray): Target values (float32).
        weights (np.ndarray): Sample weights (float32).
        sequence_length (int): Length of each sequence.
        batch_size (int): Batch size.

    Returns:
        tf.data.Dataset: Dataset yielding (inputs, targets, sample_weights).
        np.ndarray: Aligned weights.
    """
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=features,
        targets=targets,
        sequence_length=sequence_length,
        batch_size=batch_size
    )
    aligned_weights = weights[sequence_length - 1:] if len(weights) > sequence_length else weights
    dataset = dataset.map(lambda x, y: (x, y, tf.gather(aligned_weights, tf.range(tf.shape(y)[0]), axis=0)))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, aligned_weights

# Create datasets
dataset_train, train_weights = create_dataset(x_train, y_train, weights_train, SEQUENCE_LENGTH, BATCH_SIZE)
dataset_val, val_weights = create_dataset(x_val, y_val, weights_val, SEQUENCE_LENGTH, BATCH_SIZE)
dataset_test, test_weights = create_dataset(x_test, y_test, weight_test, SEQUENCE_LENGTH, BATCH_SIZE)


# Hyperparameters
SEQUENCE_LENGTH = 100
BATCH_SIZE = 256
EPOCHS = 1
BASE_LR = 3e-4
DROPOUT_RATE = 0.3
FILTERS = 256
KERNEL_SIZE = 3
DILATION_RATES = [1, 2, 4, 8, 16, 32]

# Custom WMAE Loss Function
def custom_wmae_loss(y_true, y_pred, sample_weight=None):
    """
    Weighted Mean Absolute Error loss function following evaluation criteria.
    This version strictly interprets zero sample weights as zero contribution
    and handles cases where the sum of weights is zero to prevent NaN.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if sample_weight is None:
        sample_weight = tf.ones_like(y_true, dtype=tf.float32)
    else:
        sample_weight = tf.cast(sample_weight, tf.float32)
    weighted_error = tf.abs(y_true - y_pred) * sample_weight
    sum_of_weights = tf.reduce_sum(sample_weight)
    return tf.cond(
        tf.greater(sum_of_weights, 0),
        lambda: tf.reduce_sum(weighted_error) / sum_of_weights,
        lambda: tf.constant(0.0, dtype=tf.float32)
    )

# Custom WMAE Metric
class WeightedMAEMetric(Metric):
    """
    Custom metric to compute Weighted Mean Absolute Error following evaluation criteria.
    This version strictly interprets zero sample weights as zero contribution
    and handles cases where the sum of weights is zero to prevent NaN.
    """
    def __init__(self, name='wmae', **kwargs):
        super(WeightedMAEMetric, self).__init__(name=name, **kwargs)
        self.total_weighted_error = self.add_weight(name='total_weighted_error', initializer='zeros')
        self.total_weights = self.add_weight(name='total_weights', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)
        else:
            sample_weight = tf.cast(sample_weight, tf.float32)
        weighted_error = tf.abs(y_true - y_pred) * sample_weight
        self.total_weighted_error.assign_add(tf.reduce_sum(weighted_error))
        self.total_weights.assign_add(tf.reduce_sum(sample_weight))

    def result(self):
        return tf.cond(
            tf.greater(self.total_weights, 0),
            lambda: self.total_weighted_error / self.total_weights,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

    def reset_state(self):
        self.total_weighted_error.assign(0.0)
        self.total_weights.assign(0.0)

# Learning rate scheduler with warmup
num_train_sequences = max(0, len(y_train) - SEQUENCE_LENGTH + 1) if 'y_train' in globals() else 10000  # Placeholder value
total_steps = EPOCHS * (num_train_sequences // BATCH_SIZE) if BATCH_SIZE > 0 else 0
warmup_steps = min(1000, total_steps // 10)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=BASE_LR,
    decay_steps=max(1, total_steps - warmup_steps),
    alpha=0.1
)
optimizer = Adam(learning_rate=lr_schedule)

# WaveNet Block
def wavenet_block(x, dilation_rate, filters, kernel_size, dropout_rate):
    """
    WaveNet block with dilated convolutions, gating, and residual/skip connections.
    """
    conv_filter = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='tanh')(x)
    conv_gate = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='sigmoid')(x)
    gated_output = Multiply()([conv_filter, conv_gate])
    gated_output = BatchNormalization()(gated_output)
    residual = Conv1D(filters, 1, padding='same')(gated_output)
    skip = Conv1D(filters, 1, padding='same')(gated_output)
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding='same')(x)
    return Add()([x, residual]), skip

# Transformer Encoder Layer
def transformer_encoder(x, num_heads=4, ff_dim=128, dropout_rate=0.3):
    """
    Transformer encoder layer with multi-head attention and feed-forward network.
    """
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    ff_output = Dense(ff_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(x.shape[-1])(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# Custom Layer for Clipping
class ClipLayer(tf.keras.layers.Layer):
    def __init__(self, min_value=-10.0, max_value=10.0, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

# Build WaveNet with Transformer Model
num_features = x_train.shape[1] if 'x_train' in globals() else 19  # Placeholder value
inputs = Input(shape=(SEQUENCE_LENGTH, num_features), name="input")
x = inputs
skip_connections = []

# WaveNet blocks
for dilation_rate in DILATION_RATES:
    x, skip = wavenet_block(x, dilation_rate, FILTERS, KERNEL_SIZE, DROPOUT_RATE)
    skip_connections.append(skip)

x = Add()(skip_connections)
x = Activation('relu')(x)
x = BatchNormalization()(x)

# Transformer encoder layers
x = transformer_encoder(x, num_heads=4, ff_dim=128, dropout_rate=0.3)
x = transformer_encoder(x, num_heads=4, ff_dim=128, dropout_rate=0.3)

# Final layers
x = Conv1D(FILTERS, 1, activation='relu')(x)
x = GlobalAveragePooling1D()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
x = Dropout(DROPOUT_RATE)(x)

# Apply clipping using the custom layer
outputs = ClipLayer(min_value=-10.0, max_value=10.0)(Dense(1, activation='linear', name="output", dtype='float32')(x))

# Build and compile model
model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=optimizer,
    loss=custom_wmae_loss,
    metrics=['mae'],
    weighted_metrics=[WeightedMAEMetric()]
)

model.summary()

# Train the model with sample weights
print("\n--- Training Model ---")
history = model.fit(
    dataset_train, 
    validation_data=dataset_val,  
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    ],
    shuffle=False
)


# Define the path for the .keras file
keras_model_path = "wavenet_transformer_model.keras"

# Save the model
model.save(keras_model_path, overwrite=True)
print(f"\nModel saved successfully to '{keras_model_path}' in Keras format.")

# Load the model from Keras format (.keras)
print(f"\n--- Loading Model from '{keras_model_path}' ---")
loaded_model = tf.keras.models.load_model(
    keras_model_path,
    custom_objects={
        'custom_wmae_loss': custom_wmae_loss,
        'WeightedMAEMetric': WeightedMAEMetric
    }
)
print("Model loaded successfully!")
loaded_model.summary()


print("\n--- Evaluating Model on Test Set (Scaled) ---")

evaluation_results = model.evaluate(dataset_test)
print(f"Test Set Evaluation Results (Scaled):")
for name, value in zip(model.metrics_names, evaluation_results):
    print(f"{name}: {value:.4f}")

print("\n--- Making Predictions and Calculating Unscaled Metrics ---")

# Initialize lists for true values and predictions (scaled)
y_true_scaled_list, y_pred_scaled_list, sample_weights_list = [], [], []

# Iterate through the test dataset to get predictions and corresponding true values and weights
for x_batch, y_batch_scaled, weights_batch in dataset_test:
    
    y_batch_scaled_processed = y_batch_scaled.numpy()
    if np.any(np.isnan(y_batch_scaled_processed)):
        print(f"Warning: NaN values detected in true scaled targets. Replacing with 0 for metric calculation.")
        y_batch_scaled_processed = np.nan_to_num(y_batch_scaled_processed, nan=0.0)

    y_pred_batch_scaled = model.predict(x_batch, verbose=0) # Disable verbose logging

    # NaN Handling for predictions
    if np.any(np.isnan(y_pred_batch_scaled)):
        print(f"Warning: NaN values detected in model predictions. Replacing with 0 for metric calculation.")
        y_pred_batch_scaled = np.nan_to_num(y_pred_batch_scaled, nan=0.0) # Replace NaNs with 0

    y_true_scaled_list.extend(y_batch_scaled_processed) # Use processed true values
    y_pred_scaled_list.extend(y_pred_batch_scaled.flatten())
    sample_weights_list.extend(weights_batch.numpy())

# Convert lists to NumPy arrays
y_true_scaled = np.array(y_true_scaled_list)
y_pred_scaled = np.array(y_pred_scaled_list)
sample_weights_for_metrics = np.array(sample_weights_list)

# Inverse transform predictions and true values to original scale
y_true_unscaled = np.nan_to_num(scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten(), nan=0.0)
y_pred_unscaled = np.nan_to_num(scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten(), nan=0.0)

# Compute evaluation metrics on unscaled data
mae_unscaled = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
mse_unscaled = mean_squared_error(y_true_unscaled, y_pred_unscaled)
rmse_unscaled = np.sqrt(mse_unscaled)

# Calculate WMAE on unscaled data for verification (should be similar to scaled WMAE if scaler is linear)
# Only consider samples with non-zero weights for WMAE calculation to avoid division by zero
non_zero_weight_indices = sample_weights_for_metrics > 0
if np.sum(non_zero_weight_indices) > 0:
    wmae_unscaled = np.sum(np.abs(y_true_unscaled[non_zero_weight_indices] - y_pred_unscaled[non_zero_weight_indices]) * sample_weights_for_metrics[non_zero_weight_indices]) / np.sum(sample_weights_for_metrics[non_zero_weight_indices])
else:
    wmae_unscaled = 0.0 # Handle case where all weights are zero

# Print results
print(f"Evaluation Metrics on Test Set (Unscaled):")
print(f"MAE  : {mae_unscaled:.4f}")
print(f"MSE  : {mse_unscaled:.4f}")
print(f"RMSE : {rmse_unscaled:.4f}")
print(f"WMAE : {wmae_unscaled:.4f}")

