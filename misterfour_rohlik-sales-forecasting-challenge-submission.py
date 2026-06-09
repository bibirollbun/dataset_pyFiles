%%capture
!pip install tensorflow==2.15.0
!pip install keras==2.15.0
!pip install scikit-learn==1.2.2


from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
from tensorflow.keras.layers import Input, Conv1D, Multiply, Add, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric
from sklearn.preprocessing import OrdinalEncoder
import joblib
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Add, Activation, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Multiply, LayerNormalization
)
from tensorflow.keras.layers import MultiHeadAttention


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters (must match training pipeline)
SEQUENCE_LENGTH = 100
BATCH_SIZE = 256

# Additional holiday days (unchanged)
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

# Functions (unchanged)
def fill_loss_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [pd.to_datetime(date, format='%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

def enrich_calendar(df):
    df = df.sort_values('date').reset_index(drop=True)
    df['next_holiday_date'] = df.loc[df['holiday'] == 1, 'date'].shift(-1)
    df['next_holiday_date'] = df['next_holiday_date'].bfill()
    df['date_days_to_next_holiday'] = (df['next_holiday_date'] - df['date']).dt.days
    df.drop(columns=['next_holiday_date'], inplace=True)
    df['next_shops_closed_date'] = df.loc[df['shops_closed'] == 1, 'date'].shift(-1)
    df['next_shops_closed_date'] = df['next_shops_closed_date'].bfill()
    df['date_days_to_shops_closed'] = (df['next_shops_closed_date'] - df['date']).dt.days
    df.drop(columns=['next_shops_closed_date'], inplace=True)
    df['date_day_after_closed_day'] = ((df['shops_closed'] == 0) & (df['shops_closed'].shift(1) == 1)).astype(int)
    df['date_second_closed_day'] = ((df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)).astype(int)
    df['date_day_after_two_closed_days'] = ((df['shops_closed'] == 0) & (df['date_second_closed_day'].shift(1) == 1)).astype(int)
    return df

def stack_datasets(df, calendar_extended, inventory, weights):
    df = df.merge(calendar_extended, on=['date', 'warehouse'], how='left')
    df = df.merge(inventory, on=['unique_id', 'warehouse'], how='left')
    df = df.merge(weights, on='unique_id', how='left')
    df['date'] = pd.to_datetime(df['date'])
    return df

def sort_dataframe_by_date(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)
    return df

def encode_datetime(df, datetime_column):
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month
    df['day'] = df[datetime_column].dt.day
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df.drop(datetime_column, axis=1, inplace=True)
    return df

# Load datasets
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
submission = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')

# Load and preprocess calendar
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])
calendar = fill_loss_holidays(calendar, ['Prague_1', 'Prague_2', 'Prague_3'], czech_holiday)
calendar = fill_loss_holidays(calendar, ['Brno_1'], brno_holiday)
calendar = fill_loss_holidays(calendar, ['Munich_1'], munich_holidays)
calendar = fill_loss_holidays(calendar, ['Frankfurt_1'], frankfurt_holidays)

calendar_enriched = pd.DataFrame()
for location in ['Frankfurt_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Prague_3', 'Prague_1', 'Budapest_1']:
    calendar_enriched = pd.concat([calendar_enriched, enrich_calendar(calendar.query('date >= "2020-08-01 00:00:00" and warehouse == @location'))])
calendar_enriched['year'] = calendar_enriched['date'].dt.year
calendar_enriched = calendar_enriched.rename(columns={
    'holiday_name': 'date_holiday_name',
    'year': 'date_year',
    'holiday': 'date_holiday_flag',
    'shops_closed': 'date_shops_closed_flag',
    'winter_school_holidays': 'date_winter_school_holidays_flag',
    'school_holidays': 'date_school_holidays_flag',
})

# Stack datasets
df_submission = stack_datasets(submission, calendar_enriched, inventory, weights)
df_submission['date_holiday_name'] = df_submission['date_holiday_name'].fillna('Working Day')

# Fill NaN values
for col in df_submission.select_dtypes(include=np.number).columns:
    if df_submission[col].isnull().any():
        mean_val = df_submission[col].mean()
        df_submission[col] = df_submission[col].fillna(mean_val)
        print(f"Filled NaN in df_submission['{col}'] with mean: {mean_val:.2f}")

# Sort by date
df_submission = sort_dataframe_by_date(df_submission, 'date')
submission = df_submission.copy()
# Drop noise columns
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
df_submission = df_submission.drop(columns=columns_to_drop, axis=1, errors='ignore')

# Preprocess data
def preprocess_data(df, encoder_path='/kaggle/input/rohik_encoder_v2/scikitlearn/default/1/encoder.pkl', scaler_path='/kaggle/input/rohik_scaler_v2/scikitlearn/default/1/scaler.pkl'):
    weight = df['weight'].values.astype(np.float32)
    x = df.drop(['weight'], axis=1).copy()

    datetime_cols = x.select_dtypes(include=['datetime']).columns
    if len(datetime_cols) > 0:
        for col in datetime_cols:
            x[col + '_month'] = x[col].dt.month
            x[col + '_day'] = x[col].dt.day
        x = x.drop(datetime_cols, axis=1)

    categorical_cols = x.select_dtypes(include=['object', 'category']).columns
    numeric_cols = x.select_dtypes(include=['number']).columns

    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    if len(categorical_cols) > 0:
        x[categorical_cols] = encoder.transform(x[categorical_cols])
    if len(numeric_cols) > 0:
        x[numeric_cols] = scaler.transform(x[numeric_cols])

    x = x.values.astype(np.float32)
    return x, weight

x_submission, weight_submission = preprocess_data(df_submission)

# Pad x_submission for sequence length
pad_length = SEQUENCE_LENGTH - 1
x_submission_padded = np.pad(x_submission, ((pad_length, 0), (0, 0)), mode='constant', constant_values=0)

# Function to create test dataset
def create_test_dataset(features, weights, sequence_length, batch_size):
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=features,
        targets=None,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False
    )
    aligned_weights = weights[-len(features):]
    return dataset, aligned_weights

# Create test dataset
dataset_submission, aligned_weights = create_test_dataset(x_submission_padded, weight_submission, SEQUENCE_LENGTH, BATCH_SIZE)


# Custom WMAE Loss Function
def custom_wmae_loss(y_true, y_pred, sample_weight=None):
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

# Define the path for the model file
keras_model_path = "/kaggle/input/wavenet_transformer_model.keras/keras/default/1/wavenet_transformer_model.keras"

# Load the model
print(f"\n--- Loading Model from '{keras_model_path}' ---")
model = tf.keras.models.load_model(
    keras_model_path,
    custom_objects={
        'custom_wmae_loss': custom_wmae_loss,
        'WeightedMAEMetric': WeightedMAEMetric
    }
)
print("Model loaded successfully!")
model.summary()


# Generate predictions
y_pred_list = []
for x_batch in dataset_submission:
    y_pred_batch = model.predict(x_batch, verbose=0)
    y_pred_list.append(y_pred_batch)

y_pred_padded = np.concatenate(y_pred_list, axis=0)
y_pred = y_pred_padded[-len(x_submission):]

# Load scaler for inverse transformationa
scaler_y = joblib.load('/kaggle/input/rohik_scaler_y_v2/scikitlearn/default/1/scaler_y.pkl')
y_pred_unscaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Prepare submission
submission['date'] = pd.to_datetime(submission['date']).dt.strftime('%Y-%m-%d')
# Restore unique_id since it was dropped
submission['id'] = submission['unique_id'].astype(str) + '_' + submission['date']
submission['sales_hat'] = y_pred_unscaled
submission_final = submission[['id', 'sales_hat']]

# Save submission
submission_final.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission_final

