#!pip install --upgrade holidays


import numpy as np
import tensorflow as tf
from numpy.polynomial import Polynomial
import pandas as pd
import holidays
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.optimize import minimize
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras import backend as K


df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])


df_train['date'] = pd.to_datetime(df_train['date'])
df_train['num_sold'] = np.log1p(df_train['num_sold'])


## Filling missing values (in trainning set) with a rolling mean, to preserve local trends
#df_train['num_sold'] = df_train['num_sold'].fillna(
#    df_train['num_sold'].rolling(window=7, min_periods=1).mean()
#)

# fill missing values with the mean of the first 30 days
#df_train['num_sold'] = df_train['num_sold'].fillna(
#    df_train['num_sold'].mean()
#)


df_train = df_train.dropna(subset=['num_sold'])  # Drop rows where 'num_sold' is NaN


len(df_train)


# Função para criar as janelas no formato adequado
def create_rnn_windows(X, y, n_days):
    X_windows = []
    y_targets = []
    
    # Itera pelos índices para criar as janelas
    for i in range(n_days, len(X)):
        # Janelas dos últimos n_days dias (timesteps)
        window = X.iloc[i-n_days:i].values  # Pega as features para a janela
        
        # Valor target (num_sold no próximo dia)
        target = y.iloc[i]  # Valor de saída correspondente à janela

        X_windows.append(window)
        y_targets.append(target)
    
    # Converte para numpy arrays
    return np.array(X_windows), np.array(y_targets)

def create_rnn_windows_test(X, n_days):
    """
    Create RNN input windows for testing data with n_days timesteps.
    This ensures predictions for all rows, even the first ones.
    
    Args:
        X (pd.DataFrame): Input features for test data.
        n_days (int): Number of timesteps per window.

    Returns:
        np.ndarray: Reshaped input for RNN (samples, timesteps, features).
    """
    X_windows = []
    X = X.values  # Converte para numpy para indexação direta
    
    for i in range(len(X)):
        if i < n_days - 1:
            # Se faltar histórico, preenche com o primeiro valor repetidamente
            pad = np.tile(X[0], (n_days - 1 - i, 1))
            window = np.vstack([pad, X[:i + 1]])
        else:
            # Caso normal: janela completa
            window = X[i - n_days + 1:i + 1]
        
        X_windows.append(window)
    
    return np.array(X_windows)


def include_time_based_features(df):
    # Extract time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.weekday
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Saturday=5, Sunday=6
    return df

df_train = include_time_based_features(df_train)


# Fetch holidays for each country for the years in the dataset
def get_holidays(country, years):
    holiday_dates = []
    for year in years:
        country_holidays = holidays.CountryHoliday(country, years=year)
        holiday_dates += list(country_holidays.keys())
    return pd.to_datetime(holiday_dates)

# Define the countries and their respective country codes
countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
country_codes = ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']

# Example dataset (replace with your actual dataset)
data = {
    'date': ['2010-01-01', '2010-12-25', '2011-07-04', '2012-12-31'],
    'country': ['Canada', 'Finland', 'Italy', 'Kenya'],
    'num_sold': [10, 15, 20, 25]
}

# Fetch holidays for each country for the years 2010-2019
years = range(2010, 2020)
holidays_dict = {country: get_holidays(code, years) for country, code in zip(countries, country_codes)}

# Add the holiday feature
def is_holiday(row):
    country_holidays = holidays_dict.get(row['country'], [])
    return int(row['date'] in country_holidays)

df_train['is_holiday'] = df_train.apply(is_holiday, axis=1)


# Based on https://www.kaggle.com/competitions/playground-series-s5e1/discussion/554349

import requests

def get_gdp_per_capita(alpha3, year):
    url='https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json'
    response = requests.get(url.format(alpha3,year)).json()
    return response[1][0]['value']

alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']



years = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])

gdp = np.array([
    [get_gdp_per_capita(alpha3, year) for year in years]
    for alpha3 in alpha3s
])
gdp = pd.DataFrame(gdp/gdp.sum(axis=0), index=alpha3s, columns=years)

alpha3_dict = {
    'Canada': 'CAN',
    'Finland': 'FIN',
    'Italy': 'ITA',
    'Kenya': 'KEN',
    'Norway': 'NOR',
    'Singapore': 'SGP'
}

# Create GDP feature
#df_train['GDP'] = df_train.apply(lambda s: np.log1p(gdp.loc[alpha3_dict[s['country']], s['date'].year]), axis=1)


X = df_train.drop(columns=['id', 'num_sold', 'date'])

y = df_train['num_sold']

# Encode categorical variables
X['country'] = X['country'].astype('category').cat.codes
X['store'] = X['store'].astype('category').cat.codes
X['product'] = X['product'].astype('category').cat.codes

# Adding cyclical features for better temporal encoding
#X['sin_day_of_week'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
#X['cos_day_of_week'] = np.cos(2 * np.pi * X['day_of_week'] / 7)

X['sin_month'] = np.sin(2 * np.pi * X['month'] / 12)
X['cos_month'] = np.cos(2 * np.pi * X['month'] / 12)

X['sin_day_of_year'] = np.sin(2 * np.pi * X['day_of_year'] / 365.25)
X['cos_day_of_year'] = np.cos(2 * np.pi * X['day_of_year'] / 365.25)

#X = X.drop(['day_of_week', 'month', 'day_of_year'], axis=1)

# Date column to numerical
#X['date'] = X['date'].apply(lambda x: x.toordinal())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



# Scale input features properly
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale target values properly
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))


# Reshape X
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

#n_days = 3
#X_train_reshaped, y_train_reshaped = create_rnn_windows(pd.DataFrame(X_train_scaled), pd.DataFrame(y_train_scaled), n_days)
#X_test_reshaped, y_test_reshaped = create_rnn_windows(pd.DataFrame(X_test_scaled), pd.DataFrame(y_test_scaled), n_days)

print('Shape:', X_train_reshaped.shape)


X_test.shape


# Define custom smoothed MAPE loss function
#def smoothed_mape_loss(y_true, y_pred):
#    epsilon = 1e-3
#    return tf.reduce_mean(tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), epsilon, tf.float32.max)))

def create_rnn_model(input_shape):
    model = Sequential()
    model.add(GRU(128, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mape', metrics=['mape'])
    return model


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Ensure the input data is of type float32
#X_train_reshaped = X_train_reshaped.astype('float32')
#y_train = y_train.astype('float32')

# Define the model
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])  # Timesteps, Features
#input_shape = X_train_reshaped.shape
model = create_rnn_model(input_shape)

# Define early stopping for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_reshaped, y_train_scaled,
    epochs=150,
    batch_size=64,
    validation_data=(X_test_reshaped, y_test_scaled),
    #callbacks=[early_stopping],
    verbose=1
)


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()


print(X_test_reshaped.shape)


# Evaluate on the test set
loss, mape = model.evaluate(X_test_reshaped, y_test_scaled, verbose=0)
print(f"Test Loss (MAPE): {mape:.2f}%")

# Predict and calculate MAPE
y_pred = np.expm1(model.predict(X_test_reshaped))
mape_score = mean_absolute_percentage_error(np.expm1(y_test_scaled), y_pred)
print(f"Test MAPE (manual): {mape_score:.2f}%")


y_pred[:10]


plt.plot(scaler_y.inverse_transform(np.expm1(y_test_scaled.reshape(-1, 1))))
plt.show()
plt.plot(scaler_y.inverse_transform(y_pred))


df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])

# Time based features
df_test = include_time_based_features(df_test)

# Holidays
df_test['is_holiday'] = df_test.apply(is_holiday, axis=1)

# Create GDP feature
#df_test['GDP'] = df_test.apply(lambda s: np.log1p(gdp.loc[alpha3_dict[s['country']], s['date'].year]), axis=1)


df_test


X_test = df_test.drop(columns=['id', 'date'])

# Encode categorical variables
X_test['country'] = X_test['country'].astype('category').cat.codes
X_test['store'] = X_test['store'].astype('category').cat.codes
X_test['product'] = X_test['product'].astype('category').cat.codes

# Adding cyclical features for better temporal encoding
#X_test['sin_day_of_week'] = np.sin(2 * np.pi * X_test['day_of_week'] / 7)
#X_test['cos_day_of_week'] = np.cos(2 * np.pi * X_test['day_of_week'] / 7)

X_test['sin_month'] = np.sin(2 * np.pi * X_test['month'] / 12)
X_test['cos_month'] = np.cos(2 * np.pi * X_test['month'] / 12)

X_test['sin_day_of_year'] = np.sin(2 * np.pi * X_test['day_of_year'] / 365.25)
X_test['cos_day_of_year'] = np.cos(2 * np.pi * X_test['day_of_year'] / 365.25)

#X_test = X_test.drop(['day_of_week', 'month', 'day_of_year'], axis=1)

# Date column to numerical
#X_test['date'] = X_test['date'].apply(lambda x: x.toordinal())

X_test.columns



X_test_reshaped.shape


X_test_scaled = scaler_X.transform(X_test)

#Cria janelas para treino e teste
#X_test_reshaped = create_rnn_windows_test(pd.DataFrame(X_test_scaled), n_days)

X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

predictions = model.predict(X_test_reshaped)
predictions = predictions.reshape(-1, 1)
predictions = scaler_y.inverse_transform(predictions)
predictions = np.expm1(predictions)


predictions[:5]


df_test['num_sold_predicted'] = predictions.flatten()


df_test['num_sold_predicted'] = np.round(df_test['num_sold_predicted'].astype(float)).astype(int)



mask_train = (df_train['country'] == 'Canada') & (df_train['store'] == 'Discount Stickers') & (df_train['product'] == 'Kaggle')
mask_test = (df_test['country'] == 'Canada') & (df_test['store'] == 'Discount Stickers') & (df_test['product'] == 'Kaggle')

# Aggregate num_sold and num_sold_predicted by date
actual_sales_by_date = np.expm1(df_train[mask_train].groupby(['date']).sum()['num_sold'])
predicted_sales_by_date = df_test[mask_test].groupby(['date']).sum()['num_sold_predicted'][:len(predictions)]

# Ensure the indices are properly aligned
actual_sales_by_date = actual_sales_by_date.reset_index()
predicted_sales_by_date = predicted_sales_by_date.reset_index()

#sales = scaler_y.transform(actual_sales_by_date['num_sold'].to_numpy().reshape(-1, 1)).flatten()
sales = actual_sales_by_date['num_sold'].to_numpy().reshape(-1, 1)
 
#sales_pred = scaler_y.transform(predicted_sales_by_date['num_sold_predicted'].to_numpy().reshape(-1, 1)).flatten()
sales_pred = predicted_sales_by_date['num_sold_predicted'].to_numpy()

# Visualize predictions, after X_test
plt.plot(sales, label='Actual Sales')
plt.plot(range(len(sales), len(sales) + len(sales_pred)), sales_pred, c='red', label='Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
#plt.ylim(0.95 * min(sales), 1.05 * max(sales))
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()


def create_mask(df, country, store, product):
    return (df['country'] == country) & (df['store'] == store) & (df['product'] == product)

def get_random_csp_mask(df):
    country = df['country'].unique()[np.random.randint(0, len(df['country'].unique()))]
    store = df['store'].unique()[np.random.randint(0, len(df['store'].unique()))]
    product = df['product'].unique()[np.random.randint(0, len(df['product'].unique()))]
    return country, store, product


def visualize_predictions(ax, df_train, df_test, mask_train, mask_test, sales_compare = []):

    # Aggregate num_sold and num_sold_predicted by date
    actual_sales_by_date = np.expm1(df_train[mask_train].groupby(['date']).sum()['num_sold'])
    predicted_sales_by_date = df_test[mask_test].groupby(['date']).sum()['num_sold_predicted'][:len(predictions)]

    # Ensure the indices are properly aligned
    actual_sales_by_date = actual_sales_by_date.reset_index()
    predicted_sales_by_date = predicted_sales_by_date.reset_index()

    #sales = scaler_y.transform(actual_sales_by_date['num_sold'].to_numpy().reshape(-1, 1)).flatten()
    sales = actual_sales_by_date['num_sold'].to_numpy().reshape(-1, 1)
    
    #sales_pred = scaler_y.transform(predicted_sales_by_date['num_sold_predicted'].to_numpy().reshape(-1, 1)).flatten()
    sales_pred = predicted_sales_by_date['num_sold_predicted'].to_numpy()


    # Visualize predictions, after X_test
    ax.plot(sales, label='Actual Sales')
    if len(sales_compare) > 0:
        ax.plot(range(len(sales), len(sales) + len(sales_compare)), sales_compare, c='lightgray', linestyle='dashed', label='Predicted Sales (To Compare)')
    ax.plot(range(len(sales), len(sales) + len(sales_pred)), sales_pred, c='red', label='Predicted Sales')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Sales')
    if len(sales) > 0:
        ax.set_ylim(0.90 * min(sales), 1.10 * max(sales))
    ax.legend()


df_konstantin = pd.read_csv('/kaggle/input/pgs501-model-2-additional-country-doy-factor/submission.csv')
df_konstantin['num_sold'][:5]


fig, axs = plt.subplots(3,3, figsize=(20, 10))
axs = axs.flatten()
for i in range(9):
    country, store, product = get_random_csp_mask(df_test)
    mask_train = create_mask(df_train, country, store, product)
    mask_test = create_mask(df_test, country, store, product)

    sales_compare = df_konstantin.join(df_test[mask_test].set_index('id'), on='id', how='inner')['num_sold']
    
    axs[i].set_title(f'({country},{store},{product})')
    visualize_predictions(axs[i], df_train, df_test, mask_train, mask_test, sales_compare)

plt.show()



window_size=90
n_pred = 60

country, store, product = get_random_csp_mask(df_test)
mask_train = create_mask(df_train, country, store, product)
mask_test = create_mask(df_test, country, store, product)

#mask_train = (df_train['country'] == 'Canada') & (df_train['store'] == 'Discount Stickers') & (df_train['product'] == 'Kaggle')
#mask_test = (df_test['country'] == 'Canada') & (df_test['store'] == 'Discount Stickers') & (df_test['product'] == 'Kaggle')

actual_sales = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date']).dropna(subset=['num_sold'])[mask_train]['num_sold'].to_list()[-window_size:]
_ = plt.figure(figsize=(15,5))

# Aggregate num_sold and num_sold_predicted by date
actual_sales_by_date = np.expm1(df_train[mask_train].groupby(['date']).sum()['num_sold'])

# Ensure the indices are properly aligned
actual_sales_by_date = actual_sales_by_date.reset_index()

#sales = scaler_y.transform(actual_sales_by_date['num_sold'].to_numpy().reshape(-1, 1)).flatten()
sales = actual_sales_by_date['num_sold'].to_numpy().reshape(-1, 1)

plt.title(f'({country},{store},{product})')
plt.plot(actual_sales, label='Actual Sales', ls='-')
plt.plot(sales[-window_size:], label='Sales (After processing)', ls='--')
plt.plot(np.expm1(np.log1p(actual_sales)), label='Actual Sales Transformed (manual)', ls='-.', c='lightgray')
plt.plot(range(window_size, window_size + n_pred), sales_pred[:n_pred], label='predicted sales')
plt.plot(range(window_size, window_size + n_pred), sales_compare[:n_pred], label='predicted sales (Konstantin)')
plt.legend(loc='upper left')
plt.show()


pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date']).dropna(subset=['num_sold'])[mask_train].tail(5)


df_test[mask_test].head(5)


output_df = pd.DataFrame({'id': df_test['id'], 'num_sold': df_test['num_sold_predicted']})
output_df



output_df.to_csv('/kaggle/working/submission.csv', index=False)






















































































































