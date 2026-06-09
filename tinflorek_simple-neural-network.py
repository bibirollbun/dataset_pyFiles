import numpy as np
import pandas as pd

from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')


def feature_engeenering(df):
    # Assuming df is your dataframe
    # First, make sure your date column is properly converted to datetime
    df['date'] = pd.to_datetime(df['day'])
    
    # Now create all features, but avoid mixing date objects directly with numerical values
    # Seasonal features - extract numeric components from the date
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Define seasons
    df['season'] = pd.cut(df['date'].dt.month, 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'], 
                         include_lowest=True)
    
    # Temperature features
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    # Use numeric columns only for calculations
    df['temp_gradient'] = df['temparature'].diff()
    
    # Humidity features
    df['dewpoint_depression'] = df['temparature'] - df['dewpoint']
    
    # Pressure features
    df['pressure_change'] = df['pressure'].diff()
    df['pressure_change_3d'] = df['pressure'] - df['pressure'].shift(3)
    
    # Wind features
    df['wind_rad'] = np.radians(df['winddirection'])
    df['wind_u'] = -df['windspeed'] * np.sin(df['wind_rad'])
    df['wind_v'] = -df['windspeed'] * np.cos(df['wind_rad'])
    
    # Cyclic features - based on the numeric day_of_year, not the date object
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # Interaction terms
    df['humidity_temp'] = df['humidity'] * df['temparature']
    df['cloud_temp'] = df['cloud'] * df['temparature']

    # Temperature trends
    df['temp_3d_slope'] = df['temparature'].rolling(3).apply(
        lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 2 else np.nan
    )
    
    # Pressure tendency classification
    df['pressure_change'] = df['pressure'].diff()
    df['pressure_tendency'] = pd.cut(
        df['pressure_change'],
        bins=[-np.inf, -1.0, -0.1, 0.1, 1.0, np.inf],
        labels=['Falling rapidly', 'Falling', 'Steady', 'Rising', 'Rising rapidly']
    )
    
    # More complex wind features
    df['wind_persistence'] = df['winddirection'].rolling(3).std()
    
    # Estimate wet-bulb temperature (simplified)
    df['wet_bulb_temp'] = df['temparature'] - ((100 - df['humidity']) / 5)
    
    # Moisture advection proxy
    df['moisture_advection'] = df['windspeed'] * df['humidity'] / 100
    
    # Cloud-temperature relationship
    df['cloud_temp_ratio'] = df['cloud'] / (df['temparature'] + 273.15)  # Using absolute temp
    
    # Advanced cyclical features for multiple periods
    for period in [365, 30, 7]:  # Annual, monthly, weekly
        df[f'cycle_sin_{period}d'] = np.sin(2 * np.pi * df['day_of_year'] / period)
        df[f'cycle_cos_{period}d'] = np.cos(2 * np.pi * df['day_of_year'] / period)
    
    # Temperature and dewpoint spread (indicator of atmospheric moisture capacity)
    df['temp_dewpoint_spread'] = df['temparature'] - df['dewpoint']
    
    # Estimate lifted condensation level (simplified, in meters)
    df['lcl_height'] = 125 * (df['temparature'] - df['dewpoint'])
    
    # Stability index (simplified K-index)
    df['k_index_proxy'] = df['temparature'] - df['temp_dewpoint_spread']
    
    # Interaction between wind and pressure
    df['wind_pressure_interaction'] = df['windspeed'] * df['pressure_change'].abs()
          
    return df.drop('date', axis=1)

train_data = feature_engeenering(train_data)
test_data = feature_engeenering(test_data)


X = train_data.drop('rainfall', axis=1)
y = train_data['rainfall']
test = test_data.copy()


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['season', 'pressure_tendency'])
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)
test = preprocessor.fit_transform(test)


mean_val = np.nanmean(X)
X = np.where(np.isnan(X), mean_val, X)

mean_val = np.nanmean(test)
test = np.where(np.isnan(test), mean_val, test)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.fit_transform(test)


model = Sequential([
    Dense(8, activation='relu', input_dim=X.shape[1]),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.01), 
             optimizer='Adam', 
             metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])


model.fit(X, y, validation_split=0.2, epochs=15)


preds = model.predict(test)


sub = pd.DataFrame({'id': test_data.index, 'rainfall': np.squeeze(preds)})
sub.to_csv('submission.csv', index=False)

