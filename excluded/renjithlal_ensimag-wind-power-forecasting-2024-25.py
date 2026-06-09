# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/ensimag-mmis-2024/train.csv')


df


df.describe()


df.tail()


df.isna().sum()# to confirm whetherbthere is any na


df['date']=pd.to_datetime(df['date'])


# scale the datas except date
from sklearn.preprocessing import MinMaxScaler

model1 = MinMaxScaler(feature_range = (0,1))
#model = MinMaxScaler()



# Select only the numerical columns for scaling
numerical_cols = ['u10', 'v10', 'u100', 'v100', 'production']
scaled_df = model1.fit_transform(df[numerical_cols])


df_transformed=pd.DataFrame(scaled_df,columns=['u10','v10','u100','v100','production'])


 df_transformed # we hve created a data frame with different column


X=df_transformed.iloc[:,0:4]


X


y=df_transformed['production']


y=pd.DataFrame(y,columns=['production'])


# @title production

from matplotlib import pyplot as plt
y['production'].plot(kind='line', figsize=(8, 4), title='production')
plt.gca().spines[['top', 'right']].set_visible(False)


result=pd.concat([df['date'],y['production']],axis=1)


result1=pd.concat([result,X],axis=1)


result1#all scaled except date 


col = result1.pop('production')
result1['production'] = col


result1.columns


result1


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# Corrected the date range generation
dates = pd.date_range(start='2020-01-01 01:00:00', periods=len(result1), freq='H')
n_samples = len(dates)
data = {
    'date': dates,
    'feature1': result1.u10,
    'feature2': result1.v10,
    'feature3': result1.u100,
    'feature4': result1.v100,
    'target': result1.production
}
df = pd.DataFrame(data)

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'])

# Function to create lagged features
def create_lagged_features(df, lags=3):
    df_lagged = df.copy()
    for lag in range(1, lags + 1):
        for col in ['feature1', 'feature2', 'feature3', 'feature4','target']:
            df_lagged[f'{col}_lag{lag}'] = df_lagged[col].shift(lag)
    df_lagged['target_t+1'] = df_lagged['target'].shift(-1)
    df_lagged = df_lagged.dropna()
    return df_lagged

# Create lagged features
lags = 3
df_lagged = create_lagged_features(df, lags)

# Define features and target
feature_cols = [col for col in df_lagged.columns if 'lag' in col]
X = df_lagged[feature_cols]
y = df_lagged['target_t+1']

# Split data (80% train, 20% validation)
train_size = int(0.8 * len(df_lagged))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Validation Mean Squared Error: {mse:.4f}')

# Function to predict future values iteratively
def predict_future(df, model, feature_cols, n_future=7994, lags=3):
    future_dates = pd.date_range(start='2021-01-01 00:00:00', periods=n_future, freq='H')
    predictions = []
    last_row = df.iloc[-1][['feature1', 'feature2', 'feature3','feature4','target']].copy()
    current_data = df.iloc[-lags:][['feature1', 'feature2', 'feature3', 'feature4','target']].copy()


    for _ in range(n_future):
        # Prepare input for prediction
        input_data = []
        for lag in range(1, lags + 1):
            for col in ['feature1', 'feature2', 'feature3', 'feature4','target']:
                input_data.append(current_data[f'{col}'].iloc[-lag])
        input_data = np.array(input_data).reshape(1, -1)

        # Predict next target
        pred = model.predict(input_data)[0]
        predictions.append(pred)

        # Update current_data with the new prediction
        new_row = pd.Series({
            'feature1': last_row['feature1'],  # Assume features remain constant
            'feature2': last_row['feature2'],
            'feature3': last_row['feature3'],
            'feature4': last_row['feature4'],
            'target': pred
        })
        current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
        current_data = current_data.iloc[1:]  # Keep only the last `lags` rows

    # Create result DataFrame
    result = pd.DataFrame({
        'date': future_dates,
        'predicted_target': predictions
    })
    return result

# Predict 7982 future intervals
future_preds = predict_future(df, model, feature_cols, n_future=7994, lags=lags)

# Create a dummy array with the same number of columns as the original scaled data
dummy_array = np.zeros((len(future_preds), 5))

# Place the predicted target values in the last column of the dummy array
dummy_array[:, 4] = future_preds['predicted_target']

# Use the scaler to inverse transform the dummy array
future_preds['predicted_target_unscaled'] = model1.inverse_transform(dummy_array)[:, 4]


# Display the future predictions with the unscaled values
display(future_preds)
# Output predictions

print(future_preds)





future_preds=pd.DataFrame(future_preds)


future_preds.drop(columns=['predicted_target'],inplace=True)


submission = future_preds.rename(columns={'predicted_target_unscaled':'production'})


submission


n_delete=12
submission = submission.sample(n=len(submission) - n_delete, random_state=42)



submission = submission.fillna(0)


submission.to_csv('submission.csv', index=False, encoding='utf-8')


submission

