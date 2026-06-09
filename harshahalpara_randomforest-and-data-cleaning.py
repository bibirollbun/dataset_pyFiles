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


! nproc --all


import numpy as np
import pandas as pd


df = pd.read_csv("/kaggle/input/simple-housing-price-prediction/train.csv")
test_df = pd.read_csv('/kaggle/input/simple-housing-price-prediction/test.csv')


obj_col = df.select_dtypes(include=['object']).columns
for i in obj_col:
    print(i,df[i].value_counts())


def clean_data(df, is_train=True, encodings=None):
    df = df.copy()
    
    # Drop unnecessary columns
    if 'house_id' in df.columns:
        df = df.drop(['house_id'], axis=1)
    df = df.drop(['block'], axis=1)
    
    # Handle dates
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df = df.drop(['date'], axis=1)
    
    # One-hot encode location
    location_dummies = pd.get_dummies(df['location'], prefix='loc')
    df = pd.concat([df, location_dummies], axis=1)
    df = df.drop(['location'], axis=1)
    
    # Convert room type to numeric
    def num_in_room(type_val):
        if type_val == 'EXECUTIVE':
            return 10
        if type_val == 'MULTI-GENERATION':
            return 15
        return int(type_val[0])
    
    df['type'] = df['type'].apply(num_in_room)
    
    # Building age
    df['building_age'] = df['year'] - df['commence_date']
    df = df.drop(['commence_date'], axis=1)
    
    # Group streets
    street_counts = df['street'].value_counts()
    top_streets = street_counts.head(20).index
    df['street_grouped'] = df['street'].apply(lambda x: x if x in top_streets else 'Other')
    df = df.drop(['street'], axis=1)
    
    # Storey average
    def convert_storey_to_avg(storey_range):
        start, end = storey_range.split(' TO ')
        return (int(start) + int(end)) / 2
    df['storey_avg'] = df['storey_range'].apply(convert_storey_to_avg)
    df = df.drop(['storey_range'], axis=1)
    
    # Target encoding
    if is_train:
        flat_model_means = df.groupby('flat_model')['price'].mean()
        street_means = df.groupby('street_grouped')['price'].mean()
        encodings = {'flat_model': flat_model_means, 'street': street_means}
    else:
        flat_model_means = encodings['flat_model']
        street_means = encodings['street']
    
    df['flat_model_encoded'] = df['flat_model'].map(flat_model_means).fillna(flat_model_means.mean())
    df['street_encoded'] = df['street_grouped'].map(street_means).fillna(street_means.mean())
    df = df.drop(['flat_model', 'street_grouped'], axis=1)
    
    return df, encodings



df.head()


obj_col = df.select_dtypes(include=['object']).columns
for i in obj_col:
    print(i,len(df[i].value_counts()))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Your existing data preprocessing
train_clean, encodings = clean_data(df, is_train=True)
test_clean, _ = clean_data(test_df, is_train=False, encodings=encodings)

X = train_clean.drop(['price'], axis=1)
y = train_clean['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.001, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
test_clean_scaled = scaler.transform(test_clean)

# Define multiple algorithms
model = RandomForestRegressor(n_estimators=200, random_state=42,n_jobs=-1)

# Store results
results = {}
trained_models = {}

print("="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)

# print(f"\nTraining {name}...")

model.fit(X_train_scaled, y_train)
trained_models = model

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Store results
results = {
    'Train R²': train_r2,
    'Test R²': test_r2,
    'Train RMSE': train_rmse,
    'Test RMSE': test_rmse,
    'Train MAE': train_mae,
    'Test MAE': test_mae,
    'Overfitting': train_r2 - test_r2
    }
    
    # Print results
print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
print(f"  Train RMSE: ${train_rmse:,.0f} | Test RMSE: ${test_rmse:,.0f}")
    
    # Check for overfitting
if train_r2 - test_r2 > 0.1:
    print(f"  ⚠️ Overfitting detected!")
elif test_r2 > 0.7:
    print(f"  ✅ Excellent performance!")
elif test_r2 > 0.5:
    print(f"  ✅ Good performance!")



# Load test data
sample_submission = pd.read_csv('/kaggle/input/simple-housing-price-prediction/sample_submission.csv')
predictions = model.predict(test_clean_scaled)

submission = pd.DataFrame({
    'house_id': sample_submission['house_id'],
    'price': predictions
})
print(submission.head())
submission.to_csv('submission.csv', index=False)
print("Done! submission.csv created")


