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


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df.head()


df.shape


df.describe()


df.info()


df.isnull()


df.isnull().sum()


df.head()


test.head()


def show_unique_values(dataframe, columns):
    for col in columns:
        if col in dataframe.columns:
            print(f"ğŸŸ© Unique values in '{col}':")
            print(dataframe[col].unique())
            print('------------------------------------')
        else:
            print(f"âš ï¸� Column '{col}' not found in dataframe.")
            print('------------------------------------')



columns = ['road_type', 'num_lanes', 'speed_limit', 'lighting', 
           'time_of_day', 'weather', 'num_reported_accidents']

show_unique_values(df, columns)
show_unique_values(test, columns)


def encode_categorical_features(dataframe):
    road_type_map = {'urban': 1, 'rural': 2, 'highway': 3}
    lighting_map = {'daylight': 1, 'dim': 2, 'night': 3}
    weather_map = {'rainy': 1, 'clear': 2, 'foggy': 3}
    time_of_day_map = {'afternoon': 1, 'evening': 2, 'morning': 3}

    # Safe mapping
    for col, mapping in {
        'road_type': road_type_map,
        'lighting': lighting_map,
        'weather': weather_map,
        'time_of_day': time_of_day_map
    }.items():
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].map(mapping).fillna(0)

    bool_columns = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in bool_columns:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(int)

    return dataframe



def encode_categorical_features(dataframe):

    # Define mappings
    road_type_map = {'urban': 1, 'rural': 2, 'highway': 3}
    lighting_map = {'daylight': 1, 'dim': 2, 'night': 3}
    weather_map = {'rainy': 1, 'clear': 2, 'foggy': 3}
    time_of_day_map = {'afternoon': 1, 'evening': 2, 'morning': 3}

    # Apply mappings (only if columns exist)
    if 'road_type' in dataframe.columns:
        dataframe['road_type'] = dataframe['road_type'].map(road_type_map)
    if 'lighting' in dataframe.columns:
        dataframe['lighting'] = dataframe['lighting'].map(lighting_map)
    if 'weather' in dataframe.columns:
        dataframe['weather'] = dataframe['weather'].map(weather_map)
    if 'time_of_day' in dataframe.columns:
        dataframe['time_of_day'] = dataframe['time_of_day'].map(time_of_day_map)

    # Convert boolean-like columns to integers if they exist
    bool_columns = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in bool_columns:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(int)

    return dataframe

df = encode_categorical_features(df)
test = encode_categorical_features(test)


#columns = ['road_type', 'num_lanes', 'speed_limit', 'lighting', 'time_of_day', 'weather', 'num_reported_accidents']
for i in df.columns:
    print(f"Unique values in {i}:")
    print(df[i].unique())
    print('-------')
    print('-------')


#columns = ['road_type', 'num_lanes', 'speed_limit', 'lighting', 'time_of_day', 'weather', 'num_reported_accidents']
for i in test.columns:
    print(f"Unique values in {i}:")
    print(test[i].unique())
    print('-------')
    print('-------')


df = df.drop_duplicates()
df.head()


test = test.drop_duplicates()
test.head()


print(df.duplicated().sum())
print(test.duplicated().sum())


from sklearn.preprocessing import MinMaxScaler

def normalize_numerical_features(dataframe, scaler=None):

    numeric_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

    # Fit the scaler only if not provided (for training data)
    if scaler is None:
        scaler = MinMaxScaler()
        dataframe[numeric_cols] = scaler.fit_transform(dataframe[numeric_cols])
    else:
        # Use the already fitted scaler (for test data)
        dataframe[numeric_cols] = scaler.transform(dataframe[numeric_cols])

    return dataframe, scaler



df, scaler = normalize_numerical_features(df)
test, _ = normalize_numerical_features(test, scaler)


df.head()


test.head()


df = df.drop('id', axis=1)


X = df.drop('accident_risk', axis=1)
y = df['accident_risk']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


gb_model = GradientBoostingRegressor(
    n_estimators=100,        # number of boosting stages
    learning_rate=0.1,       # smaller â†’ slower but better accuracy
    max_depth=4,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)


gb_model.score(X_test, y_test)


from sklearn.metrics import r2_score, mean_squared_error


r2 = r2_score(y_test, y_pred_gb)
mse = mean_squared_error(y_test, y_pred_gb)
print("Validation RÂ² Score:", r2)
print("Validation MSE:", mse)


print(len(test))
print(len(submission_preds))


# Make predictions without using the 'id' column
submission_preds = gb_model.predict(test.drop('id', axis=1))


submission = pd.DataFrame({
    'id': test['id'],                  # take IDs from the test file
    'accident_risk': submission_preds  # predicted values
})


# âœ… Save submission file in CSV format
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file 'submission.csv' created successfully!")
print(submission.head())







