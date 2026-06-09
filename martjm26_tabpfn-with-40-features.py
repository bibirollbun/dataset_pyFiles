!pip install autogluon
!pip install autogluon.tabular[all]


import numpy as np
import pandas as pd
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
orig = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
display(train.info(), train.head(), train.describe().T)


def feature_engineering(df):
    df = df.copy()

    # some clouds give better results
    df['cloud88'] =  (df.cloud==88).astype(int)
    df['cloud90+'] =  (df.cloud>90).astype(int)
    
    # Convert 'day' to datetime
    df['day'] = pd.to_datetime(df['day'], errors='coerce')
    
    # Extract temporal features
    df['month'] = df['day'].dt.month
    df['day_of_week'] = df['day'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Temperature features
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['avg_temp'] = (df['maxtemp'] + df['mintemp']) / 2
    df['temp_deviation'] = df['temparature'] - df['avg_temp']
    
    # Dew point depression
    df['dew_point_depression'] = df['temparature'] - df['dewpoint']
    
    # Wind direction - sine and cosine transformation
    df['wind_dir_rad'] = np.deg2rad(df['winddirection'])
    df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
    df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
    df.drop(columns=['wind_dir_rad'], inplace=True)
    
    # Wind chill factor (simplified version)
    df['wind_chill'] = 13.12 + 0.6215 * df['temparature'] - 11.37 * (df['windspeed']**0.16) + 0.3965 * df['temparature'] * (df['windspeed']**0.16)
    
    # Interaction features
    df['humidity_temp'] = df['humidity'] * df['temparature']
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    
    # Rolling statistical features
    df['rolling_temp_mean'] = df['avg_temp'].rolling(window=7).mean()
    df['rolling_wind_mean'] = df['windspeed'].rolling(window=7).mean()
    df['rolling_humidity_mean'] = df['humidity'].rolling(window=7).mean()
    
    # Lag features
    df['temp_lag_1'] = df['avg_temp'].shift(1)
    df['humidity_lag_1'] = df['humidity'].shift(1)
    df['windspeed_lag_1'] = df['windspeed'].shift(1)
    
    # Pressure-Temperature interaction
    df['pressure_temp_interaction'] = df['pressure'] * df['avg_temp']
    # Wind-Speed-Temperature interaction
    df['windspeed_temp_interaction'] = df['windspeed'] * df['avg_temp']
    
    # Sunshine-Cloud interaction
    df['sunshine_cloud_interaction'] = df['sunshine'] * df['cloud']
    
    # Season feature
    df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
                                      'Summer' if 6 <= x <= 8 else
                                      'Autumn' if 9 <= x <= 11 else 'Winter')

    for c in ['pressure', 'maxtemp', 'temparature', 'humidity']:
        for gap in [1]:
            df[c+f"_shift{gap}"] = df[c].shift(gap)
            df[c+f"_diff{gap}"] = df[c].diff(gap)

    # Binary encoding for season
    df = pd.get_dummies(df, columns=['season'], drop_first=True)
    # Drop original 'day' column
    df.drop(columns=['day'], inplace=True)
    
    return df


orig.columns = orig.columns.str.strip()
orig['rainfall'] = orig['rainfall'].str.lower().map({'yes': 1, 'no': 0})
train = train.drop(columns=['id'])


train = pd.concat([orig, train], axis=0, ignore_index=True)
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

train = train.fillna(train.mean())
test = test.fillna(test.mean())

# Drop 'id' column from train and test datasets
train.drop(columns=['id'], inplace=True, errors='ignore')
test.drop(columns=['id'], inplace=True, errors='ignore')

train = feature_engineering(train)
test = feature_engineering(test)

#train = train.fillna(train.mean())
#test = test.fillna(test.mean())


display(train.info(), test.info())


from autogluon.tabular import TabularDataset, TabularPredictor

# Convert dataset into an AutoGluon TabularDataset
train_data = TabularDataset(train)
test_data = TabularDataset(test)

# Define the target column
label = "rainfall"  # Replace with actual target column name


# Train AutoGluon model with GPU acceleration
predictor = TabularPredictor(label=label, verbosity = 2, eval_metric = "roc_auc").fit(
    train_data,
    presets="best_quality",
    time_limit=3600,
    ag_args_fit={"num_gpus": 1}  # Forces GPU usage
)


# predictor = TabularPredictor.load("/kaggle/working/AutogluonModels/ag-20250305_141137")

# predict on test
test_probs = predictor.predict_proba(test_data)

# View leaderboard of models trained
predictor.leaderboard(train_data, silent=True)


# Get feature importance
feature_importance = predictor.feature_importance(train_data)
print(feature_importance)


test_probs


# create submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = test_probs[1]
submission.to_csv('submission_final.csv', index=False)
submission.head()

