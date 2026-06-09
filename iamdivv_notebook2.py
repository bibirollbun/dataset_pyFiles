# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
train_df.head()


train_df.shape


train_df.dtypes


train_df.columns


train_df.isnull().sum()


train_df['Penalty'].value_counts()


#handling missing values in penalty column
train_df['Penalty'] = train_df['Penalty'].fillna('No Penalty')


train_df['Penalty'].isnull().sum()


penalty_map = {
    'No Penalty': 0,
    'DNS': 1,
    'DNF': 2,
    'Ride Through': 3,
    '+3s': 4,
    '+5s': 5
}
train_df['Penalty_encoded'] = train_df['Penalty'].map(penalty_map)


train_df['Penalty_encoded'].unique()


leakage_check_cols = ['Avg_Speed_kmh', 'Championship_Points',
                     'Championship_Position','position', 
                     'points', 'starts', 'finishes', 'with_points',
       'podiums', 'wins', 'Penalty_encoded' ]
plt.figure(figsize=(12,6))
sns.heatmap(train_df[leakage_check_cols + ['Lap_Time_Seconds']].corr(), annot=True, cmap='coolwarm')
plt.title("Potential Leakage Feature Correlation")
plt.show()


# dividing features 

irrelevant_cols = ['Unique ID', 'Rider_ID', 'Penalty','Session','Avg_Speed_kmh',
                  'sequence' , 'position' , 'points', 'rider', 'team', 'bike','shortname',
       'circuit_name', 'rider_name', 'team_name', 'bike_name', 'min_year', 'max_year']

numerical_cols = ['Circuit_Length_km' ,'Laps' , 
                  'Grid_Position',
                  'Championship_Points', 
                  'Championship_Position','Corners_per_Lap',
                  'Tire_Degradation_Factor_per_Lap',
                  'Pit_Stop_Duration_Seconds', 'years_active',
                  'Penalty_encoded'
                 ]

categorical_cols = ['category_x', 'Tire_Compound_Front','Tire_Compound_Rear', 'Track_Condition', 'weather', 'track']

weather_cols = ['Humidity_%', 'Ambient_Temperature_Celsius', 'Track_Temperature_Celsius', 'air', 'ground']

performance_cols = ['starts', 'finishes', 'with_points','podiums','wins']



train_df['win_rate'] = train_df['wins'] / train_df['starts'].replace(0,1)
train_df['podium_rate'] = train_df['podiums'] / train_df['starts'].replace(0,1)
train_df['points_rate'] = train_df['with_points'] / train_df['starts'].replace(0,1)
train_df['finish_rate'] = train_df['finishes'] / train_df['starts'].replace(0,1)

# weather based features
train_df['temp_diff'] = train_df['Track_Temperature_Celsius'] - train_df['Ambient_Temperature_Celsius']
train_df['air_ground_diff'] = train_df['air'] - train_df['ground']

# categorical encoding
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)


feature_cols = numerical_cols  + weather_cols
derived_cols = ['win_rate', 'podium_rate', 'points_rate', 'finish_rate', 'temp_diff', 'air_ground_diff']

# encoded categorical cols 
encoded_cols = [col for col in train_df.columns if any(prefix in col for prefix in ['category_x_',
                                                                                    'Tire_Compound_Front_','Tire_Compound_Rear_',
                                                                                    'Track_Condition_', 'weather_', 'track_'])]
# final feature list
feature_cols = feature_cols + derived_cols + encoded_cols


feature_cols


X = train_df[feature_cols]
y = train_df['Lap_Time_Seconds']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)


val_df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/val.csv')
val_df



val_df.isnull().sum()


#handling missing values in penalty column
val_df['Penalty'] = val_df['Penalty'].fillna('No Penalty')


val_df['Penalty'].isnull().sum()


penalty_mapping = {
    'No Penalty': 0,
    'DNS': 1,
    'DNF': 2,
    'Ride Through': 3,
    '+3s': 4,
    '+5s': 5
}
val_df['Penalty_encoded'] = val_df['Penalty'].map(penalty_mapping)


val_df['Penalty_encoded'].isnull().sum()


val_df['win_rate'] = val_df['wins'] / val_df['starts'].replace(0,1)
val_df['podium_rate'] = val_df['podiums'] / val_df['starts'].replace(0,1)
val_df['points_rate'] = val_df['with_points'] / val_df['starts'].replace(0,1)
val_df['finish_rate'] = val_df['finishes'] / val_df['starts'].replace(0,1)

# weather based features
val_df['temp_diff'] = val_df['Track_Temperature_Celsius'] - val_df['Ambient_Temperature_Celsius']
val_df['air_ground_diff'] = val_df['air'] - val_df['ground']

# categorical encoding
val_df = pd.get_dummies(val_df, columns=categorical_cols, drop_first=True)


X_val = val_df[feature_cols]
y_val = val_df['Lap_Time_Seconds']
x_val_scaled = scaler.transform(X_val)



# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

model = RandomForestRegressor(n_estimators=30, n_jobs = -1, random_state=42)
model.fit(X_train_scaled, y)

y_pred = model.predict(x_val_scaled)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print('Validation MSE: ', mse)
print('Validation R2: ', r2)


importances = model.feature_importances_
feature_names = X.columns

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(35), palette='viridis')
plt.title('Top 20 Important Features')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


test_df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
test_df.head()


test_df.shape


test_df.columns



test_df.isnull().sum()


#handling missing values in penalty column
test_df['Penalty'] = test_df['Penalty'].fillna('No Penalty')


penalty_mapping_test = {
    'No Penalty': 0,
    'DNS': 1,
    'DNF': 2,
    'Ride Through': 3,
    '+3s': 4,
    '+5s': 5
}
test_df['Penalty_encoded'] = test_df['Penalty'].map(penalty_mapping_test)


test_df['Penalty_encoded'].isnull().sum()


test_df['win_rate'] = test_df['wins'] / test_df['starts'].replace(0,1)
test_df['podium_rate'] = test_df['podiums'] / test_df['starts'].replace(0,1)
test_df['points_rate'] = test_df['with_points'] / test_df['starts'].replace(0,1)
test_df['finish_rate'] = test_df['finishes'] / test_df['starts'].replace(0,1)

# weather based features
test_df['temp_diff'] = test_df['Track_Temperature_Celsius'] - test_df['Ambient_Temperature_Celsius']
test_df['air_ground_diff'] = test_df['air'] - test_df['ground']

# categorical encoding
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)


Id = test_df['Unique ID']
test_df = test_df[X.columns]


test_scaled = scaler.transform(test_df)


test_predictions = model.predict(test_scaled)


submission = pd.DataFrame({
    'Unique ID': Id,
    'Lap_Time_Seconds': test_predictions
})


submission.to_csv('submission.csv', index=False)




