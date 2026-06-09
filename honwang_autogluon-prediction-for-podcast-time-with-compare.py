!pip install autogluon --quiet


# âœ… AutoGluon SOTA Notebook for Sub-12 RMSE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

# ğŸ§± Step 1: Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# âœ… Step 2: Full Feature Engineering
train.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)
test.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)

train['Host_Guest_Mean'] = (train['Host_Popularity_percentage'] + train['Guest_Popularity_percentage']) / 2
test['Host_Guest_Mean'] = (test['Host_Popularity_percentage'] + test['Guest_Popularity_percentage']) / 2

train['Ad_per_minute'] = train['Number_of_Ads'] / (train['Episode_Length_minutes'] + 1)
test['Ad_per_minute'] = test['Number_of_Ads'] / (test['Episode_Length_minutes'] + 1)

train['Pop_x_Ads'] = train['Host_Guest_Mean'] * train['Number_of_Ads']
test['Pop_x_Ads'] = test['Host_Guest_Mean'] * test['Number_of_Ads']

train['Episode_Length_per_Ad'] = train['Episode_Length_minutes'] / (train['Number_of_Ads'] + 1)
test['Episode_Length_per_Ad'] = test['Episode_Length_minutes'] / (test['Number_of_Ads'] + 1)

for col in ['Episode_Length_minutes', 'Ad_per_minute']:
    train[f'log_{col}'] = np.log1p(train[col])
    test[f'log_{col}'] = np.log1p(test[col])

# Day Encoding
day_map = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
train['Publication_Day'] = train['Publication_Day'].map(day_map)
test['Publication_Day'] = test['Publication_Day'].map(day_map)
train['day_sin'] = np.sin(2 * np.pi * train['Publication_Day'] / 7)
train['day_cos'] = np.cos(2 * np.pi * train['Publication_Day'] / 7)
test['day_sin'] = np.sin(2 * np.pi * test['Publication_Day'] / 7)
test['day_cos'] = np.cos(2 * np.pi * test['Publication_Day'] / 7)

# Time Encoding
time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
train['Publication_Time'] = train['Publication_Time'].map(time_map)
test['Publication_Time'] = test['Publication_Time'].map(time_map)

# Fill Missing
fill_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']
for col in fill_cols:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())

# Cap Outliers
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].clip(upper=train['Episode_Length_minutes'].quantile(0.995))
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].clip(upper=train['Episode_Length_minutes'].quantile(0.995))
train['Ad_per_minute'] = train['Ad_per_minute'].clip(upper=5)
test['Ad_per_minute'] = test['Ad_per_minute'].clip(upper=5)

# âœ… Features
features = [
    'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Length_minutes',
    'Genre', 'Publication_Day', 'Publication_Time', 'Number_of_Ads',
    'Host_Guest_Mean', 'Ad_per_minute', 'Pop_x_Ads', 'Episode_Length_per_Ad',
    'log_Episode_Length_minutes', 'log_Ad_per_minute', 'day_sin', 'day_cos'
]
label = 'Listening_Time_minutes'
train_data = train[features + [label]]
test_data = test[features]

# ğŸ§ª Step 3: Split
train_data_split, val_data_split = train_test_split(train_data, test_size=0.1, random_state=42)

# ğŸš€ Step 4: AutoGluon Training
predictor = TabularPredictor(label=label, eval_metric='rmse').fit(
    train_data=train_data_split,
    presets='best_quality',
    time_limit=5400,  # 90 min
    num_bag_folds=10,
    num_stack_levels=3,
    verbosity=2,
    excluded_model_types=['KNN']
)





# ğŸ”� Step 5: Full Refit
# Step 5: Full Refit - keep original object
predictor.refit_full()

# ğŸ“Š Step 6: Leaderboard
leaderboard = predictor.leaderboard(val_data_split, silent=True)
print(leaderboard)


# ğŸ“¦ Step 7: Predict and Submit
preds = predictor.predict(test_data)
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': preds
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved: submission.csv")

