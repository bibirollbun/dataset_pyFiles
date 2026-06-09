seed = 42


import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

print("train :", train.shape)
print("test :", test.shape)
print("sample_submission :", sample.shape)


train.isnull().sum().sort_values(ascending=False)


test.isnull().sum().sort_values(ascending=False)


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mode()[0])


import numpy as np

def feature_engineering(df):
    # 1. Temporal features
    df["month"] = ((df["day"] - 1) // 30) % 12 + 1  # Approximate month
    df["season"] = (df["month"] % 12) // 3
    
    df["day_of_week"] = df["day"] % 7  # Approximate day of the week
    df["is_weekend"] = df["day_of_week"].isin([6, 0]).astype(int)  # 0=Sunday, 6=Saturday

    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)
    
    # 2. Temperature Features
    df["temp_range"] = df["maxtemp"] - df["mintemp"]
    df["dewpoint_depression"] = df["temparature"] - df["dewpoint"]
    df["temp_diff"] = df["maxtemp"] - df["temparature"]

    # 3. Humidity & Pressure Features
    df["humidity_pressure_ratio"] = df["humidity"] / df["pressure"]
    df["dewpoint_humidity_ratio"] = df["dewpoint"] / df["humidity"]
    df["pressure_change"] = df["pressure"].diff().fillna(0)
    
    # 4. Cloud & Sunshine Features
    df["cloud_sunshine_ratio"] = df["cloud"] / (df["sunshine"] + 1e-6)  # Avoid division by zero
    df["sunshine_category"] = df["sunshine"] // 4
    
    # 5. Wind Features
    df["wind_speed_squared"] = df["windspeed"] ** 2
    df["wind_chill"] = df["temparature"] - (df["windspeed"] * 0.1)  # Simple approximation
    
    df["wind_x"] = np.sin(np.radians(df["winddirection"]))
    df["wind_y"] = np.cos(np.radians(df["winddirection"]))

    # 6. Get the value change rule
    for col in df.columns:
        if col in ['id', 'rainfall', 'day', 'month', 'day_of_week', 'is_weekend', 'sin_day', 'cos_day', 'season']:
            continue

        # 计算各列的差值
        diff_1 = (df[col] - df[col].shift(1)).fillna(0)
        diff_mean_3 = (df[col] - df[col].rolling(window=3, min_periods=1).mean().shift(1)).fillna(0)
        diff_mean_7 = (df[col] - df[col].rolling(window=7, min_periods=1).mean().shift(1)).fillna(0)
        diff_max_3 = (df[col] - df[col].rolling(window=3, min_periods=1).max().shift(1)).fillna(0)
        diff_max_7 = (df[col] - df[col].rolling(window=7, min_periods=1).max().shift(1)).fillna(0)
        diff_min_3 = (df[col] - df[col].rolling(window=3, min_periods=1).min().shift(1)).fillna(0)
        diff_min_7 = (df[col] - df[col].rolling(window=7, min_periods=1).min().shift(1)).fillna(0)

        # 把新列添加到列表中
        new_columns = [
            diff_1.rename(f'{col}_diff_1'),
            diff_mean_3.rename(f'{col}_diff_mean_3'),
            diff_mean_7.rename(f'{col}_diff_mean_7'),
            diff_max_3.rename(f'{col}_diff_max_3'),
            diff_max_7.rename(f'{col}_diff_max_7'),
            diff_min_3.rename(f'{col}_diff_min_3'),
            diff_min_7.rename(f'{col}_diff_min_7')
        ]

        df = pd.concat([df] + new_columns, axis=1)

    return df

    
train = feature_engineering(train)
test = feature_engineering(test)
train.head()


!pip install autogluon.tabular --no-cache-dir -q
!pip install ray==2.10.0
!pip install -U ipywidgets


from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    path='/kaggle/working/Autogluon',
    label='rainfall',
    problem_type='binary',
    eval_metric='accuracy',
    verbosity=2,
    learner_kwargs={'ignored_columns': ['ID']}
)
                                 
predictor.fit(
    train_data=train, 
    presets='best_quality',
    time_limit=None,
    num_gpus=2,
    dynamic_stacking=False,
    num_bag_folds=10,
    num_stack_levels=3,
)


predictor.leaderboard()


test_prediction = predictor.predict_proba(test)
submission = pd.DataFrame({'id': sample['id'], 'rainfall': test_prediction[1]})
print(submission.head())
submission.to_csv('submission.csv', index=False)

