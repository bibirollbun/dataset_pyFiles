# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
!pip install autogluon.tabular -q
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
!pip install pycountry holidays -q


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")

train_df.head()


test_df.head()


sample_df.head()


# Get the value counts
country_counts = train_df['country'].value_counts()

# Plot the data as a pie chart
plt.figure(figsize=(8, 8))
country_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'orange', 'green', 'red', 'purple', 'yellow'])
plt.title('Distribution of Countries', fontsize=16)
plt.ylabel('')
plt.tight_layout()


test_df['country'].value_counts()


# Null distribution
for col in train_df:
    print(col, train_df[col].isna().sum() / len(train_df) * 100)


train_df['store'].value_counts()


# Null distribution
for col in test_df:
    print(col, test_df[col].isna().sum() / len(train_df) * 100)


train_df.dropna(inplace = True)


train_df['is train'] = np.ones(len(train_df))
test_df['is train'] = np.zeros(len(test_df))

# combine the two dataframes
df = pd.concat([train_df, test_df], ignore_index = True)
df.shape


df.head()


# decompose the date

# Convert the date column to a datetime object
df['date'] = pd.to_datetime(df['date'])

# Decompose the date into components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['week'] = df['date'].dt.isocalendar().week 
df['quarter'] = df['date'].dt.quarter 
df['day_of_week'] = df['date'].dt.dayofweek
df['is_leap_year'] = df['date'].dt.is_leap_year
df['day_name'] = df['date'].dt.day_name()   # Full name of the day
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int) # True for Saturday/Sunday
df['is_month_start'] = df['date'].dt.is_month_start
df['is_month_end'] = df['date'].dt.is_month_end
df['is_year_start'] = df['date'].dt.is_year_start
df['is_year_end'] = df['date'].dt.is_year_end
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

df.head()


import pycountry
import holidays


def get_holiday_name(country_code, date_obj):
    country_holiday = holidays.CountryHoliday(country_code, years=date_obj.year)
    return country_holiday.get(date_obj)

def get_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_2  
    except KeyError:
        print(f"Unknown Country: {country_name}")
        return None

def get_holiday_for_row(row):
    country_code = get_country_code(row['country'])
    if country_code is None:
        return 'Unknown Country'
    
    try:
        date_obj = row['date']
    except ValueError:
        print(f"Invalid Date: {row['date']}")
        return 'Invalid Date'
    
    return get_holiday_name(country_code, date_obj)


df['holidays_name'] = df.apply(get_holiday_for_row, axis=1)
df.head()


df.drop("date", inplace = True, axis = "columns")
df.drop("id", inplace = True, axis  = "columns")
train_data = df[df['is train'] == 1]
test_data = df[df['is train'] == 0]


train_data.drop("is train", inplace = True, axis = "columns")
test_data.drop("is train", inplace = True, axis = "columns")
test_data.drop("num_sold", inplace = True, axis="columns")

train_data.shape, test_data.shape


train_data, val_data = train_test_split(train_data, random_state = 42, test_size = 0.2)


# convert to AutoGluon Dataset

train_data = TabularDataset(train_data)
val_data = TabularDataset(val_data)
test_data = TabularDataset(test_data)
train_data


from sklearn.metrics import mean_absolute_percentage_error
from autogluon.core.metrics import make_scorer

mape_scorer = make_scorer(name='mape',
                                 score_func= mean_absolute_percentage_error,
                                 optimum=1,
                                 greater_is_better=False)


hyperparameter_tune_kwargs = {  
    'num_trials': 40,
    'scheduler' : 'local',
    'searcher'  : 'auto',
}

predictor = TabularPredictor(label = 'num_sold',
                             eval_metric = mape_scorer,
                             problem_type = "regression",
                            )
predictor.fit(train_data,
              time_limit = 3600,
              hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
              presets = 'best_quality',
              save_space = True,
              keep_only_best = False
             )


predictor.evaluate(val_data)


LB = predictor.leaderboard(val_data)
LB


test_pred = predictor.predict(test_data)


test_pred


test_df['num_sold'] = test_pred.values


test_df.head()


test_df[["id", "num_sold"]].to_csv("submission.csv", index = False)

