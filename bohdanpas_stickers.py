# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars  as pl # data processing, CSV file I/O (e.g. pl.scan_csv)
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


kaggle_path = '/kaggle/input/playground-series-s5e1'

train_df = pl.scan_csv(f'{kaggle_path}/train.csv').collect()
test_df = pl.scan_csv(f'{kaggle_path}/test.csv').collect()
sample_df = pl.scan_csv(f'{kaggle_path}/sample_submission.csv').collect()


# Display schema and head of the training data
display(train_df.collect_schema())
display(train_df.head(5))


# Investigating unique values
print("Unique countries:", train_df.get_column('country').unique())
print("Unique stores:", train_df.get_column('store').unique())
print("Unique products:", train_df.get_column('product').unique())


# Grouped aggregations
with pl.Config(tbl_rows=20):
    display(train_df.group_by(['product', 'store']).agg(pl.col('num_sold').n_unique()))

with pl.Config(tbl_rows=30):
    display(train_df.group_by(['product', 'country']).agg(pl.col('num_sold').n_unique()))

with pl.Config(tbl_rows=20):
    display(train_df.group_by(['store', 'country']).agg(pl.col('num_sold').n_unique()))


# Exploratory Data Analysis using seaborn
import seaborn as sns
sns.histplot(train_df.filter(pl.col('num_sold').is_not_null()), x='num_sold')


# Aggregations for categorical features
cat_features = ['num_sold', 'country', 'store', 'product']
for c in cat_features:
    with pl.Config(tbl_rows=50):
        display(train_df.group_by(by=c).agg(pl.col('num_sold').mean(), pl.col('id').len()).sort('num_sold'))


# Adding holiday feature
!pip install --upgrade holidays
!pip install --upgrade autogluon
from datetime import date
import holidays

us_holidays = holidays.country_holidays("US", years=[2010, 2011, 2012, 2013, 2014, 2015, 2016])

def data_clean(raw: pl.DataFrame) -> pl.DataFrame:
    result = raw
    if 'num_sold' in raw.columns:
        result = result.filter(pl.col('num_sold').is_not_null())

    result = result.with_columns((pl.col('country') + pl.col('store')).alias('country_store'))
    result = result.with_columns(pl.col('date').str.to_date().alias('p-date'))
    result = result.with_columns(pl.col('p-date').dt.weekday().alias('weekday'))
    result = result.with_columns(pl.col('p-date').dt.quarter().alias('quarter'))
    result = result.with_columns(pl.col('p-date').dt.ordinal_day().alias('day_of_year'))
    result = result.with_columns((366 - pl.col('day_of_year')).alias('remaining_days_of_year'))
    result = result.with_columns(pl.min_horizontal(pl.col('day_of_year'), pl.col('remaining_days_of_year')).alias('close_to_season'))
    result = result.with_columns((pl.col('p-date').is_in(us_holidays.keys())).alias('is_holiday'))
    return result

train_clean_df = data_clean(train_df)
test_clean_df = data_clean(test_df)


# Shuffle and split data
train_clean_df = train_clean_df.filter(pl.col('num_sold').is_not_null()).sample(fraction=1, shuffle=True)
n = train_clean_df.shape[0]
train_size = int(np.rint(n * 0.9))
validation_size = n - train_size

train_clean_df, validation_clean_df = train_clean_df.head(train_size), train_clean_df.tail(validation_size)



# Distribution of num_sold
plt.figure(figsize=(10, 6))
sns.histplot(data=train_df.to_pandas(), x='num_sold', kde=True, bins=30, color='blue')
plt.title('Distribution of Number Sold', fontsize=16)
plt.xlabel('Number Sold', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid()
plt.show()



# Boxplot for num_sold by country
plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df.to_pandas(), x='country', y='num_sold', palette='Set2')
plt.title('Number Sold by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Number Sold', fontsize=12)
plt.xticks(rotation=45)
plt.show()




# Aggregated statistics
agg_stats = train_df.group_by('country').agg([
    pl.col('num_sold').mean().alias('mean_num_sold'),
    pl.col('num_sold').std().alias('std_num_sold'),
    pl.col('num_sold').max().alias('max_num_sold')
]).to_pandas()

# Bar plot for mean sales by country
plt.figure(figsize=(10, 6))
sns.barplot(data=agg_stats, x='country', y='mean_num_sold', palette='viridis')
plt.title('Average Sales by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Average Sales', fontsize=12)
plt.xticks(rotation=45)
plt.show()


# Cross-validation setup
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Display processed training data
display(train_clean_df.head())
display(train_clean_df.group_by(by="is_holiday").len())

test_clean_df.head()


# Model training using AutoGluon
from autogluon.tabular import TabularPredictor
import warnings

warnings.simplefilter("ignore")

predictor = TabularPredictor(
    path='/kaggle/working/Autogluon2',
    label='num_sold',
    problem_type='regression',
    eval_metric='mean_absolute_percentage_error',
    learner_kwargs={'ignored_columns': ['id']}
)

predictor.fit(
    train_data=train_clean_df.to_pandas(),
    presets='high_quality',
    time_limit=1000,
    num_gpus=0
)


# Evaluate model
predictor = TabularPredictor.load("/kaggle/working/Autogluon2")
predictor.info()
predictor.leaderboard()

validation_results = predictor.evaluate(validation_clean_df.to_pandas())
y_pred = pl.Series(predictor.predict(validation_clean_df.to_pandas()))


# Metrics calculation
y_true = validation_clean_df.get_column('num_sold').to_numpy()
mse = mean_squared_error(y_true, y_pred.to_numpy())
r2 = r2_score(y_true, y_pred.to_numpy())

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

sns.scatterplot(x=y_true, y=y_pred.to_numpy())


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_true, y=y_pred.to_numpy(), color='green', alpha=0.6)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
plt.title('True vs Predicted Values', fontsize=16)
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.grid()
plt.show()


# Make predictions on test data
predictions = pl.Series(predictor.predict(test_clean_df.to_pandas()))
submission = sample_df.with_columns((np.rint(predictions)).alias('num_sold'))
submission = submission.with_columns(pl.max_horizontal(pl.col('num_sold'), 5))


# Save submission
submission.write_csv('submission.csv')
submission.head()

