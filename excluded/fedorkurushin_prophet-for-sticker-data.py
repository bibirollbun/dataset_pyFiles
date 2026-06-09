# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


# import holidays
from prophet import Prophet


# holidays_dict = {
#     'Canada' : holidays.Canada(),
#     'Finland' : holidays.Finland(),
#     'Italy' : holidays.Italy(),
#     'Kenya' : holidays.Kenya(),
#     'Norway' : holidays.Norway(),
#     'Singapore' : holidays.Singapore(),
# }


train.num_sold = train.num_sold.ffill() # there where hav no observations will with next value
train.num_sold = train.num_sold.bfill() # for the exceptions use after value


def process_group(group, train=True):
    if train:
        group = group[['date', 'num_sold']]
    else:
        group = group[['id', 'date']]
    group.columns = ['ds', 'y'] if train else ['id', 'ds']
    return group


Prophet.make_holidays_df(country_name=key[0])


models = {}

# Train Prophet models for each group
for key, group in train.groupby(by=['country', 'store', 'product'], as_index=False):
    mini_df = process_group(group)
    model = Prophet(seasonality_mode='multiplicative')
    model.add_country_holidays(country_name=key[0])
    model.fit(mini_df)
    key = '_'.join(key)
    models[key] = model
    print(f'{key} model trained')


pd.options.mode.copy_on_write = True


# Make predictions for each group in the test set
predictions = []

for key, group in test.groupby(by=['country', 'store', 'product'], as_index=False):
    mini_df = process_group(group, train=False)
    key = '_'.join(key)
    
    # Calculate the number of periods needed for the future dataframe
    forecast = models[key].predict(mini_df)
    mini_df.reset_index(inplace=True)
    mini_df['predict'] = forecast['yhat']
    predictions.append(mini_df)

# Combine predictions into a single DataFrame
predictions_df = pd.concat(predictions)


submission = pd.merge(test, predictions_df, on='id')[['id', 'predict']]
submission.columns = ['id', 'num_sold']


submission.to_csv('./submission.csv', index=False)




