import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns


import sys
sys.path.append("eda_utility_library")

from eda_utility_library import categorize_columns, plot_pie_charts, violin_plots, missing_data_summary


ss = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train.head()


RMV = ['Price', 'id']
categorized_columns = categorize_columns(train, rmv=RMV)


for col_type in categorized_columns.keys():
    if len(categorized_columns[col_type]) > 0:
        print(col_type)


plot_pie_charts(train, categorized_columns['categorical'])


plot_pie_charts(train, categorized_columns['discrete']) 


violin_plots(train, categorized_columns['continuous']) # Haha id


missing_data_summary(train)


price_stats_per_brand = train.groupby('Brand')['Price'].describe()
price_stats_per_brand


def weighter(weight):
    if weight < 12.5:
        return 'Light'
    elif weight < 22.5:
        return 'Average'
    else:
        return 'Heavy'

train['WeightClass'] = train['Weight Capacity (kg)'].apply(lambda x: weighter(x))
price_stats_per_weight = train.groupby('WeightClass')['Price'].describe()
price_stats_per_weight


price_per_compartment = train.groupby('Compartments')['Price'].describe()
price_per_compartment


df = train_extra.copy()

FEATURES = list(set(df.columns) - set(RMV))


# Display the count of duplicate groups
duplicate_groups = df.groupby(FEATURES).size().reset_index(name='Count')
duplicate_groups = duplicate_groups[duplicate_groups['Count'] > 1]  # Keep only real duplicates

# Merge with original data to see full duplicate details
duplicate_entries = df.merge(duplicate_groups, on=FEATURES, how='inner')


len(duplicate_entries)

