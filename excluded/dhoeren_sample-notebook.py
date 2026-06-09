# imports
import pandas as pd


# load the training data
df = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/wildfire_sizes_before_2010.csv")

# print a few datapoints
df.head()


# get the predictions for 2010
last_year = df[df['month'].str[:4] == '2010']

last_year.head()


# duplicate the 2010 predictions for 2011 through 2015
dfs = []
for year in range(2011, 2016):
    new_df = last_year.copy()
    new_df['month'] = new_df['month'].str.replace('2010', str(year))
    dfs.append(new_df)
submission = pd.concat(dfs)


# add ID column that kaggle wants (order does not matter though, items are match by (STATE, month) pair)
submission['ID'] = range(len(submission))

# order columns
submission = submission[['ID', 'STATE', 'month', 'total_fire_size']]
submission.to_csv('submission.csv', index=False)

submission.head()




