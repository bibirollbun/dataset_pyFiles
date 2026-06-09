import pandas as pd
import numpy as np

df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

source = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
min_canada = source.loc[(source['country'] == 'Canada') & (source['num_sold'].notnull()), 'num_sold'].min()
min_kenya = source.loc[(source['country'] == 'Kenya') & (source['num_sold'].notnull()), 'num_sold'].min()

def replace_with_min(row):
    if row['country'] == 'Canada' and row['num_sold'] == 0:
        return min_canada
    elif row['country'] == 'Kenya' and row['num_sold'] == 0:
        return min_kenya
    return row['num_sold']
    
df['num_sold'] = np.where(
    (df['country'].isin(['Canada', 'Kenya'])) & 
    (df['store'] == 'Discount Stickers') & 
    (df['product'] == 'Holographic Goose'), 
    0, 
    100
)

# Replace 0 with the mean values
df['num_sold'] = df.apply(replace_with_min, axis=1)


df[['id', 'num_sold']].to_csv("submission.csv", index=False)


df

