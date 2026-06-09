import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

df = pd.read_csv('../input/playground-series-s5e1/train.csv')
df = df[['date', 'country', 'store', 'product', 'num_sold']]
df = df.set_index('date')
df.index = pd.to_datetime(df.index)
df = df.to_period('D').reindex(axis=1)
display(df)


wide = df.pivot(columns=['country', 'store', 'product'], values='num_sold')
display(wide)

cols_with_na = []

for col in wide.columns:
    if wide[col].isna().any():
        cols_with_na.append((wide[col].isna().sum(), col))

cols_with_na.sort(reverse=True)

print()
for na_count, col in cols_with_na:
    print(na_count, col)


s = wide[('Canada', 'Stickers for Less', 'Holographic Goose')]
_ = s.plot()


while s.isna().any():
    shift = pd.concat([s.shift(periods=p) for p in [-365, -1, 1, 365]], axis=1)
    s = s.fillna(shift.apply(np.nanmean, axis=1))

_ = s.plot()




