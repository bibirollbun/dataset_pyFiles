import pandas as pd

df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
df


print( df.shape)
print(df.columns)
print(df.dtypes)

print(df.nunique())



df.isna().mean()


import missingno as msno
msno.matrix(df)
msno.bar(df)
msno.heatmap(df)

