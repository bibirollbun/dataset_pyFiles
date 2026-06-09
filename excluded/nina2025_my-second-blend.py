import pandas as pd

p = '/kaggle/input/'

f_subm_D = '/sub_logistic-regression_0.366237.csv'

D = pd.read_csv (p+'s05e06-fertilizer-optimization-ensemble'+f_subm_D)
B = pd.read_csv (p+'optimal-fertilizers-xgb/submission.csv')

D,B = D.iloc[0:125_000],B.iloc[125_000:250_001]

df = pd.concat([D,B], axis=0)

df.to_csv('submission.csv',index=False)

df

