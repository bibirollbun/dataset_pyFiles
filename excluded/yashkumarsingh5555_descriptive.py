import pandas as pd
df=pd.read_csv('/kaggle/input/trainsample-csv/trainsample.csv')


df.shape


columnstodrop = ['AutoSampleOptIn',
'Census_InternalBatteryNumberOfCharges',
'Census_InternalBatteryType',
'Census_IsFlightingInternal',
'Census_IsFlightsDisabled',
'Census_IsWIMBootEnabled',
'Census_ProcessorClass',
'Census_ThresholdOptIn',
'DefaultBrowsersIdentifier',
'IsBeta',
'ProductName',
'PuaMode',
'UacLuaenable']


df=df.drop(columns=columnstodrop)


df.select_dtypes(include=['number']).describe().T.head(10)


df.isna().sum().sort_values(ascending=False)



for col in df.select_dtypes(include=['object']).columns:
    print(col, df[col].unique()[:20])



df.describe(include='object').T


import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(col, "outliers:", outliers)



import seaborn as sns
import matplotlib.pyplot as plt

for col in numeric_cols:
    plt.figure(figsize=(6,2))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()



df[numeric_cols].skew().sort_values(ascending=False)


df[numeric_cols].hist(figsize=(30,30))
plt.show()


