import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
train.head(5)


train.drop(columns='id', inplace=True) #discard the 'id' column as it is simply the row number and not needed

print('Name, data type and values of each column:')
vals = [(train[col].min(), train[col].max()) if train[col].dtype == 'float64' else (sorted(train[col].unique())) for col in train.columns]
imp = pd.DataFrame({'dtype': train.dtypes, 'range or values': vals})
imp


for col in train.columns:
    if train[col].dtype == 'float64':
        train[col].hist(bins=100,legend=True)
    else:
        train[col].value_counts().sort_index().plot(kind='bar', xlabel=col, ylabel='Count', rot=0)
    plt.show()


for col in train.columns:
    if train[col].dtype == 'float64':
        coldis = col + '_discretized'
        train[coldis] = pd.cut(train[col],bins=5).values #insert helper column where the float column is discretized into 5 equally width intervals
        train.boxplot(column=['accident_risk'], by=coldis, grid=False)#, color='black')
    else:
        train.boxplot(column=['accident_risk'], by=col, grid=False)#, color='black')
    plt.show()

