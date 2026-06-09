


pip install pyarrow


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
paths=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        paths+=[os.path.join(dirname, filename)]


train=pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test=pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
submit=pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


display(train[0:2])
display(test[0:2])
display(submit[0:2])


#print(train.columns.tolist())
#print(test.columns.tolist())
train.columns.tolist()==test.columns.tolist()


print(len(train),len(test),len(submit))


ts=train.index.tolist()
print(ts[0],'--',ts[-1])
#every minute for a year


print(train['label'].min(),'--',train['label'].max())


train['label'][0:3]


plt.figure(figsize=(10, 6))
plt.hist(train['label'], bins=100, edgecolor='black')
plt.xlabel('Label Value (Binned)')
plt.ylabel('Count')
plt.title('Distribution of Label')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(train['label'], bins=100, edgecolor='black')
plt.yscale('log')  # log scale
plt.xlabel('Label Value (Binned)')
plt.ylabel('Count (log scale)')
plt.title('Histogram with Log Y-Axis')
plt.grid(True)
plt.show()


def func(x):
    return np.sign(x) * np.log1p(np.abs(x))
    
def inverse_func(y):
    return np.sign(y) * (np.expm1(np.abs(y)))    
    
train['label_transformed'] = train['label'].apply(func)    
#train['label_transformed'] = train['label'].apply(lambda x:func(x))
#train['label_transformed'] = np.sign(train['label']) * np.log1p(np.abs(train['label']))

print(train['label'][0:3],train['label_transformed'][0:3])


plt.figure(figsize=(10,6))
plt.hist(train['label_transformed'], bins=100, edgecolor='black')
plt.xlabel('Transformed Label Value')
plt.ylabel('Count')
plt.title('Distribution After X-Value Transformation')
plt.grid(True)
plt.show()




