# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")


cols = train.columns

start_first = time.time()
def handle_duplicates(df):
    dupes  = set()
    
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            if (df.iloc[:,i] == df.iloc[:,j]).all():
                dupes.add(cols[j])

    return (list(dupes))

duplicates = handle_duplicates(train)
end_first = time.time()
#print(f'Here is the list of duplicated columns: {duplicates}') 
#print(f'There are {len(duplicates)} columns that have direct duplicates')
print(f"The first method took {end_first-start_first:.2f} seconds")


#Hashing
start = time.time()
hash_list = train.apply(lambda x :pd.util.hash_pandas_object(x,index=False).sum())

#Extract the indices of the non repeating hashes
_,hash_indices = np.unique(hash_list,return_index=True)

#now our dataset doesn't contain any duplicates
train = train.iloc[:,sorted(hash_indices)]
end = time.time()

print(f"The hashing method took {end-start:.2f} seconds")


#For the first method
print(f"Unique columns after the first method: {len(cols)-len(duplicates)}")

#Second Method
print(f"Unique columns after the second method: {len(hash_indices)}")

