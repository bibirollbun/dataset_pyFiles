import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')
train


test = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv')
test


mis_map = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')
mis_map


sample_sub = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/sample_submission.csv')
sample_sub


mis_list = train['MisconceptionAId'].to_list() + train['MisconceptionBId'].to_list() + train['MisconceptionCId'].to_list() + train['MisconceptionDId'].to_list()
mis_list = [int(x) for x in mis_list if pd.notna(x)]
len(set(mis_list))


from collections import Counter

# mis_list is already filtered and converted to integers
mis_count = Counter(mis_list)

# Convert Counter to a dictionary (optional, as Counter itself is a dict-like object)
mis_count_dict = dict(mis_count)



import matplotlib.pyplot as plt

plt.figure(figsize=(15, 4))
plt.scatter(mis_count_dict.keys(), mis_count_dict.values())
plt.ylabel('IDs')
plt.xlabel('Count')
plt.title('Frequency of IDs')
plt.show()





