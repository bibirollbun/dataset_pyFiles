import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import random

from itertools import combinations


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
ss = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train_0 = train_df.copy()
test_0 = test_df.copy()


train_df.head()


# Binary Encoding Sex
train_df['Sex'] = (train_0['Sex'] == 'male')

# Normalization
columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories'] 
scaler = MinMaxScaler() # just a quick solution not the best
train_df[columns] = scaler.fit_transform(train_0[columns])

train_df.head()


# Shuffle data 
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.head()


features_all = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

triplets = list(combinations(features_all, 3))
triplets


batch_size = 100 
triplet = list(triplets[0])

X = train_df[triplet][:batch_size]
X = np.array(X).reshape(100, 1, 3) # 100x1x3 RGB one pixel line

y = train_df['Calories'][:batch_size]

plt.imshow(X)
plt.axis('off')
plt.show()


combo_triplet = [list(triplet) for triplet in triplets]
dummy = pd.concat((train_df[lst] for lst in combo_triplet), axis=1)
dummy = np.array(dummy.head(100)).reshape(100, 20, 3)

y = np.array(y).reshape(100, 1)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(dummy)
axes[1].imshow(y, cmap='gray')
plt.axis('off')
plt.show()

