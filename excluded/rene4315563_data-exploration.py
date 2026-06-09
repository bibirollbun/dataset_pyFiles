import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train


pie_data   = list(train.columns)
pie_data.remove("id")
histo_data = ["age","balance","duration","pdays","previous","campaign"]

for item in histo_data:
    pie_data.remove(item)

pie_data , histo_data


for c in histo_data:
    fig, ax = plt.subplots()
    ax.title.set_text(c)
    ax.hist(train[c].value_counts().index.tolist(), bins=20, linewidth=0.5, edgecolor="white")
plt.show()


for c in pie_data:
    fig, ax = plt.subplots()
    ax.title.set_text(c)
    ax.pie( x=train[c].value_counts().values.tolist(),labels=train[c].value_counts().index.tolist()) #, autopct='%1.1f%%'

plt.show()

