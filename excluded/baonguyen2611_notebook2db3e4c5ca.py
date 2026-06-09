import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd

df = pd.read_csv("/kaggle/input/titanic-machine-learning-u-lima/train.csv")
df.head()



df.describe()



df['Sex'].value_counts()



df.groupby('Sex')['Survived'].mean()


