import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp

mp.rcParams.update({'font.size': 16, 
                    "figure.figsize": (8, 8), 
                    "figure.dpi": 128})

path = "/kaggle/input/playground-series-s5e9/train.csv"


df = pd.read_csv(path).drop(columns=['id'])


df.head()


df.info()


df.describe()


sns.pairplot(df)




