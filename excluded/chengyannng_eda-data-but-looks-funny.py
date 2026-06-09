import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


traindata = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv').drop(columns='id')
testdata = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


traindata.head()


testdata.head()


traindata.info()


traindata['Fertilizer Name'].value_counts()


traindata['Soil Type'].value_counts()


traindata['Crop Type'].value_counts()


traindata.describe().T


sns.pairplot(traindata.drop(columns=['Soil Type', 'Crop Type', 'Fertilizer Name']))

