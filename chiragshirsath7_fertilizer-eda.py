import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')


traindf = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
testdf = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


traindf


testdf


traindf.info()


traindf.describe()


for col in traindf:
    print(col,"=",traindf[col].nunique())


traindf = traindf.drop_duplicates()


plt.figure(figsize=(12,7))
plots=[traindf['Temparature'],traindf['Humidity'],traindf['Moisture'],traindf['Nitrogen'],traindf['Potassium'],traindf['Phosphorous']]
plt.boxplot(plots)
plt.show();


plt.figure(figsize=(12,7))
sns.countplot(x=traindf['Fertilizer Name'],hue=traindf['Temparature'])
plt.show()





plt.figure(figsize=(12,7))
sns.countplot(x=traindf['Soil Type'],hue=traindf['Fertilizer Name'])
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()


plt.figure(figsize=(12,7))
sns.countplot(x=traindf['Fertilizer Name'],hue=traindf['Crop Type'])
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()


num_cols = []
for col in traindf:
    if(traindf[col].dtype==int and col!='id'):
        num_cols.append(col)
        plt.figure(figsize=(12,7))
        sns.violinplot(data=traindf,x=traindf['Fertilizer Name'],y=traindf[col])
        plt.show()


plt.figure(figsize=(12,7))
corr = traindf[num_cols].corr()
sns.heatmap(corr,annot=True)
plt.show()




