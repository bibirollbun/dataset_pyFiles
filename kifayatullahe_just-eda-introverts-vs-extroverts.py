import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
print(train.shape)
print(test.shape)


train.head()



train.isnull().mean()*100 # Overview Null values


train.duplicated().mean() # Overview Null values


train.info() # Data Summarize


train.describe() # Statistical Summary of Data


print(train['Personality'].value_counts())
sns.set_style('darkgrid')

plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = train['Personality'])
plt.title('CountPlot of Target Class', fontsize=13, fontweight='bold')
plt.xlabel('Personality', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.subplot(122)
train['Personality'].value_counts().plot(kind='pie', autopct='%.2f',
                                         colors=['skyblue', 'hotpink'], shadow=True)
plt.show()


print(train['Stage_fear'].value_counts())

plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = train['Stage_fear'], palette='Blues')
plt.title('CountPlot of Stage_fear', fontsize=13, fontweight='bold')
plt.xlabel('Stage_fear', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.subplot(122)
train['Stage_fear'].value_counts().plot(kind='pie', autopct='%.2f',
                                         colors=['blue', 'teal'], shadow=True)
plt.show()


print(train['Drained_after_socializing'].value_counts())

plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = train['Drained_after_socializing'], palette='winter')
plt.title('CountPlot of Drained_after Socializing', fontsize=13, fontweight='bold')
plt.xlabel('Drained_after_socializing', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.subplot(122)
train['Drained_after_socializing'].value_counts().plot(kind='pie', autopct='%.2f',
                                         colors=['coral', 'orange'], shadow=True)
plt.show()


train.drop('id', axis=1, inplace=True)



num_cols = train.select_dtypes(include='number')

for col in num_cols.columns:
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    sns.histplot(data=train, x=col, kde=True)
    plt.title(f'Distribution of {col}')


    plt.subplot(122)
    stats.probplot(train[col], dist='norm', plot=plt)
    plt.show()



outlier_summary = {}
for col in num_cols:
    z = np.abs(stats.zscore(train[col].dropna()))
    outlier_summary[col] = (z>2).sum()   # 3-σ rule

pd.Series(outlier_summary, name="#outliers (>2σ)").sort_values(ascending=False).to_frame().style.bar()



for col in num_cols.columns:
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    sns.boxplot(data=train, x=col, y='Personality')
    plt.title(f'BoxPlot of {col}')
    plt.show()


plt.figure(figsize=(10,5))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap='winter')
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
sub.to_csv('submission.csv', index=False) # Submition csv




