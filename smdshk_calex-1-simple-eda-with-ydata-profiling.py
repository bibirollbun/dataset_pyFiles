# very basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Remove Warning Message
import warnings
warnings.filterwarnings('ignore')


# import train dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


# ydata profiling
from ydata_profiling import ProfileReport
ProfileReport(train)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.countplot(x='Sex', data = train, ax=axes[0])
sns.histplot(x='Calories', data = train, hue='Sex', kde = True, ax=axes[1])
sns.violinplot(x='Sex', y='Calories', data = train, ax=axes[2])

plt.tight_layout()
plt.show()

print('Cohen\'s d: ', end='')
print(abs(np.mean(train[train['Sex']=='male']['Calories'])-np.mean(train[train['Sex']=='female']['Calories']))/(((np.std(train[train['Sex']=='male']['Calories'], ddof=1))+np.std(train[train['Sex']=='female']['Calories'], ddof=1))/2))


num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

for col in num_cols:
    print(col)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(data = train, x = col, kde = True, ax = axes[0])
    sns.boxplot(x = train[col], ax = axes[1])
    sns.violinplot(x = 'Sex', y = col, data = train, ax = axes[2])
    plt.tight_layout()
    plt.show()
    
    print(train[col].describe())
    
    print('Cohen\'s d: ', end='')
    print(abs(np.mean(train[train['Sex']=='male'][col])-np.mean(train[train['Sex']=='female'][col]))/((np.std(train[train['Sex']=='male'][col], ddof=1)+np.std(train[train['Sex']=='female'][col], ddof=1))/2))

    plt.figure(figsize=(len(num_cols)*1.2, 1.2))
    corr = train[num_cols].corr().loc[[col]]
    sns.heatmap(corr, annot=True, cmap='coolwarm', cbar = False)
    plt.show()


train_num = train.drop(['id','Sex'], axis = 1)
corr = train_num.corr()
mask = np.triu(corr.corr())
sns.heatmap(corr, annot = True, annot_kws={"size": 8}, fmt = '.3f', cmap = 'coolwarm', square = True, mask = mask, cbar = False)


sns.pairplot(train_num)

