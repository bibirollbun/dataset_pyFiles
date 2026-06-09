import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
sns.set_theme()


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
original_df = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')


print(
    f'Shape of the train dataset: {train_df.shape}'
    '\n'
    f'Shape of the test dataset: {test_df.shape}'
    '\n'
    f'Shape of the original dataset: {original_df.shape}'
    '\n'
    f'Test dataset as the proportion of the train dataset: {100*len(test_df)/len(train_df):.2f}%'
)

print('=> Train Data')
display(train_df.head())
print('=> Original Data')
display(original_df.head())


nulls_train = train_df.isnull().sum().sum()
nulls_test = test_df.isnull().sum().sum()
nulls_original = original_df.isnull().sum().sum()

if all([data == 0 for data in [nulls_train,nulls_test,nulls_original]]): #first time using all(), this function needs a list of bools !!
    print(f'We have no nulls anywhere.')
else:
    print(f'We have nulls.')


def get_num_cols(data:pd.DataFrame):
    return data.select_dtypes(include='number').columns.tolist()
def get_cat_cols(data:pd.DataFrame):
    return data.select_dtypes(exclude='number').columns.tolist()
def enum(data:list):
    return enumerate(data)


fig,ax = plt.subplots(1,2,figsize=(12,5))
target = 'diagnosed_diabetes'
names = ['Train','Original']
for idx,data in enum([train_df,original_df]):
    sns.countplot(
        data=data,
        x = target,
        ax = ax[idx]
    )
    ax[idx].set_title(names[idx]+' Dataset')
    print(f'=> {names[idx]} Dataset')
    print(data[target].value_counts(normalize=True))
plt.tight_layout()


nums = get_num_cols(test_df)
fig,ax = plt.subplots(5,4,figsize=(12,10))
fig.suptitle('Train Dataset')
ax= ax.flatten()
new_labels = [0,1]
for idx,cols in enum(nums):
    sns.kdeplot(
        data=train_df,
        x = cols,
        hue=target,
        ax=ax[idx],
        fill=True
    )
    ax[idx].legend(title='Status', labels=new_labels)
plt.tight_layout()


nums = get_num_cols(test_df)
fig,ax = plt.subplots(5,4,figsize=(12,10))
fig.suptitle('Original Dataset')
ax= ax.flatten()
new_labels = [0,1]
for idx,cols in enum([cols for cols in nums if cols!='id']):
    sns.kdeplot(
        data=original_df,
        x = cols,
        hue=target,
        ax=ax[idx],
        # color='red'
    )
    ax[idx].legend(title='Status', labels=new_labels)
plt.tight_layout()


cats = get_cat_cols(train_df)
fig,ax = plt.subplots(3,2,figsize=(12,6))
fig.suptitle('Train Dataset')
ax = ax.flatten()

for idx,cols in enum(cats):
    sns.countplot(
        data=train_df,
        x = cols,
        hue=target,
        ax=ax[idx]
    )
    ax[idx].legend(title='Status', labels=new_labels)
plt.tight_layout()


cats = get_cat_cols(train_df)
fig,ax = plt.subplots(3,2,figsize=(12,6))
fig.suptitle('Original Dataset')
ax = ax.flatten()

for idx,cols in enum(cats):
    sns.countplot(
        data=original_df,
        x = cols,
        hue=target,
        ax=ax[idx]
    )
    ax[idx].legend(title='Status', labels=new_labels)
plt.tight_layout()


fig,ax = plt.subplots(6,2,figsize=(12,10))
ax = ax.flatten()

for idx,cols in enum(cats):
    sns.boxplot(
        data=train_df,
        x = cols,
        y=target,
        ax=ax[idx],
        showmeans=True
    )
    ax[idx].set_ylabel(f'target-{names[0]}')


for idx,cols in enum(cats):
    sns.boxplot(
        data=original_df,
        x = cols,
        y=target,
        ax=ax[idx+6],
        showmeans=True
    )
    ax[idx+6].set_ylabel(f'target-{names[1]}')
plt.tight_layout()


fig,ax = plt.subplots(1,2,figsize=(12,6),sharey=True,sharex=True)
pivot = train_df.pivot_table(
    index='income_level', 
    columns='education_level', 
    values='diagnosed_diabetes', 
    aggfunc='mean' 
)

pivot2 = original_df.pivot_table(
    index='income_level', 
    columns='education_level', 
    values='diagnosed_diabetes', 
    aggfunc='mean' 
)

sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd",ax=ax[0])
sns.heatmap(pivot2, annot=True, fmt=".1%", cmap="YlOrRd",ax=ax[1])

for i in range(2):
    ax[i].set_title(names[i])


fig,ax = plt.subplots(1,2,figsize=(12,6),sharey=True,sharex=True)
pivot = train_df.pivot_table(
    index='ethnicity', 
    columns='gender', 
    values='diagnosed_diabetes', 
    aggfunc='mean' 
)

pivot2 = original_df.pivot_table(
    index='ethnicity', 
    columns='gender', 
    values='diagnosed_diabetes', 
    aggfunc='mean' 
)

sns.heatmap(pivot, annot=True, fmt=".1%", cmap="mako",ax=ax[0])
sns.heatmap(pivot2, annot=True, fmt=".1%", cmap="mako",ax=ax[1])

for i in range(2):
    ax[i].set_title(names[i])


fig,ax = plt.subplots(1,2,figsize=(12,6),sharey=True,sharex=True)
pivot = train_df.pivot_table(
    index='smoking_status', 
    columns='gender', 
    values='diagnosed_diabetes', 
    aggfunc='mean' 
)

pivot2 = original_df.pivot_table(
    index='smoking_status', 
    columns='gender', 
    values='diagnosed_diabetes', 
    aggfunc='mean' 
)

sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd",ax=ax[0])
sns.heatmap(pivot2, annot=True, fmt=".1%", cmap="YlOrRd",ax=ax[1])

for i in range(2):
    ax[i].set_title(names[i])


plt.figure(figsize=(12,8))
c_m = train_df[[cols for cols in nums if cols!='id']].corr()
mask = np.tril(np.ones_like(c_m,dtype=bool))
sns.heatmap(
    c_m,
    mask=mask,
    cmap='viridis',
    annot=True,
    fmt='.2f'
)

