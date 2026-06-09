import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
%matplotlib inline
from sklearn.metrics import *
import warnings 
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
df.head()


print(f'Nulls in training: {df.isnull().sum().sum()}')


nums = df.select_dtypes(include='number').columns.tolist()
cats = df.select_dtypes(exclude='number').columns.tolist()


fig,ax = plt.subplots(3,int(len(nums)//3)+1,figsize=(12,8))
ax = ax.flatten()

for idx,feats in enumerate(nums):
    df[feats].plot(kind='hist',ax=ax[idx])
    ax[idx].set_xlabel(feats)
plt.tight_layout()
plt.show()


fig,ax = plt.subplots(3,int(len(cats)//3),figsize=(15,6))
ax = ax.flatten()

for idx,feats in enumerate(cats):
    # df[feats].plot(kind='hist',ax=ax[idx])
    sns.countplot(
        data=df,
        x = feats,
        hue=df['loan_paid_back'],
        ax = ax[idx]
    )
    # ax[idx].set_xlabel(feats)
plt.tight_layout()
plt.show()


from itertools import combinations
all_cross_bins_02 = [things for things in combinations(cats,2)]
print(f'=> {len(all_cross_bins_02)} potential cross bins found in the cats list')


for comb in all_cross_bins_02:
    f1 = comb[0]
    f2 = comb[1]
    df[f'{f1}x{f2}'] = df[f1].astype(str) + "_" + df[f2].astype(str)


df.head()


fig,ax = plt.subplots(5,int(len(all_cross_bins_02)//5),figsize=(15,10))
ax = ax.flatten()
for idx,feats in enumerate(all_cross_bins_02):
    sns.countplot(
        data=df,
        x = f'{feats[0]}x{feats[1]}',
        hue=df['loan_paid_back'],
        ax = ax[idx]
    )
    for container in ax[idx].containers:
            ax[idx].bar_label(container, fmt='%d', fontsize=7, padding=3)
        
    ax[idx].set_xlabel("") # Remove x-axis title
    ax[idx].tick_params(axis='x', rotation=45)
    ax[idx].set_xticks([]) # Remove the tick marks (the actual labels)
    
    title = f"{feats[0].title()} vs {feats[1].title()}"
    ax[idx].set_title(title, fontsize=8)
plt.tight_layout()
plt.show()


num_features = len(all_cross_bins_02)
num_cols = 3
num_rows = int(np.ceil(num_features / num_cols))

fig, ax = plt.subplots(
    nrows=num_rows, 
    ncols=num_cols, 
    figsize=(4 * num_cols * 1.5, 4 * num_rows * 1.2),
    sharey=False
)
ax = ax.flatten() if num_features > 1 else [ax]

for idx, (f1, f2) in enumerate(all_cross_bins_02):
    
    heatmap_data = df.groupby([f1, f2])['loan_paid_back'].mean().unstack()
    sns.heatmap(
        heatmap_data,
        annot=True,        
        fmt=".2f",         
        cmap="coolwarm_r", 
        cbar_kws={'label': 'Mean Loan Paid Back Rate'},
        linewidths=.5,    
        ax=ax[idx]
    )
    
    title = f"{f1.title()} vs {f2.title()} (Repayment Rate)"
    ax[idx].set_title(title, fontsize=10)
    ax[idx].tick_params(axis='both', rotation=45, labelsize=7)

for i in range(num_features, len(ax)):
    fig.delaxes(ax[i])

plt.tight_layout()
plt.show()


fig,ax = plt.subplots(3,int(len(nums)//3)+1,figsize=(12,8))
ax = ax.flatten()

for idx,feats in enumerate(nums):
    sns.histplot(
        data=df,
        x = feats,
        hue='loan_paid_back',
        ax=ax[idx]
    )
    ax[idx].set_xlabel(feats)
plt.tight_layout()
plt.show()


fig,ax = plt.subplots(3,int(len(cats)//3),figsize=(15,6))
ax = ax.flatten()

for idx,feats in enumerate(cats):
    sns.boxplot(
        data=df,
        x = feats,
        y='loan_paid_back',
        ax = ax[idx],
        showmeans = True
    )
    # ax[idx].set_xlabel(feats)
plt.tight_layout()
plt.show()


num_features = len(cats)
num_cols = 3
num_rows = int(np.ceil(num_features / num_cols))

fig, ax = plt.subplots(
    nrows=num_rows, 
    ncols=num_cols, 
    figsize=(18, 5 * num_rows)
)

if num_features > 1:
    ax = ax.flatten()
else:
    ax = [ax] 

for idx, feats in enumerate(cats):
    sns.histplot(
        data=df,
        x=feats,
        hue='loan_paid_back',
        multiple='fill',
        shrink=0.8,
        ax=ax[idx],
        palette='viridis' 
    )

    ax[idx].tick_params(axis='x', rotation=45)
    ax[idx].set_title(f'Target Distribution (%) in {feats.replace("_", " ").title()}', fontsize=12)
    ax[idx].set_ylabel('Proportion of Category Total (1.0 = 100%)', fontsize=10)
    ax[idx].set_xlabel(feats.replace("_", " ").title(), fontsize=10)

    handles, labels = ax[idx].get_legend_handles_labels()
    new_labels = ['Not Paid Back (0)', 'Paid Back (1)']
    ax[idx].legend(handles, new_labels, title="Loan Status", loc='upper right')
for i in range(num_features, len(ax)):
    fig.delaxes(ax[i])

plt.tight_layout()
plt.show()


sns.pairplot(data=df.sample(2000),hue='loan_paid_back',vars=nums) 
## here sample(2000) was used to speed the plotting, increasing this value will be beneficial for better representation of the data

