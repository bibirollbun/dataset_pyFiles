def plot_details(variable, target, data, **kwargs):
    plt.figure(figsize=(12, 3))
    
    # box plot 
    plt.subplot(1,3,1)
    ax = sns.boxplot(y=variable, data=data, **kwargs)
    
    # histogram
    plt.subplot(1,3,2)
    ax2 = sns.histplot(x=variable, bins=30, data=data, **kwargs)

    # correlation between variable and target
    plt.subplot(1,3,3)
    ax3 = sns.scatterplot(x=variable, y=target, data=data, **kwargs)

    if 'hue' in kwargs:
        sns.move_legend(ax2, 'upper right', bbox_to_anchor=(1, 1))
        sns.move_legend(ax3, 'upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.suptitle(variable)
    plt.show()


def detailed_binned_scatter(ftrs, target_vars):
    fig, axes = plt.subplots(nrows=len(target_vars), ncols=len(ftrs), figsize=(8 * len(ftrs), 5 * len(target_vars)))

    for row_idx, target in enumerate(target_vars):
        for col_idx, ftr in enumerate(ftrs):
            if len(ftrs) == 1 and len(target_vars) == 1:
                ax = axes
            elif len(target_vars) == 1:
                ax = axes[col_idx]
            elif len(ftrs) == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]

            sns.regplot(
                data=train[train['Sex'] == 'male'], x=ftr, y=target,
                x_bins=100, x_estimator=np.mean, x_ci=90, ax=ax, label='Male'
            )
            sns.regplot(
                data=train[train['Sex'] == 'female'], x=ftr, y=target,
                x_bins=100, x_estimator=np.mean, x_ci=90, ax=ax, label='Female', color='orange'
            )

            ax.set_title(f'{target} vs {ftr}')
            ax.legend()

    plt.tight_layout()
    plt.show()


!pip install pingouin --quiet


from pingouin import partial_corr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')


class cfg:
    train_path = '/kaggle/input/playground-series-s5e5/train.csv'
    test_path = '/kaggle/input/playground-series-s5e5/test.csv'
    original_path = '/kaggle/input/calories-burnt-prediction/calories.csv'
    categorical_features = ['Sex']
    numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    target = 'Calories'


train = pd.read_csv(cfg.train_path, index_col='id')
test = pd.read_csv(cfg.test_path, index_col='id')


train.head()


train.shape


train.info()


print('Train set shape', train.shape[0])
cols = cfg.categorical_features + cfg.numerical_features
train_duplicated = train.duplicated(subset=cols, keep=False)
duplicate_groups = train[train_duplicated].groupby(cols)

cnt = 0
for name, group in duplicate_groups:
    unq_cals = group['Calories'].unique()

    if len(unq_cals) > 1:
        cnt+=1

print('TRAIN - Duplicated rows shape:', train_duplicated.sum())
print(f'TRAIN - Duplicated rows ratio: {round((train_duplicated.sum() / train.shape[0]) * 100, 2)}%')
print('TRAIN - # of duplicated rows with different target value:', cnt)
print('TRAIN - % of duplicated rows with different target value:', round((1 - (cnt / train_duplicated.sum()))*100, 2))

print('-'*50, '\n')

print('Test set shape', test.shape[0])
test_duplicated = test.duplicated(subset=cols, keep=False)
duplicate_groups = test[test_duplicated].groupby(cols)
print('TEST - Duplicated rows shape:', test_duplicated.sum())
print(f'TEST - Duplicated rows ratio: {round((test_duplicated.sum() / train.shape[0]) * 100, 2)}%')


plt.figure(figsize=(5, 5))
sns.countplot(x='Sex', data=train);


for ftr in cfg.numerical_features:
    plot_details(ftr, cfg.target, train)


# Since it's hard to understand the correlation from scatterplots, we can use binned scatter plots
plt.figure(figsize=(15, 5))

for idx, ftr in enumerate(['Age', 'Height', 'Weight']):
    plt.subplot(1, 3, idx+1)
    plt.title(f'Binned Scatter - {ftr}')
    sns.regplot( data=train, x=ftr, y=cfg.target, x_bins=100, x_estimator=np.mean, x_ci=90)

plt.tight_layout(pad=1.0)
plt.show()


for ftr in cfg.numerical_features:
    plot_details(ftr, cfg.target, train, hue='Sex')


detailed_binned_scatter(['Age', 'Weight', 'Height'], ['Calories'])


corr_matrix = train[cfg.numerical_features + [cfg.target]].corr()
corr_matrix = corr_matrix.round(2)

plt.figure(figsize=(5,5))
sns.heatmap(corr_matrix, annot=True)


results = []
for ftr in cfg.numerical_features:
    result = partial_corr(
      data=train, 
      x=ftr, 
      y=cfg.target,
      covar=[c for c in cfg.numerical_features if c not in (ftr, cfg.target)])

    cols = result.columns.tolist()
    result['Feature'] = ftr
    result = result[['Feature'] + cols]
    results.append(result)

par_corr = pd.concat(results, axis=0)
par_corr


detailed_binned_scatter(['Duration', 'Heart_Rate', 'Body_Temp'], ['Age'])


detailed_binned_scatter(['Weight', 'Height'],
                        ['Duration', 'Heart_Rate', 'Body_Temp'])


plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
sns.histplot(x='Calories', data=train, bins=30)

plt.subplot(1, 3, 2)
sns.histplot(x='Calories', hue='Sex', data=train, bins=30)

plt.subplot(1, 3, 3)
sns.boxplot(y='Calories', hue='Sex', data=train)

plt.tight_layout(pad=.5)
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

original_df = pd.read_csv(cfg.original_path)

for ax, ftr in zip(axes, cfg.numerical_features):
    temp_df = pd.concat([
        pd.DataFrame({ftr: original_df[ftr], 'Split': 'Original'}),
        pd.DataFrame({ftr: train[ftr], 'Split': 'Train'}),
        pd.DataFrame({ftr: test[ftr], 'Split': 'Test'})
    ])

    sns.boxplot(x='Split', y=ftr, data=temp_df, ax=ax)
    ax.set_title(f'{ftr} - Original vs. Train vs. Test')

plt.tight_layout()
plt.show()

