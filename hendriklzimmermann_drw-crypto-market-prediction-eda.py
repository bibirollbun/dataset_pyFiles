import os
import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.dates as md
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cycler
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings('ignore')

colors = ["#53599A", "#068D9D", "#607BB0", "#77BECF", "#6D9DC5", "#80DED9", "#AEECEF"]
plt.rc('axes', facecolor="#E9E9E9", edgecolor='none', axisbelow=True, grid=True, prop_cycle=cycler('color', colors))



def reduce_mem_usage(dataframe):
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            else:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe


def plot_relationship(pairs, plot_df, cols=5):
    rows = np.ceil(len(pairs)/cols).astype(int)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4*rows))
    for pair in pairs:
        ax = axs[pairs.index(pair) // cols, pairs.index(pair) % cols]
        ax.scatter(plot_df[pair[0]], plot_df[pair[1]], s=0.2)
        ax.set_title(f"{pair[0]} vs {pair[1]}\nCorrelation: {pair[2]:.2f}")
        beta, alpha = np.polyfit(plot_df[pair[0]], plot_df[pair[1]], deg=1)
        ax.axline(xy1=(0, alpha), slope=1, color='red', linestyle='--')
        
    plt.tight_layout()
    plt.show()



data_dir = '/kaggle/input/drw-crypto-market-prediction/'


df_train = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
df_test = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)


print('Train shape:', df_train.shape)
print('Test shape:', df_test.shape)
print("df_train memory usage: {:.2f} GB".format(df_train.memory_usage(deep=True).sum() / 1024**3))
print("df_test memory usage: {:.2f} GB".format(df_test.memory_usage(deep=True).sum() / 1024**3))


# Check for Inf
inf_idx = (abs(df_train)==np.inf).transpose().sum()
inf_idx = inf_idx[inf_idx!=0]
print(f"Found {len(inf_idx)}\t observations with Inf values.")

inf_idx = (abs(df_train)==np.inf).sum()
inf_idx = inf_idx[inf_idx!=0]
print(f"Found {len(inf_idx)}\t columns with Inf values.")

print("Removing columns")
df_train = df_train.drop(columns=inf_idx.index)
df_test = df_test.drop(columns=inf_idx.index)

print("New Shapes:")
print("Training Data:\t", df_train.shape)
print("Test Data:\t", df_test.shape)


# Check for constant features
const_feat = df_train.var()==0
const_feat = const_feat[const_feat]
print(f"Found {len(const_feat)} constant features.")
print("Removing columns")
df_train = df_train[ list(set(df_train.columns) - set(const_feat.index)) ]
df_test = df_test[ list(set(df_test.columns) - set(const_feat.index)) ]

print("New Shapes:")
print("Training Data:\t", df_train.shape)
print("Test Data:\t", df_test.shape)


train_x = df_train.drop(columns=['label'])


n_cols = train_x.shape[1]
outlier_threshold = 0.2 * n_cols
print(f"Outlier threshold: {outlier_threshold/n_cols:.1%}")


z_score_df = (train_x - train_x.mean()) / train_x.std()
potential_outliers = z_score_df[(np.abs(z_score_df) > 3).sum(axis=1) > outlier_threshold].index
print(f"Found {len(potential_outliers)} potential outliers based on Z-score method.")
outliers = potential_outliers


z_score_df = 0.6745 * (train_x - train_x.median()) / (train_x - train_x.median()).abs().median()
potential_outliers = z_score_df[(np.abs(z_score_df) > 3).sum(axis=1) > outlier_threshold].index
print(f"Found {len(potential_outliers)} potential outliers based on modified Z-score method.")
potential_outliers = outliers.intersection(potential_outliers)


q25 = train_x.quantile(0.25, axis=0)
q75 = train_x.quantile(0.75, axis=0)
iqr = q75 - q25
potential_outliers = train_x[((train_x < q25-1.5*iqr) | (train_x > q75+1.5*iqr)).sum(axis=1) > outlier_threshold].index
print(f"Found {len(potential_outliers)} potential outliers based on IQR method.")
potential_outliers = outliers.intersection(potential_outliers)


# Drop outliers
df_train = df_train.drop(index=potential_outliers)
print(f"Dropping {len(potential_outliers)} outliers from training data.")


train_x_mkt = train_x[[col for col in train_x.columns if 'X' not in col]].sort_index(axis=1)


date = dt.date(2024,1,20)
plot_df = train_x_mkt.loc[train_x_mkt.index.date==date].copy()
fig, axs = plt.subplots(nrows=plot_df.shape[1], ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
fig.suptitle(f'Market Features on {date}', fontsize=16)
for i, col in enumerate(plot_df.columns):
    axs[i,0].scatter(plot_df.index, plot_df[col], label=col, color=colors[i], s=0.2)
    axs[i,0].legend(loc='upper right', fontsize='small')
    axs[i,0].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    axs[i,1].hist(plot_df[col][plot_df[col].between(0, plot_df[col].quantile(0.95))], color=colors[i], bins=50)
    
fig.tight_layout()
plt.show()



sns.heatmap(train_x_mkt.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)


train_x_mkt_d = train_x_mkt.groupby(train_x_mkt.index.date).aggregate(['mean', 'std', 'min', 'max', 'sum'])



width = 1
fig, axs = plt.subplots(1, 1, figsize=(14, 4))
fig.suptitle('Daily Trading Volume', fontsize=16)
axs.bar(train_x_mkt_d.index, train_x_mkt_d[('buy_qty', 'sum')], color='green', width=width, label='Bought Quantity')
axs.bar(train_x_mkt_d.index, train_x_mkt_d[('sell_qty', 'sum')], color='firebrick', width=width, label='Sold Quantity', bottom=train_x_mkt_d[('buy_qty', 'sum')])
axs.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()


train_x_X = train_x[[col for col in train_x.columns if 'X' in col]].sort_index(axis=1)


cor_mat_raw = train_x_X.corr()
cor_mat = cor_mat_raw.copy()


fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cor_mat_raw, annot=False, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)


cor_mat = pd.DataFrame(np.triu(cor_mat, k=1), index=cor_mat.index, columns=cor_mat.columns)

perfect_pairs = []
for row in cor_mat.iterrows():
    for col in row[1].index:
        if abs(row[1][col])==1:
            perfect_pairs.append((row[0], col, row[1][col]))
            
print(f"Found {len(perfect_pairs)} perfect pairs of features.")
remove = []
for pair in perfect_pairs:
    if pair[0] not in remove:
        remove.append(pair[0])
    else:
        remove.append(pair[1])


plot_relationship(perfect_pairs[:15], train_x_X, cols=5)


uncorrelated_pairs = []
for row in cor_mat.iterrows():
    for col in row[1].index:
        if row[0]!=col and abs(row[1][col])<0.01:
            uncorrelated_pairs.append((row[0], col, row[1][col]))


plot_relationship(uncorrelated_pairs[:15], train_x_X, cols=5)


# rows = int(np.ceil(train_x_X.shape[1]/80))
# fig, axs = plt.subplots(rows, 1, figsize=(14,3*rows))

# for i in range(rows):
#     plot_df = train_x_X.iloc[:, (i*80):((i+1)*80)]

#     axs[i].violinplot(plot_df, showmeans=False, showmedians=True)
#     axs[i].xaxis.grid(False)
#     axs[i].set_xticks([y + 1 for y in range(plot_df.shape[1])],
#                     labels=plot_df.columns, rotation=90)
# plt.tight_layout()
# plt.show()


target_correlation = {}
target = df_train['label']
for col in df_train.drop(columns='label'):
    target_correlation[col] = np.corrcoef(df_train[col], target)[0][1]
target_correlation = pd.DataFrame(target_correlation.values(), index=target_correlation.keys(), columns=['Corr']).sort_values(by='Corr')


fig, axs = plt.subplots(1, 1, figsize=(12,4))
axs.bar(range(len(target_correlation)), target_correlation['Corr'])
axs.set_xlabel('Feature')
axs.set_ylabel('Pearson Correlation')
axs.set_title('Feature vs Target Correlation')
plt.tight_layout()
plt.show()


target_correlation_top = pd.concat([target_correlation.head(5), target_correlation.tail(5)])


plot_relationship(list(zip(target_correlation_top.index, ['label']*len(target_correlation_top), target_correlation_top['Corr'])), df_train, cols=5)


fig = plt.figure(figsize=(14,6))

gs = GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1])

ax1.plot(df_train['label'])
ax2.plot(df_train['label'].cumsum())
ax3.hist(df_train['label'], bins=81)
ax4.boxplot(df_train['label'], vert=False, patch_artist=True)

plt.tight_layout()
plt.show()


