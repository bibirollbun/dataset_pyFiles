# install package for distribution fitting
!pip install fitter


# packages

# standard
import numpy as np
import pandas as pd
import time
import gc

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# statistics
from fitter import Fitter, get_common_distributions, get_distributions

# warnings
import warnings
warnings.filterwarnings('ignore')


# configs 

# show all columns
pd.set_option('display.max_columns', 500)

# colors
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'

# random seed
random_seed = 123


# load training data
t1 = time.time()
df_train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1, 2))


# preview
df_train.head()


# structure details
df_train.info(verbose=True, show_counts=True)


# main features
features_main = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

# anonymized features
features_x = ['X' + str(i) for i in range(1,890+1)]

# target
target = 'label'


# basic stats main features
df_train[features_main].describe()


# plot main features
for f in features_main:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax1.hist(df_train[f], bins=100, color=default_color_1)
    ax1.grid()
    ax1.set_title(f)
    # for boxplot we need to remove the NaNs first
    feature_wo_nan = df_train[~np.isnan(df_train[f])][f]
    ax2.boxplot(feature_wo_nan, vert=False)
    ax2.grid()
    ax2.set_title(f + ' - boxplot')
    plt.show()


# plot main features as time series
for f in features_main:
    plt.figure(figsize=(14,4))
    plt.scatter(df_train.index, df_train[f], 
            color=default_color_1, s=1)
    plt.title(f + ' (Time Series)')
    plt.grid()
    plt.show()


# correlation
corr_pearson = df_train[features_main+[target]].corr(method='pearson')
corr_spearman = df_train[features_main+[target]].corr(method='spearman')

plt.figure(figsize=(12,4))
ax1 = plt.subplot(1,2,1)
sns.heatmap(corr_pearson, annot=True, cmap='RdYlGn', 
            vmin=-1, vmax=+1, fmt='.3f',
            linecolor='black', linewidths=0.5)
plt.title('Pearson Correlation')

ax2 = plt.subplot(1,2,2, sharex=ax1)
sns.heatmap(corr_spearman, annot=True, cmap='RdYlGn', 
            vmin=-1, vmax=+1, fmt='.3f',
            linecolor='black', linewidths=0.5)
plt.title('Spearman Correlation')
plt.show()


# garbage collection
gc.collect();


# calc basic stats for anonymized features
df_train[features_x].describe()


# garbage collection
gc.collect();


# boxplot of all anonymized variables
for i in range(17):
    print('Columns', 50*i+1 , 'to', 50*i+50)
    df_train[features_x].iloc[:,(50*i):(50*i+50)].plot(kind='box', figsize=(15,5))
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

# separate plot for reminaing columns
print('Columns', 851 , 'to', 890)
df_train[features_x].iloc[:,850:(850+1+50)].plot(kind='box', figsize=(15,5))
plt.xticks(rotation=90)
plt.grid()
plt.show()


# garbage collection
gc.collect();


df_train['X697'].value_counts()


df_train['X864'].value_counts()


# define columns to be removed
drop_cols = ['X' + str(i) for i in range(697,717+1)]
drop_cols = drop_cols + ['X864','X867','X869','X870','X871','X872']
print(drop_cols)


# remove redundant columns
df_train.drop(drop_cols, axis=1, inplace=True)


# garbage collection
gc.collect();


# adjust list of features accordingly
features_x = [x for x in features_x if x not in drop_cols]


# calc correlations (this takes quite some time...)
t1 = time.time()
corr_pearson_x = df_train[features_x].corr(method='pearson')
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1, 2))


# create data frame to store all results
n_features = len(features_x)
n_rows = int(n_features*(n_features - 1) / 2)
corr_stats = pd.DataFrame(data=np.zeros((n_rows,3)), columns=['x','y','corr'])
corr_stats.x = corr_stats.x.astype(str)
corr_stats.y = corr_stats.y.astype(str)

# rearrange all correlations in tabular form
row = 0
for i in range(n_features):
    var_i = features_x[i]
    for j in range(n_features):
        if i<j:
            var_j = features_x[j]
            corr_x = corr_pearson_x.iloc[i,j]
            # store results
            corr_stats.loc[row,'x'] = var_i
            corr_stats.loc[row,'y'] = var_j
            corr_stats.loc[row,'corr'] = corr_x
            row = row + 1

# sort by correlation (descending)
corr_stats = corr_stats.sort_values(by=['corr'], ascending=False)
corr_stats = corr_stats.reset_index(drop=True)


# top 10 correlations
corr_stats.head(50)


# bottom 10 correlations
corr_stats.tail(25)


# plot correlations
plt.figure(figsize=(8,5))
plt.hist(corr_stats['corr'], 100, color = default_color_1)
plt.title('Feature Correlations')
plt.grid()
plt.show()


# export results
corr_stats.to_csv('correlation_stats.csv')


# histogram of target
plt.figure(figsize=(12,4))
plt.hist(df_train[target], bins=1000, color=default_color_3)
plt.title('Target (Histogram)')
plt.grid()
plt.show()


# boxplot of target
plt.figure(figsize=(12,1))
plt.boxplot(df_train[target], vert=False)
plt.title('Target (Boxplot)')
plt.grid()
plt.show()


# target as time series
plt.figure(figsize=(14,6))
plt.scatter(df_train.index, df_train[target], 
            color=default_color_3, s=1, alpha=1)
plt.title('Target (Time Series)')
plt.grid()
plt.show()


# try to fit a few distribution types to target
# we use a subset to achieve a reasonable run time
data = df_train[target].sample(50000, random_state=random_seed)
dist_fitter = Fitter(data,
                     distributions=['lognorm', 't', 'cauchy',
                                    'genhyperbolic', 'norminvgauss', 'tukeylambda', 
                                    'gennorm', 'dgamma', 'johnsonsu'], 
                     timeout=300)
dist_fitter.fit()
plt.figure(figsize=(12,5))
dist_fitter.summary(9)


# just plotting the fitted PDFs
plt.figure(figsize=(12,5))
dist_fitter.plot_pdf(Nbest=9, lw=1)


# best fit
dist_fitter.get_best()


# split target in positive and negative values
upside = df_train[target][df_train[target]>0]
dnside = -df_train[target][df_train[target]<0]


# stats for positive targets
upside.describe()


# stats for negative targets (with sign switched)
dnside.describe()


# histogram of target upside
plt.figure(figsize=(12,4))
plt.hist(np.log10(upside), bins=200, color=default_color_3)
plt.title('log10(Target Upside) (Histogram)')
plt.grid()
plt.show()


# histogram of target downside
plt.figure(figsize=(12,4))
plt.hist(np.log10(dnside), bins=200, color=default_color_3)
plt.title('-log10(Target Downside) (Histogram)')
plt.grid()
plt.show()

