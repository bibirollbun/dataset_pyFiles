# packages

# standard
import numpy as np
import pandas as pd
import time
import os

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# statistics
from scipy import stats


# show files
!ls -l '/kaggle/input/cmi-detect-behavior-with-sensor-data'


# configs
pd.set_option('display.max_columns', None) # we want to display all columns in this notebook
pd.set_option('display.max_rows', 100) # increase rows to be displayed
pd.set_option('display.max_colwidth', None) # show full cell contents

# random seed
my_random_seed = 111

# aesthetics
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'

import warnings
warnings.filterwarnings('ignore')


# load data
t1 = time.time()
df_train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv', low_memory=False)
df_test_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv', low_memory=False)
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1,4))


# preview
df_train_demo.head()


# overview
df_train_demo.info(show_counts=True, verbose=True)


# basic stats - train
df_train_demo.describe(include='all')


# define features

# numerical features
features_num_demo = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

# categorical features
features_cat_demo = ['adult_child', 'sex', 'handedness']


# plot histograms (train and test)
n_bins = 10
for f in features_num_demo:
    plt.figure(figsize=(12,3))
    ax1 = plt.subplot(1,2,1)
    df_train_demo[f].plot(kind='hist', bins=n_bins, color=default_color_1)
    plt.title(f + ' - Train')
    plt.grid()
    ax2 = plt.subplot(1,2,2)
    df_test_demo[f].plot(kind='hist', bins=n_bins, color=default_color_2)
    plt.title(f + ' - Test')
    plt.grid()
    plt.show()


# boxplots (train and test)
for f in features_num_demo:
    plt.figure(figsize=(14,1))
    ax1 = plt.subplot(1,2,1)
    df_temp = df_train_demo[f].dropna() # boxplot does not like missings...
    plt.boxplot(df_temp, vert=False)
    plt.title(f + ' - Train')
    plt.grid()
    ax2 = plt.subplot(1,2,2, sharex=ax1)
    df_temp = df_test_demo[f].dropna()
    plt.boxplot(df_temp, vert=False)
    plt.title(f + ' - Test')
    plt.grid()
    plt.show()


# plot categorical feature distributions (train and test)
for f in features_cat_demo:
    plt.figure(figsize=(14,3))
    ax1 = plt.subplot(1,2,1)
    df_train_demo[f].value_counts().sort_index().plot(kind='bar', color=default_color_1)
    plt.title(f + ' - Train')
    plt.grid()
    ax2 = plt.subplot(1,2,2)
    df_test_demo[f].value_counts().sort_index().plot(kind='bar', color=default_color_2)
    plt.title(f + ' - Test')
    plt.grid()
    plt.show()


# scatterplot of numerical features - training data
sns.pairplot(data=df_train_demo[features_num_demo],
             plot_kws = { 'alpha' : 0.5,
                          's' : 10,
                          'color' : default_color_1})
plt.show()


# Pearson correlation
corr_pearson = df_train_demo[features_num_demo].corr(method='pearson')

# visualize
plt.figure(figsize=(4,3))
sns.heatmap(corr_pearson, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1,
            fmt='.3f', linewidth=0.5, linecolor='black')
plt.title('Pearson Correlation')
plt.show()


# load data
t1 = time.time()
df_train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv', low_memory=False)
df_test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv', low_memory=False)
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1,4))


# dimension of training data
df_train.shape


# dimension of test data
df_test.shape


# preview
df_train.head()


# basic stats
df_train.describe(include='all')


# sequence types
df_train.sequence_type.value_counts()


# plot distribution
df_train.sequence_type.value_counts().plot(kind='bar', color=default_color_3)
plt.title('Sequence Type')
plt.grid()
plt.show()


# define features groups
acc = ['acc_x', 'acc_y', 'acc_z'] # acceleration
rot = ['rot_w', 'rot_x', 'rot_y', 'rot_z'] # orientation in space
thm = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'] # thermopile sensors


# target
target = 'gesture'

# frequencies without considering sequences
df_train[target].value_counts()


# plot target distribution without considering sequences
df_train[target].value_counts().plot(kind='bar', color=default_color_3)
plt.title('Target')
plt.grid()
plt.show()


# use group by to get aggregated view
df_train_group = df_train.groupby('sequence_id', as_index=False).agg(
    gesture = pd.NamedAgg(column='gesture', aggfunc='first'),
    sequence_type = pd.NamedAgg(column='sequence_type', aggfunc='first'),
    n_values = pd.NamedAgg(column='gesture', aggfunc='count'))
# display result
df_train_group


# target frequencies after grouping
df_train_group.gesture.value_counts()


# plot frequencies after grouping
df_train_group.gesture.value_counts().plot(kind='bar', color=default_color_3)
plt.title('Target after grouping by sequence id')
plt.grid()
plt.show()


# sequence type - grouped version
df_train_group.sequence_type.value_counts()


# sequence length distribution
plt.figure(figsize=(8,1))
df_train_group.n_values.plot(kind='box', vert=False)
plt.title('Length of sequence')
plt.grid()
plt.show()


# get one specific sequence for further analysis
my_seq = 'SEQ_015261'
df_seq = df_train[df_train.sequence_id==my_seq]


# stats for this sequence
df_seq.describe(include='all')


# gesture
df_seq.gesture.value_counts()


# phases
df_seq.phase.value_counts()


# behaviors
df_seq.behavior.value_counts()


# visualize acceleration - with behavior
sns.pairplot(df_seq[acc+['behavior']], 
             hue='behavior')
plt.show()


# visualize orientation in space - with behavior
sns.pairplot(df_seq[rot+['behavior']], 
             hue='behavior')
plt.show()


# visualize thermopile sensor values - with behavior
sns.pairplot(df_seq[thm+['behavior']], 
             hue='behavior')
plt.show()


# visualize acceleration - development over time
sns.pairplot(df_seq[acc+['sequence_counter']], 
             hue='sequence_counter')
plt.show()


# visualize acceleration as time series
plt.figure(figsize=(12,4))
plt.plot(df_seq.sequence_counter, df_seq.acc_x, color='blue')
plt.plot(df_seq.sequence_counter, df_seq.acc_y, color='red')
plt.plot(df_seq.sequence_counter, df_seq.acc_z, color='green')
plt.title('Acceleration')
plt.grid()
plt.show()


# visualize orientation as time series
plt.figure(figsize=(12,4))
plt.plot(df_seq.sequence_counter, df_seq.rot_w, color='grey')
plt.plot(df_seq.sequence_counter, df_seq.rot_x, color='blue')
plt.plot(df_seq.sequence_counter, df_seq.rot_y, color='red')
plt.plot(df_seq.sequence_counter, df_seq.rot_z, color='green')
plt.title('Orientation')
plt.grid()
plt.show()


# visualize thermopile sensor values as time series
plt.figure(figsize=(12,4))
plt.plot(df_seq.sequence_counter, df_seq.thm_1, color='blue')
plt.plot(df_seq.sequence_counter, df_seq.thm_2, color='red')
plt.plot(df_seq.sequence_counter, df_seq.thm_3, color='green')
plt.plot(df_seq.sequence_counter, df_seq.thm_3, color='magenta')
plt.plot(df_seq.sequence_counter, df_seq.thm_3, color='cyan')
plt.title('Thermopile Sensors')
plt.grid()
plt.show()

