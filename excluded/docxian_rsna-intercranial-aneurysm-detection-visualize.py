# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# image
import pydicom as dicom

# other stuff
import warnings
warnings.filterwarnings('ignore')

import ast


!ls -l '/kaggle/input/rsna-intracranial-aneurysm-detection'


!ls -l '/kaggle/input/rsna-intracranial-aneurysm-detection/kaggle_evaluation'


!ls -l '/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations'


!ls -l '/kaggle/input/rsna-intracranial-aneurysm-detection/series'


!ls -l '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647'


# configs
pd.set_option('display.max_columns', None) # we want to display all columns in this notebook
pd.set_option('display.max_rows', 100) # increase number of displayed rows
pd.set_option('max_colwidth', None) # make full cells content visible

# random seed
my_random_seed = 123

# aesthetics
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'


t1 = time.time()
df_train = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
df_train_local = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1,4))


# dimensions
df_train.shape


# preview
df_train.head()


# structure details
df_train.info(verbose=True, show_counts=True)


# basic stats
df_train.describe()


df_train.columns


# define features and target
features_num = ['PatientAge']

features_cat = ['PatientSex', 'Modality',
       'Left Infraclinoid Internal Carotid Artery',
       'Right Infraclinoid Internal Carotid Artery',
       'Left Supraclinoid Internal Carotid Artery',
       'Right Supraclinoid Internal Carotid Artery',
       'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
       'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
       'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
       'Right Posterior Communicating Artery', 'Basilar Tip',
       'Other Posterior Circulation']

target = 'Aneurysm Present'


# plot histograms
for f in features_num:
    plt.figure(figsize=(8,3))
    df_train[f].plot(kind='hist', bins=25, color=default_color_1)
    plt.title(f + ' - Train')
    plt.grid()


# plot categorical feature distributions
for f in features_cat:
    plt.figure(figsize=(8,3))
    df_train[f].value_counts().sort_index().plot(kind='bar', color=default_color_1)
    plt.title(f + ' - Train')
    plt.grid()


plt.figure(figsize=(4,4))
df_train[target].value_counts().sort_index().plot(kind='bar', color=default_color_3)
plt.title(target)
plt.grid()
plt.show()


# plot features distributions split by target
for f in features_num:
    plt.figure(figsize=(4,3))
    sns.violinplot(df_train, x=target, y=f,)
    plt.title(f + ' by target')
    plt.grid()
    plt.show()


# impact of categorical features - normalized cross tables
for f in features_cat:
    print('>>> Feature:', f)
    ctab = pd.crosstab(df_train[target], df_train[f])
    ctab_norm = ctab / ctab.sum()
    plt.figure(figsize=(5,2))
    g = sns.heatmap(ctab_norm, annot=True,
                    fmt='.3f', linecolor='black',
                    linewidths=0.5, cmap='Greens', 
                    vmin=0, vmax=+1)
    plt.title(f + ' vs target - train')
    plt.show()


# dimensions
df_train_local.shape


# preview
df_train_local.head()


# locations
df_train_local.location.value_counts()


# convert coordinates to dictionary
df_train_local.coordinates = df_train_local.coordinates.map(ast.literal_eval)


# extract coordinates
def extract_x(my_dict):
    return my_dict['x']

def extract_y(my_dict):
    return my_dict['y']

df_train_local['x'] = df_train_local.coordinates.map(extract_x)
df_train_local['y'] = df_train_local.coordinates.map(extract_y)


# plot coordinates
plt.figure(figsize=(9,9))
g = sns.scatterplot(data=df_train_local, x='x', y='y', hue='location')
g.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=1)
plt.show()


base_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/'


folder = '1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647/'
file = '1.2.826.0.1.3680043.8.498.10124807242473374136099471315028464450.dcm'


filename = base_path + folder + file


# load and display image
ds = dicom.dcmread(filename)
plt.imshow(ds.pixel_array)
plt.show()

