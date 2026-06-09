# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# warning handling
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# configs
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 150)

default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'


# show files
!ls -l '../input/stanford-rna-3d-folding'


# load train and validation data
df_train = pd.read_csv('../input/stanford-rna-3d-folding/train_labels.csv')
df_valid = pd.read_csv('../input/stanford-rna-3d-folding/validation_labels.csv')

df_train_seq = pd.read_csv('../input/stanford-rna-3d-folding/train_sequences.csv')
df_valid_seq = pd.read_csv('../input/stanford-rna-3d-folding/validation_sequences.csv')


# first glance
df_train.head(12)


# aux function for id extraction
def extract_seq_id(s):
    split = s.split('_')
    n = len(split)
    if n==3:
        result = split[0] + '_' + split[1]
    else:
        result = split[0]
    return result


# add sequence id removing the numbers at the end of the id
df_train['id_seq'] = df_train.ID.apply(extract_seq_id)
df_valid['id_seq'] = df_valid.ID.apply(extract_seq_id)


# structure of data frame
df_train.info(verbose=True, show_counts=True)


# show stats
df_train.describe(include='all').transpose()


# stats for "resname"
df_train.resname.value_counts()


df_train[df_train.resname=='-']


df_train[df_train.resname=='X']


# stats for "resid" - show top 25
df_train.resid.value_counts()[0:25]


# plot as histogram
df_train.resid.plot(kind='hist', bins=100, color=default_color_1)
plt.title('resid counts')
plt.grid()
plt.show()


# stats for sequence is - show top 25
df_train.id_seq.value_counts()[0:25]


# visualize coordinates
sns.pairplot(data=df_train[['x_1', 'y_1', 'z_1']],
             diag_kws = {'color' : default_color_1},
             plot_kws = {'s' : 1, 
                         'alpha' : 0.25,
                         'color' : default_color_1})
plt.show()


# sequence table
df_train_seq.head()


# first glance
df_valid.head(12)


# replace extreme values that encode missings by NaN
df_valid.replace(to_replace=-1E18, value=np.nan, inplace=True);


# structure of data frame
df_valid.info(verbose=True, show_counts=True)


# show stats
df_valid.describe(include='all').transpose()


# visualize first coordinates
sns.pairplot(data = df_valid[['x_1', 'y_1', 'z_1']],
             diag_kws = {'color' : default_color_1},
             plot_kws = {'s' : 5, 
                         'alpha' : 0.5,
                         'color' : default_color_1})
plt.show()


# visualize coordinates - plot the resname via color encoding
sns.pairplot(data=df_valid[['x_1', 'y_1', 'z_1', 'resname']],
             hue = 'resname',
             diag_kws = {'color' : default_color_1},
             plot_kws = {'s' : 5, 
                         'alpha' : 0.5,
                         'color' : default_color_1})
plt.show()


# visualize second coordinates
sns.pairplot(data = df_valid[['x_2', 'y_2', 'z_2']],
             diag_kws = {'color' : default_color_1},
             plot_kws = {'s' : 5, 
                         'alpha' : 0.5,
                         'color' : default_color_1})
plt.show()


# visualize third coordinates
sns.pairplot(data = df_valid[['x_3', 'y_3', 'z_3']],
             diag_kws = {'color' : default_color_1},
             plot_kws = {'s' : 5, 
                         'alpha' : 0.5,
                         'color' : default_color_1})
plt.show()


# sequence table
df_valid_seq.head()


# function for 3d plotting
def create_plots(i_df, i_title, i_varx='x_1', i_vary='y_1', i_varz='z_1'):
    
    # 1st plot colored by resname
    sns.pairplot(data=i_df[[i_varx, i_vary, i_varz, 'resname']],
                 hue = 'resname',
                 diag_kws = {'color' : default_color_1},
                 plot_kws = {'s' : 25, 
                             'alpha' : 1,
                             'color' : default_color_1}).fig.suptitle(i_title, y=1.05)
    plt.show()

    # 2nd plot colored by resid
    sns.pairplot(data=i_df[[i_varx, i_vary, i_varz, 'resid']],
                 hue = 'resid',             
                 diag_kws = {'color' : default_color_1},
                 plot_kws = {'s' : 25, 
                             'alpha' : 1,
                             'color' : default_color_1})
    plt.show()


# pick an example
my_id = '1SCL_A'
df_ex = df_train[df_train.id_seq == my_id].copy()
df_ex


# corresponding sequence / description
df_temp = df_train_seq[df_train_seq.target_id==my_id].reset_index(drop=True)
my_desc = df_temp.description[0]
print(my_desc)
my_seq = df_temp.sequence[0]
print(my_seq)


create_plots(df_ex, i_title = my_id + ' - ' + my_seq)


# another - very complex - example
my_id = '4V6X_A5'
df_ex = df_train[df_train.id_seq == my_id].copy()
df_ex


# corresponding sequence / description
df_temp = df_train_seq[df_train_seq.target_id==my_id].reset_index(drop=True)
my_desc = df_temp.description[0]
print(my_desc)
my_seq = df_temp.sequence[0]
print(my_seq)


create_plots(df_ex, i_title = my_id)


# interactive 3d plot using plotly
df_ex['size4plot'] = 1 # artificial column to allow size scaling
fig = px.scatter_3d(df_ex,
                    x='x_1', y='y_1', z='z_1',
                    color='resname',
                    size='size4plot',
                    size_max=8,
                    hover_data=['id_seq', 'resid'],
                    opacity=0.5)
fig.update_layout(title=my_id)
fig.show(renderer='iframe')


# sequences in validation data
df_valid.id_seq.value_counts()


# pick an example from the validation set with more than one structure
my_id = 'R1156'
df_ex = df_valid[df_valid.id_seq == my_id]
df_ex


# corresponding sequence / description
df_temp = df_valid_seq[df_valid_seq.target_id==my_id].reset_index(drop=True)
my_desc = df_temp.description[0]
print(my_desc)
my_seq = df_temp.sequence[0]
print(my_seq)


# visualize first structure
create_plots(df_ex, my_id + ' - Structure 1', 'x_1', 'y_1', 'z_1')


# visualize second structure
create_plots(df_ex, my_id + ' - Structure 2', 'x_2', 'y_2', 'z_2')


# visualize third structure
create_plots(df_ex, my_id + ' - Structure 3', 'x_3', 'y_3', 'z_3')


# interactive 3d plot using plotly
df_ex_temp = df_ex.copy()
df_ex_temp['size4plot'] = 1 # artificial column to allow size scaling
fig = px.scatter_3d(df_ex_temp,
                    x='x_1', y='y_1', z='z_1',
                    color='resname',
                    size='size4plot',
                    size_max=8,
                    hover_data=['id_seq', 'resid'],
                    opacity=0.5)
fig.update_layout(title=my_id)
fig.show(renderer='iframe')


# function for comparing two versions
def compare_structures(i_df, i_index_1, i_index_2):

    # copy first structure coordinates in temporary data frame
    df_A = df_ex[['id_seq', 'resname']].copy()
    df_A['x'] = df_ex['x_' + str(i_index_1)]
    df_A['y'] = df_ex['y_' + str(i_index_1)]
    df_A['z'] = df_ex['z_' + str(i_index_1)]
    df_A['Structure'] = 'Structure ' + str(i_index_1)

    # copy second structure coordinates in temporary data frame
    df_B = df_ex[['id_seq', 'resname']].copy()
    df_B['x'] = df_ex['x_' + str(i_index_2)]
    df_B['y'] = df_ex['y_' + str(i_index_2)]
    df_B['z'] = df_ex['z_' + str(i_index_2)]
    df_B['Structure'] = 'Structure ' + str(i_index_2)

    # concatenate the two data frames
    df_compare = pd.concat([df_A,df_B])

    # visualize comparison using newly introduced Structure variable
    sns.pairplot(data = df_compare[['x', 'y', 'z', 'Structure']],
                 hue = 'Structure',
                 plot_kws = {'s' : 10, 
                             'alpha' : 1})
    plt.show()


compare_structures(df_ex, 1, 2)


compare_structures(df_ex, 2, 3)

