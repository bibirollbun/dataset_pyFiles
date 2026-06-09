import pandas as pd

#file_path
data_path = '/kaggle/input/porto-seguro-safe-driver-prediction/'

train = pd.read_csv(data_path + 'train.csv', index_col='id')
test  = pd.read_csv(data_path + 'test.csv', index_col='id')
submission  = pd.read_csv(data_path + 'sample_submission.csv', index_col='id')


train.shape, test.shape


train.head()


test.head()


submission.head()


train.info()


import numpy as np
import missingno as msno

train_copy = train.copy().replace(-1, np.nan)

msno.bar(df = train_copy.iloc[:, 1:29],figsize=(13, 6))


msno.bar(df= train_copy.iloc[:, 29:], figsize=(13,6))


msno.matrix(df=train_copy.iloc[:, 1: 29], figsize=(12,6))


def resumtable(df):
  """
  feature summary table
  """
  print(f"Dataset Shape : {df.shape}")
  summary = pd.DataFrame(df.dtypes, columns=['Data Types'])
  summary['Missing Values Cnt'] = (df == -1).sum().values
  summary['Unique Values Cnt']  = df.nunique().values
  summary['Data Kind'] = None

  for col in df.columns:
    if 'bin' in col or col =='target':
      summary.loc[col, 'Data Kind'] = 'Binary'
    elif 'cat' in col:
      summary.loc[col, 'Data Kind'] = 'Nominal'
    elif df[col].dtype == float:
      summary.loc[col, 'Data Kind'] = 'Continuous'
    elif df[col].dtype == int:
      summary.loc[col, 'Data Kind'] = 'Ordinal'

  return summary


summary = resumtable(train)
summary


# Feature extraction with data types being nominal
summary[summary['Data Kind'] =='Nominal'].index


#Feature extraction with data types being real numbers
summary[summary['Data Types']== 'float64'].index


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline


def write_percent(ax, total_size):
  '''
  Display target value ratios at the top of bar graphs while traversing geometric bar graph objects
  '''
  for patch in ax.patches:
    height = patch.get_height()  # bar's height(data count)
    width  = patch.get_width()   # bar's width
    left_coord = patch.get_x()   # bar's left coordinate

    percent = height / total_size * 100 # target value ratio

    # Display text on X-coordinate and Y-coordinates
    ax.text(left_coord + width / 2.0  # x-coordinate
            ,height + total_size * 0.001  # y-coordinate
            ,'{:1.1f}%'.format(percent)   # target value ratio
            ,ha='center'  # align center
            )

mpl.rc('font',size=15)
plt.figure(figsize = (7,6))

ax = sns.countplot(x='target', data=train, palette=sns.color_palette('Set2'))
write_percent(ax, len(train))  # display target value ratio
ax.set_title('Target Value Distirubtion')


import matplotlib.gridspec as gridspec

def plot_target_ratio_by_features(df, features, num_rows, num_cols, size=(12,18)):
  mpl.rc('font', size=9)
  plt.figure(figsize = size)

  grid = gridspec.GridSpec(num_rows, num_cols) # adjust subplots
  plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

  for idx, feature in enumerate(features):
    ax = plt.subplot(grid[idx])

    sns.barplot(x=feature, y='target', data = df, palette='Set2', ax = ax)


bin_features = summary[summary['Data Kind']=='Binary'].index

plot_target_ratio_by_features(train, bin_features, 6, 3)


nom_features = summary[summary['Data Kind']=='Nominal'].index
plot_target_ratio_by_features(train, nom_features, 7, 2)


ord_features = summary[summary['Data Kind']=='Ordinal'].index

plot_target_ratio_by_features(train, ord_features, 8, 2, (12, 20))


cont_features = summary[summary['Data Kind']=='Continuous'].index

plt.figure(figsize=(12,16))
grid = gridspec.GridSpec(5,2)
plt.subplots_adjust(wspace = 0.2, hspace= 0.4)  # adjust space between subplots

for idx, cont_feature in enumerate(cont_features):
  train[cont_feature] = pd.cut(train[cont_feature], 5)

  ax = plt.subplot(grid[idx])
  sns.barplot(x=cont_feature, y='target', data=train, palette='Set2', ax = ax)
  ax.tick_params(axis='x', labelrotation=10)


train_copy.info() # 595212 entries, 7 to 1488027


train_copy = train_copy.dropna()  # Drop np.nan

plt.figure(figsize=(10, 8))
cont_corr = train_copy[cont_features].corr()
sns.heatmap(cont_corr, annot=True, cmap=sns.color_palette('OrRd'))

