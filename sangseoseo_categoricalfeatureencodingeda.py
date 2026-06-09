import pandas as pd
#data_path
data_path = '/content/drive/MyDrive/Kaggle/input/cat-in-the-dat/'
train   = pd.read_csv(data_path + 'train.csv', index_col='id')
test    = pd.read_csv(data_path +'test.csv', index_col='id')
submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='id')


train.shape, test.shape  # ((300000, 24), (200000, 23))


train.head().T  # Transpos()


submission.head()


test.head().T


train.dtypes


testDF = pd.DataFrame(train.dtypes, columns=['Data Types']).reset_index().rename(columns={'index': 'feature_name'})
print(f"Missing Values : {testDF.isnull().sum().values}")
print(f"Unique Values  : {testDF.nunique().values}")
print(f"1st values : {testDF.loc[0].values}")


def resumetable(df):
  """
  feature summary table
  """
  print(f"Dataset shape : {df.shape}")
  summary = pd.DataFrame(df.dtypes, columns = ['Data types'])
  summary = summary.reset_index()
  summary = summary.rename(columns = {'index': 'Feature'})
  summary['Missing values'] = df.isnull().sum().values
  summary['Unique values'] = df.nunique().values
  summary['1st value'] = df.loc[0].values
  summary['2nd value'] = df.loc[1].values
  summary['3rd value'] = df.loc[2].values

  return summary

resumetable(train)   # Dataset shape : (300000, 24)


for i in range(3):
  feature = 'ord_' + str(i)
  print(f"{feature} Unique Values : {train[feature].unique()}")


for i in range(3, 6):
  feature = 'ord_' + str(i)
  print(f"{feature} Unique Values : {train[feature].unique()}")


print(f"Day's features Unqiue Values : {train['day'].unique()}") # Day's features Unqiue Values : [2 7 5 4 3 1 6]
print(f"Month's feature Unqiue Values : {train['month'].unique()}") # Month's feature Unqiue Values : [ 2  8  1  4 10  3  7  9 12 11  5  6]
print(f"target's Unique Values : {train['target'].unique()}") # target's Unique Values : [0 1


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline


mpl.rc('font',size = 15)
plt.figure(figsize = (7,6))

ax = sns.countplot(x='target', data = train, palette=sns.color_palette('Set2'))
ax.set_title("Target Distribution")


print(ax.patches[0], ax.patches[1])


rectangle = ax.patches[0]
print(f"Rectangles' width : {rectangle.get_width()}")
print(f"Rectangle's Height : {rectangle.get_height()}")
print(f"X coordinate of the left edge of the rectangle : {rectangle.get_x()}")
print(f"Y coordinate of the left edge of the rectangel : {rectangle.get_y()}")


print(f"The X coordinate of text position  : {rectangle.get_x() + rectangle.get_width() / 2.0}")
print(f"The Y coordinate of text position  : {rectangle.get_y() + len(train) * 0.001}")


def write_percent(ax, total_size):
  """
  iterate through the shape objects and display the target value ration at the axes object
  """
  for patch in ax.patches:
    height = patch.get_height()
    width  = patch.get_width()
    left_coord = patch.get_x()  # X-coordinate of the left edge of the rectangle bar
    percent    = height / total_size * 100

    #s : text to display, ha : horizontally align
    ax.text(x = left_coord + width / 2.0, y= height + total_size * 0.003, s= f"{percent : 1.1f}%", ha='center')

plt.figure(figsize=(7, 6))

ax = sns.countplot(x = 'target', data = train, palette=sns.color_palette('Set2'))
write_percent(ax, len(train))
ax.set_title('Target Distribution')


import matplotlib.gridspec as gridspec #arrange multiple graphs in a grid pattern

mpl.rc('font', size = 12)
grid = gridspec.GridSpec(3, 2) # arrange subplot in 3 rows * 2 columns
plt.figure(figsize=(10, 16))
plt.subplots_adjust(wspace = 0.6, hspace = 0.5)

bin_features = [col for col in train.columns if 'bin' in col]

for idx, feature in enumerate(bin_features):
  ax = plt.subplot(grid[idx])

  sns.countplot(x=feature, data = train, hue='target', palette=sns.color_palette('pastel'), ax = ax, saturation=0.5)
  ax.set_title(f"{feature} Distribution by Target")
  write_percent(ax, len(train))


pd.crosstab(train['nom_0'], train['target'])


pd.crosstab(train['nom_0'], train['target'])


pd.crosstab(train['nom_0'], train['target'], normalize='index')  # normalize='index' - normalize each row


help(pd.crosstab)


crosstab = pd.crosstab(train['nom_0'], train['target'], normalize='index') * 100 # display percentage.
crosstab


crosstab.columns   # Index([0, 1], dtype='int64', name='target')


crosstab[1]   # Retrieve data with ratio of target value 1.


crosstab.index


crosstab = crosstab.reset_index()
crosstab


def get_crosstab(df, feature):
  """
  cross-tabulation table creation
  """
  crosstab = pd.crosstab(df[feature], df['target'], normalize='index') * 100
  crosstab = crosstab.reset_index()
  return crosstab


crosstab = get_crosstab(train, 'nom_0')
crosstab


#The proportion of nominal feature `nom_0' with a target value of 1.
crosstab[1]


crosstab[0]  # The ratio of nominal feature `nom_0` with target values 0


def plot_pointplot(ax, feature, crosstab):
  """
  Plot pointplot
  """
  ax2 = ax.twinx()

  ax2 = sns.pointplot(x = feature, y = 1, data = crosstab, order = crosstab[feature].values, color='black', legend=False)
  ax2.set_ylim(crosstab[1].min() - 5,  crosstab[1].max() * 1.1)  # arrange y-axis' scale
  ax2.set_ylabel("Target 1 Ratio(%)")


def plot_cat_dist_with_true_ratio(df, features, num_rows, num_cols, size=(15, 20)):
  '''
  plot_cat_dist_with_true_ratio
  '''
  plt.figure(figsize = size) #Overall figure size
  grid = gridspec.GridSpec(num_rows, num_cols)
  plt.subplots_adjust(wspace = 0.45, hspace = 0.3)

  for idx, feature in enumerate(features):
    ax = plt.subplot(grid[idx])
    crosstab = get_crosstab(df, feature)

    sns.countplot(x = feature, data = df, order = crosstab[feature].values, palette=sns.color_palette('Set2'), ax = ax)
    write_percent(ax, len(df))

    plot_pointplot(ax, feature, crosstab)
    ax.set_title(f"{feature} Distribution")


nom_features = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
plot_cat_dist_with_true_ratio(train, nom_features, num_rows=3, num_cols=2)


ord_features0123 = ['ord_0', 'ord_1', 'ord_2', 'ord_3']
plot_cat_dist_with_true_ratio(train, ord_features0123, num_rows=2, num_cols=2, size=(15,12))


ord_features45 = ['ord_4', 'ord_5']
plot_cat_dist_with_true_ratio(train, ord_features45, num_rows=2, num_cols=1, size=(15,12))


from pandas.api.types import CategoricalDtype

ord_1_values = ['Novice', 'Contributor','Expert', 'Master','Grandmaster']
ord_2_values = ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']

ord_1_type = CategoricalDtype(categories=ord_1_values, ordered=True)
ord_2_type = CategoricalDtype(categories=ord_2_values, ordered=True)

train['ord_1'] = train['ord_1'].astype(ord_1_type)
train['ord_2'] = train['ord_2'].astype(ord_2_type)





plot_cat_dist_with_true_ratio(train, ord_features0123, num_rows=2, num_cols=2, size=(15, 12))


date_features = ['day', 'month']

plot_cat_dist_with_true_ratio(train, date_features, num_rows=2, num_cols = 1, size=(10, 12))

