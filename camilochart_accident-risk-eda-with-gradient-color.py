import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
df_train.head()


df_train.info()


df_train.describe()


df_train.info()


df_train.isnull().sum()


df_train.nunique()


df_train.columns


from matplotlib.colors import LinearSegmentedColormap

colors = ['#F3F0B4', '#F7D292', '#FBB371','#FF4F4F']
custom_cmap = LinearSegmentedColormap.from_list('CustomMap', colors)


target_column = 'accident_risk'


bins = [0, 0.25, 0.5, 0.75, 1.0]
labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0']


df_train['Range_'+target_column] = pd.cut(df_train[target_column],bins=bins, labels=labels, include_lowest=True)
df_train['Range_'+target_column].value_counts().sort_index()


df_train['Range_'+target_column].value_counts().sort_index().plot.bar(color=colors)


df_train['Range_'+target_column].value_counts().sort_index().plot.pie(cmap=custom_cmap,autopct='%1.1f%%',)


str_columns = list(df_train.select_dtypes(include=["object"]).columns)
str_columns


def print_stacked_columns(df,column):

    col_to_stack = 'Range_'+target_column
    df_1 = df_train[[column,col_to_stack]]
    df_1.groupby([column,col_to_stack]).size().unstack().plot(kind='bar', stacked=True,cmap=custom_cmap)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title('columna: ' + column)
    plt.show()


for co in str_columns :
    print_stacked_columns(df_train,co)


bool_columns = list(df_train.select_dtypes(include=["bool"]).columns)
bool_columns


for co in bool_columns :
    print_stacked_columns(df_train,co)


df_train['Range_'+'curvature'] = pd.cut(df_train['curvature'],bins=bins, labels=labels, include_lowest=True)


num_columns = list(df_train.select_dtypes(exclude=["object",'bool']).columns)


num_columns.remove('curvature')
num_columns.remove('accident_risk')
num_columns.remove('Range_accident_risk')


num_columns


for co in num_columns :
    print_stacked_columns(df_train,co)


df_procesed = df_train.copy()
columns_to_encode = str_columns
df_procesed[columns_to_encode] = df_procesed[columns_to_encode].apply(lambda col : pd.Categorical(col).codes)
columns_to_encode = bool_columns
df_procesed[columns_to_encode] = df_procesed[columns_to_encode].apply(lambda col : pd.Categorical(col).codes)


df_procesed = df_procesed.drop(['Range_accident_risk','Range_curvature'],axis=1)


df_corr = df_procesed.corr()
np.fill_diagonal(df_corr.values,0)
df_corr[['accident_risk']].style.background_gradient(cmap='RdBu',axis=0)

