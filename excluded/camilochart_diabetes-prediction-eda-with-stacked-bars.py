import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
df_train.head(3)


target_column = 'diagnosed_diabetes'


str_columns = list(df_train.select_dtypes(include=["object"]).columns)

plt.figure(figsize=(14, len(str_columns) * 3))
for idx, feature in enumerate(str_columns, 1):
    df_cont = df_train[feature].value_counts().reset_index()
    df_cont.columns = [feature, 'Frequency']
    
    plt.subplot(len(str_columns), 2, idx)
    sns.barplot(x=feature, y='Frequency', data=df_cont, palette='viridis')
    plt.title(f"{feature}")

plt.tight_layout()
plt.show()


numerical_columns = df_train.select_dtypes(include=["int64", "float64"]).columns

numerical_columns =  numerical_columns.drop(target_column)

plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(df_train[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(df_train[feature].skew(), 2)}")

plt.tight_layout()
plt.show()


from matplotlib.colors import LinearSegmentedColormap

colors = ['#2ABEF4','#F4602A']
custom_cmap = LinearSegmentedColormap.from_list('CustomMap', colors)


bins = [0, 0.5, 1.0]
labels = ['negative diabetes','positive diabetes']


df_train['Range_'+target_column] = pd.cut(df_train[target_column],bins=bins, labels=labels, include_lowest=True)
df_train['Range_'+target_column].value_counts().sort_index()


df_train['Range_'+target_column].value_counts().sort_index().plot.pie(cmap=custom_cmap,autopct='%1.1f%%',)


map_dict = {
    'Low':'01 - Low', 
    'Lower-Middle':'02 - Lower-Middle', 
    'Middle':'03 - Middle', 
    'Upper-Middle':'04 - Upper-Middle',
    'High':'05 - High'
}


df_train['income_level'] = df_train['income_level'].map(map_dict)


def print_stacked_columns(df,column):
    col_to_stack = 'Range_'+target_column
    df_1 = df[[column,col_to_stack]]
    df_1.groupby([column,col_to_stack]).size().unstack().plot(kind='bar', stacked=True,cmap=custom_cmap)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title('column: ' + column)
    plt.show()


str_columns = list(df_train.select_dtypes(include=["object"]).columns)


for co in str_columns :
    print_stacked_columns(df_train,co)


print_stacked_columns(df_train,'age')


print_stacked_columns(df_train,'bmi')


print_stacked_columns(df_train,'physical_activity_minutes_per_week')


def print_both_stacked_charts(df,column):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    col_to_stack = 'Range_'+target_column
    df_1 = df[[column,col_to_stack]]
    df_1 = df_1.groupby([column,col_to_stack]).size().unstack()
    #print('df_1')
    #print(df_1)
    df_1 = df_1.reset_index()
    #print('df_1')
    #print(df_1)
    df_2 = df_1.copy()
    df_2.set_index(column)[['negative diabetes', 'positive diabetes']].plot(
        kind='bar',
        stacked=True,
        ax=axes[0],
    )
    axes[0].set_title('Stacked bars ')
    axes[0].set_ylabel('Count (units)')

    df_3 = df_1.copy()
    df_3['negative diabetes %'] = df_3['negative diabetes'] / (df_3['negative diabetes'] + df_3['positive diabetes'])
    df_3['positive diabetes %'] = df_3['positive diabetes'] / (df_3['negative diabetes'] + df_3['positive diabetes'])

    df_3.set_index(column)[['negative diabetes %', 'positive diabetes %']].plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
    )
    axes[1].set_title('100% Stacked bars ')
    axes[1].set_ylabel('Percent (%)')

    plt.tight_layout()




num_columns = list(df_train.select_dtypes(exclude=["object",'bool']).columns)


num_columns.remove(target_column)
num_columns.remove('Range_'+target_column)


for co in str_columns :
    print_both_stacked_charts(df_train,co)


for co in num_columns :
    print_both_stacked_charts(df_train,co)

