import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e6/train.csv')


df_train.head()


df_train.describe()


df_train.nunique()


df_train.isnull().sum()


df_train.apply(lambda col: (col==0).sum())


df_train.info()


str_columns = list(df_train.select_dtypes(include=["object"]).columns)
str_columns


num_columns = list(df_train.select_dtypes(exclude=["object"]).columns)
num_columns


def print_columns(df,column):
    color_value = 'skyblue'
    df_set = df[column].value_counts().sort_index()
    labels = list(df_set.index)
    values = list(df_set.values)
    plt.barh(labels, values,color=color_value)
    plt.title('columna: ' + column)
    plt.show()

for co in str_columns :
    print_columns(df_train,co)


num_columns_no_id = num_columns.copy()
num_columns_no_id.remove('id')
num_columns_no_id


for co in num_columns_no_id :
    print_columns(df_train,co)



df_train.groupby('Crop Type')[num_columns_no_id].mean().round(2).style.background_gradient('Greens')


df_train.groupby('Soil Type')[num_columns_no_id].mean().round(2).style.background_gradient('Oranges')


df_train.groupby('Fertilizer Name')[num_columns_no_id].mean().round(2).style.background_gradient('Blues')


df_train[['Nitrogen','Potassium', 'Phosphorous']].mean()


df_baseline = df_train.groupby('Fertilizer Name')['id'].count()
df_baseline.plot()


def plot_conditional_cases(df_baseline, df_cond, title):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,width_ratios=[2,2,1])
    fig.set_size_inches(13, 4)
    fig.suptitle(title)
    ax1.plot(df_baseline,)
    ax2.plot(df_cond)
    all_size = df_baseline.sum()
    cond_size = df_cond.sum()
    others_size = all_size - cond_size
    sizes = [cond_size, others_size]
    s_labels = ['subset','others']
    ax3.pie(sizes,labels = s_labels, autopct='%1.1f%%')
    fig.show()


title = 'Nitrogen is higher than potassium'
df_cond = df_train[df_train['Nitrogen'] > df_train['Potassium']].groupby('Fertilizer Name')['id'].count()
plot_conditional_cases(df_baseline, df_cond, title)


title = 'Nitrogen is lower than potassium'
df_cond = df_train[df_train['Nitrogen'] < df_train['Potassium']].groupby('Fertilizer Name')['id'].count()
plot_conditional_cases(df_baseline, df_cond, title)


title = 'Nitrogen is higher than phosphorous'
df_cond = df_train[df_train['Nitrogen'] > df_train['Phosphorous']].groupby('Fertilizer Name')['id'].count()
plot_conditional_cases(df_baseline, df_cond, title)


title = 'Nitrogen is lower than phosphorous'
df_cond = df_train[df_train['Nitrogen'] < df_train['Phosphorous']].groupby('Fertilizer Name')['id'].count()
plot_conditional_cases(df_baseline, df_cond, title)


title = 'Phosphorous is higher than Potassium'
df_cond = df_train[df_train['Phosphorous'] > df_train['Potassium']].groupby('Fertilizer Name')['id'].count()
plot_conditional_cases(df_baseline, df_cond, title)


title = 'Phosphorous is lower than Potassium'
df_cond = df_train[df_train['Phosphorous'] < df_train['Potassium']].groupby('Fertilizer Name')['id'].count()
plot_conditional_cases(df_baseline, df_cond, title)


title = 'Phosphorous is zero'
df_cond = df_train[df_train['Phosphorous'] == 0].groupby('Fertilizer Name')['id'].count()
plot_conditional_cases(df_baseline, df_cond, title)


title = 'Potassium is zero'
df_cond = df_train[df_train['Potassium'] == 0].groupby('Fertilizer Name')['id'].count()
plot_conditional_cases(df_baseline, df_cond, title)


df_procesed = df_train.copy()
columns_to_encode = str_columns
df_procesed[columns_to_encode] = df_procesed[columns_to_encode].apply(lambda col : pd.Categorical(col).codes)


#df_procesed = df_procesed.drop('id',axis=1)
df_corr = df_procesed.corr()
np.fill_diagonal(df_corr.values,0)
df_corr.style.background_gradient(cmap='RdBu',axis=0)

