import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
sample_submission


test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
test_identity


test_identity.columns


test_identity.describe()


test_identity.describe(include='all') 



# to reset this
pd.reset_option('display.max_columns')


test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_transaction


test_transaction.columns


test_transaction.describe()


# to reset this
pd.reset_option('display.max_columns')





# to reset this
pd.reset_option('display.max_columns')


train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
train_identity


train_identity.shape


train_identity.count()


train_identity[['id_'+str(i) for i in range(12,39)]] = train_identity[['id_'+str(i) for i in range(12,39)]].astype(object)


train_identity.dtypes


train_identity_categoric_columns = train_identity.loc[:,train_identity.dtypes==object].columns 
train_identity_numeric_columns = train_identity.loc[:,train_identity.dtypes!=object].columns 


pd.set_option('display.max_columns', None)
train_identity[list(train_identity_numeric_columns)].describe() 



train_identity[list(train_identity_categoric_columns)].describe() 



train_identity.isna().sum()


train_identity.isna().sum()/train_identity.shape[0]


((~train_identity.isna()).sum())/train_identity.shape[0] # ненулевые


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_transaction


train_transaction.shape


pd.set_option('display.max_rows', None)
train_transaction.count()


train_transaction[['M'+str(i) for i in range(1,10)]] = train_transaction[['M'+str(i) for i in range(1,10)]].astype(object)


train_transaction[['card'+str(i) for i in range(1,7)]] = train_transaction[['card'+str(i) for i in range(1,7)]].astype(object)


train_transaction[['addr'+str(i) for i in range(1,3)]] = train_transaction[['addr'+str(i) for i in range(1,3)]].astype(object)


train_transaction[[str(i) + '_emaildomain' for i in ['P','R'] ]] = train_transaction[[str(i) + '_emaildomain' for i in ['P','R']]].astype(object)


pd.set_option('display.max_rows', None)

train_transaction.dtypes


pd.reset_option('display.max_rows')


# to reset this
pd.reset_option('display.max_columns')


df = pd.merge(train_transaction,train_identity,'left','TransactionID')


df


df.shape


pd.set_option('display.max_rows', None)

df.dtypes


df.shape


train_identity.shape


train_identity.shape[1] + train_transaction.shape[1] # транзакций id повторяется же


df.count


df.shape


df.isna().sum()/df.shape[0]


### Изучим корелляции вещественных признаков
# to reset this
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')
categoric_columns = df.loc[:,df.dtypes==object].columns 
numeric_columns = df.loc[:,df.dtypes!=object].columns 
df[numeric_columns].corr() # слишком много колонок


%time
corr_matr = df[numeric_columns].corr() 


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

corr_matr


corr_matr.to_csv('corr_matrix_mfdp_antifraud.csv',index=False)


### функции для фильтрации признаков

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df[numeric_columns], 10))


### функции для фильтрации признаков c готвой матрицей корреляций

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df,df_corr, n=5):
    au_corr = df_corr.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df[numeric_columns],corr_matr, 1000))


def correlation(dataset,corr_matrix, threshold):
    col_corr = set() # Set of all the names of deleted columns
    #corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                #if colname in dataset.columns:
                    #del dataset[colname] # deleting the column from the dataset
    return col_corr
                    
    
correlation(df,corr_matr, 0.9)


# согласно Корреляционному анализу


# не справляется - очень долго выводит
# надо удалить колонки , которые не заполнены больше чем не половину- так как очевидно , что мало играют роли
morethanhalfna_cols = df.loc[:,(df.isna()).sum() > df.shape[0]/2].columns
#data = data.drop(morethanhalfna_cols,axis=1)


morethanhalfna_cols


len(morethanhalfna_cols) #0.5


# не справляется - очень долго выводит
# надо удалить колонки , которые не заполнены больше чем не половину- так как очевидно , что мало играют роли
morethan09_cols = df.loc[:,(df.isna()).sum() > 0.9 * df.shape[0]].columns



morethan09_cols


# не справляется - очень долго выводит
# надо удалить колонки , которые не заполнены больше чем не половину- так как очевидно , что мало играют роли
morethan08_cols = df.loc[:,(df.isna()).sum() > 0.8 * df.shape[0]].columns



morethan08_cols


# не справляется - очень долго выводит
# надо удалить колонки , которые не заполнены больше чем не половину- так как очевидно , что мало играют роли
morethan07_cols = df.loc[:,(df.isna()).sum() > 0.7 * df.shape[0]].columns



morethan07_cols


list(morethan07_cols)


# просто так не срезать - пока не будем


### Заполним средним

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())


for col in categoric_columns:
    most_recent = df.groupby(col).size().sort_values().index[-1] # по возрастанию же
    df[col] = df[col].fillna(most_recent)

df.describe(include='object') # теперь везде заполнено


# попробуем обучать на всех - заполнили самыми ходовыми 
df.isna().sum()/df.shape[0]


# попробуем обучать на всех - заполнили самыми ходовыми 
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')
df.loc[:,df.isna().sum()/df.shape[0] == 0]


# 0 незаполненных колонок
df.loc[:,df.isna().sum()/df.shape[0] != 0]


categoric_columns


pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')


pd.set_option('display.max_columns', None)

df.describe(include='object')


list(numeric_columns)


list(categoric_columns)


pd.set_option('display.max_rows', None)
df.dtypes


df[categoric_columns]


df[['M'+str(i) for i in range(1,10)]] = df[['M'+str(i) for i in range(1,10)]].astype(object)
df[['card'+str(i) for i in range(1,7)]] = df[['card'+str(i) for i in range(1,7)]].astype(object)
df[['addr'+str(i) for i in range(1,3)]] = df[['addr'+str(i) for i in range(1,3)]].astype(object)
df[[str(i) + '_emaildomain' for i in ['P','R'] ]] = df[[str(i) + '_emaildomain' for i in ['P','R']]].astype(object)
df[['id_'+str(i) for i in range(12,39)]] = df[['id_'+str(i) for i in range(12,39)]].astype(object)


df.dtypes


df.dtypes


from catboost import CatBoostClassifier # обучил, чтобы взять feature importances

X_train = df.drop(['isFraud','TransactionID'], axis=1)
y_train = df['isFraud']

cboost = CatBoostClassifier()
X_train[['M'+str(i) for i in range(1,10)]] = X_train[['M'+str(i) for i in range(1,10)]].astype('str')
X_train[['card'+str(i) for i in range(1,7)]] = X_train[['card'+str(i) for i in range(1,7)]].astype('str')
X_train[['addr'+str(i) for i in range(1,3)]] = X_train[['addr'+str(i) for i in range(1,3)]].astype('str')
X_train[[str(i) + '_emaildomain' for i in ['P','R'] ]] = X_train[[str(i) + '_emaildomain' for i in ['P','R']]].astype('str')
X_train[['id_'+str(i) for i in range(12,39)]] = X_train[['id_'+str(i) for i in range(12,39)]].astype('str')
cboost.fit(X_train,
           y_train,
           cat_features=list(categoric_columns))
# конвертнул заново из-за CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=5]=321.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.

#y_pred = cboost.predict(X_test)

#print(classification_report(y_test, y_pred, digits=4))


cboost.save_model('mfdp_simple_cboost',
           format="cbm")


X_train = df.drop(['isFraud','TransactionID'], axis=1)
y_train = df['isFraud']


!tar -zcvf outputname.tar.gz /kaggle/working


%cd /kaggle/working
from IPython.display import FileLink
FileLink('outputname.tar.gz')


from catboost import CatBoostClassifier
model = CatBoostClassifier()      # parameters not required.
model.load_model('/kaggle/input/mfdp_simple_cboost_antifraud/other/default/1/mfdp_simple_cboost')


# раньше cboost был
model.get_feature_importance()#feature_importances_


### Какие признаки оказались наиболее важны в model?
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8, 5)

import warnings
warnings.filterwarnings('ignore')
import time
fi_df = pd.DataFrame({'feature_names': df.columns.drop(['isFraud','TransactionID']),
                      'feature_importance': model.get_feature_importance()#model.feature_importances_
                     }) # на cboost грузилось так

plt.figure(figsize=(10,8))
sns.barplot(x='feature_importance', y='feature_names', 
            data=fi_df.sort_values('feature_importance', ascending=False))
plt.title('GBM catboost feature importance');


### Какие признаки оказались наиболее важны в model?
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8, 5)

import warnings
warnings.filterwarnings('ignore')
import time
fi_df = pd.DataFrame({'feature_names': df.columns.drop(['isFraud','TransactionID']),
                      'feature_importance': model.get_feature_importance()#model.feature_importances_
                     }) # на cboost грузилось так

plt.figure(figsize=(10,8))
fig = sns.barplot(x='feature_importance', y='feature_names', 
            data=fi_df.sort_values('feature_importance', ascending=False))
plot_fig = fig.get_figure()
plot_fig.savefig('GBM catboost feature importance sns.png')# пустоый вывод # так сохранилось без header
plt.title('GBM catboost feature importance');


plt.savefig('GBM catboost feature importance.png')# пустоый вывод


sns.savefig('GBM catboost feature importance sns.png')# пустоый вывод


fi_df.sort_values('feature_importance', ascending=False) # после замены на функцию - прогрузилось после загрузки модеоли из файла








import gc
del train_identity
del train_transaction
gc.collect()


import gc

gc.collect()


%whos


df_test = pd.merge(test_transaction,test_identity,'left','TransactionID')


df_test.shape


df_test.head(10)


df.columns


set(df.columns).difference(set(df_test.columns))


set(df_test.columns).difference(set(df.columns))


df_test.columns


list(df_test.columns) # ответа нет - поэтому на 1 меньше - но не ясно почему diff не лови


# наименования другие - надо переименовать


df_test.rename(columns={
    x : x.replace('-', '_') for x in list(
        set(df_test.columns).difference(set(df.columns))
                                         ) 
}
              ,inplace=True) # без inplace=True долго грузилось - так как новый обьект строился





df_test[['M'+str(i) for i in range(1,10)]] = df_test[['M'+str(i) for i in range(1,10)]].astype(object)
df_test[['card'+str(i) for i in range(1,7)]] = df_test[['card'+str(i) for i in range(1,7)]].astype(object)
df_test[['addr'+str(i) for i in range(1,3)]] = df_test[['addr'+str(i) for i in range(1,3)]].astype(object)
df_test[[str(i) + '_emaildomain' for i in ['P','R'] ]] = df_test[[str(i) + '_emaildomain' for i in ['P','R']]].astype(object)
df_test[['id_'+str(i) for i in range(12,39)]] = df_test[['id_'+str(i) for i in range(12,39)]].astype(object)


df_test.columns


for col in categoric_columns:
    most_recent = df.groupby(col).size().sort_values().index[-1] # по возрастанию же
    
    df_test[col] = df_test[col].fillna(most_recent) # заполнение данными из train!

### Заполним средним
for col in [item for item in numeric_columns if item not in ['isFraud']]: #KeyError: 'isFraud'
    df_test[col] = df_test[col].fillna(df[col].mean())# заполнение данными из train!
    
X_test = df_test.drop(['TransactionID'], axis=1)

X_test[['M'+str(i) for i in range(1,10)]] = X_test[['M'+str(i) for i in range(1,10)]].astype('str')
X_test[['card'+str(i) for i in range(1,7)]] = X_test[['card'+str(i) for i in range(1,7)]].astype('str')
X_test[['addr'+str(i) for i in range(1,3)]] = X_test[['addr'+str(i) for i in range(1,3)]].astype('str')
X_test[[str(i) + '_emaildomain' for i in ['P','R'] ]] = X_test[[str(i) + '_emaildomain' for i in ['P','R']]].astype('str')
X_test[['id_'+str(i) for i in range(12,39)]] = X_test[['id_'+str(i) for i in range(12,39)]].astype('str')

y_pred = #cboost.predict(X_test)
#print(classification_report(y_test, y_pred, digits=4)) - у нас нет y_test


y_pred = model.predict(X_test)#cboost.predict(X_test)



y_pred


pd.Series(y_pred).value_counts()


y_pred_proba = model.predict_proba(X_test)#cboost.predict_proba(X_test)


y_pred_proba


y_pred.shape


y_pred_proba.shape


Return value
Predictions for the given dataset.

The return value type depends on the number of input objects:

Single object — One-dimensional numpy.ndarray with probabilities for every class.
Multiple objects — Two-dimensional numpy.ndarray of shape (number_of_objects, number_of_classes) with the probability for every class for each object.


y_pred_proba


df_test['TransactionID'].head(5)


pd.Series(y_pred).head(5)#ValueError: Length of values (506691) does not match length of index (1)



#pd.Series(y_pred_proba)#ValueError: Data must be 1-dimensional, got ndarray of shape (506691, 2) instead
pd.DataFrame(y_pred_proba,columns=['prob_1','prob_2']).head(5)


rez = pd.concat(
    [df_test['TransactionID'],pd.Series(y_pred),pd.DataFrame(y_pred_proba,columns=['prob_1','prob_2'])]
    , axis=1)


rez.rename(columns={0:'isFraud'},inplace=True)# там название было числом


rez.head(5)


rez.to_csv('test_cboost.csv',index=False)


FileLink('test_cboost.csv') # надо возвращать prob_2 в prediction


rez_for_subm = rez[['TransactionID','prob_2']].copy()


rez_for_subm.rename(columns = {'prob_2':'isFraud'},inplace=True)


rez_for_subm.head(5)


rez_for_subm.to_csv('test_cboost_subm.csv',index=False)


FileLink('test_cboost_subm.csv') # надо возвращать prob_2 в prediction


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
X_train[['M'+str(i) for i in range(1,10)]] = X_train[['M'+str(i) for i in range(1,10)]].astype('str')
X_train[['card'+str(i) for i in range(1,7)]] = X_train[['card'+str(i) for i in range(1,7)]].astype('str')
X_train[['addr'+str(i) for i in range(1,3)]] = X_train[['addr'+str(i) for i in range(1,3)]].astype('str')
X_train[[str(i) + '_emaildomain' for i in ['P','R'] ]] = X_train[[str(i) + '_emaildomain' for i in ['P','R']]].astype('str')
X_train[['id_'+str(i) for i in range(12,39)]] = X_train[['id_'+str(i) for i in range(12,39)]].astype('str')
#CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=4]=321.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.
y_train_pred = model.predict(X_train)#cboost.predict(X_train)

print(classification_report(y_train, y_train_pred, digits=8)) #- у нас нет y_test


model.predict_proba(X_train)#cboost.predict_proba(X_train)


#pd.DataFrame(cboost.predict_proba(X_train))


from sklearn.metrics import precision_recall_curve, auc

# True binary labels
# Predicted probabilities or decision function scores
y_train_scores = pd.DataFrame(model.predict_proba(X_train))#cboost.predict_proba(X_train))
# Generate precision and recall values
precision, recall, _ = precision_recall_curve(y_train, y_train_scores[1])

# Compute PR AUC using the trapezoidal rule
pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.8f}")


print(f"accuracy score: {accuracy_score(y_train, y_train_pred):.8f}") #ValueError: Classification metrics can't handle a mix of binary and continuous targets



from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(f"precision score: {precision_score(y_train, y_train_pred):.8f}") 
print(f"recall score: {recall_score(y_train, y_train_pred):.8f}") 
print(f"f1 score: {f1_score(y_train, y_train_pred):.8f}") 









### Изучим корелляции вещественных признаков
# to reset this
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')
categoric_columns = df.loc[:,df.dtypes==object].columns 
numeric_columns = df.loc[:,df.dtypes!=object].columns 
df[numeric_columns].corr() # слишком много колонок





%time
corr_matr = df[numeric_columns].corr() 


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

corr_matr


corr_matr.to_csv('corr_matrix_mfdp_antifraud.csv',index=False)


### функции для фильтрации признаков

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df[numeric_columns], 10))


### функции для фильтрации признаков c готвой матрицей корреляций

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df,df_corr, n=5):
    au_corr = df_corr.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df[numeric_columns],corr_matr, 1000))


def correlation(dataset,corr_matrix, threshold):
    col_corr = set() # Set of all the names of deleted columns
    #corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                #if colname in dataset.columns:
                    #del dataset[colname] # deleting the column from the dataset
    return col_corr
                    
    
correlation(df,corr_matr, 0.9)


# скопировал ячейки через shift - да , удалим эти признаки из-за высокой корелляции >0.9 - за исключением


corr_more09= correlation(df,corr_matr, 0.9)


corr_more09


df_09corr = df.copy()


df_09corr


for colname in corr_more09: #у нас другие timedelta остаются
    del df_09corr[colname] # deleting the column from the dataset
#df_09corr.drop(corr_more09, axis=1) 


df_09corr


df_09corr.dtypes# гораздо меньше


list(df_09corr.columns)


from sklearn.feature_selection import VarianceThreshold
numeric_columns_09 = df_09corr.loc[:,df_09corr.dtypes!=object].columns 
cutter = VarianceThreshold(threshold=0.1)
cutter.fit(df_09corr[numeric_columns_09])# посмотрим - но тут не приводили к масштабу


cutter.get_feature_names_out()


df_09corr[cutter.get_feature_names_out()].describe()
# не константные


pd.set_option('display.max_columns', None)

constant_cols_susp = [x for x in numeric_columns_09 if x not in cutter.get_feature_names_out()]
# надо приводить - так как мкасштаб разные
df_09corr[constant_cols_susp].describe() # std - корень из дисперсии


# все величины нормального масштаба - не меньше 1 по модулю (по разности min,max) - их можно смело удалять
# и в каждой группе признаков c,d,v,id еще останется много подобных - но больше имзеняющихся!
# можно только бинарные оставить 


for i in constant_cols_susp:
    print(f'{i}_col value_counts: ',df_09corr[i].value_counts())


# v14,v41,v65,v107,V305 близки к бинарным -не будем удалять
list(cutter.get_feature_names_out()) + ['V14','V41','V65','V107','V305','isFraud']


list(df_09corr.columns)


for colname in constant_cols_susp: #у нас другие timedelta остаются
    if colname not in ['V14','V41','V65','V107','V305','isFraud']:
        del df_09corr[colname] # deleting the column from the dataset


df_09corr.columns # то есть убрали коррелирующие >0.9 и квазиконстантные!


list(df_09corr.columns)


warnings.filterwarnings('ignore')

### Установим красивые дефолтные настройки
### Может быть лень постоянно прописывать
### У графиков параметры цвета, размера, шрифта
### Можно положить их в словарь дефолтных настроек

import matplotlib as mlp

mlp.rcParams['lines.linewidth'] = 5
mlp.rcParams['xtick.major.size'] = 20
mlp.rcParams['xtick.major.width'] = 5
mlp.rcParams['xtick.labelsize'] = 20
mlp.rcParams['xtick.color'] = '#FF5533'

mlp.rcParams['ytick.major.size'] = 20
mlp.rcParams['ytick.major.width'] = 5
mlp.rcParams['ytick.labelsize'] = 20
mlp.rcParams['ytick.color'] = '#FF5533'

mlp.rcParams['axes.labelsize'] = 20
mlp.rcParams['axes.titlesize'] = 20
mlp.rcParams['axes.titlecolor'] = '#00B050'
mlp.rcParams['axes.labelcolor'] = '#00B050'



import matplotlib.pyplot as plt
import seaborn as sns



numeric_columns_09 = df_09corr.loc[:,df_09corr.dtypes!=object].columns 



list(numeric_columns_09[0:12])# без transactionID и isFraud


list(numeric_columns_09[2:12])# без transactionID и isFraud


for col in list(numeric_columns_09[2:len(numeric_columns_09)]): # без transactionID и isFraud    
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    
    sns.boxplot(y=col, x=df_09corr['isFraud'].astype('category'), data=df)
    
    plt.show()


# Все D признаки по-разному распределены по таргету
#V признаков много



df_09corr_bef_stand = df_09corr.copy()


df_09corr_bef_stand.columns


numeric_columns_09_bef_stand = df_09corr_bef_stand.loc[:,df_09corr_bef_stand.dtypes!=object].columns 



numeric_columns_09_bef_stand


len([xx for xx in list(numeric_columns_09_bef_stand) if xx not in ['TransactionID', 'isFraud']])


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
transform_cols = [xx for xx in list(numeric_columns_09_bef_stand) if xx not in ['TransactionID', 'isFraud']]
model_scaler=scaler.fit(df_09corr_bef_stand[transform_cols])

scaled_data=model_scaler.transform(df_09corr_bef_stand[transform_cols])


scaled_data.shape


df_09corr_aft_stand = df_09corr_bef_stand.copy()


df_09corr_aft_stand[transform_cols] = scaled_data


df_09corr_aft_stand


df_09corr_aft_stand.describe() # везде 1 стали максимумы и нули минимумы и na не видно


numeric_columns_09_aft_stand = df_09corr_aft_stand.loc[:,df_09corr_aft_stand.dtypes!=object].columns 

for col in list(numeric_columns_09_aft_stand[0:len(numeric_columns_09_aft_stand)]): # без transactionID и isFraud    
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    
    sns.boxplot(y=col, x=df_09corr_aft_stand['isFraud'].astype('category'), data=df)
    
    plt.show()


pd.set_option('display.max_rows', None)

fi_df.sort_values('feature_importance', ascending=False)# от 0 до 100


# срежем, как минимум незначимые (то есть значимые менее , чем 0.01)


fi_df.sort_values('feature_importance', ascending=False)[fi_df.sort_values('feature_importance', ascending=False)['feature_importance']<0.01]


for colname in list(fi_df.sort_values('feature_importance', ascending=False)[fi_df.sort_values('feature_importance', ascending=False)['feature_importance']<0.01]['feature_names']):
    if colname not in  ['TransactionID', 'isFraud'] and colname in list(df_09corr_aft_stand.columns):
        del df_09corr_aft_stand[colname] # deleting the column from the dataset


df_09corr_aft_stand.columns# удалили частично


for colname in list(fi_df.sort_values('feature_importance', ascending=False)[fi_df.sort_values('feature_importance', ascending=False)['feature_importance']<0.1]['feature_names']):
    if colname not in  ['TransactionID', 'isFraud'] and colname in list(df_09corr_aft_stand.columns):
        del df_09corr_aft_stand[colname] # deleting the column from the dataset


df_09corr_aft_stand.columns# удалили частично и еще удалили <0.1 колонок стало очеь мало


len(df_09corr_aft_stand.columns)


df_09corr_aft_stand_categoric_columns = df_09corr_aft_stand.loc[:,df_09corr_aft_stand.dtypes==object].columns 



df_09corr_aft_stand_categoric_columns


df_09corr_aft_stand.describe(include='object') # для вывода в большинстве колонок слишком много уникальных значений


import seaborn as sns
import matplotlib.pyplot as plt

# Пример данных

# Построение barplot для средних значений
plt.figure(figsize=(8, 6))
sns.barplot(
    x=df_09corr_aft_stand['card1'],#"day",           # Категориальный признак (ось X)
    y=df_09corr_aft_stand['isFraud'],    # Числовой признак (ось Y)
    #data=df_09corr_aft_stand,         # Данные
    ci=95,             # Уровень доверительного интервала (по умолчанию 95%)
    palette="viridis"  # Цветовая палитра
)
plt.title("Средний счет по дням недели")
plt.xlabel("card1")
plt.ylabel("isFraud")
plt.show()


df_09corr_aft_stand.to_csv('df_09corr_aft_stand.csv',index=False)


FileLink('df_09corr_aft_stand.csv') # надо возвращать prob_2 в prediction


from catboost import CatBoostClassifier, Pool
import numpy as np
params = {
'n_estimators': [100, 300],
          'max_depth': [2, 3, 5, 100],
          #'subsample': np.linspace(0.55, 0.6, 10), 
          'l2_leaf_reg': np.linspace(3, 3.5, 5), 
          'random_strength': np.linspace(1.1, 1.2, 10), 
          'eta': np.linspace(0.09, 0.1, 10), 
          'min_data_in_leaf': [5], 
          'random_state': [777],
    #'train_size':1,
}
#CatBoostError: catboost/private/libs/hyperparameter_tuning/hyperparameter_tuning.cpp:1139: All params in grid were invalid, last error message: catboost/private/libs/options/catboost_options.cpp:789: Error: default bootstrap type (bayesian) doesn't support 'subsample' option



kit = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'PRAUC:hints=skip_train~false', 
                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
                                        'logging_level': 'Verbose','task_type':"GPU",#,'train_size':'1.0',
                           'devices':'0'
                              })#,**params)#TypeError: CatBoostClassifier.__init__() got an unexpected keyword argument 'train_size'



%whos


del X_train


del df


gc.collect()


import pandas as pd # train
df_09corr_aft_stand = pd.read_csv('/kaggle/input/df-09corr-aft-stand-csv/df_09corr_aft_stand.csv')


df_09corr_aft_stand


#X_train = df_09corr_aft_stand.drop(['isFraud','TransactionID'], axis=1)
y_train = df_09corr_aft_stand['isFraud']

#cboost = CatBoostClassifier()
#X_train[['M'+str(i) for i in range(1,10)]] = X_train[['M'+str(i) for i in range(1,10)]].astype('str')
#X_train[['card'+str(i) for i in range(1,7)]] = X_train[['card'+str(i) for i in range(1,7)]].astype('str')
#X_train[['addr'+str(i) for i in range(1,3)]] = X_train[['addr'+str(i) for i in range(1,3)]].astype('str')
#X_train[[str(i) + '_emaildomain' for i in ['P','R'] ]] = X_train[[str(i) + '_emaildomain' for i in ['P','R']]].astype('str')
#X_train[['id_'+str(i) for i in range(12,39)]] = X_train[['id_'+str(i) for i in range(12,39)]].astype('str')
#cboost.fit(X_train,
#           y_train,
#           cat_features=list(categoric_columns))
#


df_09corr_aft_stand_categoric_columns = df_09corr_aft_stand.loc[:,df_09corr_aft_stand.dtypes==object].columns 

for col in list(df_09corr_aft_stand_categoric_columns):
    df_09corr_aft_stand[col] = df_09corr_aft_stand[col].astype('str')


pool_train = Pool(df_09corr_aft_stand.drop(['TransactionID'],axis=1), label=y_train, cat_features=list(df_09corr_aft_stand_categoric_columns))
#TypeError: Pool.__init__() got an unexpected keyword argument 'train_size'
#CatBoostError: catboost/libs/data/objects_grouping.cpp:332: Can't split with provided trainPart
#pool_train = Pool(df_09corr_aft_stand.drop(['TransactionID','isFraud'],axis=1), label=y_train, cat_features=list(df_09corr_aft_stand_categoric_columns))

#CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=5]=321.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.


#icat = CatBoostClassifier(**{'loss_function': 'Logloss',
#                                'logging_level': 'Silent',
#                                'auto_class_weights': 'Balanced',
#                                'eval_metric': 'AUC:hints=skip_train~false',
#                                'grow_policy': 'Lossguide',
#                                'min_data_in_leaf': 5,
#                                'random_seed': 777,
#                                'depth': 2,
#                                'iterations': 298,
#                                'subsample': 0.57399,
#                                'random_strength': 1.156123124308042,
#                                'learning_rate': 0.0999990001,
#                                'l2_leaf_reg': 3.40142341231})


#kitten.grid_search(params, X=pool_train) #очень долгий процесс



kit.grid_search(params, X=pool_train)#,train_size=1)# cant split provided train part
#CatBoostError: catboost/private/libs/hyperparameter_tuning/hyperparameter_tuning.cpp:1139: All params in grid were invalid, last error message: catboost/private/libs/options/catboost_options.cpp:789: Error: default bootstrap type (bayesian) doesn't support 'subsample' option



# надо сделать пул сразу на трейн и тест и подать его





from catboost import CatBoostClassifier, Pool
import numpy as np
params = {
'n_estimators': [100, 300],
          'max_depth': [2, 3, 5, 10],
          #'subsample': np.linspace(0.55, 0.6, 10), 
          'l2_leaf_reg': [0.1,0.5,1.0], 
          'random_strength': [0.1,0.5,1.0,2.0], 
          #'eta': [0.1,0.5], 
          #'min_data_in_leaf': [5], 
          'random_state': [777],
    'learning_rate': [0.03, 0.1]
    #'train_size':1,
}
kit = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'PRAUC:hints=skip_train~false', 
                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
                                        'logging_level': 'Silent','task_type':"GPU",#,'train_size':'1.0',
                           'devices':'0'
                              })
import pandas as pd # train
df_09corr_aft_stand = pd.read_csv('/kaggle/input/df-09corr-aft-stand-csv/df_09corr_aft_stand.csv')
#X_train = df_09corr_aft_stand.drop(['isFraud','TransactionID'], axis=1)
y_train = df_09corr_aft_stand['isFraud']
df_09corr_aft_stand_categoric_columns = df_09corr_aft_stand.loc[:,df_09corr_aft_stand.dtypes==object].columns 

for col in list(df_09corr_aft_stand_categoric_columns):
    df_09corr_aft_stand[col] = df_09corr_aft_stand[col].astype('str')

pool_train = Pool(df_09corr_aft_stand.drop(['TransactionID','isFraud'],axis=1), label=y_train, cat_features=list(df_09corr_aft_stand_categoric_columns))



pd.set_option('display.max_columns', None)
df_09corr_aft_stand.describe()


df_09corr_aft_stand[df_09corr_aft_stand_categoric_columns].describe()


y_train = df_09corr_aft_stand['isFraud']



from sklearn.feature_selection import VarianceThreshold
numeric_columns_09corr_aft_stand = df_09corr_aft_stand.loc[:,df_09corr_aft_stand.dtypes!=object].columns 
cutter = VarianceThreshold(threshold=0.1)
cutter.fit(df_09corr_aft_stand[numeric_columns_09corr_aft_stand])# посмотрим - но тут не приводили к масштабу


df_09corr_aft_stand_categoric_columns


cutter.get_feature_names_out()# да, почти все с малой вариацией меньше 0.1


y_train.nunique()


y_train.value_counts()


#pool_train = Pool(df_09corr_aft_stand.drop(['TransactionID','isFraud'],axis=1), label=y_train, cat_features=list(df_09corr_aft_stand_categoric_columns))



list(df_09corr_aft_stand_categoric_columns)


#kit = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'PRAUC:hints=skip_train~false', 
#                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
#                                        'logging_level': 'Verbose','task_type':"GPU",#,'train_size':'1.0',
#                           'devices':'0'
#                              })# debug слишком много


kit = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'PRAUC:hints=skip_train~false', 
                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
                                        'logging_level': 'Silent'#,#'task_type':"GPU",#,'train_size':'1.0',
                           #'devices':'0'
                              })


params = {
'n_estimators': [100, 300],
          'max_depth': [2, 3, 5, 10],
          #'subsample': np.linspace(0.55, 0.6, 10), 
          'l2_leaf_reg': [0.1,0.5,1.0], 
          'random_strength': [0.1,0.5,1.0,2.0], 
          #'eta': [0.1,0.5], 
          #'min_data_in_leaf': [5], 
          'random_state': [777],
    'learning_rate': [0.03, 0.1]
    #'train_size':1,
}
kit = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'AUC:hints=skip_train~false', 
                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
                                        'logging_level': 'Verbose'#,#'task_type':"GPU",#,'train_size':'1.0',
                           #'devices':'0'
                              })
kit.grid_search(params, X=pool_train, cv=1)


rez = kit.grid_search(params, X=pool_train, cv=1)#isFraud убрали и начало работать там же ответ был зашит!


151:	learn: 1.0000000	test: 1.0000000	best: 1.0000000 (0)	total: 47.5s	remaining: 46.3s












# Заново


sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')



train_transaction[['M'+str(i) for i in range(1,10)]] = train_transaction[['M'+str(i) for i in range(1,10)]].astype(object)
train_transaction[['card'+str(i) for i in range(1,7)]] = train_transaction[['card'+str(i) for i in range(1,7)]].astype(object)
train_transaction[['addr'+str(i) for i in range(1,3)]] = train_transaction[['addr'+str(i) for i in range(1,3)]].astype(object)
train_transaction[[str(i) + '_emaildomain' for i in ['P','R'] ]] = train_transaction[[str(i) + '_emaildomain' for i in ['P','R']]].astype(object)


train_identity[['id_'+str(i) for i in range(12,39)]] = train_identity[['id_'+str(i) for i in range(12,39)]].astype(object)


df = pd.merge(train_transaction,train_identity,'left','TransactionID')


df_min_max_scaled = df.copy()


categoric_columns = df_min_max_scaled.loc[:,df_min_max_scaled.dtypes==object].columns 
numeric_columns = df_min_max_scaled.loc[:,df_min_max_scaled.dtypes!=object].columns 


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
transform_cols = [xx for xx in list(numeric_columns) if xx not in ['TransactionID', 'isFraud']]
model_scaler=scaler.fit(df_min_max_scaled[transform_cols])

scaled_data=model_scaler.transform(df_min_max_scaled[transform_cols])


df_min_max_scaled[transform_cols] = scaled_data


pd.set_option('display.max_rows', None)

df_min_max_scaled.dtypes

# все нужные стали категориальными


pd.reset_option('display.max_rows')


df_min_max_scaled.dtypes


df_min_max_scaled.describe() # все численные привел к одному масштабу и действительно много где дисперсия меньше чем 0.1


# Прежде чем принимать решение посмотрим на feature importances


categoric_columns





X_train_categoric_columns = X_train.loc[:,(X_train.dtypes==object)  | (X_train.dtypes==str)].columns 



X_train = df_min_max_scaled.drop(['isFraud','TransactionID'], axis=1)
y_train = df_min_max_scaled['isFraud']
X_train[['M'+str(i) for i in range(1,10)]] = X_train[['M'+str(i) for i in range(1,10)]].astype('str')
X_train[['card'+str(i) for i in range(1,7)]] = X_train[['card'+str(i) for i in range(1,7)]].astype('str')
X_train[['addr'+str(i) for i in range(1,3)]] = X_train[['addr'+str(i) for i in range(1,3)]].astype('str')
X_train[[str(i) + '_emaildomain' for i in ['P','R'] ]] = X_train[[str(i) + '_emaildomain' for i in ['P','R']]].astype('str')
X_train[['id_'+str(i) for i in range(12,39)]] = X_train[['id_'+str(i) for i in range(12,39)]].astype('str')


X_train_categoric_columns


list(X_train.columns)


for col in categoric_columns:
    most_recent = df.groupby(col).size().sort_values().index[-1] # по возрастанию же
    
    df_test[col] = df_test[col].fillna(most_recent) # заполнение данными из train!

### Заполним средним
for col in [item for item in numeric_columns if item not in ['isFraud']]: #KeyError: 'isFraud'
    df_test[col] = df_test[col].fillna(df[col].mean())# заполнение данными из train!
    


from catboost import CatBoostClassifier # обучил, чтобы взять feature importances

X_train = df_min_max_scaled.drop(['isFraud','TransactionID'], axis=1)
y_train = df_min_max_scaled['isFraud']
X_train[['M'+str(i) for i in range(1,10)]] = X_train[['M'+str(i) for i in range(1,10)]].astype('str')
X_train[['card'+str(i) for i in range(1,7)]] = X_train[['card'+str(i) for i in range(1,7)]].astype('str')
X_train[['addr'+str(i) for i in range(1,3)]] = X_train[['addr'+str(i) for i in range(1,3)]].astype('str')
X_train[[str(i) + '_emaildomain' for i in ['P','R'] ]] = X_train[[str(i) + '_emaildomain' for i in ['P','R']]].astype('str')
X_train[['id_'+str(i) for i in range(12,39)]] = X_train[['id_'+str(i) for i in range(12,39)]].astype('str')
X_train_categoric_columns = X_train.loc[:,(X_train.dtypes==object)  | (X_train.dtypes==str)].columns # избыточно str
X_train_numeric_columns = X_train.loc[:,(X_train.dtypes!=object) ].columns 

for col in X_train_categoric_columns:
    most_recent = X_train.groupby(col).size().sort_values().index[-1] # по возрастанию же
    
    X_train[col] = X_train[col].fillna(most_recent) # заполнение данными из train!

### Заполним средним
for col in [item for item in X_train_numeric_columns if item not in ['isFraud','TransactionID']]: #KeyError: 'isFraud'
    X_train[col] = X_train[col].fillna(X_train[col].mean())# заполнение данными из train!
    
cboost = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'PRAUC:hints=skip_train~false', 
                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
                                        'logging_level': 'Verbose','task_type':"GPU",
                           'devices':'0'
                              })

cboost.fit(X_train,
           y_train,
           cat_features=list(X_train_categoric_columns))
# конвертнул заново из-за CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=5]=321.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.

#y_pred = cboost.predict(X_test)

#print(classification_report(y_test, y_pred, digits=4))

# ругалось на Nan до замены nan


cboost.save_model('mfdp_simple_cboost_all_numeric_minmaxed',
           format="cbm")



X_train.to_csv('X_train_min_maxed.csv',index=False)



y_train.to_csv('y_train_for_min_maxed.csv',index=False)



### Какие признаки оказались наиболее важны в model?
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8, 5)

import warnings
warnings.filterwarnings('ignore')
import time
fi_df = pd.DataFrame({'feature_names': X_train.columns,
                      'feature_importance': cboost.get_feature_importance()
                     }) # на cboost грузилось так

plt.figure(figsize=(10,8))
fig = sns.barplot(x='feature_importance', y='feature_names', 
            data=fi_df.sort_values('feature_importance', ascending=False))
plot_fig = fig.get_figure()
#plot_fig.savefig('GBM catboost feature importance sns.png')# пустоый вывод # так сохранилось без header
plt.title('GBM catboost feature importance');


pd.set_option('display.max_rows', None)

fi_df.sort_values('feature_importance', ascending=False)


fi_df.sort_values('feature_importance', ascending=False).to_csv('feature_importance_for_min_maxed.csv',index=False)


corr_matr = X_train[X_train_numeric_columns].corr() 
corr_matr.to_csv('corr_matr_for_min_maxed.csv',index=False)


from sklearn.feature_selection import VarianceThreshold
#numeric_columns_09corr_aft_stand = df_09corr_aft_stand.loc[:,df_09corr_aft_stand.dtypes!=object].columns 
cutter = VarianceThreshold(threshold=0.1)
cutter.fit(X_train[X_train_numeric_columns])# посмотрим -привели к масштабу
cutter.get_feature_names_out()# те которые оставить # типо все константные?


from sklearn.feature_selection import VarianceThreshold
#numeric_columns_09corr_aft_stand = df_09corr_aft_stand.loc[:,df_09corr_aft_stand.dtypes!=object].columns 
cutter = VarianceThreshold(threshold=0.05)
cutter.fit(X_train[X_train_numeric_columns])# посмотрим -привели к масштабу
cutter.get_feature_names_out()# те которые оставить # типо все константные? - да, получается


from sklearn.feature_selection import VarianceThreshold
#numeric_columns_09corr_aft_stand = df_09corr_aft_stand.loc[:,df_09corr_aft_stand.dtypes!=object].columns 
cutter = VarianceThreshold(threshold=0.01)
cutter.fit(X_train[X_train_numeric_columns])# посмотрим -привели к масштабу
cutter.get_feature_names_out()# те которые оставить # типо все константные? - да, получается


# малое изменение - опрометчиво было срезать до стандартизации а StandardScaler вообще сделает 1 дисперсию



df[numeric_columns].corr() # слишком много колонок


%time


