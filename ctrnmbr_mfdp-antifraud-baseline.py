import pandas as pd
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


### Заполним средним

for col in numeric_columns:
    df_min_max_scaled[col] = df_min_max_scaled[col].fillna(df_min_max_scaled[col].mean())


for col in categoric_columns:
    most_recent = df_min_max_scaled.groupby(col).size().sort_values().index[-1] # по возрастанию же
    df_min_max_scaled[col] = df_min_max_scaled[col].fillna(most_recent)

df_min_max_scaled.describe(include='object') # теперь везде заполнено


pd.set_option('display.max_columns', None)

df_min_max_scaled.describe()


# to reset this
pd.reset_option('display.max_columns')


from sklearn.preprocessing import StandardScaler# MinMaxScaler # его применять нет смысла - слишком много делает с малой вариацией лишь 5 атрибутов с вариацией >0.01 и катбуст ругается на константность признаков

scaler = StandardScaler()
transform_cols = [xx for xx in list(numeric_columns) if xx not in ['TransactionID', 'isFraud']]
model_scaler=scaler.fit(df_min_max_scaled[transform_cols])

scaled_data=model_scaler.transform(df_min_max_scaled[transform_cols])


df_min_max_scaled[transform_cols] = scaled_data


pd.set_option('display.max_columns', None)

df_min_max_scaled.describe() 


# to reset this
pd.reset_option('display.max_columns')


from sklearn.feature_selection import VarianceThreshold
#numeric_columns_09corr_aft_stand = df_09corr_aft_stand.loc[:,df_09corr_aft_stand.dtypes!=object].columns 
cutter = VarianceThreshold(threshold=0.1)
cutter.fit(X_train[X_train_numeric_columns])# посмотрим -привели к масштабу
cutter.get_feature_names_out()# те которые оставить # типо все константные?


cboost.save_model('mfdp_simple_cboost_all_numeric_standardscaled_base',
           format="cbm")



from catboost import CatBoostClassifier # обучил, чтобы взять feature importances

X_train = df_min_max_scaled.drop(['isFraud','TransactionID'], axis=1)
y_train = df_min_max_scaled['isFraud']

cboost = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'PRAUC:hints=skip_train~false', 
                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
                                        'logging_level': 'Verbose','task_type':"GPU",
                           'devices':'0'
                              })
X_train[['M'+str(i) for i in range(1,10)]] = X_train[['M'+str(i) for i in range(1,10)]].astype('str')
X_train[['card'+str(i) for i in range(1,7)]] = X_train[['card'+str(i) for i in range(1,7)]].astype('str')
X_train[['addr'+str(i) for i in range(1,3)]] = X_train[['addr'+str(i) for i in range(1,3)]].astype('str')
X_train[[str(i) + '_emaildomain' for i in ['P','R'] ]] = X_train[[str(i) + '_emaildomain' for i in ['P','R']]].astype('str')
X_train[['id_'+str(i) for i in range(12,39)]] = X_train[['id_'+str(i) for i in range(12,39)]].astype('str')
cboost.fit(X_train,
           y_train,
           cat_features=list(categoric_columns))
# конвертнул из-за CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=5]=321.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.



from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
y_train_pred = cboost.predict(X_train)

print(classification_report(y_train, y_train_pred, digits=8)) #- у нас нет y_test


from sklearn.metrics import precision_recall_curve, auc

y_train_scores = pd.DataFrame(cboost.predict_proba(X_train))
precision, recall, _ = precision_recall_curve(y_train, y_train_scores[1])

pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.8f}")
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(f"precision score: {precision_score(y_train, y_train_pred):.8f}") 
print(f"accuracy score: {accuracy_score(y_train, y_train_pred):.8f}") 

print(f"recall score: {recall_score(y_train, y_train_pred):.8f}") 
print(f"f1 score: {f1_score(y_train, y_train_pred):.8f}") 



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
                     }) 
plt.figure(figsize=(10,8))
sns.barplot(x='feature_importance', y='feature_names', 
            data=fi_df.sort_values('feature_importance', ascending=False))
plt.title('GBM catboost feature importance');


pd.set_option('display.max_rows', None)

fi_df.sort_values('feature_importance', ascending=False) 


# to reset this
pd.reset_option('display.max_rows')


not_imp_cols_0_1 = set()
for colname in list(fi_df.sort_values('feature_importance', ascending=False)[fi_df.sort_values('feature_importance', ascending=False)['feature_importance']<0.1]['feature_names']):
    if colname not in  ['TransactionID', 'isFraud'] and colname in list(X_train.columns):
        #del X_train[colname] # deleting the column from the dataset
        not_imp_cols_0_1.add(colname)


not_imp_cols_0_1


len(not_imp_cols_0_1)


X_train_not_imp_cols_0_1 = X_train.copy()


for colname in not_imp_cols_0_1:
    del X_train_not_imp_cols_0_1[colname] # deleting the column from the dataset



X_train_not_imp_cols_0_1.describe()


not_imp_cols_0 = set()
for colname in list(fi_df.sort_values('feature_importance', ascending=False)[fi_df.sort_values('feature_importance', ascending=False)['feature_importance']==0]['feature_names']):
    if colname not in  ['TransactionID', 'isFraud'] and colname in list(X_train.columns):
        #del X_train[colname] # deleting the column from the dataset
        not_imp_cols_0.add(colname)


len(not_imp_cols_0)


X_train_not_imp_cols_0_1_categoric_columns = X_train_not_imp_cols_0_1.loc[:,X_train_not_imp_cols_0_1.dtypes==object].columns 
X_train_not_imp_cols_0_1_numeric_columns = X_train_not_imp_cols_0_1.loc[:,X_train_not_imp_cols_0_1.dtypes!=object].columns 
corr_matr = X_train_not_imp_cols_0_1[X_train_not_imp_cols_0_1_numeric_columns].corr() 
corr_matr.to_csv('corr_matr_for_train_standardscaled_not_imp_cols_0_1.csv',index=False)


corr_matr


### функции для фильтрации признаков

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df,cor_matr, n=5):
    au_corr = corr_matr.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(X_train_not_imp_cols_0_1[X_train_not_imp_cols_0_1_numeric_columns],corr_matr, 100))


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
                    
    


corr_more09= correlation(X_train_not_imp_cols_0_1[X_train_not_imp_cols_0_1_numeric_columns],corr_matr, 0.9)


corr_more09


len(corr_more09)


len(X_train_not_imp_cols_0_1_numeric_columns)


X_train_not_imp_cols_0_1_and_corr0_9 = X_train_not_imp_cols_0_1.copy()


for colname in corr_more09:
    del X_train_not_imp_cols_0_1_and_corr0_9[colname] # deleting the column from the dataset



X_train_not_imp_cols_0_1_and_corr0_9_categoric_columns = X_train_not_imp_cols_0_1_and_corr0_9.loc[:,X_train_not_imp_cols_0_1_and_corr0_9.dtypes==object].columns 
X_train_not_imp_cols_0_1_and_corr0_9_numeric_columns = X_train_not_imp_cols_0_1_and_corr0_9.loc[:,X_train_not_imp_cols_0_1_and_corr0_9.dtypes!=object].columns 



len(X_train_not_imp_cols_0_1_and_corr0_9_numeric_columns)


for col in list(X_train_not_imp_cols_0_1_and_corr0_9_numeric_columns): # без transactionID и isFraud    
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    
    sns.boxplot(y=col, x=y_train.astype('category'), data=X_train_not_imp_cols_0_1_and_corr0_9)
    
    plt.show()


X_train_not_imp_cols_0_1_and_corr0_9.describe(include='object') # есть очень многочисленные группы - из-за этого EDA анализ можнт быть затруднен
# поскольку четкий смысл колонок не всегда известен =то подробно анализировать смысла нет


from catboost import CatBoostClassifier, Pool
import numpy as np
params = {
'n_estimators': [100, 300],
          'max_depth': [3, 7, 10],
          'l2_leaf_reg': [0.1,0.5], 
          'random_strength': [0.1,0.5,1.0], 
          'random_state': [777],
    'learning_rate': [0.03, 0.1]
}
kit = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'PRAUC:hints=skip_train~false', 
                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
                                        'logging_level': 'Silent','task_type':"GPU",
                           'devices':'0'
                              })
import pandas as pd 

X_train_not_imp_cols_0_1_and_corr0_9_categoric_columns = X_train_not_imp_cols_0_1_and_corr0_9.loc[:,X_train_not_imp_cols_0_1_and_corr0_9.dtypes==object].columns 

for col in list(X_train_not_imp_cols_0_1_and_corr0_9_categoric_columns):
    X_train_not_imp_cols_0_1_and_corr0_9[col] = X_train_not_imp_cols_0_1_and_corr0_9[col].astype('str')

pool_train = Pool(X_train_not_imp_cols_0_1_and_corr0_9, label=y_train, cat_features=list(X_train_not_imp_cols_0_1_and_corr0_9_categoric_columns))



rez = kit.grid_search(params, X=pool_train, cv=4)#isFraud убрали и начало работать там же ответ был зашит!


rez


dir(rez)


type(rez)


import pickle
with open('grid.pkl', 'wb') as f:
    pickle.dump(rez, f)
#with open('saved_dictionary.pkl', 'rb') as f:
#    loaded_dict = pickle.load(f)


from catboost import CatBoostClassifier # обучил, чтобы взять feature importances


cboost_grid = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'PRAUC:hints=skip_train~false', 
                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
                                        'logging_level': 'Verbose','task_type':"GPU",
                           'devices':'0'
                              },**rez['params'])

cboost_grid.fit(X_train_not_imp_cols_0_1_and_corr0_9,
           y_train,
           cat_features=list(X_train_not_imp_cols_0_1_and_corr0_9_categoric_columns))


cboost_grid.save_model('mfdp_cboost_grid',
           format="cbm")



from sklearn.metrics import precision_recall_curve, auc

y_train_scores = pd.DataFrame(cboost.predict_proba(X_train))
precision, recall, _ = precision_recall_curve(y_train, y_train_scores[1])

pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.8f}")
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(f"precision score: {precision_score(y_train, y_train_pred):.8f}") 
print(f"accuracy score: {accuracy_score(y_train, y_train_pred):.8f}") 

print(f"recall score: {recall_score(y_train, y_train_pred):.8f}") 
print(f"f1 score: {f1_score(y_train, y_train_pred):.8f}") 



from sklearn.metrics import precision_recall_curve, auc
y_train_pred_grid = cboost_grid.predict(X_train_not_imp_cols_0_1_and_corr0_9)

y_train_scores_grid = pd.DataFrame(cboost_grid.predict_proba(X_train_not_imp_cols_0_1_and_corr0_9))
precision_grid, recall_grid, _ = precision_recall_curve(y_train, y_train_scores_grid[1])

pr_auc_grid = auc(recall_grid, precision_grid)
print(f"PR AUC_grid: {pr_auc_grid:.8f}")
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(f"precision_grid score: {precision_score(y_train, y_train_pred_grid):.8f}") 
print(f"accuracy_grid score: {accuracy_score(y_train, y_train_pred_grid):.8f}") 

print(f"recall_grid score: {recall_score(y_train, y_train_pred_grid):.8f}") 
print(f"f1_grid score: {f1_score(y_train, y_train_pred_grid):.8f}") 



# продолжим eda - пока не заполнял средним


list(df_min_max_scaled.columns)


pd.set_option('display.max_columns', None)

df_min_max_scaled


df_min_max_scaled.describe()


df_min_max_scaled.describe(include='object')


df_min_max_scaled['ProductCD'].value_counts(dropna=False)


df_min_max_scaled['card1'].value_counts(dropna=False)


df_min_max_scaled['card3'].value_counts(dropna=False)


df_min_max_scaled['card5'].value_counts(dropna=False)


df_min_max_scaled['addr1'].value_counts(dropna=False)


df_min_max_scaled['P_emaildomain'].value_counts(dropna=False)


df_min_max_scaled['R_emaildomain'].value_counts(dropna=False)


df_min_max_scaled['M1'].value_counts(dropna=False)


df_min_max_scaled['M3'].value_counts(dropna=False)


df_min_max_scaled['M4'].value_counts(dropna=False)


df_min_max_scaled['M8'].value_counts(dropna=False)


df_min_max_scaled['id_12'].value_counts(dropna=False)


df_min_max_scaled['id_13'].value_counts(dropna=False)


df_min_max_scaled['id_14'].value_counts(dropna=False)


df_min_max_scaled['id_23'].value_counts(dropna=False)


df_min_max_scaled['id_25'].value_counts(dropna=False)


df_min_max_scaled['id_30'].value_counts(dropna=False)


df_min_max_scaled['id_31'].value_counts(dropna=False)


df_min_max_scaled['id_33'].value_counts(dropna=False)


df_min_max_scaled['id_34'].value_counts(dropna=False)


df_min_max_scaled['DeviceType'].value_counts(dropna=False)


df_min_max_scaled['DeviceInfo'].value_counts(dropna=False)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno  # Для визуализации пропусков


df['DeviceInfo'].value_counts().plot(kind='bar')
plt.show()


values = df['your_column'].value_counts(dropna=False).keys().tolist()
counts = df['your_column'].value_counts(dropna=False).tolist()
value_dict = dict(zip(values, counts))


print(df.isnull().sum())  # Счётчик пропусков
msno.matrix(df)  # Визуализация пропусков №слишком много
plt.show()


msno.matrix(df[categoric_columns])  # Визуализация пропусков №слишком много
plt.show()


sns.pairplot(df[['TransactionAmt','isFraud']], hue='isFraud')
plt.show()


sns.pairplot(df[['TransactionAmt','isFraud']], hue='TransactionAmt')
plt.show()


from ydata_profiling import ProfileReport
profile = ProfileReport(df, title='Advanced EDA', explorative=True,minimal=True)
profile.to_file('eda_report.html')
#In summary, while ProfileReport doesn't directly use GPUs, you can enhance its performance by using GPU-accelerated libraries like cuDF or Polars for data preprocessing before generating the report.
# 7487/143325 [1:31:25<7:08:27,  5.28it/s, scatter V227, C8]
#profile = ProfileReport(df, title='Advanced EDA', explorative=True) - 8 часов бы считался

# Базовые метрики
print(f"Размер данных: {df.shape}")
print(f"Дубликаты: {df.duplicated().sum()}")
print(f"Типы данных:\n{df.dtypes}")
print(f"Пропуски:\n{df.isnull().sum()}")


import gc
del profile
gc.collect()


pd.set_option('display.max_rows', None)

df.dtypes


pd.reset_option('display.max_rows')


from ydata_profiling import ProfileReport

from ydata_profiling.config import Settings  

# Generate the report with custom settings
profile = ProfileReport(df, title="Custom EDA Report",      correlations={
            "auto": {"calculate": False},
            "pearson": {"calculate": False},
            "spearman": {"calculate": False},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False},
            "cramers": {"calculate": False},
        },      interactions={
            "targets":  ['isFraud'],
        }, #config=config)
                       )
profile.to_file("custom_report.html")


from IPython.display import FileLink
FileLink('/kaggle/working/custom_report.html')


missing_stats = pd.DataFrame({
    'total_missing': df.isnull().sum(),
    'percent_missing': df.isnull().mean() * 100,
    'data_type': df.dtypes
}).sort_values('percent_missing', ascending=False)


pd.set_option('display.max_rows', None)

missing_stats


pd.reset_option('display.max_rows')


from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8, 5)

import warnings
warnings.filterwarnings('ignore')
import time
# Визуализация
plt.figure(figsize=(12, 6)) #numeric_columns
sns.boxplot(data=df[numeric_columns])#(data=df.select_dtypes(include=np.number))
plt.xticks(rotation=45)
plt.title('Распределение числовых признаков')
plt.show()

# Автоматическое обнаружение
clf = IsolationForest(contamination=0.05, random_state=42)
outliers = clf.fit_predict(df[numeric_columns])#(df.select_dtypes(include=np.number))
df['outlier_flag'] = np.where(outliers == -1, 1, 0)



**from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8, 5)

import warnings
warnings.filterwarnings('ignore')
import time
# Визуализация
for col in numeric_columns:
    plt.figure(figsize=(12, 6)) #numeric_columns
    sns.boxplot(data=df[col])#(data=df.select_dtypes(include=np.number))
    plt.xticks(rotation=45)
    plt.title('Распределение числовых признаков')
    plt.show()
#ничего полезного


for col in list(numeric_columns): # без transactionID и isFraud    
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    
    sns.boxplot(y=col, x=df['isFraud'].astype('category'), data=df)
    
    plt.show()


# с фродом без точного анализ и выбросы чистит опасно


# Анализ выбросов в контексте
outlier_analysis = df.groupby('outlier_flag').agg({
    'target': 'mean',
    'feature1': ['mean', 'count']
})








df_min_max_scaled['TransactionAmt'].value_counts(dropna=False)#сортировка по частоте


df_min_max_scaled['TransactionAmt'].value_counts(dropna=False,sort=False)#sort : bool, default True
    #Sort by frequencies when True. Sort by DataFrame column values when False.


df_min_max_scaled['TransactionAmt']


!pip install rfpimp


# загружаем необходимые библиотеки, классы и функции
import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split,
cross_val_score,
cross_validate)
#План предварительной подготовки данных
from sklearn.metrics import roc_auc_score,average_precision_score#https://stackoverflow.com/questions/67678705/using-precision-recall-auc-as-a-scoring-metric-in-cross-validation
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from rfpimp import (feature_dependence_matrix,
plot_dependence_heatmap,
plot_corr_heatmap)
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# отключаем экспоненциальное представление и увеличиваем
# максимальное количество отображаемых столбцов
pd.set_option('display.float_format', lambda x: '%.8f' % x)
pd.set_option('display.max_columns', 500)





import pandas as pd
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


#df_min_max_scaled = df.copy()


categoric_columns = df.loc[:,df.dtypes==object].columns 
numeric_columns = df.loc[:,df.dtypes!=object].columns 


df.head()


df.info(verbose=True,show_counts=True) # есть null в колонках, как object так и числовых


df.isnull().sum()


# заполнение пропусков





nunique_list = []
miss_list = []
type_list = []
for col in df.columns:
    nunique_list.append(df[col].nunique())
    miss_list.append(df[col].isnull().sum())
    type_list.append(df[col].dtypes)


feat_labels = df.columns
summary = np.array([nunique_list, miss_list, type_list])
columns = ['nunique', 'missing', 'type']
results = pd.DataFrame(summary.T,
index=feat_labels,
columns=columns)
results


pd.set_option('display.max_rows', 500)



results


df.describe()


df.describe(include='object')


pd.reset_option('display.max_rows')


summary.shape


summary.T.shape


from sklearn.base import BaseEstimator, TransformerMixin
import itertools
from category_encoders import TargetEncoder

class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 object_columns=[],
                 target_name='isFraud'):
        
        self.object_columns = object_columns
        self.target_name = target_name
        
                
    def fit(self, X,y):
        
        X_fit = X.copy()
        y_fit = y.copy()
        
        self.numeric_columns = [x for x in X_fit.columns if x not in self.object_columns]
        
        X_with_target = pd.concat((X_fit, y_fit), axis=1)
        
        ### Сгенерим колонки к которым применим One-Hot-Encoding
        self.cols_for_ohe = [col for col in self.object_columns
                             if 
                             X_with_target[col].nunique() <= 10]
        
        ### Запомним все ohe колонки и их названия!
        self.ohe_names = {col : sorted([f"{col}_{value}".replace(":", "__d__").replace(" ", "__w__").replace("-", "__m__") for value in X_with_target[col].unique()])
                          for col in self.cols_for_ohe}
        #https://stackoverflow.com/questions/60582050/lightgbmerror-do-not-support-special-json-characters-in-feature-name-the-same
        
        ### Сгенерим колонки к которым применим Mean-Target-Encoding
        self.cols_for_mte = [col for col in self.object_columns
                             if X_with_target[col].nunique() > 10]
        
        ### Посчитаем на валидации средние значения таргета
        #self.dict_of_means = {col : X_with_target.groupby(col)[self.target_name].mean()
        #                      for col in self.cols_for_mte}
        #encoder = TargetEncoder(smoothing=10, min_samples_leaf=5)

        #self.dict_of_means = {col : encoder.fit_transform(X_fit[col], y_fit)#np.array(encoder.fit_transform(X_fit[col], y_fit)).reshape(-1)
        #                      for col in self.cols_for_mte}
        self.dict_of_means = {}
        for col in self.cols_for_mte:
            
            encoder = TargetEncoder(smoothing=10, min_samples_leaf=5)
            encoder.fit(X_with_target[[col]],X_with_target[self.target_name])
            self.dict_of_means[col] = encoder.fit_target_encoding(X_with_target[[col]],X_with_target[self.target_name])[col]

        self.dict_of_means_for_real = {}
        for col in self.numeric_columns:

            self.dict_of_means_for_real[col] = X_fit[col].mean()
        return self
    
    def transform(self,X,y=None):
        
        X_ = X.copy()
            
        data_part = pd.get_dummies(X_[self.cols_for_ohe],
                                   prefix=self.cols_for_ohe)
        
        data_part_cols = data_part.columns
        
        X_ = X_.drop(self.cols_for_ohe, axis=1)
        X_ = pd.concat((X_, data_part), axis=1)
        for col in self.numeric_columns: # заполнение медианой - простейшее
                X_[col] = X_[col].fillna(self.dict_of_means_for_real[col])
    
        for col in self.cols_for_mte:
                X_[col] = X_[col].map(self.dict_of_means[col])
                
                mean_value = self.dict_of_means[col].values.mean()
                
                X_[col] = X_[col].fillna(mean_value)
                       
            
        all_ohe = list(itertools.chain(*list(self.ohe_names.values())))
        
        missing_columns = [x 
                           for x in all_ohe
                           if x not in X_.columns
                           and
                           x not in self.numeric_columns]

        extra_columns = [x
                         for x in data_part_cols
                         if x not in all_ohe]
        
        ### Новые категории необходимо убрать
        X_ = X_.drop(extra_columns, axis=1)
    
        ### Отсутствующие категории (бинарные колонки)
        ### необходимо добавить: заполним их просто нулями
        
        if len(missing_columns) != 0:

            zeros = np.zeros((X_.shape[0], len(missing_columns)))
            zeros = pd.DataFrame(zeros,
                                 columns=missing_columns,
                                 index=X_.index)

            X_ = pd.concat((X_, zeros), axis=1)
            
        return X_[sorted(X_.columns)]



# оттестил и запустилось


df.groupby('ProductCD')['isFraud'].mean() # надо в том же формате


encoder.mapping


train["ProductCD"]


train[["ProductCD",'isFraud']]


encoding_dict # не ясно почему лишние -1 и -2


encoding_dict['ProductCD'] # откуда лишние? -1 и -2


df['ProductCD'].value_counts()


df['card1'].mean()


type(encoding_dict['ProductCD'])


encoding_dict.items()


transformer = CustomFunctionTransformer(object_columns = categoric_columns)


from sklearn.model_selection import TimeSeriesSplit

df = df.sort_values("TransactionDT")  # Sort by time (critical!) [[3]](https://medium.com/@tomer.kaftime/split-time-series-dataset-2a7d41d0756f) 

# Define TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)  # Split into 5 folds

# Iterate over splits
for train_index, test_index in tscv.split(df):
    train = df.iloc[train_index]
    test = df.iloc[test_index]
    print(f"Train period: {train['TransactionDT'].min()} - {train['TransactionDT'].max()}")
    print(f"Test period: {test['TransactionDT'].min()} - {test['TransactionDT'].max()}")


transformer.fit(train.drop(['TransactionID','isFraud'],axis=1),train['isFraud']) # у меня использует наличие в трансформере 0 не очень - но ок


#encoding_dict = encoder.fit_target_encoding(train[["ProductCD",'isFraud']],train["isFraud"])
#AttributeError: 'NoneType' object has no attribute 'category_mapping'



cols_for_mte


!pip install category_encoders


!pip install category_encoders==2.0.0


encoding_dict


cols_for_mte


1


train_index


Why sorting matters : TimeSeriesSplit splits data in chronological order without shuffling, ensuring no future data leaks into the training set 
(https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split ).
No direct time column parameter : You must manually sort the dataframe by the time column before splitting 



print(f"Train period: {train['TransactionDT'].min()} - {train['TransactionDT'].max()}")
print(f"Test period: {test['TransactionDT'].min()} - {test['TransactionDT'].max()}")
# самый большой промежуток


train


train.info()


check_train = transformer.transform(train)


check_train.info()


list(check_train.columns)


check_train.info(verbose=True,show_counts=True)


# все заполнилось


check_train = check_train.drop(['TransactionID'],axis=1)


corr_check_train = check_train.corr()
#corr.head()


corr_check_train


# что-от пропуски таргетом не очень заполнились - забыл числовые заполнить!
# нужно было делать fit сначала - в этом была пробьлема # card2 заполнился!


for col in corr_check_train.columns:
    print(col, corr_check_train[col].sum(axis=0))


{col: corr_check_train[col].sum(axis=0) for col in corr_check_train.columns}


corr_sum = pd.DataFrame(data = {col: corr_check_train[col].sum(axis=0) for col in corr_check_train.columns},index=[0])


corr_sum


corr_sum.T


corr_sum.T[0].max(),corr_sum.T[0].min()


corr_sum.T.loc[abs(corr_sum.T[0])<10,:]


corr_select_cols_10  = list(corr_sum.T.loc[abs(corr_sum.T[0])<10,:].index)


corr_select_cols_10





corr_sum.T.loc[abs(corr_sum.T[0])>=0,:]


categorical_features_ind = np.where(check_train.dtypes != float)[0]
categorical_features_ind





# формируем обучающий пул
train_pool = Pool(
check_train,
train['isFraud'],
cat_features=categorical_features_ind)
# создаем экземпляр класса CatBoostClassifier
clf = CatBoostClassifier(learning_rate=0.08,
iterations=1200,
random_strength=0.15,
random_seed=0,
model_size_reg=0.1,
logging_level='Silent',task_type='GPU',devices='0')
# обучаем модель
clf.fit(train_pool)
# вычисляем важности по SHAP
shap_values = clf.get_feature_importance(train_pool, 'ShapValues')
shap_values = shap_values[:, :-1]
# выводим график 100 наиболее важных признаков по SHAP
shap.summary_plot(shap_values, X_train, plot_type='bar', max_display=100)


import shap
shap.summary_plot(shap_values, check_train, plot_type='bar', max_display=100)


shap_feat = list(check_train.columns[np.argsort(
np.abs(shap_values).mean(0))[::-1]])
top_shap_feat = shap_feat[:100]


top_shap_feat


shap_select_cols = ['isFraud',
 'D3',
 'V94',
 'V79',
 'V93',
 'D2',
 'V81',
 'V92',
 'V280',
 'card4_discover',
 'V308',
 'D10',
 'V333',
 'V80',
 'V323',
 'V292',
 'V315',
 'V207',
 'V219',
 'V322',
 'V259',
 'V281',
 'V204',
 'V95',
 'V130',
 'V203',
 'V205',
 'V336',
 'V312',
 'id_33',
 'V313',
 'id_16_Found',
 'V84',
 'ProductCD_W',
 'V289',
 'V143',
 'V310',
 'V217',
 'V295',
 'id_19',
 'V332',
 'V288',
 'V55',
 'D6',
 'C4',
 'V218',
 'V235',
 'V141',
 'id_25',
 'C13',
 'V59',
 'id_35_T',
 'D1',
 'V263',
 'id_02',
 'V144',
 'id_37_F',
 'V126',
 'D9',
 'V264',
 'V64',
 'V291',
 'V133',
 'id_12_NotFound',
 'id_28_New',
 'V331']


# объединяем два созданных списка в один
select_cols = set(corr_select_cols_10 + shap_select_cols)


select_cols


len(select_cols)


# убирает много типов колонок - плохо


from sklearn.model_selection import TimeSeriesSplit

df = df.sort_values("TransactionDT")  # Sort by time (critical!) [[3]](https://medium.com/@tomer.kaftime/split-time-series-dataset-2a7d41d0756f) 

# Define TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)  # Split into 5 folds

# Iterate over splits
for train_index, test_index in tscv.split(df):
    train = df.iloc[train_index]
    test = df.iloc[test_index]
    print(f"Train period: {train['TransactionDT'].min()} - {train['TransactionDT'].max()}")
    print(f"Test period: {test['TransactionDT'].min()} - {test['TransactionDT'].max()}")


list(df.columns)


# создаем экземляр класса LGBMClassifier
from sklearn.pipeline import Pipeline

lgbm_subpipeline = Pipeline(steps=[ ('scaler', CustomFunctionTransformer(object_columns = categoric_columns)),  #('scaler', MinMaxScaler()), # better to not rescale internally
                                  ('lgbm_model',    LGBMClassifier(random_state=42,
n_estimators=300,       **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0})),
])

# задаем сетку гиперпараметров
param_grid = {
'lgbm_model__learning_rate': [0.01, 0.05, 0.1]
}
# создаем экземпляр класса GridSearchCV, передав
# конвейер, сетку гиперпараметров и указав
# количество блоков перекрестной проверки
gs = GridSearchCV(lgbm_subpipeline,
param_grid,
scoring='recall',
cv=tscv)
# выполняем поиск по всем значениям сетки
gs.fit(df.drop(['isFraud','TransactionID'],axis=1), df['isFraud']);
# смотрим наилучшие значения гиперпараметров
print("Наилучшие значения гиперпараметров: {}".format(
gs.best_params_))
# смотрим наилучшее значение AUC
print("Наилучшее значение RECALL: {:.3f}".format(
gs.best_score_)) # видимо [LightGBM] [Fatal] Do not support special JSON characters in feature name.
# из-за двоеточий при dummy encoding - уберем


from sklearn.metrics import make_scorer, fbeta_score, recall_score, precision_score
from sklearn.metrics import average_precision_score, precision_recall_curve
def recall_at_5(y_true, y_pred_proba):
    n_investigate = int(len(y_true) * 0.05)
    threshold = np.sort(y_pred_proba)[-n_investigate]
    y_pred = (y_pred_proba >= threshold).astype(int)
    return recall_score(y_true, y_pred)

# Создание кастомных скореров
scorers = {
    'f2': make_scorer(fbeta_score, beta=2, average='binary'),
    'pr_auc': make_scorer(average_precision_score),#, needs_proba=True),#ValueError: Classification metrics can't handle a mix of binary and continuous targets

    'recall@5': make_scorer(recall_at_5, needs_proba=True),#,ValueError: Classification metrics can't handle a mix of binary and continuous targets

    'precision':make_scorer(precision_score),#, needs_proba=True),#ValueError: Classification metrics can't handle a mix of binary and continuous targets

}


# создаем экземляр класса LGBMClassifier
from sklearn.pipeline import Pipeline

lgbm_subpipeline_2 = Pipeline(steps=[ ('scaler', CustomFunctionTransformer(object_columns = categoric_columns)),  #('scaler', MinMaxScaler()), # better to not rescale internally
                                  ('lgbm_model',    LGBMClassifier(random_state=42,
n_estimators=300,       **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0})),
])


param_grid2 = {
'lgbm_model__learning_rate': [0.1],#уже нашли в пред раз
'lgbm_model__lambda_l1': [0, 10],
'lgbm_model__bagging_fraction': [0.5, 1],
'lgbm_model__feature_fraction': [0.5, 1]
}
# создаем экземпляр класса GridSearchCV, передав
# конвейер, сетку гиперпараметров и указав
# количество блоков перекрестной проверки
gs2 = GridSearchCV(lgbm_subpipeline_2,
param_grid2,
scoring=scorers,
refit='precision',
cv=tscv)
# выполняем поиск по всем значениям сетки
gs2.fit(df.drop(['isFraud','TransactionID'],axis=1), df['isFraud']);
# смотрим наилучшие значения гиперпараметров
print("Наилучшие значения гиперпараметров: {}".format(
gs2.best_params_))
# смотрим наилучшее значение AUC
print("Наилучшее значение по scorers: {:.3f}".format(
gs2.best_score_)) # видимо [LightGBM] [Fatal] Do not support special JSON characters in feature name.
# из-за двоеточий при dummy encoding - уберем


# создаем экземляр класса LGBMClassifier
from sklearn.pipeline import Pipeline

lgbm_subpipeline_3 = Pipeline(steps=[ ('scaler', CustomFunctionTransformer(object_columns = categoric_columns)),  #('scaler', MinMaxScaler()), # better to not rescale internally
                                  ('lgbm_model',    LGBMClassifier(
random_state=42, 
n_estimators=300, 
importance_type='gain',      **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0},**{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1}))
])

output = cross_validate(
lgbm_subpipeline_3, df.drop(['isFraud','TransactionID'],axis=1), df['isFraud'],
scoring=scorers,
#refit='precision',#TypeError: cross_validate() got an unexpected keyword argument 'refit'
cv=tscv,
return_estimator=True)




estimator._final_estimator.feature_importances_


len(estimator._final_estimator.feature_importances_)


dir(estimator)


dir(estimator._final_estimator)


type(estimator._final_estimator)


estimator._final_estimator.best_score_


estimator._final_estimator.get_params


fi = []
for estimator in output['estimator']:
    fi.append(estimator._final_estimator.feature_importances_)#estimator.feature_importances_)
#fi = pd.DataFrame(
#np.array(fi).T,
#columns=['importance ' + str(idx)
#for idx in range(len(fi))],
#index=X_train.columns)


fi


for i in fi:
    print(i.shape)


fi = pd.DataFrame(
np.array(fi).T,
columns=['importance ' + str(idx)
    for idx in range(len(fi))],
index=df.drop(['isFraud','TransactionID'],axis=1).columns)


fi = pd.DataFrame(
np.vstack(fi).T,
columns=['importance ' + str(idx)
    for idx in range(len(fi))],
index=CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud']).columns)



import numpy as np
import pandas as pd

# Step 1: Get global feature names (replace with actual feature names)
global_features = list(CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud']))

# Step 2: Create a DataFrame to store results
fi_df = pd.DataFrame(columns=[f'fold_{i}' for i in range(len(fi))], index=global_features)

# Step 3: Fill DataFrame with importances, defaulting to 0 for missing features
for fold_idx, importances in enumerate(fi):
    # Map importances to global features (adjust logic based on your feature tracking)
    for feat_idx, value in enumerate(importances):
        fi_df.iloc[feat_idx, fold_idx] = value  # Replace with actual mapping logic

# Fill NaNs with 0 (missing features in folds)
fi_df.fillna(0, inplace=True)


# Get global features (replace with actual feature names)
global_features = list(CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud']))

# Number of folds
n_folds = len(fi)

# Initialize DataFrame with zeros
fi_df = pd.DataFrame(
    0, 
    index=global_features, 
    columns=[f'fold_{i}' for i in range(n_folds)]
)


for fold_idx, importances in enumerate(fi):
    for feat_idx, value in enumerate(importances):
        if feat_idx < len(global_features):  # Prevent index overflow
            fi_df.iloc[feat_idx, fold_idx] = value
        else:
            print(f"Warning: Fold {fold_idx} has extra feature at index {feat_idx}")


fi_df


estimator._final_estimator.feature_names_in_


len(estimator._final_estimator.feature_names_in_)


estimator._final_estimator.feature_name_


len(estimator._final_estimator.feature_name_)


estimator._final_estimator.


# Assume feature_names are known for each fold (e.g., from model.feature_name_)
fi_dict = {
    f'fold_{i}': pd.Series(fi[i], index=estimator._final_estimator.feature_name_[i]) 
    for i in range(len(fi))
}

# Combine into a DataFrame
fi_df = pd.DataFrame.from_dict(fi_dict, orient='index').T.fillna(0)


print(estimator.named_steps.keys())  # Lists available step names


feature_names_list = []

for estimator in output['estimator']:
    # Get feature importances
    #fi.append(estimator.named_steps['lgbm_model'].feature_importances_)
    
    # Get feature names from the preprocessor (if using ColumnTransformer)
    preprocessor = estimator.named_steps['scaler']
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categoric_columns)
    feature_names = np.concatenate([numeric_columns, cat_features])
    feature_names_list.append(feature_names)

# Now `feature_names_list` contains feature names for each fold


# не было get_featue_names_out - определим
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import itertools
from category_encoders import TargetEncoder
import numpy as np

class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, object_columns=[], target_name='isFraud'):
        self.object_columns = object_columns
        self.target_name = target_name

    def fit(self, X, y):
        X_fit = X.copy()
        y_fit = y.copy()
        
        # Identify numeric columns
        self.numeric_columns = [x for x in X_fit.columns if x not in self.object_columns]
        
        # Prepare for One-Hot Encoding (OHE) and Mean-Target Encoding (MTE)
        X_with_target = pd.concat((X_fit, y_fit), axis=1)
        self.cols_for_ohe = [col for col in self.object_columns if X_with_target[col].nunique() <= 10]
        self.cols_for_mte = [col for col in self.object_columns if X_with_target[col].nunique() > 10]
        
        # Generate OHE feature names with sanitized values
        self.ohe_names = {
            col: sorted([f"{col}_{value}".replace(":", "__d__").replace(" ", "__w__").replace("-", "__m__") 
                         for value in X_with_target[col].unique()])
            for col in self.cols_for_ohe
        }
        
        # Fit Target Encoders for MTE columns
        self.dict_of_means = {}
        for col in self.cols_for_mte:
            encoder = TargetEncoder(smoothing=10, min_samples_leaf=5)
            encoder.fit(X_with_target[[col]], X_with_target[self.target_name])
            self.dict_of_means[col] = encoder.fit_target_encoding(X_with_target[[col]], X_with_target[self.target_name])[col]
        
        # Store numeric column means for imputation
        self.dict_of_means_for_real = {col: X_fit[col].mean() for col in self.numeric_columns}
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        
        # Apply One-Hot Encoding
        data_part = pd.get_dummies(X_[self.cols_for_ohe], prefix=self.cols_for_ohe)
        data_part_cols = data_part.columns
        X_ = X_.drop(self.cols_for_ohe, axis=1)
        X_ = pd.concat((X_, data_part), axis=1)
        
        # Impute numeric columns
        for col in self.numeric_columns:
            X_[col] = X_[col].fillna(self.dict_of_means_for_real[col])
        
        # Apply Mean-Target Encoding
        for col in self.cols_for_mte:
            X_[col] = X_[col].map(self.dict_of_means[col])
            mean_value = self.dict_of_means[col].values.mean()
            X_[col] = X_[col].fillna(mean_value)
        
        # Handle missing and extra OHE columns
        all_ohe = list(itertools.chain(*list(self.ohe_names.values())))
        #missing_columns = [x for x in all_ohe if x not in X_.columns]
        #extra_columns = [x for x in data_part_cols if x not in all_ohe]
        missing_columns = [x 
                           for x in all_ohe
                           if x not in X_.columns
                           and
                           x not in self.numeric_columns]

        extra_columns = [x
                         for x in data_part_cols
                         if x not in all_ohe]
        
        X_ = X_.drop(extra_columns, axis=1)
        
        if len(missing_columns) != 0:
            zeros = pd.DataFrame(np.zeros((X_.shape[0], len(missing_columns))), 
                                  columns=missing_columns, index=X_.index)
            X_ = pd.concat((X_, zeros), axis=1)
        
        return X_[sorted(X_.columns)]

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names after transformation.
        """
        if not hasattr(self, 'cols_for_ohe'):
            raise NotFittedError("This CustomFunctionTransformer instance is not fitted yet.")
        
        # Collect all OHE-generated columns
        ohe_cols = list(itertools.chain.from_iterable(self.ohe_names.values()))
        
        # Combine with MTE and numeric columns
        all_features = ohe_cols + self.cols_for_mte + self.numeric_columns
        
        # Ensure sorted order matches transform output
        return np.array(sorted(all_features), dtype=object) #  проверил что отличий нет - теперь - такое


# создаем экземляр класса LGBMClassifier
from sklearn.pipeline import Pipeline

lgbm_subpipeline_3 = Pipeline(steps=[ ('scaler', CustomFunctionTransformer(object_columns = categoric_columns)),  #('scaler', MinMaxScaler()), # better to not rescale internally
                                  ('lgbm_model',    LGBMClassifier(
random_state=42, 
n_estimators=300, 
importance_type='gain',      **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0},**{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1}))
])

output = cross_validate(
lgbm_subpipeline_3, df.drop(['isFraud','TransactionID'],axis=1), df['isFraud'],
scoring=scorers,
#refit='precision',#TypeError: cross_validate() got an unexpected keyword argument 'refit'
cv=tscv,
return_estimator=True)










# из-за раззных - были Feature_importances разнго размера

# Get global features (replace with actual feature names)
global_features = list(CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud']))
fi = []
for estimator in output['estimator']:
    fi.append(estimator._final_estimator.feature_importances_)#estimator.feature_importances_)

# Number of folds
n_folds = len(fi)

# Initialize DataFrame with zeros
fi_df = pd.DataFrame(
    0.0, # вместо 0 для /tmp/ipykernel_271/1743372108.py:19: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '4394.892871225253' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
    #fi_df.iloc[feat_idx, fold_idx] = value
    index=global_features, 
    columns=[f'fold_{i}' for i in range(n_folds)]
)
for fold_idx, importances in enumerate(fi):
    for feat_idx, value in enumerate(importances):
        if feat_idx < len(global_features):  # Prevent index overflow
            fi_df.iloc[feat_idx, fold_idx] = float(value)  # Cast to float #value Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '4394.892871225253' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
    #fi_df.iloc[feat_idx, fold_idx] = value
        else:
            print(f"Warning: Fold {fold_idx} has extra feature at index {feat_idx}")
list_of_all_features = []
for estimator in output['estimator']:
    preprocessor = estimator.named_steps['scaler']
    #cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categoric_columns)
    feature_names = estimator.named_steps["scaler"].get_feature_names_out()
    #feature_names = np.concatenate([numeric_columns, cat_features])
    print(feature_names)  # Consistent feature names across folds
    list_of_all_features.append(feature_names)
    
# Assume feature_names are known for each fold (e.g., from model.feature_name_)

fi_dict = {
    f'fold_{i}': pd.Series(fi[i], index=list_of_all_features[i]) 
    for i in range(len(fi))
}

# Combine into a DataFrame
fi_df = pd.DataFrame.from_dict(fi_dict, orient='index').T.fillna(0)



fi_df # вот теперь готово! # 


fi_df['mean_importance'] = fi_df.mean(axis=1)


fi_df


# записываем серию, в которой индексные метки – признаки,
# значения – важности
features_imp_mean = fi_df['mean_importance']
# сортируем индексные метки по возрастанию важностей
features_imp_mean_sort = features_imp_mean.sort_values(ascending=True)
features_imp_mean_sort


# выводим график усредненных важностей
features_imp_mean_sort.plot.barh(figsize=(10, 20));


features_li = list(features_imp_mean_sort.index)
features_li


dir(output)


output


# признаками)
score_f2 = output['test_f2'].mean()
score_pr_auc = output['test_pr_auc'].mean()
score_recall5 = output['test_recall@5'].mean()
score_precision = output['test_precision'].mean()

score_all = (score_f2 + score_pr_auc + score_recall5 + score_precision)/4


score_all


score_precision# на нем же определяли в итоге


pd.set_option('display.max_rows', None)

features_imp_mean_sort


# с 500 можно брать - это
D11                                511.93867888
V283                               515.68927926
id_30                              529.14866064
V65                                536.70169606
V76                                561.51589385
V82                                568.99207677
V12                                569.78568909
V48                                582.29309685
V67                                597.63451623
V147                               629.74765940
V323                               645.67148480
V243                               648.20692196
ProductCD_H                        667.97525773
V62                                717.41962977
C9                                 751.32545398
V187                               770.96384045
V53                                772.64904186
V91                                791.12008340
C12                                792.27850345
V315                               850.98721855
V310                               860.05869156
M4_M0                              869.64376154
V313                               911.46011591
V128                               925.96498693
card5                              929.97899888
V133                               943.74688628
D3                                 947.15354517
id_31                              951.13004234
V314                               978.22812222
id_02                             1012.11860703
D8                                1019.50887254
V201                              1037.74762778
dist1                             1054.22646420
M6_F                              1112.69165810
V83                               1115.09965338
C8                                1145.18215464
C5                                1189.30168563
V87                               1282.92835038
D4                                1312.78531916
V156                              1325.54372277
V102                              1402.23531773
C2                                1424.21393255
D10                               1468.88311166
id_33                             1530.07603393
V312                              1545.91847250
V70                               1612.41049128
V307                              1646.32427969
V45                               1655.92725029
V294                              1701.37805104
V13                               1742.07519479
D15                               1945.59583370
id_20                             2030.71983713
M5_T                              2073.52744013
V308                              2086.74657049
C6                                2094.46416261
V149                              2586.31696904
D2                                2633.55859303
D1                                2650.69285303
C11                               2878.10129326
V189                              3080.54220099
id_19                             3088.80835590
C4                                3194.56995285
P_emaildomain                     3453.83887816
card2                             3492.17451020
addr1                             4104.86171863
R_emaildomain                     4229.18452988
TransactionAmt                    4693.78481124
V317                              4866.63541373
TransactionDT                     5655.39590436
C13                               8068.09142257
C14                              11367.57714393
C1                               12969.78365183
DeviceInfo                       13554.44435907
V258                             26443.14721426
card1                            88932.94429641
Name: mean_importance, dtype: float64


# создаем экземляр класса LGBMClassifier
from sklearn.pipeline import Pipeline

lgbm_subpipeline_3 = Pipeline(steps=[ ('scaler', CustomFunctionTransformer(object_columns = categoric_columns)),  #('scaler', MinMaxScaler()), # better to not rescale internally
                                  ('lgbm_model',    LGBMClassifier(
random_state=42, 
n_estimators=300, 
importance_type='gain',      **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0},**{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1}))
])

output = cross_validate(
lgbm_subpipeline_3, df.drop(['isFraud','TransactionID'],axis=1), df['isFraud'],
scoring=scorers,
#refit='precision',#TypeError: cross_validate() got an unexpected keyword argument 'refit'
cv=tscv,
return_estimator=True)




# задаем пороговое значение разницы AUC
tol = 0.0001
print("выполнение рекурсивного удаления признаков")
# создаем список, в который будем
# записывать удаляемые признаки
features_to_remove = []
# создаем список, в который будем
# записывать значение AUC
score_mean_list = []
# создаем список, в который будем
# записывать разницу AUC
diff_score_list = []
# задаем счетчик для оценки прогресса
count = 1
# итерируем по всем признакам, признаки упорядочены по
# возрастанию важности на основе информационного выигрыша
# создаем экземляр класса LGBMClassifier
from sklearn.pipeline import Pipeline

lgbm_subpipeline_4 = Pipeline(steps=[ ('scaler', CustomFunctionTransformer(object_columns = categoric_columns)),  #('scaler', MinMaxScaler()), # better to not rescale internally
                                  ('lgbm_model',    LGBMClassifier(
random_state=42, 
n_estimators=300, 
importance_type='gain',      **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0},**{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1}))
])
tscv2 = TimeSeriesSplit(n_splits=2)  # Split into 5 folds

for feature in features_li:
    print()
    print("проверяемый признак: ", feature, " признак ", count,
    " из ", len(features_li))
    count = count + 1
    # создаем экземляр класса LGBMClassifier
#    model = Pipeline(steps=[ ('scaler', CustomFunctionTransformer(object_columns = categoric_columns)),  #('scaler', MinMaxScaler()), # better to not rescale internally
#                                  ('lgbm_model',    LGBMClassifier(
#random_state=42, 
#n_estimators=300, 
#importance_type='gain',      **{"device": "gpu",
#        "gpu_platform_id": 0,
#        "gpu_device_id": 0},**{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1}))
#])
    model = LGBMClassifier(
random_state=42, 
n_estimators=300, 
importance_type='gain',      **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0},**{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1})
    #LGBMClassifier(
    #random_state=42, 
    #n_estimators=300, 
    #importance_type='gain',      **{"device": "gpu",
    #        "gpu_platform_id": 0,
    #        "gpu_device_id": 0},**{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1}))
    
    # обучаем модели со всеми признаками минус уже удаленные признаки
    # (берем их из списка удаляемых признаков) и оцениваемый признак
    
    #scores = cross_val_score(
    #model,
    ##df.drop(['isFraud','TransactionID'],axis=1).drop(features_to_remove + [feature], axis=1), #KeyError: "['id_22_20.0'] not found in axis" динамически же
    #CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud']),
    #df['isFraud'],
    #scoring=scorers,
    #cv=tscv)# ValueError: For evaluating multiple scores, use sklearn.model_selection.cross_validate instead. {'f2': make_scorer(fbeta_score, beta=2, average=binary), 'pr_auc': make_scorer(average_precision_score), 'recall@5': make_scorer(recall_at_5, needs_proba=True), 'precision': make_scorer(precision_score)} was passed.
    scores = cross_validate(
model, CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud']),df['isFraud'],
scoring=scorers,
#refit='precision',#TypeError: cross_validate() got an unexpected keyword argument 'refit'
cv=tscv2,)# чтобы быстрее было
#return_estimator=True) # NameError: name 'scores' is not defined
    print(f"Scores ={scores} for {feature}")

    # вычисляем, усредненный по проверочным блокам
    # перекрестной проверки
    score_f2_ti = scores['test_f2'].mean()
    score_pr_auc_ti = scores['test_pr_auc'].mean()
    score_recall5_ti = scores['test_recall@5'].mean()
    score_precision_ti = scores['test_precision'].mean()

    score_mean = (score_f2_ti + score_pr_auc_ti + score_recall5_ti + score_precision_ti)/4
    #score_mean = scores.mean()
    # печатаем усредненное значение AUC
    print("AUC модели после удаления={}".format(score_mean))
    # добавляем усредненное значение AUC в список
    score_mean_list.append(score_mean)
    # печатаем AUC модели со всеми признаками
    # (опорное значение AUC)
    print("AUC модели со всеми признаками={}".format(score_all))
    # определяем разницу AUC (если отрицательное значение
    # – удаление признака улучшило AUC)
    diff_score = score_all - score_mean
    # записываем разницу AUC в список
    diff_score_list.append(diff_score)
    # сравниваем разницу AUC с порогом, заданным заранее
    # если разница AUC больше или равна порогу, сохраняем
    if diff_score >= tol:
        print("Разница AUC={}".format(diff_score))
        print("сохраняем: ", feature)
        print()
    # если разница AUC меньше порога, удаляем
    else:
        print("Разница AUC={}".format(diff_score))
        print("удаляем: ", feature)
        print()
        # если разница AUC меньше порога и мы удаляем признак,
        # мы в качестве нового опорного значения AUC задаем
        # значение AUC для модели с оставшимися признаками
        score_all = score_mean
        # добавляем удаляемый признак в список
        features_to_remove.append(feature)

# формируем датафрейм
df_rez = pd.DataFrame({'feature': features_li,
'auc_score_mean': score_mean_list,
'diff_auc_score': diff_score_list})
# цикл завершен, вычисляем количество
# удаленных признаков
print("ВЫПОЛНЕНО!!")
print("общее количество признаков для удаления: ",
len(features_to_remove))
# определяем признаки, которые мы хотим сохранить (не удаляем)
features_to_keep = [x for x in features_li
if x not in features_to_remove]
print("общее количество признаков для сохранения: ",
len(features_to_keep))








#090625


# создаем экземляр класса LGBMClassifier
from sklearn.pipeline import Pipeline

lgbm_subpipeline_3 = Pipeline(steps=[ ('scaler', CustomFunctionTransformer(object_columns = categoric_columns)),  #('scaler', MinMaxScaler()), # better to not rescale internally
                                  ('lgbm_model',    LGBMClassifier(
random_state=42, 
n_estimators=300, 
importance_type='gain',      **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0},**{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1}))
])

output = cross_validate(
lgbm_subpipeline_3, df.drop(['isFraud','TransactionID'],axis=1), df['isFraud'],
scoring=scorers,
#refit='precision',#TypeError: cross_validate() got an unexpected keyword argument 'refit'
cv=tscv,
return_estimator=True)




# из-за раззных - были Feature_importances разнго размера

# Get global features (replace with actual feature names)
global_features = list(CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud']))
fi = []
for estimator in output['estimator']:
    fi.append(estimator._final_estimator.feature_importances_)#estimator.feature_importances_)

# Number of folds
n_folds = len(fi)

# Initialize DataFrame with zeros
fi_df = pd.DataFrame(
    0.0, # вместо 0 для /tmp/ipykernel_271/1743372108.py:19: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '4394.892871225253' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
    #fi_df.iloc[feat_idx, fold_idx] = value
    index=global_features, 
    columns=[f'fold_{i}' for i in range(n_folds)]
)
for fold_idx, importances in enumerate(fi):
    for feat_idx, value in enumerate(importances):
        if feat_idx < len(global_features):  # Prevent index overflow
            fi_df.iloc[feat_idx, fold_idx] = float(value)  # Cast to float #value Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '4394.892871225253' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
    #fi_df.iloc[feat_idx, fold_idx] = value
        else:
            print(f"Warning: Fold {fold_idx} has extra feature at index {feat_idx}")
list_of_all_features = []
for estimator in output['estimator']:
    preprocessor = estimator.named_steps['scaler']
    #cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categoric_columns)
    feature_names = estimator.named_steps["scaler"].get_feature_names_out()
    #feature_names = np.concatenate([numeric_columns, cat_features])
    print(feature_names)  # Consistent feature names across folds
    list_of_all_features.append(feature_names)
    
# Assume feature_names are known for each fold (e.g., from model.feature_name_)

fi_dict = {
    f'fold_{i}': pd.Series(fi[i], index=list_of_all_features[i]) 
    for i in range(len(fi))
}

# Combine into a DataFrame
fi_df = pd.DataFrame.from_dict(fi_dict, orient='index').T.fillna(0)



fi_df['mean_importance'] = fi_df.mean(axis=1)


# записываем серию, в которой индексные метки – признаки,
# значения – важности
features_imp_mean = fi_df['mean_importance']
# сортируем индексные метки по возрастанию важностей
features_imp_mean_sort = features_imp_mean.sort_values(ascending=True)
features_imp_mean_sort


features_li = list(features_imp_mean_sort.index)
features_li
# признаками)
score_f2 = output['test_f2'].mean()
score_pr_auc = output['test_pr_auc'].mean()
score_recall5 = output['test_recall@5'].mean()
score_precision = output['test_precision'].mean()

score_all = (score_f2 + score_pr_auc + score_recall5 + score_precision)/4


score_precision


score_all


# сессия отключилась - промежуточный результа последовательного удаления по изменению метрики
remove_l_intermediate = ['id_22_20.0',
'id_24_12.0',
'id_24_11.0',
'id_23_nan',
'id_23_IP_PROXY__d__TRANSPARENT',
'id_23_IP_PROXY__d__HIDDEN',
'id_23_IP_PROXY__d__ANONYMOUS',
'id_22_nan',
'id_22_41.0',
'id_22_35.0',
'id_22_33.0',
'id_22_31.0',
'id_22_21.0',
'id_24_15.0',
'id_22_19.0',
'id_22_14.0',
'id_22_12.0',
'ProductCD_W',
'M9_nan',
'M8_nan',
'M7_nan',
'id_16_nan',
'id_16_NotFound',
'id_15_nan',
'id_35_nan',
'id_29_NotFound',
'id_35_T',
'M5_nan',
'id_32_16.0',
'V1',
'id_27_nan',
'id_27_NotFound',
'id_37_nan',
'M6_nan',
'id_36_nan',
'id_32_0.0',
'id_29_nan',
'id_12_nan',
'id_28_nan',
'M1_nan',
'M1_T',
'DeviceType_nan',
'V325',
'V32',
'id_24_nan',
'id_24_25.0',
'id_24_21.0',
'id_24_18.0',
'id_24_16.0',
'M2_nan',
'V120',
'id_12_NotFound',
'V118',
'V117',
'V114',
'V113',
'V111',
'V107',
'id_38_nan',
'M4_nan',
'M3_nan',
'V119',
'M1_F',
'id_24_26.0',
'id_24_23.0',
'id_24_19.0',
'id_35_F',
'id_34_nan',
'id_34_match_status__d____m__1',
'id_34_match_status__d__2',
'id_34_match_status__d__1',
'id_34_match_status__d__0',
'id_32_nan',
'V122',
'card6_nan',
'card6_debit__w__or__w__credit',
'card6_charge__w__card',
'V14',
'card4_nan',
'card4_american__w__express',
'V98',
'V21',
'V196',
'V195',
'V241',
'V330',
'V328',
'V27',
'V31',
'V305',
'V28',
'V121',
'V103',
'V240',
'V299',
'V41',
'id_12_Found',
'id_28_New',
'V116',
'V334',
'V146',
'V193',
'V108',
'V142',
'V324',
'V191',
'V302',
'V138',
'V179',
'id_27_Found',
'V226',
'V269',
'V237',
'V329',
'V284',
'V153',
'V157',
'V183',
'V17',
'V249',
'V112',
'id_28_Found',
'V177',
'V110',
'V181',
'V104',
'id_37_T',
'V106',
'V101',
'V123',
'V322',
'id_24',
'id_15_Found',
'V16',
'V155',
'V9',
'V182',
'id_29_Found',
'V190',
'id_16_Found',
'V297',
'V18',
'V327',
'V236',
'V227',
'V144',
'V22',
'V174',
'V173',
'V8',
'V211',
'V186',
'V331',
'V184',
'V337',
'V255',
'V247',
'V100',
'V333',
'V176',
'V151',
'V64',
'V242',
'V262',
'V202',
'V60',
'id_04',
'V235',
'V339',
'V213',
'id_38_F',
'V109',
'V115',
'V230',
'id_26',
'V214',
'M2_T',
'V298',
'V192',
'V304',
'id_37_F',
'id_15_Unknown',
'V288',
'V158',
'V301',
'V89',
'id_32_32.0',
'V273',
'id_22',
'V125',
'V212',
'V105',
'V150',
'M8_F',
'M7_T',
'V336',
'V166',
'V93',
'V231',
'V254',
'V85',
'id_10',
'V148',
'M7_F',
'V228',
'V338',
'V263',
'C3',
'V238',
'V84',
'id_38_T',
'V199',
'id_36_F',
'V6',
'V180',
'id_36_T',
'V185',
'V80',
'V159',
'V3',
'id_07',
'V204',
'id_08',
'V135',
'V286',
'V234',
'V232',
'V167',
'V188',
'id_17',
'V137',
'V257',
'V250',
'V260',
'V319',
'V10',
'V145',
'V132',
'V163',
'V161',
'V276',
'V2',
'V197',
'V141',
'V15',
'V39',
'V287',
'V58',
'V303',
'V216',
'V275',
'V239',
'V7',
'V229']


train_fitted = CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud'])


list(train_fitted.columns)


len(list(train_fitted.columns))


#train_fitted_wo = CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud']).drop(remove_l_intermediate,axis=1) # без разницы когда дропать - ведь у нас даже по таргету вдоль колонки преолбразовния - а нек нескольких - но после - чтобы колонки уже бвли после one_hot
for i in remove_l_intermediate:
    try:
        train_fitted.drop(i,axis=1,inplace=True)
    except Exception as e:
        print(e)



len(list(train_fitted.columns))


# еще один подход – смотрим, как меняются важности
# признаков по мере увеличения глубины: наиболее
# важные признаки – те, которые начинают
# использоваться раньше остальных
# задаем сетку значений глубины
max_depth_grid = [1, 2, 3, 4, 5]
# создаем список fi, в который будем сохранять
# важности признаков, и сохраняем в него важности,
# рассчитанные для каждой из моделей
fi_depth = []
# обучаем модели с разными значениями глубины, получаем
# важности и записываем важности в список

# может быть из-за разных областей хотя \то же one_fot - на большей области категорий же должно быть не менььше - вопрос
#"['id_22_20.0', 'id_24_12.0', 'id_24_11.0', 'id_22_nan', 'id_22_41.0', 'id_22_35.0', 'id_22_33.0', 'id_22_31.0', 'id_22_21.0', 'id_24_15.0', 'id_22_19.0', 'id_22_14.0', 'id_22_12.0', 'id_24_nan', 'id_24_25.0', 'id_24_21.0', 'id_24_18.0', 'id_24_16.0', 'id_24_26.0', 'id_24_23.0', 'id_24_19.0'] not found in axis"

#train_fitted = CustomFunctionTransformer(object_columns = categoric_columns).fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud']).drop(remove_l_intermediate,axis=1) # без разницы когда дропать - ведь у нас даже по таргету вдоль колонки преолбразовния - а нек нескольких - но после - чтобы колонки уже бвли после one_hot

for max_depth in max_depth_grid:
    model_all_features = LGBMClassifier(
    random_state=42,
    max_depth=max_depth,
    n_estimators=300, 
    importance_type='gain',      
        **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0},
        **{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1})
    model_all_features.fit(train_fitted, df['isFraud'])
    fi_depth.append(model_all_features.feature_importances_)

# преобразовываем список в датафрейм, индексы в котором
# будут именами наших переменных
fi_depth = pd.DataFrame(
np.array(fi_depth).T,
columns=['importance ' + str(idx)
for idx in range(len(fi_depth))],
index=train_fitted.columns)
# вычисляем усредненные важности и добавляем столбец с ними
fi_depth['mean_importance'] = fi_depth.mean(axis=1)
# сортируем по убыванию усредненных важностей
fi_depth = fi_depth.sort_values(by='mean_importance', ascending=False)
# смотрим полученный датафрейм
fi_depth


fi_depth.to_excel("fi_depth_forest_feat_imp.xlsx")


!pip install rfpimp


# вычисляем матрицу зависимостей признаков, значения – это
# пермутированные важности признаков, с помощью которых
# мы пытаемся предсказать интересующий признак
from rfpimp import (feature_dependence_matrix,
plot_dependence_heatmap,
plot_corr_heatmap)
D = feature_dependence_matrix(train_fitted, sort_by_dependence=True)
viz = plot_dependence_heatmap(D, figsize=(18, 18))
viz # очень долго



#from cuml.ensemble import RandomForestClassifier
model = LGBMClassifier(
    random_state=42,
    max_depth=5,
    n_estimators=300, 
    importance_type='gain',      
        **{"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0},
        **{'bagging_fraction': 1, 'feature_fraction': 1, 'lambda_l1': 10, 'learning_rate': 0.1})
model.fit(train_fitted, df['isFraud'])


from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
import numpy as np

# Функция перестановочной важности
def permutation_importance(model, X, y, metric=accuracy_score, n_repeats=5):
    baseline_score = metric(y, model.predict(X))
    importances = cudf.Series(0.0, index=X.columns)
    
    for col in X.columns:
        shuffled_scores = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[col] = X_shuffled[col].sample(frac=1).reset_index(drop=True)  # перемешивание
            y_pred = model.predict(X_shuffled)
            shuffled_scores.append(metric(y, y_pred))
        avg_shuffled_score = np.mean(shuffled_scores)
        importances[col] = baseline_score - avg_shuffled_score  # ухудшение метрики
    return importances.sort_values(ascending=False)


# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_fitted, df['isFraud'])

# Вычисление важности
importance = permutation_importance(model, train_fitted, df['isFraud'])
print("Перестановочная важность признаков:")
print(importance)








# F2-мера (уделяет в 4 раза больше внимания recall чем precision)
import cudf
import cupy as cp
import numpy as np
from cuml.metrics import roc_auc_score
from tqdm import tqdm
def f2_score(y_true, y_pred):
    y_true = y_true.values if hasattr(y_true, 'values') else y_true
    y_pred = y_pred.values if hasattr(y_pred, 'values') else y_pred
    
    tp = cp.sum((y_true == 1) & (y_pred == 1))
    fp = cp.sum((y_true == 0) & (y_pred == 1))
    fn = cp.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-15)
    return float(f2)

# PR-AUC (Precision-Recall AUC)
def pr_auc_score(y_true, y_pred_proba):
    y_true = y_true.values if hasattr(y_true, 'values') else y_true
    y_pred_proba = y_pred_proba.values if hasattr(y_pred_proba, 'values') else y_pred_proba
    
    # Сортировка по убыванию вероятности класса 1
    indices = cp.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[indices]
    
    # Кумулятивные суммы
    cum_tp = cp.cumsum(y_true_sorted)
    precision = cum_tp / (cp.arange(1, len(y_true_sorted) + 1))
    recall = cum_tp / cp.sum(y_true_sorted)
    
    # Вычисление AUC с помощью правила трапеций
    auc_val = cp.trapz(precision, recall)
    return float(auc_val)



# Функция перестановочной важности для GPU
def permutation_importance_fraud(model, X, y, metric='f2', n_repeats=5):
    # Базовые предсказания
    baseline_pred = model.predict(X)
    baseline_proba = model.predict_proba(X)[:, 1] if metric == 'pr_auc' else None
    
    # Базовый скор
    if metric == 'f2':
        baseline_score = f2_score(y, baseline_pred)
    elif metric == 'pr_auc':
        baseline_score = pr_auc_score(y, baseline_proba)
    else:
        raise ValueError("Unsupported metric. Use 'f2' or 'pr_auc'")
    
    importances = cudf.Series(0.0, index=X.columns)
    original_data = {col: X[col].copy() for col in X.columns}
    
    # Прогресс-бар для отслеживания выполнения
    for col in tqdm(X.columns, desc=f"Calculating PI ({metric.upper()})"):
        shuffled_scores = []
        
        for _ in range(n_repeats):
            # Сохраняем оригинальную колонку
            original_col = original_data[col]
            
            # Перемешиваем колонку
            X[col] = original_col.sample(frac=1.0).reset_index(drop=True)
            
            # Вычисляем метрику
            if metric == 'f2':
                y_pred = model.predict(X)
                score = f2_score(y, y_pred)
            elif metric == 'pr_auc':
                y_proba = model.predict_proba(X)[:, 1]
                score = pr_auc_score(y, y_proba)
            
            shuffled_scores.append(score)
            
            # Восстанавливаем оригинальную колонку
            X[col] = original_col
        
        # Среднее ухудшение метрики
        avg_shuffled_score = np.mean(shuffled_scores)
        importances[col] = baseline_score - avg_shuffled_score
    
    return importances.sort_values(ascending=False)



#print("Generating fraud dataset...")
#X, y = generate_fraud_data(n_samples=500000, n_features=50)
#print(f"Class distribution:\n{y.value_counts()}")
#print(f"Fraud rate: {y.mean().item():.4%}")
#
## Разделение на train/test
#split_index = int(len(X) * 0.8)
#X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
#X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

# пришлось перезапустить
from cuml.ensemble import RandomForestClassifier # RAPIDS (v24.02+) is incompatible with the `GPU P100` accelerator. Please consider switching to `GPU T4`.


# Обучение модели
print("\nTraining RandomForest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    n_bins=128,
    max_features=0.3,
    random_state=42,
    #class_weight='balanced', TypeError: (' The Scikit-learn variable ', 'class_weight', ' is not supported in cuML, please read the cuML documentation at (https://docs.rapids.ai/api/cuml/nightly/api.html#random-forest) for more information')

    n_streams=4  # Параллелизация на GPU
)
model.fit(train_fitted, df['isFraud'])

# Оценка модели
print("\nEvaluating model performance:")
test_pred = model.predict(train_fitted)
test_proba = model.predict_proba(train_fitted)[:, 1]

# F2-score
test_f2 = f2_score(df['isFraud'], test_pred)
print(f"Test F2-score: {test_f2:.4f}")

# PR-AUC
test_prauc = pr_auc_score(df['isFraud'], test_proba)
print(f"Test PR-AUC: {test_prauc:.4f}")

# ROC-AUC
test_rocauc = roc_auc_score(df['isFraud'], test_proba)
print(f"Test ROC-AUC: {test_rocauc:.4f}")

# Перестановочная важность с F2
print("\nComputing permutation importance with F2 metric...")
pi_f2 = permutation_importance_fraud(model, train_fitted, df['isFraud'], metric='f2', n_repeats=5)
print("\nTop 20 features by F2 importance:")
print(pi_f2.head(20).to_string())

# Перестановочная важность с PR-AUC
print("\nComputing permutation importance with PR-AUC metric...")
pi_prauc = permutation_importance_fraud(model, train_fitted, df['isFraud'], metric='pr_auc', n_repeats=5)
print("\nTop 20 features by PR-AUC importance:")
print(pi_prauc.head(20).to_string())

# Сохранение результатов
pi_f2.to_csv("f2_importance.csv")
pi_prauc.to_csv("prauc_importance.csv")
print("\nResults saved to f2_importance.csv and prauc_importance.csv")


test_f2 = f2_score(df['isFraud'], test_pred)
print(f"Test F2-score: {test_f2:.4f}")


print("\nComputing permutation importance with F2 metric...")
pi_f2 = permutation_importance_fraud(model, train_fitted, df['isFraud'], metric='f2', n_repeats=1)# 5 очень долго
print("\nTop 200 features by F2 importance:")
print(pi_f2.head(200).to_string())


type(pi_f2)


pi_f2.to_pandas().to_csv("f2_importance.csv")
# может понадобится заново получить fi)df


pi_f2.to_pandas().sort_values().to_excel("f2_importance_sorted.xlsx")



# можно оставить такие
первое из за модуля -а далее подряд c такой важности V30	0,001000938
# будем пользоваться такими фичами далее
card3
V30
V306
V321
V91
V261
V78
id_09
dist1
C9
V246
V74
V289
V131
id_02
card6_credit
V200
V86
V83
V81
V285
D11
V53
V187
V76
card6_debit
V70
V127
V20
V97
V75
D8
V280
D3
card5
V243
V156
C1
V61
V67
V320
V54
V44
id_20
V12
V134
V189
M4_M0
V296
V281
V128
V282
V310
V201
V82
V13
id_19
V295
V102
D4
V244
D10
D15
V133
V307
D1
V149
V312
V314
P_emaildomain
V87
V258
M5_T
C6
V62
C10
TransactionAmt
C7
C2
V308
D2
V318
TransactionDT
addr1
V45
V283
V294
V313
R_emaildomain
card2
V315
C11
C8
C4
C13
V317
C14
DeviceInfo
card1



pd.to_csv("f2_importance.csv")
# может понадобится заново получить fi)df


!pip install rfpimp


# выводим матрицу корреляций (на основе
# ранговой корреляции Спирмена)
from rfpimp import (feature_dependence_matrix,
plot_dependence_heatmap,
plot_corr_heatmap)
viz2 = plot_corr_heatmap(train_fitted,
figsize=(18, 18),
label_fontsize=8,
value_fontsize=7)


type(viz2)


import matplotlib.pyplot as plt

plt.tight_layout()  # Устраняет наложение меток
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()


dir(viz2)


viz2.figure.tight_layout()
viz2.figure.savefig("correlation_heatmap2.png", dpi=1000)
#plt.close(ax.figure)


viz2.save("correlation_heatmap3.png")


viz2.save('correlation_heatmap4.svg')


dir(viz2._repr_svg_)


plt.savefig("correlation_heatmap5.png", bbox_inches="tight", pad_inches=0.03)


viz2.get_figure().savefig("output_image.png", dpi=300, bbox_inches='tight')


# выводим матрицу корреляций (на основе
# ранговой корреляции Спирмена)
from rfpimp import (feature_dependence_matrix,
plot_dependence_heatmap,
plot_corr_heatmap)
viz2 = plot_corr_heatmap(train_fitted,
figsize=(18, 18),
label_fontsize=8,
value_fontsize=7)
plt.savefig("output_image.png", dpi=1000, bbox_inches='tight')



dir(viz2)


viz2.__getstate__


# 1. Вычисление матрицы корреляции (метод Спирмена)
from scipy.stats import spearmanr
spearmanr_corr_matrix = pd.DataFrame(spearmanr(train_fitted).correlation, columns=train_fitted.columns, index=train_fitted.columns)



spearmanr_corr_matrix.to_excel("spearmanr_corr_matrix.xlsx")


# там на каждом куске свои фичи и групп может быть больше в onehot


spearmanr_corr_matrix = pd.read_excel('/kaggle/input/spearmanr-corr-matrix/spearmanr_corr_matrix.xlsx')


spearmanr_corr_matrix


train_fitted.loc[:,train_fitted.dtypes!=object].columns


train_fitted.loc[:,train_fitted.dtypes==str].columns


train_fitted.dtypes


spearmanr_corr_matrix


spearmanr_corr_matrix = spearmanr_corr_matrix.set_index('Unnamed: 0')


### функции для фильтрации признаков

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df,corr_matr, n=5):
    au_corr = corr_matr.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
train_fitted_numeric = train_fitted.loc[:,train_fitted.dtypes!=object].columns
print(get_top_abs_correlations(train_fitted[train_fitted_numeric ],spearmanr_corr_matrix, 500))


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
                    
    


spearman_correlation_09 = correlation(train_fitted[train_fitted_numeric ],spearmanr_corr_matrix,0.9)


len(spearman_correlation_09)


spearman_correlation_09


fi_depth.loc[list(spearman_correlation_09),:]


pd.set_option('display.max_rows', None)

fi_depth.loc[list(spearman_correlation_09),:]


fi_depth_spearman_correlation_09 =  fi_depth.loc[list(spearman_correlation_09),:]


fi_depth_spearman_correlation_09.loc[(fi_depth_spearman_correlation_09['mean_importance']<100) & (fi_depth_spearman_correlation_09['importance 0']<100) & (fi_depth_spearman_correlation_09['importance 1']<100)& (fi_depth_spearman_correlation_09['importance 2']<100)& (fi_depth_spearman_correlation_09['importance 3']<100)& (fi_depth_spearman_correlation_09['importance 4']<100),:]
# такое условие так как есть у которых значисость больше 100 на одном из фолдов


list(fi_depth_spearman_correlation_09.loc[(fi_depth_spearman_correlation_09['mean_importance']<100) & (fi_depth_spearman_correlation_09['importance 0']<100) & (fi_depth_spearman_correlation_09['importance 1']<100)& (fi_depth_spearman_correlation_09['importance 2']<100)& (fi_depth_spearman_correlation_09['importance 3']<100)& (fi_depth_spearman_correlation_09['importance 4']<100),:].index)


len(list(fi_depth_spearman_correlation_09.loc[(fi_depth_spearman_correlation_09['mean_importance']<100) & (fi_depth_spearman_correlation_09['importance 0']<100) & (fi_depth_spearman_correlation_09['importance 1']<100)& (fi_depth_spearman_correlation_09['importance 2']<100)& (fi_depth_spearman_correlation_09['importance 3']<100)& (fi_depth_spearman_correlation_09['importance 4']<100),:].index))


len(list(fi_depth_spearman_correlation_09.loc[(fi_depth_spearman_correlation_09['mean_importance']<100),:].index)) # на 20 больше


len(train_fitted.columns)


train_fitted.drop(list(fi_depth_spearman_correlation_09.loc[(fi_depth_spearman_correlation_09['mean_importance']<100) & (fi_depth_spearman_correlation_09['importance 0']<100) & (fi_depth_spearman_correlation_09['importance 1']<100)& (fi_depth_spearman_correlation_09['importance 2']<100)& (fi_depth_spearman_correlation_09['importance 3']<100)& (fi_depth_spearman_correlation_09['importance 4']<100),:].index),axis=1,inplace=True)


len(train_fitted.columns) # remove_l_intermediate - был засчет permutatiopn удаление


f2_importance_sorted = pd.read_excel('/kaggle/input/f2-importance-sorted/f2_importance_sorted.xlsx')


f2_importance_sorted = f2_importance_sorted.set_index('Unnamed: 0')



set(f2_importance_sorted.index).intersection(set(train_fitted.columns))


len(set(f2_importance_sorted.index).intersection(set(train_fitted.columns)))


f2_importance_sorted





cols = set(f2_importance_sorted.index).intersection(set(train_fitted.columns))


f2_importance_sorted.loc[list(cols),:]


f2_importance_sorted = f2_importance_sorted.loc[list(cols),:].sort_values(by=0)


f2_importance_sorted


f2_importance_sorted.apply(abs)


f2_importance_sorted_abs = f2_importance_sorted.apply(abs).sort_values(by=0)


f2_importance_sorted_abs


train_fitted_corr = train_fitted.corr()



train_fitted_corr


train_fitted_corr.to_excel("train_fitted_corr_aft_del.xlsx")


train_fitted_numeric = train_fitted.loc[:,train_fitted.dtypes!=object].columns

linear_correlation_09 = correlation(train_fitted[train_fitted_numeric ],train_fitted_corr,0.9)


linear_correlation_09


fi_depth.loc[list(linear_correlation_09),:]


V168,V295,V321,V134 # посмотрел глазами и увидел из -что корреляция >0.9 есть


train_fitted.drop(['V168','V295','V321','V134'],axis=1,inplace=True)


len(train_fitted.columns)


fi_depth_spearman_correlation_09.loc[(fi_depth_spearman_correlation_09['mean_importance']<100) & (fi_depth_spearman_correlation_09['importance 0']<100) & (fi_depth_spearman_correlation_09['importance 1']<100)& (fi_depth_spearman_correlation_09['importance 2']<100)& (fi_depth_spearman_correlation_09['importance 3']<100)& (fi_depth_spearman_correlation_09['importance 4']<100),:]



1


train_fitted.describe()


train_fitted.describe(include='object')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for col in list(train_fitted.columns): # без transactionID и isFraud    
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    
    sns.boxplot(y=col, x=df['isFraud'].astype('category'), data=train_fitted)
    
    plt.show()


# в принципе разные за исключением откровенных выбросов


list(train_fitted.columns)


list(train_fitted.columns).index('DeviceType_desktop')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for col in list(train_fitted.columns[29:]): # без transactionID и isFraud    
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    
    sns.boxplot(y=col, x=df['isFraud'].astype('category'), data=train_fitted)
    
    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for col in list(train_fitted.columns[30:]): # без transactionID и isFraud    
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    
    sns.boxplot(y=col, x=df['isFraud'].astype('category'), data=train_fitted)
    
    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for col in list(train_fitted.columns[31:]): # без transactionID и isFraud    
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    
    sns.boxplot(y=col, x=df['isFraud'].astype('category'), data=train_fitted)
    
    plt.show()


import pandas as pd # не знает как рисовать булевы
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for col in list(train_fitted.columns[32:]): # без transactionID и isFraud    
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    
    sns.boxplot(y=col, x=df['isFraud'].astype('category'), data=train_fitted)
    
    plt.show()


for col in list(train_fitted.columns[30:]): # без transactionID и isFraud
    try:
        fig = plt.figure()
        fig.set_size_inches(16, 10)
        
        sns.boxplot(y=col, x=df['isFraud'].astype('category'), data=train_fitted)
        
        plt.show()
    except Exception as e:
        print(e)


# до зщвисания - реально все разные ! - посмотрел много


len(train_fitted.columns)


with open('train_fitted_columns_selected.txt','w') as f:
    f.write(str(list(train_fitted.columns)))


from catboost import CatBoostClassifier, Pool
import numpy as np
#params = {
#'n_estimators': [300],
#          'max_depth': [2, 4, 5, 7],
#          'l2_leaf_reg': [0.1,0.5], 
#          'random_strength': [0.1,0.5,1.0], 
#          'random_state': [777],
#    'learning_rate': [0.1]
#}

    params = {'depth':[2, 3, 4],
              'loss_function': ['Logloss', 'CrossEntropy'],
              'l2_leaf_reg':np.logspace(-20, -19, 3)
    }
  
kit = CatBoostClassifier(**{'grow_policy': 'Lossguide','eval_metric': 'PRAUC:hints=skip_train~false', 
                                        'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 
                                        'logging_level': 'Silent','task_type':"GPU",
                           'devices':'0'
                              })
import pandas as pd 

X_train_not_imp_cols_0_1_and_corr0_9_categoric_columns = X_train_not_imp_cols_0_1_and_corr0_9.loc[:,X_train_not_imp_cols_0_1_and_corr0_9.dtypes==object].columns 

for col in list(X_train_not_imp_cols_0_1_and_corr0_9_categoric_columns):
    X_train_not_imp_cols_0_1_and_corr0_9[col] = X_train_not_imp_cols_0_1_and_corr0_9[col].astype('str')

pool_train = Pool(X_train_not_imp_cols_0_1_and_corr0_9, label=y_train, cat_features=list(X_train_not_imp_cols_0_1_and_corr0_9_categoric_columns))






import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
import time

# 1. F2 Score Metric
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

# Create F2 scorer
f2_scorer = make_scorer(f2_score)

# 2. Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# 3. Parameter Grid
param_grid = {
    'iterations': [500, 1000],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64,128],
    'verbose': [False]
}

# 4. Grid Search with TimeSeriesSplit
def grid_search_catboost(X, y):
    # Split into train/test (preserving time order)
    #split_idx = int(len(X) * (1 - test_size))
    #X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    #y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Create CatBoost pool
    train_pool = Pool(X, y)
    
    # Initialize model
    model = CatBoostClassifier(task_type='GPU', devices='0:1',grow_policy='Lossguide')  # Use GPU #Warning: less than 75% GPU memory available for training. Free: 730.125 Total: 15095.0625

    
    # Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=f2_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting Grid Search...")
    start_time = time.time()
    grid_search.fit(X, y)
    print(f"Grid Search completed in {time.time() - start_time:.2f} seconds")
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best F2 Score: {grid_search.best_score_:.4f}")
    
    # Refit by accuracy
    print("\nRefitting by accuracy...")
    accuracy_model = CatBoostClassifier(
        **best_model.get_params(),
        eval_metric='Accuracy',
        early_stopping_rounds=50
    )
    
    accuracy_model.fit(
        train_pool,
        eval_set=(X, y),#eval_set=(X_test, y_test),
        verbose=100
    )
    
    return best_model, accuracy_model, (X, y)



grid_search_catboost(train_fitted, df['isFraud'])





# ПОВТОР - достаем фичи из train_fitted_columns_selected.txt
train_fitted = train_fitted[['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'DeviceInfo', 'DeviceType_desktop', 'DeviceType_mobile', 'M2_F', 'M3_F', 'M3_T', 'M4_M0', 'M4_M1', 'M4_M2', 'M5_F', 'M5_T', 'M6_F', 'M6_T', 'M8_T', 'M9_F', 'M9_T', 'P_emaildomain', 'ProductCD_H', 'ProductCD_R', 'ProductCD_S', 'R_emaildomain', 'TransactionAmt', 'TransactionDT', 'V102', 'V11', 'V12', 'V124', 'V126', 'V127', 'V128', 'V129', 'V13', 'V130', 'V131', 'V133', 'V136', 'V139', 'V140', 'V143', 'V147', 'V149', 'V152', 'V154', 'V156', 'V162', 'V164', 'V165', 'V169', 'V170', 'V172', 'V175', 'V187', 'V189', 'V19', 'V194', 'V198', 'V20', 'V200', 'V201', 'V203', 'V206', 'V208', 'V209', 'V215', 'V217', 'V219', 'V220', 'V221', 'V223', 'V224', 'V225', 'V23', 'V233', 'V243', 'V244', 'V245', 'V248', 'V25', 'V251', 'V256', 'V258', 'V261', 'V264', 'V265', 'V266', 'V270', 'V271', 'V274', 'V277', 'V279', 'V280', 'V281', 'V282', 'V283', 'V285', 'V289', 'V29', 'V290', 'V291', 'V292', 'V293', 'V294', 'V296', 'V300', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V320', 'V323', 'V326', 'V33', 'V332', 'V335', 'V35', 'V37', 'V38', 'V4', 'V40', 'V42', 'V43', 'V44', 'V45', 'V46', 'V48', 'V49', 'V5', 'V50', 'V53', 'V54', 'V55', 'V56', 'V57', 'V61', 'V62', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V75', 'V76', 'V77', 'V78', 'V79', 'V81', 'V82', 'V83', 'V86', 'V87', 'V88', 'V91', 'V94', 'V96', 'V99', 'addr1', 'addr2', 'card1', 'card2', 'card3', 'card4_discover', 'card4_mastercard', 'card4_visa', 'card5', 'card6_credit', 'card6_debit', 'dist1', 'dist2', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09', 'id_11', 'id_13', 'id_14', 'id_15_New', 'id_18', 'id_19', 'id_20', 'id_21', 'id_25', 'id_30', 'id_31', 'id_32_24.0', 'id_33']]


len(list(train_fitted.columns))# да - ок так и было


import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
import time

# 1. F2 Score Metric
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

# Create F2 scorer
f2_scorer = make_scorer(f2_score)

# 2. Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# 3. Parameter Grid
param_grid = {
    'iterations': [500, 1000],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64],
    'verbose': [False]
}

# 4. Grid Search with TimeSeriesSplit
def grid_search_catboost(X, y, test_size=0.2):
    # Split into train/test (preserving time order)
    #split_idx = int(len(X) * (1 - test_size))
    #X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    #y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Create CatBoost pool
    train_pool = Pool(X_train, y_train)
    
    # Initialize model
    model = CatBoostClassifier(task_type='GPU', devices='0:1')  # Use GPU
    
    # Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=f2_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting Grid Search...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print(f"Grid Search completed in {time.time() - start_time:.2f} seconds")
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best F2 Score: {grid_search.best_score_:.4f}")
    
    # Refit by accuracy
    print("\nRefitting by accuracy...")
    accuracy_model = CatBoostClassifier(
        **best_model.get_params(),
        eval_metric='Accuracy',
        early_stopping_rounds=50
    )
    
    accuracy_model.fit(
        train_pool,
        #eval_set=(X_test, y_test),
        verbose=100
    )
    
    return best_model, accuracy_model, (X_test, y_test)

# 5. Evaluation Function
def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    f2 = f2_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # Feature Importance
    feature_importances = model.get_feature_importance()
    features = model.feature_names_
    
    print(f"\n{' Metric ':=^40}")
    print(f"F2 Score: {f2:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    print(f"\n{' Top Features ':=^40}")
    for imp, name in sorted(zip(feature_importances, features), reverse=True)[:10]:
        print(f"{name}: {imp:.4f}")
    
    return f2, acc

# 6. Main Execution
if __name__ == "__main__":
    # Load your time-series data
    # df = pd.read_csv('your_data.csv')
    # X = df.drop(columns=['target'])
    # y = df['target']
    
    # Generate example data (replace with your data)
    print("Generating example time-series data...")
    n_samples = 10000
    X = pd.DataFrame({
        'feature1': np.cumsum(np.random.randn(n_samples)),
        'feature2': np.sin(np.linspace(0, 20, n_samples)),
        'feature3': np.log(np.arange(1, n_samples + 1)),
        'feature4': np.random.randint(0, 100, n_samples),
        'feature5': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    })
    y = (X['feature1'].rolling(10).mean() > 0).astype(int).fillna(0)
    
    # Run grid search
    f2_model, acc_model, (X_test, y_test) = grid_search_catboost(X, y)
    
    # Evaluate F2 model
    print("\nEvaluating F2-optimized model:")
    f2_score, f2_acc = evaluate_model(f2_model, X_test, y_test)
    
    # Evaluate Accuracy-refitted model
    print("\nEvaluating Accuracy-refitted model:")
    acc_score, acc_acc = evaluate_model(acc_model, X_test, y_test)
    
    # Compare results
    print(f"\n{' Comparison ':=^40}")
    print(f"F2 Model: F2={f2_score:.4f}, Accuracy={f2_acc:.4f}")
    print(f"Accuracy Model: F2={acc_score:.4f}, Accuracy={acc_acc:.4f}")
    
    # Save models
    f2_model.save_model('catboost_f2_model.cbm')
    acc_model.save_model('catboost_acc_model.cbm')
    print("\nModels saved to catboost_f2_model.cbm and catboost_acc_model.cbm")





import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import fbeta_score, accuracy_score
import time
import gc

# F2 Score Metric with GPU-friendly implementation
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

# Memory-optimized grid search with TimeSeriesSplit
def optimized_grid_search(X, y, test_size=0.2):
    # Time-based split
    #split_idx = int(len(X) * (1 - test_size))
    #X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    #y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # GPU configuration - optimized for T4 with memory constraints
    gpu_params = {
        'task_type': 'GPU',
        'devices': '0:1',  # Use both GPUs
        'bootstrap_type': 'Bernoulli',  # Less memory intensive
        'subsample': 0.8,  # Reduce memory usage
        'sampling_frequency': 'PerTree',
        'max_ctr_complexity': 1,  # Reduce memory for categorical features
        'border_count': 64,  # Reduce from default 254 to save memory
        'used_ram_limit': '10gb',  # Set explicit memory limit
    }
    
    # Parameter grid optimized for GPU memory
    param_grid = {
        'iterations': [500, 700],
        'depth': [4, 6],
        'learning_rate': [0.03, 0.05],
        'l2_leaf_reg': [3, 5],
        'grow_policy': ['Lossguide'],  # More memory efficient
        'max_leaves': [31, 63],       # Controls model size
    }
    
    # Store best params and scores
    best_score = -1
    best_params = None
    best_model = None
    
    # Manual grid search with memory management
    print("Starting memory-optimized grid search...")
    start_time = time.time()
    
    for depth in param_grid['depth']:
        for l2 in param_grid['l2_leaf_reg']:
            for lr in param_grid['learning_rate']:
                for n_est in param_grid['iterations']:
                    for policy in param_grid['grow_policy']:
                        for leaves in param_grid['max_leaves']:
                            # Configure parameters
                            params = {
                                **gpu_params,
                                'depth': depth,
                                'l2_leaf_reg': l2,
                                'learning_rate': lr,
                                'iterations': n_est,
                                'grow_policy': policy,
                                'max_leaves': leaves,
                                'verbose': False,
                                'allow_writing_files': False,  # Reduce disk I/O
                            }
                            
                            # TimeSeries Cross-Validation
                            tscv = TimeSeriesSplit(n_splits=5)
                            cv_scores = []
                            
                            for train_idx, val_idx in tscv.split(X):
                                # Create splits
                                X_tr = X.iloc[train_idx]
                                y_tr = y.iloc[train_idx]
                                X_val = X.iloc[val_idx]
                                y_val = y.iloc[val_idx]
                                
                                # Train with memory cleanup
                                model = CatBoostClassifier(**params)
                                model.fit(
                                    X_tr, y_tr,
                                    eval_set=(X_val, y_val),
                                    early_stopping_rounds=20,
                                    use_best_model=True
                                )
                                
                                # Predict and calculate F2
                                preds = model.predict(X_val)
                                score = f2_score(y_val, preds)
                                cv_scores.append(score)
                                
                                # Explicit cleanup
                                del model
                                gc.collect()
                            
                            # Average CV score
                            mean_score = np.mean(cv_scores)
                            
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = params
                                print(f"New best F2: {best_score:.4f} with {params}")
    
    print(f"Grid search completed in {time.time()-start_time:.2f} seconds")
    
    # Train best model on full data
    print("\nTraining final model with best parameters...")
    print(f"best_params {best_params}")
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X, y)
    
    return final_model, best_params, (X, y)


final_model, best_params,_ =optimized_grid_search(train_fitted, df['isFraud'])


def funci():
    return 1,2,(1,2)

aa,bb,cc = funci()


aa,bb,cc,dd = funci()


aa,bb,cc


# Memory-optimized refitting by accuracy
def refit_by_accuracy(model, X, y, X_test, y_test):
    # Get parameters from existing model
    params = model.get_params()
    
    # Update parameters for accuracy tuning
    params.update({
        'eval_metric': 'Accuracy',
        'early_stopping_rounds': 30,
        'used_ram_limit': '10gb',
        'learning_rate': params['learning_rate'] * 0.5,  # Lower LR for fine-tuning
        'iterations': 1000  # More iterations for refinement
    })
    
    print("\nRefitting model for accuracy optimization...")
    acc_model = CatBoostClassifier(**params)
    acc_model.fit(
        X, y,
        eval_set=(X_test, y_test),
        verbose=100
    )
    
    return acc_model

# Evaluate F2 model
f2_pred = final_model.predict(train_fitted)
f2_value = f2_score(df['isFraud'], f2_pred)
print(f"\nF2-optimized model performance:")
print(f"F2 Score: {f2_value:.4f}")

# Refit by accuracy
acc_model = refit_by_accuracy(final_model, train_fitted, df['isFraud'], train_fitted, df['isFraud'])

# Evaluate accuracy model
acc_pred = acc_model.predict(train_fitted)
accuracy = accuracy_score(df['isFraud'], acc_pred)
print(f"\nAccuracy-optimized model performance:")
print(f"Accuracy: {accuracy:.4f}")



f2_score(df['isFraud'], acc_pred)# модель на accuracy


accuracy_score(df['isFraud'], f2_pred)


from sklearn.metrics import precision_score

accuracy = precision_score(df['isFraud'], acc_pred)


accuracy# это precision - вроде хорошие показатели! и 0.98 accuracy





# Save models
final_model.save_model('catboost_f2_model.cbm', format='cbm')
acc_model.save_model('catboost_acc_model.cbm', format='cbm')


df_test.columns


# тестируем acc_model


#for col in categoric_columns:
#    most_recent = df.groupby(col).size().sort_values().index[-1] # по возрастанию же
#    
#    df_test[col] = df_test[col].fillna(most_recent) # заполнение данными из train!
#
#### Заполним средним
#for col in [item for item in numeric_columns if item not in ['isFraud']]: #KeyError: 'isFraud'
#    df_test[col] = df_test[col].fillna(df[col].mean())# заполнение данными из train!
df_test = pd.merge(test_transaction,test_identity,'left','TransactionID')

df_test.rename(columns={
    x : x.replace('-', '_') for x in list(
        set(df_test.columns).difference(set(df.columns))
                                         ) 
}
              ,inplace=True)
               
df_test[['M'+str(i) for i in range(1,10)]] = df_test[['M'+str(i) for i in range(1,10)]].astype('str')
df_test[['card'+str(i) for i in range(1,7)]] = df_test[['card'+str(i) for i in range(1,7)]].astype('str')
df_test[['addr'+str(i) for i in range(1,3)]] = df_test[['addr'+str(i) for i in range(1,3)]].astype('str')
df_test[[str(i) + '_emaildomain' for i in ['P','R'] ]] = df_test[[str(i) + '_emaildomain' for i in ['P','R']]].astype('str')
df_test[['id_'+str(i) for i in range(12,39)]] = df_test[['id_'+str(i) for i in range(12,39)]].astype('str')

df_test = CustomFunctionTransformer(object_columns = categoric_columns).transform(df_test.drop(['TransactionID'],axis=1))

X_test = df_test.drop(['TransactionID'], axis=1)



#y_pred = cboost.predict(X_test)


# ошибка выше - так как не сохранял трансформер в переменую!


transformer = CustomFunctionTransformer(object_columns = categoric_columns)#.fit_transform(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud'])


transformer.fit(df.drop(['isFraud','TransactionID'],axis=1),df['isFraud'])


df_test = transformer.transform(df_test.drop(['TransactionID'],axis=1))



X_test = df_test.drop(['TransactionID'], axis=1)


X_test = df_test[['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'DeviceInfo', 'DeviceType_desktop', 'DeviceType_mobile', 'M2_F', 'M3_F', 'M3_T', 'M4_M0', 'M4_M1', 'M4_M2', 'M5_F', 'M5_T', 'M6_F', 'M6_T', 'M8_T', 'M9_F', 'M9_T', 'P_emaildomain', 'ProductCD_H', 'ProductCD_R', 'ProductCD_S', 'R_emaildomain', 'TransactionAmt', 'TransactionDT', 'V102', 'V11', 'V12', 'V124', 'V126', 'V127', 'V128', 'V129', 'V13', 'V130', 'V131', 'V133', 'V136', 'V139', 'V140', 'V143', 'V147', 'V149', 'V152', 'V154', 'V156', 'V162', 'V164', 'V165', 'V169', 'V170', 'V172', 'V175', 'V187', 'V189', 'V19', 'V194', 'V198', 'V20', 'V200', 'V201', 'V203', 'V206', 'V208', 'V209', 'V215', 'V217', 'V219', 'V220', 'V221', 'V223', 'V224', 'V225', 'V23', 'V233', 'V243', 'V244', 'V245', 'V248', 'V25', 'V251', 'V256', 'V258', 'V261', 'V264', 'V265', 'V266', 'V270', 'V271', 'V274', 'V277', 'V279', 'V280', 'V281', 'V282', 'V283', 'V285', 'V289', 'V29', 'V290', 'V291', 'V292', 'V293', 'V294', 'V296', 'V300', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V320', 'V323', 'V326', 'V33', 'V332', 'V335', 'V35', 'V37', 'V38', 'V4', 'V40', 'V42', 'V43', 'V44', 'V45', 'V46', 'V48', 'V49', 'V5', 'V50', 'V53', 'V54', 'V55', 'V56', 'V57', 'V61', 'V62', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V75', 'V76', 'V77', 'V78', 'V79', 'V81', 'V82', 'V83', 'V86', 'V87', 'V88', 'V91', 'V94', 'V96', 'V99', 'addr1', 'addr2', 'card1', 'card2', 'card3', 'card4_discover', 'card4_mastercard', 'card4_visa', 'card5', 'card6_credit', 'card6_debit', 'dist1', 'dist2', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09', 'id_11', 'id_13', 'id_14', 'id_15_New', 'id_18', 'id_19', 'id_20', 'id_21', 'id_25', 'id_30', 'id_31', 'id_32_24.0', 'id_33']]





y_pred_test_proba = acc_model.predict_proba(X_test)



y_pred_test = acc_model.predict(X_test)


rez = pd.concat(
    [pd.merge(test_transaction,test_identity,'left','TransactionID')['TransactionID'],pd.Series(y_pred_test),pd.DataFrame(y_pred_test_proba,columns=['prob_1','prob_2'])]
    , axis=1)


rez.to_csv('test_acc_model.csv',index=False)



rez_for_subm_aft_ing = rez[['TransactionID','prob_2']].copy()



rez_for_subm_aft_ing.rename(columns = {'prob_2':'isFraud'},inplace=True)



rez_for_subm_aft_ing.to_csv('test_acc_model_subm.csv',index=False)



# f2 model


y_pred_testf2_proba = final_model.predict_proba(X_test)
y_pred_testf2 = final_model.predict(X_test)
rezf2 = pd.concat(
    [pd.merge(test_transaction,test_identity,'left','TransactionID')['TransactionID'],pd.Series(y_pred_testf2),pd.DataFrame(y_pred_testf2_proba,columns=['prob_1','prob_2'])]
    , axis=1)
rezf2.to_csv('test_f2_model.csv',index=False)
rezf2_for_subm_aft_ing = rezf2[['TransactionID','prob_2']].copy()
rezf2_for_subm_aft_ing.rename(columns = {'prob_2':'isFraud'},inplace=True)
rezf2_for_subm_aft_ing.to_csv('test_f2_model_subm.csv',index=False)



# усредняем вероятности с весами,
# учитывающими качество модели - не ясно
average_prob = (y_pred_testf2_proba * 0.3 +
y_pred_test_proba * 0.7 ) / 2





import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import fbeta_score, accuracy_score
import time
import gc

# Memory-optimized grid search with TimeSeriesSplit
def optimized_grid_acc_search(X, y, test_size=0.2):
    # Time-based split
    #split_idx = int(len(X) * (1 - test_size))
    #X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    #y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # GPU configuration - optimized for T4 with memory constraints
    gpu_params = {
        'task_type': 'GPU',
        'devices': '0:1',  # Use both GPUs
        'bootstrap_type': 'Bernoulli',  # Less memory intensive
        'subsample': 0.8,  # Reduce memory usage
        'sampling_frequency': 'PerTree',
        'max_ctr_complexity': 1,  # Reduce memory for categorical features
        'border_count': 64,  # Reduce from default 254 to save memory
        'used_ram_limit': '10gb',  # Set explicit memory limit
    }
    
    # Parameter grid optimized for GPU memory
    param_grid = {
        'iterations': [1000, 2000],
        'depth': [2, 4, 6],
        'learning_rate': [0.1,],
        'l2_leaf_reg': [3, 5],
        'grow_policy': ['Lossguide'],  # More memory efficient
        'max_leaves': [31, 63],       # Controls model size
    }
    
    # Store best params and scores
    best_score = -1
    best_params = None
    best_model = None
    
    # Manual grid search with memory management
    print("Starting memory-optimized grid search...")
    start_time = time.time()
    
    for depth in param_grid['depth']:
        for l2 in param_grid['l2_leaf_reg']:
            for lr in param_grid['learning_rate']:
                for n_est in param_grid['iterations']:
                    for policy in param_grid['grow_policy']:
                        for leaves in param_grid['max_leaves']:
                            # Configure parameters
                            params = {
                                **gpu_params,
                                'depth': depth,
                                'l2_leaf_reg': l2,
                                'learning_rate': lr,
                                'iterations': n_est,
                                'grow_policy': policy,
                                'max_leaves': leaves,
                                'verbose': False,
                                'allow_writing_files': False,  # Reduce disk I/O
                            }
                            
                            # TimeSeries Cross-Validation
                            tscv = TimeSeriesSplit(n_splits=5)
                            cv_scores = []
                            
                            for train_idx, val_idx in tscv.split(X):
                                # Create splits
                                X_tr = X.iloc[train_idx]
                                y_tr = y.iloc[train_idx]
                                X_val = X.iloc[val_idx]
                                y_val = y.iloc[val_idx]
                                
                                # Train with memory cleanup
                                model = CatBoostClassifier(**params)
                                model.fit(
                                    X_tr, y_tr,
                                    eval_set=(X_val, y_val),
                                    early_stopping_rounds=20,
                                    use_best_model=True
                                )
                                
                                # Predict and calculate F2
                                preds = model.predict(X_val)
                                score = accuracy_score(y_val, preds)
                                cv_scores.append(score)
                                
                                # Explicit cleanup
                                del model
                                gc.collect()
                            
                            # Average CV score
                            mean_score = np.mean(cv_scores)
                            
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = params
                                print(f"New best accuracy_score: {best_score:.4f} with {params}")
    
    print(f"Grid search completed in {time.time()-start_time:.2f} seconds")
    
    # Train best model on full data
    print("\nTraining final model with best parameters...")
    print(f"best_params {best_params}")
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X, y)
    
    return final_model, best_params, (X, y)


len(list(train_fitted.columns))


final_grid_acc_model, best_grid_acc_params,_ =optimized_grid_acc_search(train_fitted, df['isFraud'])


final_grid_acc_model.save_model('catboost_final_grid_acc_model.cbm', format='cbm')



y_pred_testgrid_acc_proba = final_grid_acc_model.predict_proba(X_test)
y_pred_testgrid_acc = final_grid_acc_model.predict(X_test)
rezgrid_acc = pd.concat(
    [pd.merge(test_transaction,test_identity,'left','TransactionID')['TransactionID'],pd.Series(y_pred_testgrid_acc),pd.DataFrame(y_pred_testgrid_acc_proba,columns=['prob_1','prob_2'])]
    , axis=1)
rezgrid_acc.to_csv('test_grid_acc_model.csv',index=False)
rezgrid_acc_for_subm_aft_ing = rezgrid_acc[['TransactionID','prob_2']].copy()
rezgrid_acc_for_subm_aft_ing.rename(columns = {'prob_2':'isFraud'},inplace=True)
rezgrid_acc_for_subm_aft_ing.to_csv('test_grid_acc_model_subm.csv',index=False)



f2_score(df['isFraud'], final_grid_acc_model.predict(train_fitted))# модель на accuracy


f2_score(df['isFraud'],acc_model.predict(train_fitted))# модель на accuracy


f2_score(df['isFraud'], final_model.predict(train_fitted))# модель на accuracy


accuracy_score(df['isFraud'], final_grid_acc_model.predict(train_fitted))# модель на accuracy


accuracy_score(df['isFraud'], acc_model.predict(train_fitted))# модель на accuracy


accuracy_score(df['isFraud'], final_model.predict(train_fitted))# модель на accuracy


precision_score(df['isFraud'], final_grid_acc_model.predict(train_fitted))# модель на accuracy


precision_score(df['isFraud'], acc_model.predict(train_fitted))# модель на accuracy


# по метрикам лучшая final_grid_acc но она получила худший скор -но это метрики трейнга - значит переобучение





precision_score(df['isFraud'], final_model.predict(train_fitted))# модель на accuracy


from IPython.display import FileLink
FileLink('/kaggle/working/catboost_f2_model.cbm')


from IPython.display import FileLink
FileLink('catboost_f2_model.cbm')








import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import fbeta_score, accuracy_score
import time
import gc

# F2 Score Metric with GPU-friendly implementation
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

# Memory-optimized grid search with TimeSeriesSplit
def optimized_grid_search(X, y, test_size=0.2):
    # Time-based split
    #split_idx = int(len(X) * (1 - test_size))
    #X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    #y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # GPU configuration - optimized for T4 with memory constraints
    gpu_params = {
        'task_type': 'GPU',
        'devices': '0:1',  # Use both GPUs
        'bootstrap_type': 'Bernoulli',  # Less memory intensive
        'subsample': 0.8,  # Reduce memory usage
        'sampling_frequency': 'PerTree',
        'max_ctr_complexity': 1,  # Reduce memory for categorical features
        'border_count': 64,  # Reduce from default 254 to save memory
        'used_ram_limit': '10gb',  # Set explicit memory limit
    }
    
    # Parameter grid optimized for GPU memory
    param_grid = {
        'iterations': [500, 700],
        'depth': [4, 6],
        'learning_rate': [0.03, 0.05],
        'l2_leaf_reg': [3, 5],
        'grow_policy': ['Lossguide'],  # More memory efficient
        'max_leaves': [31, 63],       # Controls model size
    }
    
    # Store best params and scores
    best_score = -1
    best_params = None
    best_model = None
    
    # Manual grid search with memory management
    print("Starting memory-optimized grid search...")
    start_time = time.time()
    
    for depth in param_grid['depth']:
        for l2 in param_grid['l2_leaf_reg']:
            for lr in param_grid['learning_rate']:
                for n_est in param_grid['iterations']:
                    for policy in param_grid['grow_policy']:
                        for leaves in param_grid['max_leaves']:
                            # Configure parameters
                            params = {
                                **gpu_params,
                                'depth': depth,
                                'l2_leaf_reg': l2,
                                'learning_rate': lr,
                                'iterations': n_est,
                                'grow_policy': policy,
                                'max_leaves': leaves,
                                'verbose': False,
                                'allow_writing_files': False,  # Reduce disk I/O
                            }
                            
                            # TimeSeries Cross-Validation
                            tscv = TimeSeriesSplit(n_splits=5)
                            cv_scores = []
                            
                            for train_idx, val_idx in tscv.split(X):
                                # Create splits
                                X_tr = X.iloc[train_idx]
                                y_tr = y.iloc[train_idx]
                                X_val = X.iloc[val_idx]
                                y_val = y.iloc[val_idx]
                                
                                # Train with memory cleanup
                                model = CatBoostClassifier(**params)
                                model.fit(
                                    X_tr, y_tr,
                                    eval_set=(X_val, y_val),
                                    early_stopping_rounds=20,
                                    use_best_model=True
                                )
                                
                                # Predict and calculate F2
                                preds = model.predict(X_val)
                                score = f2_score(y_val, preds)
                                cv_scores.append(score)
                                
                                # Explicit cleanup
                                del model
                                gc.collect()
                            
                            # Average CV score
                            mean_score = np.mean(cv_scores)
                            
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = params
                                print(f"New best F2: {best_score:.4f} with {params}")
    
    print(f"Grid search completed in {time.time()-start_time:.2f} seconds")
    
    # Train best model on full data
    print("\nTraining final model with best parameters...")
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    return final_model, best_params, (X_test, y_test)

# Memory-optimized refitting by accuracy
def refit_by_accuracy(model, X, y, X_test, y_test):
    # Get parameters from existing model
    params = model.get_params()
    
    # Update parameters for accuracy tuning
    params.update({
        'eval_metric': 'Accuracy',
        'early_stopping_rounds': 30,
        'used_ram_limit': '10gb',
        'learning_rate': params['learning_rate'] * 0.5,  # Lower LR for fine-tuning
        'iterations': 1000  # More iterations for refinement
    })
    
    print("\nRefitting model for accuracy optimization...")
    acc_model = CatBoostClassifier(**params)
    acc_model.fit(
        X, y,
        eval_set=(X_test, y_test),
        verbose=100
    )
    
    return acc_model

# Main execution with GPU memory management
if __name__ == "__main__":
    # Generate example data
    n_samples = 500000
    print(f"Generating {n_samples} samples...")
    X = pd.DataFrame({
        'feat1': np.cumsum(np.random.randn(n_samples)),
        'feat2': np.sin(np.linspace(0, 20, n_samples)),
        'feat3': np.log(np.arange(1, n_samples + 1)),
        'feat4': np.random.rand(n_samples),
    })
    y = (X['feat1'].rolling(50).mean() > 0).astype(int).fillna(0)
    
    # Reduce memory usage
    for col in X.columns:
        if X[col].dtype == 'float64':
            X[col] = X[col].astype('float32')
    
    # Run optimized grid search
    f2_model, best_params, (X_test, y_test) = optimized_grid_search(X, y)
    
    # Evaluate F2 model
    f2_pred = f2_model.predict(X_test)
    f2_value = f2_score(y_test, f2_pred)
    print(f"\nF2-optimized model performance:")
    print(f"F2 Score: {f2_value:.4f}")
    
    # Refit by accuracy
    acc_model = refit_by_accuracy(f2_model, X, y, X_test, y_test)
    
    # Evaluate accuracy model
    acc_pred = acc_model.predict(X_test)
    accuracy = accuracy_score(y_test, acc_pred)
    print(f"\nAccuracy-optimized model performance:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Compare feature importances
    print("\nFeature importances comparison:")
    f2_importances = pd.Series(f2_model.feature_importances_, index=X.columns)
    acc_importances = pd.Series(acc_model.feature_importances_, index=X.columns)
    print(pd.DataFrame({
        'F2 Importance': f2_importances,
        'Accuracy Importance': acc_importances
    }).sort_values('F2 Importance', ascending=False))
    
    # Save models
    f2_model.save_model('catboost_f2_model.cbm', format='cbm')
    acc_model.save_model('catboost_acc_model.cbm', format='cbm')
    print("\nModels saved in CatBoost binary format")


# Compare feature importances
print("\nFeature importances comparison:")
f2_importances = pd.Series(final_model.feature_importances_, index=train_fitted.columns)
acc_importances = pd.Series(acc_model.feature_importances_, index=train_fitted.columns)
final_grid_acc_importances = pd.Series(final_grid_acc_model.feature_importances_, index=train_fitted.columns)
print(pd.DataFrame({
    'F2 Importance': f2_importances,
    'Accuracy Importance': acc_importances,
    "final_grid_acc_importances_pereobuch_train":final_grid_acc_importances
}).sort_values('F2 Importance', ascending=False))



pd.DataFrame({
    'F2 Importance': f2_importances,
    'Accuracy Importance': acc_importances,
    "final_grid_acc_importances_pereobuch_train":final_grid_acc_importances
}).sort_values('F2 Importance', ascending=False).to_excel('importances.xlsx')




