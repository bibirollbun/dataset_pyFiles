
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install ray==2.10.0
!pip install autogluon.tabular[all]
!pip install -U ipywidgets


#!pip install --upgrade pip


from autogluon.tabular import TabularPredictor
predictor_uoload = TabularPredictor.load("/kaggle/input/autogluon3.4/other/default/1/ag-20250304_043654")


predictor_uoload.leaderboard()


pip uninstall lightgbm -y	pip install lightgbm --install-option=--gpu


#!pip install --upgrade scikit-learn
#!pip install --upgrade imbalanced-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import gc
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')


df_train.head()


df_train.info()


df_test.head()


df_test.info()


ID = df_test['id']


print('Check for percentage of missing values in train dataset:')
df_train.isnull().mean()*100


print('Check for percentage of missing values in test dataset:')
df_test.isnull().mean()*100


from datetime import datetime
current_yr = datetime.now().year
df_train['age'] = current_yr - df_train['model_year']
df_test['age'] = current_yr - df_test['model_year']


NumCol_train = df_train.select_dtypes(include = 'int64').columns.drop(['id','model_year']).tolist()
NumCol_train.remove('price')
CatCol_train = df_train.select_dtypes(include = 'object').columns.tolist()
print('List of numerical features for train data set:%s'%(NumCol_train))
print('List of categorical features for train data set:%s'%(CatCol_train))


NumCol_test = df_test.select_dtypes(include = 'int64').columns.drop(['id','model_year']).tolist()
CatCol_test = df_test.select_dtypes(include = 'object').columns.tolist()
print('List of numerical features for train data set:%s'%(NumCol_test))
print('List of categorical features for train data set:%s'%(CatCol_test))


for col in CatCol_train:
    df_train[col].value_counts().nlargest(20).plot(kind = 'bar',figsize = (10,5))
    plt.title('Number of cars by %s'%(col))
    plt.ylabel('Number of cars')
    plt.xlabel(col)
    plt.xticks(rotation=75)
    plt.show()


for col in CatCol_test:
    df_test[col].value_counts().nlargest(20).plot(kind = 'bar',figsize = (10,5))
    plt.title('Number of cars by %s'%(col))
    plt.ylabel('Number of cars')
    plt.xlabel(col)
    plt.xticks(rotation=75)
    plt.show()


import seaborn as sns


train_heatmap = sns.heatmap(df_train.corr(numeric_only = True),annot = True)


def remove_outliers(df, column):
    q1 = np.percentile(df[column], 25, method='midpoint')
    q3 = np.percentile(df[column], 75, method = 'midpoint')
    IQR = q3-q1
    lowerbound = q1 - 1.5*IQR
    upperbound = q3 + 1.5*IQR
    df_returned = df[(df[column]>=lowerbound) & (df[column]<=upperbound)]
    return df_returned

for col in NumCol_train:
    train = remove_outliers(df_train, col)


plt.figure(figsize = (12,6))
plt.scatter(train['age'], train['milage'], cmap = 'terrain', c = train['price'])
plt.xlabel('age')
plt.ylabel('milage')
plt.colorbar(label = 'price')
plt.show()


train[CatCol_train] = train[CatCol_train].fillna('Unknown')
df_test[CatCol_test] = df_test[CatCol_test].fillna('Unknown')


train.isnull().sum()


df_test.isnull().sum()


#train = pd.read_csv('/kaggle/working/ProcessedTrain.csv')


def Unique(df, col):
    bar = 200
    for m in col:
        count = df[m].value_counts()
        uni_value = count[count < bar].index
        df[m] = df[m].apply(lambda x: 'NOISE' if x in uni_value else x)
    return df

Unique(train[CatCol_train], CatCol_train)
Unique(df_test[CatCol_test], CatCol_test)


train.info()


df_test.info()


#price = train['price']


train = train.drop(columns = ["id","model_year"])
train.info()


df_test = df_test.drop(columns = ["id","model_year"])
df_test.info()


train["accident"] = train["accident"].apply(lambda x: 0 if x == "None reported" else 1)
df_test["accident"] = df_test["accident"].apply(lambda x: 0 if x == "None reported" else 1)


#price.astype('float32')


#price.shape


train.info()


df_test.info()


#train[CatCol_train] = train[CatCol_train].astype('category')
#train[NumCol_train] = train[NumCol_train].astype('float32')


#df_test[CatCol_test] = df_test[CatCol_test].astype('category')
#df_test[NumCol_test] = df_test[NumCol_test].astype('float32')


train.info()


df_test.info()


from autogluon.tabular import TabularPredictor
Predictor = TabularPredictor(label = 'price',
                             eval_metric = 'rmse', 
                             problem_type = 'regression')
Predictor.fit(train_data=train, presets=['best_quality','optimize_for_deployment'], time_limit = 3600, 
              verbosity = 2,excluded_model_types =["NN","KNN","RF","XT"],
              ag_args_fit={'num_gpus': 2,'num_cpus':4})


Predictor.save()


import shutil


shutil.make_archive("autogluon_model",'zip',"/kaggle/working/AutogluonModels")


Predictor.leaderboard()


Predictor.info()


model_info = Predictor.info()[best_model]
print(model_info)


Predictor.fit_summary(show_plot = True)


best_model_name = Predictor.model_best


print(best_model_name)


model_info = Predictor.info()['model_info'][best_model_name]


Predictor.feature_importance(train)


child_info = model_info['children_info']['S1F1']['model_weights']
print("Ensemble Model Weights:",child_info)


sorted_model_weights = dict(sorted(child_info.items(), key = lambda item:item[1],reverse = True))


print(sorted_model_weights)


best_model = model_info['name']


print("Best Model Info:", "\nModel Name:",best_model,"\nModel Type:",
      model_info['model_type'],"\nModel Score:",model_info['val_score'] ,
     "\nChild Models and Weights:\n",sorted_model_weights)


print(model_info)


print("model type:",model_info['model_type'])


test_pred = Predictor.predict(df_test,model = best_model)


df = pd.DataFrame({
    'id': ID,  
    'price': test_pred            
})

# Display the DataFrame
print(df.head())


df.to_csv('prediction_WeightedEnsemble.csv', index=False)

