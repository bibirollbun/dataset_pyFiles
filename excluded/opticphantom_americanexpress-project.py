# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: /kaggle/input/amex-default-prediction/train_data.csv
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library
import seaborn as sns # library for advanced plotting
from tqdm import tqdm # to make use of progress bar
from IPython.display import clear_output
import gc # for calling python garbage collection or memory handling

import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_parquet('/kaggle/input/amex-parquet/train_data.parquet')
print(f'Loaded: {df.index.size} rows')


df['S_2']=pd.to_datetime(df['S_2']) # to convert date data into datetime type


# to remove collumns with more than 20% of the data null
columns_toremove=[]

for col in df.columns:
    null_percent=round((df[col].isna().sum()/df.index.size)*100,2)
    clear_output(wait=True)
    if(null_percent>=20):
        columns_toremove.append(col)
        print(f'Column ({col}) : Null Percent: {null_percent}% ')
    
df=df.drop(columns_toremove,axis=1)
print(f'Deleted {len(columns_toremove)} columns')


# Extract Numerical and Categorical Columns
target_columns=[c for c in df.columns if c not in ['customer_ID','S_2','target']] # taking all columns except customer id and S_2 (contains date data)
cat_features=['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_68'] # contains all categorical features (taken from data card)
num_features=[c for c in target_columns if c not in cat_features] # to get all numerical features
print('Column Extraction Complete')

#notes:
# D_66 was removed from the cat_features list as it was removed at the null removing part of the code


# Aggregating the Numerical Features based on Customer ID
num_agg=df.groupby(['customer_ID'])[num_features].agg(['mean','max','last'])
num_agg.columns=['_'.join(x) for x in num_agg.columns]
num_agg=num_agg.reset_index(drop=False)
print('Completed Operation!')


# Converting the customer_ID feature from object to string for efficient memory merge

#to remove all the rows with any nulls in any of the features from the df dataframe
print('Deleting Rows with any null in features...(for df dataframe)')
df=df[df[df.columns[1:]].isnull().all(axis=1)==False]

df = df[['customer_ID','target'] + cat_features].dropna().groupby('customer_ID').last().reset_index()
df['customer_ID']=df['customer_ID'].astype('category')

df[cat_features]=df[cat_features].astype('category')

gc.collect()
print(f'Rows After: {df.index.size}')
print(f'Columns: {df.columns.size}')

#to remove all the rows with any nulls in any of the features from the num_agg dataframe
print('\nDeleting Rows with any null in features...(for num_agg dataframe)')
print(f'Rows Before: {num_agg.index.size}')
num_agg['customer_ID']=num_agg['customer_ID'].astype('category') #convert customer_ID from object to string for efficnet memory use
gc.collect()
print(f'Rows After: {num_agg.index.size}')
print(f'Columns: {num_agg.columns.size}')

print('Row clean-up complete!')


# To merge the numerical, categorical and target features into the same dataframe
df=num_agg.merge(df,on='customer_ID',how='inner')
print(f'Tables have been merged {df.index.size}')

#For clearing memory
del num_agg
gc.collect()


print(f"""
Summary (df)
Rows: {df.index.size}
Cols: {df.columns.size} 
""")
df.head(3)


    #official evaluation metric function drom the compition hosters
    def amex_metric(y_pred, train_data) -> tuple:
        
        y_true=train_data.get_label()
        y_true=pd.DataFrame({'target':y_true})
        
        y_pred=pd.DataFrame({'prediction':y_pred})
        
        # print(y_true.head())
        # print(y_pred.head())
        # raise Exception('Force breaker')
        
        def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
            df = (pd.concat([y_true, y_pred], axis='columns')
                  .sort_values('prediction', ascending=False))
            df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
            four_pct_cutoff = int(0.04 * df['weight'].sum())
            df['weight_cumsum'] = df['weight'].cumsum()
            df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
            return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
            
        def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
            df = (pd.concat([y_true, y_pred], axis='columns')
                  .sort_values('prediction', ascending=False))
            df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
            df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
            total_pos = (df['target'] * df['weight']).sum()
            df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
            df['lorentz'] = df['cum_pos_found'] / total_pos
            df['gini'] = (df['lorentz'] - df['random']) * df['weight']
            return df['gini'].sum()
    
        def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
            y_true_pred = y_true.rename(columns={'target': 'prediction'})
            return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)
    
        g = normalized_weighted_gini(y_true, y_pred)
        d = top_four_percent_captured(y_true, y_pred)
    
        return 'amex_metric',0.5 * (g + d),True
    
    print('Created Official Evaluation Metric Function!')


import lightgbm as lgb
from sklearn.model_selection import train_test_split

print('Starting model building process')

X=df.drop(columns=['customer_ID','target'])
y=df['target']

xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=45,test_size=0.2,stratify=y)

train_data=lgb.Dataset(xtrain,label=ytrain,categorical_feature=cat_features)
test_data=lgb.Dataset(xtest,label=ytest,categorical_feature=cat_features)

params={
    'objective':'binary',
    'metric':'None',
    'boosting_type':'gbdt',
    'num_leaves':31,
    'learning_rate':0.05,
    'feature_fraction':0.9,
    'bagging_fraction':0.8,
    'bagging_freq':5
}

print('Starting Training')
model=lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)
    ],
    feval=amex_metric
)

print('Training Complete!')

# model.save_model('lgb_modelAMEX.txt')


model.save_model('lgb_modelAMEX.txt')
print('Model saved!')



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library
import seaborn as sns # library for advanced plotting
from tqdm import tqdm # to make use of progress bar
from IPython.display import clear_output
import gc # for calling python garbage collection or memory handling

import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import lightgbm as lgb
# model=lgb.Booster(model_file='/kaggle/input/lightgbm_amex_defaulter_predictor/AE_Defaulter_Model.cbm')
model=lgb.Booster(model_file='/kaggle/working/lgb_modelAMEX.txt')


#Import the Data
testdata_df=pd.read_parquet('/kaggle/input/amex-parquet/test_data.parquet')

testdata_df['S_2']=pd.to_datetime(testdata_df['S_2']) # to convert date data into datetime
print('Operation Complete!')

# to remove columns with more than 20% of the data null
columns_toremove=[]

for col in testdata_df.columns:
    null_percent=round((testdata_df[col].isna().sum()/testdata_df.index.size)*100,2)
    clear_output(wait=True)
    if(null_percent>=20):
        columns_toremove.append(col)
        print(f'Column ({col}) : Null Percent: {null_percent}% ')
    
testdata_df=testdata_df.drop(columns_toremove,axis=1)
print(f'Deleted {len(columns_toremove)} columns')


# Extract Numerical and Categorical Columns
target_columns=[c for c in testdata_df.columns if c not in ['customer_ID','S_2']] # taking all columns except customer id and S_2 (contains date data)
cat_features=['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_68'] # contains all categorical features (taken from data card)
num_features=[c for c in target_columns if c not in cat_features] # to get all numerical features
print('Column Extraction Complete')


#notes:
# D_66 was removed from the cat_features list as it was removed at the null removing part of the code

# Aggregating the Numerical Features based on Customer ID
num_agg=testdata_df.groupby(['customer_ID'])[num_features].agg(['mean','max','last'])
num_agg.columns=['_'.join(x) for x in num_agg.columns]
num_agg=num_agg.reset_index(drop=False) #fix indexing issue
print('Completed Operation!')




# Converting the customer_ID feature from object to string for efficient memory merge

#Additional row operations (convert to int & category type for memory efficiency, remove any rows with null and remove duplicates)
print('Deleting null rows in features...(for testdata_df dataframe)')

# PROBLEM IS HERE
test_df=testdata_df[testdata_df[testdata_df.columns[1:]].isnull().all(axis=1)==False] #new code

testdata_df = testdata_df[['customer_ID'] + cat_features].dropna().groupby('customer_ID').last().reset_index()
testdata_df['customer_ID']=testdata_df['customer_ID'].astype('category') #convert customer_ID from object to string for efficient memory use
# raise Exception('force breaker')

testdata_df[cat_features]=testdata_df[cat_features].astype('category')

gc.collect()
print(f'Rows After: {testdata_df.index.size}')
print(f'Columns: {testdata_df.columns.size}')



#to remove all the rows with any nulls in any of the features from the num_agg dataframe
print('\nDeleting Rows with any null in features...(for num_agg dataframe)')
print(f'Rows Before: {num_agg.index.size}')

num_agg['customer_ID']=num_agg['customer_ID'].astype('category') #convert customer_ID from object to string for efficnet memory use
gc.collect()
print(f'Rows After: {num_agg.index.size}')
print(f'Columns: {num_agg.columns.size}')

print('Row clean-up complete!')

# To merge the numerical, categorical and target features into the same dataframe
testdata_df=num_agg.merge(testdata_df,on='customer_ID',how='inner')

#For clearing memory
del num_agg
gc.collect()

#Running model
X=testdata_df.drop(columns=['customer_ID'])

# test_data=lgb.Dataset(X,categorical_feature=cat_features)
print('prediction')
y_pred=model.predict(X,num_iteration=model.best_iteration)
print('Prediction Complete!')



submission_data=pd.DataFrame({'customer_ID':testdata_df['customer_ID'],'prediction':np.round(y_pred,2)})
submission_data.to_csv('submission.csv',index=False)
print('Saved submission.csv file.')

