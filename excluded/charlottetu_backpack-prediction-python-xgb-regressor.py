# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.pyplot as py
import seaborn as sns
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from statistics import mean
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import shap


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
y_df = train_df['Price']


#Identify the columns
train_df.info()


#Classify the columns

num_cols = [colname for colname in train_df.columns if train_df[colname].dtype in ['int64','float64']]
cat_cols = [colname for colname in train_df.columns if (train_df[colname].dtype in ['object']) ]
print(num_cols)
print(cat_cols)


#Finding out how many distint items there are and filtering them
for i in cat_cols:
    print (f'Value Count for {i}')
    print(train_df[i].value_counts())
    print('_'*20)


#Analyse the categorical data
for i in cat_cols:
    fig , axes = plt.subplots(1,2, figsize=(10,6))
    sns.countplot(train_df , x=i, ax= axes[0])
    sns.boxplot(train_df, x = i, y = y_df, ax= axes[1])
    plt.show()


#Analyse distribution
fig = py.figure(figsize = (18,16))
for index,col in enumerate(num_cols[:19]):
    py.subplot(5,4,index+1)
    sns.distplot(train_df.loc[:,col].dropna())
fig.tight_layout(pad = 1.0)


#Analyse correlation
dfnumerical = train_df[num_cols]
correlation = dfnumerical.corr()
correlation['Price'].sort_values(ascending = False)

fig4 = sns.heatmap(dfnumerical.corr())
sns.set(rc = {'figure.figsize':(40,30)})


correlation


#Number of blanks by column
train_df.isna().sum()


train_df.head()


#Feature engineering
train_df['Leather'] = train_df['Material'].apply(lambda x:1 if x == 'Leather' else 0) 
test_df['Leather'] = test_df['Material'].apply(lambda x:1 if x == 'Leather' else 0) 

train_df['Good colour'] = train_df['Color'].apply(lambda x: 1 if x in ['Green', 'Blue'] else 0)
test_df['Good colour'] = test_df['Color'].apply(lambda x: 1 if x in ['Green', 'Blue'] else 0)

train_df['Canvas'] = train_df['Material'].apply(lambda x: 1 if x == 'Canvas' else 0)
test_df['Canvas'] = test_df['Material'].apply(lambda x: 1 if x == 'Canvas' else 0)


train_df


print(num_cols)
print(cat_cols)


#Columns to include
num_cols_inc = ['Compartments', 'Weight Capacity (kg)', 'Leather', 'Good colour', 'Canvas']
cat_cols_inc = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color', ]
all_cols_inc = num_cols_inc + cat_cols_inc
train_df_filtered = train_df[all_cols_inc]
test_df_filtered = test_df[all_cols_inc]


for i in cat_cols_inc:
    train_df_filtered[i] = train_df_filtered[i].astype('category')
    test_df_filtered[i] = test_df_filtered[i].astype('category')




train_df_filtered.info()


#Split into train and validation
#x_train, x_val, y_train, y_val = train_test_split(train_df_filtered, y_df, random_state=4)

x_train = train_df_filtered
y_train = y_df

#Final test data set
test_df_final = test_df_filtered


cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)


model = XGBRegressor(enable_categorical = True)


scores = cross_val_score(model, x_train, y_train, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
print(scores.mean()*-1)


model.fit(x_train,y_train)


model.predict(test_df_final)



explainer = shap.Explainer(model)



#shap_values = explainer(x_train)
#shap.summary_plot(shap_values, x_train)
#shap.waterfall_plot(shap_values[0]) 


submission = pd.DataFrame(data = test_df['id'], index = None, columns = ['id'])
submission['Price'] = model.predict(test_df_final)
submission.to_csv("submission.csv", index=False)


submission.head()




