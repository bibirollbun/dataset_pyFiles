# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


df1 = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df1


df2 = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
df2


df = pd.concat([df1,df2])
df


df.info()


df.isna().sum()


df = df.dropna()
df.isna().sum()


df.columns


import scipy.stats as stats

lis = ['Brand', 'Material', 'Size', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color']
signi = []

for cate in lis: 

    grouped_prices = [df[df[cate] == cat]["Price"] for cat in df[cate].unique()]
    
    f_stat, p_value = stats.f_oneway(*grouped_prices)
    
    print('-----------------', cate,'-----------------' )
    print(f"F-statistic: {f_stat:.2f}")
    print(f"P-value: {p_value:.4f}")
    
    # Interpretation
    if p_value < 0.05:
        signi.append(cate)
        print("Significant difference in price across categories.")
    else:
        print("No significant difference in price across categories.")

print("Categories that are significant difference:",signi)


df.corr(numeric_only = True)


signi.append('Weight Capacity (kg)')


df_post = df[signi]
df_post


df_post = pd.get_dummies(df_post,dtype=int)
df_post


x,y = df_post, df[['Price']]
print(x.shape, y.shape)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=1,test_size=0.2)


import xgboost as xgb
xgb1 = xgb.XGBRegressor(objective ='reg:squarederror')


%%time
xgb1.fit(x_train,y_train)


y_pred = xgb1.predict(x_test)


#Compute rmse
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
score = (mse)**(1/2)
score


from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [.03,.05], 
    'max_depth': [5,6],
    'min_child_weight': [4,5],
    'n_estimators': [300,400]
}

xgb_grid = GridSearchCV(xgb1,param_grid,cv = 2,n_jobs = -1,verbose=True)


%%time
xgb_grid.fit(x_train,
         y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


model =  xgb_grid.best_estimator_
y_pred2 = model.predict(x_test)


mse = mean_squared_error(y_test, y_pred2)
score = (mse)**(1/2)
score


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test


test = test[signi]
test


test = pd.get_dummies(test, dtype=int)
test


sub = model.predict(test)
sub


sub_df = pd.DataFrame(sub)
sub_df


sub_df.index = sub_df.index + 300000
sub_df = sub_df.reset_index(names='id')
sub_df


sub_df = sub_df.rename(columns={0: 'Price'})
sub_df


sub_df.to_csv('submission.csv',index=False)




