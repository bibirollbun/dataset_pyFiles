!pip install -q --no-deps scikit_learn==1.3.1


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures, MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, BayesianRidge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from colorama import Style, Fore
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import KFold
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib.ticker import MaxNLocator
from scipy.stats import skew
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split



train = pd.read_csv('/kaggle/input/computer-prices-2025/computer_prices_all.csv',index_col='ID')
test =  pd.read_csv('/kaggle/input/computer-prices-2025/computer_prices_test.csv',index_col='ID')


display(train.shape, test.shape)


fig, axes = plt.subplots(1, 2, figsize=(10, 3)) 
train['price'].hist(ax=axes[0])
axes[0].set_title('Price')
axes[0].set_ylabel('Freq')
axes[0].set_xlabel('Price')

np.log(train['price']).hist(ax=axes[1])
axes[1].set_title('Log(Price)')
axes[1].set_ylabel('Freq')
axes[1].set_xlabel('Log(Price)');
plt.suptitle('Target',fontsize=20, y=1.06, fontweight='bold');



cols_to_drop = [
    'bluetooth',
    'warranty_months',
    'storage_drive_count',
    'wifi',
    'model',
    'cpu_model',
    'display_size_in',
    'weight_kg',
    'cpu_boost_ghz',
    'battery_wh'
]
for df in [train,test]:
    df['brand'] = df['model'].str.split().str[0]
    #df['model_line'] = df['model'].str.split().str[1]
    #df['model_code'] = df['model'].str.split().str[-1]    
    df['cpu_brand'] = df['cpu_model'].str.split().str[0]
    #df['cpu_family'] = df['cpu_model'].str.split().str[1]
    #df['cpu_gen_code'] = df['cpu_model'].str.split().str[-1]    
    df.drop(cols_to_drop,axis=1,inplace=True)


cat_features = list(train.select_dtypes('object').columns)
TARGET = 'price'

high_card = []
low_card = []

print(f" {Fore.GREEN} Categorical features: {Fore.BLACK} {cat_features} ")
for col in cat_features:
    dtype = pd.CategoricalDtype(list(set(train[col]).union(set(test[col]))))
    train[col] = train[col].astype(dtype)
    test[col] = test[col].astype(dtype)
    if train[col].nunique() >= 15:
        high_card.append(col)
    else:
        low_card.append(col)
        
num_features = list(test.select_dtypes('int').columns) + list(test.select_dtypes('float').columns)
num_features = [f for f in num_features if f not in ['price']]

print(f' {Fore.BLUE} Numeric features: {Fore.BLACK}{num_features}')
features = list(test.columns)


column_transformer_h = ColumnTransformer(
    [('te', TargetEncoder(target_type='continuous', random_state=42), high_card)],
    verbose_feature_names_out=False,
    remainder='passthrough'
).set_output(transform='pandas')
column_transformer_l = ColumnTransformer(
    [('ohe', OneHotEncoder(sparse_output=False), low_card)],
    verbose_feature_names_out=False,
    remainder='passthrough'
).set_output(transform='pandas')
skewed_cols = train[num_features].apply(skew).sort_values(ascending=False)
skewed_pos = skewed_cols[skewed_cols > 0].index.tolist()
log_transformer = FunctionTransformer(np.log1p, feature_names_out='one-to-one')
column_transformer_skew = ColumnTransformer(
    [('log', log_transformer, skewed_pos)],
    verbose_feature_names_out=False,
    remainder='passthrough' 
).set_output(transform='pandas')


class InverseFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols):
        self.numerical_cols = numerical_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.numerical_cols:
            new_col_name = f'{col}_inverse'            
            X_copy[new_col_name] = np.where(X_copy[col] != 0, 1 / X_copy[col], 0)
        return X_copy


%%time
pt = PowerTransformer(method='box-cox')
model = TransformedTargetRegressor(
            make_pipeline(                    
            column_transformer_h,
            column_transformer_l, 
            column_transformer_skew,            
            MinMaxScaler(),            
            PolynomialFeatures(interaction_only=False,degree=2),                  
            Ridge()),                        
            transformer=pt        
         )
print(-np.mean(cross_val_score(model, X=train.drop('price',axis=1),y=train.price, scoring='neg_root_mean_squared_error')))


model.fit(train.drop('price',axis=1),train.price)
y_pred = model.predict(test)


sub = pd.read_csv('/kaggle/input/computer-prices-2025/sample_submission.csv')
sub['price'] = y_pred
sub.to_csv('submission.csv',index=False)

