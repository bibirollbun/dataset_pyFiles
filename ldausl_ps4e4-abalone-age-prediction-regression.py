                                                            import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"
import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
tqdm_notebook.get_lock().locks = []
from prettytable import PrettyTable
%matplotlib inline
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
from copy import deepcopy
from functools import partial
from itertools import combinations

from sklearn.cluster import KMeans
# !pip install yellowbrick
# from yellowbrick.cluster import KElbowVisualizer

import random
from random import uniform
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_squared_log_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler,PowerTransformer, FunctionTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
import statsmodels.api as sm
import math
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
!pip install optuna
!pip install cmaes
import optuna
import xgboost as xgb
# !pip install catboost
!pip install lightgbm --install-option=--gpu --install-option="--boost-root=C:/local/boost_1_69_0" --install-option="--boost-librarydir=C:/local/boost_1_69_0/lib64-msvc-14.1"
import lightgbm as lgb
!pip install category_encoders
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder, CatBoostEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression, RidgeCV, ElasticNetCV
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor,GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoost, CatBoostRegressor,CatBoostClassifier
from catboost import Pool
from sklearn.neighbors import KNeighborsRegressor
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
pd.pandas.set_option('display.max_columns',None)


train=pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')
submission=pd.read_csv("/kaggle/input/playground-series-s4e4/sample_submission.csv")

original=pd.read_csv("/kaggle/input/playgrounds4e04originaldata/Original.csv")

train_copy=train.copy()
test_copy=test.copy()

# Tag Orignal
original["original"]=1
train["original"]=0
test["original"]=0

train.drop(columns=["id"],inplace=True)
test.drop(columns=["id"],inplace=True)
original.drop(columns=["id"],inplace=True)

train=train.rename(columns={'Whole weight':'Whole_weight','Whole weight.1':'Shucked_weight', 'Whole weight.2':'Viscera_weight', 'Shell weight':'Shell_weight'})
test=test.rename(columns={'Whole weight':'Whole_weight','Whole weight.1':'Shucked_weight', 'Whole weight.2':'Viscera_weight', 'Shell weight':'Shell_weight'})

train=pd.concat([train,original],axis='rows')
train.reset_index(inplace=True,drop=True)

device='cpu'

target='Rings'

train.head()


table = PrettyTable()

table.field_names = ['Column Name', 'Data Type', 'Train Missing %', 'Test Missing %']
for column in train.columns:
    data_type = str(train[column].dtype)
    non_null_count_train= 100-train[column].count()/train.shape[0]*100
    if column!=target:
        non_null_count_test = 100-test[column].count()/test.shape[0]*100
    else:
        non_null_count_test="NA"
    table.add_row([column, data_type, non_null_count_train,non_null_count_test])
print(table)


class_0 = train[train['original'] == 0][target]
class_1 = train[train['original'] == 1][target]

mean_0 = np.mean(class_0)
median_0 = np.median(class_0)
mean_1 = np.mean(class_1)
median_1 = np.median(class_1)

fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(class_0, bins=20, density=True, alpha=0.5, label='Original=0 Histogram')
ax.hist(class_1, bins=20, density=True, alpha=0.5, label='Original=1 Histogram')

x_values_0 = np.linspace(class_0.min(), class_0.max(), len(class_0))
density_values_0 = (1 / (np.sqrt(2 * np.pi) * np.std(class_0))) * np.exp(-0.5 * ((x_values_0 - mean_0) / np.std(class_0))**2)
ax.plot(x_values_0, density_values_0, color='red', label='Original=0 Density')

x_values_1 = np.linspace(class_1.min(), class_1.max(), len(class_1))
density_values_1 = (1 / (np.sqrt(2 * np.pi) * np.std(class_1))) * np.exp(-0.5 * ((x_values_1 - mean_1) / np.std(class_1))**2)
ax.plot(x_values_1, density_values_1, color='green', label='Original=1 Density')

ax.axvline(mean_0, color='blue', linestyle='dashed', linewidth=2, label='Mean (Original=0)')
ax.axvline(median_0, color='green', linestyle='dashed', linewidth=2, label='Median (Original=0)')
ax.axvline(mean_1, color='blue', linestyle='dashed', linewidth=2, label='Mean (Original=1)')
ax.axvline(median_1, color='red', linestyle='dashed', linewidth=2, label='Median (Original=1)')

ax.set_xlabel(target)
ax.set_ylabel('Frequency / Density')
ax.set_title('Histograms and Density Plots')

x_min = min(min(class_0), min(class_1))
x_max = max(max(class_0), max(class_1))
ax.set_xlim([x_min, x_max])

ax.legend(bbox_to_anchor=(1,1),fancybox=False,shadow=False, loc='upper left')

plt.tight_layout()
plt.show()



class_0 = np.log1p(train[train['original'] == 0][target])
class_1 = np.log1p(train[train['original'] == 1][target])

mean_0 = np.mean(class_0)
median_0 = np.median(class_0)
mean_1 = np.mean(class_1)
median_1 = np.median(class_1)

fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(class_0, bins=20, density=True, alpha=0.5, label='Original=0 Histogram')
ax.hist(class_1, bins=20, density=True, alpha=0.5, label='Original=1 Histogram')

x_values_0 = np.linspace(class_0.min(), class_0.max(), len(class_0))
density_values_0 = (1 / (np.sqrt(2 * np.pi) * np.std(class_0))) * np.exp(-0.5 * ((x_values_0 - mean_0) / np.std(class_0))**2)
ax.plot(x_values_0, density_values_0, color='red', label='Original=0 Density')

x_values_1 = np.linspace(class_1.min(), class_1.max(), len(class_1))
density_values_1 = (1 / (np.sqrt(2 * np.pi) * np.std(class_1))) * np.exp(-0.5 * ((x_values_1 - mean_1) / np.std(class_1))**2)
ax.plot(x_values_1, density_values_1, color='green', label='Original=1 Density')

ax.axvline(mean_0, color='blue', linestyle='dashed', linewidth=2, label='Mean (Original=0)')
ax.axvline(median_0, color='green', linestyle='dashed', linewidth=2, label='Median (Original=0)')
ax.axvline(mean_1, color='blue', linestyle='dashed', linewidth=2, label='Mean (Original=1)')
ax.axvline(median_1, color='red', linestyle='dashed', linewidth=2, label='Median (Original=1)')

ax.set_xlabel(target)
ax.set_ylabel('Frequency / Density')
ax.set_title('Log Transformed Histograms and Density Plots')

x_min = min(min(class_0), min(class_1))
x_max = max(max(class_0), max(class_1))
ax.set_xlim([x_min, x_max])

ax.legend(bbox_to_anchor=(1,1),fancybox=False,shadow=False, loc='upper left')

plt.tight_layout()
plt.show()



cont_cols=[f for f in train.columns if train[f].dtype in [float,int] and train[f].nunique()>2 and f not in [target]]

# Calculate the number of rows needed for the subplots
num_rows = (len(cont_cols) + 2) // 3

# Create subplots for each continuous column
fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows*5))

# Loop through each continuous column and plot the histograms
for i, col in enumerate(cont_cols):
    # Determine the range of values to plot
    max_val = max(train[col].max(), test[col].max(), original[col].max())
    min_val = min(train[col].min(), test[col].min(), original[col].min())
    range_val = max_val - min_val
    
    # Determine the bin size and number of bins
    bin_size = range_val / 20
    num_bins_train = round(range_val / bin_size)
    num_bins_test = round(range_val / bin_size)
    num_bins_original = round(range_val / bin_size)
    
    # Calculate the subplot position
    row = i // 3
    col_pos = i % 3
    
    # Plot the histograms
    sns.histplot(train[col], ax=axs[row][col_pos], color='orange', kde=True, label='Train', bins=num_bins_train)
    sns.histplot(test[col], ax=axs[row][col_pos], color='green', kde=True, label='Test', bins=num_bins_test)
    sns.histplot(original[col], ax=axs[row][col_pos], color='blue', kde=True, label='Original', bins=num_bins_original)
    axs[row][col_pos].set_title(col)
    axs[row][col_pos].set_xlabel('Value')
    axs[row][col_pos].set_ylabel('Frequency')
    axs[row][col_pos].legend()

# Remove any empty subplots
if len(cont_cols) % 3 != 0:
    for col_pos in range(len(cont_cols) % 3, 3):
        axs[-1][col_pos].remove()

plt.tight_layout()
plt.show()


sns.pairplot(data=original, vars=cont_cols+[target], hue='Sex')
plt.show()


plt.subplots(figsize=(16, 5))
sns.violinplot(x='Sex', y=col, data=train)
plt.title('Rings Distribution by Sex', fontsize=14)
plt.xlabel('Sex', fontsize=12)
plt.ylabel('Rings', fontsize=12)
sns.despine()
fig.tight_layout()
plt.show()


features=[f for f in test.columns if f!='Sex']
corr = train[features].corr()
plt.figure(figsize = (8, 6), dpi = 300)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, cmap = 'magma', annot = True, annot_kws = {'size' : 6})
plt.title('Features Correlation Matrix\n', fontsize = 15, weight = 'bold')
plt.show()


def min_max_scaler(train, test, column):
    '''
    Min Max just based on train might have an issue if test has extreme values, hence changing the denominator uding overall min and max
    '''
    sc=MinMaxScaler()
    
    max_val=max(train[column].max(),test[column].max())
    min_val=min(train[column].min(),test[column].min())

    train[column]=(train[column]-min_val)/(max_val-min_val)
    test[column]=(test[column]-min_val)/(max_val-min_val)
    
    return train,test  

def rmse(y1,y2):
    ''' RMSE Evaluator'''
    return(np.sqrt(mean_squared_error(np.array(y1),np.array(y2))))

def nearest(y_predicted):
    
    y_original=y_unique
    modified_prediction = np.zeros_like(y_predicted)

    for i, y_pred in enumerate(y_predicted):
        nearest_value = min(y_original, key=lambda x: abs(x - y_pred))
        modified_prediction[i] = nearest_value

    return modified_prediction

global y_unique
y_unique=train[target].unique()
y_unique_log=np.log1p(train[target]).unique()

xgb_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.0116,
            'colsample_bytree': 1,
            'subsample': 0.6085,
            'min_child_weight': 9,
            'reg_lambda': 4.879e-07,
            'max_bin': 431,
            'n_jobs': -1,
            'eval_metric': 'mae',
            'objective': "reg:absoluteerror",
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': 42,
        }

def fill_missing_numerical(train,test,target, max_iterations=10):
    '''Iterative Missing Imputer: Updates filled missing values iteratively using CatBoost Algorithm'''
    train_temp=train.copy()
    if target in train_temp.columns:
        train_temp=train_temp.drop(columns=target)
        
    
    df=pd.concat([train_temp,test],axis="rows")
    df=df.reset_index(drop=True)
    features=[ f for f in df.columns if df[f].isna().sum()>0]
    if len(features)>0:
        # Step 1: Store the instances with missing values in each feature
        missing_rows = store_missing_rows(df, features)

        # Step 2: Initially fill all missing values with "Missing"
#         for f in features:
#             df[f]=df[f].fillna(df[f].median())

        cat_features=[f for f in df.columns if not pd.api.types.is_numeric_dtype(df[f])]
        dictionary = {feature: [] for feature in features}

        for iteration in tqdm(range(max_iterations), desc="Iterations"):
            for feature in features:
#                 print(feature)
                # Skip features with no missing values
                rows_miss = missing_rows[feature].index
                replace_dict={}
                rev_replace_dict={}
                for col in  cat_features:
                    df[col]=df[col].astype(str)
                    int_cat=dict(zip(df[col].unique(),np.arange(0, df[col].nunique())))
                    rev_int_cat=dict(zip(np.arange(0, df[col].nunique()), df[col].unique()))
                    df[col]=df[col].replace(int_cat)
                
                    replace_dict[col]=int_cat
                    rev_replace_dict[col]=rev_int_cat

                missing_temp = df.loc[rows_miss].copy()
                non_missing_temp = df.drop(index=rows_miss).copy()
                y_pred_prev=missing_temp[feature]
                missing_temp = missing_temp.drop(columns=[feature])


                # Step 3: Use the remaining features to predict missing values using Random Forests
                X_train = non_missing_temp.drop(columns=[feature])
                y_train = non_missing_temp[[feature]]

#                 model1 = CatBoostRegressor(**cb_params)
#                 model1.fit(X_train, y_train,cat_features=cat_features, verbose=False)
                
                model2 = xgb.XGBRegressor(**xgb_params)
                model2.fit(X_train, y_train, verbose=False) 
                
                # Step 4: Predict missing values for the feature and update all N features
                y_pred = np.array(model2.predict(missing_temp))
                
                df.loc[rows_miss, feature] = y_pred
#                 error_minimize=rmse(y_pred,y_pred_prev) #mean_squared_error
                error_minimize=np.sqrt(mean_squared_error(y_pred,y_pred_prev) )#mean_squared_error
                dictionary[feature].append(error_minimize)  # Append the error_minimize value
                
                for col in  cat_features:
                    df[col]=df[col].replace(rev_int_cat)


        for feature, values in dictionary.items():
            values=np.array(values)/sum(values)
            iterations = range(1, len(values) + 1)  # x-axis values (iterations)
            plt.plot(iterations, values, label=feature)  # plot the values
            plt.xlabel('Iterations')
            plt.ylabel('RMSE')
            plt.title('Minimization of RMSE with iterations')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        train[features] = np.array(df.iloc[:train.shape[0]][features])
        test[features] = np.array(df.iloc[train.shape[0]:][features])

    return train,test


def new_features(data):
    df=data.copy()
    
    # Clean the weights by capping the over weights with total body weights
    df['Shell_weight']=np.where(df['Shell_weight']>df['Whole_weight'],df['Whole_weight'],df['Shell_weight'])
    df['Viscera_weight']=np.where(df['Viscera_weight']>df['Whole_weight'],df['Whole_weight'],df['Viscera_weight'])
    df['Shucked_weight']=np.where(df['Shucked_weight']>df['Whole_weight'],df['Whole_weight'],df['Shucked_weight'])
    
    # Abalone Surface area
    df["surface_area"]=df["Length"]*df["Diameter"]
    df['total_area']=2*(df["surface_area"]+df["Height"]*df["Diameter"]+df["Length"]*df["Height"])
    
    # Abalone density approx
    df['approx_density']=df['Whole_weight']/(df['surface_area']*df['Height']+1e-5)
    
    # Abalone BMI
    df['bmi']=df['Whole_weight']/(df['Height']**2+1e-5)
    
    # Measurement derived
    df["length_dia_ratio"]=df['Length']/(df['Diameter']+1e-5)
    df["length_height_ratio"]=df['Length']/(df['Height']+1e-5)
    df['shell_shuck_ratio']=df["Shell_weight"]/(df["Shucked_weight"]+1e-5)
    df['shell_viscera_ratio']=df['Shell_weight']/(df['Viscera_weight']+1e-5)
    
    df['viscera_tot_ratio']=df['Viscera_weight']/(df['Whole_weight']  +1e-5)
    df['shell_tot_ratio']=df['Shell_weight']/(df['Whole_weight']    +1e-5)
    df['shuck_tot_ratio']=df['Shucked_weight']/(df['Whole_weight']   +1e-5)
    df['shell_body_ratio']=df['Shell_weight']/(df['Shell_weight']+df['Whole_weight']+1e-5)
    df['flesh_ratio']=df['Shucked_weight']/(df['Whole_weight']+df['Shucked_weight']+1e-5)
    
    df['inv_viscera_tot']= df['Whole_weight'] / (df['Viscera_weight']+1e-5)
    df['inv_shell_tot']= df['Whole_weight'] /( df['Shell_weight']+1e-5)
    df['inv_shuck_tot']= df['Whole_weight'] / (df['Shucked_weight']+1e-5)
    
    
    # Water Loss during experiment
    df["water_loss"]=df["Whole_weight"]-df["Shucked_weight"]-df['Viscera_weight']-df['Shell_weight']
    df["water_loss"]=np.where(df["water_loss"]<0,min(df["Shucked_weight"].min(),df["Viscera_weight"].min(),df["Shell_weight"].min()),df["water_loss"])
    return df

train=new_features(train)
test=new_features(test)
original=new_features(original)


cont_cols = [f for f in train.columns if train[f].dtype != 'O' and train[f].nunique()>10000]

sc=MinMaxScaler()

global unimportant_features
global overall_best_score
global overall_best_col
unimportant_features=[]
overall_best_score=1e5
overall_best_col='none'

# for col in cont_cols:
#      train, test=min_max_scaler(train, test, col)

def transformer(train, test,cont_cols, target):
    '''
    Algorithm applies multiples transformations on selected columns and finds the best transformation using a single variable model performance
    '''
    global unimportant_features
    global overall_best_score
    global overall_best_col
    train_copy = train.copy()
    test_copy = test.copy()
    table = PrettyTable()
    table.field_names = ['Feature', 'Original RMSLE', 'Transformation', 'Tranformed RMSLE']

    for col in cont_cols:
        train_copy, test_copy=min_max_scaler(train_copy, test_copy, col)
        for c in ["log_"+col, "sqrt_"+col, "bx_cx_"+col, "y_J_"+col, "log_sqrt"+col, "pow_"+col, "pow2_"+col]:
            if c in train_copy.columns:
                train_copy = train_copy.drop(columns=[c])
        
        # Log Transformation after MinMax Scaling (keeps data between 0 and 1)
        train_copy["log_"+col] = np.log1p(train_copy[col])
        test_copy["log_"+col] = np.log1p(test_copy[col])
        
        # Square Root Transformation
        train_copy["sqrt_"+col] = np.sqrt(train_copy[col])
        test_copy["sqrt_"+col] = np.sqrt(test_copy[col])
        
        # Box-Cox transformation
        combined_data = pd.concat([train_copy[[col]], test_copy[[col]]], axis=0)
        epsilon = 1e-5
        transformer = PowerTransformer(method='box-cox')
#         scaled_data = transformer.fit_transform(combined_data + epsilon)

#         train_copy["bx_cx_" + col] = scaled_data[:train_copy.shape[0]]
#         test_copy["bx_cx_" + col] = scaled_data[train_copy.shape[0]:]
        train_copy["bx_cx_" + col] = transformer.fit_transform(train_copy[[col]]+epsilon)
        test_copy["bx_cx_" + col] = transformer.transform(test_copy[[col]]+epsilon)
        # Yeo-Johnson transformation
        transformer = PowerTransformer(method='yeo-johnson')
        train_copy["y_J_"+col] = transformer.fit_transform(train_copy[[col]])
        test_copy["y_J_"+col] = transformer.transform(test_copy[[col]])
        
        # Power transformation, 0.25
        power_transform = lambda x: np.power(x + 1 - np.min(x), 0.25)
        transformer = FunctionTransformer(power_transform)
        train_copy["pow_"+col] = transformer.fit_transform(train_copy[[col]])
        test_copy["pow_"+col] = transformer.transform(test_copy[[col]])
        
        # Power transformation, 2
        power_transform = lambda x: np.power(x + 1 - np.min(x), 2)
        transformer = FunctionTransformer(power_transform)
        train_copy["pow2_"+col] = transformer.fit_transform(train_copy[[col]])
        test_copy["pow2_"+col] = transformer.transform(test_copy[[col]])
        
        # Log to power transformation
        train_copy["log_sqrt"+col] = np.log1p(train_copy["sqrt_"+col])
        test_copy["log_sqrt"+col] = np.log1p(test_copy["sqrt_"+col])
        
        temp_cols = [col, "log_"+col, "sqrt_"+col, "bx_cx_"+col, "y_J_"+col,  "pow_"+col , "pow2_"+col,"log_sqrt"+col]
        
        train_copy,test_copy = fill_missing_numerical(train_copy,test_copy,target,3)
#         train_copy[temp_cols] = train_copy[temp_cols].fillna(0)
#         test_copy[temp_cols] = test_copy[temp_cols].fillna(0)
        
        pca = TruncatedSVD(n_components=1)
        x_pca_train = pca.fit_transform(train_copy[temp_cols])
        x_pca_test = pca.transform(test_copy[temp_cols])
        x_pca_train = pd.DataFrame(x_pca_train, columns=[col+"_pca_comb"])
        x_pca_test = pd.DataFrame(x_pca_test, columns=[col+"_pca_comb"])
        temp_cols.append(col+"_pca_comb")
        
        test_copy = test_copy.reset_index(drop=True)
        
        train_copy = pd.concat([train_copy, x_pca_train], axis='columns')
        test_copy = pd.concat([test_copy, x_pca_test], axis='columns')
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        rmse_scores = []
        
        for f in temp_cols:
            X = train_copy[[f]].values
            y = train_copy[target].values
            
            rmses = []
            for train_idx, val_idx in kf.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                x_val, y_val = X[val_idx], y[val_idx]
                model=LinearRegression()
                model.fit(X_train,np.log1p(y_train))
                y_pred=nearest(np.expm1(model.predict(x_val)))
                rmses.append(rmse(np.log1p(y_val),np.log1p(y_pred)))
            rmse_scores.append((f,np.mean(rmses)))
            
            if overall_best_score > np.mean(rmses):
                overall_best_score = np.mean(rmses)
                overall_best_col = f

            if f == col:
                orig_rmse = np.mean(rmses)
                
        best_col, best_rmse = sorted(rmse_scores, key=lambda x: x[1], reverse=False)[0]
        cols_to_drop = [f for f in temp_cols if f != best_col]
        final_selection = [f for f in temp_cols if f not in cols_to_drop]
        
        if cols_to_drop:
            unimportant_features = unimportant_features+cols_to_drop
        table.add_row([col,orig_rmse,best_col ,best_rmse])
    print(table)   
    print("overall best CV RMSLE score: ",overall_best_score)
    return train_copy, test_copy

train, test= transformer(train, test,cont_cols, target)
train, test=fill_missing_numerical(train,test,target, max_iterations=3)


table = PrettyTable()
table.field_names = ['Cluster WOE Feature', 'RMSLE (CV-TRAIN)']
for col in cont_cols:
    sub_set=[f for f in unimportant_features if col in f]
#     print(sub_set)
    temp_train=train[sub_set]
    temp_test=test[sub_set]
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    model = KMeans()

    # print(ideal_clusters)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(np.array(temp_train))
    labels_train = kmeans.labels_

    train[col+"_unimp_cluster_WOE"] = labels_train
    test[col+"_unimp_cluster_WOE"] = kmeans.predict(np.array(temp_test))
    
    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    
    X=train[[col+"_unimp_cluster_WOE"]].values
    y=train[target].values

    best_rmse=[]
    for train_idx, val_idx in kf.split(X,y):
        X_train,y_train=X[train_idx],y[train_idx]
        x_val,y_val=X[val_idx],y[val_idx]
        model=LinearRegression()
        model.fit(X_train,np.log1p(y_train))
        y_pred=nearest(np.expm1(model.predict(x_val)))
        best_rmse.append(rmse(np.log1p(y_val),np.log1p(y_pred)))
        
    table.add_row([col+"_unimp_cluster_WOE",np.mean(best_rmse)])
    if overall_best_score<np.mean(best_rmse):
            overall_best_score=np.mean(best_rmse)
            overall_best_col=col+"_unimp_cluster_WOE"
    
print(table)


cat_cols = [f for f in test.columns if (train[f].dtype != 'O' and train[f].nunique()<2000 and train[f].nunique()>2 and "WOE" not in f) or (train[f].dtype == 'O') ]
print(train[cat_cols].nunique())


def nearest_val(target):
    return min(common, key=lambda x: abs(x - target))

cat_cols_updated=['Sex']
for col in cat_cols:
    if train[col].dtype!="O":
        train[f"{col}_cat"]=train[col]
        test[f"{col}_cat"]=test[col]
        cat_cols_updated.append(f"{col}_cat")
        uncommon=list((set(test[col].unique())| set(train[col].unique()))-(set(test[col].unique())& set(train[col].unique())))
        if uncommon:
            common=list(set(test[col].unique())& set(train[col].unique()))
            train[f"{col}_cat"]=train[col].apply(nearest_val)
            test[f"{col}_cat"]=test[col].apply(nearest_val)
print(train[cat_cols_updated].nunique())


global overall_best_score
# overall_best_score = 0
def OHE(train_df,test_df,cols,target):
    '''
    Function for one hot encoding, it first combines the data so that no category is missed and
    the category with least frequency can be dropped because of redundancy
    '''
    combined = pd.concat([train_df, test_df], axis=0)
    for col in cols:
        one_hot = pd.get_dummies(combined[col])
        counts = combined[col].value_counts()
        min_count_category = counts.idxmin()
        one_hot = one_hot.drop(min_count_category, axis=1)
        one_hot.columns=[str(f)+col+"_OHE" for f in one_hot.columns]
        combined = pd.concat([combined, one_hot], axis="columns")
        combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # split back to train and test dataframes
    train_ohe = combined[:len(train_df)]
    test_ohe = combined[len(train_df):]
    test_ohe.reset_index(inplace=True,drop=True)
    test_ohe.drop(columns=[target],inplace=True)
    return train_ohe, test_ohe

def high_freq_ohe(train, test, extra_cols, target, n_limit=50):
    '''
    If you wish to apply one hot encoding on a feature with so many unique values, then this can be applied, 
    where it takes a maximum of n categories and drops the rest of them treating as rare categories
    '''
    train_copy=train.copy()
    test_copy=test.copy()
    ohe_cols=[]
    for col in extra_cols:
        dict1=train_copy[col].value_counts().to_dict()
        ordered=dict(sorted(dict1.items(), key=lambda x: x[1], reverse=True))
        rare_keys=list([*ordered.keys()][n_limit:])
#         ext_keys=[f[0] for f in ordered.items() if f[1]<50]
        rare_key_map=dict(zip(rare_keys, np.full(len(rare_keys),9999)))
        
        train_copy[col]=train_copy[col].replace(rare_key_map)
        test_copy[col]=test_copy[col].replace(rare_key_map)
    train_copy, test_copy = OHE(train_copy, test_copy, extra_cols, target)
    drop_cols=[f for f in train_copy.columns if "9999" in f or train_copy[f].nunique()==1]
    train_copy=train_copy.drop(columns=drop_cols)
    test_copy=test_copy.drop(columns=drop_cols)
    
    return train_copy, test_copy

def cat_encoding(train, test,cat_cols_updated, target):
    global overall_best_score
    global overall_best_col
    table = PrettyTable()
    table.field_names = ['Feature', 'Encoded Features', 'RMSLE Score']
    train_copy=train.copy()
    test_copy=test.copy()
    train_dum = train.copy()
    for feature in cat_cols_updated:
#         print(feature)
#         cat_labels = train_dum.groupby([feature])[target].mean().sort_values().index
#         cat_labels2 = {k: i for i, k in enumerate(cat_labels, 0)}
#         train_copy[feature + "_target"] = train[feature].map(cat_labels2)
#         test_copy[feature + "_target"] = test[feature].map(cat_labels2)

        dic = train[feature].value_counts().to_dict()
        train_copy[feature + "_count"] =train[feature].map(dic)
        test_copy[feature + "_count"] = test[feature].map(dic)

        dic2=train[feature].value_counts().to_dict()
#         list1=np.arange(len(dic2.values()),0,-1) # Higher rank for high count
        list1=np.arange(len(dic2.values())) # Higher rank for low count
        dic3=dict(zip(list(dic2.keys()),list1))
        
        train_copy[feature+"_count_label"]=train[feature].replace(dic3).astype(float)
        test_copy[feature+"_count_label"]=test[feature].replace(dic3).astype(float)

        temp_cols = [ feature + "_count", feature + "_count_label"]#,feature + "_target"

        train_copy[feature]=train_copy[feature].astype(str)+"_"+feature
        test_copy[feature]=test_copy[feature].astype(str)+"_"+feature
        
        if train_copy[feature].nunique()<=15:
            train_copy[feature]=train_copy[feature].astype(str)+"_"+feature
            test_copy[feature]=test_copy[feature].astype(str)+"_"+feature
            train_copy, test_copy = OHE(train_copy, test_copy, [feature], target)
            
        else:
            train_copy,test_copy=high_freq_ohe(train_copy,test_copy,[feature], target, n_limit=15)
            
        train_copy=train_copy.drop(columns=[feature])
        test_copy=test_copy.drop(columns=[feature])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        rmse_scores = []

        for f in temp_cols:
            X = train_copy[[f]].values
            y = train_copy[target].astype(int).values

            rmses = []
            for train_idx, val_idx in kf.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                x_val, y_val = X[val_idx], y[val_idx]
                model=LinearRegression()
                model.fit(X_train,np.log1p(y_train))
                y_pred=nearest(np.expm1(model.predict(x_val)))
                rmses.append(rmse(np.log1p(y_val),np.log1p(y_pred)))
            rmse_scores.append((f,np.mean(rmses)))
            if overall_best_score > np.mean(rmses):
                overall_best_score = np.mean(rmses)
                overall_best_col = f
        best_col, best_auc = sorted(rmse_scores, key=lambda x: x[1], reverse=False)[0]

        corr = train_copy[temp_cols].corr(method='pearson')
        corr_with_best_col = corr[best_col]
        cols_to_drop = [f for f in temp_cols if corr_with_best_col[f] > 0.5 and f != best_col]
        final_selection = [f for f in temp_cols if f not in cols_to_drop]
        if cols_to_drop:
            train_copy = train_copy.drop(columns=cols_to_drop)
            test_copy = test_copy.drop(columns=cols_to_drop)

        table.add_row([feature, best_col, best_auc])
        print(feature)
    print(table)
    print("overall best CV score: ", overall_best_score)
    return train_copy, test_copy

train, test= cat_encoding(train, test,cat_cols_updated, target)
train, test=fill_missing_numerical(train,test,target, max_iterations=3)


first_drop=[ f for f in unimportant_features if f in train.columns]
train=train.drop(columns=first_drop)
test=test.drop(columns=first_drop)


final_drop_list=[]

table = PrettyTable()
table.field_names = ['Original', 'Final Transformation', "RMSLE(CV)- Regression"]
dt_params={'criterion': 'absolute_error'}
threshold=0.85
# It is possible that multiple parent features share same child features, so store selected features to avoid selecting the same feature again
best_cols=[]

for col in cont_cols:
    sub_set=[f for f in train.columns if col in f and train[f].nunique()>100]
    print(sub_set)
    if len(sub_set)>2:
        correlated_features = []

        for i, feature in enumerate(sub_set):
            # Check correlation with all remaining features
            for j in range(i+1, len(sub_set)):
                correlation = np.abs(train[feature].corr(train[sub_set[j]]))
                # If correlation is greater than threshold, add to list of highly correlated features
                if correlation > threshold:
                    correlated_features.append(sub_set[j])

        # Remove duplicate features from the list
        correlated_features = list(set(correlated_features))
        print(correlated_features)
        if len(correlated_features)>=2:

            temp_train=train[correlated_features]
            temp_test=test[correlated_features]
            #Scale before applying PCA
            sc=StandardScaler()
            temp_train=sc.fit_transform(temp_train)
            temp_test=sc.transform(temp_test)

            # Initiate PCA
            pca=TruncatedSVD(n_components=1)
            x_pca_train=pca.fit_transform(temp_train)
            x_pca_test=pca.transform(temp_test)
            x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb_final"])
            x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb_final"])
            train=pd.concat([train,x_pca_train],axis='columns')
            test=pd.concat([test,x_pca_test],axis='columns')

            # Clustering
            model = KMeans()
            kmeans = KMeans(n_clusters=28)
            kmeans.fit(np.array(temp_train))
            labels_train = kmeans.labels_

            train[col+'_final_cluster'] = labels_train
            test[col+'_final_cluster'] = kmeans.predict(np.array(temp_test))

            cat_labels=cat_labels=train.groupby([col+"_final_cluster"])[target].mean()
            cat_labels2=cat_labels.to_dict()
            train[col+"_final_cluster"]=train[col+"_final_cluster"].map(cat_labels2)
            test[col+"_final_cluster"]=test[col+"_final_cluster"].map(cat_labels2)

            correlated_features=correlated_features+[col+"_pca_comb_final",col+"_final_cluster"]

            # See which transformation along with the original is giving you the best univariate fit with target
            kf=KFold(n_splits=5, shuffle=True, random_state=42)

            rmse_scores = []

            for f in temp_cols:
                X = train_copy[[f]].values
                y = train_copy[target].astype(int).values

                rmses = []
                for train_idx, val_idx in kf.split(X, y):
                    X_train, y_train = X[train_idx], y[train_idx]
                    x_val, y_val = X[val_idx], y[val_idx]
                    model=LinearRegression()
                    model.fit(X_train,np.log1p(y_train))
                    y_pred=nearest(np.expm1(model.predict(x_val)))
                    rmses.append(rmse(np.log1p(y_val),np.log1p(y_pred)))
                    
                if f not in best_cols:
                    rmse_scores.append((f,np.mean(rmses)))
            best_col, best_rmse=sorted(rmse_scores, key=lambda x:x[1], reverse=False)[0]
            best_cols.append(best_col)

            cols_to_drop = [f for f in correlated_features if  f not in best_cols]
            if cols_to_drop:
                final_drop_list=final_drop_list+cols_to_drop
            table.add_row([col,best_col ,best_acc])

print(table)      


final_features=[f for f in train.columns if f not in [target]]
final_features=[*set(final_features)]

sc=StandardScaler()

train_scaled=train.copy()
test_scaled=test.copy()
train_scaled[final_features]=sc.fit_transform(train[final_features])
test_scaled[final_features]=sc.transform(test[final_features])
len(final_features)


def post_processor(train, test):
    cols=train.drop(columns=[target]).columns
    train_cop=train.copy()
    test_cop=test.copy()
    drop_cols=[]
    for i, feature in enumerate(cols):
        for j in range(i+1, len(cols)):
            if sum(abs(train_cop[feature]-train_cop[cols[j]]))==0:
                if cols[j] not in drop_cols:
                    drop_cols.append(cols[j])
    print(drop_cols)
    train_cop.drop(columns=drop_cols,inplace=True)
    test_cop.drop(columns=drop_cols,inplace=True)
    
    return train_cop, test_cop
                    
train_cop, test_cop=   post_processor(train_scaled, test_scaled)


data=train_cop[train_cop[target].isin([7,8,9,10,11])]
data_=train_cop[~train_cop[target].isin([7,8,9,10,11])]

from sklearn.ensemble import IsolationForest

def isolation_forest(data):
    data_pure=data.copy()
    
#     model = IsolationForest(contamination=0.010650427383702971, random_state=0)
    model = IsolationForest(contamination=0.01, random_state=0)
    
    drop_cols=[f for f in data_pure.columns if f in target]
    model.fit(data_pure.drop(columns=drop_cols))

    # Predict the anomaly scores for each data point
    anomalies = model.predict(data_pure.drop(columns=drop_cols))

    outliers = anomalies == -1

    # Combine the outlier information with the original data and labels
    data_pure['outlier_ISF'] = outliers

    # Print the identified outliers
    identified_outliers = data_pure[data_pure['outlier_ISF']]
    print(f"Number of detected Potential outliers: {identified_outliers.shape[0]}")
    
    data_clean=data[~data_pure['outlier_ISF']]
    print(data_clean.shape)
    
    return data_clean


data_clean=isolation_forest(data).reset_index(drop=True)

from sklearn.neighbors import LocalOutlierFactor

# params={'n_neighbors': 13, 'contamination': 0.01019797151100069}
params={'n_neighbors': 13, 'contamination': 0.01}


def lof(data):
    data_pure=data.copy()
    drop_cols=[f for f in data_pure.columns if f in target]

    features = data_pure.drop(columns=drop_cols)
    lof = LocalOutlierFactor(**params)  # Adjust contamination based on your data
    anomalies = lof.fit_predict(features)  # Negative scores are outliers

    outliers = anomalies == -1

    # Combine the outlier information with the original data and labels
    data_pure['outliers_LOF'] = outliers

    # Print the identified outliers
    identified_outliers = data_pure[data_pure['outliers_LOF']]
    print(f"Number of detected Potential outliers: {identified_outliers.shape[0]}")
    
    data_clean=data[~data_pure['outliers_LOF']]
    print(data_clean.shape)
    
    return data_clean

data_clean=lof(data_clean).reset_index(drop=True)

def pca_anamolies(data):
    data_pure=data.copy()
    drop_cols=[f for f in data_pure.columns if f in target]

    features = data_pure.drop(columns=drop_cols)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=2)  # Choose the number of components for visualization
    principal_components = pca.fit_transform(scaled_features)

    # Calculate the reconstruction error (MSE) for each data point
    reconstruction_errors = ((scaled_features - pca.inverse_transform(principal_components)) ** 2).mean(axis=1)

    # Set a threshold for anomaly detection
    threshold = 3.5 # Adjust the threshold based on your data and desired sensitivity

    # Identify potential outliers
    potential_outliers = [index for index, error in enumerate(reconstruction_errors) if error > threshold]

    # Create a new column 'outliers' in the DataFrame
    data_pure['outliers_PCA'] = False
    data_pure.loc[potential_outliers, 'outliers_PCA'] = True

    print(f"Number of detected Potential outliers: {len(potential_outliers)}")
    
    data_clean=data[~data_pure['outliers_PCA']]
    print(data_clean.shape)

    # Plot the data with potential outliers highlighted
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c='green', label='Normal Data')
    plt.scatter(principal_components[potential_outliers, 0], principal_components[potential_outliers, 1], c='red', label='Potential Outliers')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA with Potential Outliers')
    plt.show()
    return data_clean

data_clean=pca_anamolies( data_clean).reset_index(drop=True)


# train_cop=pd.concat([data_clean,data_],axis="rows").reset_index(drop=True)


X_train = train_cop.drop(columns=[target])
y_train = train_cop[target]

X_test = test_cop.copy()

print(X_train.shape, X_test.shape)


def get_most_important_features(X_train, y_train, n,model_input):
    
    lgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            "num_leaves": 16,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            #'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'regression_l2',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbose':-1
        }
    cb_params = {
            'iterations': 300,
            'depth': 6,
            'learning_rate': 0.01,
            'l2_leaf_reg': 0.5,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 80,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'IncToDec',
            'eval_metric': 'MSLE',
            'loss_function': 'RMSE',
            'random_state': 42,
             'verbose':False
        }
     
    xgb_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.0116,
            'colsample_bytree': 1,
            'min_child_weight': 9,
            'n_jobs': -1,
            'eval_metric': 'rmsle',
            'objective': "reg:squarederror",
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': 42,
        }
    if 'xgb' in model_input:
        model = xgb.XGBRegressor(**xgb_params)
    elif 'cat' in model_input:
        model=CatBoostRegressor(**cb_params)
    else:
        model=lgb.LGBMRegressor(**lgb_params)
    
    

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kfold.split(X_train,y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        if 'lgb' in model_input:
            model.fit(X_train_fold, np.log1p(y_train_fold))
        else:
            model.fit(X_train_fold, np.log1p(y_train_fold),verbose=False)

        y_pred = model.predict(X_val_fold)

        rmses = np.sqrt(mean_squared_error(np.log1p(y_val_fold),y_pred))
        rmse_scores.append(rmses)

    avg_rmse = np.mean(rmse_scores)

    feature_importances = model.feature_importances_

    feature_importance_list = [(X_train.columns[i], importance) for i, importance in enumerate(feature_importances)]

    sorted_features = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

    top_n_features = [feature[0] for feature in sorted_features[:n]]
    print(avg_rmse)
    return top_n_features



n_imp_features_cat=get_most_important_features(X_train.reset_index(drop=True), y_train,75, 'cat')
n_imp_features_xgb=get_most_important_features(X_train.reset_index(drop=True), y_train,75, 'xgb')
n_imp_features_lgbm=get_most_important_features(X_train.reset_index(drop=True), y_train, 75, 'lgbm')


n_imp_features=[*set(n_imp_features_lgbm+n_imp_features_cat)]#
print(f"{len(n_imp_features)} features have been selected from three algorithms for the final model")

X_train=X_train[n_imp_features]
X_test=X_test[n_imp_features]


classes = np.unique(y_train)  # Get unique class labels
class_to_index = {cls: idx for idx, cls in enumerate(classes)}
y_train_numeric = np.array([class_to_index[cls] for cls in y_train])

class_counts = np.bincount(y_train_numeric)

total_samples = len(y_train_numeric)

class_weights = total_samples / (len(classes) * class_counts)

class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

print("Class counts:", class_counts)
print("Total samples:", total_samples)
print("Class weights:", class_weights)
print("Class weights dictionary:", class_weights_dict)



import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

sgd=tensorflow.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True)
rms = tensorflow.keras.optimizers.RMSprop()
nadam=tensorflow.keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam"
)
lrelu = lambda x: tensorflow.keras.activations.relu(x, alpha=0.1)

from tensorflow.keras import backend as K

def root_mean_squared_log_error(y_true, y_pred):
    '''
    Compute RMSLE between actuals & predictions
    '''
    return K.sqrt(K.mean(K.square(K.log(abs(y_pred+1)) - K.log(abs(y_true+1)))))
    
def root_mean_squared_error(y_true, y_pred):
    '''
    Compute RMSE between actuals & predictions
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


    
ann = Sequential()
ann.add(Dense(64, input_dim=X_test.shape[1], kernel_initializer='he_uniform', activation='relu'))
ann.add(Dropout(0.1))
ann.add(Dense(16,  kernel_initializer='he_uniform', activation='relu'))
ann.add(Dropout(0.1))
ann.add(Dense(4,  kernel_initializer='he_uniform', activation='relu'))
ann.add(Dropout(0.1))

ann.add(Dense(1,  kernel_initializer='he_uniform'))
ann.compile(loss=root_mean_squared_log_error, optimizer="Adamax",metrics=['mse'])

    
ann2 = Sequential()
ann2.add(Dense(12, input_dim=X_test.shape[1], kernel_initializer='he_uniform', activation='relu'))
ann2.add(Dropout(0.1))
ann2.add(Dense(8,  kernel_initializer='he_uniform', activation='relu'))
ann2.add(Dropout(0.1))
# ann2.add(Dense(4,  kernel_initializer='he_uniform', activation='relu'))
# ann2.add(Dropout(0.1))

ann2.add(Dense(1,  kernel_initializer='he_uniform'))
ann2.compile(loss=root_mean_squared_log_error, optimizer="Adam",metrics=['mse'])



class Splitter:
    def __init__(self, test_size=0.2, kfold=True, n_splits=5):
        self.test_size = test_size
        self.kfold = kfold
        self.n_splits = n_splits

    def split_data(self, X, y, random_state_list):
        if self.kfold:
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            for random_state in random_state_list:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
                yield X_train, X_val, y_train, y_val

class Regressor:
    def __init__(self, n_estimators=100, device="gpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.reg_models = self._define_reg_model()
        self.len_models = len(self.reg_models)
        
    def _define_reg_model(self):
        
        xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 6,
            'learning_rate': 0.0116,
            'colsample_bytree': 1,
            'subsample': 0.6085,
            'min_child_weight': 9,
            'reg_lambda': 4.879e-07,
            'max_bin': 431,
            #'booster': 'dart',
            'n_jobs': -1,
            'eval_metric': 'rmsle',
            'objective': "reg:squaredlogerror",
            #'tree_method': 'hist',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        xgb_params1=xgb_params.copy()
        xgb_params1['subsample']=0.7
        xgb_params1['max_depth']=10
        xgb_params1['learning_rate']=0.002
        xgb_params1['colsample_bytree']=0.6

        xgb_params2=xgb_params.copy()
        xgb_params2['subsample']=0.5
        xgb_params2['max_depth']=8
        xgb_params2['learning_rate']=0.047
        xgb_params2['colsample_bytree']=0.9
        xgb_params2['tree_method']='approx'
     
        lgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 6,
            "num_leaves": 16,
            'learning_rate': 0.002,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            #'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'regression_l2',
            'metric': 'mean_squared_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
            'verbose':-1
        }
        lgb_params1=lgb_params.copy()
        lgb_params1['subsample']=0.9
        lgb_params1['reg_lambda']=0.8994221730208598
        lgb_params1['reg_alpha']=0.6236579699090548
        lgb_params1['max_depth']=6
        lgb_params1['learning_rate']=0.01
        lgb_params1['colsample_bytree']=0.5

        lgb_params2=lgb_params.copy()
        lgb_params2['subsample']=0.1
        lgb_params2['reg_lambda']=0.5940716788024517
        lgb_params2['reg_alpha']=0.4300477974434703
        lgb_params2['max_depth']=8
        lgb_params2['learning_rate']=0.019000000000000003
        lgb_params2['colsample_bytree']=0.8
        lgb_params3 = {
            'n_estimators': self.n_estimators,
            'num_leaves': 45,
            'max_depth': 5,
            'learning_rate': 0.0684383311038932,
            'subsample': 0.5758412171285148,
            'colsample_bytree': 0.8599714680300794,
            'reg_lambda': 1.597717830931487e-08,
            'objective': 'regression_l2',
            'metric': 'mean_squared_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
            'verbosity': 0,
            'force_col_wise': True,
            'verbose':-1
        }
        lgb_params4=lgb_params.copy()
        lgb_params4['subsample']=0.3
        lgb_params4['reg_lambda']=0.5488355125638069
        lgb_params4['reg_alpha']=0.23414681424407247
        lgb_params4['max_depth']=7
        lgb_params4['learning_rate']=0.019000000000000003
        lgb_params4['colsample_bytree']=0.5
        lgb_params4['verbose']=-1
        
        lgb_class = {
            'n_estimators': self.n_estimators,
            'max_depth': 8,
            'learning_rate': 0.02,
            'subsample': 0.20,
            'colsample_bytree': 0.56,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-08,
            'objective': 'multi_logloss',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
            'class_weight':class_weights_dict,
        }
        
        cb_params = {
            'iterations': self.n_estimators,
            'depth': 6,
            'learning_rate': 0.002,
            'l2_leaf_reg': 0.5,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 80,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'IncToDec',
            'eval_metric': 'MSLE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb_sym_params = cb_params.copy()
        cb_sym_params['grow_policy'] = 'SymmetricTree'
        cb_loss_params = cb_params.copy()
        cb_loss_params['grow_policy'] = 'Lossguide'
    
        cb_params1 = {
            'iterations': self.n_estimators,
            'depth': 15,
            'learning_rate': 0.05,
            'l2_leaf_reg': 0.1,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 50,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bernoulli',
            'od_type': 'Iter',
            'eval_metric': 'MSLE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb_params2= {
            'n_estimators': self.n_estimators,
            'depth': 10,
            'learning_rate': 0.08827842054729117,
            'l2_leaf_reg': 4.8351074756668864e-05,
            'random_strength': 0.21306687539993183,
            'max_bin': 483,
            'od_wait': 97,
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MSLE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
            'silent': True
        }
        cat_class1={
             'iterations': 500,
            'depth': 6,
            'learning_rate': 0.002,
            'l2_leaf_reg': 0.7,
            'random_strength': 0.2,
            'max_bin': 200,
            'od_wait': 65,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MultiClass',
            'loss_function': 'MultiClass',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
}
        cat_class2={
            'iterations': self.n_estimators,
            'random_strength': 0.1, 
            'one_hot_max_size': 70, 
            'max_bin': 100, 
            'learning_rate': 0.003, 
            'l2_leaf_reg': 0.3, 
            'grow_policy': 'Depthwise', 
            'depth': 9, 
            'max_bin': 200,
            'od_wait': 65,
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MultiClass',
            'loss_function': 'MultiClass',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
        }
        dt_params= {'min_samples_split': 8, 'min_samples_leaf': 4, 'max_depth': 16, 'criterion': 'squared_error'}
        knn_params= {'weights': 'uniform', 'p': 1, 'n_neighbors': 12, 'leaf_size': 20, 'algorithm': 'kd_tree'}

        reg_models = {
            'xgb_reg': xgb.XGBRegressor(**xgb_params),
           'xgb_reg1': xgb.XGBRegressor(**xgb_params1),
           'xgb_reg2': xgb.XGBRegressor(**xgb_params2),
            'lgb_reg': lgb.LGBMRegressor(**lgb_params),
#             'lgb2_reg': lgb.LGBMRegressor(**lgb_params1),
#            'lgb3_reg': lgb.LGBMRegressor(**lgb_params2),
#             'lgb4_reg': lgb.LGBMRegressor(**lgb_params3),
#           'lgb5_reg': lgb.LGBMRegressor(**lgb_params4),
#            "hgbm": HistGradientBoostingRegressor(max_iter=self.n_estimators, learning_rate=0.01, loss="squared_error", 
#                                                  n_iter_no_change=300,random_state=self.random_state),
            'cat_reg': CatBoostRegressor(**cb_params),
#            'cat_class1':CatBoostClassifier(**cat_class1),
#            'cat_class2':CatBoostClassifier(**cat_class2),
#             'cat_reg2': CatBoostRegressor(**cb_params1),
#             'cat_reg3': CatBoostRegressor(**cb_params2),
           "cat_sym": CatBoostRegressor(**cb_sym_params),
#             "cat_loss": CatBoostRegressor(**cb_loss_params),
#             'etr': ExtraTreesRegressor(min_samples_split=12, min_samples_leaf= 6, max_depth=10,
#                                        n_estimators=500,random_state=self.random_state),
 #           "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,loss="squared_error", random_state=self.random_state),
#             "RandomForestRegressor": RandomForestRegressor(max_depth= 6,max_features= 'auto',min_samples_split= 4,
#                                                            min_samples_leaf= 4,  n_estimators=500, random_state=self.random_state, n_jobs=-1),
#            'dt': DecisionTreeRegressor(**dt_params),
#            "lr":LinearRegression(),
# #             "knn":KNeighborsRegressor(**knn_params),
#             "PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=3000, tol=1e-3, n_iter_no_change=60, random_state=self.random_state),
            "HuberRegressor": HuberRegressor(max_iter=3000),
#             'ann':ann,
#             "ann2":ann2
            
            
        }


        return reg_models


class OptunaWeights:
    def __init__(self, random_state):
        self.study = None
        self.weights = None
        self.random_state = random_state

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", -1, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)
        weighted_pred=np.where(weighted_pred<1,1,weighted_pred)

        # Calculate the RMSE score for the weighted prediction
        score = rmse(np.log1p(y_true), np.log1p(weighted_pred))
        return score

    def fit(self, y_true, y_preds, n_trials=2000):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds, n_trials=2000):
        self.fit(y_true, y_preds, n_trials=n_trials)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights



kfold = True
n_splits = 1 if not kfold else 10
random_state = 42
random_state_list = [42] 
n_estimators = 9999 
early_stopping_rounds = 600
verbose = False
device = 'cpu'

splitter = Splitter(kfold=kfold, n_splits=n_splits)


# Initialize an array for storing test predictions
test_predss = np.zeros(X_test.shape[0])
ensemble_score = []
weights = []
trained_models = dict(zip(Regressor().reg_models.keys(), [[] for _ in range(Regressor().len_models)]))
trained_models = {'xgb_reg':[]}

# Evaluate on validation data and store predictions on test data
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits

    # Get a set of Regressor models
    reg = Regressor(n_estimators, device, random_state)
    models = reg.reg_models

    # Initialize lists to store oof and test predictions for each base model
    oof_preds = []
    test_preds = []

    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
    for name, model in models.items():
        if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
            if ('lgb' in name) and ("reg" in name): #categorical_feature=cat_features
                model.fit(X_train_, np.log1p(y_train_), eval_set=[(X_val, np.log1p(y_val))]),#,categorical_feature=cat_features,)
            elif 'class' in name:
                model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)],#cat_features=cat_features,
                          early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            elif "xgb" in name:
                model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            else:
                model.fit(X_train_, np.log1p(y_train_), eval_set=[(X_val, np.log1p(y_val))], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        elif name=='ann':
            model.fit(X_train_, np.log1p(y_train_), validation_data=(X_val, np.log1p(y_val)),batch_size=50, epochs=200,verbose=verbose)
        else:
            model.fit(X_train_, np.log1p(y_train_))
            
        if name=="ann":
            y_val_pred = np.expm1(model.predict(X_val)[:,0])
            test_pred = np.expm1(model.predict(X_test)[:,0])
        elif "class" in name:
            y_val_pred =np.array(model.predict(X_val))[:,0]
            test_pred = np.array(model.predict(X_test))[:,0]
        elif "xgb" in name:
            y_val_pred =np.array(model.predict(X_val))
            test_pred = np.array(model.predict(X_test))            
        else:
            y_val_pred = np.expm1(model.predict(X_val))
            test_pred = np.expm1(model.predict(X_test))


#         # Convert predicted values back to their original scale by applying the expm1 function
#         y_val_pred = np.expm1(y_val_pred)
#         test_pred = np.expm1(test_pred)

        score = rmse(np.log1p(y_val), np.log1p(y_val_pred))
#         print("pure RMSE",rmse(np.log1p(y_val), np.log1p(y_val_pred)))

#         score = rmse(np.expm1(y_val), y_val_pred)

        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] RMSLE score: {score:.5f}')

        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))

    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state)
    
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
#     y_val_pred = optweights.fit_predict(np.expm1(y_val.values), oof_preds)

    score = rmse(np.log1p(y_val), np.log1p(y_val_pred))
#     score = rmse(np.expm1(y_val), y_val_pred)

    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] RMSLE score -------> {score:.5f}')
    ensemble_score.append(score)
    weights.append(optweights.weights)
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))

    gc.collect()


mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble RMSLE score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name} {mean_weight:.5f} ± {std_weight:.5f}')


def visualize_importance(models, feature_cols, title, top=15):
    importances = []
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["importance"] = model.feature_importances_
        _df["feature"] = pd.Series(feature_cols)
        _df["fold"] = i
        _df = _df.sort_values('importance', ascending=False)
        _df = _df.head(top)
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)
        
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    # display(feature_importance.groupby(["feature"]).mean().reset_index().drop('fold', axis=1))
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance, color='brown', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance [Top {top}]', fontsize=15)
    plt.grid(True, axis='x')
    plt.show()
    
for name, models in trained_models.items():
    visualize_importance(models, list(X_train.columns), name)


submission[target] = test_predss
submission[target]=np.clip(submission[target],1,29)
submission.to_csv('submission.csv',index=False)


target=[target]
sub_ext1=pd.read_csv("/kaggle/input/clean-code-voting-regressor-base-3-models/submission.csv")
sub_ext2=pd.read_csv("/kaggle/input/regression-with-an-abalone-dataset-ml-regressor/submission.csv")

submission1=pd.read_csv("/kaggle/input/ps4e4-ensemble-ancillary/sub_pure_14648.csv")
submission2=pd.read_csv("/kaggle/input/ps4e4-ensemble-ancillary/sub_pure_14651.csv")
submission3=pd.read_csv("/kaggle/input/ps4e4-ensemble-ancillary/sub_pure_14685.csv")


sub_list=[sub_ext1,sub_ext2, submission1,submission2, submission] # list all the results

"""
Assign Weights of each submissions, a general method is to rank them.
"""
weights=[3,3,1,1,3]


weighted_list = [item for sublist, weight in zip(sub_list, weights) for item in [sublist] * weight]

def ensemble_mean(sub_list,cols, mean="AM"):
    
    """
    The function computes Arithmetic Mean/Geometric Mean/Harmonic Mean given a list of results with specific results.
    """
    
    sub_out=sub_list[0].copy()
    if mean=="AM":
        for col in cols:
            sub_out[col]=sum(df[col] for df in sub_list)/len(sub_list)
    elif mean=="GM":
        for df in sub_list[1:]:
            for col in cols:
                sub_out[col]*=df[col]
        for col in cols:
            sub_out[col]=(sub_out[col])**(1/len(sub_list))
    elif mean=="HM":
        for col in cols:
            sub_out[col]=len(sub_list)/sum(1/df[col] for df in sub_list)
    
    return sub_out
    
sub_ensemble=ensemble_mean(weighted_list,target,mean="AM")
sub_ensemble.to_csv('submission_comb.csv',index=False)
sub_ensemble.head()

