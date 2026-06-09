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


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


## first 5 rows of train data
train.head()


## first 5 rows of test data
test.head()


sample_submission


train.isnull().sum() ## There is no any null values in train data


test.isnull().sum() ## There is no any null values in test data


train.describe()


train.info()


print("Shape of train:",train.shape)
print("Shape of test:",test.shape)


import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline


# ## function to plot some graphs and print the dtype of feature
# def plot_graphs(df,df_col):
#     col_dtype=df[df_col].dtype
#     print("="*50,"Feature:",df_col,"="*50)
#     print("Data type:",col_dtype)
#     print("Total missing/null values:",df[df_col].isnull().sum())

#     print()
#     if(col_dtype=='O'):
#         print("Total Unique Values:",df[df_col].nunique())
#         plt.figure(figsize=(15,6))

#         plt.subplot(1,2,1)
#         plt.title("Bar Plot for {}".format(df_col))
#         plt.ylabel("Count")
#         df[df_col].value_counts().plot(kind='bar')

#         plt.subplot(1,2,2)
#         plt.title("Pie Chart for {}".format(df_col))
#         df[df_col].value_counts().plot(kind='pie', autopct='%.2f%%')
#         plt.show()


#     elif(col_dtype!='O'):
#         print("Mean:",np.round(df[df_col].mean(),2))
#         print("Median:",np.round(df[df_col].median(),2))
#         print("Minimum:",df[df_col].min())
#         print("Maximum:",df[df_col].max())
#         print("Std:",np.round(df[df_col].std(),2))
#         print("Skew:",df[df_col].skew())
        
#         plt.figure(figsize=(18,15))

#         plt.subplot(2,2,1)
#         plt.title("Histogram for '{}'".format(df_col))
#         df[df_col].plot(kind='hist', color='skyblue', edgecolor='black')

#         plt.subplot(2,2,2)
#         plt.title("KDE plot for '{}'".format(df_col))
#         df[df_col].plot(kind='kde', color='green')

#         plt.subplot(2,2,3)
#         plt.title("Box Plot for '{}'".format(df_col))
#         df[df_col].plot(kind='box', color='orange')

#         plt.subplot(2,2,4)
#         plt.title("Displot for '{}'".format(df_col))
#         # sns.displot(data=df,x=df[df_col],kind='hist',kde=True)
#         sns.histplot(data=df, x=df[df_col], kde=True, color='purple')
        
#         plt.show()


#     else:
#         print("Datatype of feature is neither numeric not categorical...")


# ## univariate plots for each feature
# for feature in train.columns:
#     plot_graphs(train,feature)


# sns.pairplot(data=train)


## there is no need of id feature so let's drop it
train.drop('id', axis=1, inplace=True)


X = train.drop('BeatsPerMinute',axis=1)
y = train['BeatsPerMinute']


X


y


from sklearn.preprocessing import PolynomialFeatures, StandardScaler


X.columns


def feature_engineering(df, target='BeatsPerMinute', poly_degree=2):
    # Make a copy to avoid modifying the original DataFrame
    df_engineered = df.copy()
    
    df_engineered['Log_TrackDurationMs'] = np.log1p(df_engineered['TrackDurationMs'])
    df_engineered['Log_AudioLoudness'] = np.log1p(abs(df_engineered['AudioLoudness']))
    
    df_engineered['Rhythm_Loudness_Interaction'] = df_engineered['RhythmScore'] * df_engineered['AudioLoudness']
    df_engineered['Energy_per_Duration'] = df_engineered['Energy'] / df_engineered['TrackDurationMs']

    df_engineered['Overall_Score'] = df_engineered['RhythmScore'] + df_engineered['InstrumentalScore'] + df_engineered['MoodScore']
    
    return df_engineered


X = feature_engineering(X)


X


test = feature_engineering(test)


from sklearn.metrics import r2_score,mean_squared_error


## Model Training and Model Selection

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost 
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor


from sklearn.model_selection import KFold


test.drop('id', axis=1, inplace=True)


sample_submission.head(3)


lgbm_best_params = {'n_estimators': 507,
                     'max_depth': 4,
                     'learning_rate': 0.011601596659910327,
                     'num_leaves': 1950,
                     'min_child_samples': 77,
                     'subsample': 0.5530516591087266,
                     'colsample_bytree': 0.5014930205871159,
                     'reg_alpha': 3.1108310696136283,
                     'reg_lambda': 5.698589444074868
                   }


## Model training
models={
    "XGBRegressor":XGBRegressor(tree_method="gpu_hist"),
    "LGBMRegressor":LGBMRegressor(device="gpu", verbosity=-1),
    "LGBMRegressor_with_Optuna":LGBMRegressor(device="gpu", verbosity=-1, **lgbm_best_params),
    "CatBoostRegressor":CatBoostRegressor(task_type="GPU", devices="0", verbose=False),
}

n_splits=5
state=42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=state)

for model_name, model in models.items():
    print(model_name,"=============================>")
    print(model)

    print()
    test_pred = np.zeros(test.shape[0])
    score_list = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X,y), 1):
        # define the model 
        fold_model = model
        
        X_train_fold,X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold,y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # training the model
        fold_model.fit(X_train_fold,y_train_fold)
        y_test_fold_pred = fold_model.predict(X_test_fold)
        
        score = np.sqrt(mean_squared_error(y_test_fold,y_test_fold_pred))
        
        print(f"Fold {fold}: RMSE : {score:.4f}")
        score_list.append(score)

        test_pred+=fold_model.predict(test)
    

    print(f"\n Average RMSE : {np.mean(score_list):.4f}\n")
       
    # Average test predictions over all folds
    test_pred /= n_splits
    
    ## saving prediction in submission file
    sample_submission['BeatsPerMinute'] = test_pred
    sample_submission.to_csv(f"{model_name}_prediction.csv", index=False) 
    display(sample_submission.head())
    print(f"File saved as {model_name}_prediction.csv.....\n")





from cuml.ensemble import RandomForestRegressor

import cudf
# Load data into a cuDF DataFrame
X_rf = cudf.DataFrame(X)
y_rf = cudf.Series(y)

# Create a RandomForestRegressor object
rf_regressor_gpu = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)


test_rf = cudf.DataFrame(test)


models={
    "RandomForestRegressor": rf_regressor_gpu,
}

n_splits=5
state=42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=state)

for model_name, model in models.items():
    print(model_name,"=============================>")
    print(model)

    print()
    test_pred = np.zeros(test.shape[0])
    score_list = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_rf,y_rf), 1):
        # define the model 
        fold_model = model
        
        X_train_fold,X_test_fold = X_rf.iloc[train_idx], X_rf.iloc[test_idx]
        y_train_fold,y_test_fold = y_rf.iloc[train_idx], y_rf.iloc[test_idx]

        # training the model
        fold_model.fit(X_train_fold,y_train_fold)
        y_test_fold_pred = fold_model.predict(X_test_fold)
        
        score = np.sqrt(mean_squared_error(y_test_fold.to_numpy(),y_test_fold_pred.to_numpy()))
        
        print(f"Fold {fold}: RMSE : {score:.4f}")
        score_list.append(score)

        test_pred+=fold_model.predict(test_rf).to_numpy()
    

    print(f"\n Average RMSE : {np.mean(score_list):.4f}\n")
       
    # Average test predictions over all folds
    test_pred /= n_splits
    
    ## saving prediction in submission file
    sample_submission['BeatsPerMinute'] = test_pred
    sample_submission.to_csv(f"{model_name}_prediction.csv", index=False) 
    display(sample_submission.head())
    print(f"File saved as {model_name}_prediction.csv.....\n")




