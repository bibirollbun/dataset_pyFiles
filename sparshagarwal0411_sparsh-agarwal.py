# Leaving this here for person whose gonna review my notebook for GDSC...

# My notebook has 94 versions..All version from Ver 1, have smthng unique and new experiments done by me...
# So request the person to view all. Am listing versions with best public and private score to stress on them...
# PUBLIC SCORE - Version 82
# PRIVATE SCORE- Version 89


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


# Warnings
import warnings
warnings.filterwarnings("ignore")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# GPU acceleration
import cudf.pandas

# Optimization
import optuna

# Gradient Boosting
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score

# Sklearn tools
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error


# Importing csv files
train = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv")
test = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv")
final = pd.read_csv('/kaggle/input/recruitment-task-for-gdsc-ml/SPECIMEN.csv')


train.shape


test.shape


final.shape


train.head()


train.tail()


test.head()


test.tail()


# Since symbols in feature's names can be difucult to engineer.. We can do thi instead- 

train.columns = [f"f{i}" for i in range(train.shape[1])]
test.columns = [f"f{i}" for i in range(test.shape[1])]


# Dropping Id column

train = train.drop(['f0'], axis = 1)
test = test.drop(['f0'], axis = 1)


train.head()


test.head()


test.info()


id = final['LOCAL_IDENTIFIER']


# Columns to encode
cat_cols = ['f14', 'f22', 'f24']

# Create encoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train and transform
train[cat_cols] = encoder.fit_transform(train[cat_cols])

# Transform test using the same mapping
test[cat_cols] = encoder.transform(test[cat_cols])


# # Double - Checking for vulgarity in Data
# for col in train.columns[:44]:
#     plt.figure(figsize=(6,4))
#     sns.kdeplot(train[col], label="Train", color = 'cyan')
#     sns.kdeplot(test[col], label="Test", color = 'red')
#     plt.title(col)
#     plt.legend()
#     plt.show()


# Setting correlation in train features
corr= train.corr().abs()


# dropping colums with correlation more than 90%

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > .9)]
print("Features to drop:", to_drop)


# # Keeping f33 to represent these dropped features...It was found most important among others....

# train = train.drop(['f4', 'f17', 'f21','f31','f45'], axis = 1)
# test = test.drop(['f4', 'f17', 'f21','f31','f45'], axis = 1)


# # HIST-PLOT to check whether features are normal, uniform or multi-modal....

# for col in train:
#     plt.figure(figsize = (10,10))
#     plt.style.use('dark_background')
#     sns.histplot(train[col].dropna(), bins= 100, kde=True)
#     plt.title(f"{col}", color = 'red')
#     plt.show()


# for col in test:
#     plt.figure(figsize = (10,10))
#     plt.style.use('dark_background')
#     sns.histplot(train[col].dropna(), bins= 100, kde=True)
#     plt.title(f"{col}", color = 'red')
#     plt.show()    


# Types of data in our features
Normal =['f2','f5','f7','f9','f15','f16','f19','f23','f25','f31','f26','f27','f28','f29','f30','f32','f37','f39','f42','f43','f44']
Uniform = ['f1','f3','f6','f8','f11','f12','f18','f20','f33','f34','f35','f36', 'f4', 'f17', 'f21', 'f33']
Multimodal = ['f10','f13','f38','f40','f41','f46','f45','f14', 'f22', 'f24']


# Filing NaN values of Normal features with median
for col in Normal:
    median_val = train[col].median()
    median_test = test[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_test)
    
#Filling NaN values of Uniform and Multimodal features with Iterative Imputer
imputer = KNNImputer(n_neighbors = 1)
train[Uniform] = imputer.fit_transform(train[Uniform])
test[Uniform] = imputer.fit_transform(test[Uniform])


train[Multimodal] = imputer.fit_transform(train[Multimodal])
train['f47'] = imputer.fit_transform(train[['f47']])
test[Multimodal] = imputer.fit_transform(test[Multimodal])


# # Heatmap built by correlation

# plt.figure(figsize = (20,10))
# plt.style.use('dark_background')
# sns.heatmap(corr, cmap = 'viridis', cbar = True, annot = True)


# # Checking first 5 features relation with Target

# for col in train.columns[:5]:  
#     plt.figure(figsize=(2,2))
#     sns.scatterplot(x=train[col], y=train['f47'], alpha=0.3)
#     plt.title(f"{col} vs Target")
#     plt.show()


# Defining X and y
X = train.drop(['f47'], axis =1)
y = train['f47']


# Performing train-test split 
X_train, X_value, y_train, y_value = train_test_split(X, y, test_size = 0.3, random_state = 16064)
X_test = test.copy()


# Using Optuna to decide hyper-parameters
def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-5, 10, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)
    selection = trial.suggest_categorical("selection", ["cyclic", "random"])
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    max_iter = trial.suggest_int('max_iter', 1000, 20000, step=1000)
    random_state = trial.suggest_int('random_state', 0 , 100)

 # Choosing ElasticNet model
    
    model = ElasticNet(alpha = alpha,
                       l1_ratio = l1_ratio,
                       tol = tol,
                       selection = selection,
                       fit_intercept = fit_intercept,
                       max_iter = max_iter,
                       random_state = random_state
        )
    
    score = cross_val_score(
        model, X_train, y_train, cv = 2,
        scoring = "neg_root_mean_squared_error",
        n_jobs = -1
    ).mean()
    
    return -score
    
# Creating study by Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials = 1000)

# Getting best parameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")
print("Best CV RMSE:", study.best_value)


# Defining Best parameters to be modeled

best_alpha= best_params['alpha']
best_l1_ratio = best_params['l1_ratio']
best_tol = best_params['tol']
best_selection = best_params['selection']
best_fit_intercept = best_params['fit_intercept']
best_max_iter = best_params['max_iter']
best_random_state = best_params['random_state']


# K- Folding and modeling

cv = KFold(n_splits= 100, shuffle= True)

oof_preds = np.zeros(len(X_train))   
test_preds = np.zeros(len(X_test))  
fold_rmse = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    
    
    model = ElasticNet(alpha = best_alpha,
                       l1_ratio = best_l1_ratio,
                       tol = best_tol,
                       selection = best_selection,
                       fit_intercept = best_fit_intercept,
                       max_iter = best_max_iter)     
    
    model.fit(X_tr, y_tr)
    
    # Validating Predictions
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds

    # Test predictions (average across folds)
    test_preds += model.predict(X_test) / cv.n_splits

    # RMSE per fold
    mse = mean_squared_error(y_val, val_preds)
    rmse = np.sqrt(mse)
    fold_rmse.append(rmse)


# Finding Fold with least RMSE
value = min(fold_rmse)
i = fold_rmse.index(min(fold_rmse)) 
print(f"Fold {i + 1} provides the least RMSE value ({value})...")


# Submission
submission = pd.DataFrame({
    "LOCAL_IDENTIFIER": final["LOCAL_IDENTIFIER"].astype(int),
    "CORRUCYSTIC_DENSITY": test_preds.astype(float)
})
# Save CSV
submission.to_csv("submission.csv", index=False)

# Preview
submission.head(100)

