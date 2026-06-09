import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# general
import pandas as pd
import numpy as np
import math

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")
sns.set_palette("Paired")

#ML Modeling
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
import catboost as catb

import optuna

# Avoid unnecessary warnings
import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
samp_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print("Train Data:")
display(train_data.head())

print("\nTest Data:")
display(test_data.head())

print("\nSample Submission:")
display(samp_sub.head())


print("!!!! TRAIN DATA STATISTICS !!!!\n")

print("1. Shape : " , train_data.shape)
print("=====================================")

display(train_data.describe(include="all").T)
print("=====================================")

print("2. Column Info :\n")
display(train_data.info())
print("=====================================")

print("\n3.Null Values Info : \n")
display(train_data.isnull().sum())


TARGET = 'accident_risk'
N_SPLITS = 5

train_x = train_data.drop([TARGET , 'id'] , axis=1)
train_y = train_data[TARGET]

test_x = test_data.drop('id' , axis=1)

CAT_COLS = train_x.select_dtypes(include=['object' , 'bool']).columns.tolist()
NUM_COLS = train_x.select_dtypes(include=['int' , 'float']).columns.tolist()

print("Categorical Features : " , CAT_COLS)
print("Numerical Features : " , NUM_COLS)


# Target Distribution
def target_dist_plot(target , name):
    fig, ax = plt.subplots(2 , 1 , figsize=(18, 10 * 1))
    ax = ax.flatten()

    sns.kdeplot(target , ax=ax[0] , label = target , color="skyblue" , linestyle = "-." , linewidth=2)
    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Density")
    ax[0].set_title("KDE Plot")

    sns.histplot(target , ax=ax[1] , kde = False , color = "skyblue" , edgecolor = 'blue')
    ax[1].set_xlabel("Accident Risk")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Histogram")

    plt.tight_layout()
    plt.show() 

# Feature Distribution
def cat_feature_distribution(data):
    n_cols = 2
    n_rows = math.ceil(len(CAT_COLS) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(CAT_COLS):
        ax = axes[i]
        sns.countplot(data=data, x=col, ax=ax)
        ax.set_title(f"{col}", fontsize = 14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize = 14)
        
    plt.tight_layout()
    plt.show()

def num_feature_distribution(data):
    n_cols = 2
    n_rows = math.ceil(len(NUM_COLS) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(NUM_COLS):
        if col == "curvature" :
            sns.histplot(train_x[col], ax=axes[i], kde=False)
        else:
            sns.histplot(train_x[col], ax=axes[i], kde=False, binwidth=0.3)
            
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        
    plt.tight_layout()
    plt.show()

# Correlation Matrix
def correlation_plot(data , name):
    plt.figure(figsize=(20,8))

    num_cols = data.select_dtypes(include=["int" , "float"]).columns
    corr_matrix = data[num_cols].corr()

    plt.gcf().set_facecolor('#FFFDD0') 
    sns.heatmap(corr_matrix , annot = True, cmap='coolwarm' , fmt = ".2f" , linewidths = 0.5)
    plt.title(name)
    plt.show()

print("## Accident Risk Distribution ##\n")
target_dist_plot(train_data[TARGET] , "Accident Risk Distribution")

print("\n## Feature Distribution ##\n")
print("Categorical")
cat_feature_distribution(train_x)
print("Numerical")
num_feature_distribution(train_x)

print("\n## Correlation Matrix ##\n")
correlation_plot(train_x , "Correlation between Numerical features")


def rmse(y_true , y_pred):
    return np.sqrt(mean_squared_error(y_true , y_pred))


def ensemble_model(train_x , train_y , test_x , N_SPLITS=5):

    # CatBoost Modeling
    catb_params = {
    'learning_rate' : 0.06846073551293783,
    'l2_leaf_reg' : 0.03358471172334371,
    'colsample_bylevel' : 0.4681301004890497,
    'depth' : 8,
    'min_data_in_leaf' : 12,
    'subsample' : 0.6306114676661142
    }

    catb_model = catb.CatBoostRegressor(**catb_params)
    catb_oof_preds = np.zeros(len(train_x))
    catb_models , catb_scores = [] , []
    kf = KFold(n_splits = N_SPLITS , shuffle=True , random_state=0)

    print("## CatBoost Model ##\n")
    for train_idx , val_idx in kf.split(train_x , train_y):
        print('\nFold:' , len(catb_models) + 1)
        X_train , X_val = train_x.iloc[train_idx] , train_x.iloc[val_idx]
        Y_train , Y_val = train_y.iloc[train_idx] , train_y.iloc[val_idx]
        train_pool = catb.Pool(X_train , Y_train , cat_features = CAT_COLS)
        test_pool = catb.Pool(X_val , Y_val , cat_features = CAT_COLS)

        catb_model.fit(train_pool , eval_set = test_pool , verbose=100)
        catb_val_pred = catb_model.predict(X_val)
        catb_oof_preds[val_idx] = catb_val_pred
        catb_val_rmse = rmse(Y_val , catb_val_pred)
        catb_models.append(catb_model) , catb_scores.append(catb_val_rmse)

    # LightGBM Modeling
    lgb_params = {
    'learning_rate' :0.08848094516681457,
    'num_leaves' : 32,
    'max_depth' : 5,
    'reg_alpha' : 0.0686341087751724,
    'reg_lambda' : 0.040740581260332503,
    'subsample' : 0.5970157462771167
    }

    lgb_oof_preds = np.zeros(len(train_x))
    kf = KFold(n_splits = N_SPLITS , shuffle=True , random_state=0)

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_models , lgb_scores = [] , []

    print("\n## LightGBM Model ##\n")
    for train_idx , val_idx in kf.split(train_x , train_y):
        print('\nFold:' , len(lgb_models) + 1)
        X_train , X_val = train_x.iloc[train_idx] , train_x.iloc[val_idx]
        Y_train , Y_val = train_y.iloc[train_idx] , train_y.iloc[val_idx]
        X_test = test_x.copy()
        X_train[CAT_COLS] = X_train[CAT_COLS].astype('category')    
        X_val[CAT_COLS] = X_val[CAT_COLS].astype('category') 
        X_test[CAT_COLS] = X_test[CAT_COLS].astype('category')

        lgb_model.fit(X_train , Y_train , eval_set = [(X_val , Y_val)])
        lgb_val_pred = lgb_model.predict(X_val)
        lgb_oof_preds[val_idx] = lgb_val_pred
        lgb_model_rmse = rmse(Y_val , lgb_val_pred)
        lgb_models.append(lgb_model) , lgb_scores.append(lgb_model_rmse)

    # XGBoost Modeling
    xgb_params = {
        'learning_rate': 0.01175895769174525, 
         'n_estimators': 2155, 
         'max_depth': 6, 
         'subsample': 0.9280196779407106, 
         'colsample_bytree': 0.4869279064549016, 
         'early_stopping_round': 357,
         'enable_categorical' : True
    }

    xgb_oof_preds = np.zeros(len(train_x))
    kf = KFold(n_splits = N_SPLITS , shuffle=True , random_state=0)
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_models , xgb_scores = [] , []

    print("\n## XGBoost Model ##\n")
    for train_idx , val_idx in kf.split(train_x , train_y):
        print('\nFold:' , len(xgb_models) + 1)
        X_train , X_val = train_x.iloc[train_idx] , train_x.iloc[val_idx]
        Y_train , Y_val = train_y.iloc[train_idx] , train_y.iloc[val_idx]
        X_test = test_x.copy()
        X_train[CAT_COLS] = X_train[CAT_COLS].astype('category')    
        X_val[CAT_COLS] = X_val[CAT_COLS].astype('category')    
        X_test[CAT_COLS] = X_test[CAT_COLS].astype('category')

        xgb_model.fit(X_train , Y_train , eval_set = [(X_val , Y_val)] , verbose=500)
        xgb_val_pred = xgb_model.predict(X_val)
        xgb_oof_preds[val_idx] = xgb_val_pred
        xgb_model_rmse = rmse(Y_val , xgb_val_pred)
        xgb_models.append(xgb_model) , xgb_scores.append(xgb_model_rmse)

    # Ensemble
    print("\n## Ensemble ##\n")
    catb_test_preds = sum(catb_model.predict(X_test) for catb_model in catb_models) / len(catb_models)
    lgb_test_preds = sum(lgb_model.predict(X_test) for lgb_model in lgb_models) / len(lgb_models)
    xgb_test_preds = sum(xgb_model.predict(X_test) for xgb_model in xgb_models) / len(xgb_models)

    final_preds = 0.4*catb_test_preds + 0.3*lgb_test_preds + 0.3*xgb_test_preds

    print("Catb Scores : " , catb_scores)
    print("LightGBM Scores : " , lgb_scores)
    print("XGBoost Scores : " , xgb_scores)
    print("\nFinal Test preds : " , final_preds)
    return final_preds

results = ensemble_model(train_x , train_y , test_x)


results = np.round(results , 3)

submission = pd.DataFrame({'id': test_data['id'], 'accident_risk': results})
submission.to_csv('submission.csv', index=False)
display(submission.head())

