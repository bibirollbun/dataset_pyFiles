# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# %load_ext cudf.pandas
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


''' Import Libraries '''
# %load_ext cudf.pandas
import numpy as np
import pandas as pd
# import cudf
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


import xgboost as xgb
import lightgbm as lgb

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
# from cuml.neighbors import KNeighborsRegressor as cuKNN


''' Load Data '''
# Read in Train Data
dft1 = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
dft2 = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
train_df = pd.concat([dft1, dft2], axis=0, ignore_index=True)
# Read in Test Data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
# Read in original dataset
og_df = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")



''' Preprocess Data '''
# Generate & Apply Price mapping from original dataset
STATS_num = ["mean","std","count","nunique","median","min","max","skew"]
og_priceMap = og_df.groupby("Weight Capacity (kg)").Price.agg(STATS_num)
og_priceMap.columns = [f"og_Price_{stat}" for stat in STATS_num]
og_PriceCols = og_priceMap.columns.tolist()
# og_priceMap.name = "og_Price"
train_df = train_df.merge(og_priceMap, on="Weight Capacity (kg)", how="left")
test_df = test_df.merge(og_priceMap, on="Weight Capacity (kg)", how="left")
# Merge rest of original dataset (except Price) into train/test
og_df = og_df.loc[(og_df["Weight Capacity (kg)"] > 5) & (og_df["Weight Capacity (kg)"] < 30)]
og_df.columns = [f"og_{column}" for column in og_df.columns]
train_df = train_df.merge(og_df.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="og_Weight Capacity (kg)", how="left")
test_df = test_df.merge(og_df.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="og_Weight Capacity (kg)", how="left")


Cols = train_df.columns
ogCols = og_df.columns
newCATS = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
ogCATS = ['og_Brand', 'og_Material', 'og_Size', 'og_Compartments', 'og_Laptop Compartment', 'og_Waterproof', 'og_Style', 'og_Color']
CATS = newCATS + ogCATS
NUMS = ['Weight Capacity (kg)', 'og_Weight Capacity (kg)'] + og_PriceCols

# Factorize categorical columns 
for column in CATS:
    # Combine so cats have same encoding for train/test
    combine = pd.concat([train_df[column], test_df[column]], axis=0)
    combine,_ = pd.factorize(combine)
    train_df[column] = combine[:len(train_df)]
    test_df[column] = combine[len(train_df):]

print(NUMS)
train_df.head()



''' Preprocess Data 2 '''
# Downscale data format for faster processing
# for col in train_df.columns: train_df[col] = train_df[col].astype('float32')
# for col in test_df.columns: test_df[col] = test_df[col].astype('float32')


# Combine Weight Capacity with all other categorical columns
CombinedCols = []
for column in CATS:
    train_df[f"{column}_wCap"] = (train_df[column] * 100) + train_df["Weight Capacity (kg)"]
    test_df[f"{column}_wCap"] = (test_df[column] * 100) + test_df["Weight Capacity (kg)"]
    CombinedCols.append(f"{column}_wCap")


FEATURES = CATS + NUMS + CombinedCols


print(train_df.isna().sum())
print("================================")
print(test_df.isna().sum())


''' Train KNN Model '''
# X_ktrain, X_kval, y_ktrain, y_kval = train_test_split(train_df[FEATURES], train_df["Price"], test_size=0.2, random_state=42)

# KNNModel = cuKNN(n_neighbors=1, algorithm='rbc')
# KNNModel.fit(X_ktrain, y_ktrain)

# kvalidations = KNNModel.predict(X_kval)
# kvalidations, y_kval = kvalidations.to_numpy(), y_kval.to_numpy()
# kscore = np.sqrt(mean_squared_error(y_kval, kvalidations))
# print(f"KNN RMSE: {kscore}")

# kpredictions = KNNModel.predict(KX_test)





# kdefaults = np.ones(len(KX_test)) * -1

# Find duplicates to start off our Price Predictions
# count = 0
# distances, indices = KNNModel.kneighbors(KX_test)
# for i in range(len(KX_test)):
#     if distances[i] == 0: 
#         count += 1
#         kdefaults[i] = kpredictions[i]

# print(f"{count} duplicates found!")

# kvalidations = np.array(kvalidations).flatten()
# kscore = np.sqrt(mean_squared_error(y_kval, kvalidations))
# print(f"KNN RMSE: {kscore}")


''' Nested K-Fold Feature Engineering and LGBM/XGB Training'''

FOLDS = 7
kf_outer = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
 

STATS_num = ["mean","std","count","nunique","median","min","max","skew"]
STATS_cat = ["mean", "std", "count", "nunique", "median"]

predictions = np.zeros((len(test_df)))
predictions1 = np.zeros((len(test_df)))
predictions2 = np.zeros((len(test_df)))


for i, (train_idx, val_idx) in enumerate(kf_outer.split(train_df)):

    start_time = time.time()
    
    # Split entire dataset into current train/valid K-Fold split
    X_train = train_df.loc[train_idx, FEATURES + ['Price']].reset_index(drop=True).copy() # include Price in training data for target encoding
    y_train = train_df.loc[train_idx, 'Price']
    X_valid = train_df.loc[val_idx, FEATURES].reset_index(drop=True).copy()
    y_valid = train_df.loc[val_idx, 'Price']
    X_test = test_df[FEATURES].copy()

    
    # Generate TARGET ENCODING (nested K-Fold)
    kf_inner = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for j, (train_idx_2, val_idx_2) in enumerate(kf_inner.split(X_train)):
        # Split current train split further into train/valid split
        X_train_2 = X_train.loc[train_idx_2, FEATURES + ["Price"]].copy()
        X_valid_2 = X_train.loc[val_idx_2, FEATURES].copy()
        for column in NUMS:
            # Create TARGET ENCODING from inner train set, apply to the inner validation set
            TE = X_train_2.groupby(column).Price.agg(STATS_num)#.astype('float32')
            TE.columns = [f"TEnum_{column}_{stat}" for stat in STATS_num]
            X_valid_2 = X_valid_2.merge(TE, on=column, how="left")
            for col in TE.columns:
                # Apply TARGET ENCODING to the outer fold training set
                X_train.loc[val_idx_2, col] = X_valid_2[col].values
        for column in CombinedCols:
            # Create TARGET ENCODING from inner train set, apply to the inner validation set
            TE = X_train_2.groupby(column).Price.agg(STATS_cat)#.astype('float32')
            TE.columns = [f"TEcat_{column}_{stat}" for stat in STATS_cat]
            X_valid_2 = X_valid_2.merge(TE, on=column, how="left")
            for col in TE.columns:
                # Apply TARGET ENCODING to the outer fold training set
                X_train.loc[val_idx_2, col] = X_valid_2[col].values
        

    # Apply numerical TARGET ENCODING to the outer valid/test
    for column in NUMS:
        TE = X_train.groupby(column).Price.agg(STATS_num)#.astype('float32')
        TE.columns = [f"TEnum_{column}_{stat}" for stat in STATS_num]
        X_valid = X_valid.merge(TE, on=column, how="left")
        X_test = X_test.merge(TE, on=column, how="left")
    # Apply categorical TARGET ENCODING to the outer valid/test
    for column in CombinedCols:
        TE = X_train.groupby(column).Price.agg(STATS_cat)#.astype('float32')
        TE.columns = [f"TEcat_{column}_{stat}" for stat in STATS_cat]
        X_valid = X_valid.merge(TE, on=column, how="left")
        X_test = X_test.merge(TE, on=column, how="left")
    # Apply categorical FEATURE ENCODING to the outer valid/test
    for column in CATS:
        FE = X_train.groupby(column)["Weight Capacity (kg)"].agg(STATS_cat)#.astype('float32')
        FE.columns = [f"FEcat_{column}_wCap_{stat}" for stat in STATS_cat]
        X_train = X_train.merge(FE, on=column, how="left")
        X_valid = X_valid.merge(FE, on=column, how="left")
        X_test = X_test.merge(FE, on=column, how="left")

    X_train[CATS] = X_train[CATS].astype("category")
    X_valid[CATS] = X_valid[CATS].astype("category")
    X_test[CATS] = X_test[CATS].astype("category")

    X_train = X_train.drop(['Price'], axis=1)



    XGBModel = XGBRegressor(
        random_state=42,
        max_depth=6,
        n_estimators=10000,
        enable_categorical=True,
        # device="cuda",
        subsample=0.8,
        colsample_bytree=0.5,
        learning_rate=0.02,
        min_child_weight=10,
        early_stopping_rounds=100,
        eval_metric='rmse',
        verbosity=0
    )

    LGBMModel = LGBMRegressor(
        random_state=42,
        max_depth=6,
        n_estimators=10000, 
        # device='gpu',
        # gpu_platform_id=0,
        # gpu_device_id=1,
        subsample=0.75,
        colsample_bytree=0.6,
        learning_rate=0.02,
        min_child_weight=10,
        early_stopping_rounds=100,
        verbose=-1
    )


    
    # Train Model and Record Test Predictions
    XGBModel.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=300)
    validations1 = XGBModel.predict(X_valid)
    predictions1 += XGBModel.predict(X_test)
    score1 = np.sqrt(mean_squared_error(y_valid, validations1))
    # if i == 0:
    #     xgb.plot_importance(XGBModel, max_num_features=20, importance_type='gain')
    #     plt.title("Feature Importances (XGBRegressor)")
    #     plt.show()

    # LGBMModel.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse')
    # validations2 = LGBMModel.predict(X_valid)
    # predictions2 += LGBMModel.predict(X_test)
    # score2 = np.sqrt(mean_squared_error(y_valid, validations2))
    # if i == 0:
    #     lgb.plot_importance(LGBMModel, max_num_features=20, importance_type='gain')
    #     plt.title("Feature Importances (LGBMRegressor)")
    #     plt.show()

    
    # COMBOvalidations = (0.5 * validations1) + (0.5 * validations2)
    # COMBOscore = np.sqrt(mean_squared_error(y_valid, COMBOvalidations))

    
    print(f"========================================================== K-Fold {i+1}")
    print(f"XGBModel  Validation RMSE: {score1}") 
    # print(f"LGBMModel Validation RMSE: {score2}")
    # print(f"Combined  Validation RMSE: {COMBOscore}")
    print(f"Runtime: {round(time.time() - start_time)} seconds")
    print(f"==========================================================")


predictions = predictions1 / FOLDS
# predictions = (0.5 * predictions1) + (0.5 * predictions2)
# predictions /= FOLDS


''' Generate submission.csv '''

sub_df = pd.DataFrame({'id': test_df["id"], 'Price': predictions})
sub_df.to_csv('submission.csv', index=False)
sub_df.head()

