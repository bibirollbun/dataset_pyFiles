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


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
import warnings 
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
import optuna 
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor , Pool
from tqdm.notebook import tqdm
from sklearn.pipeline import make_pipeline, Pipeline
import pandas as pd, numpy as np, polars as pl
from gc import collect
from tqdm.notebook import tqdm

from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.base import clone
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline, Pipeline

from warnings import filterwarnings 
filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt


!pip install category_encoders
from category_encoders import TargetEncoder





df_train = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv")
df_test = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv")


df_train.head()


df_test.head()


X_train = df_train.drop(['LOCAL_IDENTIFIER','CORRUCYSTIC_DENSITY' ],axis =1)
y_train = df_train['CORRUCYSTIC_DENSITY']
X_test = df_test.drop(['LOCAL_IDENTIFIER'],axis=1)


display(X_train.shape)
display(y_train.shape)




X_train.describe()


display(X_train.isnull().mean()*100)



y_train.isnull().sum()


X_test.isnull().mean()*100


plt.figure(figsize=(12, 8))
sns.heatmap(df_train.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("correlogram)", fontsize=14)
plt.show()

numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    plt.hist(X_train[col], bins=30, color='blue', alpha=0.7, density=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()


num_cols_train = X_train.select_dtypes(include=['int64', 'float64']).columns
num_cols_test = X_test.select_dtypes(include=['int64', 'float64']).columns
num_imputer = SimpleImputer(strategy='mean')
X_train[num_cols_train] = num_imputer.fit_transform(X_train[num_cols_train])
X_test[num_cols_test] = num_imputer.transform(X_test[num_cols_test])
y_train = y_train.fillna(y_train.mean())


cat_cols_train = X_train.select_dtypes(include='object').columns
cat_cols_test = X_test.select_dtypes(include='object').columns
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_cols_train] = cat_imputer.fit_transform(X_train[cat_cols_train])
X_test[cat_cols_train] = cat_imputer.transform(X_test[cat_cols_train])


cv = KFold(5, shuffle = True, random_state = 42)

Mdl_Master = {
    "XGB1R" : XGBRegressor(
                    n_estimators  = 600,
                    learning_rate = 0.005,
                    max_depth     = 7,
                    random_state  = 42,
                    colsample_bytree = 0.80,
                    reg_alpha        = 0.01,
                    reg_lambda       = 0.001,
                    enable_categorical = True,
                    verbosity          = 0,
                   ),
    "LGBM1R" : LGBMRegressor(
                        n_estimators     = 600,
                        learning_rate    = 0.005,
                        max_depth        = -1,
                        num_leaves       =31,
                        random_state     = 42,
                        subsample        = 0.60,
                        reg_alpha        = 0.01,
                        reg_lambda       = 0.001,                     
                        verbosity        = -1,
                            ),

    "CB1R" : CatBoostRegressor(
                    iterations       = 500,
                    learning_rate    = 0.005,
                    max_depth        = 5,
                    l2_leaf_reg      = 0.65,
                    loss_function    = "RMSE",
                    colsample_bylevel = 0.55,
                    verbose           = 0,
                    random_state      = 42,
                ),
    "lasso" : Lasso(alpha=0.001, max_iter=5000, random_state=42)
    
    
}



OOF_Preds, Mdl_Preds = [], []

for fold_nb, (train_idx, dev_idx) in tqdm(enumerate( cv.split(X_train, y_train) ) ):

    print(f"---> Starting Fold {fold_nb + 1}")

    Xtr, ytr   = X_train.iloc[train_idx], y_train.iloc[train_idx]
    Xdev, ydev = X_train.iloc[dev_idx],   y_train.iloc[dev_idx]
    Xt         = X_test.copy()

    oof_preds, test_preds = [], []
    
    for method, mymodel in tqdm( Mdl_Master.items() ):

        model = make_pipeline(*[TargetEncoder(), mymodel])
        model.fit(Xtr.values, ytr.values)
        dev_preds = pd.DataFrame( model.predict(Xdev.values), index = Xdev.index, columns = ["Preds"])
        mdl_preds = pd.DataFrame( model.predict(Xt.values), index = X_test.index, columns = ["Preds"])

        oof_preds.append(dev_preds)
        test_preds.append(mdl_preds)

    oof_preds = pd.concat(oof_preds, axis= 0).groupby(level = 0).mean()
    test_preds = pd.concat(test_preds, axis= 0).groupby(level = 0).mean()
    OOF_Preds.append(oof_preds)
    Mdl_Preds.append(test_preds)
    

OOF_Preds = pd.concat(OOF_Preds, axis= 0).sort_index(ascending = True)
Mdl_Preds = (
    pd.concat(Mdl_Preds, axis= 0).
    sort_index(ascending = True).
    groupby(level = 0).
    mean()
)

score = np.sqrt(mean_squared_error(y_train, OOF_Preds.values.flatten()))
print(f"\n---> Combined Score = {score:,.8f}\n\n")        


submission = pd.DataFrame({
    "LOCAL_IDENTIFIER": df_test["LOCAL_IDENTIFIER"].astype(int),
    "CORRUCYSTIC_DENSITY": Mdl_Preds["Preds"].values.astype(float)
})

submission.to_csv("submission.csv", index=False)
display(submission)

