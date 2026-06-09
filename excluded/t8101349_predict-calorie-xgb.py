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


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import KBinsDiscretizer
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df = df.drop('id',axis = 1)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test = df_test.drop('id',axis=1)
df = df.drop_duplicates()


df.describe()


df.isnull().sum()


sns.heatmap(df.corr(numeric_only=True), annot = True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()



sns.histplot(df["Calories"], kde=True)
plt.title("原始 Calories 分布")
plt.show()

sns.histplot(np.log1p(df["Calories"]), kde=True)
plt.title("Log1p 後的 Calories 分布")
plt.show()




from scipy.stats import boxcox

x = df["Calories"].copy()
x = x[x > 0]  # Box-Cox 不能有0或負數

x_boxcox, lam = boxcox(x)

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.histplot(x, kde=True)
plt.title("原始分布")

plt.subplot(1, 2, 2)
sns.histplot(x_boxcox, kde=True)
plt.title(f"Box-Cox 後分布 (λ={lam:.2f})")
plt.show()



# one-hot
df['male'] = (df['Sex'] == 'male').astype(int)
df['female'] = (df['Sex'] == 'female').astype(int)



#label encoding
df['Age_bin'] = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 100], labels=False)



df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
#df["HR_dur"] = df["Heart_Rate"] * df["Duration"]


#(Body_Temp − mean)² 
df["Temp_squar"] = (df["Body_Temp"] - df["Body_Temp"].mean())**2



# standard normalize
numerical_cols = ["Duration","Heart_Rate","Temp_squar","BMI"]
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])



df


def feature_engineering(df : pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    df['male'] = (df['Sex'] == 'male').astype(int)
    df['female'] = (df['Sex'] == 'female').astype(int)
    df['Age_bin'] = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 100], labels=False)
    
           
    return df.drop(["Age","Sex"], axis = 1 )


def feature_engineering0(df : pd.DataFrame, numeric_cols: list) -> pd.DataFrame:


    df['male'] = (df['Sex'] == 'male').astype(int)
    df['female'] = (df['Sex'] == 'female').astype(int)
    df['Age_bin'] = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 100], labels=False)
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    #df["HR_dur"] = df["Heart_Rate"] * df["Duration"]
    df["Temp_squar"] = (df["Body_Temp"] - df["Body_Temp"].mean())**2
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df.drop(["Age","Sex","Weight","Height","Body_Temp"], axis = 1 )


#y = np.log1p(df["Calories"])   #logp1
y = df["Calories"]
X = df.drop(['Calories',"Age","Sex","Weight","Height","Body_Temp"], axis = 1 )



"""
df["BMI"] = train["Weight"] / (df["Height"]/100)**2
df["Duration_per_kg"] = df["Duration"] / df["Weight"]
df["HR_per_min"] = df["Heart_Rate"] / df["Duration"]
df["HR_dur"] = df["Heart_Rate"] * df["Duration"]
df["Temp_squar"]= (df["Body_Temp"]- mean(df["Body_Temp"]))**2
"""


X_test = feature_engineering0(df_test,numerical_cols)


FOLDS = 40
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))          # OOF 預測（訓練集用）
pred_xgb = np.zeros(len(X_test))    # 測試集預測

## XGBOOST
xgb_model = XGBRegressor(
    max_depth=10,
    colsample_bytree=0.75,
    subsample=0.9,
    n_estimators=2000,
    learning_rate=0.01,
    gamma=0.01,
    max_delta_step=2,
    early_stopping_rounds=100,
    eval_metric="rmse",
    enable_categorical=True,
    device = 'cuda',
    tree_method='gpu_hist')


from scipy.special import inv_boxcox

y_boxcox, lam = boxcox(y)


for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/{FOLDS}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_boxcox[train_idx], y.iloc[val_idx]

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_boxcox[val_idx])],
        verbose=False
    )

    preds_val = xgb_model.predict(X_val)
    preds_val_inv = inv_boxcox(preds_val, lam) 

    # 預測驗證集
    oof_xgb[val_idx] = preds_val_inv
    
    # 累加測試集預測（待會會除以 FOLDS 得到平均）
    preds_test = xgb_model.predict(X_test)
    preds_test_inv = inv_boxcox(preds_test, lam)
    
    pred_xgb += preds_test_inv
    
    # fold 的 RMSE
    rmse = mean_squared_error(y_val, preds_val_inv, squared=False)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

pred_xgb /= FOLDS
# ➤ 評估：用原始 y 來比 RMSE
rmse = mean_squared_error(y, oof_xgb, squared=False)
print(f"RMSE after Box-Cox on y: {rmse:.4f}")


"""
# logp1 is better
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}/{FOLDS}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    
    # 預測驗證集
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)
    
    # 累加測試集預測（待會會除以 FOLDS 得到平均）
    pred_xgb += xgb_model.predict(X_test) / FOLDS
    
    # fold 的 RMSE
    rmse = mean_squared_error(y_valid, oof_xgb[valid_idx], squared=False)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

# 全部資料的 OOF RMSE
oof_rmse = mean_squared_error(y, oof_xgb, squared=False)
print(f"\nOverall OOF RMSE: {oof_rmse:.4f}")
"""


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

submission["Calories"] = pred_xgb
#submission["Calories"] = np.expm1(pred_xgb)   #logp1
submission.to_csv("submission.csv", index=False)
print('submission saved')
submission.head()

