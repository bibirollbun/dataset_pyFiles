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


import pandas as pd
print("Hello")


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import mean_squared_error 




df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')




df_train.head()


df_train.describe()


df_test.head()


df_sub.head()




df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)



df_train.shape,df_test.shape

df_train.isnull().sum()


df_train.shape,df_test.shape,df_sub.shape


df_train.select_dtypes(include='number').corr()


df_train.dtypes
common_cols=df_train.columns.intersection(df_test.columns)


print(len(common_cols),len(df_train.columns),len(df_test.columns))

dif=df_train.columns.difference(df_test.columns)
dif


numerical_cols=df_train[common_cols].select_dtypes(include=['int64','float']).columns
categorical_cols=df_train[common_cols].select_dtypes(include=['object']).columns


df_train['Episode_Length_minutes'].head().isnull().sum()


len(df_train['Genre'].unique())


df_train[numerical_cols]=df_train[numerical_cols].fillna(df_train[numerical_cols].median())
df_test[numerical_cols]=df_test[numerical_cols].fillna(df_test[numerical_cols].median())
df_train[categorical_cols]=df_train[categorical_cols].apply(lambda x:x.fillna(x.mode()[0]))
df_test[categorical_cols]=df_test[categorical_cols].apply(lambda x:x.fillna(x.mode()[0]))


df_train[numerical_cols].head()


print("Train Columns:",list(df_train.columns))


!pip install --upgrade pip
!pip install xgboost
!pip install lightgbm
!pip install category_encoders
!pip install catboost


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import gc
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
import xgboost as xgb


y=df_train['Listening_Time_minutes']


df_train = pd.read_csv('/kaggle/input/podcasts/train.csv')
df_test  = pd.read_csv('/kaggle/input/podcasts/test.csv')




X = df_train
X_test = df_test


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

xgb_params = {
    'n_estimators': 565,
    'max_depth': 14,
    'learning_rate': 0.04222221,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist', 
    'n_jobs': -1,
    'eval_metric': 'rmse'
}


n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

scores = []
test_preds = np.zeros(len(X_test)) 

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits}...")    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]   
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=100
        
    )    
    val_pred = model.predict(X_val)
    score = rmse(y_val, val_pred)
    scores.append(score)
    test_preds += model.predict(X_test) / n_splits      
    print(f"Fold {fold + 1} RMSE: {score:.4f}")

print(f'Optimized Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

sns.set(style="whitegrid")

feature_importance = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
residuals = y_val - val_pred

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

sns.barplot(
    x="Importance", 
    y="Feature", 
    data=importance_df, 
    palette="viridis", 
    ax=axes[0, 0]
)
axes[0, 0].set_title("Feature Importance (XGBoost)")
axes[0, 0].set_xlabel("Importance Score")
axes[0, 0].set_ylabel("Features")
for i, v in enumerate(importance_df['Importance']):
    axes[0, 0].text(v + 0.001, i, f"{v:.3f}", va='center', fontsize=9)

sns.scatterplot(x=y_val, y=val_pred, alpha=0.6, edgecolor="k", ax=axes[0, 1])
axes[0, 1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r', lw=2)
axes[0, 1].set_title("Actual vs. Predicted Listening Time")
axes[0, 1].set_xlabel("Actual Listening Time")
axes[0, 1].set_ylabel("Predicted Listening Time")

sns.histplot(residuals, bins=30, kde=True, color='skyblue', ax=axes[1, 0])
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_title("Residual Distribution")
axes[1, 0].set_xlabel("Residual Value")
axes[1, 0].set_ylabel("Frequency")

sns.histplot(test_preds, bins=30, kde=True, color='lightgreen', ax=axes[1, 1])
axes[1, 1].set_title("Distribution of Test Predictions")
axes[1, 1].set_xlabel("Predicted Listening Time")
axes[1, 1].set_ylabel("Frequency")

sns.scatterplot(x=val_pred, y=residuals, alpha=0.6, edgecolor="k", ax=axes[2, 0])
axes[2, 0].axhline(0, color='red', linestyle='--')
axes[2, 0].set_title("Residuals vs. Predicted Values")
axes[2, 0].set_xlabel("Predicted Listening Time")
axes[2, 0].set_ylabel("Residuals")

stats.probplot(residuals, dist="norm", plot=axes[2, 1])
axes[2, 1].set_title("QQ Plot of Residuals")

plt.tight_layout()
plt.show()




df_sub['Listening_Time_minutes'] = test_preds
df_sub.to_csv('submission.csv', index=False)










df_sub.head(10)



