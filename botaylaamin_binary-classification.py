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


import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway


df = pd.read_csv(r"/kaggle/input/playground-series-s5e8/train.csv")
df.head()


df_test = pd.read_csv(r"/kaggle/input/playground-series-s5e8/test.csv")
df_test_id = df_test['id']


df.columns


df.info()


df_test.info()


df.describe().T


df.describe(include = "O").T


cat_cols = df.select_dtypes("object").columns
num_cols = df.select_dtypes(["int","float"]).columns


cat_cols


num_cols


len(cat_cols)


plt.figure(figsize=(10,8))

for i,col in enumerate(cat_cols):
    plt.subplot(3,3,i+1)
    sns.countplot(data= df , x=col)
    plt.title(f"The Distribution of {col}")
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10,8))

for i,col in enumerate(cat_cols):
    plt.subplot(3,3,i+1)
    sns.countplot(data= df , x=col , hue ="y")
    plt.title(f"The Distribution of {col}")
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


for col in cat_cols:
    print(f"colmn {col} has {df[col].nunique()} labels")


for col in cat_cols:
    print(f"colmn  {df[col].value_counts()} labels")
    print(20*'*')


len(num_cols)


plt.figure(figsize=(10,8))

for i,col in enumerate(num_cols):
    plt.subplot(3,3,i+1)
    sns.histplot(data= df , x=col , kde =True)
    plt.title(f"The Distribution of {col}")
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


from scipy.stats import skew

for col in num_cols:
    print(f"Skewness of {col} is {skew(df[col])}")


# get target distribution

df["y"].value_counts()


df["y"].value_counts(normalize = True)


corr_mat = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_mat))
sns.heatmap(corr_mat ,mask = mask , annot = True,fmt='.2f' )
plt.title("Heatmap of numerical columns")
plt.show()


corr_mat


for col in num_cols:
    group1= df[df['y'] == 0][col]
    group2 = df[df['y'] == 1][col]
    f_statics, p_value = f_oneway(group1, group2)
    print(f"P_value of {col}  : {p_value}")


Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5* IQR
upper_bound = Q3 + 1.5* IQR

outliers = (df[num_cols] < lower_bound) | (df[num_cols] > upper_bound)

outliers.sum()


df.duplicated().sum()


df.drop("id" , axis = 1 , inplace = True)


df_test.drop("id" , axis = 1 , inplace = True)


df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
df['previous_log'] = np.log1p(df['previous'].clip(lower=0))

df_test['balance_log'] = np.log1p(df_test['balance'].clip(lower=0))
df_test['previous_log'] = np.log1p(df_test['previous'].clip(lower=0))




df["contact_before"] = (df['pdays'] != -1).astype(int)

df_test["contact_before"] = (df_test['pdays'] != -1).astype(int)


df= pd.get_dummies(df , columns = cat_cols , drop_first = True)


df_test = pd.get_dummies(df_test , columns = cat_cols , drop_first = True)


X = df.drop('y' , axis =1)
Y =df['y']


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


X_scaled = pd.DataFrame(X_scaled , columns = X.columns)


X_scaled


df_test_scaled = pd.DataFrame(scaler.transform(df_test) , columns = df_test.columns)


from sklearn.model_selection import train_test_split

x_train, x_test ,y_train,y_test = train_test_split(X_scaled , Y , test_size =0.2 , random_state = 41)


x_test.shape 


y_test.shape


import xgboost as xgb
from sklearn.metrics  import accuracy_score,classification_report , roc_auc_score

clf = xgb.XGBClassifier()
clf.fit(x_train , y_train)

pred = clf.predict(x_test)

print(classification_report(y_test , pred , digits =4))
print(f"ROC_AUC score : {roc_auc_score(y_test , pred)}")


from lightgbm import LGBMClassifier
from sklearn.metrics  import accuracy_score,classification_report , roc_auc_score


clf = LGBMClassifier()
clf.fit(x_train, y_train)

pred = clf.predict(x_test)

print(classification_report(y_test , pred , digits =4))
print(f"ROC_AUC score : {roc_auc_score(y_test , pred)}")


feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
})

feature_importances.sort_values(by="importance", ascending=False, inplace=True)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importances, x='importance', y='feature', palette="viridis")
plt.title('Feature Importances from LGBMClassifier')
plt.tight_layout()
plt.show()


feature_importances


top_36_features = feature_importances['feature'].head(36).tolist()
x_train_top36 = x_train[top_36_features]
x_test_top36 = x_test[top_36_features]

clf = LGBMClassifier()
clf.fit(x_train_top36, y_train)

pred = clf.predict(x_test_top36)

print(classification_report(y_test, pred, digits=4))
print(f"ROC_AUC score : {roc_auc_score(y_test , pred)}")



df_test_scaled = df_test_scaled[top_36_features]



test_pred = clf.predict_proba(df_test_scaled)[:, 1]



submission = pd.DataFrame({
    "id": df_test_id,  
    "y": test_pred
})
submission.to_csv("submission.csv", index=False)



test_pred




