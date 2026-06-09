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
df_train=pd.read_csv('/kaggle/input/churn-challenge-ai/train.csv')
df_test=pd.read_csv('/kaggle/input/churn-challenge-ai/test.csv')


print(f'train nan value={df_train.isnull().sum().sum()}')
print(f'test nan value={df_test.isnull().sum().sum()}')


df_all = pd.concat([df_train, df_test]).reset_index(drop=True)


group_count_percentage =df_all.groupby(['CustomerId', 'Surname', 'Geography', 'Gender']).ngroups/len(df_all)
print(f"surname percent unique groups: {group_count_percentage*100}%")
group_count_percentage = df_all.groupby(['Surname', 'Geography', 'Gender']).ngroups/len(df_all)
print(f"surname percent unique groups: {group_count_percentage*100}%")


for col in ['Surname','CustomerId','Geography', 'Gender','Age']:
    per=df_all[col].nunique()/len(df_all[col])
    print(f'{col} unique same percentage ={per*100}')


group_count_percentage = df_all.groupby(['Surname']).ngroups/len(df_all)
print(f"surname percent unique groups: {group_count_percentage*100}%")


group_count_percentage = df_all.groupby(['CustomerId']).ngroups/len(df_all)
print(f"surname percent unique groups: {group_count_percentage*100}%")


group_count_percentage = df_all.groupby(['Surname','CustomerId']).ngroups/len(df_all)
print(f"surname percent unique groups: {group_count_percentage*100}%")


group_count_percentage = df_all.groupby(['CustomerId','Surname', 'Gender', 'Geography']).ngroups/len(df_all)
print(f"percent unique groups: {group_count_percentage*100}%")


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


obj_feats=list(df_train.select_dtypes(include=['object','category']).columns)
df_x=df_train.drop('Exited',axis=1,inplace=False)
df_y=df_train['Exited']
X_train, X_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.2, random_state=1
)
cat_interp = CatBoostClassifier(verbose=False, 
                                cat_features=obj_feats, 
                                early_stopping_rounds=200)

cat_interp.fit(X_train, y_train, eval_set=(X_test, y_test))

feature_importance = cat_interp.get_feature_importance()
feature_names = X_train.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)


import matplotlib.pyplot as plt


# 繪圖
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1])  # 反向畫圖讓最大在上
plt.xlabel('Importance')
plt.title(f'Feature Importances')
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
col_name=df_train.select_dtypes(include=['int64','float64']).columns
col_name=col_name.drop('Exited')
col=4
row=int(np.ceil(len(col_name)/col))
fig, axes = plt.subplots(row, col, figsize=(5*row, 4*col))
axes = axes.flatten() 
for i,c in enumerate(col_name):
    uniquevalue=df_train[c].unique()
    #print(f'{c} unique value={len(uniquevalue)}')

    if len(uniquevalue)>10:
        sns.boxplot(x='Exited', y=f'{c}', data=df_train, ax=axes[i])
        axes[i].set_title(f'{c}')
    if len(uniquevalue)<=10:
        exit_rates =df_train.groupby(f'{c}')['Exited'].mean() * 100
        sns.barplot(x=exit_rates.index, y=exit_rates,ax=axes[i])
        axes[i].set_title(f'{c}')

plt.tight_layout()
plt.show()
        




