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


import random
import seaborn
from  matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier


df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv ("/kaggle/input/playground-series-s5e8/test.csv")


df_train 


df_train.info()


df_train.describe()


df_train['y'].value_counts() ,df_train['y'].value_counts()[0]/df_train['y'].value_counts()[1]


index_0 = list(df_train[df_train['y'] == 0].index)
index_1 = list(df_train[df_train['y'] == 1].index)


index_0_removed_excess_lbl = random.sample(index_0,90488)
len(index_0_removed_excess_lbl)


final_indexes = index_0_removed_excess_lbl + index_1
len(final_indexes)


df_train_fin = df_train.iloc[final_indexes,:]
df_train_fin


seaborn.set(style = 'whitegrid') 
seaborn.violinplot(x ="y", 
             y ="duration", 
             data = df_train)
## when people spend more time on call , they open a deposit account 


seaborn.set(style = 'whitegrid') 
seaborn.violinplot(x ="y", 
             y ="balance", 
             data = df_train)



seaborn.set(style = 'whitegrid') 
seaborn.violinplot(x ="y", 
             y ="age", 
             data = df_train)



cross_tab = pd.crosstab(df_train['month'], df_train['y'])

cross_tab

# Plotting the stacked bar chart
#cross_tab.plot(kind='bar', stacked=True, figsize=(8, 6))
ax = cross_tab.plot.bar(stacked=True, figsize=(8, 6))

for container in ax.containers:
    ax.bar_label(container)
plt.title('y(deposit acct opened / not opnened ) vs month')
plt.xlabel('month')
plt.ylabel('count')
plt.xticks(rotation=0)
plt.legend(title='y')
plt.show()



df_train_fin.head(5)


cross_tab = pd.crosstab(df_train['job'], df_train['y'])

cross_tab

# Plotting the stacked bar chart
#cross_tab.plot(kind='bar', stacked=True, figsize=(8, 8))
ax = cross_tab.plot.bar(stacked=True, figsize=(8, 6))

for container in ax.containers:
    ax.bar_label(container)
plt.title('y(deposit acct opened / not opnened ) vs month')
plt.xlabel('month')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.legend(title='y')
plt.show()



cross_tab = pd.crosstab(df_train_fin['marital'], df_train_fin['y'])

cross_tab

# Plotting the stacked bar chart
#cross_tab.plot(kind='bar', stacked=True, figsize=(8, 8))
ax = cross_tab.plot.bar(stacked=True, figsize=(8, 6))

for container in ax.containers:
    ax.bar_label(container)
plt.title('y(deposit acct opened / not opnened ) vs marital')
plt.xlabel('marital')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.legend(title='y')
plt.show()



cross_tab = pd.crosstab(df_train['education'], df_train['y'])

cross_tab

# Plotting the stacked bar chart
#cross_tab.plot(kind='bar', stacked=True, figsize=(8, 8))
ax = cross_tab.plot.bar(stacked=True, figsize=(8, 6))

for container in ax.containers:
    ax.bar_label(container)
plt.title('y(deposit acct opened / not opnened ) vs education')
plt.xlabel('education')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.legend(title='y')
plt.show()



cross_tab = pd.crosstab(df_train['housing'], df_train['y'])

cross_tab

# Plotting the stacked bar chart
#cross_tab.plot(kind='bar', stacked=True, figsize=(8, 8))
ax = cross_tab.plot.bar(stacked=True, figsize=(8, 6))

for container in ax.containers:
    ax.bar_label(container)
plt.title('y(deposit acct opened / not opnened ) vs housing loan')
plt.xlabel('housing loan')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.legend(title='y')
plt.show()



cross_tab = pd.crosstab(df_train['loan'], df_train['y'])

cross_tab

# Plotting the stacked bar chart
#cross_tab.plot(kind='bar', stacked=True, figsize=(8, 8))
ax = cross_tab.plot.bar(stacked=True, figsize=(8, 6))

for container in ax.containers:
    ax.bar_label(container)
plt.title('y(deposit acct opened / not opnened ) vs personal loan')
plt.xlabel('personal loan')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.legend(title='y')
plt.show()



df_train.head()


cross_tab = pd.crosstab(df_train['day'], df_train['y'])

cross_tab

# Plotting the stacked bar chart
#cross_tab.plot(kind='bar', stacked=True, figsize=(8, 8))
ax = cross_tab.plot.bar(stacked=True, figsize=(15, 6))

for container in ax.containers:
    ax.bar_label(container)
plt.title('y(deposit acct opened / not opnened ) vs day')
plt.xlabel('day of month when contact made')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.legend(title='y')
plt.show()



cross_tab = pd.crosstab(df_train['campaign'], df_train['y'])

cross_tab

# Plotting the stacked bar chart
#cross_tab.plot(kind='bar', stacked=True, figsize=(8, 8))
ax = cross_tab.plot.bar(stacked=True, figsize=(15, 6))

for container in ax.containers:
    ax.bar_label(container)
plt.title('y(deposit acct opened / not opnened ) vs no of contacts made in this campaign')
plt.xlabel('no of contacts made in this campaign')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.legend(title='y')
plt.show()



seaborn.set(style = 'whitegrid') 
seaborn.violinplot(x ="y", 
             y ="pdays", 
             data = df_train)



cross_tab = pd.crosstab(df_train['poutcome'], df_train['y'])

cross_tab

# Plotting the stacked bar chart
#cross_tab.plot(kind='bar', stacked=True, figsize=(8, 8))
ax = cross_tab.plot.bar(stacked=True, figsize=(20, 20))

for container in ax.containers:
    ax.bar_label(container)
plt.title('y(deposit acct opened / not opnened ) vs poutcome')
plt.xlabel('poutcome')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.legend(title='y')
plt.show()



df_train_fin.columns


y,X= df_train_fin['y'].values,df_train_fin.drop(['id','y'],axis=1).values


X_col = df_train_fin.drop(['id','y'],axis=1).columns



X_train,X_val,y_train,y_val =  train_test_split(X,y,test_size= 0.25)


cat_feature_idxs = [] 
for i in range(X.shape[1]):
    if 'str' in str(type(X[0,i])):
        print(type(X[0,i]))
        cat_feature_idxs.append(i)





model = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.1, cat_features=cat_feature_idxs,
                           loss_function='Logloss', custom_metric=['AUC'], random_seed=42)


model.fit(X_train,y_train ,plot=True,use_best_model=True,eval_set = (X_val,y_val),cat_features =cat_feature_idxs)


# predicting accuracy
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

# print accuracy
print(f"Accuracy: {accuracy:.2f}")


importances = model.get_feature_importance()
feature_names = X_col
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[sorted_indices])
plt.xticks(range(len(feature_names)), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importance")
plt.show()


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_val, y_pred)
# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
seaborn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


id, X_test =df_test['id'], df_test.drop(['id'],axis=1).values


y_ = model.predict(X_test)



fin_dict = {'id':id,'y':y_}


fin_df = pd.DataFrame(fin_dict)
fin_df.to_csv('output.csv',index=False)




