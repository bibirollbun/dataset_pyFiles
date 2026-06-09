import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_sample = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
df_Train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_Train.head()


df_Train.info()


df_Train.isnull().sum()


df_Train.duplicated().sum()


df_Train.shape


df_Train.describe().T


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(df_Train.drop(['id','rainfall'],axis=1),df_Train['rainfall'])
print("Score of Train: ",model.score(df_Train.drop(['id','rainfall'],axis=1),df_Train['rainfall']))

result = df_test.merge(df_sample, on='id', how='left')

result.dropna(axis=0,inplace=True)

print("Score of Test: ", model.score(result.drop(['id','rainfall'],axis=1),result['rainfall']))


df_Train.hist(figsize=(15,10))
plt.show()


# log Transform.
right_skewed_Futers = ['sunshine' ,'winddirection','windspeed']
for i in right_skewed_Futers:
    df_Train[i] = np.log(df_Train[i] + 1)

# inverse Log Transform.

left_skewed_Futers = ['maxtemp','temparature', 'mintemp', 'dewpoint', 'humidity','cloud']
for i in left_skewed_Futers:
    df_Train[i] = np.log(df_Train[i].max() - df_Train[i] + 1)



df_Train.hist(figsize=(15,10))
plt.show()


plt.figure(figsize=(15,7))
plt.title("Correlation of Feuters")
sns.heatmap(df_Train.corr(),annot=True)
plt.show()


df_Train.info()     


def quarter(x):
    if (x%365) <91:
        return "Q1"
    elif (x%365)<181:
        return "Q2"
    elif (x%365)<270:
        return "Q3"
    return "Q4"


df_Train['quarters'] = df_Train['day'].apply(quarter)


exclude_cols = {'id', 'day','quarters'}
for i in df_Train.columns.difference(exclude_cols):
    plt.figure(figsize=(10,7))
    sns.lineplot(data=df_Train, x='quarters', y=i, hue='rainfall')
    plt.show()


def countplot_ratio(data, x, hue=None, ax=None, rotate_xlabel=0):
    ax = sns.countplot(data=data, x=x, hue=hue, ax=ax)
    # Rotate x-axis labels based on the rotate_xlabel parameter
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_xlabel)
    ax.set_title(x + " Distributions")
    
    total = float(len(data))
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2., height + 3,
                    '{:.2f}%'.format((height / total) * 100), 
                    fontsize=12, weight='bold', ha="center")


plt.figure(figsize=(10,7))
countplot_ratio(df_Train,x='rainfall')
plt.show()


accur_train = []
accur_test = []
name = ['LogisticRegression','KNeighbors','svm','DecisionTree','RandomForest','naive_bayes','GradientBoosting']


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])
print("Score of Train: ",model.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000//10)
accur_train.append(model.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000//10)


df_sample.shape


df_test.shape


df_test.head()


df_sample.head()


result = df_test.merge(df_sample, on='id', how='left')


result.dropna(axis=0,inplace=True)


result.isnull().sum()


print("Score of Test: ",model.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000//10)
accur_test.append(model.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000//10)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])
print("Score of Training: ",knn.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000 // 10)
accur_train.append(knn.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000 // 10)

print("Score of Training: ",knn.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000 // 10)
accur_test.append(knn.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000 // 10)


from sklearn.svm import SVC

svm = SVC(kernel='linear')

svm.fit(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])
print("Score of Training: ",svm.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000 // 10)
accur_train.append(svm.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000 // 10)
print("Score of Training: ",svm.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000 // 10)
accur_test.append(svm.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000 // 10)


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5)

tree.fit(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])
print("Score of Training: ",tree.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000 // 10)
accur_train.append(tree.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000 // 10)
print("Score of Training: ",tree.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000 // 10)
accur_test.append(tree.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000 // 10)


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

forest.fit(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])
print("Score of Training: ",forest.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000 // 10)
accur_train.append(forest.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000 // 10)
print("Score of Training: ",forest.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000 // 10)
accur_test.append(forest.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000 // 10)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])
print("Score of Training: ",nb.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000//10)
accur_train.append(nb.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000//10)
print("Score of Training: ",nb.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000//10)
accur_test.append(nb.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000//10)


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)

gb.fit(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])
print("Score of Training: ",gb.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000//10)
accur_train.append(gb.score(df_Train.drop(['id','rainfall','quarters'],axis=1),df_Train['rainfall'])*1000//10)
print("Score of Training: ",gb.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000//10)
accur_test.append(gb.score(result.drop(['id','rainfall'],axis=1),result['rainfall'])*1000//10)


Evaluation_Table = pd.DataFrame(data=[name,accur_train,accur_test],index=[['Model Name','Accurcy of Train','Accurcy of Test']])

Evaluation_Table = Evaluation_Table.T

Evaluation_Table.head(7)

