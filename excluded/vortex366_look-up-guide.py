import pandas as pd


data = pd.read_csv('/kaggle/input/capstone-agri/train.csv').drop('Id',axis=1)
data


data.isnull().sum()


data.info()


data.dtypes


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


data['label'] = le.fit_transform(data['label'])
print(le.classes_)


data.dtypes


import seaborn as sns
import matplotlib.pyplot as plt


data.hist(bins=50,figsize=(20,15))
plt.show()


correlation_figure, correlation_axis = plt.subplots(figsize = (30,25))
corr_mtrx = data.corr()
correlation_axis = sns.heatmap(corr_mtrx, annot= True)

plt.xticks(rotation = 30, horizontalalignment = 'right', fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()


corr_matrix = data.corr()
corr_matrix['label'].sort_values(ascending=False)


from pandas.plotting import scatter_matrix


attributes = ['label','humidity','temperature','rainfall']
scatter_matrix(data[attributes],figsize=(12,8))


label = data['label']
data = data.drop(['label','ph','N','P','K'],axis=1)
data.head(5)


from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(data,label,test_size=0.2, random_state=42)


def display_results(y_val,prediction):
    # Print the Confusion Matrix and slice it into four pieces
    cm = confusion_matrix(y_val,prediction)
    # visualize confusion matrix with seaborn heatmap
    cm_matrix = pd.DataFrame(data=cm)
    print("Model Accuracy:",accuracy_score(y_val,prediction))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score


# instantiate the DecisionTreeClassifier model with criterion gini index

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=0)
# fit the model
clf_gini.fit(X_train, y_train)


prediction = clf_gini.predict(X_val)


print(classification_report(y_val,prediction))


plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_gini.fit(X_train, y_train))


display_results(y_val,prediction)


test = pd.read_csv('/kaggle/input/capstone-agri/test.csv').drop(['Id','ph','N','P','K'],axis=1)
test


results = clf_gini.predict(test)


class_encode = ['apple','banana','blackgram','chickpea','coconut','coffee','cotton',
 'grapes' ,'jute' ,'kidneybeans' ,'lentil', 'maize', 'mango', 'mothbeans',
 'mungbean', 'muskmelon', 'orange' ,'papaya', 'pigeonpeas','pomegranate',
 'rice', 'watermelon']


outcome =[]
for x in results:
    outcome.append(class_encode[x])


col0 = pd.read_csv("/kaggle/input/capstone-agri/submission.csv").drop(["label"],axis=1)


col1 = pd.Series(outcome, name="label")
submission = pd.concat([col0,col1],axis = 1)
submission.to_csv("submission.csv",index=False)

