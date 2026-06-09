import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import roc_curve, auc,classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score


train=pd.read_csv("/kaggle/input/rainfall/train.csv",na_filter=True,index_col=0)
test=pd.read_csv("/kaggle/input/rainfall/test.csv",na_filter=True)


train = train.rename(columns={'temparature': 'temperature'})
test = test.rename(columns={'temparature': 'temperature'})

for colname in train:
    print(colname)


# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.pairplot(train.iloc[:, 1:12], hue = 'rainfall', diag_kind='kde')



train.plot.line(y=["mintemp","temperature","maxtemp"],figsize=(20,10))
plt.title("Temperatures")
plt.show


rows_with_na = test[test.isna().any(axis=1)]
print(rows_with_na.head())


val=int(test["winddirection"].median())
test=test.fillna(val)


#Checking that there are no remainings missing values 
rows_with_na = test[test.isna().any(axis=1)]
print(rows_with_na)


X=train.loc[:,"pressure":"windspeed"].values
y =train.rainfall.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


#Create the LR model 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
model = LogisticRegression()


grid = { 'C':[0.1, 1, 10, 100, 1000] }
model = GridSearchCV(LogisticRegression(penalty="l2", max_iter=1000),
grid, scoring="accuracy", cv=5)
model.fit(X_train, y_train)


print(model.best_params_)


coef = pd.Series(model.best_estimator_.coef_[0,:])
coef.index =train.columns[1:-1]
print(coef)


from sklearn.metrics import accuracy_score
print(f"The accuracy is {accuracy_score(y_train, model.predict(X_train)):,.3f}")


#The predictions in the X_test
y_pred=model.predict(X_test)


print(classification_report(y_test,y_pred))


#Visual heatmap
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt=".0f")


#Calculations to get the ROC curve
y_predprob=model.predict_proba(X_train)
fpr,tpr,threshold = roc_curve(y_train,y_predprob[:,1]) #target 1 probability


plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig('ROC_curve.png')


print(f"The area under the curve of the ROC plot is {auc(fpr,tpr):.2f}")


test2=test.loc[:,"pressure":"windspeed"].values
r_test=pd.Series(model.predict_proba(test2)[:,1])


output= pd.DataFrame({'id':test["id"],'rainfall':r_test})


output.to_csv('/kaggle/working/submission.csv',index=False)

