import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


plt.style.use('seaborn-v0_8-whitegrid')


train = pd.read_csv('/kaggle/input/bank-chrun-classification/train.csv')
test = pd.read_csv('/kaggle/input/bank-chrun-classification/test.csv')


train.shape,test.shape


train.head()


train.tail()


train.info()


test.info()


train.drop(columns=['id','CustomerId','Surname'],inplace=True)
test.drop(columns=['id','CustomerId','Surname'],inplace=True)


sns.histplot(data=train,x=train['CreditScore'])
plt.show()


sns.countplot(data=train,x=train['Geography'],width=0.6,palette='flare')
plt.show()


sns.countplot(data=train,x='Gender',palette='deep')
plt.show()



sns.histplot(data=train,x='Age',palette='flare')
plt.show()


sns.countplot(data=train,x='Tenure')
plt.show()


sns.histplot(data=train,x='Balance',bins=[55000,65000,75000,85000,95000,105000,115000,125000,135000,145000,155000])
plt.show()


sns.countplot(data=train,x='NumOfProducts')
plt.show()


sns.countplot(data=train,x='HasCrCard',palette='deep')
plt.show()


sns.countplot(data=train,x=train['IsActiveMember'],width=0.4,palette='deep')
plt.show()


sns.histplot(data=train,x=train['EstimatedSalary'],kde=True)
plt.show()


sns.countplot(data=train,x=train['Exited']) ## output column
plt.show()


### Bivariate Analysis
pd.crosstab(train['Geography'],train['Gender'])


pd.crosstab(train['Geography'],train['Gender']).plot(kind='bar',color=['orange','teal'])
plt.show()


pd.crosstab(train['Gender'],train['Tenure'])



pd.crosstab(train['Gender'],train['Tenure']).plot(kind='bar')
plt.show()


num_cols = train.select_dtypes('number')
corr = num_cols.corr()
sns.heatmap(corr,cmap='coolwarm',annot=True)


sns.pairplot(data=train)


X = train.drop('Exited', axis=1)  
y = train['Exited']            

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
transformer = ColumnTransformer(transformers=[
    ('ohe',OneHotEncoder(sparse_output=False),[1,2]),
    ('std_scaler', StandardScaler(),[0, 3, 4, 5, 6]),
],remainder='passthrough')


parameters = {
    'C' : [0.001,0.01,0.1,1,2,5,10,20,30,40,50,60,70,100],
    'solver':['liblinear','newton-cholesky'],
    'max_iter': [10,100,1000]
}
clf = LogisticRegression()
cv = GridSearchCV(clf,parameters,cv=10,scoring='neg_mean_squared_error')
pipe = Pipeline(steps=[
    ('trf',transformer),
    ('classifier',cv)
])


pipe.fit(X_train,y_train)


y_pred = pipe.predict(X_val)
accuracy_score(y_pred,y_val)


print(classification_report(y_pred,y_val))


cv.best_estimator_


print(cv.best_params_)


cv.best_score_


cv.scoring

