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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
import statistics  
from sklearn.metrics import classification_report as cr
from sklearn.metrics import classification_report,roc_curve
from sklearn import metrics 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVC as svc
from sklearn.linear_model import LogisticRegression as lr 
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda 
from sklearn.svm import SVC
from xgboost import XGBRFClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix as cm, ConfusionMatrixDisplay as cmd
from sklearn.linear_model import LassoCV
from numpy import arange
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error, mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error as MAPE
#Libraries for model selection
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

#Libraries for models
from sklearn.decomposition import PCA 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor,Ridge
from sklearn.tree import plot_tree, ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error as mlse
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression


df_train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
X_test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')


df_train.head()


df_train.shape


df_train.describe()


sns.countplot(x="NObeyesdad", data = df_train)
 
plt.xlabel("Category")
plt.ylabel("Frequency")
plt.title("Obesity Class Distribution")
plt.xticks(rotation = 45)
plt.show()


# Assuming df is your DataFrame
def plot_histograms(df_train):
    numeric_columns = df_train.select_dtypes(include=['number']).columns
    df_train[numeric_columns].hist(bins=5, figsize=(15, 10), layout=(len(numeric_columns) // 3 + 1, 3))
    plt.tight_layout()
    plt.show()
df_train = df_train.drop(['id'],axis = 1)    
plot_histograms(df_train) 


sns.boxplot( data = df_train, orient="h")
# display
plt.show()


sns.boxplot( data = df_train.Age, orient="h")
# display
plt.show()


sns.boxplot( data = df_train.Height, orient="h")
# display
plt.show()


sns.boxplot( data = df_train.Weight, orient="h")
# display
plt.show()


df_train.corr(numeric_only=True)


plt.figure(figsize=(10,10))
sns.heatmap(df_train.corr(numeric_only=True),annot=True,cmap='Blues');


df_train['NObeyesdad'].unique()


df_train.info()


df_train.isnull().sum()


df_train.duplicated().sum()


df_train.describe()


la = LabelEncoder()
la1 = LabelEncoder()


# Seperate train data into object and non-object
df_train_obj = df_train.select_dtypes(include='object')
df_train_non_obj = df_train.select_dtypes(exclude='object')


# Seperate test data into object and non-object
X_test_obj = X_test.select_dtypes(include='object')
X_test_non_obj = X_test.select_dtypes(exclude='object')


# Tranform object data to numeric in df_train 
for i in range(0 , df_train_obj.shape[1]):
    df_train_obj.iloc[:,i]=la.fit_transform(df_train_obj.iloc[:,i])
df_train_obj = df_train_obj.astype('int')

# Tranform object data to numeric in X_test 
for i in range(0 , X_test_obj.shape[1]):
    X_test_obj.iloc[:,i]=la1.fit_transform(X_test_obj.iloc[:,i])
X_test_obj = X_test_obj.astype('int')


print(la.classes_)


df_train_obj.head()


df_train_obj["NObeyesdad"].value_counts()


sns.countplot(x="NObeyesdad", data = df_train_obj)
plt.xlabel("NObeyesdad")
plt.ylabel("Frequency")
plt.title("Obeisity Category")
plt.show()


X_test_obj


df_train_obj.info()


train_data =pd.concat([df_train_obj, df_train_non_obj], axis=1)
X_test =pd.concat([X_test_obj, X_test_non_obj], axis=1)


ss =StandardScaler()


train_data['Age'] = ss.fit_transform(train_data[['Age']]) 
train_data['Weight'] = ss.fit_transform(train_data[['Weight']])

X_test['Age'] = ss.fit_transform(X_test[['Age']])
X_test['Weight'] = ss.fit_transform(X_test[['Weight']])


train_data


train_data.head()


train_data.head()


X_test.head()


#x = train_data.drop(['NObeyesdad', 'id'], axis=1)
X_train = train_data.drop(['NObeyesdad'], axis=1)
y = train_data['NObeyesdad']


test_data = X_test.drop('id', axis=1)


x_train, x_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=21)


X_train.shape


model_lr = lr(multi_class='multinomial', solver='lbfgs')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
# evaluate the model and collect the scores 
n_scores = cross_val_score(model_lr, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)


print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores))) 
mylr=model_lr.fit(x_train, y_train) 
print(mylr.score(x_train, y_train)) 
mypred=mylr.predict(x_val) 
print(cr(y_val,mypred)) 


tmp=cm(y_val.astype('int'), mypred)
disp=cmd(tmp)
disp.plot()
print("Figure 1: Confusion Matrix for Multinomial Logistic Regression Model")
print('[0, 1, 2, 3, 4, 5, 6] = [Insufficient_Weight,Normal_Weight,Obesity_Type_I,Obesity_Type_II,Obesity_Type_III,Overweight_Level_I,Overweight_Level_II]')


model_lda=LDA() 
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
# evaluate the model and collect the scores 
n_scores = cross_val_score(model_lda, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)


print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores))) 
mylda=model_lda.fit(x_train, y_train) 
print(mylda.score(x_train, y_train)) 
mypred=mylda.predict(x_val) 
print(cr(y_val,mypred)) 


tmp=cm(y_val.astype('int'), mypred)
disp=cmd(tmp)
disp.plot()
print("Figure 2: Confusion Matrix for LDA Model")
print('[0, 1, 2, 3, 4, 5, 6] = [Insufficient_Weight,Normal_Weight,Obesity_Type_I,Obesity_Type_II,Obesity_Type_III,Overweight_Level_I,Overweight_Level_II]')


model_nb=GaussianNB() 
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
# evaluate the model and collect the scores 
n_scores = cross_val_score(model_nb, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)


print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores))) 
mygnb=model_nb.fit(x_train, y_train) 
print(mygnb.score(x_train, y_train)) 
mypred=mygnb.predict(x_val) 
print(cr(y_val,mypred)) 


tmp=cm(y_val.astype('int'), mypred)
disp=cmd(tmp)
disp.plot()
print("Figure 3: Confusion Matrix for Gaussian Naive Bayes Model")
print('[0, 1, 2, 3, 4, 5, 6] = [Insufficient_Weight,Normal_Weight,Obesity_Type_I,Obesity_Type_II,Obesity_Type_III,Overweight_Level_I,Overweight_Level_II]')


model_svc = SVC(kernel='linear')
#model_svc = svc()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
#evaluate the model and collect the scores 
n_scores = cross_val_score(model_svc, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)


print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores))) 
mysvc=model_svc.fit(x_train, y_train) 
print(mysvc.score(x_train, y_train)) 
mypred=mysvc.predict(x_val)
print(cr(y_val,mypred)) 


tmp=cm(y_val.astype('int'), mypred)
disp=cmd(tmp)
disp.plot()
print('Figure 4: Confusion Matrix for SVM Model')
print('[0, 1, 2, 3, 4, 5, 6] = [Insufficient_Weight,Normal_Weight,Obesity_Type_I,Obesity_Type_II,Obesity_Type_III,Overweight_Level_I,Overweight_Level_II]')


prediction = model_svc.predict(test_data)


submission = pd.DataFrame({'id': X_test['id'], 'NObeyesdad': prediction})


submission.head()


submission['NObeyesdad'].unique()


submission['NObeyesdad'] = submission['NObeyesdad'].replace(to_replace=[0, 1, 2, 3, 4, 5, 6], value=[
    'Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II', 
    'Obesity_Type_III', 'Overweight_Level_I','Overweight_Level_II'])
submission



submission.to_csv('submission01.csv', index=False)

