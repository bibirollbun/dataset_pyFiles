from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from pandas.plotting import scatter_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.utils import shuffle
 
# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.linear_model import Perceptron
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split

#metrics
from sklearn.metrics import log_loss,make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


#Ensemble Models
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier

# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import StratifiedKFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate,StratifiedShuffleSplit

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.impute import SimpleImputer
from category_encoders import BinaryEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score

# GridSearchCV
from sklearn.model_selection import GridSearchCV


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Load training and test data into respective dataframes
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df =  pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


#Data clean up required
#remove all NaNs with column means

# Fill NaNs in all numeric columns with their respective means
# First lets do this for numeric data
for col in train_df.select_dtypes(include=float).columns:
    train_df.fillna({col:train_df[col].median()}, inplace=True)
# Next lets do this for boolean data
for col in train_df.select_dtypes(include=object).columns:
    print( col)
    train_df.fillna({col:train_df[col].mode()[0]}, inplace=True)

#repeat above for Test data
# First lets do this for numeric data
for col in test_df.select_dtypes(include=float).columns:
    test_df.fillna({col:test_df[col].median()}, inplace=True)
# Next lets do this for boolean data
for col in test_df.select_dtypes(include=object).columns:
    print( col)
    test_df.fillna({col:test_df[col].mode()[0]}, inplace=True)


X_train = train_df.copy()
X_train['Stage_fear'] = train_df['Stage_fear'].map({'Yes': 1, 'No': 0}) #converting yes/no to a 1/0
X_train['Drained_after_socializing'] = train_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
X_train['Personality'] = train_df['Personality'].map({'Introvert': 1, 'Extrovert': 0})


# Create a scatter matrix plot
scatter_matrix(train_df, alpha=0.8, diagonal='kde')
plt.suptitle('Scatter Matrix of DataFrame Columns', y=1.02) # Add a title
plt.show()


# # Calculate the correlation matrix
# corr_matrix = train_df.corr()

# # Create a heatmap
# plt.figure(figsize=(10, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix Heatmap')
# plt.show()


# Drop personality and id from training data
X_train = X_train.drop(columns='Personality') #taking out the label to use later in y
X_train = X_train.drop(columns='id')


X_test = test_df.copy()
X_test['Stage_fear'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
X_test['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
# Drop personality and id from testing  data
X_test = X_test.drop(columns='id')


y_train = train_df['Personality'].map({'Introvert': 1, 'Extrovert': 0})
y_train = pd.DataFrame(data=y_train)


std_scaler = StandardScaler()
X_test = std_scaler.fit_transform(X_test) # How cool that one line can do all your scaling 
X_train = std_scaler.fit_transform(X_train) 
X_train = pd.DataFrame(data=X_train)
X_test = pd.DataFrame(data=X_test)


X_train.columns


y_train.head()


X_test.head(10)


#Grid Search cross validations
#Logistic regression
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20] #,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50]
algorithm  = ['auto']
weights = ['uniform','distance']
LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
leaf_size = list(range(1,25,5)) # 1-50 in the increments of 5
#Define hyperparamenters
hyperparams = {}
gd=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=hyperparams,verbose=False,cv=10,scoring="roc_auc")
#gd=GridSearchCV(estimator=LogisticRegression(penalty="l1",solver="saga"),param_grid=hyperparams,verbose=True,cv=10,scoring=LogLoss)
gd.fit(X_train,y_train)


print(gd.best_score_)
print(gd.best_estimator_)
#Pick the bet estimator chosen by the grid search and use it to train our model
y_pred = gd.best_estimator_.predict(X_test)
y_pred = pd.DataFrame(y_pred)


y_pred.head()


temp = pd.DataFrame(pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")['id'])
temp['Personality'] = y_pred # Appending the output to the id read from the test data file
temp['Personality'] = temp['Personality'].map({1:'Introvert',0:'Extrovert'})
temp.to_csv("../working/submission.csv", index = False)

