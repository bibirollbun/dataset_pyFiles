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
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from scipy.stats import zscore


train=pd.read_csv('/kaggle/input/dsaa-6100-titanic-survival-using-decision-trees/train.csv')


train.head()


print('Training data shape :',train.shape)


train.describe()


train.info()


plt.figure(figsize=(10,10))
sns.heatmap(train.isnull(),yticklabels=False,cbar=True)


train.nunique()


train.groupby('Pclass')['Survived'].value_counts()


train.groupby('Sex')['Survived'].value_counts()


train.pivot_table(index='Sex',columns='Survived',values='Fare',aggfunc='mean')


train.isnull().sum()


train.drop('Cabin',axis=1,inplace=True)


male_age_fill=train[train['Sex']=='male']['Age'].mean().astype(int)
female_age_fill=train[train['Sex']=='female']['Age'].mean().astype(int)
male_age_fill,female_age_fill


train.loc[(train.Age.isnull()) & (train.Sex=='male'),'Age']=male_age_fill # fill with mean of males to all null males

train.loc[(train.Age.isnull()) & (train.Sex=='female'),'Age']=female_age_fill# fill with mean of females to all null females


train.isna().sum()


train['age_category']=pd.cut(train['Age'],4,labels=[1,2,3,4]) 



survived_by_title = train.groupby('Pclass')['Survived'].sum()
total_by_title = train['Pclass'].value_counts()
percent_survived = (survived_by_title / total_by_title) * 100

labels = percent_survived.index
sizes = percent_survived.values

colors = plt.cm.RdBu(np.linspace(0, 1, len(labels)))

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=140, textprops={'color': 'black'})

plt.title('Distribution of Title over Survived', fontsize=14, color='black')
plt.gca().set_facecolor('#000000')  
plt.tight_layout()
plt.show()


train['family_size']=train['SibSp']+train['Parch']+1 


train['fare_category']=pd.cut(train['Fare'],4,labels=[1,2,3,4]) 


train.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','Fare'],axis=1,inplace=True)


#for training data

to_categorical=['Sex','Embarked']

from sklearn.preprocessing import LabelEncoder

L_encoder=LabelEncoder()

for label in to_categorical:
    train[label]=L_encoder.fit_transform(train[label])


train.head()


X_train = train.drop('Survived',axis=1)
y_train = train['Survived']


X_test=pd.read_csv('/kaggle/input/dsaa-6100-titanic-survival-using-decision-trees/test.csv')
y_test=pd.read_csv('/kaggle/input/dsaa-6100-titanic-survival-using-decision-trees/gender_submission.csv')


X_test.head()


print('Testing data shape :',X_test.shape)


X_test.describe()


X_test.info()


plt.figure(figsize=(10,10))
sns.heatmap(X_test.isnull(),yticklabels=False,cbar=True)


X_test.nunique()


X_test.isnull().sum()


X_test.drop('Cabin',axis=1,inplace=True)


male_age_fill=X_test[X_test['Sex']=='male']['Age'].mean().astype(int)
female_age_fill=X_test[X_test['Sex']=='female']['Age'].mean().astype(int)
male_age_fill,female_age_fill


X_test.loc[(X_test.Age.isnull()) & (X_test.Sex=='male'),'Age']=male_age_fill # fill with mean of males to all null males

X_test.loc[(X_test.Age.isnull()) & (X_test.Sex=='female'),'Age']=female_age_fill# fill with mean of females to all null females


"""
X_test.dropna(subset=['Fare'], inplace=True)

dropped_index = X_test.index

y_test = y_test.loc[dropped_index]"""


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
X_test[['Fare']] = imputer.fit_transform(X_test[['Fare']])



X_test.isna().sum()


X_test['age_category']=pd.cut(X_test['Age'],4,labels=[1,2,3,4]) 


X_test['family_size']=X_test['SibSp']+X_test['Parch']+1 


X_test['fare_category']=pd.cut(X_test['Fare'],4,labels=[1,2,3,4]) 


X_test.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','Fare'],axis=1,inplace=True)


to_categorical=['Sex','Embarked']

from sklearn.preprocessing import LabelEncoder

L_encoder=LabelEncoder()

for label in to_categorical:
    X_test[label]=L_encoder.fit_transform(X_test[label])


X_test.head()


X_train = train.drop(['Survived'], axis=1)

y_trrain = train['Survived']


common_cols = X_train.columns.intersection(X_test.columns)
only_in_df1 = X_train.columns.difference(X_test.columns)
only_in_df2 = X_test.columns.difference(X_train.columns)


print("âœ… Common columns:", list(common_cols))
print("ğŸŸ¥ Columns only in df1:", list(only_in_df1))
print("ğŸŸ¦ Columns only in df2:", list(only_in_df2))


y_test.head()


y_test.drop('PassengerId',axis=1,inplace=True)


print(y_train.shape)
print(y_test.shape)


def convert_to_numeric(X):
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype('category').cat.codes
    return X


def remove_outliers_zscore(X):
    X = convert_to_numeric(X)  
    z_scores = zscore(X, axis=0)
    mask = np.abs(z_scores) > 3  
    return X[~mask.any(axis=1)], mask 



def remove_outliers_iqr(X):
    X = convert_to_numeric(X)  
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    mask = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))  
    return X[~mask.any(axis=1)], mask 



def remove_outliers_isolation_forest(X):
    X = convert_to_numeric(X)  
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    preds = iso_forest.fit_predict(X)  
    mask = preds == -1  
    return X[~mask], mask  



def remove_outliers_lof(X):
    X = convert_to_numeric(X)  
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    mask = lof.fit_predict(X) == -1 
    return X[~mask], mask



outlier_methods = {
    'Z-Score': remove_outliers_zscore,
    'IQR': remove_outliers_iqr,
    'Isolation Forest': remove_outliers_isolation_forest,
    'LOF': remove_outliers_lof
}


results = {}


def train_evaluate(X_train, X_test, y_train, y_test, model_name):
    print(f"Training with {model_name}...")
    
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    class_report = classification_report(y_test, y_pred)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }
    
    print(f"Results for {model_name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    


for method_name, method_func in outlier_methods.items():
    print(f"\nğŸ› ï¸� Applying {method_name} outlier detection...")

    X_train_filtered, mask = method_func(X_train)

    if len(mask.shape) == 1:
        y_train_filtered = y_train[~mask]
    else:
        y_train_filtered = y_train[~mask.any(axis=1)]

    train_evaluate(X_train_filtered, X_test, y_train_filtered, y_test, method_name)



def train_evaluate(X_train, X_test, y_train, y_test, model_name):
    print(f"Training with {model_name}...")
    model = BaggingClassifier(GridSearchCV(DecisionTreeClassifier(random_state=42), 
                                         [{'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}], 
                                         cv=3, verbose=1), 
                             n_estimators=1000, max_samples=100, bootstrap=True, n_jobs=-1)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    class_report = classification_report(y_test, y_pred)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }
    
    print(f"Results for {model_name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
 


for method_name, method_func in outlier_methods.items():
    print(f"\nğŸ› ï¸� Applying {method_name} outlier detection...")

    X_train_filtered, mask = method_func(X_train)

    if len(mask.shape) == 1:
        y_train_filtered = y_train[~mask]
    else:
        y_train_filtered = y_train[~mask.any(axis=1)]

    train_evaluate(X_train_filtered, X_test, y_train_filtered, y_test, method_name)





