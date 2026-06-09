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
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import classification_report as cr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # import the train_test_split function
from sklearn.metrics import accuracy_score, precision_score, recall_score ,f1_score, confusion_matrix, classification_report
#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier  # or DecisionTreeRegressor
#from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore')
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.tree import plot_tree


#read the local csv files
file_path_train = '/kaggle/input/playground-series-s4e2/train.csv'
df_train = pd.read_csv(file_path_train)

file_path_test = '/kaggle/input/playground-series-s4e2/test.csv'
df_test = pd.read_csv(file_path_test)


#Making all train data numerical
df_train['NObeyesdad'] = df_train['NObeyesdad'].map({'Overweight_Level_II':3, 'Normal_Weight':1, 'Insufficient_Weight':0,'Obesity_Type_III':6, 'Obesity_Type_II':5, 'Overweight_Level_I':2,'Obesity_Type_I':4})
df_train['family_history_with_overweight'] = df_train['family_history_with_overweight'].map({'no':0,'yes':1})
df_train['Gender'] = df_train['Gender'].map({'Male':0,'Female':1})
df_train['FAVC'] = df_train['FAVC'].map({'no':0,'yes':1})
df_train['CAEC'] = df_train['CAEC'].map({'no':0,'Sometimes':1,'Frequently':2,'Always':3})
df_train['SMOKE'] = df_train['SMOKE'].map({'no':0,'yes':1})
df_train['SCC'] = df_train['SCC'].map({'no':0,'yes':1})
df_train['CALC'] = df_train['CALC'].map({'no':0,'Sometimes':1,'Frequently':2})
df_train['MTRANS'] = df_train['MTRANS'].map({'Public_Transportation':0, 'Automobile':1, 'Walking':2, 'Motorbike':3,'Bike':4})


df_train


#Let's do the same for test df
df_test['family_history_with_overweight'] = df_test['family_history_with_overweight'].map({'no':0,'yes':1})
df_test['FAVC'] = df_test['FAVC'].map({'no':0,'yes':1})
df_test['Gender'] = df_test['Gender'].map({'Male':0,'Female':1})
df_test['SCC'] = df_test['SCC'].map({'no':0,'yes':1})
df_test['CALC'] = df_test['CALC'].map({'no':0,'Sometimes':1,'Frequently':2})
df_test['SMOKE'] = df_test['SMOKE'].map({'no':0,'yes':1})
df_test['CAEC'] = df_test['CAEC'].map({'no':0,'Sometimes':1,'Frequently':2,'Always':3})
df_test['MTRANS'] = df_test['MTRANS'].map({'Public_Transportation':0, 'Automobile':1, 'Walking':2, 'Motorbike':3,'Bike':4})


df_test


df_train=df_train.reindex()
df_train.describe().T


#Do not need id for training df
df_train=df_train.drop(['id'], axis = 1)


# Correlation heatmap
numeric_df = df_train.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


Predictors = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
X=df_train[Predictors]
y = df_train['NObeyesdad']
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


#Data Splitting for training
from sklearn.model_selection import train_test_split # import the train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize a global DataFrame to store models results
results_df = pd.DataFrame(columns=['Model', 'Train Score', 'Test Score', 'Precision', 'Recall', 'F1-Score'])

def train_model(model):
    global results_df  # Access the global DataFrame

    # Train the model
    model.fit(X_train, y_train)

    # Compute train and test scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f'Train Score => {train_score}')
    print(f'Test Score => {test_score}')

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate classification report dictionary
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extract weighted average precision, recall, and F1-score
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    # Append results to the global DataFrame using pd.concat
    # Create a new DataFrame for the current model's results
    new_row_df = pd.DataFrame([{
        'Model': type(model).__name__,
        'Train Score': train_score,
        'Test Score': test_score,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    }])

    # Concatenate the new row DataFrame with the existing results_df
    results_df = pd.concat([results_df, new_row_df], ignore_index=True)


model=tree()
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


mytree=model.fit(X,y) # algorithm learns from data
print( mytree.score(X,y)) #Score - the percentage of correctly classified data points. 


mypred=mytree.predict(X)
print(cr(y,mypred))


train_model(mytree)


from sklearn.ensemble import BaggingClassifier as bag


myBaggingClassifier=bag(n_estimators=30)
# evaluate the model and collect the scores
n_scores = cross_val_score(myBaggingClassifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


bag=myBaggingClassifier.fit(X,y)
print( bag.score(X,y))


mypred=bag.predict(X)
print(cr(y,mypred))


train_model(bag)


from sklearn.ensemble import RandomForestClassifier as rfc


myRandomForestClassifier=rfc(n_estimators=100)
# evaluate the model and collect the scores
n_scores = cross_val_score(myRandomForestClassifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


myrf=myRandomForestClassifier.fit(X,y)
print( myrf.score(X,y))


mypred=myrf.predict(X)
print(cr(y,mypred))


train_model(myrf)


feat_importances = pd.Series(myRandomForestClassifier.feature_importances_, index=X.columns)
feat_importances.nlargest(16).plot(kind='barh')


from sklearn.ensemble import GradientBoostingClassifier as gbc


myGradientBoostingClassifier=gbc(n_estimators=100, max_depth=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(myGradientBoostingClassifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


mygbc=myGradientBoostingClassifier.fit(X,y)
print( mygbc.score(X,y))


mypred=mygbc.predict(X)
print(cr(y,mypred))


train_model(mygbc)


feat_importances = pd.Series(myRandomForestClassifier.feature_importances_, index=X.columns)
feat_importances.nlargest(16).plot(kind='barh')


results_df


df_test[Predictors].describe()


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')  # Or other strategies like 'median' mean, 'most_frequent'
df_test_imputed = df_test[Predictors].copy()  # Create a copy to avoid modifying the original DataFrame
df_test_imputed[Predictors] = imputer.fit_transform(df_test_imputed[Predictors])

obesity_for_submission = myrf.predict(df_test_imputed)
#obesity_for_submission = lda_model.predict(df_test[Predictors])


kaggle_submisson = pd.DataFrame({"id":df_test["id"],"NObeyesdad":obesity_for_submission})


kaggle_submisson["NObeyesdad"] = kaggle_submisson["NObeyesdad"].map({3:'Overweight_Level_II', 1:'Normal_Weight', 0:'Insufficient_Weight', 6:'Obesity_Type_III', 5:'Obesity_Type_II', 2:'Overweight_Level_I', 4:'Obesity_Type_I'})



kaggle_submisson


kaggle_submisson.to_csv('/kaggle/working/obesity_kaggle_submisson_alina.csv', index = False)

