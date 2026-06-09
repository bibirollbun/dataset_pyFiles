# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns
import pandas as pd
import numpy as np


train = pd.read_csv('/kaggle/input/leaf-classification/train.csv.zip')
test = pd.read_csv('/kaggle/input/leaf-classification/test.csv.zip')
test.head()


#train = train.drop('id', axis=1)
train.describe()


train.info()


columns_with_missing  = train.columns[train.isnull().any()].tolist()
print(columns_with_missing)



train.duplicated().sum()


train_without_species = train.drop(["species", "id"], axis=1)


skewed_columns = train_without_species.columns[train_without_species.skew().sort_values()>0.5]

skewed_columns
print(len(skewed_columns)) # 182


sns.histplot(train_without_species['texture61'])


skewed_train_data = train[skewed_columns]
skewed_train_data.head(3)


def observing_outliers(data):
    
    percentile = np.percentile(data, [25, 75])
    
    q1 = percentile[0]
    q3 = percentile[1]
    
    iqr = q3 - q1
    
    lower_limit = q1 - (1.5*iqr)
    
    upper_limit = q3 + (1.5*iqr)

    outliers = [x for x in data if x < lower_limit or x>upper_limit]
    
    #print(len(outliers))

    return len(outliers)


outliers_data = []
for col in skewed_columns:
    outliers_data.append({
        "column": col,
        "no_of_outliers": observing_outliers(skewed_train_data[col])
    })


df = pd.DataFrame([obj for obj in outliers_data])
df


from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)),
    ('scaler', StandardScaler())
])


train_transformed_array = pipeline.fit_transform(train_without_species)
#test_transformed_array = pipeline.fit_transform(test_without_id)


# cnverting array to dataframe
train_transformed_df = pd.DataFrame(train_transformed_array, columns = train_without_species.columns, index = train_without_species.index)


skewed_columns_after = train_transformed_df.columns[train_transformed_df.skew().sort_values()>0.5]

skewed_columns_after

print(len(skewed_columns_after)) # 62


outliers_data_after = []
for col in skewed_columns_after:
    outliers_data_after.append({
        "column": col,
        "no_of_outliers": observing_outliers(train_transformed_df[col])
    })


print(outliers_data_after)



df = pd.DataFrame([obj for obj in outliers_data_after])
df


X = train.drop(["species", "id"], axis = 1)
y = train['species']


X_transformed = train_transformed_df



from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

y = le.fit_transform(train['species'])


from sklearn.model_selection import StratifiedShuffleSplit


sss = StratifiedShuffleSplit(10, test_size=0.2, random_state=42)


print(X.shape) # (990, 192)
print(y.shape) # (990,)
print(X_transformed.shape) # (990, 192)



for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y[train_index], y[test_index]


X.iloc[207]

X.columns


for train_index, test_index in sss.split(X_transformed, y):
    X_train_trans, X_test_trans = X_transformed.values[train_index], X_transformed.values[test_index]
    y_train_trans, y_test_trans = y[train_index], y[test_index]


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, log_loss



classification = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression(),
    SGDClassifier(loss='log_loss'),
    XGBClassifier()    
]

log_cols=["Classifier", "Accuracy", "Log Loss", "precision", "recall", "f1_score"]
log = pd.DataFrame(columns=log_cols)


for clf in classification:


    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    print('----------- Classifier name -----------')
    print(name)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='macro')
    recall = recall_score(y_test, y_pred,average='macro')
    f1score = f1_score(y_test, y_pred,average='macro')
    cr = classification_report(y_test, y_pred)


    y_pred_probab = clf.predict_proba(X_test)
    ll = log_loss(y_test, y_pred_probab)

    # print(f"acc: {acc}")
    # print(f"precision: {precision}")
    # print(f"recall: {recall}")
    # print(f"f1score: {f1score}")
    # print(f"ll: {ll}")
    #print(f"cr: {cr}")

    log = pd.concat([log, pd.DataFrame([[name, acc*100, ll, precision*100, recall*100, f1score*100 ]], columns =log_cols )])
    


# ["Classifier", "Accuracy", "Log Loss", "precision", "recall", "f1_score"]
log.sort_values(by='Accuracy', ascending=False)


log.sort_values(by='Log Loss', ascending=True)



classification = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression(),
    SGDClassifier(loss='log_loss'),
    XGBClassifier()    
]

log_cols=["Classifier", "Accuracy", "Log Loss", "precision", "recall", "f1_score"]
log_trans = pd.DataFrame(columns=log_cols)


for clf in classification:


    clf.fit(X_train_trans, y_train_trans)

    name = clf.__class__.__name__

    print('----------- Classifier name -----------')
    print(name)

    y_pred_trans = clf.predict(X_test_trans)

    acc = accuracy_score(y_test_trans, y_pred_trans)
    precision = precision_score(y_test_trans, y_pred_trans,average='macro')
    recall = recall_score(y_test_trans, y_pred_trans,average='macro')
    f1score = f1_score(y_test_trans, y_pred_trans,average='macro')
    cr = classification_report(y_test_trans, y_pred_trans)


    y_pred_probab_trans = clf.predict_proba(X_test_trans)
    ll = log_loss(y_test_trans, y_pred_probab_trans)

    # print(f"acc: {acc}")
    # print(f"precision: {precision}")
    # print(f"recall: {recall}")
    # print(f"f1score: {f1score}")
    # print(f"ll: {ll}")
    #print(f"cr: {cr}")

    log_trans = pd.concat([log_trans, pd.DataFrame([[name, acc*100, ll, precision*100, recall*100, f1score*100 ]], columns =log_cols )])
    


log_trans.sort_values(by="Accuracy", ascending=False)


log_trans.sort_values(by="Log Loss", ascending=True)


log.sort_values(by='Classifier')



log_trans.sort_values(by='Classifier')


test_ids = test['id']
test_without_id = test.drop(["id"], axis=1)
classes = test_without_id.columns


classes


from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)),
    ('scaler', StandardScaler())
])


#train_transformed_array = pipeline.fit_transform(train_without_species)
test_transformed_array = pipeline.fit_transform(test_without_id)


# Predict Test Set
final_clf = RandomForestClassifier()
final_clf.fit(X_train_trans, y_train_trans)
test_predictions = final_clf.predict_proba(test_transformed_array)


#print(test_predictions[0])
# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=le.classes_)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('submission.csv', index = None)
submission.tail()



submission.to_csv("submission.csv",  index = None)





