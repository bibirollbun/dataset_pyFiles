# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
import sklearn.metrics as metrics

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')


train.head()


train.columns


train.dtypes


train.shape


# check the distribution of target col.

train['Personality'].value_counts(normalize=True)


# plot the personality Distribution.

sns.set_theme(style='whitegrid', palette='viridis')
ax = sns.countplot(x=train['Personality'])
ax.bar_label(ax.containers[0], padding=-10, label_type='center', color='white')
plt.title('Personality Distribution')
plt.show()


plt.figure(figsize=(12,6))
sns.pairplot(train)


# first we will convert categorical columns into numerical.
train_corr = train.copy()

# yes or no columns
cat_cols = ['Stage_fear','Drained_after_socializing']
mapping = {'Yes':1, 'No':0}

for col in cat_cols:
    train_corr[col] = train_corr[col].map(mapping)

# personality column
person_type = {'Extrovert':1, 'Introvert':0}
train_corr['Personality'] = train_corr['Personality'].map(person_type)


train_corr.head()


corr_matrix = train_corr.drop('id', axis=1).corr()

plt.figure(figsize=(10,5))
sns.heatmap(corr_matrix, annot=True)


original.head()


original.shape


original.isna().sum()


original.duplicated().sum()


# Dropping duplicated 
original = original.drop_duplicates()


train.isna().sum()


# Percetage of missing values in the dataset
ratio = train.isna().sum() / len(train) *100
ratio


# Dropping missing values from the most important features

important_features = ['Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing']
train = train.dropna(subset=important_features)


# Filling missing values in less important features with the mode value

less_important = ['Time_spent_Alone', 'Friends_circle_size', 'Post_frequency']

for col in less_important:
    mode_value = train[col].mode()[0]
    train[col] = train[col].fillna(mode_value)



train.duplicated().sum()


train.shape


test.duplicated().sum()


test.isna().sum()


test_features = test.drop('id', axis=1)

for col in test_features:
    mode_value = test[col].mode()[0]
    test[col] = test[col].fillna(mode_value)

test.isna().sum()


test.duplicated().sum()


# first we will change the name of personality col in original data
original = original.rename(columns={'Personality':'Personality_type'})

# Next we will merge datasets together
merge_cols = ['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing'
              ,'Friends_circle_size','Post_frequency']

train = train.merge(original, how='left', on=merge_cols)
test = test.merge(original, how='left', on=merge_cols)


train.head()


train.shape


# Filling the NaN values in the personality_type col with 'Unkown'
train['Personality_type'] = train['Personality_type'].fillna('Unknown')
test['Personality_type'] = test['Personality_type'].fillna('Unknown')



train.head()


# Drop duplicates
train = train.drop_duplicates()
test = test.drop_duplicates()


# Converting categorical cols to numeric
# first yes/no cols
yes_no_cols = ['Stage_fear', 'Drained_after_socializing']
yes_no_map = {'Yes':1, 'No':0}

for col in yes_no_cols:
    train[col] = train[col].map(yes_no_map)
    test[col] = test[col].map(yes_no_map)

# personality feature col
feature_map = {'Extrovert':2, 'Introvert':1, 'Unknown':0}

train['Personality_type'] = train['Personality_type'].map(feature_map)
test['Personality_type'] = test['Personality_type'].map(feature_map)

# personality target col
target_map = {'Extrovert':1, 'Introvert':0}

train['Personality'] = train['Personality'].map(target_map)


# Splitting the data

X = train.drop(['id', 'Personality'], axis=1)
Y = train['Personality']
valid_data = test.drop('id', axis=1)


x_train, x_test, y_train, y_test = train_test_split(X,Y,
                                                    test_size=0.25,
                                                    stratify=Y,
                                                    random_state=0
                                                   )


decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(x_train,y_train)

dt_preds = decision_tree.predict(x_test)


# create table of results

Accuracy = metrics.accuracy_score(y_test,dt_preds)
Precision = metrics.precision_score(y_test,dt_preds)
Recall = metrics.recall_score(y_test,dt_preds)
F1 = metrics.f1_score(y_test,dt_preds)
Roc_Auc = metrics.roc_auc_score(y_test,dt_preds)

decision_tree_results = pd.DataFrame({'Model' : ['Decision Tree'],
                                      'Accuracy' : [Accuracy],
                                      'Precision': [Precision],
                                      'Recall' : [Recall],
                                      'F1' : [F1],
                                      'Roc_Auc' : [Roc_Auc]
                                       })
decision_tree_results


cm = metrics.confusion_matrix(y_test, dt_preds, labels=decision_tree.classes_)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=decision_tree.classes_)
disp.plot(values_format='.2f')
plt.grid(False)


param = {'max_depth': [10,20,30,None],
         'min_samples_split': [10,20,50,100],
         'min_weight_fraction_leaf':[0.1,0.2,0.25,0.3]
        }

scoring = ['accuracy', 'precision', 'recall', 'roc_auc','f1']


tuned_decision_tree = DecisionTreeClassifier(random_state=0)

clf = GridSearchCV(tuned_decision_tree,
                   param,
                   scoring = scoring,
                   cv = 5,
                   refit='f1')

clf.fit(x_train,y_train)


clf.best_estimator_


def make_results(model_name,model_object):

    cv_results = pd.DataFrame(model_object.cv_results_)

    # isolate the row with the max f1 score
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(),:]
    
    f1 = best_estimator_results.mean_test_f1
    accuracy = best_estimator_results.mean_test_accuracy
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    roc_auc = best_estimator_results.mean_test_roc_auc

    # create table of results
    table = pd.DataFrame({'Model': [model_name],
                          'Accuracy' : [accuracy],
                          'Precision': [precision],
                          'Recall' : [recall],
                          'F1' : [f1],
                          'Roc_Auc' : [roc_auc]
                         })
    return table

results = make_results('Tuned Decision Tree',clf)

    
final_results = pd.concat([results,decision_tree_results], axis=0)
final_results


opt_dt = DecisionTreeClassifier(max_depth=10, min_samples_split=10,
                                min_weight_fraction_leaf=0.1, random_state=0
                               )
opt_dt.fit(x_train,y_train)


valid_data.head()


final_predictions = opt_dt.predict(valid_data)


# Reverse encoded values to its original meanings
names = {1:'Extrovert', 0:'Introvert'}

final_predictions = pd.Series(final_predictions).map(names)


final_predictions.head()


submission = pd.DataFrame({'id': test['id'], 'Personality': final_predictions})
submission.head()


dup = submission.duplicated()
submission[dup]


submission = submission.drop_duplicates()


submission.to_csv('/kaggle/working/submission.csv', index=False)

