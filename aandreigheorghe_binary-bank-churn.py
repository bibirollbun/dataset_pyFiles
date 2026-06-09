import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m

import os
import seaborn as sns
pd.set_option('future.no_silent_downcasting', True)
sns.set()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from mlxtend.evaluate import bootstrap_point632_score
from sklearn.pipeline import Pipeline


for root, folders, filenames in os.walk('/kaggle/input'):
   print(root, folders)


# Load the data sets

train = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/train.csv')
test = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/test.csv')
submission = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/sample_submission.csv')

# convert column names to lowercase
train = train.rename(columns = lambda x: x.lower())
test = test.rename(columns = lambda x: x.lower())


# Glance at the training set 

train.head()


# Glance at the test set 

test.head()


train.info()


test.info()


# Drop unnecessary columns from the training/test sets
test_ids = test.loc[:, 'id']
train = train.drop(['id', 'customerid', 'surname'], axis = 1)
test = test.drop(['id', 'customerid', 'surname'], axis = 1)


# Check if all of the credit scores were wrongly input as float

sum(train['creditscore'].value_counts().index[:]), sum(test['creditscore'].value_counts().index[:])


# Convert training set features which were wrongly registered as float64 to int64

train['age'] = train['age'].astype(np.int64)
train['tenure'] = train['tenure'].astype(np.int64)
train['hascrcard'] = train['hascrcard'].astype(np.int64)
train['numofproducts'] = train['numofproducts'].astype(np.int64)
train['isactivemember'] = train['isactivemember'].astype(np.int64)
train['exited'] = train['exited'].astype(np.int64)
train['creditscore'] = train['creditscore'].astype(np.int64)


# Convert test features which were wrongly registered as float64 to int64

test['age'] = test['age'].astype(np.int64)
test['tenure'] = test['tenure'].astype(np.int64)
test['hascrcard'] = test['hascrcard'].astype(np.int64)
test['numofproducts'] = test['numofproducts'].astype(np.int64)
test['isactivemember'] = test['isactivemember'].astype(np.int64)
test['creditscore'] = test['creditscore'].astype(np.int64)


# Take a look at how features correlate to one another through scatter plots in our training set
# Also check feature distribution in our training set

sns.pairplot(data = train.drop(['geography', 'gender', 'isactivemember', 'hascrcard', 'exited'], axis = 1)) 


# Take a look at how features correlate to one another through scatter plots in our test set
# Also check feature distribution in our test set

sns.pairplot(data = test.drop(['geography', 'gender', 'isactivemember', 'hascrcard'], axis = 1))


# Look at a few countplots to try glean some relationship between churn and a different features

_, axes = plt.subplots(nrows = 2, ncols = 2, sharey=True, figsize=(36, 24))

sns.countplot(x = "hascrcard", hue = "exited", data = train, ax = axes[0][0]);
sns.countplot(x = "tenure", hue = "exited", data = train, ax = axes[0][1]);
sns.countplot(x = "isactivemember", hue = "exited", data = train, ax = axes [1][0]);

# kde histplot for the 'balance' features to get a better look at its distribution
sns.histplot(kde = True, data = train['balance'], ax = axes[1][1])


# Look at a few countplots to try glean some relationship between churn and a different features


_, axes = plt.subplots(nrows = 1, ncols = 2, sharey=True, figsize=(30, 12))

# bin features with many values
sns.countplot(x = pd.cut(train['creditscore'], 1 + m.floor(m.log(len(train['creditscore']), 2)/2)) , hue = train["exited"], ax=axes[0])
sns.countplot(x = pd.cut(train['balance'], 1 + m.floor(m.log(len(train['creditscore']), 2)/2)), hue = train["exited"], ax=axes[1]);


sns.countplot(x = "geography", hue = "exited", data = train);


train.describe(include = np.number) # check the statistics for numerical features of training set


test.describe(include = np.number) # check the statistics for numerical features of test set


train['exited'].value_counts(normalize = True) # Check for class imbalance


train['geography'].value_counts() # Check for unique values for 'geography' feature


# Map categorical features to nominal

train['geography'] = train['geography'].map({'Spain': 0, 'France': 1, 'Germany': 2})
train['gender'] = train['gender'].map({'Female': 0, 'Male': 1})

test['geography'] = test['geography'].map({'Spain': 0, 'France': 1, 'Germany': 2})
test['gender'] = test['gender'].map({'Female': 0, 'Male': 1})


train.head() # Check the cleaned up training set


test.head() # Check the cleaned up test set


# Split the training data frame into features (X) and labels (y)

X, y = train.loc[:, :'estimatedsalary'].values, train.loc[:, 'exited'].values
print(f'Features: {X}')
print(f'Labels: {y}')


print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 123, stratify = y) # 80/20 train/test split of our data


print(f"Training set dim: {X_train.shape, y_train.size}")
print(f"Test set dim: {X_test.shape, y_test.size}")


# We'll use Nested - CV for Algorithm Selection

# Create some classifier objects
dec_tree = DecisionTreeClassifier(random_state = 123)
rand_forest = RandomForestClassifier(random_state = 123)
log_reg = LogisticRegression(solver = 'newton-cg', random_state = 123)
KNN = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size = 50)

# Create a pipeline object for the KNN algorithm
knn_pipe = Pipeline([('std', StandardScaler()), ('KNN', KNN)])

# Create a list of classifiers
classifiers = [dec_tree, rand_forest, log_reg, knn_pipe]

# parameter dictonary for GridSearchCV's param_grid
params = [{'criterion': ['entropy', 'gini'],'splitter': ['best', 'random']},
          {'criterion': ['entropy', 'gini'], 'n_estimators' : [100, 200, 300]},
          {'C': [1.0, 1.25, 1.5, 1.75, 2.0], 'class_weight': ['balanced', None]},
          {'KNN__n_neighbors': list(range(1, 10)), 'KNN__p': [1, 2]}]

names = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'KNN']

# Dictionary to store our models
gridcvs = {}

# Build a StratifiedKFold object that will be used for the inner loop. This will split our data set into two folds, one used for training, the other for
# validation
inner_cv = StratifiedKFold(n_splits = 2, shuffle = True, random_state = 1)

# Fill the gridcvs dictionary with GridSearchCV objects, one for each of our classifiers
for classifier, param, name in zip(classifiers, params, names):
    gcv = GridSearchCV(estimator = classifier,
                      param_grid = param,
                      cv = inner_cv,
                      n_jobs = -1,
                      refit = True)
    gridcvs[name] = gcv


# Create a StratifiedKFold object for our outer loop. This time we split our data set into 5 folds, 4 used for training, 1 for validation

outer_cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 1)

# Perform the Nested CV. We'll use AUC as a metric, since that's the performance metric that interests us.
for name, gs_est in gridcvs.items():
    nested_score = cross_val_score(gs_est, 
                                   X = X_train, 
                                   y = y_train, 
                                   cv = outer_cv,
                                   n_jobs = -1,
                                   scoring = 'roc_auc')
    print(f'{name} | outer AUC {nested_score.mean():.2f} +/- {nested_score.std() * 100:.2f}')  


cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 123)

gcv_model_select = GridSearchCV(estimator = RandomForestClassifier(),
                                param_grid = params[1],
                                scoring = 'roc_auc',
                                n_jobs = -1,
                                cv = cv,
                                verbose = 1,
                                refit = True)

gcv_model_select.fit(X_train, y_train)


best_model = gcv_model_select.best_estimator_
gcv_model_select.best_params_


acc = best_model.score(X_test, y_test)
print(f"Accuracy = {acc*100:.2f}%")

f1_val = f1_score(y_true = y_test, y_pred = best_model.predict(X_test))
print(f"F1 score = {f1_val:.2f}")

pre_val = precision_score(y_true = y_test, y_pred = best_model.predict(X_test))
print(f"Precision score = {pre_val:.2f}")

rec_val = recall_score(y_true = y_test, y_pred = best_model.predict(X_test))
print(f"Recall score = {rec_val:.2f}")


y_pred_rf = best_model.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 6))

fpr, tpr, _ = roc_curve(y_test, y_pred_rf)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label = f' Random Forest (area = {roc_auc:.2f})', c = 'green')
plt.plot([0, 1], [0, 1], 'r--', label = f'random guessing (area = {0.5})')

plt.plot([0, 0, 1], [0, 1, 1], 'b--', label = f'perfect performance (area = {1})',)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve for the best performing Random Forest model')
plt.legend()
plt.show()


# 95% Confidence interval via normal approximation using roc_auc as a metric
# I use normal approximation mainly because I don't want to use the 632.+ Bootstrap method on a Random Forest Model. (Mainly because I would have to
# fit a lot of trees, which would take a really long time

z_value = 1.96

ci = z_value * np.sqrt((roc_auc * (1 - roc_auc)) / y_test.shape[0])

print(roc_auc - ci, roc_auc + ci)


# Refit the model on the entire data set using the best parameters

rcf = RandomForestClassifier(criterion = 'entropy', 
                             n_estimators= 300,
                             random_state = 123,
                             n_jobs = -1)
rcf.fit(X, y)


final_labels = rcf.predict(test.values)
final_output = pd.DataFrame({'col1' : test_ids,  'col2':pd.Series(final_labels)})
final_output.rename(columns = {'col1':'id', 'col2':'exited'}).to_csv('out.csv', index = False)

