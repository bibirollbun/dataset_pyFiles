import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
pd.set_option('future.no_silent_downcasting', True)
sns.set()

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from mlxtend.evaluate import bootstrap_point632_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


os.listdir('/kaggle/input')


for root, folders, filenames in os.walk('/kaggle/input'):
   print(root, folders)


# Read the csvs as pandas df. Dataset URL = https://www.kaggle.com/competitions/star-type-classification/data

test_df = pd.read_csv('/kaggle/input/star-type-classification/test_star.csv')
train_df = pd.read_csv('/kaggle/input/star-type-classification/train_star.csv')


# Convert the column names to lowercase

train_df = train_df.rename(columns = lambda x: x.lower().strip())
test_df = test_df.rename(columns = lambda x: x.lower().strip())


print(f"Dimensions of the training set: {train_df.shape}")
print(f"Dimensions of the test set: {test_df.shape}")


train_df.head()


test_df.head()


# Check the 95th percentiles of the e_plx feature

print(f"Train e_plx 95th percentile: {train_df['e_plx'].quantile(0.95)}")
print(f"Test e_plx 95th percentile: {test_df['e_plx'].quantile(0.95)}")


# Drop the rows with e_plx values > 95th percentile

train_df = train_df.drop(train_df[train_df['e_plx']> 2].index)
test_df = test_df.drop(test_df[test_df['e_plx']> 1.7].index)


train_df.loc[:, :"amag"].describe()


_, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
sns.boxplot(x = "e_plx", data = train_df, ax = axes[0])
axes[0].set_title("Train Set")
axes[1].set_title("Test Set")
sns.boxplot(x = "e_plx", data = test_df, ax = axes[1])


# Look at the correlation between every feature in our data set, as well as their distributions

sns.pairplot(train_df.loc[:, :"amag"].drop("sptype", axis = 1 ), kind = 'scatter')


sns.heatmap(train_df.drop(["targetclass", "sptype"], axis = 1).corr(), annot = True, cmap = 'viridis') # corr matrix


train_df['targetclass'].value_counts(normalize = True) # Look at the class imbalance of our labels


le = LabelEncoder() # Use the label encoder to convert from Categorical to numerical

train_df['sptype'] = le.fit_transform(train_df['sptype'])
test_df['sptype'] = le.fit_transform(test_df['sptype'])
train_df['targetclass'] = le.fit_transform(train_df['targetclass'])
train_df.head()


X = train_df.loc[:, :"amag"].values # Store the features in X
y = train_df.loc[:, 'targetclass'].values # Store the labels in y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 12345) # 80/20 Train-Test split of our data


# Let's start by using GridSearchCV for model selection on a lazy learner. We will use the f1-score as a scoring metric since we're dealing with
# a binary classification problem with class imbalance

knn_clf = KNeighborsClassifier(metric = 'minkowski', p = 2, 
                               algorithm = 'ball_tree',
                               leaf_size = 50)
KNN_Pipe = Pipeline([('std', StandardScaler()),('KNN', knn_clf)])

param_grid = [{"KNN__n_neighbors": list(range(1, 10)),
               "KNN__weights": ['uniform', 'distance']}]

cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 123)

gs_knn = GridSearchCV(estimator = KNN_Pipe,
                  param_grid = param_grid,
                  refit = True,
                  cv = cv,
                  scoring = 'f1',
                  n_jobs = -1)
gs_knn.fit(X_train, y_train)


# Let's check the model's preformance:
acc_knn = gs_knn.score(X_test, y_test)
print(f"Accuracy = {acc_knn:.2f}")

f1_val = f1_score(y_true = y_test, y_pred = gs_knn.predict(X_test))
print(f"F1 score = {f1_val:.2f}")

pre_val = precision_score(y_true = y_test, y_pred = gs_knn.predict(X_test))
print(f"Precision score = {pre_val:.2f}")

rec_val = recall_score(y_true = y_test, y_pred = gs_knn.predict(X_test))
print(f"Recall score = {rec_val:.2f}")


# Let's fit a more sophisticated classifier

parameter_grid = [{'min_samples_split':[2, 3],
                   'max_features':["sqrt", "log2"],
                   'n_estimators':[100, 200, 300]}]

# Again performing model selection via GridSearchCV

gs_forest = GridSearchCV(estimator = RandomForestClassifier(n_jobs = -1, random_state = 1),
                  param_grid = parameter_grid,
                  refit = True,
                  cv = cv,
                  scoring = 'f1',
                  n_jobs = -1)

gs_forest.fit(X_train, y_train)


acc_forest = gs_forest.score(X_test, y_test)
print(f"Accuracy = {acc_forest:.2f}")

f1_val_forest = f1_score(y_true = y_test, y_pred = gs_forest.predict(X_test))
print(f"F1 score = {f1_val_forest:.2f}")

pre_val_forest = precision_score(y_true = y_test, y_pred = gs_forest.predict(X_test))
print(f"Precision score = {pre_val_forest:.2f}")

rec_val_forest = recall_score(y_true = y_test, y_pred = gs_forest.predict(X_test))
print(f"Recall score = {rec_val_forest:.2f}")


# Plot ROC AUC curves for the two classifiers

# Store the prediction probabilities
y_pred_rf_knn = gs_knn.predict_proba(X_test)[:, 1]
y_pred_rf_forest = gs_forest.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 6))

fpr_forest, tpr_forest, _ = roc_curve(y_test, y_pred_rf_forest)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_rf_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_forest = auc(fpr_forest, tpr_forest)

plt.plot(fpr_knn, tpr_knn, label = f' KNN (area = {roc_auc_knn:.2f})', c = 'green')
plt.plot(fpr_forest, tpr_forest, label = f' Random Forest (area = {roc_auc_forest:.2f})', c = 'black')
plt.plot([0, 1], [0, 1], 'r--', label = f'random guessing (area = {0.5})')
plt.plot([0, 0, 1], [0, 1, 1], 'b-.', label = f'perfect performance (area = {1})',)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve for the best performing Random Forest model')
plt.legend()
plt.show()


# 95% Confidence interval via normal approximation using f1 score as a metric
# I use normal approximation mainly because I don't want to use the 632.+ Bootstrap method on a Random Forest Model. (Mainly because I would have to
# fit a lot of trees, which would take a really long time

z_value = 1.96

ci = z_value * np.sqrt((f1_val_forest * (1 - f1_val_forest)) / y_test.shape[0])

print(f1_val_forest - ci, f1_val_forest + ci)


gs_forest.best_params_


# Refit the Random Forest using the entire X data set

rf_classifier = RandomForestClassifier(max_features = 'sqrt', min_samples_split = 3, n_estimators = 200)
rf_classifier.fit(X, y)


final_labels = rf_classifier.predict(test_df.values)


pd.Series(final_labels).to_csv('out.csv', index = False)

