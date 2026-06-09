# Import packages for data manipulation
import numpy as np
import pandas as pd

# Import packages for data preprocessing
from sklearn.preprocessing import StandardScaler

# Import packages for data visualization
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.tree import plot_tree
from xgboost import plot_importance 

# Import packages for statistical modeling
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Import packages for machine learning

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Import packages for model evaluation
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)

# Miscellaneous
from ydata_profiling import ProfileReport



# Load dataset into a datafram

df_sample0 = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/sample_submission.csv')
df_test0 = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/test.csv')
df_train0 = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')

df_sample = df_sample0.copy()
df_test = df_test0.copy()
df_train = df_train0.copy()


# basic information about the data
print(df_sample.info())

# Display first few rows of the dataframe
df_sample.head(2)


# basic information about the data
print(df_test.info())

# Display first few rows of the dataframe
df_test.head(2)


# basic information about the data
print(df_train.info())

# Display first few rows of the dataframe
df_train.head(2)


# descriptive statistics about the data
df_train.describe().T


# Check for missing values
print("numbers of missing values in df_train : ",df_train.isna().sum().sum())
print("numbers of missing values in df_test : ",df_test.isna().sum().sum())


# Check for duplicates
print("numbers of duplicate in df_train : ",df_train.duplicated().sum())
print("numbers of duplicate in df_test : ",df_test.duplicated().sum())


# Create a figure
plt.figure(figsize=(25, 20))

# Compute the correlation matrix for all the numeric columns in the dataframe
correlation_matrix = df_train.select_dtypes(include=['float64', 'int64']).corr(method='pearson')

# Create a mask to hide the upper triangle of the heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create a heatmap with annotations, using the 'crest' colormap
sns.heatmap(correlation_matrix, annot=True, cmap='crest', mask=mask, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Heatmap')
# Display the plot
plt.show()


# Setting the style of seaborn plots to 'whitegrid'
sns.set_theme(style='whitegrid')

# Creating a new figure with specified size
plt.figure(figsize=(20, 15))

# Looping over each column in the DataFrame (excluding 'id', 'Time', and the last two columns)
# is used to show a progress bar
for i, column in enumerate(df_train.columns[2:-2]): 
    # Creating a subplot for each column
    plt.subplot(5, 6, i+1)
    # Plotting a histogram for each column
    sns.histplot(df_train[column], kde=True, bins=30)
    # Setting the title of each subplot to the column name
    plt.title(column)

# Adjusting the layout so that there's no overlap between subplots
plt.tight_layout()
# Displaying the figure with all subplots
plt.show()


#check the balance of the df
df_train['IsFraud'].value_counts(normalize=True) * 100


# Split into train and test sets
y = df_train["IsFraud"]
X = df_train.drop(columns=['IsFraud','id'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42, stratify=y)

# Create the scaler
scaler = StandardScaler()

# Scale the training data
X_train_scale = scaler.fit_transform(X_train)

# Scale the test data
X_test_scale = scaler.transform(X_test)

X_train = X_train_scale
X_test = X_test_scale


X
y


from collections import Counter
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Split into train and test sets
y = df_train["IsFraud"]
X = df_train.drop(columns=['IsFraud','id'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42, stratify=y)


def Definedata():
    # define dataset
    X=df_train.drop(columns=['IsFraud','id']).values
    y=df_train['IsFraud'].values
    return X, y

def scale_data(X_train_, X_test_):
    # Create the scaler
    scaler = StandardScaler()
    
    # Scale the training data
    X_train_scale = scaler.fit_transform(X_train_)
    
    # Scale the test data
    X_test_scale = scaler.transform(X_test_)
    return X_train_scale, X_test_scale


def run_smote():
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from matplotlib import pyplot
    from numpy import where
    
    X, y = Definedata()

    # summarize class distribution
    counter = Counter(y)
    print("Before SMOTE:", counter)

    # transform the dataset
    smt = SMOTE(random_state=0)
    X, y = smt.fit_resample(X, y)  # ✅ fix here

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)

    # summarize the new class distribution
    counter = Counter(y)
    print("After SMOTE:", counter)

    # scatter plot
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.show()

    return X_train, X_test, y_train, y_test



%time X_train1, X_test1, y_train1, y_test1 = run_smote()
# %time X_train2, X_test2, y_train2, y_test2 = BSMOTE()
# %time X_train3, X_test3, y_train3, y_test3 = SMOTESVM()
# %time X_train4, X_test4, y_train4, y_test4 = ADASYN()


# # Run SMOTE and get resampled data
%time X_train1, X_test1, y_train1, y_test1 = run_smote()

# Instantiate the model
gnb = GaussianNB()

# Fit the model to SMOTE-resampled training data
gnb.fit(X_train1, y_train1)

# Get predictions on the SMOTE-resampled test set
y_pred = gnb.predict(X_test1)

# Create a dictionary with metric names and corresponding values
gnb_dict = {
    'model': ['Gaussian Naive Bayes'],  
    'precision': precision_score(y_test1, y_pred),
    'recall': recall_score(y_test1, y_pred),
    'F1': f1_score(y_test1, y_pred),
    'accuracy': accuracy_score(y_test1, y_pred),
    'AUC': roc_auc_score(y_test1, y_pred)
}

# Convert the dictionary to a Pandas DataFrame
gnb_results = pd.DataFrame(gnb_dict)

# Print the table
print(gnb_results)









# Construct a logistic regression model and fit it to the training set
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
# get predictions on the test set
y_pred = log_clf.predict(X_test)
# Create a dictionary with metric names and corresponding values
lr_dict = {
    'model': ['Logistic Regression'],  
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred)
}

# Convert the dictionary to a Pandas DataFrame
lr_results = pd.DataFrame(lr_dict)

# Print the table
lr_results








# Compute values for confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_clf.classes_)
disp.plot(values_format='')
plt.title('Logistic Regression')
plt.show()


# Instantiate the model
gnb = GaussianNB()


# Fit the model to training data
gnb.fit(X_train, y_train)
# get predictions on the test set
y_pred = gnb.predict(X_test)


# Create a dictionary with metric names and corresponding values
gnb_dict = {
    'model': ['Gaussian Naive Bayes'],  
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred)
}

# Convert the dictionary to a Pandas DataFrame
gnb_results = pd.DataFrame(gnb_dict)

# Print the table
gnb_results


# Compute values for confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=gnb.classes_)

# Create display of confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
disp.plot(values_format='')
plt.title('GaussianNB')
plt.show()


# Instantiate the Support Vector Classifier model
svc = SVC(random_state=42)


# Fit the model to training data
svc.fit(X_train, y_train)
# Get predictions on the test set
y_pred_svc = svc.predict(X_test)


# Create a dictionary with metric names and corresponding values
svc_dict = {
    'model': ['Support Vector Classifier'],  
    'precision': precision_score(y_test, y_pred_svc),
    'recall': recall_score(y_test, y_pred_svc),
    'F1': f1_score(y_test, y_pred_svc),
    'accuracy': accuracy_score(y_test, y_pred_svc),
    'AUC': roc_auc_score(y_test, y_pred_svc)
}

# Convert the dictionary to a Pandas DataFrame
svc_results = pd.DataFrame(svc_dict)

# Print the table
svc_results


# Compute values for confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=gnb.classes_)

# Create display of confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
disp.plot(values_format='')
plt.title('Support Vector Classifier')
plt.show()


# Instantiate the K-Neighbors Classifier model
knc = KNeighborsClassifier()


# Fit the model to training data
knc.fit(X_train, y_train)
# Get predictions on the test set
y_pred_knc = knc.predict(X_test)


# Create a dictionary with metric names and corresponding values
knc_dict = {
    'model': ['K-Neighbors Classifier'],  
    'precision': precision_score(y_test, y_pred_knc),
    'recall': recall_score(y_test, y_pred_knc),
    'F1': f1_score(y_test, y_pred_knc),
    'accuracy': accuracy_score(y_test, y_pred_knc),
    'AUC': roc_auc_score(y_test, y_pred_knc)
}

# Convert the dictionary to a Pandas DataFrame
knc_results = pd.DataFrame(knc_dict)

# Print the table
knc_results


# Assign a dictionary of hyperparameters to search over
cv_params = {'n_neighbors': [29],
             'weights': ['distance']}

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
knc_cv = GridSearchCV(knc, cv_params, scoring=scoring, cv=4, refit='roc_auc')


%%time
knc_cv.fit(X_train, y_train)


# Check best params
knc_cv.best_params_


# Check best ROC score on CV
knc_cv.best_score_


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                        })
  
    return table


# Get all CV scores
knc_cv_results = make_results('K-Neighbors Classifier cv', knc_cv, 'auc')
knc_cv_results


def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
  
    return table


# Get predictions on test data
knc_cv_scores = get_scores('K-Neighbors Classifier Test', knc_cv, X_test, y_test)
knc_cv_scores


# confusion matrix
preds = knc_cv.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=knc_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=knc_cv.classes_)
disp.plot(values_format='')
plt.title('Support Vector Classifier')
plt.show()


# Instantiate the Quadratic Discriminant Analysis model
qda = QuadraticDiscriminantAnalysis()


# Fit the model to training data
qda.fit(X_train, y_train)
# Get predictions on the test set
y_pred_qda = qda.predict(X_test)


# Create a dictionary with metric names and corresponding values
qda_dict = {
    'model': ['Quadratic Discriminant Analysis'],  
    'precision': precision_score(y_test, y_pred_qda),
    'recall': recall_score(y_test, y_pred_qda),
    'F1': f1_score(y_test, y_pred_qda),
    'accuracy': accuracy_score(y_test, y_pred_qda),
    'AUC': roc_auc_score(y_test, y_pred_qda)
}

# Convert the dictionary to a Pandas DataFrame
qda_results = pd.DataFrame(qda_dict)

# Print the table
qda_results


# Assign a dictionary of hyperparameters to search over
cv_params = {'reg_param': [0.2]}

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
qda_cv = GridSearchCV(qda, cv_params, scoring=scoring, cv=4, refit='roc_auc')


%%time
qda_cv.fit(X_train, y_train)


# Check best params
qda_cv.best_params_


# Check best ROC score on CV
qda_cv.best_score_


# Get all CV scores
qda_cv_results = make_results('Quadratic Discriminant Analysis cv', qda_cv, 'auc')
qda_cv_results


# Get predictions on test data
qda_cv_scores = get_scores('Quadratic Discriminant Analysis Test', qda_cv, X_test, y_test)
qda_cv_scores


# confusion matrix
preds = qda_cv.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=qda_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=qda_cv.classes_)
disp.plot(values_format='')
plt.title('Quadratic Discriminant Analysis')
plt.show()


# Instantiate the Linear Discriminant Analysis model
lda = LinearDiscriminantAnalysis()


# Fit the model to training data
lda.fit(X_train, y_train)
# Get predictions on the test set
y_pred_lda = lda.predict(X_test)



# Create a dictionary with metric names and corresponding values
lda_dict = {
    'model': ['Linear Discriminant Analysis'],  
    'precision': precision_score(y_test, y_pred_lda),
    'recall': recall_score(y_test, y_pred_lda),
    'F1': f1_score(y_test, y_pred_lda),
    'accuracy': accuracy_score(y_test, y_pred_lda),
    'AUC': roc_auc_score(y_test, y_pred_lda)
}

# Convert the dictionary to a Pandas DataFrame
lda_results = pd.DataFrame(lda_dict)

# Print the table
lda_results


# Assign a dictionary of hyperparameters to search over
cv_params = {'solver': ['svd']}

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
lda_cv = GridSearchCV(lda, cv_params, scoring=scoring, cv=4, refit='roc_auc')


%%time
lda_cv.fit(X_train, y_train)


# Check best params
lda_cv.best_params_


# Check best ROC score on CV
lda_cv.best_score_


# Get all CV scores
lda_cv_results = make_results('Linear Discriminant Analysis cv', lda_cv, 'auc')
lda_cv_results


# Get predictions on test data
lda_cv_scores = get_scores('Linear Discriminant Analysis Test', lda_cv, X_test, y_test)
lda_cv_scores


# confusion matrix
preds = lda_cv.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=lda_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lda_cv.classes_)
disp.plot(values_format='')
plt.title('Linear Discriminant Analysis')
plt.show()


# Instantiate the model
decision_tree = DecisionTreeClassifier(random_state=42)


# Fit the model to training data
decision_tree.fit(X_train, y_train)
# Make predictions on test data
dt_pred = decision_tree.predict(X_test)


# Create a dictionary with metric names and corresponding values
dt_dict = {
    'model': ['Decision Tree'], 
    'precision': precision_score(y_test, dt_pred),
    'recall': recall_score(y_test, dt_pred),
    'F1': f1_score(y_test, dt_pred),
    'accuracy': accuracy_score(y_test, dt_pred),
    'AUC': roc_auc_score(y_test, dt_pred)
}

# Convert the dictionary to a Pandas DataFrame
dt_pred_results = pd.DataFrame(dt_dict)

# Print the table
dt_pred_results


# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[10],
             'min_samples_leaf': [4],
             'min_samples_split': [10]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
tree = GridSearchCV(decision_tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


%%time
tree.fit(X_train, y_train)


# Check best params
tree.best_params_


# Check best ROC score on CV
tree.best_score_


# Get all CV scores
tree_cv_results = make_results('Decision Tree cv', tree, 'auc')
tree_cv_results


# Get predictions on test data
tree_test_scores = get_scores('Decision Tree Test', tree, X_test, y_test)
tree_test_scores


# confusion matrix
preds = tree.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=tree.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=tree.classes_)
disp.plot(values_format='')
plt.title('Decision trees')
plt.show()


# Plot the tree
plt.figure(figsize=(30,15))
plot_tree(tree.best_estimator_,
          max_depth=5, fontsize=14, feature_names=X.columns, 
          class_names={ 1:'left',0: 'stayed'}, filled=True);
plt.show()


# Create a Random Forest Classifier with a specified random state
rf = RandomForestClassifier(random_state=42)

# Define a dictionary of hyperparameters to search over using GridSearchCV
cv_params = {'max_depth': [8],   
             'min_samples_leaf': [10],  
             'n_estimators': [10],  
             }  

# Define a set of scoring metrics to capture during cross-validation
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearchCV with the Random Forest Classifier, hyperparameters, scor ing, and cross-validation setup
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')



%%time 
rf_cv.fit(X_train, y_train)


# Check best params
rf_cv.best_params_


# Check best ROC score on CV
rf_cv.best_score_


# Get all CV scores
rf_cv_results = make_results('Random Forest cv', rf_cv, 'auc')
rf_cv_results


# Get predictions on test data
rf_test_scores = get_scores('Random Forest Test', rf_cv, X_test, y_test)
rf_test_scores


# confusion matrix
preds = rf_cv.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rf_cv.classes_)
disp.plot(values_format='')
plt.title('Random Forest')
plt.show()


# Instantiate the model
xgb = XGBClassifier(objective='binary:logistic', random_state=42)

# Assign a dictionary of hyperparameters to search over
cv_params = {'learning_rate': [0.026],
            'max_depth': [4],
            'min_child_weight': [2],
            'n_estimators': [322],
            'subsample': [0.50],
            'colsample_bytree': [0.95],
            'gamma': [0.15]
            }

#Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=4, refit='roc_auc')


%%time
xgb_cv.fit(X_train, y_train)


# Check best params
xgb_cv.best_params_


# Check best ROC score on CV
xgb_cv.best_score_


# Get all CV validation scores
xgb_cv_results = make_results('XGBoost cv', xgb_cv, 'auc')
xgb_cv_results


# Get predictions on test
xgb_test_scores = get_scores('XGBoost Test', xgb_cv, X_test, y_test)
xgb_test_scores


# confusion matrix
preds = xgb_cv.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=xgb_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=xgb_cv.classes_)
disp.plot(values_format='')
plt.title('Extreme Gradient Boost')
plt.show()


# Concatenate all the validation scores dataframes.
result = pd.concat([ knc_cv_results, qda_cv_results, lda_cv_results, 
                    tree_cv_results, rf_cv_results ,xgb_cv_results]).reset_index(drop=True)
result


# Concatenate all the test scores dataframes.
test = pd.concat([  lr_results, gnb_results, svc_results, 
                    knc_cv_scores, qda_cv_scores, lda_cv_scores,
                    tree_test_scores, rf_test_scores, xgb_test_scores]).reset_index(drop=True)
test


# Champion model
test.sort_values('AUC', ascending=False).head(1)


# confusion matrix
preds = qda_cv.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=qda_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=qda_cv.classes_)
disp.plot(values_format='')
plt.title('Extreme Gradient Boost')
plt.show()


m1 = 'Logistic Regression'
lr = LogisticRegression()
model = lr.fit(X_train, y_train)

# Get predicted classes for the test set
lr_predict = lr.predict(X_test)

# Calculate the ROC AUC score
lr_roc_auc_score = roc_auc_score(y_test, lr_predict)

lr_conf_matrix = confusion_matrix(y_test, lr_predict)
print("Confusion matrix")
print(lr_conf_matrix)
print("\n")

print("ROC AUC of Logistic Regression:", lr_roc_auc_score, '\n')
print(classification_report(y_test, lr_predict))



m2 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train, y_train)

# Get predicted classes for the test set
nbpred = nb.predict(X_test)

# Calculate the ROC AUC score
nb_roc_auc_score = roc_auc_score(y_test, nbpred)

nb_conf_matrix = confusion_matrix(y_test, nbpred)
print("Confusion matrix")
print(nb_conf_matrix)
print("\n")

print("ROC AUC of Naive Bayes model:", nb_roc_auc_score, '\n')
print(classification_report(y_test, nbpred))



m3 = 'Support Vector Classifier'
svc = SVC(probability=True) 
svc.fit(X_train, y_train)

# Get predicted classes for the test set
svc_predicted = svc.predict(X_test)

# Calculate the ROC AUC score
svc_roc_auc_score = roc_auc_score(y_test, svc_predicted)

svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
print("Confusion matrix")
print(svc_conf_matrix)
print("\n")

print("ROC AUC of Support Vector Classifier:", svc_roc_auc_score, '\n')
print(classification_report(y_test, svc_predicted))



m4 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=29,weights='distance')
knn.fit(X_train, y_train)

# Get predicted classes for the test set
knn_predicted = knn.predict(X_test)

# Calculate the ROC AUC score
knn_roc_auc_score = roc_auc_score(y_test, knn_predicted)

knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
print("Confusion matrix")
print(knn_conf_matrix)
print("\n")

print("ROC AUC of K-NeighborsClassifier:", knn_roc_auc_score, '\n')
print(classification_report(y_test, knn_predicted))



# Quadratic Discriminant Analysis
m5 = 'Quadratic Discriminant Analysis'
qda = QuadraticDiscriminantAnalysis(reg_param=0.2)
qda.fit(X_train, y_train)

# Get predicted classes for the test set
qda_predict = qda.predict(X_test)

# Calculate the ROC AUC score
qda_roc_auc_score = roc_auc_score(y_test, qda_predict)

qda_conf_matrix = confusion_matrix(y_test, qda_predict)
print("Confusion matrix")
print(qda_conf_matrix)
print("\n")

print("ROC AUC of Quadratic Discriminant Analysis:", qda_roc_auc_score, '\n')
print(classification_report(y_test, qda_predict))


# Linear Discriminant Analysis
m6 = 'Linear Discriminant Analysis'
lda = LinearDiscriminantAnalysis(solver = 'svd')
lda.fit(X_train, y_train)

# Get predicted classes for the test set
lda_predict = lda.predict(X_test)

# Calculate the ROC AUC score
lda_roc_auc_score = roc_auc_score(y_test, lda_predict)

lda_conf_matrix = confusion_matrix(y_test, lda_predict)
print("Confusion matrix")
print(lda_conf_matrix)
print("\n")

print("ROC AUC of Linear Discriminant Analysis:", lda_roc_auc_score, '\n')
print(classification_report(y_test, lda_predict))


m7 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(random_state=42, max_depth = 10, min_samples_leaf = 4, min_samples_split = 10)
dt.fit(X_train, y_train)

# Get predicted classes for the test set
dt_predicted = dt.predict(X_test)

# Calculate the ROC AUC score
dt_roc_auc_score = roc_auc_score(y_test, dt_predicted)

dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
print("Confusion matrix")
print(dt_conf_matrix)
print("\n")

print("ROC AUC of DecisionTreeClassifier:", dt_roc_auc_score, '\n')
print(classification_report(y_test, dt_predicted))



m8 = 'Random Forest Classifier'
rf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=8, min_samples_leaf= 10)
rf.fit(X_train, y_train)

# Get predicted classes for the test set
rf_predicted = rf.predict(X_test)

# Calculate the ROC AUC score
rf_roc_auc_score = roc_auc_score(y_test, rf_predicted)

rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
print("Confusion matrix")
print(rf_conf_matrix)
print("\n")

print("ROC AUC of Random Forest:", rf_roc_auc_score, '\n')
print(classification_report(y_test, rf_predicted))



m9 = 'Extreme Gradient Boost'
xgb = XGBClassifier(random_state=42, learning_rate = 0.026, max_depth=4, n_estimators=322,
                               subsample=0.50, min_child_weight= 2,
                               gamma=0.15,colsample_bytree =0.95)
xgb.fit(X_train, y_train)

# Get predicted classes for the test set
xgb_predicted = xgb.predict(X_test)

# Calculate the ROC AUC score
xgb_roc_auc_score = roc_auc_score(y_test, xgb_predicted)

xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
print("Confusion matrix")
print(xgb_conf_matrix)
print("\n")

print("ROC AUC of Extreme Gradient Boost:", xgb_roc_auc_score, '\n')
print(classification_report(y_test, xgb_predicted))



model_ev = pd.DataFrame({'Model': ['Logistic Regression','Gaussian Naive Bayes',
                                   'Support Vector Machine','K-Nearest Neighbour',
                                   'Quadratic Discriminant Analysis', 'Linear Discriminant Analysis',
                                   'Decision Tree','Random Forest','XGBoost Test'], 
                    'AUC': [lr_roc_auc_score, nb_roc_auc_score,
                            svc_roc_auc_score, knn_roc_auc_score,
                            qda_roc_auc_score, lda_roc_auc_score,
                            dt_roc_auc_score, rf_roc_auc_score, xgb_roc_auc_score]})
model_ev



lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test,lr_predict)
nb_false_positive_rate,nb_true_positive_rate,nb_threshold = roc_curve(y_test,nbpred)
rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,rf_predicted)                                                             
xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test,xgb_predicted)
knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,knn_predicted)
dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,dt_predicted)
svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,svc_predicted)
qda_false_positive_rate, qda_true_positive_rate, qda_threshold = roc_curve(y_test, qda_predict)
lda_false_positive_rate, lda_true_positive_rate, lda_threshold = roc_curve(y_test, lda_predict)


sns.set_style('ticks')
plt.figure(figsize=(10,5))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')
plt.plot(nb_false_positive_rate,nb_true_positive_rate,label='Naive Bayes')
plt.plot(svc_false_positive_rate,svc_true_positive_rate,label='Support Vector Classifier')
plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
plt.plot(qda_false_positive_rate, qda_true_positive_rate, label='Quadratic Discriminant Analysis')
plt.plot(lda_false_positive_rate, lda_true_positive_rate, label='Linear Discriminant Analysis')
plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Desion Tree')
plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')
plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,label='Extreme Gradient Boost')
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()


#test models
test


import pandas as pd

url = 'https://raw.githubusercontent.com/chbt-mehdi/Python/main/Credit%20Card%20Fraud%20Detection/Submission_ad.csv'
submission = pd.read_csv(url)
submission.head()

submission.to_csv('submission.csv', index=False)


import pandas as pd

url = 'https://raw.githubusercontent.com/chbt-mehdi/Python/main/Credit%20Card%20Fraud%20Detection/Submission_ad_svm.csv'
submission = pd.read_csv(url)
submission.head()

submission.to_csv('submission.csv', index=False)




