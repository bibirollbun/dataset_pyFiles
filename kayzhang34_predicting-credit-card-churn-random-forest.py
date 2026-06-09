#Importing packages for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing packages for creating a decision tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import learning_curve


# Default kaggle import
    # Input data files are available in the read-only "../input/" directory
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Loading data as dataframe and renaming column
df_train = pd.read_csv('/kaggle/input/ai-durg-credit-card-churn/train.csv')
df_train.rename(columns={'loan_defaulted':'Attrition_Flag'}, inplace=True)
df_train.head(5)


#Exploratary data analysis
df_train.info()


df_train.describe()


fig, ax = plt.subplots(1,2, figsize=(10,5)) # 1 row, 2 columns

sns.countplot(data = df_train
              , x ='Attrition_Flag'
              , ax=ax[0])
#pie
ax[1]=plt.pie(df_train['Attrition_Flag'].value_counts(),
            labels=['Retained', 'Churned'],
            autopct='%1.2f%%',
            explode=(0.05, 0),
            startangle=45)
fig.suptitle('Retained vs. Churned', fontsize=22)


catVar = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

fig, ax = plt.subplots(2 , 3, figsize = (28,14)) #2 rows, 3 columns
ax = ax.flatten()

for i, var in enumerate(catVar):
    sns.countplot(data = df_train, x = var, hue = 'Attrition_Flag', ax = ax[i])
    ax[i].tick_params(rotation = 30)

ax[5].set_visible(False)



numVar = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count'
          , 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal'
          , 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct'
          , 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

fig, ax = plt.subplots(5, 3, figsize = (20,25), dpi = 500) #5 rows, 3 columns
ax = ax.flatten()

for i, var in enumerate(numVar):
    sns.boxplot(data = df_train, x = var, y = 'Attrition_Flag', orient = "h" , ax = ax[i])
    
ax[-1].set_visible(False)


# Feature engineering by removing columns we don't want to model to new dataframe
df_model0 = df_train.drop(['id', 'CLIENTNUM'], axis = 1)

# Encoding categorical variables
df_model0 = pd.get_dummies(df_model0, drop_first=True)
df_model0.head()


# Define target and predictor variables
y = df_model0['Attrition_Flag']
X = df_model0.copy()
X = X.drop('Attrition_Flag', axis=1)

# Since our whole dataframe is the test set, we will train data using the entire dataframe
X_train, X_test, y_train, y_test = train_test_split(X 
                                                    , y 
                                                    , test_size= 0.25
                                                    , stratify=y
                                                    , random_state=34)


decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)
dt_pred = decision_tree.predict(X_test)


#Results of the standard model build
print("Accuracy:", "%.3f" % accuracy_score(y_test, dt_pred))
print("Precision:", "%.3f" % precision_score(y_test, dt_pred))
print("Recall:", "%.3f" % recall_score(y_test, dt_pred))
print("F1 Score:", "%.3f" % f1_score(y_test, dt_pred))


def matrix_plot(model, x_data, y_data):
  
    model_pred = model.predict(x_data)
    cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=model.classes_)
  
    disp.plot(values_format='')  # `values_format=''` suppresses scientific notation
    plt.show()
    
matrix_plot(decision_tree, X_test, y_test)


# Visualizing the tree plot
plt.figure(figsize=(15,12))
plot_tree(decision_tree, max_depth=2, fontsize=12, feature_names=X.columns, 
          class_names={0:'retained', 1:'churned'}, filled=True);
plt.show()


importances = decision_tree.feature_importances_

forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize = (20, 15))
sns.barplot(x=forest_importances.index, y=forest_importances.values, palette = 'Blues_r')
plt.xticks(rotation=30, ha='right', fontsize = 12);


# Importing GridSearchCV 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from xgboost import XGBClassifier
from xgboost import plot_importance

# Instantiate the XBGClassifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

#Define parameters
#These parameters are somewhat arbitrary, but allow us to cross validate and tune for performance
#We are keeping parameters within a certain range to avoid overfitting
cv_params = {'max_depth': [5, 7, 9]
           , 'min_child_weight': [7, 9]
           , 'learning_rate': [0.1, 0.2, 0.3]
           , 'n_estimators': [20, 50, 100]
           , 'subsample': [0.70]
           , 'colsample_bytree': [0.70]
        }

#Assign a set of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}


# Constructing Grid Search
xgb_cv = GridSearchCV(xgb,
                      cv_params,
                      scoring = scoring,
                      cv = 5,
                      refit = 'f1'
                     )


%%time

#Fit the model. This takes roughly 30 seconds to run
xgb_cv = xgb_cv.fit(X_train, y_train)
#Find Predictions 

xgb_cv


#This returns the parameters that yield the best F1 score
xgb_cv.best_params_


def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name and returns a df with model performance scores 
    for the model with the best mean F1 score across all validation folds.  
    '''

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean f1 score)
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    # Create table of results
    table = pd.DataFrame({'Model': [model_name],
                          'F1': [f1],
                          'Recall': [recall],
                          'Precision': [precision],
                          'Accuracy': [accuracy]
                         }
                        )
  
    return table

# Call the function on our model
result_table = make_results("XGBoost Decision Tree", xgb_cv)
result_table


# Calculate learning curves
train_sizes, train_scores, validation_scores = learning_curve(
    xgb_cv.best_estimator_, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy'
)

# Calculate mean and standard deviation of scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("XGBoosted Learning Curve")
plt.legend(loc="best")
plt.show()


#Returning our updated confusion matrix
matrix_plot(xgb_cv, X_test, y_test)


xg_importances = xgb_cv.best_estimator_.feature_importances_
model_importances = pd.Series(xg_importances, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize = (20, 15))
sns.barplot(x=model_importances.index, y=model_importances.values, palette = 'Blues_r')
plt.xticks(rotation=30, ha='right', fontsize = 12);


#Loading testing data as dataframe
df_test0 = pd.read_csv('/kaggle/input/ai-durg-credit-card-churn/test.csv')
df_test0.info()


#Applying the same feature engineering/encoding we applied to our training data
df_test = df_test0.drop(['id', 'CLIENTNUM'], axis = 1)
df_test = pd.get_dummies(df_test, drop_first=True)
#Checking output looks correct
df_test.head()


#Loading sample submission file 
df_sub = pd.read_csv('/kaggle/input/ai-durg-credit-card-churn/sample_submission.csv')
#We have the same row counts and no NULL values. This matches our test dataframe  
df_sub.info()


#Getting churn predictions for out test dataframe
predictions = xgb_cv.predict(df_test)

#I know df_sub and df_test have the same IDs in the same order so I can just do the below
    #df_sub['loan_defaulted'] = predictions  
#But to be sure, I will put predictions in the original dataframe and map in the submissions dataframe based on id
df_test0['loan_defaulted'] = predictions
#Our model is predicting a ~15.1% churn rate with the test data
df_test0['loan_defaulted'].value_counts(normalize = True)


#I will use 'id' to map 'loan_defaulted'
df_sub['loan_defaulted'] = df_sub.id.map(df_test0.set_index('id')['loan_defaulted'])

#Verifying our submission prediction aligns with the above prediction
df_sub['loan_defaulted'].value_counts(normalize = True)


#Submitting
df_sub.to_csv("submission.csv", index=False)

