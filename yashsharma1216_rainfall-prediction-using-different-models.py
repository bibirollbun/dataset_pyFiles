import pandas as pd
import numpy as np
import optuna
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv('train.csv')


df.head()


df.shape


df.info()


df.duplicated().sum()


df.drop(columns='id',inplace=True)


df.describe()


df['rainfall'].value_counts()


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


from sklearn.model_selection import train_test_split,cross_val_score


x = df.drop(columns='rainfall')
y = df['rainfall']

x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=69)


x_train.shape


x_test.shape


x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



x_train_scaled


## Logistic regression 


from sklearn.linear_model import LogisticRegression
lr  = LogisticRegression()


lr.fit(x_train_scaled,y_train)


train_pred = lr.predict(x_train_scaled)
test_pred = lr.predict(x_test_scaled)


from sklearn.metrics import accuracy_score


accuracy_score(y_train,train_pred)


accuracy_score(y_test,test_pred)



train_pred_prob = lr.predict_proba(x_train_scaled)[:,1]
test_pred_prob = lr.predict_proba(x_test_scaled)[:,1]



from sklearn.metrics import roc_curve, auc


fpr, tpr, thresholds = roc_curve(y_test, test_pred_prob)
roc_auc = auc(fpr, tpr) 


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')  # Diagonal Line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()








test = pd.read_csv('test.csv')


test.head()


test.info()


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the DataFrame
test = pd.DataFrame(imputer.fit_transform(test), columns=test.columns)


import seaborn as sns
plt.figure(figsize=(8, 5))
sns.boxplot(y=test['winddirection'])
plt.title("Box Plot of Values (Outliers in Red)")
plt.show()


test[test['winddirection'].isnull()]


lr_prediction = lr.predict_proba(test.drop(columns='id'))[:,1]


test['id'] = test['id'].astype('int')


lr_prediction = [f"{p:.20f}" for p in lr_prediction]
# print(formatted_probs)


lr_output = pd.DataFrame({'id': test['id'], 'rainfall': lr_prediction})



lr_output


lr_output.to_csv('lr_submission.csv',index=False)


## SVM


from sklearn.svm import SVC


# Create SVM model
svm_model = SVC(kernel='rbf', C=1.0,probability =True)


svm_model.fit(x_train_scaled,y_train)


train_pred = svm_model.predict(x_train_scaled)
test_pred = svm_model.predict(x_test_scaled)


accuracy_score(y_train,train_pred)


accuracy_score(y_test,test_pred)


train_pred_prob = svm_model.predict_proba(x_train_scaled)[:,1]
test_pred_prob = svm_model.predict_proba(x_test_scaled)[:,1]



fpr, tpr, thresholds = roc_curve(y_test, test_pred_prob)
roc_auc = auc(fpr, tpr) 


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')  # Diagonal Line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()





def run_model(model,parameters=None):

    if parameters is None:
        random_search = model
    else:
        random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=parameters, 
        n_iter=50, 
        scoring='accuracy', 
        cv=5, 
        random_state=42
        )
        
    random_search.fit(x_train_scaled,y_train)

    print(random_search.best_params_)

    train_pred = random_search.predict(x_train_scaled)
    test_pred =  random_search.predict(x_test_scaled)

    print("Accuracy on train data:", accuracy_score(y_train,train_pred))
    print("Accuracy on test data:", accuracy_score(y_test,test_pred))

    train_pred_prob = random_search.predict_proba(x_train_scaled)[:,1]
    test_pred_prob =  random_search.predict_proba(x_test_scaled)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, test_pred_prob)
    roc_auc = auc(fpr, tpr) 

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')  # Diagonal Line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    answer = input("want to get prediction file?")
    if answer == "yes":
        prediction = random_search.predict_proba(test.drop(columns='id'))[:,1]
        output = pd.DataFrame({'id': test['id'], 'rainfall': prediction})

        model_name = input("Enter the model name")
        output.to_csv(f'{model_name}_submission.csv',index=False)
    else:
        print("end")





run_model(SVC(kernel='rbf', C=1.0,probability =True))


run_model(LogisticRegression())



accuracy_score(y_test,y_pred)


## Random Forest


from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier()


param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],  # Number of trees
    'criterion': ['gini', 'entropy', 'log_loss'],  # Splitting criteria
    'max_depth': [None, 5, 10, 20, 50, 100],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10, 20],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4, 10],  # Minimum samples at leaf nodes
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],  # Minimum weighted fraction of samples at a leaf
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider for split
    'max_leaf_nodes': [None, 10, 20, 50, 100],  # Max number of leaf nodes
    'min_impurity_decrease': [0.0, 0.1, 0.2],  # Minimum impurity decrease required to split
    'bootstrap': [True, False],  # Whether bootstrap samples are used
    'oob_score': [True, False],  # Whether to use out-of-bag samples for scoring
    'random_state': [42],  # Ensures reproducibility
    'ccp_alpha': [0.0, 0.01, 0.1],  # Complexity parameter for pruning
    'max_samples': [None, 0.5, 0.75, 1.0]  # Percentage of samples used for training each tree
}


run_model(rf,param_grid)


## XGBoost


from xgboost import XGBClassifier


xgb = XGBClassifier()


param_grid = {
    'n_estimators': [100, 200, 500, 1000],  # Number of trees in the ensemble
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],  # Step size shrinkage for boosting
    'max_depth': [3, 5, 7, 10, 15],  # Maximum tree depth
    'min_child_weight': [1, 3, 5, 10],  # Minimum sum of weights for a child node split
    'subsample': [0.6, 0.8, 1.0],  # Fraction of samples to grow trees
    'colsample_bytree': [0.4, 0.6, 0.8, 1.0],  # Fraction of features for each tree
    'colsample_bylevel': [0.4, 0.6, 0.8, 1.0],  # Fraction of features for each split
    'colsample_bynode': [0.4, 0.6, 0.8, 1.0],  # Fraction of features per node split
    'gamma': [0, 0.1, 0.2, 0.5, 1],  # Minimum loss reduction for a split
    'reg_alpha': [0, 0.1, 0.5, 1, 10],  # L1 regularization
    'reg_lambda': [1, 0.5, 0.1, 10, 100],  # L2 regularization
    'scale_pos_weight': [1, 10, 25, 50],  # Balancing classes in case of imbalance
    'booster': ['gbtree', 'gblinear', 'dart'],  # Type of booster
    'tree_method': ['auto', 'exact', 'approx', 'hist', 'gpu_hist'],  # Tree construction algorithm
    'random_state': [42]  # Ensures reproducibility
}


run_model(xgb,param_grid)





## Gradient boosting


from sklearn.ensemble import GradientBoostingClassifier


param_grid = {
    'n_estimators': [100, 200, 500, 1000],  # Number of boosting stages
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],  # Shrinkage rate
    'max_depth': [3, 5, 7, 10],  # Maximum depth of individual estimators
    'min_samples_split': [2, 5, 10, 20],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4, 10],  # Minimum samples required to be a leaf node
    'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider for split
    'subsample': [0.6, 0.8, 1.0],  # Fraction of samples for training each estimator
    'criterion': ['friedman_mse', 'mse'],  # Criterion to measure the quality of a split
    'loss': ['deviance', 'exponential'],  # Loss function to optimize
    'random_state': [42],  # For reproducibility
    'warm_start': [True, False],  # Reuse results of previous call to fit
    'max_leaf_nodes': [None, 10, 20, 50],  # Maximum leaf nodes
    'min_impurity_decrease': [0.0, 0.1, 0.2],  # Minimum impurity decrease for a split
    'ccp_alpha': [0.0, 0.01, 0.1],  # Complexity parameter for pruning
}


gb = GradientBoostingClassifier()


run_model(gb,param_grid)






## Ada Boost


from sklearn.ensemble import AdaBoostClassifier

# Comprehensive parameter grid for AdaBoostClassifier
param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000, 1500],  # Number of estimators (boosting rounds)
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 1.5],  # Shrinkage rate applied to weights
    'algorithm': ['SAMME', 'SAMME.R'],  # Algorithm for boosting
    'estimator': [None],  # Optionally, you could define custom base estimators like DecisionTreeClassifier
    'random_state': [42],  # Ensures reproducibility
}


ab = AdaBoostClassifier()


run_model(ab,param_grid)


## Light GBM


from lightgbm import LGBMClassifier

# Comprehensive parameter grid for LGBMClassifier
param_grid = {
    'n_estimators': [100, 200, 500, 1000, 2000],  # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # Learning rate
    'max_depth': [-1, 3, 5, 7, 10, 15],  # Maximum depth of the trees (-1 means no limit)
    'num_leaves': [31, 50, 70, 100, 150],  # Maximum number of leaves per tree
    'min_child_samples': [10, 20, 30, 50, 100],  # Minimum number of samples in a child node
    'min_child_weight': [1e-3, 1e-2, 0.1, 1, 5],  # Minimum sum of instance weight in a leaf
    'subsample': [0.6, 0.8, 1.0],  # Fraction of samples for training each iteration
    'subsample_freq': [0, 1, 5, 10],  # Frequency for subsampling
    'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features used for training each tree
    'reg_alpha': [0.0, 0.1, 0.5, 1.0, 5.0],  # L1 regularization term
    'reg_lambda': [0.0, 0.1, 0.5, 1.0, 5.0],  # L2 regularization term
    'max_bin': [255, 512, 1024],  # Maximum number of bins for numeric features
    'boosting_type': ['gbdt', 'dart', 'goss'],  # Boosting method
    'objective': ['binary', 'multiclass', 'multiclassova'],  # Learning task and objective
    'scale_pos_weight': [1, 10, 25, 50],  # Weight for balancing classes in imbalanced datasets
    'random_state': [42],  # Seed for reproducibility
    'verbose': [-1]  # Suppresses output logging
}



lgb = LGBMClassifier()


run_model(lgb,param_grid)




