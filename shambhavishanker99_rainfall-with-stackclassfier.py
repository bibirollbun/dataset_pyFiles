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


#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
#import joblib
import xgboost as xgb
#from sklearn.svm import SVC
import lightgbm as lgb
from scipy.stats import randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression



df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df.head()


#dropping id column as it is not needed
df.drop(columns='id',axis=1,inplace=True)


#splitting df into feature X and target Y

X = df.drop(columns='rainfall',axis=1)
Y = df['rainfall']


#Handling Outliers using IQR and Capping
def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    #calcluating upar bound and lower bound
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    # Cast bounds to the same dtype as the column
    lower_bound = lower_bound.astype(df[col].dtype)
    upper_bound = upper_bound.astype(df[col].dtype)

    
    df.loc[df[col] > upper_bound, col] = upper_bound
    df.loc[df[col] < lower_bound, col] = lower_bound


#invoking cap_outlier for feature with outleir in them 
cap_out=['pressure','mintemp','humidity','dewpoint','windspeed','cloud']

for col in cap_out:
    cap_outliers(X,col)


#standardizing the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)


# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=X.columns)


#creating a new df conatining only max_temp,min_temp and dew point
corr_col = ['maxtemp','mintemp','dewpoint']



def PCA_fun(df,corr_col):
    corr_df= df[corr_col]
    
    #Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 components
    df_pca_output = pca.fit_transform(corr_df)
    
    #Convert PCA result to DataFrame
    df_pca_output = pd.DataFrame(df_pca_output, columns=['PC1', 'PC2'])
    
    # Add PCA components back to the original scaled DataFrame
    df_scaled_pca = pd.concat([df, df_pca_output], axis=1)

    #dropping corr_col from output
    df_scaled_pca.drop(columns=corr_col,axis=1,inplace=True)

    #df_scaled_pca.head()
    return df_scaled_pca


#invoking PCA 
df_scaled_pca = PCA_fun(df_scaled,corr_col)


df_scaled_pca.head()


#print shape before smote
print(df_scaled_pca.shape,Y.shape)



    #loading smote
smote = SMOTE()
    
    #performing SMOTE
df_scaled_pca,Y = smote.fit_resample(df_scaled_pca,Y)




#print shape after smote
print(df_scaled_pca.shape,Y.shape)


#split the data
X_train, X_test, Y_train, Y_test = train_test_split(df_scaled_pca,Y,test_size = 0.2, stratify = Y, random_state=42) 


#StraightKFold
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)



# Define the parameter distribution
param_dist = {
    'num_leaves': randint(20, 150),
    'max_depth': randint(-1, 50),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(50, 500),
    'min_child_samples': randint(10, 50),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

# Initialize the model
lgb_model = lgb.LGBMClassifier()

# Perform randomized search
random_search_lgb = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_dist, n_iter=100, cv=stratified_kfold, scoring='accuracy', n_jobs=-1, random_state=42)
random_search_lgb.fit(X_train, Y_train)

# Best parameters and score
print(f'Best parameters for LightGBM: {random_search_lgb.best_params_}')
print(f'Best score for LightGBM: {random_search_lgb.best_score_}')



#DecisionTreeClassifier

#defining the parameters
param_dist = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [np.random.randint(2, 20)],
    'min_samples_leaf': [np.random.randint(1, 20)],
    'max_features': [None, 'sqrt', 'log2']
}

#gridsearchcv
tree = GridSearchCV(DecisionTreeClassifier(), param_dist, cv=stratified_kfold, n_jobs=-1) 

#fitting the model in gridsearch
tree.fit(X_train, Y_train)

#printing the best hyperparameters
print(tree.best_params_)

#printing the best accuracy score
print(tree.best_score_)


#RandomForestClassifier
# Initialize the DecisionTreeClassifier with some hyperparameters
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [np.random.randint(2, 20)],
    'min_samples_leaf': [np.random.randint(1, 20)],
    'max_features': [None, 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'oob_score': [True, False]
}

#gridsearchcv
grid = RandomizedSearchCV(RandomForestClassifier(), param_dist, cv=stratified_kfold, n_jobs=-1)  

#fitting the model in gridsearch
grid.fit(X_train, Y_train)

#printing the best hyperparameters
print(grid.best_params_)

#printing the best accuracy score
print(grid.best_score_)


# Define the parameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 300]
}

# Initialize GridSearchCV with Logistic Regression
log = GridSearchCV(LogisticRegression(), param_grid, cv=stratified_kfold, n_jobs=-1)

# Fit the GridSearchCV on the training data
log.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best hyperparameters for Logistic Regression:", log.best_params_)

# Print the best score
print("Best score for Logistic Regression:", log.best_score_)


#XGBoost
# Defining the parameters
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# GridSearchCV
xgb_model = RandomizedSearchCV(xgb.XGBClassifier(objective='binary:logistic', random_state=42), param_dist, cv=stratified_kfold, n_jobs=-1)

# Fitting the model in GridSearch
xgb_model.fit(X_train, Y_train)

# Printing the best hyperparameters
print(xgb_model.best_params_)

# Printing the best accuracy score
print(xgb_model.best_score_)


param_LightGBM = random_search_lgb.best_params_
param_tree = tree.best_params_
param_forest = grid.best_params_
param_log = log.best_params_
param_xgb = xgb_model.best_params_


#training base models

#LightGBM
model1_lgb = lgb.LGBMClassifier(**param_LightGBM , verbose=-1)
model1_lgb.fit(X_train, Y_train)

y_probs = model1_lgb.predict_proba(X_test)[:, 1]

print(f'ROC AUC Score": {roc_auc_score(Y_test, y_probs)}')



#DecisionTree
model2_tree = DecisionTreeClassifier(**param_tree)
model2_tree.fit(X_train, Y_train)

y_probs = model2_tree.predict_proba(X_test)[:, 1]

print(f'ROC AUC Score": {roc_auc_score(Y_test, y_probs)}')


#random forest
model3_forest = RandomForestClassifier(**param_forest)
model3_forest.fit(X_train,Y_train)

y_probs = model3_forest.predict_proba(X_test)[:, 1]

print(f'ROC AUC Score": {roc_auc_score(Y_test, y_probs)}')


#xgbBoost
model4_xgb = xgb.XGBClassifier(**param_xgb)
model4_xgb.fit(X_train,Y_train)

y_probs = model4_xgb.predict_proba(X_test)[:, 1]

print(f'ROC AUC Score": {roc_auc_score(Y_test, y_probs)}')


meta_model = LogisticRegression(**param_log)


stack_clf = StackingClassifier(
    estimators=[  
        ("tree", model2_tree),  
        ("forest", model3_forest),
        ("xgb",model4_xgb),
        ("lgb", model1_lgb),
    ],
    final_estimator=meta_model,
    cv=stratified_kfold,
    passthrough=True 
)

stack_clf.fit(X_train, Y_train)

#predicting the target values
y_probs = stack_clf.predict_proba(X_test)[:, 1]

print(f'ROC AUC Score": {roc_auc_score(Y_test, y_probs)}')


#loading test df

test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


#hanlding missing value
# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
test_df['winddirection'] = imputer.fit_transform(test_df[['winddirection']])


#extracting passengerid in different 

ID = test_df['id'].tolist()


test_df.drop(columns='id',axis=1,inplace=True)


#standardizing the data
scaler = StandardScaler()
test_scaled = scaler.fit_transform(test_df)


# Convert the scaled data back to a DataFrame
test_scaled = pd.DataFrame(test_scaled, columns=X.columns)


#creating a new df conatining only max_temp,min_temp and dew point
test_col = ['maxtemp','mintemp','dewpoint']


test_scaled_pca = PCA_fun(test_scaled,test_col)


#making prediction
prediction = stack_clf.predict(test_scaled_pca)


# Making probability predictions
probabilities = stack_clf.predict_proba(test_scaled_pca)[:, 1]


#saving
results = pd.DataFrame({
    'id': ID,
    'rainfall': probabilities
})

# Check the result
results.head()


# Convert the DataFrame to a CSV file and drop the index
results.to_csv('rainfall_predictions.csv', index=False)

