import matplotlib as plt # import for graphing
import seaborn as sns # import for graphing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier #importing the ML model being used
from time import time # importing to determine how long model fitting works
from sklearn.preprocessing import OrdinalEncoder # importing encoder for columns with a natural order
# below two imports are for the HalvingGridSearchCV
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV, RandomizedSearchCV


# shows all columns in any displayed DFs
pd.set_option('display.max_columns', None)

# reading the train and test data into the respective variables
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')

# defining the name of the target column for easy retrieval later
TARGET = 'diagnosed_diabetes'

print('All files loaded successfully!')
train


# identifying numerical and categorical columns
num_cols = [col for col in test.columns if test[col].dtype in ['float64', 'int64']]
cat_cols = [col for col in train.columns if train[col].dtype in ['object']]


def cat_group_percentage(cat_cols):
   
    for column_name1 in cat_cols:
        unique_columns1 = train[column_name1].unique().tolist()
        cat_cols.remove(column_name1)
        for column_name in cat_cols:
            unique_columns = train[column_name].unique().tolist()
            per_dict = {}

            print(f"{column_name1} and {column_name} Percentage List:")
            for col1 in unique_columns1:
                
                for col in unique_columns:
                    temp_per = (train[ (train[column_name] == col) & (train[column_name1] == col1)& (train[TARGET] == 1.0) ][TARGET].count() / ( (train[train[column_name1]==col1]) + (train[ train[column_name] == col])).count().sum()) * 100
                    

                    print(f"{col1} & {col}: {temp_per:.2f}")    


# examining all the unique cat columns groups in reference to having diabetes
#     No groups have significantly higher percentages 
#cat_group_percentage(cat_cols)


# One Hot Encoding Cat columns without any order
train = pd.get_dummies(train, columns=['gender', 'ethnicity', 'employment_status'], drop_first=False)
test = pd.get_dummies(test, columns=['gender', 'ethnicity', 'employment_status'], drop_first=False)


# Ordinal Encoding cat columns with order
ord_cols = ['education_level', 'income_level', 'smoking_status']
ordinal_encoder = OrdinalEncoder()
train[ord_cols] = ordinal_encoder.fit_transform(train[ord_cols])
test[ord_cols] = ordinal_encoder.fit_transform(test[ord_cols])


# Feature Engineering
train["activity_diet_score"]= train["physical_activity_minutes_per_week"] / train['diet_score']
test["activity_diet_score"]= test["physical_activity_minutes_per_week"] / test['diet_score']


# separating data into the final X and y
y = train['diagnosed_diabetes']

X = train.drop(TARGET,axis=1)


# defining the grid of hyper parameters for the xgb model
param_grid = {
    'n_estimators': [500, 750, 1000, 1500, 1750, 2000],#[250, 400, 800, 1250, 1500, 2000], # number of trees the model will create
    'learning_rate': [.1, .05, .03, .025], # how fast the model moves towards minimums
    'max_depth': [1, 2, 3, 5], # max depth a tree can go
    'min_child_weight': [3, 5, 7]
}

# instantiating the xgb model
xgb = XGBClassifier(random_state=0)

# calling the experimental HalvingGridSearchCV
#     HalvingGridSearchCV tests all combinations within the above param grid
#       however, it tests all combinations with a small amount of resources (AKA data rows) and eliminates low performers
#       after each iteration/elimination of low performers, the amount of resources doubles with each iteration until only the top model remains
h_gridsearch = HalvingGridSearchCV(estimator=xgb, param_grid=param_grid, cv=4, scoring='roc_auc', verbose=1, factor=2, random_state=0, min_resources=15000)


# using the time lib to determine the runtime of the HalvingGridSearch
tick = time()
h_gridsearch.fit(X,y) # fitting the model to the data while testing all params

# getting time and printing how many minutes it took the HalvingGridSearch to run
tick2 = time()
elapsed = tick2 - tick
print(f"Time Passed: {elapsed/60} minutes")


# historical best score
best_score = 0.7237262180719664

# pulling out the best model to use for submission predictions
best_model = h_gridsearch.best_estimator_ 
# pulling out the best combination of hyperparamters to later print
best_params = h_gridsearch.best_params_

#printing the best score and the best combination of hyper parameters
print("Best Score:", h_gridsearch.best_score_)
print("Best Params:", best_params)

if (best_score < h_gridsearch.best_score_):
    print()
    print("New Best Score!")

# collecting the model's prediction from the testing data
preds = best_model.predict_proba(test)[:,1]


#creating submission file for the competition
submission = pd.DataFrame({'id': test.index, 'diagnosed_diabetes': preds})
submission.to_csv('submission.csv', index=False)

