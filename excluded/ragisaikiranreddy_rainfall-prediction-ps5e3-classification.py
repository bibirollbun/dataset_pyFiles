import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


X_full = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col = 'id')
X_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 'id')

print(X_full.shape, X_test.shape)


X_full.info()


X_full.head()


X_full.isna().sum()  #no null values in training data


y = X_full['rainfall']
X = X_full.drop(['rainfall'],axis=1)
print(X.shape, y.shape)


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size= 0.2, random_state=1)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)


X_train.info()


X_train.head()


cat_cols = [col for col in X_train.columns if X_train[col].dtype== 'object'] 
num_cols = [col for col in X_train.columns if col not in cat_cols]
cat_cols


# columns to be scaled
scaling_cols = [col for col in X_train.columns if X_train[col].dtype in ['float64','int64'] and col!='day']
day_col = ['day']
scaling_cols



# plotting imbalance of target values in a data
idx = y_train.value_counts().index
val = y_train.value_counts().values
plt.bar(idx,val)
plt.xticks(idx)


X_train.corrwith(y_train).sort_values(ascending=False)


# absolute correlation of features with a target rainfall  (required for further feature engineering)
abs(X_train.corrwith(y_train)).sort_values(ascending=True).plot(kind='barh')


print(abs(X_train[scaling_cols].skew()))
# can use median imputation as most of the values are greater then 0.5


from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

day_col_transformer = make_pipeline(SimpleImputer(strategy='median'))

numerical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('day',day_col_transformer, day_col),
        ('num', numerical_transformer,scaling_cols)
    ])


# cross val with a single model (XGB_classifier)

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


xgb_model = XGBClassifier(n_estimators=150, random_state=1)

clf = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',xgb_model)])


scores = cross_val_score(clf,X,y,
                             cv = 5,
                              scoring = 'roc_auc')

print(scores)
print("Average roc_auc score :")
print(scores.mean())


# # Transform data manually (since Pipeline won't pass extra parameters)
# X_train_transformed = preprocessor.fit_transform(X_train)
# X_valid_transformed = preprocessor.transform(X_valid)


# # clf.named_steps['model'].fit()
# xgb_model.fit(X_train_transformed, y_train, 
#              early_stopping_rounds=5, 
#              eval_set=[(X_valid_transformed, y_valid)], 
#              verbose=False)

# print("-------------------")
# # # alternate way
# # clf.set_params(model__early_stopping_rounds=5, model__eval_set=[(X_valid, y_valid)], model__verbose=False)
clf.fit(X, y)
test_preds = clf.predict_proba(X_test)[:,1]
# test_preds


# y_preds_score = xgb_model.predict_proba(X_valid_transformed)[:,1]
# roc_auc_score(y_valid, y_preds_score)


# # Preprocessing of test data, predict
# X_test_transformed = preprocessor.transform(X_test)

# y_preds_test = clf.named_steps['model'].predict_proba(X_test_transformed)[:,1]


# Save test predictions to file
output = pd.DataFrame()
output['id'] = X_test.index
output['rainfall'] = test_preds

# output.to_csv(f'xgb_pipeline_cv_ps5e3_submission.csv',index= False)
print(f'Submission file using XGB classifier model created successfully!')


output.head()


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score


# model_1 = DecisionTreeClassifier(random_state=1)
model_1 = SVC(probability=True)
model_2 = RandomForestClassifier(random_state=42)
model_3 = XGBClassifier(n_estimators= 150, random_state=1)
model_4 = LGBMClassifier()
model_5 = DecisionTreeClassifier(min_samples_split=15,max_depth= 7,max_leaf_nodes =100, random_state=1)
model_6 = RandomForestClassifier(n_estimators=150,min_samples_split=15, random_state= 1)
model_7 = RandomForestClassifier(n_estimators=100,max_depth= 7, random_state= 1)
model_8 = RandomForestClassifier(n_estimators=150,min_samples_split=15,max_depth= 7, random_state= 1)


# function to check the cross_val_score mean of roc_auc_score of a respective model 
# note: no need to create validation data seperately..can directly pass whole training data (X,y)

def get_cross_val_score(model, X,y):
    clf = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model)])
    
    scores = cross_val_score(clf,X,y,
                             cv = 5,
                              scoring = 'roc_auc')
    print(model)
    print('\nvalidation scores:',scores)

    # clf.fit(X_train,y_train)
    # y_pred_score = clf.predict_proba(X_valid)[:,1]
    # return roc_auc_score(y_valid, y_pred_score)

    # print(f'cross_val_score mean of ROC AUC Scores using {model} model:\n')
    print(f'\nmean of val scores using (ROC AUC Scores) with this model:\n')
    return scores.mean()


# Model validating comparing ROC AUC Scores
models = [model_1, model_2, model_3,model_4, model_5, model_6,model_7,model_8]

for model in models:
    print("-----------------------\n")
    
    print(get_cross_val_score(model, X,y))


# function to train the pipeline, get cross_val_score and return predict probs of X_test
def get_test_preds(model, X,y,X_test):
    clf = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model)])
    
    scores = cross_val_score(clf,X,y,
                             cv = 5,
                              scoring = 'roc_auc')
    print('\nvalidation scores:',scores)
    print('cross_val_score: ',scores.mean())
    

    clf.fit(X,y)
    # y_pred_score = clf.predict_proba(X_valid)[:,1]
    # return roc_auc_score(y_valid, y_pred_score)
    return clf.predict_proba(X_test)[:,1]


# better_models = [model_3,model_4,model_8]

models = {
    'SupportVector':model_1,
    'DecisionTree': model_5,
    'RandomForest': model_8,
    'XGB': model_3,
    'LGBM': model_4
}

for model_name in models:
    # print()
    print(f'Training: {model_name} ===>\n')
    model = models[model_name]
    
    # model.fit(X,y)
    y_prob_scores = get_test_preds(model,X,y,X_test)
    
    output = pd.DataFrame()
    output['id'] = X_test.index
    output['rainfall'] = y_prob_scores
    
    print('\n pred prob scores of of X_test: \n',output.head())
    
    output.to_csv(f'{model_name}_pipeline_cv_ps5e3_submission.csv',index= False)
    print(f'\nSubmission file using {model_name} classifier model created successfully!')
    print('-----------------------------------\n')







