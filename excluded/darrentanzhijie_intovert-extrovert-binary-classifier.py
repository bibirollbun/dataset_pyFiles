import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
import matplotlib.pyplot as plt


train_df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
datasert_df= pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


test_df


train_df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
datasert_df= pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

datasert_df = (
    datasert_df
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']

train_df = train_df.merge(datasert_df, how='left', on=merge_cols)
test_df = test_df.merge(datasert_df, how='left', on=merge_cols)

train_df['Personality'] = train_df['Personality'].replace({'Extrovert': 1, 'Introvert': 0})

train_ID = train_df['id']
test_ID = test_df['id']

train_df.drop(columns=['id','match_p'],inplace=True)
test_df.drop(columns=['id','match_p'],inplace=True)




train_df


train_df.isna().sum()


print(train_df.info())
print(train_df.describe())



sns.heatmap(train_df.corr(numeric_only=True),annot=True,cmap='coolwarm')


from pandas.plotting import scatter_matrix

scatter_matrix(train_df,figsize=(20, 8))


from sklearn.base import BaseEstimator, TransformerMixin

#dataframe X
class CascadingImputer(BaseEstimator,TransformerMixin):

    def __init__(self,bin_columns=None,impute_steps=None):
        '''
        bin_columns : list , columns to create bins
        impute_steps: list of tuples , [(target_col_to_fill_nan_values,[grp_col1,grp_col2,...])]
        '''
        self.bin_columns= bin_columns or []
        self.impute_steps = impute_steps or []
        
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        df=X.copy()
        for col in self.bin_columns:
            
            if col in df.columns:
                df[f'{col}_bin']=pd.qcut(df[col],4,labels=['Q1','Q2','Q3','Q4'])

        for target_col,grp_cols in self.impute_steps:
            for grp_col in grp_cols:
                grp_col=grp_col+'_bin'
                df[target_col]=df[target_col].fillna(df.groupby(grp_col)[target_col].transform('median'))
                
        for col in self.bin_columns:
            df.drop(columns=[f'{col}_bin'],inplace=True)
            
        return df

casading_imputer = CascadingImputer(
    bin_columns=['Social_event_attendance', 'Going_outside',
                 'Friends_circle_size', 'Post_frequency'],
    impute_steps=[
        ('Time_spent_Alone',       ['Social_event_attendance', 'Going_outside']),
        ('Social_event_attendance',['Going_outside', 'Friends_circle_size', 'Post_frequency']),
        ('Going_outside',          ['Social_event_attendance', 'Friends_circle_size', 'Post_frequency']),
        ('Friends_circle_size',    ['Social_event_attendance', 'Going_outside', 'Post_frequency']),
        ('Post_frequency',         ['Going_outside', 'Social_event_attendance','Friends_circle_size'])
    ]
)





from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

def fill_stage_drained_with_unKnow(X):
    df = X.copy()
    
    if 'Stage_fear' in df.columns:
        df['Stage_fear'] = df['Stage_fear'].fillna('UnKnow')
    
    if 'Drained_after_socializing' in df.columns:
        df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('UnKnow')
    
    return df


cat_pipe = Pipeline([
    ('fill_unknown', FunctionTransformer(fill_stage_drained_with_unKnow)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

num_pipe= Pipeline(
    [('fill_nan',casading_imputer),
    ('ss',StandardScaler())]
)




preprocessing=ColumnTransformer(
    [('nan_imputer_num_and_others',num_pipe,['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']),
     ('nan_imputer_cat_and_others',cat_pipe,['Stage_fear','Drained_after_socializing'])
    ], remainder='passthrough'
)


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

X=train_df.drop(columns=['Personality'])
y=train_df['Personality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)

def models_performance_cv(preprocessing_pipeline,model,X,y,cv=5):
    model_pipeline=Pipeline(
        [('pp',preprocessing_pipeline),
        ('model',model)]
    )

    cvs=cross_val_score(model_pipeline,X,y,scoring='roc_auc',cv=cv,verbose=0)
    print(f'{model} score : {cvs.mean()}')
    return 
    
    


lg=models_performance_cv(preprocessing,LogisticRegression(),X_train,y_train)
sgd=models_performance_cv(preprocessing,SGDClassifier(),X_train,y_train)
svc=models_performance_cv(preprocessing,SVC(),X_train,y_train)
rf= models_performance_cv(preprocessing,RandomForestClassifier(),X_train,y_train)


preprocessing


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

lg_pipeline=Pipeline(
    [('pp',preprocessing),
    ('logistic_regression',LogisticRegression())]
)

lg_pipeline.fit(X_train,y_train)

pred=lg_pipeline.predict(X_test)

print(f'roc auc score : {roc_auc_score(y_test,pred)}')
print(f'accuracy score : {accuracy_score(y_test,pred)}')


from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate_model(X_train, y_train, X_test, y_test, preprocessing, model):
    """
    Fit the model pipeline and print ROC AUC and accuracy scores.

    Parameters:
    - X_train: training features (DataFrame or array-like)
    - y_train: training labels (array-like)
    - X_test: test features (DataFrame or array-like)
    - y_test: test labels (array-like)
    - preprocessing: sklearn Transformer or ColumnTransformer pipeline
    - model: sklearn estimator object (e.g., LogisticRegression(), RandomForestClassifier())

    Returns:
    - pipeline: the fitted pipeline (Pipeline object)
    - scores: dictionary with keys 'roc_auc' and 'accuracy'
    """
    # Create a pipeline combining preprocessing and model
    pipeline = Pipeline([
        ('pp', preprocessing),
        ('model', model)
    ])

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Obtain predicted probabilities for roc_auc_score (binary classification)
    try:
        pred_proba = pipeline.predict_proba(X_test)[:, 1]
    except AttributeError:
        # Some models might not have predict_proba
        pred_proba = None

    # Obtain predicted classes
    pred = pipeline.predict(X_test)

    # Calculate scores
    if pred_proba is not None:
        roc_auc = roc_auc_score(y_test, pred_proba)
    else:
        # fallback to using predicted classes (less ideal)
        roc_auc = roc_auc_score(y_test, pred)

    accuracy = accuracy_score(y_test, pred)

    print(f'ROC AUC score : {roc_auc:.4f}')
    print(f'Accuracy score : {accuracy:.4f}')

    scores = {'roc_auc': roc_auc, 'accuracy': accuracy}
    return pipeline, scores

pipeline, scores = evaluate_model(X_train, y_train, X_test, y_test, preprocessing, LogisticRegression())
print(f'LogisticRegression : {scores}')
print('\n')

pipeline, scores = evaluate_model(X_train, y_train, X_test, y_test, preprocessing, SVC())
print(f'SVC : {scores}')
print('\n')

pipeline, scores = evaluate_model(X_train, y_train, X_test, y_test, preprocessing, SGDClassifier())
print(f'SGDClassifier : {scores}')
print('\n')

pipeline, scores = evaluate_model(X_train, y_train, X_test, y_test, preprocessing, RandomForestClassifier())
print(f'random forest : {scores}')


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform,randint

def hyperparameter_search(model, param_grid, X_train, y_train, X_test, y_test,preprocessing=None, model_name='model', n_iter=20, cv=3,scoring='accuracy', verbose=1):
    
    """
    returns best estimator from random search and results [dict]
    """
    
    #create pipeline
    if preprocessing is not None:
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            (model_name, model)
        ])
        #add correct prefix to param_grid keys if not already present
        formatted_param_grid = {}
        expected_prefix = f'{model_name}__'
        
        for key, value in param_grid.items():
            if not key.startswith(expected_prefix):
                formatted_param_grid[f'{expected_prefix}{key}'] = value
            else:
                formatted_param_grid[key] = value
    else:
        pipeline = model
        formatted_param_grid = param_grid


    rs = RandomizedSearchCV(pipeline, 
        formatted_param_grid, 
        n_iter=n_iter, 
        cv=cv, 
        scoring=scoring,
        verbose=verbose)
    
    rs.fit(X_train, y_train)
    
   
    train_score = rs.score(X_train, y_train)
    test_score = rs.score(X_test, y_test)
    y_pred = rs.predict(X_test)
    
   
    results = {
        'best_estimator': rs.best_estimator_,
        'best_params': rs.best_params_,
        'best_cv_score': rs.best_score_,
        'train_score': train_score,
        'test_score': test_score,
        'y_pred': y_pred,
        'randomized_search': rs
    }
    print(f'Model: {model_name}')
    print(f"Best CV Score: {rs.best_score_:.4f}")
    print(f"Test Score: {test_score:.4f}")
    print(f"Best Parameters: {rs.best_params_}")
    print('\n')
    #rs.best_estimator returns the entire pipeline including preprocessing step so lets just return the model only
    return rs.best_params_,results


lg_pipeline=Pipeline(
    [('pp',preprocessing),
    ('logistic_regression',LogisticRegression())]
)

lg_param_grid = {
    'logistic_regression__C': loguniform(0.01, 100),           # Regularization strength
    'logistic_regression__penalty': ['l1', 'l2'],              # Regularization type
    'logistic_regression__solver': ['liblinear'],              # Works with both l1 and l2
    'logistic_regression__max_iter': [1000]                    # Fixed iterations
}



svc_pipeline=Pipeline(
    [('pp',preprocessing),
    ('svc',SVC())]
)

svc_param_grid = {
    'svc__C': loguniform(0.1, 100),                    # Regularization parameter
    'svc__kernel': ['linear', 'rbf', 'poly'],          # Kernel type
    'svc__gamma': loguniform(1e-4, 1e-1),              # Kernel coefficient for rbf/poly
    'svc__degree': [2, 3, 4],                          # Degree for polynomial kernel
    'svc__probability': [True]                         # Enable probability estimates
}

rf_pipeline=Pipeline(
    [('pp',preprocessing),
     ('random_forest',RandomForestClassifier())]
)

rf_param_grid = {
    'random_forest__n_estimators': randint(50, 300),           # Number of trees
    'random_forest__max_depth': [3, 5, 7, 10, None],          # Tree depth
    'random_forest__min_samples_split': randint(2, 10),       # Min samples to split
    'random_forest__min_samples_leaf': randint(1, 5),         # Min samples in leaf
    'random_forest__max_features': ['sqrt', 'log2', None],    # Features per split
    'random_forest__bootstrap': [True, False]                 # Bootstrap sampling
}



lg_final_param,res_lg=hyperparameter_search(lg_pipeline,lg_param_grid,X_train,y_train,X_test,y_test,model_name='logistic_regression')
svc_final_param,res_svc=hyperparameter_search(svc_pipeline,svc_param_grid,X_train,y_train,X_test,y_test,model_name='svc')
rf_final_param,res_rf=hyperparameter_search(rf_pipeline,rf_param_grid,X_train,y_train,X_test,y_test,model_name='random_forest')


def strip_prefix(params, prefix):
    """Remove pipeline prefix from parameter names"""
    stripped_params = {}
    prefix_with_separator = f"{prefix}__"
    
    for key, value in params.items():
        if key.startswith(prefix_with_separator):
            # Remove the prefix and separator
            new_key = key[len(prefix_with_separator):]
            stripped_params[new_key] = value
        else:
          
            stripped_params[key] = value
    
    return stripped_params

# Strip prefixes from your best parameters
rf_clean_params = strip_prefix(res_rf['best_params'], 'random_forest')
svc_clean_params = strip_prefix(res_svc['best_params'], 'svc')  
lg_clean_params = strip_prefix(res_lg['best_params'], 'logistic_regression')


from sklearn.ensemble import StackingClassifier

#fresh models with best parameters

final_model=StackingClassifier([('random_forest',RandomForestClassifier(**rf_clean_params)),('svc',SVC(**svc_clean_params))],final_estimator=LogisticRegression(**lg_clean_params))
final_model_pipeline=Pipeline([('pp',preprocessing),('final_model',final_model)])



final_model_pipeline


final_model_pipeline.fit(X_train,y_train)


pred=final_model_pipeline.predict(test_df)
# Create submission
submission = pd.DataFrame({
    'id': test_ID,
    'Personality': pred
})

print(submission.head())
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv('submission.csv', index=False)


submission




