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


import pandas as pd
import sklearn
import xgboost as xgb
import cupy as cp
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, roc_auc_score



#models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier



train_path = '/kaggle/input/playground-series-s5e8/train.csv'
test_path = '/kaggle/input/playground-series-s5e8/test.csv'


df = pd.read_csv(train_path)
print(df.shape)
df.head()


cat_cols = [col for col in df.columns if df[col].dtype == 'object']
df[cat_cols].head()


for col in cat_cols:
    print(f'column: {col}, cardinality: {len(df[col].unique())}')


def bin_map(df):
    binary_cols = ['default','housing','loan']
    bin_map = {'yes':1,'no':0}
    
    for col in binary_cols:
        df[col] = df[col].map(bin_map)
    return(df)

df = bin_map(df)
binary_cols = ['default','housing','loan']
df[binary_cols].head()


print(df['job'].unique())
def job_map(df):
    job_mapping = {
        'unemployed':0, 'housemaid':0, 'retired':0,
        'student':1, 'self-employed':1,
        'unknown':2,
        'blue-collar':3, 'technician':3, 'services':3,
        'admin.':4, 'management':4, 'entrepreneur':4
    }
    
    df['job'] = df['job'].map(job_mapping)
    return df


def month_map(df):
    month_mapping = {
        'jan':1, 'feb':2, 'mar':3, 'apr':4,
        'may':5, 'jun':6, 'jul':7, 'aug':8,
        'sep':9, 'oct':10, 'nov':11, 'dec':12
    }
    days_before_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    df['month'] = df['month'].map(month_mapping)
    
    df['date'] = df['month'].apply(lambda x: days_before_month[x - 1]) + df['day'] 
    return df
df.head()



#feature engineering
def engineered_features(df):
    df['balance'] = np.log1p(np.maximum(df['balance'], 0))
    df['age_by_balance'] = np.log1p(np.maximum(df['balance'], 0))/df['age'] #ratio age balance (log is to keep the fraction from exploding with high balances)
    df['campaign_by_duration'] = df['duration']/(1 + df['campaign'])
    df['house_n_loan'] = (df['housing'] + df['loan']) * (1 - df['default']) 
    df['duration_cos'] = np.cos(2*np.pi + df['duration']/np.max(df['duration']))
    df['duration_sin'] = np.sin(2*np.pi + df['duration']/np.max(df['duration']))
    df['contacted_before'] = (df['pdays'] != -1).astype(int)

    return df


#One Hot Encoding the rest

def ohe_encode(df):    
    cat_left = [col for col in df.columns if df[col].dtype == 'object']
    num_cols = [col for col in df.columns if col not in cat_left]
    
    ohe = OneHotEncoder()
    
    cat_data = df[cat_left]
    transformed_cat_data = ohe.fit_transform(cat_data).toarray()
    
    transformed_cat_data = pd.DataFrame(transformed_cat_data, columns=ohe.get_feature_names_out(), index=df.index)
    
    df = pd.concat([df[num_cols], transformed_cat_data], axis=1)
    return df

df = ohe_encode(df)


df.head(10)


#data is ready, we check target value distribution
def check_distribution(df):
    df_y_copy = df.value_counts()
    df_y_copy.plot(kind='bar')

df_y = df['y']
check_distribution(df_y)


df_y = df['y']
df = df.drop(columns=['id','y'])

X_train,X_val, y_train, y_val = train_test_split(df, df_y, test_size=0.2,
                                                stratify=df_y, random_state=0)
sample_weights = compute_sample_weight('balanced', y_train)




#we test some basiline models
models = {
    'decision_tree': DecisionTreeClassifier(class_weight='balanced', random_state=0),
    'logistic_regression': LogisticRegression(class_weight='balanced'),
    'random_forest': RandomForestClassifier(class_weight='balanced', random_state=0),
    'XGBoost': XGBClassifier(random_state=0,
                            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])),
    'Gaussian': GaussianNB(),
    'neural_network': MLPClassifier(random_state=0)
}
#and evaluate with a function
results = {}
logs = []
def evaluate_models(models: dict, X_train, X_val, y_train, y_val):
    
    for name, model in models.items(): 
        try:
            print(f'fitting {name}...')
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            print('model fit')
            
            report = classification_report(y_true=y_val, y_pred=preds)
            print('metrics computed')
            print(report)

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, proba)
            else:
                auc = 'N/A'
            
            results[name] = {
                'report': report,
                'AUC': auc
            }

            print('-'*50)

        except Exception as e:
            logs.append(f'{name}: {str(e)}')
            print(f'Couldn\'t train {name}, check logs')
        
        



#evaluate_models(models)


#for name, result in results.items():
#    roc_auc = result['AUC']
#    print(f'ROC-AUC Score of {name} is {roc_auc}')


!nvidia-smi


print(f'nans: {df.isna().sum().sum()}')


xgb_model = XGBClassifier(random_state=0,
                          scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                          tree_method='hist',  
                          device='cuda')

def all_data_prep(df):
    df = bin_map(df)
    df = job_map(df)
    df = month_map(df)
    df = ohe_encode(df)
    df = engineered_features(df)
    
    return df


df = pd.read_csv(train_path)
df = all_data_prep(df)
y = df['y']
X = df.drop(labels=['id','y'], axis=1)

dtrain = xgb.DMatrix(X, label=y)

xgb_param = {
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': 0.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective': 'binary:logistic'
}

cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=999, nfold=5,
                 metrics='auc', early_stopping_rounds=50)



cvresult.sort_values(by='test-auc-mean', ascending=False).head()


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(cvresult.index, cvresult['train-auc-mean'], label='Train AUC')
plt.plot(cvresult.index, cvresult['test-auc-mean'], label='Validation AUC')
plt.axvline(x=237, color='red', linestyle='--', label=f'Best iteration: {262}')
plt.xlabel('Boosting Rounds')
plt.ylabel('AUC Score')
plt.legend()
plt.title('XGBoost Cross-Validation Results')
plt.show()


#new_config = {'n_estimators':237,
#            'tree_method': 'hist',
#            'device': 'cuda',
#            'eta': 0.3,
#            'subsample': 1,
#            'colsample_bytree': 1,
#            'objective': 'binary:logistic'
#            }
             
#xgb_model.set_params(**new_config)

#we start by complexity of the model to avoid overfitting from the get go
#param_ranges1 = {
#    'max_depth': [3, 4, 5, 6],
#    'min_child_weight': [1, 3, 5, 7]
#}

#gsearch1 = GridSearchCV(
#    estimator=xgb_model,
#    param_grid=param_ranges1, scoring='roc_auc', cv=5
#    ).fit(np.array(X),np.array(y))


#print(gsearch1.best_params_)
#print(gsearch1.best_score_)


#new_config.update(gsearch1.best_params_)
#xgb_model.set_params(**new_config)

#param_ranges2 = {
#    'reg_alpha': [0, 0.1, 1, 10],  # L1
#    'reg_lambda': [1, 5, 10, 20]   # L2
#}

#gsearch2 = GridSearchCV(
#    estimator=xgb_model,
#    param_grid=param_ranges2, scoring='roc_auc', cv=5
#    ).fit(np.array(X),np.array(y))


#print(gsearch2.best_params_)
#print(gsearch2.best_score_)


#new_config.update(gsearch2.best_params_)
#xgb_model.set_params(**new_config)
#print(new_config)

#param_ranges3 = {
#    'subsample': [0.6, 0.8, 1.0],
#    'colsample_bytree': [0.6, 0.8, 1.0]
#}

#gsearch3 = GridSearchCV(
#    estimator=xgb_model,
#    param_grid=param_ranges3, scoring='roc_auc', cv=5
#    ).fit(np.array(X),np.array(y))


#print(gsearch3.best_params_)
#print(gsearch3.best_score_)


#new_config.update(gsearch3.best_params_)
##xgb_model.set_params(**new_config)
#print(new_config)

#param_ranges4 = {
#    'eta': [0.01, 0.05, 0.1, 0.2],
#    'n_estimators': [300, 500, 800, 1200]
#}

#gsearch4 = GridSearchCV(
#    estimator=xgb_model,
#    param_grid=param_ranges4, scoring='roc_auc', cv=5
#    ).fit(np.array(X),np.array(y))


#print(gsearch4.best_params_)
#print(gsearch4.best_score_)


#new_config.update(gsearch4.best_params_)
#xgb_model.set_params(**new_config)
#print(new_config)

#param_ranges5 = {
#    'max_bin': [256, 512, 1024]  # larger = more precise splits, slower
#}
#
#gsearch5 = GridSearchCV(
#    estimator=xgb_model,
#    param_grid=param_ranges5, scoring='roc_auc', cv=5
#    ).fit(np.array(X),np.array(y))


#print(gsearch5.best_params_)
#print(gsearch5.best_score_)


new_config = {'n_estimators': 1200, 'tree_method': 'hist', 'device': 'cuda',
              'eta': 0.1, 'subsample': 1.0, 'colsample_bytree': 0.8,
              'objective': 'binary:logistic', 'max_depth': 5, 'min_child_weight': 5,
              'reg_alpha': 10, 'reg_lambda': 1, 'max_bin': 1024}


X_test = pd.read_csv(test_path)
X_train = pd.read_csv(train_path)
xgb_model = XGBClassifier()
xgb_model.set_params(**new_config)
print(new_config)




from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=0,
    n_jobs=-1
)

cat_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.3,
    random_state=0
)

neural_net = MLPClassifier(random_state=0)

final_model = StackingClassifier(estimators=[('xgb', xgb_model),('lgbm', lgbm_model),
                                             ('cat', cat_model),('neural_net', neural_net)],
                                stack_method='predict_proba',
                                n_jobs=-1
                                )



X_train_prepared = all_data_prep(X_train)
y_train = X_train_prepared['y']
X_train_prepared = X_train_prepared.drop(labels=['y','id'], axis=1)

X_test_prepared = all_data_prep(X_test)
X_id = np.array(X_test_prepared['id'])
X_test_prepared = X_test_prepared.drop(labels='id', axis=1)


final_model.fit(X_train_prepared, y_train)
preds = final_model.predict_proba(X_test_prepared)
y_pred = np.array([p for _,p in preds])

for p in y_pred:
    if p>0.8:
        p=1
    elif p<0.2:
        p=0


submission = pd.DataFrame({
    'id': X_id,
    'target': y_pred
})
submission.to_csv('submission.csv', index=False)




