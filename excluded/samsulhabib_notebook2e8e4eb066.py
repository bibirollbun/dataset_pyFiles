import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


### checking if all the features in train.csv are also present in test.csv

features=train.drop(columns='diagnosed_diabetes').columns


test.columns==features


### yes! all the features are common in both the datasets.


### information of the dataset

train.info()


### no of missing values in the train and test dataset

print(train.isna().sum().sum())
print(test.isna().sum().sum())


### train and test both the datasets are completely clean with no missing values.


### train dataset has some columns with object data type

for col in train.columns:
    if train[col].dtype=='object':
        print(col)


### let's check which columns in test data are also of object data type.

for col in test.columns:
    if test[col].dtype=='object':
        print(col)


### train and test data both have same colmumns with object data type. No further transformation needed.


numerical_data=train.select_dtypes(include=['number'])


numerical_data.drop(columns=['id'],inplace=True)


categorical_data=train.select_dtypes(include=['object'])


### correlation heatmap between the numerical features

cor_mat=numerical_data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(cor_mat,cmap='coolwarm',fmt='.2f')
plt.title('Correlation heatmap')
plt.show()


### correlation coefficient between the target and other numerical features

f,c=[],[]
for col in numerical_data.columns:
    f.append(col)
    c.append(numerical_data[col].corr(numerical_data['diagnosed_diabetes']))
frame=pd.DataFrame({'features':f,'correlation coefficient':c})
frame.sort_values('correlation coefficient',ascending=False).reset_index(drop=True)


### impact of different numerical features on diabetes. How to implement this?

import seaborn as sns

print("Bar plot of different features vs total diabetes case. This is not histogram of different features.")
for feature in set(numerical_data.columns)-{'diagnosed_diabetes'}:
    f=pd.DataFrame(numerical_data.groupby(feature)['diagnosed_diabetes'].sum()).reset_index().sort_values(feature)
    ax=f.plot.bar(y='diagnosed_diabetes', legend=False)
    
    # index positions of min and max
    min_idx = f[feature].idxmin()
    max_idx = f[feature].idxmax()

    ax.set_xticks([min_idx, max_idx])
    ax.set_xticklabels([f[feature].min(), f[feature].max()])
    
    #plt.title(f"Bar Plot of {feature} vs Total diabetes case")
    plt.xlabel(feature)
    plt.ylabel('total diagnosed diabetes')
    plt.show()


### How many unique values are there in each categorical column?

for col in categorical_data.columns:
    print(f"unique elements in {col} is ---> {categorical_data[col].unique()}")


### impact of different categorical features on diabetes.

import seaborn as sns

print("Bar plot of different categorical features vs total diabetes case. This is not histogram of different features.")
for feature in categorical_data.columns:
    f=pd.DataFrame(train.groupby(feature)['diagnosed_diabetes'].sum())
    ax=f.plot.bar(y='diagnosed_diabetes', legend=False)
    
    plt.xlabel(feature)
    plt.ylabel('total diagnosed diabetes')
    plt.show()


for col in set(numerical_data.columns)-{'diagnosed_diabetes'}:
    plt.hist(numerical_data[col],bins=100)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Value")
    plt.show()


for col in set(numerical_data.columns)-{'diagnosed_diabetes'}:
    print(f"skewness of {col} is {numerical_data[col].skew()}")
    print("\n")
    print("*"*50)
    print("\n")


for col in categorical_data.columns:
    counts=categorical_data[col].value_counts()
    counts.plot(kind='bar',color='skyblue')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


X=train.iloc[:,1:25]   # feature matrix
y=train['diagnosed_diabetes']     # target matrix


from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
import optuna


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



# 1st encode the categorical features
# feature selection using SelectKBest
# define models
# use optuna for hpt


from copy import deepcopy


X_copy=deepcopy(X)
y_copy=deepcopy(y)


cat_ord=['education_level', 'income_level']  # should be ordinal encoded
cat_ohe=['gender','ethnicity','smoking_status','employment_status'] # should be OneHot encoded

for col in cat_ord:
    X_copy[col]=OrdinalEncoder().fit_transform(np.array(X_copy[col]).reshape(-1,1))

"""
ordinal_encoding=Pipeline(
    steps=[
        ("ordinal_encode",OrdinalEncoder())
    ]
)
ohe_encoding=Pipeline(steps=[
    ("ohe_encode",OneHotEncoder(sparse_output=False))
])


encode=ColumnTransformer(
    transformers=[
        ("ordinal_encoding",ordinal_encoding,cat_ord),
        ("ohe_encoding",ohe_encoding,cat_ohe)
    ],
    remainder="passthrough"
)
"""


# One hot encoding the other categorical features

X_copy=pd.get_dummies(X_copy,columns=cat_ohe).astype(int)


X_copy.shape


X_copy.isna().sum().sum()  # sanity checking for any missing value


X_copy.select_dtypes(include=['object']).shape[1]   # all the features are numerical now


cols=X_copy.columns
for col in cols:
    s=0
    q1=np.quantile(X_copy[col],0.25)
    q3=np.quantile(X_copy[col],0.75)
    IQR=q3-q1
    for elem in X_copy[col]:
        if (elem > (q3+1.5*IQR)) or (elem<(q1-1.5*IQR)):
            s+=1
    print(f"Total outliers in {col} is ----->  {(s/X_copy[col].shape[0])*100:.2f}%")


X_copy['education_level'].unique()


X_copy['income_level'].unique()


# One hot and ordinal encoded features don't need to be checked for outliers.
# One hot encoded column has only 0 and 1, so no outliers.
# ordinal encoded columns 'education_level' and 'income_level'
# have integers 0 to 4. So, no outliers.


# other features like 'hypertension_history','family_history_diabetes' have only 0 and 1. So, no outliers.


selector=SelectKBest(score_func=f_classif,k='all')


selector.fit_transform(X_copy,y_copy)


feature_score=pd.DataFrame({
    'features':X_copy.columns,
    'score':selector.scores_
}).sort_values('score',ascending=False)


feature_score


X_copy_new=X_copy.loc[:,feature_score['features'][:10]]   # taking the 1st 10 highest scored features


X_copy_new


X_copy_new.isna().sum().sum()


# total number of data points in each class

y_copy.value_counts()


# above result shows the dataset is imbalanced, since class 1 dominates class 0 


from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score


from sklearn.model_selection import KFold


def objective1(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'n_jobs': -1,
        'objective': 'binary',
        'verbosity':-1
    }

    model = LGBMClassifier(**params)

    # 5 fold CV
    cv = KFold(n_splits=5)
    roc_auc = []

    for train_idx, val_idx in cv.split(X_copy_new):
        X_trainn, X_val = X_copy_new.iloc[train_idx], X_copy_new.iloc[val_idx]
        y_trainn, y_val = y_copy.iloc[train_idx], y_copy.iloc[val_idx]

        model.fit(
                X_trainn, y_trainn,
                eval_set=(X_val, y_val),
                eval_metric='auc'
            )

        y_pred = model.predict(X_val)
        roc_auc.append(roc_auc_score(y_val, y_pred))

    return np.mean(roc_auc)



optuna.logging.set_verbosity(optuna.logging.ERROR)


study1 = optuna.create_study(direction='maximize',study_name='lightgbm study')
study1.optimize(objective1, n_trials=30, show_progress_bar=True)


print("Best R2 Score:", study1.best_value)
print("Best Parameters:")
for key, value in study1.best_params.items():
   print(f"  {key}: {value}")


best_params=study1.best_params
best_params['verbose']=-1
best_params['n_jobs']=-1
best_params['objective']='binary'
model1 = LGBMClassifier(
        **best_params
    )


# 5 fold cross validation on entire data

cv = KFold(n_splits=5)

fold=1
for train_idx, val_idx in cv.split(X_copy_new):
    X_trainn, X_val = X_copy_new.iloc[train_idx], X_copy_new.iloc[val_idx]
    y_trainn, y_val = y_copy.iloc[train_idx], y_copy.iloc[val_idx]

    model1.fit(
                X_trainn, y_trainn,
                eval_set=(X_val, y_val),
                eval_metric='auc'
            )

    y_pred = model1.predict(X_val)
    print(f"roc_auc score on fold {fold} is {roc_auc_score(y_val, y_pred)}")
    fold+=1


# training the model on the entire data

model1.fit(X_copy_new,y_copy)


# list of final features

f=list(X_copy_new.columns)


final_test=test[f]   # taking the necessary features from test dataframe


prediction=pd.DataFrame(model1.predict_proba(final_test))


prediction


pred_df=pd.DataFrame({
    'id':test['id'],
    'diagnosed_diabetes':round(prediction[1],2)
})


pred_df.head()


pred_df.to_csv('submission.csv',index=False)

