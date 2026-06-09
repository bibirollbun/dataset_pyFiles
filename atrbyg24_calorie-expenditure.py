import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import shap
import optuna

import warnings
warnings.simplefilter("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv',index_col = 'id')


train.head()


train.info()


train.describe()


train['Sex'].value_counts()


num_cols = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Calories']
for col in num_cols:
    plt.figure()
    sns.histplot(data = train[num_cols], x = col)
    plt.title(f"Histogram of {col}")
    plt.show()
    plt.clf()


for col in num_cols:
    plt.figure()
    sns.histplot(data = train, x = col,hue='Sex')
    plt.title(f"Histogram of {col}")
    plt.show()
    plt.clf()


corr = train[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(data=corr,annot=True)


y_train = np.log1p(train['Calories'])
X_train = train.drop('Calories',axis=1)

y_f_train = np.log1p(train[train['Sex']=='female']['Calories'])
y_m_train = np.log1p(train[train['Sex']=='male']['Calories'])
X_f_train = X_train[X_train['Sex'] == 'female']
X_m_train = X_train[X_train['Sex']=='male']


def bmi_feature(X):
    return X[:,[0]] / ((X[:,[1]] / 100) ** 2)

def bmi_name(function_transformer,feature_names_in):
    return ['BMI']

def bmi_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(bmi_feature,feature_names_out=bmi_name),
        StandardScaler()
    )

def interaction_feature(X):
    return X[:,[0]] * X[:,[1]]

def interaction_name(function_transformer,feature_names_in):
    return ['interaction']

def interaction_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(interaction_feature,feature_names_out=interaction_name),
        StandardScaler()
    )

def square_feature(X):
    return X[:,[0]] ** 2

def square_name(function_transformer,feature_names_in):
    return ['square']

def square_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(square_feature,feature_names_out=square_name),
        StandardScaler()
    )


cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

preprocessor = ColumnTransformer(
    transformers=[
        ('bmi', bmi_pipeline(), ['Weight','Height']),
        ('age_height',interaction_pipeline(),['Age','Height']),
        ('age_weight',interaction_pipeline(),['Age','Weight']),
        ('age_duration',interaction_pipeline(),['Age','Duration']),
        ('age_heart_rate',interaction_pipeline(),['Age','Heart_Rate']),
        ('age_body_temp',interaction_pipeline(),['Age','Body_Temp']),
        ('height_weight',interaction_pipeline(),['Height','Weight']),
        ('height_duration',interaction_pipeline(),['Height','Duration']),
        ('height_heart_rate',interaction_pipeline(),['Height','Heart_Rate']),
        ('height_body_temp',interaction_pipeline(),['Height','Body_Temp']),
        ('weight_duration',interaction_pipeline(),['Weight','Duration']),
        ('weight_heart_rate',interaction_pipeline(),['Weight','Heart_Rate']),
        ('weight_body_temp',interaction_pipeline(),['Weight','Body_Temp']),
        ('duration_heart_rate',interaction_pipeline(),['Duration','Heart_Rate']),
        ('duration_body_temp',interaction_pipeline(),['Duration','Body_Temp']),
        ('heart_rate_body_temp',interaction_pipeline(),['Heart_Rate','Body_Temp']),
        ('duration_square',square_pipeline(),['Duration']),
        ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
        ('cat', cat_pipeline, make_column_selector(dtype_include=object)),
    ],remainder= 'passthrough')

preprocessor2 = ColumnTransformer(
    transformers=[
        ('bmi', bmi_pipeline(), ['Weight','Height']),
        ('age_height',interaction_pipeline(),['Age','Height']),
        ('age_weight',interaction_pipeline(),['Age','Weight']),
        ('age_duration',interaction_pipeline(),['Age','Duration']),
        ('age_heart_rate',interaction_pipeline(),['Age','Heart_Rate']),
        ('age_body_temp',interaction_pipeline(),['Age','Body_Temp']),
        ('height_weight',interaction_pipeline(),['Height','Weight']),
        ('height_duration',interaction_pipeline(),['Height','Duration']),
        ('height_heart_rate',interaction_pipeline(),['Height','Heart_Rate']),
        ('height_body_temp',interaction_pipeline(),['Height','Body_Temp']),
        ('weight_duration',interaction_pipeline(),['Weight','Duration']),
        ('weight_heart_rate',interaction_pipeline(),['Weight','Heart_Rate']),
        ('weight_body_temp',interaction_pipeline(),['Weight','Body_Temp']),
        ('duration_heart_rate',interaction_pipeline(),['Duration','Heart_Rate']),
        ('duration_body_temp',interaction_pipeline(),['Duration','Body_Temp']),
        ('heart_rate_body_temp',interaction_pipeline(),['Heart_Rate','Body_Temp']),
        ('duration_square',square_pipeline(),['Duration']),
        ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
    ],remainder= 'drop')


xgb_reg = Pipeline([
    ('preprocess',preprocessor),
    ('xgb',XGBRegressor(random_state=42))
])
xgb_f_reg = Pipeline([
    ('preprocess',preprocessor2),
    ('xgb',XGBRegressor(random_state=42))
])
xgb_m_reg = Pipeline([
    ('preprocess',preprocessor2),
    ('xgb',XGBRegressor(random_state=42))
])
xgb_reg.fit(X_train,y_train)


xgb_rmses = -cross_val_score(xgb_reg,X_train,y_train,scoring="neg_root_mean_squared_error",cv=5)
pd.Series(xgb_rmses).describe()


xgb_f_reg.fit(X_f_train,y_f_train)
xgb_m_reg.fit(X_m_train,y_m_train)


xgb_f_rmses = -cross_val_score(xgb_f_reg,X_f_train,y_f_train,scoring="neg_root_mean_squared_error",cv=5)
pd.Series(xgb_rmses).describe()


xgb_m_rmses = -cross_val_score(xgb_m_reg,X_m_train,y_m_train,scoring="neg_root_mean_squared_error",cv=5)
pd.Series(xgb_rmses).describe()


xgb_model = xgb_reg.named_steps['xgb']
xgb_f_model = xgb_f_reg.named_steps['xgb']
xgb_m_model = xgb_m_reg.named_steps['xgb']


xgb_importances = xgb_model.feature_importances_
sorted(zip(xgb_importances, xgb_reg.named_steps['preprocess'].get_feature_names_out()),reverse=True)


xgb_f_importances = xgb_f_model.feature_importances_
sorted(zip(xgb_importances, xgb_f_reg.named_steps['preprocess'].get_feature_names_out()),reverse=True)


xgb_m_importances = xgb_m_model.feature_importances_
sorted(zip(xgb_importances, xgb_m_reg.named_steps['preprocess'].get_feature_names_out()),reverse=True)


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 450, 650)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.02, log=True) 
    max_depth = trial.suggest_int('max_depth', 6, 9)
    subsample = trial.suggest_float('subsample', 0.7, 0.8) 
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.9, 1.0)

    xgb = XGBRegressor(n_estimators=n_estimators,
                       learning_rate=learning_rate,
                       max_depth=max_depth,
                       subsample=subsample,
                       colsample_bytree=colsample_bytree,
                       random_state=42)

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('xgb', xgb)
    ])

    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=5) 

print("Best trial:", study.best_trial)
best_xgb_reg_optuna = Pipeline([
    ('preprocess', preprocessor),
    ('xgb', XGBRegressor(**study.best_params, random_state=42))
])
best_xgb_reg_optuna.fit(X_train, y_train)


sample = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv',index_col='id')


sample['Calories'] = np.clip(np.expm1(best_xgb_reg_optuna.predict(test)),1,314)


sample.to_csv('submission.csv')




