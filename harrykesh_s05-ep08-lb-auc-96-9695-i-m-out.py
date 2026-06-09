import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import RandomOverSampler
%matplotlib inline
root = r'/kaggle/input/playground-series-s5e8/train.csv'
df = pd.read_csv(root)


df.head()


df.shape


df.info()


df.isnull().sum().sum() ## YAAAAAA!!!! 


df.y.value_counts(normalize=True) 
# 13% positives and 87% negatives 


df.describe()


mapping = {'jan':1,'feb':2,'mar':3,'apr':4,
                'may':5,'jun':6,'jul':7,'aug':8,'sep':9,
                'oct':10,'nov':11,'dec':12}
df['month_map'] = df['month'].map(mapping)


df.month_map


xtrain,xtest,ytrain,ytest = train_test_split(df.drop(columns=['id','y']),df.y,test_size=0.00001,stratify=df.y,random_state=1337)
combine_train = pd.concat([xtrain,ytrain],axis=1)
combine_test = pd.concat([xtest,ytest],axis=1)


plt.figure(figsize=(12,20))
for idx,columns in enumerate(xtrain.select_dtypes(include='number')):
    plt.subplot(10,5,idx+1)
    plt.hist(xtrain[columns])
    plt.xlabel(columns)
    plt.grid()
plt.tight_layout()
## age, balance(contains negatives as well), duration, campaign, pdaysf


plt.figure(figsize=(25,25))
for idx,feat in enumerate(['age','balance','day','duration','campaign','pdays','previous']):
    plt.subplot(5,3,idx+1)
    sns.boxplot(x=combine_train.y,y=f'{feat}',data=combine_train)
plt.tight_layout()


def feature_engineering(combine_train,train_data):
    combine_train = combine_train.copy()
    combine_train.columns = combine_train.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True)##this was included to remove any JSON characters which can't be interpreted by LGBM
    combine_train['is_senior'] = (combine_train['age'] >= 60).astype(int)
    combine_train['is_young_adult'] = combine_train['age'].between(18, 30).astype(int)
    combine_train['age_decade'] = (combine_train['age'] // 10) * 10
    combine_train['age_zscore'] = (combine_train['age'] - combine_train['age'].mean()) / combine_train['age'].std()

    job_freq = train_data['job'].value_counts(normalize=True)
    combine_train['job_freq'] = combine_train['job'].map(job_freq).fillna(0)
    combine_train['job_is_high_profile'] = combine_train['job'].isin(['management', 'admin.', 'technician']).astype(int)
    combine_train['is_self_employed'] =combine_train['job'].isin(['self-employed', 'entrepreneur']).astype(int)
    
    combine_train['age_bin'] = pd.cut(combine_train['age'], bins=range(15, 100, 5), labels=False)
    combine_train['HouseAndLoan'] =((combine_train['housing'] == 'yes') & (combine_train['loan'] == 'yes')).astype(int)
    combine_train['JobxEdu'] = combine_train['job'] + "_" + combine_train['education']
    combine_train['MaritalxHouseAndLoan'] = combine_train['marital'] + "_" + combine_train['HouseAndLoan'].map({0:"no",1:"yes"})
    combine_train['ContactxPoutcome'] = combine_train['contact'] + "_" + combine_train['poutcome']
    combine_train['JobxDef'] = combine_train['job'] + "_" + combine_train['default']
    combine_train['DefxLoanAndHousing'] = combine_train['default'] + "_" + combine_train['HouseAndLoan'].map({0:"no",1:"yes"})
    combine_train['month_sin'] = np.sin(2*np.pi*df['month_map']/12)
    combine_train['month_cos'] = np.cos(2*np.pi*df['month_map']/12)
    combine_train['WasContactedLastTime'] = ((combine_train['previous'] > 0) | (combine_train['pdays'] != -1)).astype(int)
    combine_train['totalContacts'] = combine_train['previous'] + combine_train['campaign']
    combine_train['addthemup'] = combine_train['previous'] + combine_train['pdays'] + combine_train['campaign']
    combine_train['DurxCamp1'] = combine_train['duration']/(combine_train['campaign']+1)
    combine_train['Durxmonth_sin'] = combine_train['duration']*combine_train['month_sin']
    combine_train['Durxmonth_cos'] = combine_train['duration']*combine_train['month_cos']

    season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}  # winter=0,spring=1,summer=2,fall=3
    df['season'] = df['month_map'].map(season_map).fillna(-1).astype(int)
    df['is_month_end'] = (df['day'] > 25).astype(int)
    df['day_of_week_estimate'] = df['day'] % 7
    df['is_q2'] = df['month'].isin(['apr','may','jun']).astype(int)
    #target encoding 
    age_duration_map = combine_train.groupby(['age'])['duration'].mean()
    combine_train['AgeDurMean'] = combine_train['age'].map(age_duration_map)

    camp_day_map = combine_train.groupby(['day'])['campaign'].mean()
    combine_train['CampDayMean'] = combine_train['day'].map(camp_day_map)

    day_dur_map = combine_train.groupby(['day'])['duration'].mean()
    combine_train['DayDur'] = combine_train['day'].map(day_dur_map)
    
    target_encoded_features = ['JobxEdu','MaritalxHouseAndLoan','ContactxPoutcome','JobxDef']
    mean_y = train_data['y'].mean()
    for feats in target_encoded_features:
        if feats in train_data.columns:
            mapping_target = train_data.groupby(feats)['y'].mean()
            combine_train[f"{feats}_mean_target"] = combine_train[feats].map(mapping_target).fillna(mean_y)
    combine_train['duration_bin'] = pd.cut(combine_train['duration'],[0, 60, 120, 300, 600, 1200, 2400, 3600],labels=[f'db{i}' for i in range(0,7)],include_lowest=True)
    return combine_train


feature_engineering(xtrain,combine_train)
feature_engineering(xtest,combine_train)

cats = xtrain.select_dtypes(exclude='number').columns.tolist()
nums = xtrain.select_dtypes(include='number').columns.tolist()

xtrain = pd.get_dummies(xtrain,columns=cats)


nums


import xgboost
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
scaler = StandardScaler()


ytrain.value_counts()


lgbm = LGBMClassifier(verbose=-1,class_weight = {0:1,1:7.288},random_state=42)
xgb1 = xgboost.XGBClassifier(scale_pos_weight = 7.288,random_state=42)
xgb2 = xgboost.XGBClassifier(scale_pos_weight = 7.288,random_state=1337)
logistic = LogisticRegression(solver='newton-cholesky',max_iter=1000,class_weight = {0:1,1:7.288}, random_state=42 )
models = [logistic,xgb1,xgb2]
names  =['logi-reg','xgb1','xgb2']
## I changes it here so that the can be reused 
## one may change to whatever he/she wishes 


log = FunctionTransformer(np.log1p)
transformer = ColumnTransformer(
    [("LogTransform", log, ['age','duration']),
    ("standard",scaler,nums)],remainder='passthrough'
)


##base line scores for the model choice
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for idx, model in enumerate(models):
    print(names[idx])
    if names[idx]!='logi-reg':
        print(cross_val_score(model,xtrain,ytrain,cv=cv,scoring='roc_auc',n_jobs=-1))
    else: 
        training = transformer.fit_transform(xtrain)
        print(cross_val_score(model,training,ytrain,cv=cv,scoring='roc_auc',n_jobs=-1))

## hyper parameter tuning *MAY* improve the baseline scores 


#necessary imports
import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from optuna.pruners import MedianPruner ##pruner was used to decrease the time required per study 


def objective_lgbm(trial):
    params = {
       "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 250),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "max_bin": trial.suggest_int("max_bin", 127, 255),
        "random_state":42 
        }
    
    model = LGBMClassifier(class_weight={0:1,1:7.288},**params,verbose=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, xtrain, ytrain, cv=cv, scoring="roc_auc", n_jobs=-1)
    return np.mean(scores)

lgbm_study = optuna.create_study(direction="maximize",study_name='lgbm_opt',pruner=MedianPruner(n_warmup_steps=1))
lgbm_study.optimize(objective_lgbm, n_trials=25,show_progress_bar=True)



##the best result should be over roughly over 0.9646


lgbm_params = lgbm_study.best_params


xtrain.shape


def objective_xgb(trial):
    params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
    "n_estimators": trial.suggest_int("n_estimators", 100, 250),
    "max_depth": trial.suggest_int("max_depth", 3, 12),   # no -1 in XGB
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    "gamma": trial.suggest_float("gamma", 0.0, 0.5),      # like min_split_gain
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    "max_bin": trial.suggest_int("max_bin", 127, 255),     # used if tree_method='hist'
    "scale_pos_weight": 7.288,  # your class weight ratio
    "tree_method": "hist",      # fastest for large data
    "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
    "random_state":42
    }

    model = xgboost.XGBClassifier(**params,device='cuda',verbosity=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, xtrain, ytrain, cv=cv, scoring="roc_auc", n_jobs=-1)
    return np.mean(scores)

xgb_study = optuna.create_study(direction="maximize",study_name='lgbm_opt',pruner=MedianPruner(n_warmup_steps=1))
xgb_study.optimize(objective_xgb, n_trials=50,show_progress_bar=True)


xgb_params1 = xgb_study.best_params
# 0.9674310320209359


def objective_xgb2(trial):
    params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.8, log=True),
    "n_estimators": trial.suggest_int("n_estimators", 150, 500),
    "max_depth": trial.suggest_int("max_depth", 2, 20),   # no -1 in XGB
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    "gamma": trial.suggest_float("gamma", 0.0, 1.5),      # like min_split_gain
    "subsample": trial.suggest_float("subsample", 0.2, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    "max_bin": trial.suggest_int("max_bin", 127, 300),     # used if tree_method='hist'
    "scale_pos_weight": 7.288,  # your class weight ratio
    "tree_method": "hist",      # fastest for large data
    "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
    "random_state":13379
    }

    model = xgboost.XGBClassifier(**params,device='cuda',verbosity=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, xtrain, ytrain, cv=cv, scoring="roc_auc", n_jobs=-1)
    return np.mean(scores)

xgb_study2 = optuna.create_study(direction="maximize",study_name='xgb2_opt',pruner=MedianPruner(n_warmup_steps=1))
xgb_study2.optimize(objective_xgb2, n_trials=50,show_progress_bar=True)


xgb_params2 = xgb_study2.best_params


def objective_logi_reg(trial):
    param = {
        "penalty":'l2',
        "tol":trial.suggest_float("tol",1e-6,1e-4),
        "C":trial.suggest_float("C",1e-3,5),
        # "class_weight":trial.suggest_categorical("balanced"),
        "max_iter":trial.suggest_int("max_iter",100,1500),
        "multi_class":"ovr"
    }
    model = LogisticRegression(**param,class_weight={0:1,1:7.288},random_state=42,n_jobs=-1)
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    xtrain_transformed = transformer.fit_transform(xtrain)
    scores = cross_val_score(model,xtrain_transformed,ytrain,cv=cv,scoring='roc_auc',n_jobs=-1)
    return np.mean(scores)
    
logi_study = optuna.create_study(direction="maximize",study_name='logi-opt',pruner=MedianPruner(n_warmup_steps=1))
logi_study.optimize(objective_logi_reg, n_trials=20,show_progress_bar=True)


logi_params = logi_study.best_params


model1 = xgboost.XGBClassifier(**xgb_params1)
model2 = xgboost.XGBClassifier(**xgb_params2)
model3 = LogisticRegression(**logi_params,class_weight={0:1,1:7.288},random_state=42)
names = ['XGBoost1','XGBoost2','Logistic Regression']
for idx,models in enumerate([model1,model2,model3]):
    print(f'Training {names[idx]}')
    if names[idx]!= 'Logistic Regression':
        models.fit(xtrain,ytrain)
    else:
        xtrain_transformed = transformer.fit_transform(xtrain)
        models.fit(xtrain_transformed,ytrain)
    print(f'{names[idx]} trained')


Model1 = xgboost.XGBClassifier(**xgb_params1)
# Model2 = LGBMClassifier(class_weight={0:1,1:7.288},**lgbm_params,verbose=-1)
Model2 = xgboost.XGBClassifier(**xgb_params2)
Model3 = LogisticRegression(**logi_params,class_weight={0:1,1:7.288},random_state=42)
oof_train_dataset = np.zeros((xtrain.shape[0],4))
names = ['XGBOOST1','XGBOOST2','LogisticRegression']
cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
for idx,models in enumerate([Model1,Model2,Model3]):
    print(f'---{names[idx]}---')
    num_fold = 0
    for train_idx, valid_idx in cv.split(xtrain, ytrain):
        num_fold+=1
        X_train = xtrain.iloc[train_idx]
        X_valid = xtrain.iloc[valid_idx]
        y_train = ytrain.iloc[train_idx]
        y_valid = ytrain.iloc[valid_idx]
        if names[idx]!='LogisticRegression':
            models.fit(X_train,y_train)
            preds = models.predict_proba(X_valid)[:,1]
        else:
            x_train_trans = transformer.fit_transform(X_train)
            x_valid_trans = transformer.fit_transform(X_valid)
            models.fit(x_train_trans,y_train)
            preds = models.predict_proba(x_valid_trans)[:,1]
        oof_train_dataset[valid_idx,idx] = preds
        oof_train_dataset[valid_idx,-1] = y_valid
        print(f'Fold => {num_fold} CV => {roc_auc_score(y_valid,preds)}')
    print('-'*25)

meta_data = pd.DataFrame(oof_train_dataset,columns=['xgb1','xgb2','logi','y'])
for idx, features in enumerate(meta_data.columns):
    if(features!='y'):
        meta_data[f'{features}_logits'] = np.log(meta_data[features]/(1-meta_data[features]))
meta_data.head(10)


X,y = meta_data.drop(columns=['y']),meta_data.y
# scale = StandardScaler()
# # X = scale.fit_transform(X)


from sklearn.linear_model import LinearRegression
meta_model1 = LogisticRegression(class_weight='balanced',random_state=23)
# meta_model2 = LinearRegression()
for model in [meta_model1]:
    print(model)
    model.fit(X,y)
    pred_probab = model.predict_proba(X)[:,1]
    print(roc_auc_score(y,pred_probab))
    print(f'{model} trained')
    # print('--------------')


import optuna  
from optuna.pruners import MedianPruner

def objective_meta(trial):
    params = {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "max_iter": 1000,
        "random_state":23
    }
    model = LogisticRegression(**params)
    # print(f'eval started...')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    # print(f'eval ended...')
    return (np.mean(scores))

# Create and run Optuna study
meta_study = optuna.create_study(direction="maximize",study_name='MetaModelOptimization',pruner=MedianPruner(n_warmup_steps=1))
meta_study.optimize(objective_meta, n_trials=12,show_progress_bar=True)


meta_params = meta_study.best_params


meta_model1 = LogisticRegression(**meta_params)
for model in [meta_model1]:
    print(model)
    model.fit(X,y)
    pred_probab = model.predict_proba(X)[:,1]
    print(roc_auc_score(y,pred_probab))
    print(f'{model} trained')
    print('--------------')


testing = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
testing.head()


ids = testing.id
testing['month_map'] = testing['month'].map(mapping)
testing.drop(columns=['id'],inplace=True)


testing = feature_engineering(testing,combine_train)


testing.shape


testing = pd.get_dummies(testing,columns=cats)
testing = testing.reindex(columns=xtrain.columns,fill_value=0)


oof_predictions = np.zeros((testing.shape[0],3))
names = ['xgb1','xgb2','logi']
# cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=1337)
for idx,models in enumerate([model1,model2,model3]):
    print(f'{names[idx]}')
    if names[idx]!='logi':
        oof_predictions[:,idx]=models.predict_proba(testing.to_numpy())[:,1]
    else:
        test = transformer.fit_transform(testing)
        oof_predictions[:,idx]=models.predict_proba(test)[:,1]


oof = pd.DataFrame(oof_predictions,columns=['xgb1','xgb2','logi'])
for idx,features in enumerate(oof.columns):
    oof[f'{features}_logits'] = np.log(oof[features]/(1-oof[features]))


oof = oof.reindex(columns=X.columns)


oof


# preds1=model1.predict_proba(oof)[:,1]
preds2=meta_model1.predict_proba(oof)[:,1]


# preds = (preds1+preds2)/2
# preds = preds1
preds = preds2


sub = pd.DataFrame({
    'id':ids,
    'y':preds
})


preds


sub.to_csv('submit-today.csv',index=False)




