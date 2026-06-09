import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import *
from sklearn.model_selection import *
from sklearn.metrics import *
import xgboost as xgb


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col=['id'])
df.head()


org01 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
org02 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
org03 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')

ORG = pd.concat([org01,org02,org03],axis=0).drop_duplicates()
ORG.shape


df.info()


df.describe().T #only the numeric columns are described here


ORG.describe().T


cats = df.select_dtypes(exclude='number').columns.tolist()
print(f'All Categorical features in original dataset:\n{cats}')


fig,ax = plt.subplots(3,int(len(cats)/3)+1,figsize=(6,6))
ax = ax.flatten()
# fig.title('Submission')
target = 'accident_risk'
for idx,feats in enumerate(cats):
    sns.boxplot(x=feats, y=target, data=df, ax=ax[idx],showmeans=True)
plt.tight_layout()
plt.show()


fig,ax = plt.subplots(3,int(len(cats)/3)+1,figsize=(6,6))
ax = ax.flatten()
# fig.title('Submission')
target = 'accident_risk'
for idx,feats in enumerate(cats):
    sns.boxplot(x=feats, y=target, data=ORG, ax=ax[idx],showmeans=True)
plt.tight_layout()
plt.show()


plt.hist((df['accident_risk']))
plt.show()


import scipy

def AddTargetEncoding(data,cats,target='accident_risk',mapping=None):
    df = data.copy()
    mapping_dict = {}
    for cat in cats:
        if mapping is None:
            Mapping = df.groupby(cat)[target].mean() 
            mapping_dict[cat] = Mapping
        else:
            Mapping = mapping[cat]
        df[f'_{cat}_TE'] = df[cat].map(Mapping)
    return df,mapping_dict

def CrossMatchCats(data,cross:list[tuple]):
    df = data.copy()
    for crosses in cross:
        #crosses is a tuple with any number of combinations
        n_feat = "x".join(crosses)
        df[n_feat] = df[list(crosses)].astype(str).agg('_'.join, axis=1)
    return df    

def GetTargetEncoding(working_data,original_data,cats,target='accident_risk',mapping=None):
    make_map = None
    if mapping is None:
        # if no mapping is given, make the mapping to return
        make_map = {}
        for cols in cats:
            make_map[cols] = original_data.groupby(cols)[target].mean()
    else:
        #if mapping is given use that 
        make_map = mapping

    df = working_data.copy()
    for cat in cats:
        df[f'_orig_{cat}_TE'] = df[cat].map(make_map[cat])
    return df,make_map
        
def NumFeats(data):
    df = data.copy()
    k = 1
    df['MaxLimitedAngularVelocity'] = df['speed_limit']**2/(df['curvature']+1e-3)
    df['TotalAreaPresent'] = k*(df['num_lanes']**2)
    df['SpeedPerLane'] = df['speed_limit'] / (df['num_lanes']) #new feature
    return df

def AddBinaryFeature(data):
    '''
    AnyReportedIncidents
    IsCurvedRoad
    IsHighSpeedRoad (only these as of now!)
    '''
    df = data.copy()
    df['AnyReportedIncidents'] = (df['num_reported_incidents'] > 0).astype(int)
    df['IsCurvedRoad'] = (df['curvature'] > 0).astype(int)
    df['IsHighSpeedRoad'] = (df['speed_limit'] > 60).astype(int) #changed from > -> â‰¥
    
    return df

def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f


def PreProcess(data,mapping_01=None,target='accident_risk'):
    df = data.copy()
    cats = df.select_dtypes(exclude='number').columns.tolist()
    df['rule_based_score'] = clip(f)(df)
    get_map = None
    if mapping_01 is not None:
        get_map = mapping_01
    else:
        get_map = {}
        for cat in cats:
            get_map[cat] = df.groupby(cat)[target].mean()
    for cat in cats:
        df[f'_{cat}_TE'] = df[cat].map(get_map[cat])
    
    df = NumFeats(df)
    df = CrossMatchCats(df,[
        ('lighting','weather'), #CHANGES
        ('road_type','road_signs_present','public_road'),
        ('lighting','time_of_day'),
        ('holiday','school_season'),
        ##baseline, past this new experiments are added 
        # ('weather','holiday','lighting')        
    ])
    
    ##changes features
    df["CurvBin"] = pd.qcut(df['curvature'],q=4,labels=['straight','slight','curvy','very_curvy'])
    df['LaneBin'] = pd.qcut(df['num_lanes'],q=3,labels=['one','two','three'])
    df['ReportBin'] = pd.qcut(df['num_reported_accidents'],q=3,labels=['low_reports','mid_reports','high_reports'])
    #till here
    
    bools = df.select_dtypes(include='bool').columns.tolist()
    for cols in bools:
        df[cols] = df[cols].astype(int)
    #don't make dummies for 'bool' data type columns
    cats = df.select_dtypes(exclude='number').columns.tolist()
    df = pd.get_dummies(df,cats)
    return df,get_map


# x,te_map_01,te_map_02 = PreProcess(df,org)
# x = x.drop(columns=['accident_risk'])
# y = df['accident_risk']


cats = df.select_dtypes(exclude='number').columns.tolist()
x,map_original = GetTargetEncoding(df,ORG,cats)
x.head()


print(f'Dataset Shape: {x.shape}')


n_splits = 5
splitter = KFold(n_splits=n_splits,shuffle=True, random_state=42)


np.random.seed(42)
hold_idx = np.random.randint(0,len(x),int(0.01*len(x)))
active_idx = [i for i in range(len(x)) if i not in hold_idx]

active = x.iloc[active_idx]
hold = x.iloc[hold_idx]


import time
scores = np.zeros(n_splits)
to_split = x

for idx,(train_idx,val_idx) in enumerate(splitter.split(to_split)):
    print(f'=> Fold: {idx+1}')
    train,val = x.iloc[train_idx],x.iloc[val_idx]
    IN = time.time()
    train,fold_map = PreProcess(train)
    val,_ = PreProcess(val,mapping_01=fold_map)

    trainx,trainy = train.drop(columns=['accident_risk']),train.accident_risk
    valx,valy = val.drop(columns=['accident_risk']),val.accident_risk
    # print(f'Shapes: trainx: {trainx.shape}, valx: {valx.shape}, trainy: {trainy.shape}, valy: {valy.shape}')
    # print(trainx.columns.tolist())
    # break
    m_reg = xgb.XGBRegressor(n_estimators=10000, #this is a standard practice is Xgboost training
                         subsample=0.8, #a good value to start with 
                         early_stopping_rounds=100,## this is choosen by a heuristic, we start with 100 and increase or decrease based on the result
                         device='gpu',
                         verbosity=0,
                         random_state=-9100
                        )

    m_reg.fit(
        trainx,trainy,
        eval_set=[(valx,valy)],
        verbose=0
    )
   
    predictions_reg = m_reg.predict(valx)
    scores[idx] = np.sqrt(mean_squared_error(valy,predictions_reg))
    print(f'{time.time()-IN} seconds')

# print(scores.mean())
print(f'---BaseLine Model Assesment---')
print(f'Score: {scores.mean()}Â±{scores.std()}')


scores


# #pre computing DMatrices for the xgb_models
# dmatrices = []
# for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(active)):
#     train, val = active.iloc[train_idx], active.iloc[val_idx]
#     train_processed, fold_map = PreProcess(train)
#     val_processed, _ = PreProcess(val, mapping_01=fold_map)

#     train_x, train_y = train_processed.drop(columns=['accident_risk']), train_processed.accident_risk
#     val_x, val_y = val_processed.drop(columns=['accident_risk']), val_processed.accident_risk

#     dtrain = xgb.DMatrix(train_x, label=train_y) # <-- CHANGED
#     dval = xgb.DMatrix(val_x, label=val_y) 
#     dmatrices.append((dtrain,dval))


import optuna
import time
from tqdm.notebook import trange
import gc 
import warnings
warnings.filterwarnings('ignore')

def objective_xgb(trial,dmatrices,random_state=42):
    IN = time.time()
    params = {
        'eval_metric':'rmse',
        'learning_rate':trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'max_depth':trial.suggest_int('max_depth', 3, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': random_state, #CHANGED
        'objective': 'reg:squarederror',
        'booster': 'gbtree', 
        'device': 'gpu',          
    }
    scores = []
    trees = []
    for (dtrain,dval) in dmatrices:
        # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-rmse')
        model = xgb.train(                                    
            params=params,
            dtrain=dtrain,
            num_boost_round=10000,                            
            early_stopping_rounds=150,
            evals=[(dval, 'validation')],
            verbose_eval=False,
            # callbacks=[pruning_callback]
        )
        val_y = dval.get_label()
        trees.append(model.best_iteration)
        preds = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(val_y, preds))
        scores.append(rmse)
    OUT = time.time()
    print(f'Exit in: {OUT-IN} sec')
    print(f'Avg Trees: {np.mean(trees)}')
    return np.mean(scores)


xgb_params_dict = []

n_splits = 5
for i in trange(10): # a maximum of 10 hours
    state=np.random.randint(0,100)
    print(f'Take-{i+1}')
    splitter = KFold(n_splits=n_splits,shuffle=True,random_state=state)
    dmatrices = []
    IN = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(active)):
        train, val = active.iloc[train_idx], active.iloc[val_idx]
        train_processed, fold_map = PreProcess(train)
        val_processed, _ = PreProcess(val, mapping_01=fold_map)
    
        train_x, train_y = train_processed.drop(columns=['accident_risk']), train_processed.accident_risk
        val_x, val_y = val_processed.drop(columns=['accident_risk']), val_processed.accident_risk
    
        dtrain = xgb.DMatrix(train_x, label=train_y) # <-- CHANGED
        dval = xgb.DMatrix(val_x, label=val_y) 
        dmatrices.append((dtrain,dval))
    print(f'=> Data Ready - {time.time()-IN} seconds')
    IN = time.time()
    study = optuna.create_study(study_name=f'Tune_XGB_{i+1}',direction='minimize',pruner=optuna.pruners.MedianPruner(n_warmup_steps=15))
    study.optimize(lambda trial:objective_xgb(trial,dmatrices=dmatrices,random_state=state),n_trials=50,timeout=60*60,show_progress_bar=True)
    xgb_params_dict.append({
        'Params':study.best_params,
        'State':state,
        'Score':study.best_value
    })
    print(f'=> Study DONE - {(time.time()-IN)/60} minutes')
    del dmatrices
    del splitter 
    del train_idx,val_idx
    del train,val
    del train_processed,val_processed,fold_map
    del train_x,train_y,val_x,val_y
    del dtrain,dval
    gc.collect()


##storing the params in a json file
import json
FILE = 'config.json'

with open(FILE,'a') as json_file:
    json.dump(xgb_params_dict, json_file)


active_processed,active_map = PreProcess(active)
hold_processed,_ = PreProcess(hold,mapping_01=active_map)
activex,activey = active_processed.drop(columns=['accident_risk']),active_processed.accident_risk
holdx,holdy = hold_processed.drop(columns=['accident_risk']),hold_processed.accident_risk
X,total_map = PreProcess(x)

xgb_models = []

for idx,things in enumerate(xgb_params_dict):
    # print(f'Model-{idx+1}')
    params = things['Params']
    random_state = things['State']
    model = xgb.XGBRegressor(
            n_estimators = 10000,
            early_stopping_rounds =150,
            device='gpu',
            random_state=random_state,
            **params,
        )
    model.fit(activex,activey,eval_set=[(holdx,holdy)],verbose=0)
    n_trees = model.best_iteration
    model = xgb.XGBRegressor(
            n_estimators = n_trees,
            device='gpu',
            random_state=random_state,
            **params,
        )
    model.fit(X.drop(columns=['accident_risk']),X.accident_risk)
    xgb_models.append(model)
    print(f'=> Model-{idx+1}\nN-Trees: {n_trees}\nBest Score: {things["Score"]}')
print(f'=> Trained and Stored {len(xgb_models)} XGBoost Models..')


xgb_models


num_models = len(xgb_params_dict)
oof_predictions = np.zeros((len(x), num_models + 1))
splitter = KFold(n_splits=5)
for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x)):
    print(f'=> Fold: {fold_idx + 1}')
    train, val = x.iloc[train_idx], x.iloc[val_idx]
    train_processed, fold_map = PreProcess(train)
    val_processed, _ = PreProcess(val, mapping_01=fold_map)
    
    train_x, train_y = train_processed.drop(columns=['accident_risk']), train_processed.accident_risk
    val_x, val_y = val_processed.drop(columns=['accident_risk']), val_processed.accident_risk
    
    oof_predictions[val_idx, num_models] = val_y.values

    
    for model_idx, config in enumerate(xgb_params_dict):
        print(f'  Training model {model_idx + 1}/{num_models}...')
        IN = time.time()
        
        params = config['Params']
        random_state = config['State']
        

        model = xgb.XGBRegressor(
            n_estimators=10000,
            early_stopping_rounds=150,
            device='gpu',
            random_state=random_state,
            **params
        )
        

        model.fit(
            train_x, train_y,
            eval_set=[(val_x, val_y)],
            verbose=0
        )
        

        predictions = model.predict(val_x)
        

        oof_predictions[val_idx, model_idx] = predictions
        
        print(f'    Done in {time.time()-IN:.2f} seconds. Best iteration: {model.best_iteration}')

print('\n--- OOF Prediction Generation Complete ---')

# 4. (Optional but Recommended) Convert the result to a clean DataFrame
oof_column_names = [f'model_{i+1}_preds' for i in range(num_models)] + ['target']
oof_df = pd.DataFrame(oof_predictions, columns=oof_column_names)

print("OOF DataFrame Head:")
print(oof_df.head())


oof_df


from sklearn.linear_model import LinearRegression
IN = time.time()
meta_model = LinearRegression().fit(oof_df.drop(columns=['target']),oof_df.target)
print(f'Meta Model Trained...... - {time.time()-IN}')


oof_df.to_csv('oof.csv',index=False)





testing = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
testing.head()


Ids = testing['id']
cats = df.select_dtypes(exclude='number').columns.tolist()
testing,_ = GetTargetEncoding(testing,ORG,cats,mapping=map_original)

testing,_ = PreProcess(testing,mapping_01=total_map)

# preds = model.predict(testing.drop(columns=['id']))
# preds_02 = model.predict(testing.drop(columns=['id']))
# preds_03 = model.predict(testing.drop(columns=['id']))
# preds_04 = model.predict(testing.drop(columns=['id']))
# preds_05 = model.predict(testing.drop(columns=['id']))
# preds_07 = model.predict(testing.drop(columns=['id']))
# preds_08 = model.predict(testing.drop(columns=['id']))
# preds_09 = model.predict(testing.drop(columns=['id']))
# preds_08 = pd.read_csv('/kaggle/input/submission05/submission-5.csv').accident_risk.values


base_predictions


base_predictions = np.zeros((len(testing),10))
for idx,models in  enumerate(xgb_models):
    base_predictions[:,idx] = models.predict(testing.drop(columns=['id']))

PREDICTIONS =meta_model.predict(base_predictions)
PREDICTIONS


PREDICTIONS = meta_model.predict(base_predictions)


PREDICTIONS


PREDICTIONS


print(f'Current Predictions: {preds_08}')
# collective_predictions.append(preds_09)


# collective_predictions = [preds,preds_02,preds_03]
# collective_predictions.append(preds_04)


last_best_submission = pd.read_csv('/kaggle/input/last-submission/submission-3.csv').accident_risk.values
# n_submissions = 5
# collective_predictions  = [last_best_submission['accident_risk'].values]


PREDICTIONS = (0.9999*preds_08 + 0.0001*preds_09)


PREDICTIONS


# PREDICTIONS = np.zeros(len(preds))
# for i in range(len(collective_predictions)):
#     PREDICTIONS += (1/len(collective_predictions))*collective_predictions[i]

# PREDICTIONS = (n_submissions*collective_predictions[0]+preds_07+preds_08)/(n_submissions+2)

# PREDICTIONS = (preds_07+preds_08)/2


print(f'Final Prediction: {PREDICTIONS}')


submission = pd.DataFrame(
    {'id':Ids,
    'accident_risk':PREDICTIONS.reshape(-1)},
dtype = np.float64
)
submission['id'] = submission['id'].astype(int)

submission.to_csv('submission.csv',index=False)


print(f'Submission File:')
print(submission.head())




