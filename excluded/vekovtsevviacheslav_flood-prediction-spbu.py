
import kagglehub
kagglehub.login()

playground_series_s4e5_path = kagglehub.competition_download('playground-series-s4e5')

print('Data source import complete.')

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna

df_train=pd.read_csv("/kaggle/input/playground-series-s4e5/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s4e5/test.csv")
#privat_path = kagglehub.dataset_download('vekovtsev/extradata')
subb = pd.read_csv('/kaggle/input/extradata/ensemble.csv')
#

num_cols=['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
       'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
       'Siltation', 'AgriculturalPractices', 'Encroachments',
       'IneffectiveDisasterPreparedness', 'DrainageSystems',
       'CoastalVulnerability', 'Landslides', 'Watersheds',
       'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
       'InadequatePlanning', 'PoliticalFactors']

unique_vals = []
for df in [df_train, df_test]:
    for col in num_cols:
        unique_vals += list(df[col].unique())

unique_vals = list(set(unique_vals))
#
def getFeats(df):
    
    scaler = StandardScaler()
    
    df['ClimateAnthropogenicInteraction'] = (df['MonsoonIntensity'] + df['ClimateChange']) * (df['Deforestation'] + df['Urbanization'] + df['AgriculturalPractices'] + df['Encroachments'])
    df['InfrastructurePreventionInteraction'] = (df['DamsQuality'] + df['DrainageSystems'] + df['DeterioratingInfrastructure']) * (df['RiverManagement'] + df['IneffectiveDisasterPreparedness'] + df['InadequatePlanning'])
    df['sum'] = df[num_cols].sum(axis=1)
    df['std']  = df[num_cols].std(axis=1)
    df['mean'] = df[num_cols].mean(axis=1)
    df['max']  = df[num_cols].max(axis=1)
    df['min']  = df[num_cols].min(axis=1)
    df['mode'] = df[num_cols].mode(axis=1)[0]
    df['median'] = df[num_cols].median(axis=1)
    df['q_25th'] = df[num_cols].quantile(0.25, axis=1)
    df['q_75th'] = df[num_cols].quantile(0.75, axis=1)
    df['skew'] = df[num_cols].skew(axis=1)
    df['kurt'] = df[num_cols].kurt(axis=1)
    df['sum_72_76'] = df['sum'].isin(np.arange(72, 76))
    for i in range(10,100,10):
        df[f'{i}th'] = df[num_cols].quantile(i/100, axis=1)
    df['harmonic'] = len(num_cols) / df[num_cols].apply(lambda x: (1/x).mean(), axis=1)
    df['geometric'] = df[num_cols].apply(lambda x: x.prod()**(1/len(x)), axis=1)
    df['zscore'] = df[num_cols].apply(lambda x: (x - x.mean()) / x.std(), axis=1).mean(axis=1)
    df['cv'] = df['std'] / df['mean']
    df['Skewness_75'] = (df[num_cols].quantile(0.75, axis=1) - df[num_cols].mean(axis=1)) / df[num_cols].std(axis=1)
    df['Skewness_25'] = (df[num_cols].quantile(0.25, axis=1) - df[num_cols].mean(axis=1)) / df[num_cols].std(axis=1)
    df['2ndMoment'] = df[num_cols].apply(lambda x: (x**2).mean(), axis=1)
    df['3rdMoment'] = df[num_cols].apply(lambda x: (x**3).mean(), axis=1)
    df['entropy'] = df[num_cols].apply(lambda x: -1*(x*np.log(x)).sum(), axis=1)
    df['MonsoonIntensity_squared'] = df['MonsoonIntensity'] ** 2
    # Новые производные признаки
    df['Urbanization_vs_Drainage'] = df['Urbanization'] / (df['DrainageSystems'] + 1e-6) # Добавляем малое число во избежание деления на ноль
    df['Preparedness_Score'] = df['RiverManagement'] + df['DamsQuality'] + df['DrainageSystems'] + df['IneffectiveDisasterPreparedness'] + df['InadequatePlanning']
    df['Environmental_Pressure'] = df['Deforestation'] + df['Urbanization'] + df['Siltation'] + df['AgriculturalPractices'] + df['Encroachments']
    df['Preparedness_vs_Pressure'] = df['Preparedness_Score'] / (df['Environmental_Pressure'] + 1e-6) # Добавляем малое число
    df['Vulnerability_Score'] = df['CoastalVulnerability'] + df['Landslides'] + df['WetlandLoss'] + df['PopulationScore']
    
    for v in unique_vals:
        if v<16:
            df['cnt_{}'.format(v)] = (df[num_cols] == v).sum(axis=1)
            df['cnt_wtd_{}'.format(v)] = ((df['cnt_{}'.format(v)])*v)/df['sum']
    
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df


print("Train:",len(df_train))
sample_sub=pd.read_csv("/kaggle/input/playground-series-s4e5/sample_submission.csv")
#
df_train.head()

df_train['typ']=0
df_test['typ']=1
#
df_all=pd.concat([df_train,df_test],axis=0)
df_all=getFeats(df_all)
df_all.head()

df_train=df_all[df_all['typ']==0]
df_test=df_all[df_all['typ']==1]
#
X=df_train.drop(['id','FloodProbability','typ'],axis=1)
y=df_train['FloodProbability']
#
feats=list(X.columns)

print('done')



lgb_params = {
    'boosting_type': 'gbdt', 
    'n_estimators':3000, 
    'learning_rate' :  0.007, 
    #'device':'gpu',
    'num_leaves' : 150, 
    'subsample_for_bin': 140000, 
    'min_child_samples': 150, 
    'reg_alpha': 6.075e-06, 
    'reg_lambda': 1.139e-07, 
    'colsample_bytree': 0.798,
    'subsample': 0.963, 
    'max_depth': 9,
    'random_state':0,
    'verbosity':-1}


xgb_params ={'n_estimators':4000,
             'max_depth': 12,
             #'tree_method': 'gpu_hist',
             'learning_rate': 0.02,
             'random_state':0,
             }
             

cat_params = {'n_estimators':10000,
             'l2_leaf_reg': 0.0048, 
             'max_bin': 134, 
             'learning_rate': 0.0106, 
             'max_depth': 7, 
             'random_state': 0, 
             'min_data_in_leaf': 350
             }

## Function for Cross Validation
def cross_val_train(X,y,df_test,params,mName):
    
    spl=7
    test_preds = np.zeros((len(df_test)))
    val_preds = np.zeros((len(X)))
    val_scores, train_scores = [],[]
    
    cv = KFold(spl, shuffle=True, random_state=42)
    
    for fold, (train_ind, valid_ind) in enumerate(cv.split(X,y)):
        
        X_train = X.iloc[train_ind]
        y_train = y[train_ind]
        X_val = X.iloc[valid_ind]
        y_val = y[valid_ind]
        
        if mName=='LGB':
            model = lgb.LGBMRegressor(**params, early_stopping_rounds=50)
            model.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        )

        if mName=='XGB':
            model = XGBRegressor(**params, early_stopping_rounds=3, verbose=500)
            model.fit(X_train, y_train,
                              eval_set=[(X_train, y_train), (X_val, y_val)]
                              )

        if mName=='CAT':
            model = CatBoostRegressor(**params, early_stopping_rounds=50, verbose=500)
            model.fit(X_train, y_train,
                              eval_set=[(X_train, y_train), (X_val, y_val)]
                              )
        
        y_pred_trn=model.predict(X_train)
        y_pred_val=model.predict(X_val)
        train_r2 = r2_score(y_train, y_pred_trn)
        val_r2 = r2_score(y_val, y_pred_val)
        print("Fold:",fold, " Train R2:",np.round(train_r2,5), " Val R2:",np.round(val_r2,5))
        
        test_preds += model.predict(df_test[feats])/spl
        val_preds[valid_ind] = model.predict(X_val)
        val_scores.append(val_r2)
        print("-"*50)
        
    return val_scores, val_preds, test_preds

# Evaluate the model
def modelEval(y,val_preds):
    mse = mean_squared_error(y,val_preds)
    rmse = np.sqrt(mean_squared_error(y, val_preds))
    r2 = r2_score(y, val_preds)
    #
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')


val_scores_cat, val_preds_cat, test_preds_cat=cross_val_train(X,y,df_test,cat_params,'CAT')
# Evaluate the model
modelEval(y,val_preds_cat)

val_scores_lgb, val_preds_lgb, test_preds_lgb=cross_val_train(X,y,df_test,lgb_params,'LGB')
# Evaluate the model
modelEval(y,val_preds_lgb)

val_scores_xgb, val_preds_xgb, test_preds_xgb=cross_val_train(X,y,df_test,xgb_params,'XGB')
# Evaluate the model
modelEval(y,val_preds_xgb)

val_preds = val_preds_lgb*0.6 + val_preds_xgb*0.3 + val_preds_cat*0.1
test_preds = test_preds_lgb*0.6 + test_preds_xgb*0.3 + test_preds_cat*0.1

# Evaluate the Ensemble
modelEval(y,val_preds)


plt.figure(figsize=(12,6))
# Calculate residuals
residuals = y - val_preds
# Plot residuals
plt.scatter(val_preds, residuals)
plt.axhline(y=0, color='red', linestyle='--',linewidth=3)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Plot histogram of residuals
plt.figure(figsize=(12,6))
plt.hist(residuals, bins=100, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

sub=sample_sub[['id']]
sub['FloodProbability'] = test_preds
subb.to_csv('submission.csv', index=False)
sub.head()


