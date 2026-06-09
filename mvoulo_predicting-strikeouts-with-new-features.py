import numpy as np 
from sklearn.impute import KNNImputer
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics




df_train=pd.read_csv('/kaggle/input/nwds-k/train.csv')
df_test=pd.read_csv('/kaggle/input/nwds-k/test.csv')
df_sample_solution=pd.read_csv('/kaggle/input/nwds-k/sample_solution.csv')



df_train_swing=df_train[~df_train['swing_length'].isna()]
df_train_NOswing=df_train[df_train['swing_length'].isna()]


y_swing=np.array(df_train_swing[['k','is_strike']])
y_NOswing=np.array(df_train_NOswing[['k','is_strike']])



df_test_swing=df_test[~df_test['swing_length'].isna()]
df_test_NOswing=df_test[df_test['swing_length'].isna()]


def clean_data(df_train, df_test):
    pitches=['Sinker', 'Slider', '4-Seam Fastball', 'Sweeper', 'Changeup',
           'Split-Finger', 'Cutter', 'Curveball', 'none', 'Knuckle Curve',
           'Slurve', 'Knuckleball', 'Forkball', 'Eephus', 'Screwball',
           'Other', 'Slow Curve', 'Pitch Out']
    for p in pitches:
        df_train[p]=np.where(df_train['pitch_name']==p, 1, 0)
        df_test[p]=np.where(df_test['pitch_name']==p, 1, 0)

    df_train["pitch_group_fastball"] = np.where(
    df_train["pitch_type"].isin(["FF","SI"]),
        1,0
        )
    df_train["pitch_group_bendy"] = np.where(
        df_train["pitch_type"].isin(["ST","SL","KC","CU"]),
           1, 0)
    df_train['pitch_offspeed']= np.where(
                df_train["pitch_type"].isin(["FS","CH","FC"]),
                1, 0)
    df_test["pitch_group_fastball"] = np.where(
    df_test["pitch_type"].isin(["FF","SI"]),
        1,0
        )
    df_test["pitch_group_bendy"] = np.where(
        df_test["pitch_type"].isin(["ST","SL","KC","CU"]),
           1, 0)
    df_test['pitch_offspeed']= np.where(
                df_test["pitch_type"].isin(["FS","CH","FC"]),
                1, 0)
    df_train['inning_top']=np.where(df_train['inning_topbot']=='Top', 1, 0)
    df_train['stand_R']=np.where(df_train['stand']=='R', 1, 0)
    df_train['pitcher_R']=np.where(df_train['p_throws']=='R', 1, 0)
    df_train[['on_3b', 'on_2b', 'on_1b']]=df_train[['on_3b', 'on_2b', 'on_1b']].astype(int)
    df_train['bases']= df_train['on_1b'] + 2*df_train['on_2b'] + 3*df_train['on_3b']
    df_train.drop(columns=['pitch_type', 'pitch_name', 'inning_topbot', 'stand', 'p_throws'], inplace=True)
    df_test['inning_top']=np.where(df_test['inning_topbot']=='Top', 1, 0)
    df_test['stand_R']=np.where(df_test['stand']=='R', 1, 0)
    df_test['pitcher_R']=np.where(df_test['p_throws']=='R', 1, 0)
    df_test[['on_3b', 'on_2b', 'on_1b']]=df_test[['on_3b', 'on_2b', 'on_1b']].astype(int)
    df_test['bases']= df_test['on_1b'] + 2*df_test['on_2b'] + 3*df_test['on_3b']
    df_test.drop(columns=['pitch_type', 'pitch_name', 'inning_topbot', 'stand', 'p_throws'], inplace=True)
    df_train['bat_speed'].fillna(0, inplace=True)
    df_test['bat_speed'].fillna(0, inplace=True)
    df_train['swing_length'].fillna(0, inplace=True)
    df_test['swing_length'].fillna(0, inplace=True)
      
    return df_train, df_test
def impute_missing(df_train, df_test):
    
    columns=df_test.columns 
    df_train_to_impute=df_train[columns]
    imputer = KNNImputer(n_neighbors=3)
    imputer.fit(df_train_to_impute)
    df_train_impute=pd.DataFrame(imputer.transform(df_train_to_impute), columns = df_train_to_impute.columns)
    df_test=pd.DataFrame(imputer.transform(df_test), columns =df_test. columns)
    df_train_impute[['k', 'is_strike']]=df_train[['k', 'is_strike']]
    return df_train_impute, df_test

#feature engineering
def feature_engineering(df):
    df['sz_dist']=df['sz_top']-df['sz_bot']
    df['pitcher_batter']=np.where(df['pitcher_R']==df['stand_R'], 1, 0)
    df['vertical']=np.where((df['pfx_z']>df['sz_top'])|(df['pfx_z']<df['sz_bot']), 1, 0)
    df['balls-strikes']=(df['balls']+1)/(df['strikes']+1)
    df['vertical_dist_top']=abs(df['pfx_z']-df['sz_top'])
    df['vertical_dist_top2']=abs(df['release_pos_z']-df['sz_top'])
    df['vertical']=np.where((df['pfx_z']>df['sz_top'])|(df['pfx_z']<df['sz_bot']), 1, 0)
    df['vertical_dist_bot']=abs(df['pfx_z']-df['sz_bot'])
    df['vertical_dist_bot2']=abs(df['release_pos_z']-df['sz_bot'])
    df['speed_diff']=np.where(df['bat_speed']==0, 0, abs(df['bat_speed']-df['release_speed']))
    df['ext_posz']=df['release_pos_z']-df['release_extension']
    df['spin_rate_axis']=df['release_spin_rate']/(df['spin_axis']+1)
    df['pitch_speed_ext']=df['release_speed']/(df['release_extension']+1)
    df['axis_fastball']=df['spin_axis']-180
    df['horizontal_mismatch']=np.where((abs(df['pfx_x'])>abs(df['swing_length'])), 1, 0)
    df['horizontal_dist']=np.where(df['swing_length']==0, 0, abs(df['pfx_x']-df['swing_length']))
    df['swing-homeplate']=df['swing_length']/(17/12) #home plate is 17 inches
    df['horizontal_dist2']=np.where(df['swing_length']==0, 0, abs(df['release_pos_x']-df['swing_length']))
    df['sin_armangle']=np.sin(df['arm_angle'])
    df['cos_armangle']=np.cos(df['arm_angle'])
    df['tan_armangle']=np.tan(df['arm_angle'])
    df['arm_angle90ext']=(df['arm_angle']/90)*df['release_extension']
    df['arm_angle45']=(df['arm_angle']/45)*df['release_extension']
    df['arm_angle30']=(df['arm_angle']/30)*df['release_extension']
    df['arm_angle60']=(df['arm_angle']/60)*df['release_extension']
    df['arm_angle180']=(df['arm_angle']/180)*df['release_extension']
    df['dist/speed']=60.5/(df['release_speed'])
    df['extension_60.5']=df['release_extension']/60.5 #60.5 feet pitchers mound to home plate
    df['60.5extension']=60./5/df['release_extension']
    #following pitch groups are taken from https://www.kaggle.com/code/stephensuttonbrown/movement-vs-expected
 
    df['swing_short']=np.where(((df['swing_length']<7.3)& (df['swing_length']!=0)), 1, 0) #compare to average swing length
    df['swing_fast']=np.where((df['bat_speed']>75& (df['swing_length']!=0)), 1, 0) #compare to bat speed considered 'fast'
    df['strikes/thru']=df['strikes']/(df['n_thruorder_pitcher']+1)
    df['balls/thru']=df['balls']/(df['n_thruorder_pitcher']+1)
    df['strikes/balls']=df['strikes']/(df['balls']+1)
    df['pfx_distance']=(df['pfx_x']**2+df['pfx_z']**2)**.5
    df['release_distance']=(df['release_pos_x']**2+df['release_pos_x']**2)**.5
    df['distance_difference']=abs(df['release_distance']-df['pfx_distance'])
    df['batspeed_length']=df['bat_speed']/df['swing_length'] #thank you, @danpettyaz
    df['neg_arm_angle']=np.where(df['arm_angle']<=0, 1, 0)
    df['fatigue'] =  df['release_speed'] / (df['n_thruorder_pitcher'] +1.2)**2
    return df
def pitch_columns(df_train_df_test):

    return df


df_train_swing, df_test_swing=clean_data(df_train_swing, df_test_swing)

df_train_NOswing, df_test_NOswing=clean_data(df_train_NOswing, df_test_NOswing)




df_train_swing_impute, df_test_swing=impute_missing(df_train_swing, df_test_swing)



df_train_NOswing_impute, df_test_NOswing=impute_missing(df_train_NOswing, df_test_NOswing)



df_train_swing=feature_engineering(df_train_swing_impute)
df_test_swing=feature_engineering(df_test_swing)

df_train_NOswing=feature_engineering(df_train_NOswing_impute)
df_test_NOswing=feature_engineering(df_test_NOswing)


df_train_swing[['k', 'is_strike']]=y_swing
df_train_NOswing[['k', 'is_strike']]=y_NOswing


def get_final_features(df):
    correlations=df.corr()[['is_strike']]
    final_features=list(correlations[(correlations['is_strike']>=.003) |(correlations['is_strike']<=-.003)].T.columns.values)
    #remove other redundant features
    threshold = .7
    correlation_matrix = df[final_features].drop(columns=['k', 'is_strike']).corr()
    highly_correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                highly_correlated_features.add(colname)
    
    final_features=list(set(final_features)-highly_correlated_features)
    final_features.remove('is_strike')
    final_features.remove('k')
    return final_features
final_features_swing=get_final_features(df_train_swing)
final_features_NOswing=get_final_features(df_train_NOswing)



len(final_features_swing)


len(final_features_NOswing)


df_train_swing=df_train_swing[df_train_swing['strikes']==2]
df_train_Noswing=df_train_NOswing[df_train_NOswing['strikes']==2]
#final_features.remove('index')





X=df_train_swing[final_features_swing]

y=df_train_swing['k']
SEED=42

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []

models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LGBM', LGBMClassifier(verbose=-1)))
models.append(('XGB', xgb.XGBClassifier()))
models.append(('cat', CatBoostClassifier(logging_level='Silent')))
# evaluate each model in turn
results = []
names = []

scoring = 'neg_log_loss'

for name, model in models:
    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', model)])
    cv_results = model_selection.cross_val_score(pipeline, X, y, cv=SKF, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




from sklearn.model_selection import KFold, cross_val_score
import optuna  # pip install optuna
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
def optuna_call(transformer,X,y,SKF):

    def tune(objective):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=True)

        params = study.best_params
        best_score = study.best_value
        print(f"Best score: {best_score}\n")
        print(f"Optimized parameters: {params}\n")
        return params

    def objective(trial):
        
        param = {
        "learning_rate": trial.suggest_float("learning_rate", 4e-2, 7e-2, log=True),
         "depth": trial.suggest_int("depth",4,8),
        "subsample": trial.suggest_float("subsample", 0.2, .95),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, .95),
        "iterations":trial.suggest_int("iterations",250, 600 )
        
        }   
        cat = CatBoostClassifier(**param,  logging_level='Silent')  
        pipeline=Pipeline([('tr', transformer), ('cat',  cat)], verbose = False)
        scores = cross_val_score( pipeline, X, y, cv=SKF, scoring="neg_log_loss")
        return scores.mean()
    
    return tune(objective)
transformer=MinMaxScaler()
#optuna_call(transformer, X, y, SKF)


cat_params={'learning_rate': 0.040180976598548655,
 'depth': 8,
 'subsample': 0.9464574043874763,
 'colsample_bylevel': 0.6559838533004978,
 'iterations': 574}



clf2 = pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('cat', CatBoostClassifier(**cat_params, logging_level='Silent'))])

X=np.array(df_train_swing[final_features_swing])

y=np.array(df_train_swing['k'])



clf2.fit(X,y)
probs=clf2.predict_proba(df_test_swing[final_features_swing])


model=CatBoostClassifier(**cat_params, logging_level='Silent')
model.fit(X,y)



# Get feature importances
importances = model.feature_importances_

# Sort the importances
sorted_indices = np.argsort(importances)[::-1]

# Print the feature importances
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f+1}. Feature {final_features_swing[f]}: {importances[sorted_indices[f]]:.4f}")

# Visualize the feature importances
plt.figure(figsize=(35, 25))
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), final_features_swing, rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()



submission1=df_test_swing[['index']]
submission1['k']=probs[:,1]
submission1.head()



X=df_train_NOswing[final_features_NOswing]

y=df_train_NOswing['k']
SEED=42

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []

models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LGBM', LGBMClassifier(verbose=-1)))
models.append(('XGB', xgb.XGBClassifier()))
models.append(('cat', CatBoostClassifier(logging_level='Silent')))
# evaluate each model in turn
results = []
names = []

scoring = 'neg_log_loss'
'''
for name, model in models:
    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', model)])
    cv_results = model_selection.cross_val_score(pipeline, X, y, cv=SKF, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''


SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# Define the helper function so that it can be reused
def tune(objective):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")
    return params
def objective(trial):
    """
    Objective function to be minimized.
    """
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
             
        "num_class": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 13.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 13.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.05, 1.0),
       
        "max_depth": trial.suggest_int("max_depth", 3, 11),
        "n_estimators": trial.suggest_int("n_estimators", 20, 600),
        "learning_rate": trial.suggest_float('learning_rate', 1e-2, .1, log=True)
    }
    gbm = LGBMClassifier(**param, verbose=-1)

   
 

    scores = cross_val_score(
        gbm, X, y, cv=SKF, scoring="neg_log_loss"
    )
    return scores.mean()
#lgb_params = tune(objective)


lgb_params={'lambda_l1': 0.003972142775117704, 'lambda_l2': 0.0017523052274764732, 'num_leaves': 59, 'feature_fraction': 0.3687924776813824, 'max_depth': 4, 'n_estimators': 332, 'learning_rate': 0.05338446795399978}


def optuna_call(transformer,X,y,SKF):

    def tune(objective):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10, show_progress_bar=True)

        params = study.best_params
        best_score = study.best_value
        print(f"Best score: {best_score}\n")
        print(f"Optimized parameters: {params}\n")
        return params

    def objective(trial):
        
        param = {
        "learning_rate": trial.suggest_float("learning_rate", 3e-2, 1e-1, log=True),
         "depth": trial.suggest_int("depth",5,9),
        "subsample": trial.suggest_float("subsample", 0.2, .95),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, .95),
        "iterations":trial.suggest_int("iterations",250, 700 )
        
        }   
        cat = CatBoostClassifier(**param,  logging_level='Silent')  
        pipeline=Pipeline([('tr', transformer), ('cat',  cat)], verbose = False)
        scores = cross_val_score( pipeline, X, y, cv=SKF, scoring="neg_log_loss")
        return scores.mean()
    
    return tune(objective)
transformer=MinMaxScaler()
#optuna_call(transformer, X, y, SKF)


cat_params={'learning_rate': 0.06169725852683252,
 'depth': 5,
 'subsample': 0.7062993982925228,
 'colsample_bylevel': 0.25507081239555685,
 'iterations': 612}


from mlxtend.plotting import plot_decision_regions
clf1 = pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('lgbm', LGBMClassifier(**lgb_params, verbose=-1))])
clf2 = pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('cat', CatBoostClassifier(**cat_params, logging_level='Silent'))])

X=np.array(df_train_NOswing[final_features_NOswing])

y=np.array(df_train_NOswing['k'])



from sklearn.ensemble import VotingClassifier
voting_clf_soft = VotingClassifier(
    estimators=[
        ('LGBM', clf1),  
        ('CAT', clf2),
   
    ],
    voting='soft'  # Specify soft voting, where class probabilities are combined
)


#scores = cross_val_score( voting_clf_soft, X, y, cv=SKF, scoring="neg_log_loss")
#print(scores.mean())



clf2.fit(X,y)
probs=clf2.predict_proba(df_test_NOswing[final_features_NOswing])
model=CatBoostClassifier(**cat_params, logging_level='Silent')
model.fit(X,y)



# Get feature importances
importances = model.feature_importances_

# Sort the importances
sorted_indices = np.argsort(importances)[::-1]

# Print the feature importances
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f+1}. Feature {final_features_NOswing[f]}: {importances[sorted_indices[f]]:.4f}")

# Visualize the feature importances
plt.figure(figsize=(35, 25))
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), final_features_NOswing, rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()



submission2=df_test_NOswing[['index']]
submission2['k']=probs[:,1]
submission2.head()


submission=pd.concat([submission1, submission2])


submission.to_csv('submission.csv', index=False)


submission.head()




