import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')


np.random.seed(42)


data_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


len(data_train)


len(data_test)


data_test.info()


data_train.drop(['id', 'day'], axis=1).hist(figsize=(10, 10));


sns.heatmap(data_train.corr(), cmap='coolwarm')


fig, axs = plt.subplots(2,2, figsize=(11,10))

# boxenplot is an enhanced version of boxplot
sns.boxenplot(data=data_train[['dewpoint', 'rainfall']], x='rainfall', y='dewpoint', ax=axs[0, 0])  
sns.boxenplot(data=data_train[['humidity', 'rainfall']], x='rainfall', y='humidity', ax=axs[1, 0])  

# kdeplot is a kernel distribution plot -  https://en.wikipedia.org/wiki/Kernel_density_estimation   
sns.kdeplot(data=data_train[['dewpoint', 'rainfall']], x='dewpoint', ax=axs[0, 1])  
sns.kdeplot(data=data_train[['humidity', 'rainfall']], x='humidity', ax=axs[1, 1])  


plt.hist(data_train['day']);


from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold #we have clear class imbalance so there's a need for stratified k-fold


def naive_test(X, y, n_splits=5, model_ = CatBoostClassifier):
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    metrics_values = []
    for train_ids, test_ids in folds.split(X, y):
        model = model_(verbose=0)
        model.fit(X[train_ids], y[train_ids])
        metrics_values.append(roc_auc_score(y[test_ids], model.predict(X[test_ids])))
    return np.array(metrics_values)

X = data_train.drop(['id', 'rainfall'], axis=1).values
y = data_train.rainfall.to_numpy()
metrics_values = naive_test(X, y)


print(f'Naive Approach leads to {np.mean(metrics_values):.4} ROC-AUC')


def preproccess(data_new):
    data_new['winddirection'] = data_new['winddirection'].fillna(data_new['winddirection'].median())
    '''

    data_new['day_cos'] = np.cos(data_new['day'])
    data_new['day_sin'] = np.sin(data_new['day'])
    data_new['temp_diff'] = data_new.maxtemp - data_new.mintemp
    data_new['tem_dew_diff'] = data_new.dewpoint - data_new.temparature 
    data_new['weather_severnity'] = (data_new['cloud'] * data_new['humidity']) / (data_new['pressure'] * (data_new['sunshine'] + 1))

    #data_new['humidity_prev'] = data_new['humidity'].shift(1) 
    #data_new['cloud_prev']    = data_new['cloud'].shift(1) 
    #data_new['dewpoint_prev'] = data_new['dewpoint'].shift(1) 
    #data_new['pressure_prev'] = data_new['pressure'].shift(1) 
    data_new['humidity_dew_diff'] = data_new.dewpoint - data_new.humidity
    #data_new['high_cloud_humidity'] = ((data_new['humidity'] > 80) & (data_new['cloud'] > 60)).astype(int)

    data_new["dew_humidity"] = data_new["dewpoint"] * data_new["humidity"] # ***
    data_new["cloud_windspeed"] = data_new["cloud"] * data_new["windspeed"] # ***
    data_new["cloud_to_humidity"] = data_new["cloud"] / data_new["humidity"]
    data_new["temp_to_sunshine"] = data_new["sunshine"] / data_new["temparature"] # ***
    '''
    data_new['temp_range'] = data_new['maxtemp'] - data_new['mintemp']
    data_new['max_min_temp_ratio'] = data_new['maxtemp'] / data_new['mintemp']
    data_new['cloud_coverage'] = data_new['cloud'] / 100  
    data_new['weather_severity'] = (data_new['cloud'] * data_new['humidity']) / (data_new['pressure'] * (data_new['sunshine'] + 1))
    data_new['temp_humidity_index'] = (data_new['temparature'] * data_new['humidity']) / 100
    data_new['pressure_temp_humidity'] = (data_new['pressure'] * data_new['temparature']) / data_new['humidity']



    return data_new


data_new = data_train.copy()
data_new = preproccess(data_new)
data_new


X = data_new.drop(['id', 'rainfall'], axis=1).values
y = data_new.rainfall.to_numpy()
metrics_values = naive_test(X, y)


print(f'Temperautre-connected and day-year statistics approach leads to {np.mean(metrics_values):.4} ROC-AUC')


from imblearn.over_sampling import SMOTE


data_new.drop(['id', 'rainfall'], axis=1)


X = data_new.drop(['id', 'rainfall'], axis=1).values
y = data_new.rainfall.to_numpy()

sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X, y)

metrics_values = naive_test(X_smote, y_smote)


print(f'Addition of SMOTE leads to {np.mean(metrics_values):.4} ROC-AUC')


from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)


model1 = LogisticRegression()
model2 = RandomForestClassifier(random_state=42)
model3 = XGBClassifier(objective='binary:logistic', random_state=42)
model4 = LGBMClassifier(random_state=42)
model5 = CatBoostClassifier(random_state=42, verbose=0)


pipeline1 = Pipeline([
    ('scaler', StandardScaler()),  
    ('logreg', model1)
])

pipeline2 = Pipeline([
    ('scaler', StandardScaler()),  
    ('rf', model2)
])

pipeline3 = Pipeline([
    ('scaler', StandardScaler()), 
    ('xgb', model3)
])

pipeline4 = Pipeline([
    ('scaler', StandardScaler()),  
    ('lgb', model4)
])

pipeline5 = Pipeline([
    ('scaler', StandardScaler()),  
    ('catboost', model5)
])


voting_model = VotingClassifier\
(
    estimators=[
    ('log_reg', pipeline1),
    ('rf',  pipeline2),
    ('lgb', pipeline4),
    ('xgb', pipeline3),
    ('cb',  pipeline5)]
    , voting='soft'
)



voting_model.fit(X_train, y_train)
y_pred_proba = voting_model.predict_proba(X_test)[:, 1] 
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc}")


train_stack = preproccess(data_train.drop(['id', 'rainfall'], axis=1)).copy()
test_stack  = preproccess(data_test.drop(['id'], axis=1)).copy()

train_stack['voting_pred'] = voting_model.predict_proba(train_stack.values)[:, 1] 
test_stack['voting_pred'] = voting_model.predict_proba(test_stack.values)[:, 1] 


!pip3 install optuna
from optuna import trial
import optuna


def objective_catboost(trial):
    n_estimators  = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth     = trial.suggest_int("depth", 3, 10)
    l2_reg        = trial.suggest_float("l2_leaf_reg", 1e-4, 1e-1, log=True)
    subsample     = trial.suggest_float("subsample", 0.5, 1.0)

    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics_values = []
    for train_ids, test_ids in folds.split(X_smote, y_smote):
        model = CatBoostClassifier(verbose=0, 
                                   n_estimators=n_estimators,
                                   depth=max_depth,
                                   learning_rate=learning_rate,
                                   l2_leaf_reg=l2_reg,
                                   subsample=subsample,
                                   random_state=42)
        
        model.fit(X_smote[train_ids], y_smote[train_ids])
        metrics_values.append(roc_auc_score(y_smote[test_ids], model.predict(X_smote[test_ids])))

    metrics_value = np.mean(metrics_values)
    return metrics_value


study_cb = optuna.create_study(direction="maximize")
study_cb.optimize(objective_catboost, n_trials=10) #put here at least 50


import optuna.visualization as vis

# Plot the optimization history
fig_history = vis.plot_optimization_history(study_cb)
fig_history.show()


study_cb.best_params


pd.concat([data_train.drop('rainfall', axis=1), data_test], join='inner')


for_submission = preproccess(pd.concat([data_train.drop('rainfall', axis=1), data_test], join='inner'))
for_submission = for_submission[for_submission['id'] >= 2190]


for_submission = preproccess(data_test)


model = model_best = CatBoostClassifier(
    **study_cb.best_params,
    random_seed=42
).fit(train_stack, data_train['rainfall'])

y_pred = model.predict_proba(test_stack.values)[:, 1]


submission['rainfall'] = y_pred


submission.to_csv('sumbit-vanila.csv', index=False)

