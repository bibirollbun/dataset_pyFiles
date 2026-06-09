import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import os


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
train.drop(['id'],axis = 1,inplace = True)


train


train.describe()


print(train.isnull().sum())
nanlist=[col for col in train.columns if train[col].isnull().any()]
nanlist


print(test.isnull().sum())
testnanlist=[col for col in test.columns if test[col].isnull().any()]
testnanlist


numlist=[]
objlist=[col for col in test.columns if test[col].dtype == object]


for col in test.columns:
    print(f'{col}:',test[col].dtype)
    if col not in objlist:
        numlist.append(col)

numlist.remove('id')
print(f"\nnumlist:{numlist}\nobjlist:{objlist}")


def outliers(df):
    for col in numlist:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (test[col] > upper_bound)]

        print(f"{col} outlier values",outliers[col].unique())

outliers(test)
outliers(train)


def fixoutlier(df):
    for col in numlist:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        for idx, value in df[col].items():
            if value < lower or value > upper:
                df.loc[idx, col] = np.nan


fixoutlier(test)
fixoutlier(train)


train['nancount'] = 0

for col in train.columns:
    nans = train[col].isna()
    train.loc[nans, 'nancount'] += 1

test['nancount'] = 0

for col in test.columns:
    nanss = test[col].isna()
    test.loc[nanss, 'nancount'] += 1


for col in nanlist:
    train.fillna({col:train[col].median()}, inplace=True)
    test.fillna({col:test[col].median()}, inplace=True)

print(train.isnull().sum().sum(),test.isnull().sum().sum())


for col in train.columns:
    if col in objlist:
        print(train[col].unique())

print('\n#########################################################################################################\n')

for col in test.columns:
    if col in objlist:
        print(test[col].unique())


sentiment={
        "Neutral":0,
        'Positive':1,
        'Negative':-1,
}

train['Episode_Sentiment'] = train['Episode_Sentiment'].map(sentiment)
test['Episode_Sentiment'] = test['Episode_Sentiment'].map(sentiment)

test['Episode_Sentiment']=test['Episode_Sentiment'].astype('int')
test['Episode_Sentiment']=test['Episode_Sentiment'].astype('int')

objlist.remove('Episode_Sentiment')


train['Episode_Title'] = train['Episode_Title'].astype(str).str.replace('Episode ', '', regex=False)
train['Episode_Title']=train['Episode_Title'].astype('int')

test['Episode_Title'] = test['Episode_Title'].astype(str).str.replace('Episode ', '', regex=False)
test['Episode_Title']=test['Episode_Title'].astype('int')

objlist.remove('Episode_Title')


for col in train.columns:
    if col in objlist:        
        mean = train.groupby(col)['Listening_Time_minutes'].mean()
        train[col] = train[col].map(mean)
        test[col] = test[col].map(mean)


mask = train['Listening_Time_minutes'] > train['Episode_Length_minutes']
train = train[~mask]


mask = train['Host_Popularity_percentage'] > 100
median_value = train[train['Host_Popularity_percentage'] < 100]['Host_Popularity_percentage'].median()
train.loc[mask, 'Host_Popularity_percentage'] = median_value

mask = train['Guest_Popularity_percentage'] > 100
median_value = train[train['Guest_Popularity_percentage'] < 100]['Guest_Popularity_percentage'].median()
train.loc[mask, 'Guest_Popularity_percentage'] = median_value

mask = train['Number_of_Ads'] % 1 != 0
median_value = train[train['Number_of_Ads'] % 1 == 0]['Number_of_Ads'].median()
train.loc[mask, 'Number_of_Ads'] = median_value



mask = test['Host_Popularity_percentage'] > 100
median_value = test[test['Host_Popularity_percentage'] < 100]['Host_Popularity_percentage'].median()
test.loc[mask, 'Host_Popularity_percentage'] = median_value

mask = test['Guest_Popularity_percentage'] > 100
median_value = test[test['Guest_Popularity_percentage'] < 100]['Guest_Popularity_percentage'].median()
test.loc[mask, 'Guest_Popularity_percentage'] = median_value

mask = test['Number_of_Ads'] % 1 != 0
median_value = test[test['Number_of_Ads'] % 1 == 0]['Number_of_Ads'].median()
test.loc[mask, 'Number_of_Ads'] = median_value


train.drop(['Podcast_Name'],axis = 1,inplace = True)


numlist=['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

for col in numlist:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(train[col], train['Listening_Time_minutes'])
    ax.set_xlabel(col)
    ax.set_ylabel('Listening_Time_minutes')
    plt.show()


id_col_test= test['id']
test.drop(['id'],axis = 1,inplace = True)


plt.figure(figsize=(10,6))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap='summer',fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


from sklearn.model_selection import train_test_split,cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,classification_report
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna


y = train['Listening_Time_minutes']
train = train.drop(['Listening_Time_minutes'],axis=1)
X = train


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=607)


def objective(trial):
    params={
        'colsample_bytree':trial.suggest_float('colsample_bytree',0,1),
        'n_estimators':trial.suggest_int('n_estimators', 100,1000),
        'learning_rate':trial.suggest_float('learning_rate', 0.01,0.1),
        'reg_lambda':trial.suggest_float('reg_lambda', 0,4),
        'reg_alpha':trial.suggest_float('reg_alpha', 0,4),
        'max_depth':trial.suggest_int('max_depth', 2,10),
        'gamma':trial.suggest_float('gamma', 0,1),
        'objective':'reg:squarederror',
        'eval_metric':'rmse',
    }
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

    return -score.mean()


study = optuna.create_study(direction='minimize',sampler=optuna.samplers.RandomSampler(seed=607))
optuna.logging.set_verbosity(optuna.logging.WARNING)

def log_best_trial(study, trial):
    if study.best_trial == trial:
        print(f"New best trial: {trial.number} with value: {trial.value} and params: {trial.params}")


study.optimize(objective, n_trials=12,n_jobs=-1,callbacks=[log_best_trial])


best_params = study.best_params
best_score = study.best_value
print(f"Best Hyperparameters: {best_params}")
print(f"Best Score: {best_score:.6f}")


best_xgb=XGBRegressor(**study.best_params)

eval_set = [(X_train, y_train), (X_test, y_test)]

best_xgb.fit(X_train,y_train,eval_set=eval_set,verbose=False)

y_pred = best_xgb.predict(X_test)

score=cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

print(f'score: {-score}')


y_pred = best_xgb.predict(X_train)
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred, alpha=0.7, color='royalblue', edgecolor='k')
plt.plot([y_train.min(), y_train.max()], [y_pred.min(), y_pred.max()], 'r--', lw=2)
plt.xlabel("Real Value")
plt.ylabel("Predicted Value")
plt.title("Real vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()


X_df = X_train.copy() 

errors = np.abs(y_pred - y_train)

df_errors = pd.DataFrame({
    'Real': y_train,
    'Predicted': y_pred,
    'Error': errors
})

df_most_wrong = df_errors.sort_values(by='Error', ascending=False)

df_most_wrong


df_most_wrong.describe()


results = best_xgb.evals_result()

epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

plt.figure(figsize=(8,4))
plt.plot(x_axis, results['validation_0']['rmse'], label='Train rmse')
plt.plot(x_axis, results['validation_1']['rmse'], label='Test rmse')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('XGBoost RMSE Curve')
plt.legend()
plt.grid(True)
plt.show()


x_test_df = test[X_train.columns]

y_test_pred = best_xgb.predict(x_test_df)

test['predicted'] = y_test_pred

print(test[['predicted']])


sub["id"]=id_col_test
sub["Listening_Time_minutes"]=test['predicted']
sub.to_csv('submission.csv', index=False)
print(sub)
sub.info()

