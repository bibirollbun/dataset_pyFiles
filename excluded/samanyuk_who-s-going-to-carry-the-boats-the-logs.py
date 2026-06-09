# importing
import pandas as pd
import numpy as np


df=pd.read_csv(r'/kaggle/input/playground-series-s5e5/train.csv')
df


te=pd.read_csv(r'/kaggle/input/playground-series-s5e5/test.csv')
tte=te.copy()
te


# importing
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df.info()


te.info()


df['Sex'] = df['Sex'].apply(lambda x: 1 if x == "male" else 0 if pd.notnull(x) else -1)
te['Sex'] = te['Sex'].apply(lambda x: 1 if x == "male" else 0 if pd.notnull(x) else -1)


fig,ax=plt.subplots(2,4,figsize=(20,10))
ax=ax.flatten()
i=0
for col in df.columns:
    if col!='id':
        sns.kdeplot(data=df,x=col,ax=ax[i])
        i+=1
plt.tight_layout()
plt.show()


fig,ax=plt.subplots(3,2,figsize=(20,10))
ax=ax.flatten()
i=0
cols = [col for col in df.columns if col not in ['id', 'Calories', 'Sex']]
for col in cols:
    sns.scatterplot(data=df,x=col,y='Calories',ax=ax[i])
    i+=1
plt.tight_layout()
plt.show()


plt.figure(figsize=(20,5))
sns.boxplot(data=df,y='Calories',x='Sex')
plt.show()


from sklearn.feature_selection import mutual_info_regression


plt.figure(figsize=(20,5))
sns.heatmap(df.corr(),annot=True)
plt.show()


x=df.drop(columns=['id','Calories'])
y=df['Calories']
mi=mutual_info_regression(x,y)
mi_df=pd.DataFrame({'cols':x.columns,'mi':mi})
mi_df.sort_values(by='mi',ascending=False)

plt.figure(figsize=(20,5))
sns.barplot(data=mi_df.sort_values(by='mi', ascending=False), x='cols', y='mi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import numpy as np
from itertools import combinations

cols = ['Duration', 'Heart_Rate', 'Body_Temp']
for feature in cols:
    df[f'{feature}_squared'] = df[feature] ** 2
    df[f'{feature}_cubed'] = df[feature] ** 3
    df[f'{feature}_log'] = np.log1p(df[feature])
    df[f'{feature}_sqrt'] = np.sqrt(df[feature])
    df[f'{feature}_inv'] = 1 / (df[feature] + 1e-6)  
for f1, f2 in combinations(cols, 2):
    df[f'{f1}_x_{f2}'] = df[f1] * df[f2]

for feature in cols:
    te[f'{feature}_squared'] = te[feature] ** 2
    te[f'{feature}_cubed'] = te[feature] ** 3
    te[f'{feature}_log'] = np.log1p(te[feature])
    te[f'{feature}_sqrt'] = np.sqrt(te[feature])
    te[f'{feature}_inv'] = 1 / (te[feature] + 1e-6)  
for f1, f2 in combinations(cols, 2):
    te[f'{f1}_x_{f2}'] = te[f1] * te[f2]


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
plt.show()


x=df.drop(columns=['id','Calories'])
y=df['Calories']
mi=mutual_info_regression(x,y)
mi_df=pd.DataFrame({'cols':x.columns,'mi':mi})
mi_df.sort_values(by='mi',ascending=False)

plt.figure(figsize=(20,5))
sns.barplot(data=mi_df.sort_values(by='mi', ascending=False), x='cols', y='mi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


X = df.drop(columns=['id', 'Calories'])
y = np.log1p(df['Calories'])

def objective(trial):
    params = {
        'loss_function': 'RMSE',
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': 0,
        'random_seed': 42,
        'cat_features': [col for col in X.select_dtypes(include=['object']).columns],
        'task_type': 'GPU',
        'devices': '0'
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X, y, verbose=False)
    val_rmse = model.best_score_['learn']['RMSE']
    return val_rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params.update({
    'loss_function': 'RMSE',
    'iterations': 1000,
    'random_seed': 42,
    'cat_features': [col for col in X.select_dtypes(include=['object']).columns],
    'task_type': 'GPU',
    'devices': '0'
})

model = CatBoostRegressor(**best_params)
model.fit(X, y, verbose=False)


log_preds = model.predict(te)
final_predictions = np.expm1(log_preds)
final_predictions = np.clip(final_predictions, 1, 314)

submission = pd.DataFrame({'id': tte['id'], 'Calories': final_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")

