import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


train=pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s4e11/sample_submission.csv')


train.head()


train.columns = [col.replace(' ', '_') for col in train.columns]
test.columns = [col.replace(' ', '_') for col in test.columns]
train.columns


print(train.dtypes,"\n\n",test.dtypes)


train['Sleep_Duration'].unique()


sleep={
        "More than 8 hours":9,
        'Less than 5 hours':4,
        '5-6 hours':5.5,
        '7-8 hours':7.5,
        '1-2 hours':1.5,
        '6-8 hours':7,
        '4-6 hours':5,
        '6-7 hours':6.5,
        '10-11 hours':10.5,
        '8-9 hours':8.5,
        '9-11 hours':10,
        '2-3 hours':2.5,
        '3-4 hours':3.5,
        'Moderate':6,
        '4-5 hours':4.5,
        '9-6 hours':7.5,
        '1-3 hours':2,
        '1-6 hours':4,
        '8 hours':8,
        '10-6 hours':8,
        'Unhealthy':3,
        'Work_Study_Hours':6,
        '3-6 hours':3.5,
        '9-5':7,
        '9-5 hours':7,
}

train['Sleep_Duration'] = train['Sleep_Duration'].map(sleep)
test['Sleep_Duration'] = test['Sleep_Duration'].map(sleep)


sleep_med=train['Sleep_Duration'].median()
train.fillna({'Sleep_Duration':sleep_med}, inplace=True)
test.fillna({'Sleep_Duration':sleep_med}, inplace=True)


print('Gender:',train['Gender'].unique())
print('Thoughts:',train['Have_you_ever_had_suicidal_thoughts_?'].unique())
print('Working:',train['Working_Professional_or_Student'].unique())
print('History:',train['Family_History_of_Mental_Illness'].unique())


gender={
    'Male':0,
    'Female':1,
}

work={
    'Working Professional':1,
    'Student':0,
}

Thoughts={
    'No':0,
    'Yes':1,
}

History={
    'No':0,
    'Yes':1,
}

train['Working_Professional_or_Student'] = train['Working_Professional_or_Student'].map(work)
test['Working_Professional_or_Student'] = test['Working_Professional_or_Student'].map(work)

train['Gender'] = train['Gender'].map(gender)
test['Gender'] = test['Gender'].map(gender)

train['Have_you_ever_had_suicidal_thoughts_?'] = train['Have_you_ever_had_suicidal_thoughts_?'].map(Thoughts)
test['Have_you_ever_had_suicidal_thoughts_?'] = test['Have_you_ever_had_suicidal_thoughts_?'].map(Thoughts)

train['Family_History_of_Mental_Illness'] = train['Family_History_of_Mental_Illness'].map(History)
test['Family_History_of_Mental_Illness'] = test['Family_History_of_Mental_Illness'].map(History)

print('Gender:',train['Gender'].unique(),'Working:',train['Working_Professional_or_Student'].unique(),'Thoughts:',train['Have_you_ever_had_suicidal_thoughts_?'].unique(),'History:',train['Family_History_of_Mental_Illness'].unique())


train['Work_Hours'] = train.apply(
    lambda row: np.nan if pd.isna(row['Work/Study_Hours']) 
    else row['Work/Study_Hours'] if row['Working_Professional_or_Student'] == 1
    else 0,
    axis=1
)

train['Study_Hours'] = train.apply(
    lambda row: np.nan if pd.isna(row['Work/Study_Hours']) 
    else row['Work/Study_Hours'] if row['Working_Professional_or_Student'] == 0
    else 0,
    axis=1
)


test['Work_Hours'] = test.apply(
    lambda row: np.nan if pd.isna(row['Work/Study_Hours']) 
    else row['Work/Study_Hours'] if row['Working_Professional_or_Student'] == 1
    else 0,
    axis=1
)

test['Study_Hours'] = test.apply(
    lambda row: np.nan if pd.isna(row['Work/Study_Hours']) 
    else row['Work/Study_Hours'] if row['Working_Professional_or_Student'] == 0
    else 0,
    axis=1
)


train.drop(['Work/Study_Hours'], axis=1, inplace=True)
test.drop(['Work/Study_Hours'], axis=1, inplace=True)


train['Academic_Pressure'] = train.apply(
    lambda row: 0 if row['Working_Professional_or_Student'] == 1
    else (np.nan if pd.isna(row['Academic_Pressure']) else row['Academic_Pressure']),
    axis=1
)

test['Academic_Pressure'] = test.apply(
    lambda row: 0 if row['Working_Professional_or_Student'] == 1
    else (np.nan if pd.isna(row['Academic_Pressure']) else row['Academic_Pressure']),
    axis=1
)


train['Work_Pressure'] = train.apply(
    lambda row: 0 if row['Working_Professional_or_Student'] == 0
    else (np.nan if pd.isna(row['Work_Pressure']) else row['Work_Pressure']),
    axis=1
)

test['Work_Pressure'] = test.apply(
    lambda row: 0 if row['Working_Professional_or_Student'] == 0
    else (np.nan if pd.isna(row['Work_Pressure']) else row['Work_Pressure']),
    axis=1
)


train['Dietary_Habits'].unique()


diet={
    'More Healty':0,
    'Healthy':1,
    'Less than Healthy':2,
    'Less Healthy':2,
    'Moderate':3,
    'Unhealthy':4,   
    'No Healthy':4,
}

train['Dietary_Habits'] = train['Dietary_Habits'].map(diet)
test['Dietary_Habits'] = test['Dietary_Habits'].map(diet)

train['Dietary_Habits'].unique()


train['Degree'].unique()


degree = {
    "BCom": "B.Com", "B.Com": "B.Com", "B.Comm": "B.Com",
    "B.Tech": "B.Tech", "BTech": "B.Tech", "B.T": "B.Tech",
    "BSc": "B.Sc", "B.Sc": "B.Sc", "Bachelor of Science": "B.Sc",
    "BArch": "B.Arch", "B.Arch": "B.Arch",
    "BA": "B.A", "B.A": "B.A",
    "BBA": "BBA", "BB": "BBA",
    "BCA": "BCA",
    "BE": "BE",
    "BEd": "B.Ed", "B.Ed": "B.Ed",
    "BPharm": "B.Pharm", "B.Pharm": "B.Pharm",
    "BHM": "BHM",
    "LLB": "LLB", "LL B": "LLB", "LL BA": "LLB", "LL.Com": "LLB", "LLCom": "LLB",
    "MCom": "M.Com", "M.Com": "M.Com",
    "M.Tech": "M.Tech", "MTech": "M.Tech", "M.T": "M.Tech",
    "MSc": "M.Sc", "M.Sc": "M.Sc", "Master of Science": "M.Sc",
    "MBA": "MBA",
    "MCA": "MCA",
    "MD": "MD",
    "ME": "ME",
    "MEd": "M.Ed", "M.Ed": "M.Ed",
    "MArch": "M.Arch", "M.Arch": "M.Arch",
    "MPharm": "M.Pharm", "M.Pharm": "M.Pharm",
    "MA": "MA", "M.A": "MA",
    "MPA": "MPA",
    "LLM": "LLM",
    "PhD": "PhD",
    "MBBS": "MBBS",
    "CA": "CA",
    "Class 12": "Class 12", "12th": "Class 12",
    "Class 11": "Class 11", "11th": "Class 11"
}

train['Degree'] = train['Degree'].map(degree)
test['Degree'] = test['Degree'].map(degree)

train['Degree'].unique()


print(train.isnull().sum(),"\n\n",test.isnull().sum())


nanlist=['Work_Hours','Study_Hours','Profession','Academic_Pressure','Work_Pressure','CGPA','Study_Satisfaction','Job_Satisfaction','Dietary_Habits','Degree','Financial_Stress']


for col in nanlist:
    napercent=(train[col].isna().sum()/train[col].shape[0])*100
    print(f"{col} is %{napercent:.2f} null")


train['Work_Stress'] = train.apply(
    lambda row:(row['Financial_Stress'] + row['Work_Pressure'] - row['Job_Satisfaction'])if row['Working_Professional_or_Student'] == 1 
    else 0,
    axis=1
)

train['Academic_Stress'] = train.apply(
    lambda row:(row['Financial_Stress'] + row['Academic_Pressure'] - row['Study_Satisfaction'])if row['Working_Professional_or_Student'] == 0 
    else 0,
    axis=1
)

test['Work_Stress'] = test.apply(
    lambda row:(row['Financial_Stress'] + row['Work_Pressure'] - row['Job_Satisfaction'])if row['Working_Professional_or_Student'] == 1 
    else 0,
    axis=1
)

test['Academic_Stress'] = test.apply(
    lambda row:(row['Financial_Stress'] + row['Academic_Pressure'] - row['Study_Satisfaction'])if row['Working_Professional_or_Student'] == 0 
    else 0,
    axis=1
)


nanlist=['Work_Stress','Academic_Stress','Work_Hours','Study_Hours','Profession','Academic_Pressure','Work_Pressure','CGPA','Study_Satisfaction','Job_Satisfaction','Dietary_Habits','Degree','Financial_Stress']


for col in nanlist:
    print(f"{col}:",train[col].dtype)


train.fillna({'Degree': 'Unknown', 'Profession': 'Unknown'}, inplace=True)
test.fillna({'Degree': 'Unknown', 'Profession': 'Unknown'}, inplace=True)


from sklearn.model_selection import StratifiedKFold
from category_encoders import TargetEncoder


n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


target_encoder = TargetEncoder(cols=['Profession', 'Degree'])


for train_index, val_index in kf.split(train, train['Depression']):
    train_fold = train.iloc[train_index]
    val_fold = train.iloc[val_index]
    

    train_fold_encoded = target_encoder.fit_transform(train_fold[['Profession', 'Degree']], train_fold['Depression'])
    val_fold_encoded = target_encoder.transform(val_fold[['Profession', 'Degree']])
    

    train.loc[val_index, ['Profession', 'Degree']] = val_fold_encoded

test_encoded = target_encoder.transform(test[['Profession', 'Degree']])
test[['Profession', 'Degree']] = test_encoded


mean_n = train.groupby('Name')['Depression'].mean()
train['Name'] = train['Name'].map(mean_n)
test['Name'] = test['Name'].map(mean_n)

mean_n = train.groupby('City')['Depression'].mean()
train['City'] = train['City'].map(mean_n)
test['City'] = test['City'].map(mean_n)


for col in nanlist:
    med=train[col].median()
    train.fillna({col:med}, inplace=True)
    test.fillna({col:med}, inplace=True)

print(train.isnull().sum(),"\n\n",test.isnull().sum())


newnan=["Name","City"]

for col in newnan:
    med=train[col].median()
    test.fillna({col:med}, inplace=True)

test.isnull().sum()


train.head(10)


id_col_tra= train['id']
train.drop(['id'],axis = 1,inplace = True)
id_col_test= test['id']
test.drop(['id'],axis = 1,inplace = True)


from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(train.drop('Depression', axis=1), train['Depression'])
mi_series = pd.Series(mi, index=train.drop('Depression', axis=1).columns)

feature_importance = pd.DataFrame({
    'Feature': train.drop('Depression', axis=1).columns,
    'Importance': mi_series
})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()


plt.figure(figsize=(15,10))
sns.heatmap(train.corr(), annot=True, cmap='summer',fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import optuna
from sklearn.preprocessing import StandardScaler


y = train['Depression']
train = train.drop(['Depression'],axis=1)
X = train


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=607)

def objective(trial):
    colsample_bytree= trial.suggest_float('colsample_bytree',0,1)
    n_estimators = trial.suggest_int('n_estimators', 400,1000)
    learning_rate = trial.suggest_float('learning_rate', 0.01,0.1)
    reg_lambda = trial.suggest_float('reg_lambda', 0,4)
    reg_alpha = trial.suggest_float('reg_alpha', 0,4)
    max_depth = trial.suggest_int('max_depth', 2,10)
    gamma = trial.suggest_float('gamma', 0,0.5)
    eval_metric='auc'
    
    model = XGBClassifier(
    colsample_bytree = colsample_bytree,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda,
    gamma=gamma,
    eval_metric='auc',
    random_state=607
)
    model.fit(X_train, y_train)
    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return score


study = optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler(seed=607))
optuna.logging.set_verbosity(optuna.logging.WARNING)

def log_best_trial(study, trial):
    if study.best_trial == trial:
        print(f"New best trial: {trial.number} with value: {trial.value} and params: {trial.params}")


study.optimize(objective, n_trials=100,callbacks=[log_best_trial])


best_params = study.best_params
best_score = study.best_value
print(f"Best Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_score:.6f}")


optuna.visualization.plot_param_importances(study)


best_params = study.best_params
best_score = study.best_value
print(f"Best Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_score:.6f}")

n_estimators = best_params['n_estimators']
reg_alpha = best_params['reg_alpha']
learning_rate = best_params['learning_rate']
reg_lambda = best_params['reg_lambda']
max_depth = best_params['max_depth']
colsample_bytree = best_params['colsample_bytree']
gamma = best_params['gamma']


best_xgb=XGBClassifier(
    colsample_bytree = colsample_bytree,
    n_estimators = n_estimators,
    learning_rate = learning_rate,
    reg_alpha = reg_alpha,
    reg_lambda = reg_lambda,
    max_depth = max_depth,
    gamma=gamma,
    eval_metric='auc',
    random_state=607
)

eval_set = [(X_train, y_train), (X_test, y_test)]

best_xgb.fit(X_train,y_train,eval_set=eval_set,verbose=False)

y_pred = best_xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1])
print(roc_auc)


feature_importance = pd.DataFrame({
    'Feature': train.columns,
    'Importance': best_xgb.feature_importances_
})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()


results = best_xgb.evals_result()

epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

plt.figure(figsize=(8,4))
plt.plot(x_axis, results['validation_0']['auc'], label='Train AUC')
plt.plot(x_axis, results['validation_1']['auc'], label='Test AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('XGBoost AUC Learning Curve')
plt.legend()
plt.grid(True)
plt.show()


x_test_df = test[X_train.columns]

y_test_pred = best_xgb.predict(x_test_df)

test['predicted'] = y_test_pred

print(test[['predicted']])


sub["id"]=id_col_test
sub["Depression"]=test['predicted']
sub.to_csv('submission.csv', index=False)
print(sub)
sub.info()

