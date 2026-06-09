import pandas as pd 
import numpy as np
import matplotlib .pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train


test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test


train.info()


test.info()


train.describe()


test.describe()


train.isnull().sum()


test.isnull().sum()


train.duplicated().sum()


test.duplicated().sum()


round(100*train["Personality"].value_counts(normalize=True),2)


train['Personality'].value_counts()


df_t=train.drop('id',axis=1)
df_t


df_s=test.drop('id',axis=1)
df_s


def fillna(data):
    for col in data.columns:
        if data[col].dtype=='object':
            data[col]=data[col].fillna(data[col].mode()[0])
        elif  data[col].dtype in['float','int']:
            data[col]=data[col].fillna(data[col].mean())
            
    return data  


fillna(df_t)


fillna(df_s)


df_t.isnull().sum()


df_s.isnull().sum()


train['Time_spent_Alone']=df_t['Time_spent_Alone'].astype(int)
test['Time_spent_Alone']=df_s['Time_spent_Alone'].astype(int)


def bar_chart(features):
    
    extrovert=df_t[df_t['Personality']=='Extrovert'][features].value_counts()
    introvert=df_t[train['Personality']=='Introvert'][features].value_counts()
    df=pd.DataFrame({'Extrovert': extrovert, 'Introvert': introvert})
    df.plot(kind='bar',stacked=False,figsize=(10,5))
    


bar_chart('Time_spent_Alone')


bar_chart('Stage_fear')


bar_chart('Social_event_attendance')


bar_chart('Friends_circle_size')


sns.boxplot(x='Personality',y='Time_spent_Alone',data=train)


sns.boxplot(x='Personality',y='Post_frequency',hue='Personality',data=train)


num_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside",
            "Friends_circle_size", "Post_frequency"]
for col in num_cols:
    plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    sns.histplot(data=df_t,x=col,kde=True,color='mediumorchid',edgecolor="black")
    plt.title(f"Distribution of {col}")


features=['Stage_fear','Drained_after_socializing','Personality']
le=LabelEncoder()
for col in features:
    df_t[col]=le.fit_transform(df_t[col])



df_t


features=['Stage_fear','Drained_after_socializing']
le=LabelEncoder()
for col in features:
    df_s[col]=le.fit_transform(df_s[col])


df_s


correlation=df_t.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation,cmap='inferno', annot=True, fmt=".2f", linewidths=0.5, linecolor='black')
plt.title('correlation',fontsize=20)
plt.xticks(rotation=90)
plt.tight_layout()


df_t.Personality.value_counts()


x=df_t.drop('Personality',axis=1)
x


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
x


xs=scaler.fit_transform(df_s)
xs


y=df_t['Personality']
y


from sklearn .model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train



y_train


x_test


y_test


from sklearn .ensemble import RandomForestClassifier,GradientBoostingClassifier
model=RandomForestClassifier(max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1)
model.fit(x_train,y_train)


y_predict=model.predict(x_test)
y_predict


acc=accuracy_score(y_test,y_predict)
acc


model2=GradientBoostingClassifier()
model2.fit(x_train,y_train)


y_predict=model2.predict(x_test)
y_predict


acc=accuracy_score(y_test,y_predict)
acc


model2=GradientBoostingClassifier()
model2.fit(x_train,y_train)


y_predict=model2.predict(x_test)
y_predict


acc=accuracy_score(y_test,y_predict)
acc


from sklearn.model_selection import StratifiedKFold,cross_val_score
kf=StratifiedKFold(n_splits=6,shuffle=True,random_state=42)


model=RandomForestClassifier(n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1)
score=cross_val_score(model,x,y,cv=kf,scoring='accuracy').mean()
score
   


model3=XGBClassifier()
score=cross_val_score(model3,x,y,cv=kf,scoring='accuracy').mean()
score


import optuna


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1)
    }
    model = XGBClassifier(**param)
    score = cross_val_score(model, x, y, cv=kf, scoring='accuracy').mean()
    return score
    
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(study.best_params)    



model = XGBClassifier()
model.fit(x_train,y_train)


y_pred=model.predict(xs)
y_pred


final=test['id']


label_map = {0: 'Extrovert', 1: 'Introvert'}

# حوّلي التوقعات الرقمية إلى نصوص
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': [label_map[pred] for pred in y_pred]  # أو y_pred
})

submission


submission.to_csv('submission.csv', index=False)




