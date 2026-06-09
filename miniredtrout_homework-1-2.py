# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PolynomialFeatures,LabelEncoder, MaxAbsScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy import stats
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('/kaggle/input/st-2-ml-bootcamp/train.csv')
test = pd.read_csv('/kaggle/input/st-2-ml-bootcamp/test.csv')
sampl = pd.read_csv('/kaggle/input/st-2-ml-bootcamp/sample_submission.csv')


train.describe(include = 'object').T


train.describe(include = [np.number]).T


train.head()


nan_counts = train.isnull().sum()
nan_percentage = (train.isnull().sum()/len(train))*100
nan_sum = pd.DataFrame({'NaN_Count':nan_counts, 'NaN_Percentage':nan_percentage}).sort_values(by='NaN_Count',ascending=False)
nan_sum[nan_sum['NaN_Count'] > 0]


def pass_id(df):
    df[['PassengerGroup','PassengerId']] = df['PassengerId'].str.split('_',expand=True)
    
    return df


def cabin(df): 
    df[['Deck','Num','Side']] = df['Cabin'].str.split('/',expand=True)
    return df


def to_bool(df):
    df['VIP'] = df['VIP'].astype("boolean")
    df['CryoSleep'] = df['CryoSleep'].astype("boolean")
    
    return df


def conv(df):
    df = pass_id(df)
    df = to_bool(df)
    df = cabin(df)
    df= df.drop(columns=['Name','Cabin'])
    return df


def null_group(df):
    df['Age'] = df['Age'].fillna(int(df['Age'].mean()))
    df['Deck'] = df.groupby('PassengerGroup')['Deck'].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
    df['Side'] = df.groupby('PassengerGroup')['Side'].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
    df['Num'] = df['Num'].fillna(df['Num'].mode()[0])
    df['Destination'] = df.groupby('PassengerGroup')['Destination'].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
    df['HomePlanet'] = df.groupby('PassengerGroup')['HomePlanet'].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
    df['Num'] = df['Num'].fillna(df['Num'].mode()[0])
    df['Deck'] = df['Deck'].fillna(df['Deck'].mode()[0])
    df['Side'] = df['Side'].fillna(df['Side'].mode()[0])
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['RoomService'] = df['RoomService'].fillna(0)
    df['RoomService'] = df['RoomService'].astype(float)
    df['FoodCourt'] = df['FoodCourt'].fillna(0)
    df['FoodCourt'] = df['FoodCourt'].astype(float)
    df['ShoppingMall'] = df['ShoppingMall'].fillna(0)
    df['ShoppingMall'] = df['ShoppingMall'].astype(float)
    df['Spa'] = df['Spa'].fillna(0)
    df['Spa'] = df['Spa'].astype(float)
    df['VRDeck'] = df['VRDeck'].fillna(0)
    df['VRDeck'] = df['VRDeck'].astype(float)
    df['TotalBill'] = df['RoomService']+ df['FoodCourt']+ df['ShoppingMall']+ df['Spa']+ df['VRDeck']
    df['VIP'] = df['VIP'].fillna(False)
    df['CryoSleep'] = np.where(df['CryoSleep'].isna(),df['TotalBill'] == 0, df['CryoSleep'])
    return df


def age(df):
    df['AgeGroup'] = pd.qcut(df['Age'], q = 4, labels=['Baby','Young','Middle','Old'])
    return df


def agecount(df):
    y_m = df[df['AgeGroup'].isin(['Baby'])]
    count_by_group = y_m.groupby('PassengerGroup').size()
    df['AgeCount'] = df['PassengerGroup'].map(count_by_group).fillna(0).astype(int)
    return df


def null_out(df):
    df = null_group(df)
    df = age(df)
    df = agecount(df)
    return df


def met(df):
    df = conv(df)
    df = null_out(df)
    return df


train = met(train)
test = met(test)


train.nunique()


nan_counts = train.isnull().sum()
nan_percentage = (train.isnull().sum()/len(train))*100
nan_sum = pd.DataFrame({'NaN_Count':nan_counts, 'NaN_Percentage':nan_percentage}).sort_values(by='NaN_Count',ascending=False)
nan_sum


train.groupby('Transported')['Transported'].count()


corr = train.drop(columns=['Deck', 'Num','Side', 'HomePlanet', 'Destination','AgeGroup','PassengerGroup','PassengerId']).corr()


sns.heatmap(corr)


cat = ['HomePlanet','Destination','Deck','Side','AgeGroup','PassengerId']
fig, axes = plt.subplots(2, 3, figsize=(15, 5))
for i, col in enumerate(cat):
    k = i% 3
    j = i // 3
    value_counts = train[col].value_counts()
    sns.countplot(data=train, x=col, ax=axes[j,k])

plt.tight_layout()
plt.show()


train['Num'] = pd.cut(train['Num'].astype(int),bins=5,labels=[1,2,3,4,5])
train['PassengerId'] = pd.cut(train['PassengerId'].astype(int),bins=4,labels=[1,2,3,4])
train['PassengerGroup'] = pd.cut(train['PassengerGroup'].astype(int),bins=4,labels=[1,2,3,4])


fig, axes = plt.subplots(2, 2, figsize=(15, 5))
un_cat = ['PassengerGroup','Num']
for i, col in enumerate(un_cat):
    k = i% 2
    value_counts = train[col].value_counts()
    sns.countplot(data=train, x=col, ax=axes[0,k])
    sns.barplot(data=train,x=col,y='Transported',ax=axes[1,k])

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 5))
for i, col in enumerate(cat):
    k = i% 3
    j = i // 3
    value_counts = train[col].value_counts()
    sns.barplot(data=train, x=col,y='Transported', ax=axes[j,k])

plt.tight_layout()
plt.show()


numb = ['Age','TotalBill','AgeCount']
fig, axes = plt.subplots(2, 3, figsize=(15, 5))
for i, col in enumerate(numb):
    k = i% 3
    value_counts = train[col].value_counts()
    sns.histplot(data=train, x=col, hue='Transported', kde=True, ax=axes[0,k])
    sns.boxplot(data=train,x=col,ax=axes[1,k])
    

plt.tight_layout()
plt.show()


total_cols = ['FoodCourt','RoomService','ShoppingMall','Spa','VRDeck','TotalBill']
for col in total_cols:
    zero_count = (train[col] == 0).sum()
    percentage_zero = (zero_count/len(train[col]))*100
    mean_zero = train['Transported'][train[col] == 0].mean()
    print(col)
    print(f"Количество нулевых:{zero_count}  Процент:{percentage_zero}%  Среднее:{mean_zero}")
    print('-'*100)

    


fig, axes = plt.subplots(2, 3, figsize=(15, 5))
for i,col in enumerate(total_cols):
    new_df = train[train[col] != 0]
    k = i % 3
    j = i // 3
    axes[j,k].hist(new_df[col],bins=50,alpha=0.7)
    axes[j,k].set_title(col)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 5))
for i,col in enumerate(total_cols):
    new_df = train[train[col] != 0]
    k = i % 3
    j = i // 3
    bins = np.array_split(train.sort_values(col),10)
    means = [bin[col].mean() for bin in bins]
    variances = [bin[col].var() for bin in bins]
    axes[j,k].scatter(means,variances,alpha=0.7)
    axes[j,k].set_title(col)

plt.tight_layout()
plt.show()



def group_bool(df):
    l = []
    for col in total_cols:
        df[col],lambdaa = stats.boxcox(df[col]+1)
        l.append(lambdaa)
    return df
#train = group_bool(train)
#test = group_bool(test)


train.nunique()



def ttest(train, feature_list, target):
    ttest_results = []
    
    for feature in feature_list:
        group_0 = train[train[target] == 0][feature]
        group_1 = train[train[target] == 1][feature]
        
        t_stat, p_val = stats.ttest_ind(group_0, group_1, nan_policy='omit')
        
        ttest_results.append({'feature':feature,
                             't_stat':t_stat,
                              'p_value':p_val
                            })
    tt_df = pd.DataFrame(ttest_results)
    return tt_df.set_index('feature')


num = train.select_dtypes(include=[np.number]).columns
ttest(train,num,'Transported')


sns.pairplot(train.drop(columns=cat), height = 2 ,kind ='scatter',diag_kind='kde')



sns.catplot(x='CryoSleep', y='Transported',hue='VIP',aspect=2, data=train, kind='point')


fig,axes = plt.subplots(1,2,figsize=(9,6))
for i,col in enumerate(['VIP','CryoSleep']):
    val = train[col].value_counts()
    axes[i].pie(val.values,labels=val.index,)
    axes[i].set_title(col)
plt.legend(loc='lower right')
plt.show()



fig, axes = plt.subplots(1,2,figsize=(10,3))
cross_tab = pd.crosstab(train['VIP'],train['Transported'])
cross_tab.plot(kind='bar',ax=axes[0])
cross_tab = pd.crosstab(train['CryoSleep'],train['Transported'])
cross_tab.plot(kind='bar',ax=axes[1])





X = train.drop(columns=['Transported'])
y = train['Transported']
X['TotalBill'] = np.log10(X['TotalBill']*100 + 1)
poly = PolynomialFeatures(degree=3,include_bias=False)
le = LabelEncoder()
y = le.fit_transform(y)


prepr = ColumnTransformer(transformers=[('n',StandardScaler(),['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','TotalBill']),('b','passthrough',['CryoSleep','VIP']),('c',OrdinalEncoder(),['HomePlanet','Destination','PassengerId','PassengerGroup','Num','AgeGroup','Side','Deck','AgeCount'])])


def pipe1(model0,X_t,y_t):
    pipeline = Pipeline([('prepr',prepr),('poly',poly),('model',model0)])
    pipeline.fit(X_t,y_t)
    return pipeline

def pipe(model0,X_t,y_t):
    pipeline = make_pipeline(StandardScaler(),model0)
    pipeline.fit(X_t,y_t)
    return pipeline



test['Num'] = pd.cut(test['Num'].astype(int),bins=5,labels=[1,2,3,4,5])
test['PassengerId'] = pd.cut(test['PassengerId'].astype(int),bins=4,labels=[1,2,3,4])
test['PassengerGroup'] = pd.cut(test['PassengerGroup'].astype(int),bins=4,labels=[1,2,3,4])
test['TotalBill'] = np.log10(test['TotalBill']*100 + 1)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


model = lgb.LGBMClassifier(n_estimators=195,learning_rate=0.05,max_depth=9,num_leaves=31,reg_alpha=0.1,reg_lambda=0.1,min_child_samples=20,min_split_gain=0.001,subsample=0.8,colsample_bytree=0.8,n_jobs=-1,verbosity=-1)
pp = pipe1(model,X_train,y_train)
y_t = pp.predict(X_test)
accuracy_score(y_t,y_test)


y_pred = pipe1(model, X,y).predict(test)


y_pred = y_pred.astype("bool")
submission = pd.DataFrame({'PassengerId': list(sampl['PassengerId']),'Transported':y_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)


def feature_imp(df):
    X = df.drop('Transported', axis=1)
    y = df['Transported']  # Убрали лишние скобки
    
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # Logistic Regression
    model1 = LogisticRegression(max_iter=1000)
    pipeline_lr = pipe(model1, X_train, y_train)
    coef_lr = pipeline_lr[1].coef_[0] 
    acc_train_lr = accuracy_score(y_true=y_train, y_pred=pipeline_lr.predict(X_train))
    acc_test_lr = accuracy_score(y_true=y_test, y_pred=pipeline_lr.predict(X_test))
    label_lr = f'LR (train={acc_train_lr:.2%}, test={acc_test_lr:.2%})'
    
    # Random Forest
    model2 = RandomForestClassifier(random_state=0)
    pipeline_rf = pipe(model2, X_train, y_train)
    coef_rf = pipeline_rf[1].feature_importances_ # Правильный доступ
    acc_train_rf = accuracy_score(y_true=y_train, y_pred=pipeline_rf.predict(X_train))
    acc_test_rf = accuracy_score(y_true=y_test, y_pred=pipeline_rf.predict(X_test))
    label_rf = f'RF (train={acc_train_rf:.2%}, test={acc_test_rf:.2%})'

    # LightGBM
    model3 = lgb.LGBMClassifier(n_estimators=195,learning_rate=0.05,max_depth=9,num_leaves=31,reg_alpha=0.1,reg_lambda=0.1,min_child_samples=20,min_split_gain=0.001,subsample=0.8,colsample_bytree=0.8,n_jobs=-1,verbosity=-1,random_state=0)
    pipeline_lgb = pipe(model3, X_train, y_train)
    coef_lgb = pipeline_lgb[1].feature_importances_
    acc_train_lgb = accuracy_score(y_true=y_train, y_pred=pipeline_lgb.predict(X_train))
    acc_test_lgb = accuracy_score(y_true=y_test, y_pred=pipeline_lgb.predict(X_test))
    label_lgb = f'LGB (train={acc_train_lgb:.2%}, test={acc_test_lgb:.2%})'
    
    # Создаем DataFrame с важностями признаков
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'LR': np.abs(coef_lr),
        'RF': coef_rf,
        'LightBoost': coef_lgb
    })
    
    # Сортируем по LightGBM важности
    feature_importances = feature_importances.sort_values(by='LightBoost', ascending=False)
    
    # Масштабируем важности
    sc = MaxAbsScaler()
    feature_importances['LR_scaled'] = sc.fit_transform(feature_importances[['LR']])
    feature_importances['RF_scaled'] = sc.fit_transform(feature_importances[['RF']])
    feature_importances['LightBoost_scaled'] = sc.fit_transform(feature_importances[['LightBoost']])
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    
    # Берем топ-15 признаков для лучшей визуализации
    top_features = feature_importances.head(15)
    
    x = np.arange(len(top_features))
    width = 0.25
    
    plt.barh(x - width, top_features['LR_scaled'], height=width, label=label_lr, alpha=0.7)
    plt.barh(x, top_features['RF_scaled'], height=width, label=label_rf, alpha=0.7)
    plt.barh(x + width, top_features['LightBoost_scaled'], height=width, label=label_lgb, alpha=0.7)
    
    plt.ylabel('Features')
    plt.xlabel('Scaled Importance')
    plt.title('Feature Importance Comparison')
    plt.yticks(x, top_features['Feature'])
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Выводим таблицу с важностями
    print("Top 10 Most Important Features:")
    print(feature_importances[['Feature', 'LR', 'RF', 'LightBoost']].head(10))
    
    return ([acc_train_lr, acc_train_rf, acc_train_lgb], [acc_test_lr, acc_test_rf, acc_test_lgb])



feature_imp(train)













