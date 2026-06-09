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


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import catboost as cb
from lightgbm import LGBMClassifier
import xgboost as xgb


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()


test.head()


#null values in train dataset
# there are no null values in train dataset
(train.isnull().sum()/train.shape[0])*100


#null values in test dataset
#we have some missing data in testing dataset that we should take into account (13.6% in winddirection)
(test.isnull().sum()/test.shape[0])*100


rainfall = pd.DataFrame(train['rainfall'].value_counts()/train.shape[0]).reset_index().rename(columns = {'count':'% rainfall'})


fig , ax = plt.subplots(figsize = (10,5))
color = ['red','green']
ax.bar(x = rainfall['rainfall'].values,height =rainfall['% rainfall'].values, color = color)
ax.set_ylabel("% of rainfall days")
ax.set_xlabel("rainfall")
ax.set_xticks([0,1])
ax.set_xticklabels(['rain','no rain'])
ax.spines['top'].set_color(None)
ax.spines['right'].set_color(None)
ax.set_ylim([0,1])
ax.set_title("Count of rainfall days")
values = rainfall['% rainfall'].values[::-1]
for i, v  in enumerate(np.round(values,3)):
    ax.text(i, v +0.02, s = v,  ha = 'center')


fig, ax = plt.subplots(figsize = (15,9))
sns.heatmap(train[[col for col in train.columns if col not in ('id')]].corr(), ax = ax, annot = True ,fmt='.1%',cmap="coolwarm")
ax.set_title("Correlation between variables");


fig,ax = plt.subplots(1,2, figsize = (10,5))
sns.lineplot(x="id", y="pressure", hue="rainfall",data=train, ax = ax[0])
ax[0].set_title("Pressure vs id")
sns.lineplot(x="day", y="pressure", hue="rainfall",data=train, ax = ax[1])
ax[1].set_title("Pressure vs day")
sns.despine()


cols_to_plot = [col for col in train.columns if col not in ['id','rainfall','day']]


fig,ax = plt.subplots(5,2,figsize = (10,15))
for i in list(zip(list(itertools.product(range(5),range(2))), cols_to_plot)):
    sns.lineplot(x="id", y=i[1], hue="rainfall",data=train, ax = ax[i[0][0],i[0][1]])
    ax[i[0][0],i[0][1]].set_title(f"Rainfall vs {i[1]}")
plt.suptitle("Weather variable vs id")
plt.tight_layout();



fig,ax = plt.subplots(10,2, figsize = (10,15))
for i, col in list(zip(range(10), cols_to_plot)):
    sns.lineplot(x="id", y=col, data=train[train['rainfall'] ==0], ax = ax[i][0])
    ax[i][0].set_title(f"{col} vs no rain")
    sns.lineplot(x="id", y=col ,data=train[train['rainfall'] ==1], ax = ax[i][1])
    ax[i][1].set_title(f"{col} vs rain")
plt.tight_layout()


train[train['rainfall'] == 1].describe()[cols_to_plot]


train[train['rainfall'] == 0].describe()[cols_to_plot]


train.tail(1)


test.head(1)


fig,ax = plt.subplots(5,2,figsize = (10,15))
for i in list(zip(list(itertools.product(range(5),range(2))), cols_to_plot)):
    sns.lineplot(x="id", y=i[1], data=train, ax = ax[i[0][0],i[0][1]], label = 'train',color = 'blue')
    sns.lineplot(x="id", y=i[1], data=test, ax = ax[i[0][0],i[0][1]], label = 'test', color = 'red')
plt.suptitle("Weather variables for train and test")
plt.tight_layout();


def create_new_features(df):
    df['temp_diff1'] = df['maxtemp'] - df['mintemp']
    features = [ 'pressure', 'maxtemp', 'temparature', 'mintemp',
           'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
           'windspeed','temp_diff1']
    #shifted features
    shifts = [1,2,3,4,5]
    for f in features:
        for s in shifts:
            df[f'{f}_shift_{s}'] = df[f].shift(s)
    df['months'] = (df['id']%365) // 30 +1
    df = df.fillna(method = 'bfill', axis = 0)
    return df


train, test = create_new_features(train) , create_new_features(test)


train.head(2)


test.head(2)


models_ensamble = {
    "Random_Forest" : RandomForestClassifier(n_estimators=10000,max_depth=10, random_state=42),
    "XGBoostClassifier" : xgb.XGBClassifier(n_estimators=10000, objective="binary:logistic",eval_metric="auc",use_label_encoder=False, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=10000, learning_rate=0.05, max_depth=5, random_state=42,verbose=-1),
    "CatBoost": cb.CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=5, verbose=0, random_state=42),
    "Logistic_Regression" : LogisticRegression(penalty='l2',random_state = 42)
}


X = train[[col for col in train.columns if col not in ['rainfall','id','day']]].copy()
y = train['rainfall'].copy()


X_train,X_test,y_train, y_test = X[:1533],X[1533:], y[:1533],y[1533:] 


test_pred = test[[col for col in test.columns if col not in ['id','day']]]


assert (X.columns == test_pred.columns).all()
assert X.shape[0] == len(y)


predictions = pd.DataFrame(columns = ['Model'])
predictions_test = pd.DataFrame(columns = ['Model'])


for name, model in models_ensamble.items():
    model = Pipeline([('imputer',SimpleImputer(strategy = 'most_frequent')),('scaler',StandardScaler()),(name, model)])
    print(f"Fitting model --> {model}")
    model.fit(X_train, y_train)
    #fitting
    prob = model.predict_proba(X_test)[:,1]
    temp_results = pd.Series(name =  name, data =  prob)
    predictions = pd.concat([predictions,temp_results],axis = 1 , ignore_index = True)
    #predicting
    print(f"Predicting model {name}")
    prob_test = model.predict_proba(test_pred)[:,1]
    temp_results_test = pd.Series(name =  name, data =  prob_test)
    predictions_test = pd.concat([predictions_test,temp_results_test],axis = 1 , ignore_index = True)


predictions.drop(columns = 0, inplace = True)
predictions.columns = [i for  i in models_ensamble.keys()]
predictions['Ensemble'] = np.mean(predictions, axis = 1)


predictions


roc_auc_score(y_test.values,predictions['Ensemble'])


predictions_test['Ensemble'] = np.mean(predictions_test, axis = 1)


submission = pd.DataFrame({
    'id': test['id'],         
    'rainfall': predictions_test['Ensemble'].values    
})


assert submission.shape[0] == test.shape[0]


submission


# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
print("Submission created")

