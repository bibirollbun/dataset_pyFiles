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


import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('/kaggle/input/chocolate-rating-prediction-ai-edu/chocolate_train.csv')
data.info()


data = data.dropna(subset=['Bean Type', 'Broad Bean Origin'])
y = data['Rating']
X = data.drop(columns=['Rating'], inplace=False)
X.info()


X['Cocoa Percent']
X['Cocoa Percent'] = X['Cocoa Percent'].map(lambda x: x[:-1])
X['Cocoa Percent'] = X['Cocoa Percent'].astype(dtype='float64')


!pip install association-metrics


import association_metrics as am

XC = X.apply(
        lambda x: x.astype("category") if x.dtype == "object" else x)

cramersv = am.CramersV(XC)

cramersv.fit()


X.drop(columns=['Company Location', 'Specific Bean Origin'], inplace=True)


X[['REF', 'Review', 'Cocoa Percent']].corr()


X.drop(columns=['Review'], inplace=True)



for c in X.columns:
    print(c, len(X[X[c] == '\xa0']))


X['Bean Type'] = X['Bean Type'].replace({'\xa0' : 'Unknown'})
X['Broad Bean Origin'] = X['Broad Bean Origin'].replace({'\xa0' : 'Unknown'})
X = X.assign(Indicator1=lambda x: x['Bean Type']=='Unknown')
X['Indicator1'] = X['Indicator1'].replace({True: 1, False: 0})
X = X.assign(Indicator2=lambda x: x['Broad Bean Origin']=='Unknown')
X['Indicator2'] = X['Indicator2'].replace({True: 1, False: 0})




# Словарь соответствий (дочерние -> основное). Названия регионов
region_mapping = {
    'Venezuela': ['Ven.', 'Ven', 'Venez', 'Venezuela/ Ghana', ],
    'Dominican Republic': ['Domincan Republic', 'Dom. Rep', 'D.R.', 'Domin. Rep', 'Dom. Rep.', 'Dominican Rep.', 'DR'],
    'Costa Rica': ['Cost Rica'],
    'Ecuador': ['Ecuad.', 'Ecu.'],
    'Nicaragua': ['Nic.'],
    'Madagascar': ['Mad.', 'Mad'],
    'Brazil': ['Brasil'],
    'Mexico': ['Mex'],
    'Guatemala': ['Guat.'],
    'Sao Tome & Principe': ['Sao Tome', 'Principe'],
    'Papua New Guinea': ['PNG'],
    'Carribean': ['Carribean(DR/Jam/Tri)'],
    'Indonesia': ['Java', 'Bali'],
    'West Africa': ['Africa'],
    'Trinidad': ['Trinidad-Tobago', 'Trinidad']
}

# Функция для замены названий
def standardize_region(region):
    for main_region, aliases in region_mapping.items():
        if region in aliases:
            return main_region
    return region # Если нет в словаре, оставить как есть
X['Broad Bean Origin'] = X['Broad Bean Origin'].map(standardize_region)


from category_encoders.target_encoder import TargetEncoder

te = TargetEncoder()
te.fit(X, y)
X = te.transform(X)
X.head()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)



def data_transform(X):
    X['Cocoa Percent'] = X['Cocoa Percent'].map(lambda x: x[:-1])
    X['Cocoa Percent'] = X['Cocoa Percent'].astype(dtype='float64')
    X.drop(columns=['Company Location', 'Specific Bean Origin'], inplace=True)
    X.drop(columns=['Review'], inplace=True)
    X['Bean Type'] = X['Bean Type'].replace({'\xa0' : 'Unknown'})
    X['Broad Bean Origin'] = X['Broad Bean Origin'].replace({'\xa0' : 'Unknown'})
    X = X.assign(Indicator1=lambda x: x['Bean Type']=='Unknown')
    X['Indicator1'] = X['Indicator1'].replace({True: 1, False: 0})
    X = X.assign(Indicator2=lambda x: x['Broad Bean Origin']=='Unknown')
    X['Indicator2'] = X['Indicator2'].replace({True: 1, False: 0})
    X['Broad Bean Origin'] = X['Broad Bean Origin'].map(standardize_region)
    X = te.transform(X)
    X = scaler.transform(X)
    return X
    


!pip install catboost -q


!pip install optuna -q

import optuna
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor

def objective_cat(trial):    
    max_depth = trial.suggest_int("max_depth", 2, 16)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 2000)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
    random_strength = trial.suggest_float("random_strength", 0, 10)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1)
    score = cross_val_score(CatBoostRegressor(
    early_stopping_rounds = 50, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, l2_leaf_reg=l2_leaf_reg, random_strength=random_strength, bagging_temperature=bagging_temperature, verbose=100),
                            X, y, cv=3, scoring='r2').mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective_cat, n_trials=150)



from catboost import CatBoostRegressor
model = CatBoostRegressor(**study.best_params)
model.fit(X, y)



X = pd.read_csv('/kaggle/input/chocolate-rating-prediction-ai-edu/chocolate_test_new.csv')
X


X = data_transform(X)
pred = model.predict(X)
result = pd.DataFrame()
result['id'] = np.arange(540)
result['Rating'] = pred
result[['id','Rating']].to_csv("submission.csv", index=False)




