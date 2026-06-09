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


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier


import warnings
warnings.filterwarnings("ignore")


df  = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
orginal = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


df.head()


orginal.columns = [col.strip() for col in orginal.columns]


orginal['rainfall'] = orginal['rainfall'].map({"yes":1, 'no':0})


df = pd.concat([df, orginal], axis='index')


df.head()


numerical_cols = df.select_dtypes(include='number').columns.difference(['id'])

num_cols = 3  # Number of columns in the grid
num_rows = (len(numerical_cols) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], ax=axes[i], kde=True)
    axes[i].set_title(col)

    ax_box = axes[i].inset_axes([0.2, -0.3, 0.6, 0.2])  # [x, y, width, height]
    sns.boxplot(x=df[col], ax=ax_box, orient='h')
    ax_box.set(xlabel='')
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.show()



corr_matrix = df.corr()

# Affichage avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matrice de Corrélation")
plt.show()


plt.scatter(df['temparature'], df['maxtemp'])


plt.hist(df['rainfall'])


def create_feature(df):
    df = df.copy()
    df['temparature_norm'] = (df["maxtemp"] - df['temparature'] )/(df['maxtemp'] + df['mintemp'])
    df['humidity_temerature'] =  df['humidity'] /(df['pressure']+df['temparature'])
    df['temparature_diff'] =  df['maxtemp'] -  df['mintemp']
    df['cloud_per_humidity'] =  df['cloud'] / df['humidity']
    df['humidity_per_windspeed'] = df['humidity']/df['windspeed']
    df['winddirection_windspeed'] = df['windspeed']/df['winddirection']
    df['sunshine_maxtemp'] = df['maxtemp'] * df["sunshine"] 
    df['sunshine_humidity'] = df['humidity'] * df['sunshine']
    df[['cloud', 'dewpoint', 'humidity', 'mintemp', 'temparature']] = np.log1p(df[['cloud', 'dewpoint', 'humidity', 'mintemp', 'temparature']])
    return df

def date_features(data, date_column="date", startedDate = 2020):
    df  = data.copy()
    df['year'] = (df.index / 365).astype(int)
    df['date'] = pd.to_datetime((startedDate + df["year"]).astype(str) + df['day'].astype(str), format='%Y%j')
    df.set_index(df['date'],inplace=True)
    df['day'] = df[date_column].dt.day.astype('int')
    df['day_of_year'] = df[date_column].dt.dayofyear.astype('int')
    df['week_of_year'] = df[date_column].dt.isocalendar().week.astype('int')
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df.drop(['date', 'week_of_year'], axis=1, inplace=True)
    return df




df.isna().sum()


winddirection_mean = df['winddirection'].mean()
windspeed_mean =  df["windspeed"].mean()


df['winddirection'] = df['winddirection'].fillna(winddirection_mean)
df['windspeed'] = df['windspeed'].fillna(windspeed_mean)


df = date_features(df)
df = create_feature(df)


X = df.drop(['rainfall', 'id'], axis='columns')
y =  df.rainfall


encoder  = StandardScaler()
X = pd.DataFrame(encoder.fit_transform(X), columns=X.columns)


import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

feature_importances =[]
def objective(trial):
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",   
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "verbose":-1
    }
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = LGBMClassifier(**params)
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict_proba(X_test)[:, 1]
        
        # Calcul du score
        score = roc_auc_score((y_test), (y_pred))
        print(score)
        scores.append(score)
        feature_importances.append(model.feature_importances_)
        return score
    # Afficher les résultats
    print(f"Scores pour chaque fold : {scores}")
    print(f"Score moyen : {np.mean(scores):.4f}±{np.std(scores)}")
    return np.mean(scores) #- np.std(scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_value)
print(study.best_params)


LGBM_params  = {'learning_rate': 0.011618941678254543, 'n_estimators': 359, 'max_depth': 4, 'num_leaves': 73, 'min_child_samples': 41, 'subsample': 0.5384456313547902, 'colsample_bytree': 0.7804990765825439, 'reg_alpha': 0.5211975435604865, 'reg_lambda': 0.08300529091803749, 'verbose':-1}
XGB_params = {'max_depth': 10, 'learning_rate': 0.12524062640471165, 'n_estimators': 848, 'min_child_weight': 8.912237368631079, 'gamma': 4.5731207793171045, 'subsample': 0.968857229906754, 'colsample_bytree': 0.6665914820989893, 'lambda': 5.541807139214233, 'alpha': 2.1391455345529167}
LGBM_params2  =  {'learning_rate': 0.012711415541291425, 'n_estimators': 7016, 'max_depth': 4, 'num_leaves': 229, 'min_child_samples': 43, 'subsample': 0.849160372450066, 'colsample_bytree': 0.8476726247823237, 'reg_alpha': 9.77630012223171, 'reg_lambda': 9.470277087238564, "verbose":-1}
LGBM_params3= {'learning_rate': 0.010248900399532015, 'n_estimators': 197, 'max_depth': 9, 'num_leaves': 126, 'min_child_samples': 43, 'subsample': 0.9156699930157592, 'colsample_bytree': 0.8040039189341261, 'reg_alpha': 0.2076660934643052, 'reg_lambda': 3.5567756201744616}
RF_params =  {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 300}


test_predict = test.drop('id', axis='columns')


test_predict


test_predict.isna().sum()


test_predict['winddirection']=test_predict['winddirection'].fillna(winddirection_mean)


test_predict = date_features(test_predict, startedDate=2025)
test_predict= create_feature(test_predict)


test_predict = encoder.transform(test_predict)


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
prediction =[]
# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
model = VotingClassifier(estimators=[
                                        ('LGBMClassifier',  LGBMClassifier(**LGBM_params3)), 
                                        ('RandomForestClassifier',RandomForestClassifier(**RF_params) ),
                                        ],
                                        voting='soft')
    # model = LGBMClassifier(**LGBM_params3)
        # Entraînement
model.fit(X_train, y_train)
    
    # Prédiction
y_pred = model.predict_proba(X_test)[:, 1]
    
    # Calcul du score
    # score = roc_auc_score((y_test), (y_pred))
    # print(score)
    # scores.append(score)
prediction.append(model.predict_proba(test_predict)[:, 1])
print(scores)
print(f"{np.mean(scores)} ± {np.std(scores)}" )


submission =  pd.DataFrame({"id":test.id, "rainfall": np.mean(prediction, axis=0)})


submission.head()


submission.to_csv('submission.csv',index=False)

