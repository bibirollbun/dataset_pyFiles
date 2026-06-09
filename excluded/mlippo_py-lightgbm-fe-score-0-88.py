import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import optuna
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import StratifiedKFold

import plotly.io as pio
from IPython.core.display import display, HTML
from IPython.display import IFrame


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
from scipy.stats import pointbiserialr

warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
display(train.head(3))
print(train.shape)


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test.head(3)


print("!-- null values --!")

print("train:")
for cols in train.columns:
    if train[cols].isnull().sum() >= 1:
        print(f'{cols} = {train[cols].isnull().sum()}')


print("\ntest:")
for cols in test.columns:
    if test[cols].isnull().sum() >= 1:
        print(f'{cols} = {test[cols].isnull().sum()}')


test.winddirection = test.winddirection.fillna(test.winddirection.median())


train.rainfall.value_counts()


train.describe().T


day_df_train = train[['id','day']]
day_df_test = test[['id', 'day']]





from plotly.subplots import make_subplots

fig = make_subplots(rows = 1, cols = 2)


g0 = px.line(day_df_train, x = 'id', y = 'day')
g1 = px.line(day_df_test, x = 'id', y = 'day')

for trace in g0.data:
    fig.add_trace(trace, row=1, col=1)

for trace in g1.data:
    fig.add_trace(trace, row=1, col=2)
    
fig.update_layout(
    width=1000, height=500,
)

fig.update_xaxes(title_text="Train", row=1, col=1)
fig.update_xaxes(title_text="Test", row=1, col=2)

pio.write_html(fig, "fig1.html")
display(IFrame("fig1.html", width="85%", height=550))


def FE_time(df):
    df['expected_day'] = (df['id']) % 365 + 1
    df['day_mislabelled'] = df['day'] != df['expected_day']
    df['month'] = pd.cut(df['expected_day'], bins = [0,31,59,90,120,151,181,212,243,273,304,334,365], labels = [1,2,3,4,5,6,7,8,9,10,11,12], right = True)
    df['season'] = pd.cut(df['month'], bins = [1,4,7,10,13], labels = [1,2,3,4], include_lowest = True, right = False)
    return df


day_df_train = FE_time(day_df_train)
day_df_test = FE_time(day_df_test)


print(day_df_train.day_mislabelled.value_counts())


fig = px.line(day_df_train, x = 'id', y = 'expected_day', width = 500)

pio.write_html(fig, "fig2.html")
display(IFrame("fig2.html", width="45%", height=550))


train = pd.concat([day_df_train[['expected_day', 'month', 'season']], train.iloc[:, 2:]], axis = 1)
train.head(3)


id = test.id

test = pd.concat([day_df_test[['expected_day', 'month', 'season']], test.iloc[:, 2:]], axis = 1)
test.head(3)


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))  

sns.histplot(data = train, x = 'expected_day', bins = 15, hue = 'rainfall', ax = ax[0])
sns.histplot(data = train, x = 'month', bins = 15, hue = 'rainfall', ax = ax[1])
sns.histplot(data = train, x = 'season', bins = 15, hue = 'rainfall', ax = ax[2])


cols= ["pressure",	"maxtemp",	"temparature",	"mintemp",	"dewpoint",	"humidity",	"cloud",	"sunshine",	"winddirection","windspeed"]


fig = px.box(train, y = cols, log_y = True)

pio.write_html(fig, "fig3.html")
display(IFrame("fig3.html", width="100%", height=400))


def change_values(df):
    df.loc[df.sunshine == 0, "sunshine"] = 0.01
    df.loc[df.dewpoint <= 0, "dewpoint"] = 0.01
    return df


train = change_values(train)
test = change_values(test)


correlations = {}
for col in train.columns[1:-1]:  
    corr, _ = pointbiserialr(train['rainfall'], train[col])
    correlations[col] = corr
print(correlations)


plt.figure(figsize=(10, 6))
sns.barplot(x=list(correlations.values()), y=list(correlations.keys()), palette="coolwarm")

plt.xlabel("Correlation coefficient")
plt.ylabel("Vars")
plt.title("Correlations with Y variable (Rainfall)")
plt.xlim(-1, 1)  

plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)  
plt.show()


plt.figure(figsize = (10, 6))
sns.heatmap(train.corr(), annot = True)
plt.title('Correlations before Feature Enginnering');


fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(10, 30))  
j = 0
for i, col in enumerate(cols):
    sns.histplot(data=train, x=col, hue='rainfall', kde=True, ax=ax[i//2, j%2])  
    ax[i//2, j%2].set_title(col) 
    j+=1

plt.tight_layout() 
plt.show()


train_corr = train.copy()
test_corr = test.copy()


def FE(df):
    #df['temp_min_diff'] = df['temparature'] - df['mintemp']
    #df['temp_max_diff'] = df['maxtemp'] - df['temparature']
    #df['cloud_sunshine_ratio'] = df['cloud'] / df['sunshine'] # it decreased the correlation between cloud to rainfall and sunshine to rainfall
    #df['pressure_dewpoint_ratio'] = df['pressure'] / df['dewpoint'] # it decreased the correlation between presssure to rainfall and dewpoint to rainfall
    #df['pressure_dewpoint_avg'] = (df['pressure'] + df['dewpoint']) / 2
    #df['sqrt_pressure_dewpoint'] =  np.sqrt(df['pressure'] * df['dewpoint']) 
    df['temp_diff'] = df['maxtemp'] - df['mintemp']  
    df['log_pressure_dewpoint'] = np.log1p(df['pressure']) + np.log1p(df['dewpoint'])
    df['wind_cat'] = pd.cut(df['winddirection'], bins=[0, 90, 180, 270, 360], labels=['1', '2', '3', '4'], include_lowest=True)
    df['log_cloud_sunshine'] = np.log1p(df['cloud']) + np.log1p(df['sunshine'])
    return df


def DROP(df, cols = []):
    df = df.drop(columns = cols)
    return df


train_corr = FE(train_corr)
test_corr = FE(test_corr)


drop_cols = ['expected_day', 'season','temparature', 'maxtemp', 'mintemp',  'winddirection', 'pressure', 'dewpoint','cloud', 'sunshine']

train_corr = DROP(train_corr, cols = drop_cols)
test_corr = DROP(test_corr, cols = drop_cols)


plt.figure(figsize = (10, 6))
sns.heatmap(train_corr.corr(), annot = True)
plt.title('Correlations after Feature Enginnering');


train = train_corr.copy()
test = test_corr.copy()


X_train = train.drop(columns = ["rainfall"]) #LightGBM works with df directly (without converting to numpy)
y_train = train["rainfall"]

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",  
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 10, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
    }

    scores = []  

    for train_idx, val_idx in kf.split(X_train_split, y_train_split):  
        X_fold_train, X_fold_val = X_train_split.iloc[train_idx], X_train_split.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_split.iloc[train_idx], y_train_split.iloc[val_idx]

        dtrain = lgb.Dataset(X_fold_train, label=y_fold_train)
        dval = lgb.Dataset(X_fold_val, label=y_fold_val)
        
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            callbacks=[early_stopping(50), log_evaluation(500)]
        )

        probs = model.predict(X_fold_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_fold_val, probs)  
        scores.append(auc)  

    return np.mean(scores) 

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  


dtrain = lgb.Dataset(X_train_split, label=y_train_split) 
model = lgb.train(study.best_params, dtrain, valid_sets=[dtrain], callbacks=[lgb.early_stopping(50)])


y_train_pred = model.predict(X_val, num_iteration = model.best_iteration)
roc_auc_score(y_val, y_train_pred)


test.columns


test_preds = model.predict(test, num_iteration=model.best_iteration)
pred = np.clip(test_preds, 0, 1)


submission = pd.DataFrame({"id": id, "rainfall": pred})
submission.to_csv("submission.csv", index=False)


submission

