!pip install h2o requests wbgapi tabulate future flaml ephem polars optuna plotly nbformat optuna-dashboard pyarrow > /dev/null 2>&1
print("process complete")


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ignoring the future is a great way to prepare for it!

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from datetime import date

# h2o imports
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch

# Math 
import math

# Viz imports 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#numpy 
import numpy as np 

import os
import random

# optuna imports
import optuna
from optuna.visualization import plot_optimization_history,plot_contour, plot_edf, plot_intermediate_values, plot_parallel_coordinate, plot_param_importances, plot_rank, plot_slice, plot_timeline

# ğŸ�¼ğŸ�¼ğŸ�¼ğŸ�¼
import pandas as pd 

import shap

#sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold


# scipy
from scipy import stats
from scipy.stats import anderson, norm,normaltest,skew,shapiro
from scipy.special import boxcox1p

# tf imports
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout,Input,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

#misc 
import psutil
import webcolors
from pytz import all_timezones
from difflib import get_close_matches
import calendar
import time

#xgboost imports
import xgboost as xgb
from xgboost import XGBRegressor 
import re
# regEx and df imports and load-in (Kaggle-specific)
# because who has time to name, like, 3 different files? I sure don't. 
# Way better to just auto-name each df based on like 15 lines of code! 

paths=[]
names=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        nm=(os.path.join(dirname, filename))
        paths.append(os.path.join(dirname, filename))
        newfilename= re.sub(r"\.csv$", "", filename)
        names.append(newfilename)
        print(os.path.join(dirname, filename))

dframesDict=dict(zip(names, paths))
print(dframesDict)

for key, value in dframesDict.items():
    key.strip("Path")
    globals()[key] = pd.read_csv(value)

# Among my many talents is an inability to countenance careless misspellings of column names.
train = train.rename(columns={"temparature": "temperature"})
test = test.rename(columns={"temparature": "temperature"})


#Let's get the df's basic shape and info
train.shape
trows=train.shape[0]
tcols=train.shape[1]
train.info()


test.info()

testrows=train.shape[0]
testcols=train.shape[1]


# I like to mix things up & keep them exciting, so I'm doing .sample(5) instead of . head()
train.sample(5)


# Get the numerical columns but not id, but exclude it in a fancy way that's harder than just writing a second line of code with .remove()
numcols = train.select_dtypes(include=['number']).columns.difference(['id']).tolist()
print(numcols)


# code modified from https://www.kaggle.com/code/rv1922/rainfall-prediction-eda-roc-auc
rainfall_counts = train['rainfall'].value_counts()  

# It's 2025, and emoji > words
lbls=["â˜”ï¸�","â˜€ï¸�"]

plt.figure(figsize=(6, 7))  
plt.pie(rainfall_counts, labels=lbls, autopct='%1.1f%%',  
        colors=['lightslategrey', '#F8D210'],  wedgeprops={'edgecolor': 'black'}, textprops={'fontsize':18})  

plt.title("It's Usually Pretty Rainy in SyntheticDataDelphia")  

plt.show()  


num_cols = len(numcols) 
num_rows = math.ceil(num_cols / 3)  
plt.figure(figsize=(15, num_rows * 5)) 
    
for i, col in enumerate(numcols, 1):  
    plt.subplot(num_rows, 4, i)  
    sns.histplot(train[col], bins=10, kde=True, color='#41729F', edgecolor='#274472')  
    plt.title(f"{col} Distribution")  

plt.tight_layout()  
plt.show()


#Shapiro-Wilk Normality test

fig, axes = plt.subplots(nrows=len(numcols), figsize=(6, len(numcols) * 3))
fig.subplots_adjust(wspace=0.3, hspace=0.4) 

for ax, col in zip(axes, numcols):
    data = train[col].dropna()
    shapiro_test = shapiro(train[col])
    sns.histplot(data, bins=20, kde=True, ax=ax, color="#120a8f", edgecolor="#808080")
    ax.text(1.15, 0.95, f"Shapiro-Wilk test\nAÂ² = {shapiro_test.statistic:.3f}\n"
                        f"Critical values: {shapiro_test.statistic:.3f}\n"
                        f"\np-value = {shapiro_test.pvalue:.12f}\n",
            fontsize=10, ha='left', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    if shapiro_test.pvalue < 0.05: 
        verdict = "Not Normal!" 
        color = "red"
    else:
        verdict = "Normal!"
        color = "green"
    
    ax.text(0.5, .75, verdict, fontsize=10, ha='center', color=color, transform=ax.transAxes, weight='bold')
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#plt.tight_layout(pad=2) 
plt.show()

#TODO: someday get this so the textbox aligns with the graph


fig, axes = plt.subplots(nrows=len(numcols), figsize=(6, len(numcols) * 3))
fig.subplots_adjust(wspace=0.3, hspace=0.4) 

for ax, col in zip(axes, numcols):
    data = train[col].dropna()
    
    # Anderson-Darling Test
    anderson_test = anderson(data)

    sns.histplot(data, bins=20, kde=True, ax=ax, color="mediumaquamarine", edgecolor="black")
    
    
    ax.text(1.15, 0.95, f"Anderson-Darling test\nAÂ² = {anderson_test.statistic:.3f}\n"
                        f"Critical values: {anderson_test.critical_values}\n"
                        f"Significance levels: {anderson_test.significance_level}",
            fontsize=10, ha='left', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    if anderson_test.statistic > anderson_test.critical_values[2]: 
        verdict = "Not Normal!"
        color = "red"
    else:
        verdict = "Normal!"
        color = "green"
    
    ax.text(0.5, .75, verdict, fontsize=10, ha='center', color=color, transform=ax.transAxes, weight='bold')
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  
plt.show()


rain_color = "#238BAD"  # I'm only #238BAD when it rains...
color_palette = sns.color_palette("Spectral", n_colors=len(numcols)+15)  
color_iter = iter(color_palette[::2])
boxplotcols =numcols
boxplotcols.remove("rainfall")

n_cols=2
n_rows = (len(boxplotcols) + 1)//n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(boxplotcols):
    #plt.figure(figsize=(4, 4))
    dry_color = next(color_iter) 
    sns.boxplot(x=train['rainfall'],  y=train[col], palette={0: dry_color, 1: rain_color},ax=axes[i])
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(["No Rain", "Rain"])
    axes[i].set_xlabel("") 
    axes[i].set_title(f'Rainfall Distribution vs {col}')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j]) 

plt.tight_layout()    
plt.show()
    


def corr(df):
    plt.figure(figsize=(20,20))
    if 'id' in df.columns:
        df=df.drop(columns=['id'],inplace=True) 
    numeric_cols=train.select_dtypes(include='number')
    corr_matrix = numeric_cols.corr()
    mask = np.triu(np.ones_like(corr_matrix))
    sns.heatmap(corr_matrix, vmax=.8, cmap='viridis',square=False, linewidth=.5,mask=mask)

corr(train)


# in which we Rube Goldberg the heck out of this weather by making ~in~numerable interactions  and ratios and so on
def feat_eng(df):
    df = df.copy()

    
    new_features = {}

    df['day'] = pd.to_numeric(df['day'], errors='coerce').fillna(0)
    max_day = df['day'].max()
    if max_day > 1:  
        new_features['day_sin'] = np.sin(2 * np.pi * df['day'] / max_day)
        new_features['day_cos'] = np.cos(2 * np.pi * df['day'] / max_day)
    else:
        new_features['day_sin'] = 0
        new_features['day_cos'] = 0

    # CLOUDS
    new_features['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1e-5)
    new_features['cloud_sun_interaction'] = df['cloud'] * df['sunshine']
    new_features['cloud_pressure_ratio'] = df['cloud'] / (df['pressure'] + 1e-5)
    new_features['cloud_humidity_ratio'] = df['cloud'] / (df['humidity'] + 1e-5)
    new_features['cloud_pressure_interaction'] = df['cloud'] * df['pressure']
    new_features['cloud_humidity_interaction'] = df['cloud'] * df['humidity']
    new_features['cloud_dewpoint_interaction'] = df['cloud'] * df['dewpoint']
    new_features['cloud_pressure_ratio_change_1d'] = new_features['cloud_pressure_ratio'].diff(1)
    new_features['cloud_pressure_ratio_change_3d'] = new_features['cloud_pressure_ratio'].diff(3)
    new_features['cloud_pressure_ratio_acceleration_3d'] = new_features['cloud_pressure_ratio_change_3d'].diff(1)
    new_features['cloud_pressure_ratio_cloud_humidity_interaction_interaction']= new_features['cloud_pressure_ratio']*new_features['cloud_humidity_interaction']
    new_features['cloud_temperature_interaction'] = df['cloud']*df['temperature']
    new_features['cloud_windspeed_interaction'] = df['cloud']*df['windspeed']
    new_features['cloud_windspeed_ratio'] = df['cloud']/( df['windspeed'] + 1e-5)
    new_features['cloud_windirection_ratio'] = df['cloud']/( df['winddirection'] + 1e-5)
    new_features['cloud_winddirection_interaction'] = df['cloud']*df['winddirection']


    # DEWPOINT
    new_features['dewpoint_pressure_interaction'] = df['dewpoint'] * df['pressure']
    new_features['dewpoint_humidity_ratio'] = df['dewpoint'] / (df['humidity'] + 1e-5)
    new_features['dewpoint_windspeed_ratio'] = df['dewpoint'] / (df['windspeed'] + 1e-5)
    new_features['dewpoint_sunshine_interaction']= df['dewpoint'] * df['sunshine']
    new_features['dewpoint_humidity_interaction']= df['dewpoint'] * df['humidity']
    new_features['dewpoint_windspeed_interaction']= df['dewpoint'] * df['windspeed']
    new_features['dewpoint_cloud_interaction']= df['dewpoint'] * df['cloud']
    new_features['dewpoint_cloud_ratio'] = df['dewpoint'] / (df['cloud'] + 1e-5)
    new_features['dewpoint_temperature_interaction']=df['dewpoint'] * df['temperature'] 
    
    
    # Is df['dewpoint'].min()... the don't point?! 

    # HUMIDITY
    new_features['humidity_pressure_interaction'] = df['humidity'] * df['pressure']
    new_features['humidity_sunshine_interaction'] = df['humidity'] * df['sunshine']
    new_features['humidity_trend_3d'] = df['humidity'].diff(3)
    new_features['humidity_maxtemp_interaction'] = df['humidity'] * df['maxtemp']
    new_features['humidity_temperature_interaction'] = df['humidity'] * df['temperature']
    new_features['humidity_pressure_ratio'] = df['humidity']/(df['pressure'] +1e-5)

    # PRESSURE
    new_features['pressure_trend_3d'] = df['pressure'].diff(3)
    new_features['low_pressure_flag'] = (df['pressure'] < df['pressure'].quantile(0.25)).astype(int)
    new_features['pressure_temperature_interaction'] = df['pressure'] * df['temperature']
    new_features['pressure_temperature_ratio'] = df['pressure'] /(df['temperature']+1e-5)
    new_features['pressure_wind_interaction']= df['pressure'] * df['windspeed']

    #SUNSHINE
    new_features['sunshine_trend_3d'] = df['sunshine'].diff(3)
    new_features['sunshine_pressure_interaction'] = df['sunshine'] * df['pressure']
    new_features['sunshine_humidity_interaction'] = df['sunshine'] * df['humidity']
    new_features['sunshine_dewpoint_ratio'] = df['sunshine'] /( df['dewpoint'] + 1e-5)

    # TEMPERATURE
    
    new_features['temp_variance'] = abs(df['maxtemp'] - df['mintemp'])
    new_features['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    new_features['temp_sunshine_interaction'] = df['temperature'] * df['sunshine']
    new_features['temp_sunshine_ratio'] = df['temperature'] / (df['sunshine'] + 1e-5)
    new_features['temp_windspeed_ratio'] = df['temperature'] / (df['windspeed'] + 1e-5)
    new_features['temp_windspeed_interaction'] = df['temperature'] * df['windspeed']
    new_features['tempyesterday'] = df['temperature'].shift(1)
    new_features['temp_cloud_humidity_ratio_interaction']=new_features['cloud_humidity_ratio']*df['temperature']
    new_features['temp_cloud_sun_ratio_interaction']=new_features['cloud_sun_ratio']*df['temperature']
    #new_features['temp_cloud_humidity_interaction_interaction']=df['temperature']*new_features['cloud_humidity_interaction']
    
    # WIND
    new_features['wind_sin'] = np.sin(np.deg2rad(df['winddirection'].fillna(df['winddirection'].median())))
    new_features['wind_cos'] = np.cos(np.deg2rad(df['winddirection'].fillna(df['winddirection'].median())))
    new_features['windyesterday'] = df['windspeed'].shift(1)
    
    new_features['windspeed_change_3d'] = df['windspeed'].diff(3)
    new_features['windspeed_cloud_interaction'] = df['windspeed'] * df['cloud']
    new_features['windspeed_cloud_ratio'] = df['windspeed'] / (df['cloud'] + 1e-5)
    new_features['windspeed_windirection_interaction']= df['windspeed']*df['winddirection']
    new_features['windspeed_windirection_ratio']= df['windspeed']/(df['winddirection'] +1e-5)


    # ROLLING WINDOWS
    rolling_windows = [3, 7, 14]
    for window in rolling_windows:
        new_features[f'cloud_rolling_{window}d'] = df['cloud'].rolling(window, min_periods=1).mean()
        new_features[f'max_temp_rolling_{window}d'] = df['maxtemp'].rolling(window, min_periods=1).mean()
        new_features[f'min_temp_rolling_{window}d'] = df['mintemp'].rolling(window, min_periods=1).mean()
        new_features[f'pressure_rolling_{window}d'] = df['pressure'].rolling(window, min_periods=1).mean()
        new_features[f'humidity_rolling_{window}d'] = df['humidity'].rolling(window, min_periods=1).mean()
        new_features[f'windspeed_rolling_{window}d'] = df['windspeed'].rolling(window, min_periods=1).mean()
        new_features[f'temperature_rolling_{window}d'] = df['temperature'].rolling(window, min_periods=1).mean()
        new_features[f'dewpoint_rolling_{window}d'] = df['dewpoint'].rolling(window, min_periods=1).mean()
        new_features[f'sunshine_rolling_{window}d'] = df['sunshine'].rolling(window, min_periods=1).mean()

    new_features['pressure_acceleration'] = new_features['pressure_rolling_3d'].diff().fillna(0)
  
    # EXTREME WEATHER FLAGS
    for feature in ['cloud', 'humidity','pressure', 'sunshine', 'windspeed', 'winddirection', 'dewpoint', 'maxtemp', 'mintemp', 'temperature']:
        new_features[f'extreme_{feature}'] = ((df[feature] > df[feature].quantile(0.95)) |
                                              (df[feature] < df[feature].quantile(0.05))).astype(int)

    # Convert dictionary to DataFrame and concatenate in one step
    new_features_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)

    # Clean up NaN values
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


train = feat_eng(train)  
test = feat_eng(test)

# Get the numerical columns but not id, but exclude it in a fancy way that's harder than just writing a second line of code with .remove()
numcols = train.select_dtypes(include=['number']).columns.difference(['id']).tolist()
print("This phase of data torture is now complete. Features have been engineered.")




def moreFE(df):
    df['temp_variance_dewpoint_interaction'] = df['temp_variance'] * df['dewpoint']
    print("I've added the temp variance dewpoint interaction.")
    return df

moreFE(train)
moreFE(test)


def corr(df):
    plt.figure(figsize=(20,20))
    if 'id' in df.columns:
        df=df.drop(columns=['id'],inplace=True) 
    numeric_cols=train.select_dtypes(include='number')
    corr_matrix = numeric_cols.corr()
    mask = np.triu(np.ones_like(corr_matrix))
    sns.heatmap(corr_matrix, vmax=.8, cmap='viridis',square=False, linewidth=.5,mask=mask)

corr(train)


#print(train.columns)
# Get the numerical columns but not id, but exclude it in a fancy way that's way more complicated than just writing a second line of code with .remove()
numcols = train.select_dtypes(include=['number']).columns.difference(['id']).tolist()
print(numcols)


predictors = [col for col in train.columns if col != 'rainfall']
if 'id' in predictors:
    predictors.remove('id')

xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.08, random_state=42,enable_categorical=True)
xgb_model.fit(train[predictors], train['rainfall'])

feature_importances = pd.DataFrame({'feature': predictors, 'importance': xgb_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

top_features = feature_importances.head(50)

plt.figure(figsize=(12, 9))
sns.barplot(x=top_features['importance'], y=top_features['feature'], palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Top 50 Most Important Features")
plt.show()



feats=feature_importances.shape[0]
# Feats of strength for a rainy Festivus perhaps



N_FEATURES = feats 

# Get top features dynamically
top_features = feature_importances.head(N_FEATURES)['feature'].tolist()
# Apply feature selection to train & test data
train = train[top_features + ['rainfall']]
test = test[top_features]
dropped = feats-N_FEATURES

print(f"âœ… Retained {len(top_features)} best features based on importance! Dropped {dropped} features!")
feature_importances.head(N_FEATURES)

predictors = [col for col in train.columns if col != 'rainfall']
if 'id' in predictors:
    predictors.remove('id')

X_train=train[predictors]
y_train=train['rainfall']

estimator = RandomForestClassifier(random_state=42)
rfecv = RFECV(estimator, step=1, cv=StratifiedKFold(15, shuffle=True, random_state=42),
              scoring='roc_auc')
rfecv.fit(X_train, y_train)

# X_train_selected now holds the features chosen by RFECV
X_train_selected = rfecv.transform(X_train)
print("Optimal number of features : %d" % rfecv.n_features_)



numTrials=250
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 300), 
        "max_depth": trial.suggest_int("max_depth", 4, 8),  
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.18), 
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),  
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 8), 
        "gamma": trial.suggest_float("gamma", 0.01, 0.6), 
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
        "enable_categorical": True,
        "random_state": 42
    }

    X_train, X_val, y_train, y_val = train_test_split(
        train.drop(columns=['rainfall']), train['rainfall'], test_size=0.25, random_state=42
    )

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    return auc

# Disable annoying logging
optuna.logging.set_verbosity(optuna.logging.ERROR)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=numTrials) 
plot_optimization_history(study).show()
#plot_intermediate_values(study)
#plot_parallel_coordinate(study)
#plot_parallel_coordinate(study, params=["lr", "n_layers"])
#plot_contour(study).show()
plot_param_importances(study).show()
print("\nğŸ”¹ Best Hyperparameters Found:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")
print(f"\nâœ… Best AUC Score: {study.best_value:.5f}")
bayesianAUC=study.best_value


xstudy = optuna.create_study(direction="maximize") 
xstudy.optimize(objective, n_trials=numTrials)  
print("\nğŸ”¹ Best Hyperparameters Found:")
for key, value in xstudy.best_params.items():
    print(f"{key}: {value}")

print(f"\nâœ… Best AUC Score: {xstudy.best_value:.5f}")
plot_optimization_history(xstudy)

df_trials = xstudy.trials_dataframe()
df_trials = df_trials.sort_values(by="value", ascending=False) 

#print("\nğŸ”¹ Top 5 Trials:")
#df_trials.head(5)

print("Best hyperparameters:", xstudy.best_params)
nonBayesianAUC=xstudy.best_value
plot_optimization_history(xstudy).show()



notBayesian=xstudy.best_value
bysian=study.best_value
if nonBayesianAUC > bayesianAUC:
    best_params = xstudy.best_params
    print(f"No Bayesian today; {notBayesian}")
else:
    best_params = study.best_params
    print(f"yes Bayesian today: {bysian}")

best_params["enable_categorical"] = True  
best_params["random_state"] = 42  

final_xgb = xgb.XGBClassifier(**best_params)
final_xgb.fit(train.drop(columns=['rainfall']), train['rainfall'])

print("âœ… The Final XGBoost model has been trained dynamically with Optuna's best parameters!")
print("It is the almost literal champagne of feature selection for figuring out if it will rain or not")



explainer = shap.Explainer(final_xgb, train.drop(columns=['rainfall']))
shap_values = explainer(test)

shap.summary_plot(shap_values, test)
#whoa, shocking, the feature that just agglomerates a bunch of other predictive features had a big impact on the model whoa amazing



test = test[train.drop(columns=['rainfall']).columns]  

sample_submission['rainfall'] = final_xgb.predict_proba(test)[:, 1]  

# Save model
final_xgb.save_model("best_xgb_model_viaOptuna331.json")

# Save final submission (using sample_submission as template)
sample_submission.to_csv('3_31_optunaXGboostV3.csv', index=False)

print("âœ…âœ… Submission successfully saved!")
sample_submission.head(5)



# Initial values
min_mem_size = 7
run_time = 3600
print(f"OK, we got {min_mem_size} for minmemsize and {run_time} for the max runtime")

pct_memory = 0.7
available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  
min_mem_size = round(pct_memory * available_memory_gb)

print(f"Adjusted minMemSize is {min_mem_size}")

port_no = random.randint(5555, 55555)

try:
    h2o.init(
        strict_version_check=False, min_mem_size_GB=min_mem_size, port=port_no
    )  # start h2o
except:
    logging.critical("h2o.init")
    h2o.download_all_logs(dirname=logs_path, filename=logfile)
    h2o.cluster().shutdown()
    sys.exit(2)


#  ğŸ�¼ -->  ğŸ’§ Frames
h2o_train = h2o.H2OFrame(train)
h2o_test = h2o.H2OFrame(test)

#  ğŸ�¯ column --> ğŸ�ˆâ€�â¬›
h2o_train['rainfall'] = h2o_train['rainfall'].asfactor()
print("ğŸ�¼ -->  ğŸ’§ Frames complete")


# Does the same thing, but as a function. ooh, ahh. 
def pandaWater(train, test, target):
    h2o_train = h2o.H2OFrame(train)
    h2o_test = h2o.H2OFrame(test)

    #  ğŸ�¯ column --> ğŸ�ˆâ€�â¬›
    h2o_train[target] = h2o_train[target].asfactor()
    print("ğŸ�¼ -->  ğŸ’§ Frames complete")
    return h2o_train, h2o_test


# âœ… Train H2O  AutoML
aml = H2OAutoML(
    max_runtime_secs=run_time,  
    max_models=120,  
    stopping_metric="AUC",  
    nfolds=15,  
    exclude_algos=["DeepLearning"],  
    seed=42
)
aml.train(x=features, y="rainfall", training_frame=h2o_train)

# âœ… Now get the best model from AutoML
best_aml_model = aml.leader
print("ğŸ�†Best AutoML Model:", best_aml_model.model_id)

# âœ… Optunize if and only if XGboost was selected by AutoML
if "XGBoost" in best_aml_model.model_id:
    print("ğŸ�¯ AutoML selected XGBoost! Running Optuna for hyperparameter tuning...")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "enable_categorical": True,  
            "random_state": 42
        }

        # âœ… train / test split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train.drop(columns=['rainfall']), train['rainfall'], test_size=0.2, random_state=42
        ) #real missed opportunity to call it 'trainfall'

        # âœ… Train XGBoost with suggested hyperparameters
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # âœ… Compute AUC on validation set
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)  # Maximize AUC

    # âœ… Optunize! 
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=numTrials)  

    # âœ… Train final XGBoost model with best Optuna parameters
    best_params = study.best_params  
    best_params["enable_categorical"] = True  
    best_params["random_state"] = 42  
    final_xgb = xgb.XGBClassifier(**best_params)
    final_xgb.fit(train.drop(columns=['rainfall']), train['rainfall'])

    print("âœ… Final XGBoost model has been trained with Optunaâ€™s best parameters!")

# âœ… Get Predictions from the best model (whether it was AutoML or Optuna XGBoost)
if "XGBoost" in best_aml_model.model_id:
    predictions = final_xgb.predict_proba(test)[:, 1]  
else:
    predictions = best_aml_model.predict(h2o_test).as_data_frame()["p1"]

# âœ… Use sample_submission as a template for the final output
sample_submission['rainfall'] = predictions
sample_submission.to_csv('331_submissionAMLxgV3.csv', index=False)

print("âœ… Submission successfully saved with optimized model selection!")


best_aml_model.explain(h2o_train)


run_time=5400
# Convert target column to categorical for classification
#h2o_train['rainfall'] = h2o_train['rainfall'].asfactor()

# âœ… Train H2O AutoML
aml = H2OAutoML(
    max_runtime_secs=run_time,  
    max_models=250,  
    stopping_metric="AUC",  
    nfolds=15,  
    exclude_algos=["DeepLearning"],  
    seed=42
)
aml.train(x=[col for col in train.columns if col != 'rainfall'], y="rainfall", training_frame=h2o_train)

#  Get the best model from AutoML
best_aml_model = aml.leader
model_name = best_aml_model.model_id
print(f"ğŸ�† Best AutoML Model: {model_name}")

#  Determine Model Type and Set Hyperparameter Space for Optuna
def get_hyperparameter_space(model_type, trial):
    if model_type == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "enable_categorical": True,
            "random_state": 42
        }
    elif model_type == "LightGBM":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 20, 350),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42
        }
    elif model_type == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 3, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength": trial.suggest_float("random_strength", 0, 1),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_state": 42
        }
    else:
        return None

#  If AutoML selects a model that can be Optunized, do it
supported_models = ["XGBoost", "LightGBM", "CatBoost"]
selected_model = None

for model in supported_models:
    if model in model_name:
        selected_model = model
        break

if selected_model:
    print(f"ğŸ�¯ I\"m AutoML, and I selected {selected_model}! Running Optuna for some high-end hyperparameter tuning now...Op-tune in later for the results!")

    def objective(trial):
        params = get_hyperparameter_space(selected_model, trial)

        if params is None:
            return float('-inf')  

        #  Ye Olde Train Test Split! 
        X_train, X_val, y_train, y_val = train_test_split(
            train.drop(columns=['rainfall']), train['rainfall'], test_size=0.2, random_state=42
        )

        #  Train the selected model with the suggested hyperparameters
        if selected_model == "XGBoost":
            model = xgb.XGBClassifier(**params)
        elif selected_model == "LightGBM":
            model = lgb.LGBMClassifier(**params)
        elif selected_model == "CatBoost":
            model = cb.CatBoostClassifier(**params, verbose=0)

        model.fit(X_train, y_train)

        #  Compute AUC for the validation set
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred) 

    # Optunize!! 
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=numTrials)

    #  Now train the final model using the best Optuna parameters
    best_params = study.best_params
    best_params["random_state"] = 42

    if selected_model == "XGBoost":
        final_model = xgb.XGBClassifier(**best_params)
    elif selected_model == "LightGBM":
        final_model = lgb.LGBMClassifier(**best_params)
    elif selected_model == "CatBoost":
        final_model = cb.CatBoostClassifier(**best_params, verbose=0)

    final_model.fit(train.drop(columns=['rainfall']), train['rainfall'])

    print(f"âœ… Final {selected_model} model trained with Optunaâ€™s best parameters!")

#  Use the BEST model for predictions ! 
if selected_model:
    predictions = final_model.predict_proba(test)[:, 1]  # Use Optuna-Tuned Model
else:
    predictions = best_aml_model.predict(h2o_test).as_data_frame()["p1"]  # Use AutoML Best Model

#  now put it in sample formatting 

sample_submission['rainfall'] = predictions
sample_submission.to_csv('f_submission_optimized_sink329_V1.csv', index=False)

print("âœ… Kitchen Sink successfully saved with optimized model selection!")
sample_submission.head(5)




explain_model = aml.explain(frame=h2o_train, figsize=(10,8), exclude_explanations=['pdp', 'ice'])
explain_model

