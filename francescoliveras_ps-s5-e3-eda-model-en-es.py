import os 
import re
import gc
import sys
import math
import time
import random
import warnings
import catboost
import datetime
import numpy as np 
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import missingno as msno
import plotly.express as px
import category_encoders as ce
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.colors as mcolors

from tqdm import tqdm

from lightgbm import early_stopping  
from IPython.display import clear_output
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_log_error 
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import FeatureHasher
from catboost import CatBoostRegressor, CatBoostClassifier, Pool


# Put theme of notebook 
from colorama import Fore, Style

# Colors
red = Fore.RED + Style.BRIGHT
mgta = Fore.MAGENTA + Style.BRIGHT
yllw = Fore.YELLOW + Style.BRIGHT
cyn = Fore.CYAN + Style.BRIGHT
blue = Fore.BLUE + Style.BRIGHT

# Reset
res = Style.RESET_ALL
plt.style.use({"figure.facecolor": "#282a36"})


# Colors
YELLOW = "#F7C53E"

CYAN_G = "#0CF7AF"
CYAB_DARK = "#11AB7C"

PURPLE = "#D826F8"
PURPLE_DARJ = "#9309AB"
PURPLE_L = "#b683d6"

BLUE = "#0C97FA"
RED = "#FA1D19"
ORANGE = "#FA9F19"
GREEN = "#0CFA58"
LIGTH_BLUE = "#01FADC"
S_BLUE = "#81c9e6"
DARK_BLUE = "#394be6"
# Palettes
PALETTE_2 = [CYAN_G, PURPLE]
PALETTE_3 = [YELLOW, CYAN_G, PURPLE]
PALETTE_4 = [YELLOW, ORANGE, PURPLE, LIGTH_BLUE]
PALETTE_5 = [PURPLE_DARJ, PURPLE_L, PURPLE, BLUE, LIGTH_BLUE]
PALETTE_6 = [BLUE, RED, ORANGE, GREEN, LIGTH_BLUE, PURPLE]

# Vaporwave palette by Francesc Oliveras
PALETTE_7 = [PURPLE_DARJ, PURPLE_L, PURPLE, BLUE, LIGTH_BLUE, DARK_BLUE, S_BLUE]
PALETTE_7_C = [PURPLE_DARJ, BLUE, PURPLE, LIGTH_BLUE, PURPLE_L, S_BLUE, DARK_BLUE]
sns.palplot(sns.color_palette(PALETTE_7))

# Set Style
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)

cmap = mcolors.LinearSegmentedColormap.from_list("", PALETTE_2)
cmap_2 = mcolors.LinearSegmentedColormap.from_list("", [S_BLUE, PURPLE_DARJ])

font_family = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=10), width=1000, height=500))

warnings.filterwarnings('ignore')


PATH = "/kaggle/input/playground-series-s5e3"
SUBMISSION_FILENAME = "sample_submission.csv"
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"

TARGET = "rainfall"

SUBMISSION_DIR = os.path.join(PATH, SUBMISSION_FILENAME)
TRAIN_DIR = os.path.join(PATH, TRAIN_FILENAME) 
TEST_DIR = os.path.join(PATH, TEST_FILENAME)
ORIGINAL_DIR = "/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv"

SEED = 250


def show_corr_heatmap(df, title):
    
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize = (15, 10))
    plt.title(title)
    # sns.heatmap(corr, annot = False, linewidths=.5, fmt=".2f", square=True, mask = mask, cmap=cmap_2)
    if df.shape[1] < 25:
        sns.heatmap(corr, annot=True, linewidths=.5, fmt=".2f", square=True, mask=mask, cmap=cmap_2)
    else:
        sns.heatmap(corr, annot=False, linewidths=.5, square=True, mask=mask, cmap=cmap_2)

    plt.show()


def data_description(df):
    print("Data description")
    print(f"Total number of records {df.shape[0]}")
    print(f'number of features {df.shape[1]}\n\n')
    columns = df.columns
    data_type = []
    
    # Get the datatype of features
    for col in df.columns:
        data_type.append(df[col].dtype)
        
    n_uni = df.nunique()
    # Number of NaN values
    n_miss = df.isna().sum()
    
    names = list(zip(columns, data_type, n_uni, n_miss))
    variable_desc = pd.DataFrame(names, columns=["Name","Type","Unique levels","Missing"])
    print(variable_desc)


def plot_cont(col, ax, color=PALETTE_7[0]):
    sns.histplot(data=comb_df, x=col,
                hue="set",ax=ax, hue_order=labels,
                common_norm=False, **histplot_hyperparams)
    
    ax_2 = ax.twinx()
    ax_2 = plot_cont_dot(
        comb_df.query('set=="train"'),
        col, TARGET, ax_2,
        color=color
    )
    
    ax_2 = plot_cont_dot(
        comb_df, col,
        TARGET, ax_2,
        color=color
    )


def show_pie_mult(dataframe, target = TARGET):
    target_counts = dataframe[target].sum()

    # Creando el grÃ¡fico de pastel con un agujero en el centro
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(target_counts, labels=target, autopct='%1.1f%%', startangle=140, colors=PALETTE_7_C)

    # Agregando un cÃ­rculo blanco en el centro para hacer un agujero
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Ajustando el aspecto para que sea un cÃ­rculo y mostrando el grÃ¡fico
    plt.title('DistribuciÃ³n de los Targets')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def show_pie_categorical(dataframe, target=TARGET):
    target_counts = dataframe[target].value_counts()

    # Creando el grÃ¡fico de pastel con un agujero en el centro
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=140, colors=[PALETTE_7_C[0],PALETTE_7_C[1],PALETTE_7_C[2],
                                                                                                                            PALETTE_7_C[3],PALETTE_7_C[4],PALETTE_7_C[5]])

    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Ajustando el aspecto para que sea un cÃ­rculo y mostrando el grÃ¡fico
    plt.title('DistribuciÃ³n de los Targets')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def show_box_plot(dataframe):
    # numerical_features_for_boxplot = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('id')
    numerical_features_for_boxplot = dataframe.select_dtypes(include=['int64', 'float64'])

    plt.figure(figsize=(20, 15))

    for i, feature in enumerate(numerical_features_for_boxplot, 1):
        plt.subplot(7, 5, i)
        sns.boxplot(y=train_df[feature], color=PALETTE_7_C[i % len(PALETTE_7_C)])
        plt.title(feature)

    plt.tight_layout()
    plt.show()



def show_hist(dataframe):
    # Filtrando las columnas numÃ©ricas para sus histogramas
    numerical_features = dataframe.select_dtypes(include=['int64', 'float64']).columns

    # Configurando el tamaÃ±o de la figura
    plt.figure(figsize=(20, 15))

    # Creando un histograma para cada caracterÃ­stica numÃ©rica
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(7, 5, i) # Ajustar segÃºn el nÃºmero de caracterÃ­sticas numÃ©ricas
        dataframe[feature].hist(bins=20, color=PALETTE_7_C[int(i%7)])
        plt.title(feature)

    plt.tight_layout()
    plt.show()


train_df = pd.read_csv(TRAIN_DIR)
test_df = pd.read_csv(TEST_DIR)
original_df = pd.read_csv(ORIGINAL_DIR)
submission_df = pd.read_csv(SUBMISSION_DIR)


data_description(train_df)
data_description(test_df)
data_description(original_df)


original_df['rainfall'] = original_df['rainfall'].map({"yes": 1, "no": 0})





msno.matrix(df=train_df, figsize=(15,5), color=(0,0.6,0.5))


numerical_variables = ['winddirection', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
categorical_variables = []

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()
original_df.columns = original_df.columns.str.strip()


display(show_corr_heatmap(train_df, "Train dataframe heatmap"))
display(show_corr_heatmap(test_df, "Test dataframe heatmap"))
display(show_corr_heatmap(original_df, "Original dataframe heatmap"))


show_hist(train_df)


show_hist(test_df)


show_hist(original_df)


numerical_variables = ['winddirection', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = []


train_df['Dataset'] = 'Train'
test_df['Dataset'] = 'Test'
original_df['Dataset'] = 'Original'

variables = [col for col in train_df.columns if col in numerical_variables]


def create_variable_plots(var):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_df, test_df,original_df.dropna()]), x=var, y="Dataset", palette=PALETTE_7_C)
    plt.xlabel(var)
    plt.title(f"Box Plot for {var}")

    plt.subplot(1, 2, 2)
    sns.histplot(data=train_df, x=var, color=PALETTE_7_C[0], kde=True, bins=30, label="Train dataframe")
    sns.histplot(data=test_df, x=var, color=PALETTE_7_C[1], kde=True, bins=30, label="Test dataframe")
    sns.histplot(data=original_df.dropna(), x=var, color=PALETTE_7_C[2], kde=True, bins=30, label="Original dataframe")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {var} [TRAIN, TEST & ORIGINAL]")
    plt.legend()

    plt.tight_layout()

    plt.show()


for var in variables:
    create_variable_plots(var)


train_df = pd.read_csv(TRAIN_DIR)
test_df = pd.read_csv(TEST_DIR)
original_df = pd.read_csv(ORIGINAL_DIR)
submission_df = pd.read_csv(SUBMISSION_DIR)

original_df[TARGET] = original_df[TARGET].map({ 'yes': 1, 'no': 0 })

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()
original_df.columns = original_df.columns.str.strip()


discrete_features = train_df[numerical_variables].dtypes == int


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(train_df[numerical_variables], train_df[target_variable])
mi_scores 


if 'id' in train_df.columns:
    train_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)


train_df = pd.concat([original_df, train_df], axis=0, ignore_index=True)
train_df['temparature'].max()


def fe(df):
    df['day'] = pd.to_datetime(df['day'])
    
    df['month'] = df['day'].dt.month
    df['day_of_week'] = df['day'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['avg_temp'] = (df['maxtemp'] + df['mintemp']) / 2
    df['temp_deviation'] = df['temparature'] - df['avg_temp']
    
    df['dew_point_depression'] = df['temparature'] - df['dewpoint']
    df['wind_dir_rad'] = np.deg2rad(df['winddirection'])
    df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
    df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
    df.drop(columns=['wind_dir_rad'], inplace=True)
    
    df['wind_chill'] = 13.12 + 0.6215 * df['temparature'] - 11.37 * (df['windspeed']**0.16) + 0.3965 * df['temparature'] * (df['windspeed']**0.16)
    df['humidity_temp'] = df['humidity'] * df['temparature']
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    df['rolling_temp_mean'] = df['avg_temp'].rolling(window=7).mean()
    df['rolling_wind_mean'] = df['windspeed'].rolling(window=7).mean()
    df['rolling_humidity_mean'] = df['humidity'].rolling(window=7).mean()
    
    df['temp_lag_1'] = df['avg_temp'].shift(1)
    df['humidity_lag_1'] = df['humidity'].shift(1)
    df['windspeed_lag_1'] = df['windspeed'].shift(1)
    
    df['pressure_temp_interaction'] = df['pressure'] * df['avg_temp']
    df['windspeed_temp_interaction'] = df['windspeed'] * df['avg_temp']
    df['sunshine_cloud_interaction'] = df['sunshine'] * df['cloud']
    df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
                                      'Summer' if 6 <= x <= 8 else
                                      'Autumn' if 9 <= x <= 11 else 'Winter')

    for c in ['pressure', 'maxtemp', 'temparature', 'humidity']:
        for gap in [1]:
            df[c+f"_shift{gap}"] = df[c].shift(gap)
            df[c+f"_diff{gap}"] = df[c].diff(gap)

    df = pd.get_dummies(df, columns=['season'], drop_first=True)
    df.drop(columns=['day'], inplace=True)


    df['temp_diff']=df['maxtemp']-df['mintemp']

    df['windspeed_product_pressure'] = df['windspeed']*df['pressure']
    
    df['sunshine_product_maxtemp'] = df['sunshine']*df['maxtemp']
    
    
    df['cloud + humidity'] = df['cloud'] + df['humidity']
    df['cloud * humidity'] = df['cloud'] * df['humidity']
    df['cloud + humidity + sunshine'] = df['cloud'] + df['humidity'] + df['sunshine']
    df['cloud * sunshine'] = df['cloud'] * df['sunshine']
    
    df['humidity * sunshine'] = df['humidity'] * df['sunshine']
    df['wci'] = (10*np.sqrt(df['windspeed']) - df['windspeed'] + 10.5) * (33-df['temparature']) 
    df['temp_fahren'] = (df['temparature']*1.8)+32 
    df['heat_index'] = -8.784+1.611*df['temparature']+2.338*df['humidity']-0.146*df['temparature']*df['humidity']-0.0123*df['temparature']**2-0.0164*df['humidity']**2+0.0022*df['temparature']**2*df['humidity']+0.0007*df['temparature']*df['humidity']**2
 
    df['wind_chill'] = 13.12 + 0.6215 * df['temparature'] - 11.37 * (df['windspeed']**0.16) + 0.3965 * df['temparature'] * (df['windspeed']**0.16)
    df['vapour_pressure'] = (df['humidity']/100)*6.105*np.exp((17.27*df['temparature'])/(237.7+df['temparature']))
    
    df['at'] = df['temparature']+0.33*df['vapour_pressure']-0.7*df['windspeed']-4.00

    df['wind_chill'] = 35.74 + 0.6215*df['temp_fahren']- 35.75*df['windspeed']**0.16 + 0.4275*df['temp_fahren']*df['windspeed']**0.16

    df.drop(['maxtemp'], axis=1, inplace=True)
    
    
    

    return df


train_df = fe(train_df)
test_df = fe(test_df)


train_df.head()


display(show_corr_heatmap(train_df, "Train dataframe heatmap"))
display(show_corr_heatmap(test_df, "Test dataframe heatmap"))


y = train_df.pop('rainfall')


from sklearn.preprocessing import StandardScaler
def scaling(df):
    ss = StandardScaler()
    return ss.fit_transform(df)


cv = StratifiedKFold(5, shuffle=True, random_state=1)
cv_splits = cv.split(train_df, y)
scores = []
test_preds = []
X_test_pool = Pool(scaling(test_df))
for i, (train_idx, val_idx) in enumerate(cv_splits):
    model = CatBoostClassifier(eval_metric='AUC')
    X_train_fold, X_val_fold = train_df.loc[train_idx], train_df.loc[val_idx]
    y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
    X_train_pool = Pool(scaling(X_train_fold), y_train_fold)
    X_valid_pool = Pool(scaling(X_val_fold), y_val_fold)
    model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=0, early_stopping_rounds=100)
    val_pred = model.predict_proba(X_valid_pool)[:, 1]
    score = roc_auc_score(y_val_fold, val_pred)
    scores.append(score)
    test_pred = model.predict_proba((X_test_pool))[:, 1]
    test_preds.append(test_pred)
    print(f'{mgta}Fold {i + 1} roc_auc_score:{res} {cyn}{score}{res}')

print(f'\n {blue}{"-"*50}{res} \n')
print(f'{red}Cross-validated ROC AUC score:{res} {cyn}{np.mean(scores):.3f} +/- {np.std(scores):.3f}{res}')
print(f'{red}Max ROC AUC score:{res} {cyn}{np.max(scores):.3f}{res}')
print(f'{red}Min ROC AUC score:{res} {cyn}{np.min(scores):.3f}{res}')


import matplotlib.pyplot as plt
import seaborn as sns
def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='feature_importance', 
        y='feature_names', 
        data=fi_df,
        palette=PALETTE_7_C  # aquÃ­ se aplica tu paleta personalizada
    )
    
    plt.title(f'{model_type} FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()


plot_feature_importance(model.get_feature_importance(), train_df.columns,'CATBOOST')


test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].median())


submission_df[TARGET] = np.mean(test_preds, axis=0)
submission_df.to_csv('submission.csv', index=False)
submission_df.head(10)

