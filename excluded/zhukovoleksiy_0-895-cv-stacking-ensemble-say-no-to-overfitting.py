!pip install cmaes


# Misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from copy import deepcopy
from functools import partial
import gc
import warnings
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing

# Import libraries for Hypertuning
import optuna

# Import libraries for gradient boosting
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool


# Seaborn
rc = {
    #FAEEE9
    "axes.facecolor": "#243139",
    "figure.facecolor": "#243139",
    "axes.edgecolor": "#000000",
    "grid.color": "#000000",
    "font.family": "arial",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#FFFFFF",
    "ytick.color": "#FFFFFF",
    "grid.alpha": 0.4
}
sns.set(rc=rc)

# Useful line of code to set the display option so we could see all the columns in pd dataframe
pd.set_option('display.max_columns', None)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Functions
def print_sl():
    print("=" * 50)
    print()


# Load Data
train_PATH    = '/kaggle/input/playground-series-s5e3/train.csv'
test_PATH     = '/kaggle/input/playground-series-s5e3/test.csv'
sub_PATH      = '/kaggle/input/playground-series-s5e3/sample_submission.csv'

train_ex_PATH = '/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv'

train_df      = pd.read_csv(train_PATH)
train_orig_df = pd.read_csv(train_ex_PATH)
test_df       = pd.read_csv(test_PATH)
sub_df        = pd.read_csv(sub_PATH)

# Encode 'rainfall' feature for original data
train_orig_df['rainfall'] = train_orig_df['rainfall'].map({'yes': 1, 'no': 0})

train_df.drop('id',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)
# train_orig_df.drop('id',axis=1,inplace=True)

print('Data Loaded Succesfully!')
print_sl()

# Fast Data Check
print(f'Train Data Shape: {train_df.shape}')
print(f'Are there any null values in train? - {train_df.isnull().any().any()}\n')

print(f'Train Orig Data Shape: {train_orig_df.shape}')
print(f'Are there any null values in train? - {train_orig_df.isnull().any().any()}\n')

print(f'Test Data Shape:  {test_df.shape}')
print(f'Are there any null values in test? - {test_df.isnull().any().any()}\n')
print_sl()

# Traget
target = 'rainfall'

train_df.head()


train_orig_df.columns = train_orig_df.columns.str.strip()
train_orig_df['windspeed'].fillna(train_orig_df['windspeed'].mean(), inplace=True)
train_orig_df['winddirection'].fillna(train_orig_df['winddirection'].mean(), inplace=True)
test_df['winddirection'].fillna(test_df['winddirection'].mean(), inplace=True)

# Merge with original data:
train_df = pd.concat([train_df, train_orig_df], ignore_index=True)


def plot_count(df: pd.core.frame.DataFrame, col: str, title_name: str='Train') -> None:
    # Set background color
    f, ax = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.2)

    s1 = df[col].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    colors = sns.color_palette("mako")
    # hex_colors = [matplotlib.colors.to_hex(color) for color in colors]
    # print(hex_colors)
    
    outer_colors = ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']
    inner_colors = ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']
    #inner_colors = ['#59b3a3',] #'#433C64']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90, frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={'linewidth' : 1, 'edgecolor' : 'black'}, 
        textprops={'fontsize': 12, 'weight': 'bold', 'color': 'white'}
    )

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='#243139', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette=colors, orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,       
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='white', fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col, fontweight="bold", color='white')
    ax[1].set_ylabel('count', fontweight="bold", color='white')

    f.suptitle(f'{title_name}', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout() 
    plt.show()

plot_count(train_df, 'rainfall', 'Target Distribution of Data')


## PRESSURE

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['pressure'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Pressure in train_df', color='white')
axes[0].set_xlabel('Pressure')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['pressure'], ax=axes[1])
axes[1].set_title('Box plot of Pressure in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


## MAXTEMP

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['maxtemp'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Max Temparature in train_df', color='white')
axes[0].set_xlabel('Temp')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['maxtemp'], ax=axes[1])
axes[1].set_title('Box plot of Max Temparature in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


## TEMPARATURE

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['temparature'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Temparature in train_df', color='white')
axes[0].set_xlabel('Temp')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['pressure'], ax=axes[1])
axes[1].set_title('Box plot of Temparature in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


## MINTEMP

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['mintemp'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Min Temparature in train_df', color='white')
axes[0].set_xlabel('Temp')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['pressure'], ax=axes[1])
axes[1].set_title('Box plot of Min Temparature in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


## DEWPOINT

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['dewpoint'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Dewpoint in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['pressure'], ax=axes[1])
axes[1].set_title('Box plot of Dewpoint in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


## HUMIDITY

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['humidity'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Humidity in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['pressure'], ax=axes[1])
axes[1].set_title('Box plot of Humidity in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


## CLOUD

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['cloud'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Cloud in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['pressure'], ax=axes[1])
axes[1].set_title('Box plot of Cloud in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


## SUNSHINE

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['sunshine'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Sunshine in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['pressure'], ax=axes[1])
axes[1].set_title('Box plot of Sunshine in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


## WINDDIRECTION

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['winddirection'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Winddirection in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['pressure'], ax=axes[1])
axes[1].set_title('Box plot of Winddirection in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


## WINDSPEED

# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['windspeed'], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Windspeed in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['pressure'], ax=axes[1])
axes[1].set_title('Box plot of Windspeed in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Create a copy of the dataframe
df = train_df.copy()

def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str = 'Train correlation') -> None:
    excluded_columns = ['id']
    columns_without_excluded = [col for col in df.columns if col not in excluded_columns]
    corr = df[columns_without_excluded].corr()
    
    fig, axes = plt.subplots(figsize=(14, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='mako', annot=True, annot_kws={"size": 6})
    plt.title(title_name, color='white')
    plt.show()

# Plot correlation heatmap for encoded dataframe
plot_correlation_heatmap(df, 'Dataset Correlation')


def feature_engineering(df):
    # Existing Features
    df["temp_range"] = df["maxtemp"] - df["mintemp"]
    df["dew_point_depression"] = df["temparature"] - df["dewpoint"]
    df["humidity_pressure_ratio"] = df["humidity"] / df["pressure"]
    df["saturation_deficit"] = 100 - df["humidity"]
    
    # Wind Features
    df["wind_u"] = df["windspeed"] * np.cos(np.radians(df["winddirection"]))
    df["wind_v"] = df["windspeed"] * np.sin(np.radians(df["winddirection"]))

    # Strongest Correlated Features
    df["relative_sunshine"] = df["sunshine"] / (100 - df["cloud"] + 1e-5)
    df["humidity_cloud_interaction"] = (df["humidity"] * df["cloud"]) / 10000
    df["inv_humidity_cloud"] = 100 - df["humidity"] - df["cloud"]
    df["sunshine_ratio"] = df["sunshine"] / (df["cloud"] + df["humidity"] + 1e-5)

    # Handling Time Features (Assuming 'day' represents sequential days)
    #df["month"] = ((df["day"] - 1) // 30) % 12 + 1  # Approximate month
    df["date"] = pd.to_datetime(df["day"], format="%j")  # Converts day-of-year to date
    df["month"] = df["date"].dt.month  # Extracts the month
    df.drop(columns=["date"], inplace=True) 
    
    # df["season"] = df["month"].map({
    #     12: "Winter", 1: "Winter", 2: "Winter", 
    #     3: "Spring", 4: "Spring", 5: "Spring",
    #     6: "Summer", 7: "Summer", 8: "Summer",
    #     9: "Autumn", 10: "Autumn", 11: "Autumn"
    # })
    # df["day_of_week"] = df["day"] % 7  # Approximate day of the week
    # df["is_weekend"] = df["day_of_week"].isin([6, 0]).astype(int)  # 0=Sunday, 6=Saturday

    return df


# Apply to Train and Test Data
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

train_df.head()


# le = LabelEncoder()
# train_df["season"] = le.fit_transform(train_df["season"])
# test_df["season"] = le.transform(test_df["season"])

X_train = train_df.drop(columns=[target]).reset_index().drop(columns=['index'])
y_train = train_df.rainfall.astype(int).reset_index().drop(columns=['index'])
X_test = test_df.reset_index().drop(columns=['index'])

def scale(x):
    scaler = preprocessing.RobustScaler()
    robust_df = scaler.fit_transform(x)
    robust_df = pd.DataFrame(robust_df, columns =x.columns)
    return robust_df

num_cols = [
 'cloud',
 'dew_point_depression',
 'dewpoint',
 'humidity',
 'humidity_cloud_interaction',
 'humidity_pressure_ratio',
 'inv_humidity_cloud',
 'maxtemp',
 'mintemp',
 'pressure',
 'relative_sunshine',
 'saturation_deficit',
 'sunshine',
 'sunshine_ratio',
 'temp_range',
 'temparature',
 'wind_u',
 'wind_v',
 'winddirection',
 'windspeed']

# X_train[num_cols] = scale(X_train[num_cols])
# X_test[num_cols] = scale(X_test[num_cols])
# X_train = scale(X_train)
# X_test = scale(X_test)

X_train.drop('day',axis=1,inplace=True)
X_test.drop('day',axis=1,inplace=True)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')

X_train.head()


class Splitter:
    def __init__(self, kfold=True, n_splits=5):
        self.n_splits = n_splits
        self.kfold = kfold

    def split_data(self, X, y, random_state_list):
        if self.kfold == 'skf':
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    if type(X) is np.ndarray:
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                    else:
                        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            raise ValueError(f"Invalid kfold: Must be True")


class Classifier:
    def __init__(self, n_estimators=200, device="cpu", random_state=42):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)
        
    def _define_model(self):
        
        xgb_optuna0 = {
            'n_estimators': 1000,
            'learning_rate': 0.01752354328845971,
            'booster': 'gbtree',
            'lambda': 0.08159630121074074,
            'alpha': 0.07564858712175693,
            'subsample': 0.5065979400270813,
            'colsample_bytree': 0.6187340851873067,
            'max_depth': 4,
            'min_child_weight': 5,
            'eta': 0.2603059902806757,
            'gamma': 0.6567360773618207,
            #'scale_pos_weight': scale_pos_weight,
            'random_state': random_state
        }
        
        lgb_params0 = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'goss',
            'random_state': 42,
            'colsample_bytree': 0.50,
            'subsample': 0.70,
            'learning_rate': 0.0625,
            'max_depth': 6,
            'n_estimators': 1000,
            'num_leaves': 110, 
            'reg_alpha': 0.0001,
            'reg_lambda': 2.0,
            'verbosity': -1,
            'random_state': random_state,
        }


        
    ### All those models are from previous binary classification competitions that I participated in. They are not tuned for this particular competition and I use them for baseline solution    
        xgb_params0 = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.09641232707445854,
            'booster': 'gbtree',
            'lambda': 4.666002223704784,
            'alpha': 3.708175990751336,
            'subsample': 0.6100174145229473,
            'colsample_bytree': 0.5506821152321051,
            'max_depth': 7,
            'min_child_weight': 3,
            'eta': 1.740374368661041,
            'gamma': 0.007427363662926455,
            'grow_policy': 'depthwise',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': self.random_state,
            #'scale_pos_weight': scale_pos_weight
        }
        
        xgb_params1 = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.012208383405206188,
            'booster': 'gbtree',
            'lambda': 0.009968756668882757,
            'alpha': 0.02666266827121168,
            'subsample': 0.7097814108897231,
            'colsample_bytree': 0.7946945784285216,
            'max_depth': 3,
            'min_child_weight': 4,
            'eta': 0.5480204506554545,
            'gamma': 0.8788654128774149,
            'scale_pos_weight': 4.71,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': self.random_state,
           # 'scale_pos_weight': scale_pos_weight
        }

        
        
        xgb_params2 = {
            'n_estimators': self.n_estimators,
            'colsample_bytree': 0.5646751146007976,
            'gamma': 7.788727238356553e-06,
            'learning_rate': 0.1419865761603358,
            'max_bin': 824,
            'min_child_weight': 1,
            'random_state': 811996,
            'reg_alpha': 1.6259583347890365e-07,
            'reg_lambda': 2.110691851528507e-08,
            'subsample': 0.879020578464637,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'n_jobs': -1,
            'verbosity': 0,
            'random_state': self.random_state,
           # 'scale_pos_weight': scale_pos_weight
        }
        
        xgb_params3 = {
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'colsample_bytree': 0.4836462317215041,
            'eta': 0.05976752607337169,
            'gamma': 1,
            'lambda': 0.2976432557733288,
            'max_depth': 6,
            'min_child_weight': 1,
            'n_estimators': 550,
            'objective': 'binary:logistic',
            'scale_pos_weight': 4.260162886376033,
            'subsample': 0.7119282378433924,
           # 'scale_pos_weight': scale_pos_weight
        }
        
        xgb_params4 = {
            'n_estimators': self.n_estimators,
            'colsample_bytree': 0.8757972257439255,
            'gamma': 0.11135738771999848,
            'max_depth': 7,
            'min_child_weight': 3,
            'reg_alpha': 0.4833998914998038,
            'reg_lambda': 0.006223568555619563,
            'scale_pos_weight': 8,
            'subsample': 0.7056434340275685,
            'random_state': self.random_state,
           # 'scale_pos_weight': scale_pos_weight
        }
        
        xgb_params5 = {
            'n_estimators': self.n_estimators,
            'max_depth': 5, 
            'min_child_weight': 2.934487833919741,
            'learning_rate': 0.11341944575807082, 
            'subsample': 0.9045063514419968,
            'gamma': 0.4329153382843715,
            'colsample_bytree': 0.38872702868412506,
            'colsample_bylevel': 0.8321880031718571,
            'colsample_bynode': 0.802355707802605,
            'random_state': self.random_state,
            #'scale_pos_weight': scale_pos_weight
       }
        
        xgb_base = {
            'n_estimators': self.n_estimators,
           # 'scale_pos_weight': scale_pos_weight,
            'verbosity': 0,
            'random_state': self.random_state,
        }
        
        xgb_params6 = {
            'objective': 'binary:logistic',
            'colsample_bytree': 0.7, 
            'gamma': 2, 
            'learning_rate': 0.01, 
            'max_depth': 7, 
            'min_child_weight': 10, 
            'n_estimators': 500, 
            'subsample':0.7,
            'random_state': self.random_state,
           # 'scale_pos_weight': scale_pos_weight
        }

        xgb_params7 = {
            'n_estimators': 190, 
            'learning_rate': 0.017792963423540194, 
            'max_depth': 6, 
            'subsample': 0.2579692108675591, 
            'colsample_bytree': 0.2487767930540334, 
            'min_child_weight': 4,
            'random_state': self.random_state
        }

        xgb_params8 = {
            'n_estimators': 407,
             'max_depth': 7, 
             'learning_rate': 0.0012420086569174989, 
             #'scale_pos_weight': 10.71605747228545, 
             'subsample': 0.6452516976062799, 
             'colsample_bytree': 0.7854972792598017, 
             'min_child_weight': 8, 
             'reg_alpha': 9.493547785077148e-05, 
             'reg_lambda': 0.9222350455360162, 
             'gamma': 1.1156600244243586, 
             'max_delta_step': 0,
             'colsample_bylevel': 0.9261592982325488, 
             'colsample_bynode': 0.9247934082270379, 
             'grow_policy': 'lossguide'
        }

        xgb_params9 = {
             'n_estimators': 493,
             'learning_rate': 0.005044728330173781,
             'max_depth': 8,
             'scale_pos_weight': 0.6874137917773947,
             'subsample': 0.7075768096876544,
             'colsample_bytree': 0.36594577594144206,
             'min_child_weight': 3, 
             'max_delta_step': 6,
             'random_state': self.random_state,
        }

        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
       
        models = {
            
            # XGBoost
            #'xgb': xgb.XGBClassifier(random_state=self.random_state),
            'xgbOp': xgb.XGBClassifier(**xgb_optuna0),
            'xgb0': xgb.XGBClassifier(**xgb_params0),
            #'xgb1': xgb.XGBClassifier(**xgb_params1),
            #'xgb2': xgb.XGBClassifier(**xgb_params2),
            #'xgb3': xgb.XGBClassifier(**xgb_params3),
            #'xgb4': xgb.XGBClassifier(**xgb_params4),
            #'xgb5': xgb.XGBClassifier(**xgb_params5),
            #'xgbb': xgb.XGBClassifier(**xgb_base),
            #'xgb6': xgb.XGBClassifier(**xgb_params6),
            #'xgb7': xgb.XGBClassifier(**xgb_params7),
            'xgb8': xgb.XGBClassifier(**xgb_params8),
            'xgb9': xgb.XGBClassifier(**xgb_params9),
            
            # Misc
            #'lgb0': lgb.LGBMClassifier(**lgb_params0),
            
            # add some models with default params to "simplify" ensemble
           # 'svc': SVC(random_state=self.random_state, probability=True),
            'brf': BalancedRandomForestClassifier(max_depth = 6, random_state=self.random_state),

        }
        
        return models


class OptunaWeights:
    def __init__(self, random_state, n_trials=100):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-14, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the score for the weighted prediction
        # score = log_loss(y_true, weighted_pred)
        score = roc_auc_score(y_true, weighted_pred)
        
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights


%%time

# Config
kfold = 'skf'
n_splits = 12
n_reapts = 3
random_state = 42
n_estimators = 9999
early_stopping_rounds = 333
verbose = False
device = 'cpu'

# Fix seed
random.seed(random_state)
random_state_list = random.sample(range(9999), n_reapts)
#random_state_list = [42]

# Initialize an array for storing test predictions
classifier = Classifier(n_estimators, device, random_state)
test_predss = np.zeros((X_test.shape[0]))
oof_predss = np.zeros((X_train.shape[0], n_reapts))

# Store scores and weights
ensemble_score = []
weights = []

# Predictions and models
oof_each_predss = []
oof_each_preds = np.zeros((X_train.shape[0], classifier.len_models))
test_each_predss = []
test_each_preds = np.zeros((X_test.shape[0], classifier.len_models))
trained_models = {'xgb':[],}
score_dict = dict(zip(classifier.models_name, [[] for _ in range(classifier.len_models)]))

# Loop over KFold splits
splitter = Splitter(kfold=kfold, n_splits=n_splits)
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits
            
    # Get a set of classifier models
    classifier = Classifier(n_estimators, device, random_state_list[m])
    models = classifier.models
    
    # Initialize lists to store oof and test predictions for each base model
    oof_preds = []
    test_preds = []
    
    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
    for name, model in models.items():
        if ('xgb' in name) or ('lgb' in name) or ('cat' in name):
            if 'xgb' in name:
                model.fit(
                    X_train_, y_train_, 
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            elif 'lgb' in name:
                model.fit(
                    X_train_, y_train_, 
                    eval_set=[(X_val, y_val)])
            elif 'cat' in name:
                model.fit(
                    Pool(X_train_, y_train_), 
                    eval_set=Pool(X_val, y_val),
                    early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
            
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))
        
        test_pred = model.predict_proba(X_test)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]
        
        # Calculate recall and precision scores
        y_val_pred_binary = (y_val_pred > 0.5).astype(int)
        recall = recall_score(y_val, y_val_pred_binary)
        precision = precision_score(y_val, y_val_pred_binary)
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] Recall score: {recall:.5f}')
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] Precision score: {precision:.5f}')

        score = roc_auc_score(y_val, y_val_pred)
        score_dict[name].append(score)
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] ROC score: {score:.5f}')
        print('-'*50)
        
        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
    
    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state_list[m])
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
    
    score_ = roc_auc_score(y_val, y_val_pred)
    print(f'--> Ensemble [FOLD-{n} SEED-{random_state_list[m]}] ROC score {score_:.5f}')
    print_sl()
    ensemble_score.append(score_)
    weights.append(optweights.weights)
    
    # Predict to X_test by the best ensemble weights
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    oof_predss[X_val.index, m] += optweights.predict(oof_preds)
    oof_each_preds[X_val.index] = np.stack(oof_preds).T
    test_each_preds += np.array(test_preds).T / n_splits
    
    if n == (n_splits - 1):
        oof_each_predss.append(oof_each_preds)
        oof_each_preds = np.zeros((X_train.shape[0], classifier.len_models))
        test_each_predss.append(test_each_preds)
        test_each_preds = np.zeros((X_test.shape[0], classifier.len_models))
    
    gc.collect()
    
oof_each_predss = np.mean(np.array(oof_each_predss), axis=0)
test_each_predss = np.mean(np.array(test_each_predss), axis=0)
oof_each_predss = np.concatenate([oof_each_predss, np.mean(oof_predss, axis=1).reshape(-1, 1)], axis=1)
test_each_predss = np.concatenate([test_each_predss, test_predss.reshape(-1, 1)], axis=1)


# Calculate the mean score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Mean Optuna Ensemble {mean_score:.5f} ± {std_score:.5f} \n')

print('--- Optuna Weights---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')


my_palette = sns.cubehelix_palette(n_colors = 7, start=.46, rot=-.45, dark = .2, hue=0.95, as_cmap=True)

def show_confusion_roc(oof, title='Model Evaluation Results'):
    f, ax = plt.subplots(1, 2, figsize=(16, 6))
    df = pd.DataFrame({'preds': oof[0], 'target': oof[1]})
    cm = confusion_matrix(df.target, df.preds.ge(0.5).astype(int))
    cm_display = ConfusionMatrixDisplay(cm).plot(cmap=my_palette, ax=ax[0])
    ax[0].grid(False)
    RocCurveDisplay.from_predictions(df.target, df.preds, ax=ax[1])
    ax[1].grid(True)
    plt.suptitle(f'{title}', fontsize=12, fontweight='bold')
    plt.tight_layout()

show_confusion_roc(oof=[oof_each_predss[:, 5], y_train['rainfall']], title='OOF Evaluation Results')


%%time

stack_test_predss = np.zeros((X_test.shape[0]))
stack_scores = []
stack_models = []
splitter = Splitter(kfold=kfold, n_splits=n_splits)
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(oof_each_predss, np.array(y_train), random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits
    
    classifier = Classifier(n_estimators, device, random_state_list[m])
    models = classifier.models
    model = models['xgb8']
    
    model.fit(
    X_train_, y_train_,
   # eval_set=[(X_val, y_val)],
   # early_stopping_rounds=early_stopping_rounds,
   # verbose=verbose
)
    
    test_pred = model.predict_proba(test_each_predss)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]

    score = roc_auc_score(y_val, y_val_pred)
    stack_scores.append(score)
    stack_models.append(deepcopy(model))
    
    stack_test_predss += test_pred / (n_splits * len(random_state_list))


# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble ROC score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')
print('')

# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(stack_scores)
std_score = np.std(stack_scores)
print(f'Stacking ROC score {mean_score:.5f} ± {std_score:.5f}\n')


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    X_train_, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    model = xgb.XGBClassifier(
        n_estimators = 493,
        learning_rate = 0.005044728330173781,
        max_depth = 8,
        scale_pos_weight = 0.6874137917773947,
        subsample = 0.7075768096876544,
        colsample_bytree = 0.36594577594144206,
        min_child_weight = 3,
        max_delta_step = 6,
        random_state = 42,
    )

    # model = RandomForestClassifier(random_state = 42)
    model.fit(X_train_, y_train_)

    # Get predicted probabilities
    oof_preds[val_index] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits  # Averaging across folds

# Find optimal threshold using all OOF predictions
fpr, tpr, thresholds = roc_curve(y_train, oof_preds)
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

# Apply threshold for final classification
oof_binary_preds = (oof_preds >= optimal_threshold).astype(int)
test_binary_preds = (test_preds >= optimal_threshold).astype(int)

# Final evaluation
oof_auc = roc_auc_score(y_train, oof_preds)  # Still using probabilities for AUC
print(f'Final OOF AUC: {oof_auc:.7f}')
print(f'Optimal Threshold: {optimal_threshold:.7f}')


sub = pd.read_csv(os.path.join(sub_PATH))

sub['rainfall'] = stack_test_predss * 0.5 + test_preds * 0.5
sub.to_csv('submission.csv', index=False)
sub

