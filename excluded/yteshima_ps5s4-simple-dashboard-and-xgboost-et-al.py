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


import seaborn as sns
import matplotlib.pyplot as plt
import holidays

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from datetime import datetime, timedelta

import ipywidgets as widgets
from ipywidgets import interact, Layout
from IPython.display import HTML, display, clear_output
from IPython.display import IFrame

import matplotlib.pyplot as plt

from scipy import optimize

from xgboost import XGBRegressor,XGBClassifier, DMatrix
from lightgbm import LGBMRegressor, LGBMClassifier, log_evaluation, early_stopping
import lightgbm as lgb

from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA

from matplotlib.colors import LinearSegmentedColormap
%matplotlib inline 

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap', ['blue', 'white', 'red']
)

import random

from tqdm import tqdm

from gc import collect
from colorama import Fore, Style, init;

import optuna
import shap

from optuna.samplers import TPESampler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from scipy import optimize

# ignore wornings
import warnings
warnings.filterwarnings("ignore")

# Get the execution mode of the Kaggle environment
run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive')


# Load the training and test data from CSV files

df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv',index_col = 0)
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=0)

print(f'N_train = {len(df_train)}, N_test = {len(df_test)}')

# Assign a new column 'train_test' with the value 'train' to the training dataset, and 'test' to the test dataset respectively
df_train['train_test'] = 'train'
df_test['train_test'] = 'test'

# Create reduced versions of the training and test datasets by randomly sampling 1/20th of the rows
df_train_reduced = df_train.sample(len(df_train) // 20)
df_test_reduced = df_test.sample(len(df_test) // 20)
print(f'N_train_reduced = {len(df_train_reduced)}, N_test_reduced = {len(df_test_reduced)}')

# Specify the target column for the analysis or model
target_col = 'Listening_Time_minutes'

# Combine the training and test datasets into a single DataFrame for unified processing
df_all = pd.concat([
    df_train,
    df_test,
    # Uncomment the lines below to include the reduced datasets if needed
    # df_train_reduced,
    # df_test_reduced
])



df_all_pp = df_all.copy()


# Define a custom format function to format float values for display
def custom_format(x):
    if isinstance(x, float):
        # Format float values to 3 decimal places and remove trailing zeros and decimal points
        return ('{0:.3f}'.format(x)).rstrip('0').rstrip('.')
    return x

# Display detailed information about the DataFrame, including statistics and metadata
def display_dfinfo(df):
    display(HTML('<br><h2>Display head of the dataframe</h2>'))
    display(df_all.sample(3))
    display(HTML('<br><h2>Display numerical data infomations</h2>'))
    df_disp = []
    for tt in df['train_test'].unique():
        # Generate descriptive statistics for numeric columns
        tmp = df.select_dtypes(include=[int, float]).loc[df['train_test'] == tt].describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        )

        # Add skewness and kurtosis for numeric columns
        tmp.loc['skew'] = df.loc[df['train_test'] == tt].select_dtypes(include=[int, float]).skew()
        tmp.loc['kurtosis'] = df.loc[df['train_test'] == tt].select_dtypes(include=[int, float]).kurtosis()

        # Add data type and NaN count for all columns
        tmp.loc['dtype'] = df.loc[df['train_test'] == tt].dtypes
        tmp.loc['NaN count'] = df.loc[df['train_test'] == tt].isna().sum(axis=0)

        tmp.loc['N unique'] = df.loc[df['train_test'] == tt].nunique()
        tmp.columns = pd.MultiIndex.from_product([tmp.columns, [tt]])  # Add multi-level columns
        df_disp.append(tmp)

    df_disp = pd.concat(df_disp, axis=1)  # Combine statistics for all 'train_test' groups
    df_disp = df_disp[df_disp.columns.get_level_values(0).unique()]  # Remove duplicate columns
    # Reorganize the DataFrame and filter relevant statistics
    df_disp = df_disp.T
    df_disp = df_disp.loc[
        df.select_dtypes(include=[int, float]).columns, [
            'count', 'NaN count', 'N unique', 'dtype',
            'mean', 'min', '5%', '25%', '50%', '75%', '95%', 'max',
            'std', 'skew', 'kurtosis'
        ]
    ]
    formatter = {}
    # Display the DataFrame with custom formatting and background gradients for numeric stats
    display(
        df_disp.style.format(formatter=custom_format).background_gradient(
            subset=['mean', 'min', '5%', '25%', '50%', '75%', '95%', 'max'],
            cmap='Reds', axis=1
        )
    )
    
    if len([col for col in df.select_dtypes(include='object') if col != 'train_test']) > 0:
        display(HTML('<br><h2>Display categorical data infomations</h2>'))
        col = df.columns[0]
        for col in [col for col in df.select_dtypes(include='object') if col != 'train_test']:
            df_disp = []
            for tt in df['train_test'].unique():
                tmp = df.loc[df['train_test']==tt,col]
                tmp.fillna('nan', inplace = True)
                tmp = pd.DataFrame(tmp.value_counts()).T
                df_disp.append(tmp)
    
            df_tmp = df[[col, target_col]].copy()
            df_tmp[col].fillna('nan', inplace = True)
    
            df_disp.append(df_tmp.groupby(col, dropna = False)[target_col].describe()[['mean', 'std']].T)
            df_disp = pd.concat(df_disp)
    
            df_disp.index = pd.MultiIndex.from_product([
                [col],
                [f'{trates} count' for trates in list(df['train_test'].unique())] + [f'{target_col} mean', f'{target_col} std']
            ])
    
            df_disp.columns.names = ['']
            display(df_disp.style.set_table_styles([
                {'selector': 'th.index_name', 'props': [('width', '60px')]},  # width of 1st row
                {'selector': 'th.row_heading', 'props': [('width', '90px')]},  # width of 2nd row
            ]
           ).background_gradient(cmap='Reds', axis=1))


# Function to plot a correlation matrix for numeric columns
def plot_correlation_matrix(df, num_cols, plottype='sns'):
    if plottype == 'plotly':
        fig = px.imshow(
            df[num_cols].corr(), zmax=1, zmin=-1, color_continuous_scale='rdbu_r',  # Red-blue color scale
            text_auto=".2f"
        )
        # Customize the layout of the Plotly figure
        fig.update_layout(
            width=max(min(len(num_cols) * 100, 600), 400),
            height=max(min(len(num_cols) * 80, 600), 400),
            title='Correlation matrix'
        )
        fig.show()
    elif plottype == 'sns':
        plt.figure(figsize=(min(len(df.columns) * 0.6, 12), min(len(df.columns) * 0.25, 12)))
        sns.heatmap(
            df[num_cols].corr(), annot=True, vmax=1, vmin=-1,
            cmap=custom_cmap, fmt='.2f'  # Use a custom color map and format values
        )
        plt.title('Correlation Matrix')
        plt.show()



num_cols = df_all_pp.select_dtypes(include=[float, int]).columns
disp_cols = df_all_pp.columns
cat_cols = [c for c in df_all_pp.columns if c!= 'date']
display_dfinfo(df_all_pp)
plot_correlation_matrix(df_all_pp, num_cols,  plottype='sns')


def target_encoder(df, input_col, target_col):
    tmp = df[[input_col, target_col]]
    means = df.groupby(input_col)[target_col].mean()
    for ind in means.index:
        tmp.loc[tmp[f'{input_col}']==ind, f'{input_col}_te'] = means[ind]

    return tmp[f'{input_col}_te'].values

def preprocessing(df, num_cols, cat_cols, target_col,train_test = 'train_test'):
    df_pp = df[num_cols].copy()
    for i, cat_col in enumerate(cat_cols):
        print(cat_col, end = ' / ')
        # target encoding
        df_pp[f'{cat_col}_te'] = target_encoder(df, cat_col, target_col)
    df_pp[target_col] = df[target_col]
    df_pp[train_test] = df[train_test]
    return df_pp

def adversarial_validation(df_adv):
#     Return a list of train data indistinguishable from test data
    xgb = XGBClassifier()
    X_adv = df_adv.drop('train_test',axis = 1)
    y_adv = df_adv['train_test'].map({'train':0,'train_extra':0, 'original':0, 'test':1})
    
    xgb.fit(X_adv, y_adv)
    predict_adv = pd.DataFrame(
        xgb.predict_proba(X_adv.loc[y_adv==0])[:,0], columns=['train'],
        index = X_adv.index[y_adv==0]
    )
    predict_adv.sort_values(by='train',inplace = True)
    return predict_adv.index


print(df_all_pp.select_dtypes(exclude=[int,float]).columns)
print(df_all_pp.select_dtypes(include=[int,float]).columns)


df_all[target_col] = df_all[target_col].astype(float)

cat_cols = [
    'Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
    'Publication_Time', 'Episode_Sentiment'
]
num_cols = [
    'Episode_Length_minutes', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads',
    'Listening_Time_minutes'
]
print('Preprocessing start', end=' → ')
# df_all_pp = catboostimputer(df_all, target_col)
df_all_pp = preprocessing(
    df_all_pp,
    num_cols, cat_cols,
    target_col
)


df_all_pp[target_col]=df_all_pp[target_col].astype(float)

input_coaggregatels = list(df_all_pp.columns)
input_cols = list(df_all_pp.columns)
input_cols.remove(target_col)

df_target = df_all_pp.loc[:, input_cols + [target_col]].copy()
df_target.drop(
    df_target.index[
        (df_target['train_test'].isin(['train'])) &
        (df_target[target_col].isna())
    ], axis=0, inplace = True
)

train_data = df_target.loc[
    df_target['train_test'].isin(['train'])
].drop('train_test', axis=1)
test_data =  df_target.loc[
    df_target['train_test']=='test'
].drop('train_test', axis=1)

valid_indices = adversarial_validation(df_target.drop(target_col, axis = 1))
valid_indices = valid_indices[:round(len(valid_indices)*0.2)]

X_train = train_data.drop(target_col,axis=1).drop(valid_indices)
X_val = train_data.drop(target_col,axis=1).loc[valid_indices]
X_test = test_data.drop(target_col, axis = 1)

y_train = train_data[target_col].drop(valid_indices)
y_val = train_data[target_col].loc[valid_indices]



print(f'X_train.shape:{X_train.shape}, X_val.shape:{X_val.shape}, X_test.shape:{X_test.shape}')
print(f'y_train.shape:{y_train.shape}, y_val.shape:{y_val.shape}')


def bayese_objective(X, y, Classifier, metric, n_sample = np.nan):
    
    def bayese_trial(trial):
        if Classifier in [XGBClassifier, XGBRegressor]:
            params = {
                'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                'gamma' : trial.suggest_float('gamma', 1e-9, 0.5),
                'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'max_depth': trial.suggest_int('max_depth', 0, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),

                'random_state': 42,
                'booster':'gbtree',
                'device':"cuda",
                'verbosity': 0,
                'tree_method':"hist",
                "timeout_request_budget": 180,
                'eval_metric': metrics['XGB'],

                
            }
        elif Classifier in [LGBMClassifier, LGBMRegressor]:
            params = {
                "n_estimators": trial.suggest_int('n_estimators', 50, 1000, step=10),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                "max_depth": trial.suggest_int('max_depth', 3, 15),
                "min_child_samples": trial.suggest_int('lgbm_min_child_samples', 1, 20),
                "subsample": trial.suggest_float('subsample', 0.5, 1.0),
                "colsample_bytree": trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'verbose':-1,
                
                'random_state': 42,
                "time_budget": 180,
                'verbose':-1,
                'metric': metrics['LGBM']
            }
        elif Classifier in [CatBoostClassifier, CatBoostRegressor]:
            params = {
                "iterations": trial.suggest_int('iterations', 50, 1000, step=10),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                "depth": trial.suggest_int('depth', 3, 15),
                "l2_leaf_reg": trial.suggest_float('l2_leaf_reg', 1e-3, 1),
                
                'random_state': 42,
                "verbose": False,
                # 'time_limit':300,
                'eval_metric': metrics['CatBoost'],
            }
        
        cv = KFold(n_splits=20, shuffle=True, random_state=0)

        if np.isnan(n_sample):
            X_reduced = X.copy()
            y_reduced = y.copy()
        else:
            X_reduced = X.sample(n_sample)
            y_reduced = y.loc[X_reduced.index]

        cv_splits = cv.split(X_reduced, y = y_reduced)
        cv_scores = list()
        
        for train_idx, val_idx in cv_splits:
            model = Classifier()
            model.set_params(**params)
            X_train_fold, X_val_fold = X_reduced.iloc[train_idx], X_reduced.iloc[val_idx]
            y_train_fold, y_val_fold = y_reduced.iloc[train_idx], y_reduced.iloc[val_idx]
            model.fit(X_train_fold, y_train_fold)
            
            y_val_prob = model.predict(X_val_fold)
            score = mean_squared_error(y_val_fold, y_val_prob)

            cv_scores.append(score)
        return np.mean(cv_scores)
    return bayese_trial


%%time
best_params1 = []
best_scores1 = []
_optim = False

metrics = {
    'XGB': 'rmse',
    'LGBM': 'rmse',
    'CatBoost': 'RMSE'
}
models = {
    'XGB':XGBRegressor,
    'LGBM':LGBMRegressor,
    'Cat':CatBoostRegressor
}
if _optim:# run Bayese Optimization
    for key, model in models.items():
        print(f'{key} {model}:')
        try:
            study = optuna.create_study(
                direction = 'minimize',
                sampler=optuna.samplers.TPESampler(seed=0),
                study_name=f"{key}_study", storage=f"sqlite:///{key}_study.db", load_if_exists=True
            )
                        
            study.optimize(
                bayese_objective(X_train, y_train, model, metrics, n_sample = np.nan),
                n_trials=1000, timeout=3600 * 3, n_jobs = -1
            )

            best_params1.append(study.best_trial.params)
            best_scores1.append(study.best_trial.value)
            print(f'{key}:')
            print('best params:')
            print(best_params1[-1])
            print('best scores:')
            print(best_scores1[-1])
        except:
            print(f'{model} failed')

else:
    best_params1 = {
        'XGB':{
            'grow_policy': 'lossguide', 'n_estimators': 881, 'learning_rate': 0.010002859598284185, 'gamma': 0.42112547221031443, 'subsample': 0.5089850589088577, 'colsample_bytree': 0.8074295596396124, 'max_depth': 5, 'min_child_weight': 5, 'reg_lambda': 0.004798224304308994, 'reg_alpha': 5.7767510609461336e-08,
            
            'random_state': 42,
            'booster':'gbtree',
            'device':"cuda",
            'verbosity': 0,
            'tree_method':"hist",
            'eval_metric': metrics['XGB'],
        },'LGBM':{
            'n_estimators': 200, 'learning_rate': 0.036876878605913356, 'max_depth': 4, 'lgbm_min_child_samples': 4, 'subsample': 0.5962038888245833, 'colsample_bytree': 0.981190687562547, 'num_leaves': 100,
            
            'random_state': 42,
            'verbose':-1,
            'metric': metrics['LGBM']
        },'CatBoost':{
            'iterations': 440, 'learning_rate': 0.020931272720137666, 'depth': 7, 'l2_leaf_reg': 0.5923369750739447,
            
            'random_state': 42,
            "verbose": False,
            'eval_metric': metrics['CatBoost'],
        }}



%%time
models_best = [
    XGBRegressor(), LGBMRegressor(),
    CatBoostRegressor()
]
predict_cols = [
    'XGB', 'LGBM',
    'CatBoost',
    # 'keras',
    'blend']
predict_trains = pd.DataFrame(index = X_train.index)
predict_vals = pd.DataFrame(index = X_val.index)
predict_tests = pd.DataFrame(X_test.index)


for i,model in enumerate(models_best):
    print(predict_cols[i], end = ' / ')
    model.set_params(**best_params1[predict_cols[i]])
    model.fit(X_train, y_train)
    predict_trains[predict_cols[i]] = model.predict(X_train)
    predict_vals[predict_cols[i]] = model.predict(X_val)
    predict_tests[predict_cols[i]] = model.predict(X_test)
print('finished')


# imputer = KNNImputer()
# ss_input = StandardScaler()
# train_data_input_ss = ss_input.fit_transform(train_data.drop(target_col,axis=1))
# train_data_input_ss = pd.DataFrame(
#     imputer.fit_transform(train_data_input_ss),
#     index = train_data.index, columns = train_data.drop(target_col,axis=1).columns
# )
# test_data_input_ss = ss_input.transform(test_data.drop(target_col,axis=1))
# test_data_input_ss = pd.DataFrame(
#     imputer.transform(test_data_input_ss),
#     index = test_data.index, columns = test_data.drop(target_col,axis=1).columns
# )

# X_train_ss = train_data_input_ss.drop(valid_indices)
# X_val_ss = train_data_input_ss.loc[valid_indices]
# X_test_ss = test_data_input_ss


# early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# # Initialize Neural Network
# model = Sequential([
#     Dense(
#         128, activation='relu', kernel_initializer='he_normal',
#         input_shape=(X_train_ss.shape[1],)
#     ),
#     Dropout(0.3),
#     Dense(
#         64, activation='relu', kernel_initializer='he_normal'
#     ),
#     Dropout(0.2),
#     Dense(
#         32,activation='relu', kernel_initializer='he_normal'
#     ),
#     Dropout(0.2),
#     Dense(
#         16,activation='relu', kernel_initializer='he_normal'
#     ),
#     Dense(1, activation='linear')  # Binary classification
# ])

# # Compile Model
# optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# # Train Model
# history = model.fit(X_train_ss, y_train, epochs=500, batch_size=32, validation_split=0.2, 
#                     callbacks=[early_stopping], verbose=0)

# # Make Predictions
# predict_trains['keras'] = model.predict(X_train_ss).flatten()
# predict_vals['keras'] = model.predict(X_val_ss).flatten()

# y_test_pred_keras = model.predict(X_test_ss).flatten()


predict_trains['blend'] = predict_trains.mean(axis=1)
predict_vals['blend'] = predict_vals.mean(axis=1)
predict_tests['blend'] = predict_tests.mean(axis=1)

predict_trains['True'] = y_train
predict_vals['True'] = y_val
predict_vals['train_val'] = 'val'


target_max = max(predict_trains.max()[:-1].max(), predict_vals.max()[:-1].max())
target_max = max(target_max, df_all[target_col].max())
target_min = min(np.abs(predict_trains.min()[:-1]).min(), np.abs(predict_vals.min()[:-1]).min())
target_min = min(target_min, df_all[target_col].min())
fig, ax = plt.subplots(nrows = 1, ncols = len(predict_cols), figsize = (len(predict_cols)*3,4))
for i, model in enumerate(predict_trains.columns[:-1]):
    score_train = np.sqrt(mean_squared_error(predict_trains['True'], predict_trains[model]))
    score_val = np.sqrt(mean_squared_error(predict_vals['True'], predict_vals[model]))
    ax[i].scatter(predict_trains['True'],predict_trains[model], c='r', label=f'train auc={score_train:.4f}', s=2, alpha=0.1)
    ax[i].scatter(predict_vals['True'],predict_vals[model], c='b', label = f'val auc={score_val:.4f}',s = 2, alpha= 0.1)
    ax[i].legend()
    ax[i].grid()
    ax[i].set_title(model)
    ax[i].set_xlabel('true')
    ax[i].set_ylabel('predict');
    ax[i].set_aspect(1)
    
plt.tight_layout()  


# ipywidgets　will run only in interactive mode
if run_type == 'Interactive':
    shap.initjs()
    model_button = widgets.ToggleButtons(
        options = ['XGB', 'LGBM', 'CatBoost'],
        button_style='info',description = 'model:'
    )
    model_button.style.button_width = f'100px'
    model_button.style.description_width = '90px'
    
    type_button = widgets.ToggleButtons(
        options = ['dot','bar'],
        button_style='warning',description = 'type:'
    )
    type_button.style.button_width = f'100px'
    type_button.style.description_width = '90px'
    
    max_disp_slider = widgets.IntSlider(
        value=min(6,len(df_train.columns)), min=0, max=len(X_train.columns), step=1, 
        description='max_display:', orientation='horizontal'
    )
    max_disp_slider.style.button_width = f'100px'
    max_disp_slider.style.description_width = '90px'
    
    
    @interact(model_name = model_button, plot_type = type_button, max_display = max_disp_slider)
    def plot_re(model_name, plot_type, max_display):
        df_train = X_train.sample(1000)
        i = list(predict_cols).index(model_name)
    
        model = models_best[i]
            
        explainer = shap.TreeExplainer(model=model, model_output='raw')
        shap_values = explainer.shap_values(X=df_train)
        shap.summary_plot(shap_values, df_train, plot_type=plot_type, max_display=max_display)
else:
    print('Run in interactive mode to display plots.')
    df_train = X_train.sample(1000)
    model = models_best[0]
    explainer = shap.TreeExplainer(model=model, model_output='raw')
    shap_values = explainer.shap_values(X=df_train)
    shap.summary_plot(shap_values, df_train, plot_type='dot', max_display=7)



models_final = [XGBRegressor(), LGBMRegressor(), CatBoostRegressor()]
predict_tests = []
for i,model in enumerate(models_final):
    print(predict_cols[i], end = ' / ')
    model.set_params(**best_params1[predict_cols[i]])
    model.fit(train_data.drop(target_col,axis=1), train_data[target_col])
    predict_tests.append(model.predict(X_test))

predict_tests = pd.DataFrame(
    np.array(predict_tests).T, columns = ['XGB', 'LGBM', 'CatBoost'], index = X_test.index
)

# predict_tests['keras'] = y_test_pred_keras

predict_tests['blend'] = predict_tests['XGB'] * 0.5 + predict_tests['LGBM'] * 0.25 + predict_tests['CatBoost'] * 0.25 # + predict_tests['keras'] * 0.0


y_test_predict = predict_tests['blend']
# y_test_predict = df_all['prediction']
df_submit = df_sample_submission.copy()
df_submit[target_col] = y_test_predict
df_submit.to_csv('submission.csv', index = True)
print(df_submit.isna().sum())
display(df_submit)




