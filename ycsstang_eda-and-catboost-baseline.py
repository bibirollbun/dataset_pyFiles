def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


!pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.4.*"


%load_ext cudf.pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import cudf
warnings.filterwarnings('ignore')
from colorama import Fore, Style, init


def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

def set_cn_visualization_style():
    # Seaborn样式设置
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="deep",
        font="sans-serif",
        font_scale=1.1,
        rc={
            'font.family': ['sans-serif'],
            'font.sans-serif': [
                'SimHei',      
                'Microsoft YaHei', 
                'WenQuanYi Zen Hei', 
                'Arial Unicode MS', 
                'DejaVu Sans', 
                'sans-serif'
            ],
            
            'axes.unicode_minus': False, 
            
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'axes.edgecolor': '0.15',
            'axes.linewidth': 1,
            
            'grid.color': '.8',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.color': '0.4',
            'ytick.color': '0.4',
            
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'legend.edgecolor': '0.8',
            
            'lines.linewidth': 2,
            'lines.markersize': 7,
            
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.transparent': True,
            'figure.constrained_layout.use': True
        }
    )

    rcParams['mathtext.fontset'] = 'stix' 

set_cn_visualization_style()

CUSTOM_PALETTE = [
    '#2E86AB', 
    '#F24236',  
    '#5BBA6F',  
    '#FF9F1C',  
    '#6C5B7B', 
    '#F15BB5'   
]
sns.set_palette(CUSTOM_PALETTE)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv',index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv',index_col='id')


all_df = pd.concat([train_df,test_df])


all_df.info()


number_col = all_df.select_dtypes(include=np.number).columns
cat_col = all_df.select_dtypes(exclude=np.number).columns


target = 'Listening_Time_minutes'
number_col = number_col.drop('Listening_Time_minutes')


def histPlot(df,num_col,ax = None,bin = 20,kde=True):
    if not ax:
        fig,ax = plt.subplots()
    sns.histplot(df[num_col],bins=bin,kde=True,ax = ax)


fig,ax = plt.subplots(2,2)
for i,num_col in enumerate(number_col):
    histPlot(all_df,num_col,ax = ax[i // 2][i - i // 2 * 2])
fig.tight_layout()


histPlot(train_df,target)


upper = all_df[number_col].quantile(0.99999)
musk = (all_df[number_col] < upper).all(axis=1)
clean_all_df = all_df[musk]


fig,ax = plt.subplots(2,2)
for i,num_col in enumerate(number_col):
    histPlot(all_df[number_col][musk],num_col,ax = ax[i // 2][i - i // 2 * 2])
fig.tight_layout()


all_df[number_col][musk]['Number_of_Ads'].value_counts()


for cat in cat_col:
    cat_value_count = len(all_df[cat].unique())

    PrintColor(f'Col name:{cat}   Unique value:{cat_value_count}')


miss_value_percent = all_df.isna().sum() / len(all_df)
miss_value_percent = miss_value_percent[miss_value_percent > 0]
miss_value_percent = miss_value_percent.drop(target)
miss_value_percent.sort_values(ascending=False,inplace=True)
missing_col = miss_value_percent.index



fig,ax = plt.subplots()
fig.set_size_inches(10,6)
sns.barplot(x=miss_value_percent.index,y=miss_value_percent,ax = ax)
ax.set_xlabel('Missing Feature')
ax.set_ylabel('Missing Percent')
ax.set_title('Percent of Missing Data')


def boxPlot(df,cat_col,num_col,ax = None):
    if not ax:
        fig,ax = plt.subplots()
    sns.boxplot(data = df, x = cat_col , y = num_col, ax = ax)
    if len(df[cat_col].unique()) >= 10:
        ax.set_xticklabels(df[cat_col].unique(),rotation = 90)


fig,ax = plt.subplots(1,len(missing_col))
fig.set_size_inches(20,5)
for i,ml in enumerate(missing_col):
    boxPlot(clean_all_df,'Genre',ml,ax=ax[i])



fig,ax = plt.subplots(1,len(missing_col))
fig.set_size_inches(20,5)
for i,ml in enumerate(missing_col):
    boxPlot(clean_all_df,'Publication_Day',ml,ax=ax[i])



fig,ax = plt.subplots(1,len(missing_col))
fig.set_size_inches(20,5)
for i,ml in enumerate(missing_col):
    boxPlot(clean_all_df,'Episode_Sentiment',ml,ax=ax[i])



fig,ax = plt.subplots(1,len(missing_col))
fig.set_size_inches(20,5)
for i,ml in enumerate(missing_col):
    boxPlot(clean_all_df,'Publication_Time',ml,ax=ax[i])



### It doesn't look skewed on any Cat_col. So inpute NA with mean
### If skewed on certain CatCol. We can Inpute by that col
all_df[missing_col] = all_df[missing_col].fillna(all_df[missing_col].mean())


cat_target_std = []
for col in cat_col:
    cat_target_std.append(all_df.groupby(col)['Listening_Time_minutes'].mean().std())
    


PrintColor(f"Target std is {all_df['Listening_Time_minutes'].std()}",Fore.YELLOW)


fig,ax = plt.subplots()
sns.barplot(x = list(cat_col), y = cat_target_std,ax = ax)
ax.set_xticklabels(cat_col,rotation = 45)

## if some cat_col std is too large, we can use it to postprocess


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import sklearn
from tqdm import tqdm
from itertools import combinations


ohe = OneHotEncoder()
le = LabelEncoder()
add_cat_col = []


def FE(df):
    df_fe = df.copy()
    
    df_fe['is_weekend'] = df_fe['Publication_Day'].isin(['Saturday','Sunday']).astype('int').astype('category')
    add_cat_col.append('is_weekend')
    ##generate cross Feature
    columns_to_encode = ['Episode_Length_minutes', 
                     'Episode_Title', 
                     'Host_Popularity_percentage', 
                     'Number_of_Ads', 
                     'Episode_Sentiment', 
                     'Publication_Day', 
                     'Publication_Time']
    num_process_col = number_col.drop('Number_of_Ads')
    for col in num_process_col:
        df_fe[f'{col}_sin'] = np.sin(df_fe[col])
        df_fe[f'{col}_cos'] = np.cos(df_fe[col])

    for cat in cat_col:
        for num in number_col:
            tmp = df_fe.groupby(cat)[num].mean()
            tmp.name = f'{cat}_{num}_mean'
            df_fe = df_fe.merge(tmp,how='left',on=cat)

            tmp = df_fe.groupby(cat)[num].std()
            tmp.name = f'{cat}_{num}_std'
            df_fe = df_fe.merge(tmp,how='left',on=cat)
            
    
    pair_size = [2,3,4]
    for r in pair_size: 
        for cols in tqdm(list(combinations(columns_to_encode, r))): 
            new_col_name = '_'.join(cols)

            df_fe[new_col_name] = df_fe[list(cols)].astype(str).agg('_'.join, axis=1) 
            df_fe[new_col_name] = df_fe[new_col_name].astype('category')
            add_cat_col.append(new_col_name)
    new_cat_col = list(cat_col) + add_cat_col
    ##encoding cat Feature
    for col in new_cat_col:
        # count = len(df_fe[col].unique())
        # if count <= 10:
            
        #     ohe_arr = ohe.fit_transform(df_fe[[col]].values).toarray()
        #     ohe_df = pd.DataFrame(ohe_arr,columns=ohe.get_feature_names_out(),index=df_fe.index)
        #     df_fe = pd.concat([df_fe,ohe_df],axis=1)
        #     df_fe.drop(col,axis=1,inplace=True)
        # else:
        df_fe[col] = le.fit_transform(df_fe[col])
    
    return df_fe


all_df_FE = FE(all_df)


all_df_FE = reduce_mem_usage(all_df_FE)


train_df_FE = all_df_FE[all_df_FE.index < len(train_df)]
test_df_FE = all_df_FE[all_df_FE.index >= len(train_df)]


from catboost import CatBoostRegressor,CatBoostClassifier,Pool
from xgboost import XGBRegressor,XGBClassifier
import xgboost as xgb
from lightgbm import LGBMRegressor,LGBMClassifier
import torch
from sklearn.model_selection import train_test_split
import optuna


kf = KFold(n_splits= 5,shuffle=True,random_state=42)


# import cupy, cudf
# class IterLoadForDMatrix(xgb.core.DataIter):
#     def __init__(self, df=None, features=None, y=None, batch_size=256*1024):
#         self.y = y
#         self.df = df
#         self.it = 0 # set iterator to 0
#         self.batch_size = batch_size
#         self.batches = int( np.ceil( len(df) / self.batch_size ) )
#         super().__init__()

#     def reset(self):
#         '''Reset the iterator'''
#         self.it = 0

#     def next(self, input_data):
#         '''Yield next batch of data.'''
#         if self.it == self.batches:
#             return 0 # Return 0 when there's no more batch.
        
#         a = self.it * self.batch_size
#         b = min( (self.it + 1) * self.batch_size, len(self.df) )
#         dt = cudf.DataFrame(self.df.iloc[a:b])
#         input_data(data=dt, label=self.y) #, weight=dt['weight'])
#         self.it += 1
#         return 1




def get_gpu_count():
    try:
        return torch.cuda.device_count()
    except:
        return 1  


class ModelTrainer:
    def __init__(self,
                 n_splits = 5,
                 random_state = 42,
                 verbose = 0):
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.models = []
        self.oof = None
        self.feature_importances = None
        self.cv_score = None
    def load_data(self, X, y):
        self.X = X
        self.y = y

    def Trainmodel(self,model='catboost',
                   params=None,
                   type='regression',
                   metric='rmse',
                   kf=None,
                   early_stopping_rounds=50,
                   use_gpu=False):
        """
        model: str, default='catboost'
            The type of model to use. Options are 'catboost', 'xgboost', or 'lightgbm'.
        params: dict, default=None
            The parameters for the model. If None, default parameters will be used.
        type: str, default='regression'
            The type of model to use. Options are 'regression' or 'classification'.
        metric: str, default='rmse'
            The evaluation metric to use. Options are 'rmse' for regression and 'logloss' for classification.
        kf: KFold object, default=None
            The KFold object to use for cross-validation. If None, a default KFold object will be created.
        early_stopping_rounds: int, default=50
            The number of rounds for early stopping. If None, no early stopping will be used.
        """
        X,y = self.X, self.y
        cat_col = X.select_dtypes(exclude=np.number).columns
        if kf is None:
            kf = KFold(n_splits=self.n_splits, shuffle=True)
        
        
        oof_preds = np.zeros((self.X.shape[0],))
        self.feature_importances = np.zeros((X.shape[1],))
        if params is None:
            base_params = {
                'catboost': {
                    'iterations': 1000,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'random_seed': self.random_state,
                    'early_stopping_rounds': early_stopping_rounds,
                    'verbose': self.verbose,
                    'cat_features': list(cat_col),
                },
                'xgboost': {
                    'n_estimators': 1000,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                    'enable_categorical': True,
                },
                'lightgbm': {
                    'n_estimators': 1000,
                    'learning_rate': 0.1,
                    'max_depth': -1,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                }
            }
            self.params = base_params[model]
        else:
            self.params = params.copy()

        gpu_config = {
            'catboost': {
                True: {'task_type': 'GPU', 'devices': list(range(get_gpu_count()))},
                False: {'task_type': 'CPU'}
            },
            'xgboost': {
                True: {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor','n_gpus': get_gpu_count(),},
                False: {'tree_method': 'hist', 'predictor': 'cpu_predictor'}
            },
            'lightgbm': {
                True: {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': list(range(get_gpu_count())),'num_gpu': get_gpu_count()},
                False: {'device': 'cpu'}
            }
        }
        self.params.update(gpu_config[model][use_gpu])


        if type == 'regression':
            if model == 'catboost':
                self.models = [CatBoostRegressor(**self.params) for _ in range(self.n_splits)]
            elif model == 'xgboost':
                self.models = [XGBRegressor(**self.params) for _ in range(self.n_splits)]
            elif model == 'lightgbm':
                self.models = [LGBMRegressor(**self.params) for _ in range(self.n_splits)]
            else:
                raise ValueError("Unsupported model type. Choose from 'catboost', 'xgboost', or 'lightgbm'.")
        elif type == 'classification':
            if model == 'catboost':
                self.models = [CatBoostClassifier(**self.params) for _ in range(self.n_splits)]
            elif model == 'xgboost':
                self.models = [XGBClassifier(**self.params) for _ in range(self.n_splits)]
            elif model == 'lightgbm':
                self.models = [LGBMClassifier(**self.params) for _ in range(self.n_splits)]
            else:
                raise ValueError("Unsupported model type. Choose from 'catboost', 'xgboost', or 'lightgbm'.")
        else:
            raise ValueError("Unsupported model type. Choose from 'regression' or 'classification'.")



        PrintColor(f"Training {model} model...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            PrintColor(f"Fold {fold + 1}/{self.n_splits}", Fore.YELLOW)
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            if model == 'xgboost':
                self.models[fold].fit(X_train, y_train,
                                  eval_set=[(X_val, y_val)],
                                  verbose=self.verbose)
            else:
                self.models[fold].fit(X_train, y_train,
                                      eval_set=[(X_val, y_val)],
                                      )


            oof_preds[val_idx] = self.models[fold].predict(X_val)
            self.feature_importances += self.models[fold].feature_importances_ / self.n_splits
            if metric == 'rmse':
                PrintColor(f"Fold {fold + 1} RMSE: {np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))}", Fore.CYAN)
            elif metric == 'logloss':
                PrintColor(f"Fold {fold + 1} Logloss: {sklearn.metrics.log_loss(y_val, oof_preds[val_idx])}", Fore.CYAN)
        if metric == 'rmse':
            self.cv_score = np.sqrt(mean_squared_error(y, oof_preds))
        elif metric == 'logloss':
            self.cv_score = sklearn.metrics.log_loss(y, oof_preds)
        else:
            raise ValueError("Unsupported metric. Choose from 'rmse' or 'logloss'.")
        self.oof = oof_preds
        PrintColor(f"CV Score: {self.cv_score}", Fore.GREEN)
        return self.models
    


    
    def make_predictions(self, test_data):
        """
        Make predictions on the test data using the trained models.
        """
        test_preds = np.zeros((test_data.shape[0],))
        for model in self.models:
            test_preds += model.predict(test_data) / self.n_splits
        return test_preds
    
    def plot_feature_importance(self):
        """
        Plot the feature importance of the trained models.
        """
        if self.feature_importances is None:
            raise ValueError("Feature importances are not available. Train the model first.")
        
        try:
            feature_names = self.models[0].feature_names_
        except:
            try:
                feature_names = self.models[0].feature_names_in_
            except:
                feature_names = self.models[0].feature_name_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.feature_importances, y=feature_names)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()


target = train_df_FE.pop('Listening_Time_minutes')
test_df_FE.pop('Listening_Time_minutes')


X_train, X_test, y_train, y_test = train_test_split(train_df_FE, target, test_size=0.2, random_state=42)





# def objective(trial):
#     params = {
#         'iterations': trial.suggest_int('iterations', 100, 1000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'depth': trial.suggest_int('depth', 4, 10),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
#         'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
#         'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
#         'od_wait': trial.suggest_int('od_wait', 10, 50),
#         'verbose': False,
#         'random_seed': 42,
#         'task_type': 'GPU', 
#         'devices': list(range(get_gpu_count()))
#     }
    
#     model = CatBoostRegressor(**params)
#     model.fit(X_train,y_train, early_stopping_rounds=50)
    
#     preds = model.predict(X_test)
#     rmse = mean_squared_error(y_test, preds, squared=False)
    
#     return rmse


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=20)

# print('Number of finished trials:', len(study.trials))
# print('Best trial:')
# cat_best_params = study.best_trial

# print('  Value: {:.4f}'.format(cat_best_params.value))
# print('  Params: ')
# for key, value in cat_best_params.params.items():
#     print('    {}: {}'.format(key, value))



cat_params = {'iterations': 588, 
 'learning_rate': 0.03839426280073349,
 'depth': 7,
 'l2_leaf_reg': 1.1432771839920444,
 'random_strength': 6.276626304423243e-06,
 'bagging_temperature': 4.051203925374356,
 'od_type': 'Iter', 
 'od_wait': 24}


mt = ModelTrainer(n_splits=4,verbose=0)
mt.load_data(train_df_FE,target)



cat_models = mt.Trainmodel(model='catboost',type='regression',params=cat_params,metric='rmse',early_stopping_rounds=50,use_gpu=True)

##before num_col FE
# Training catboost model...
# Fold 1/5
# Fold 1 RMSE: 12.777225502944116
# Fold 2/5
# Fold 2 RMSE: 12.806031933509326
# Fold 3/5
# Fold 3 RMSE: 12.827681902334913
# Fold 4/5
# Fold 4 RMSE: 12.795241056753127
# Fold 5/5
# Fold 5 RMSE: 12.758352666879855
# CV Score: 12.792928709998645


mt.plot_feature_importance()


cat_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv',index_col='id')
cat_submission['Listening_Time_minutes'] = mt.make_predictions(test_df_FE)


cat_submission.to_csv('submission.csv')





