import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from tqdm import tqdm
from colorama import Fore, Style, init


from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import *

from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')

from IPython.display import clear_output


# importing the data

train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


train_df.drop('id', axis = 1, inplace = True)
test_df.drop('id', axis = 1, inplace = True)


def display_info(train_df, test_df):
    '''Displays head, info, describe, missing values of both train_df and test_df'''
    for data, label in zip([train_df, test_df], ['Train', 'Test']):
        print(Style.BRIGHT + Fore.BLUE + f'\n{label} head \n')
        display(data.head())

        print(Style.BRIGHT + Fore.BLUE + f'\n{label} info \n' + Style.RESET_ALL)
        display(data.info())

        print(Style.BRIGHT + Fore.BLUE + f'\n{label} describe \n')
        display(data.describe().T)

        print(Style.BRIGHT + Fore.BLUE + f'\n{label} missing values \n' + Style.RESET_ALL)
        display(data.isnull().sum())
        print("------------------------------------------------------------------")

display_info(train_df, test_df)


for col in ('country', 'store', 'product'):
    print(f"Unique values in {col} -> ", train_df[col].unique())


# For starter analysis we drop nan containing rows

train_df.dropna(inplace = True)


def process_date(df : pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'])

    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    
    df.drop('date',axis=1,inplace=True)
    
    return df

def preprocess(df : pd.DataFrame):
    # Convert columns to category
    df[ df.select_dtypes(include='object').columns ] = df.select_dtypes(include='object').astype("category")
    return process_date(df)
    


train_df = preprocess(train_df)
test_df = preprocess(test_df)



cat_cols = train_df.select_dtypes(exclude = ['number']).columns.to_list()


X = train_df.drop('num_sold', axis = 1)
y = train_df['num_sold']
test = test_df


class Config:
    
    seed = 42
    n_splits = 10
    n_repeats = 1
    fold_type = 'RKF'
    cat_features = None
    
    OOF_preds = pd.DataFrame()
    TEST_preds = pd.DataFrame()
    scores_df = pd.DataFrame(columns = ['Score'])


class Model(Config):
    
    def __init__(self, X, X_enc, y, test, test_enc, models):
        super().__init__()
        
        self.y = y
        self.models = models
        self.cat_c = list(X.select_dtypes(exclude = ['number']).columns)

    def getCVScheme(self):
        if self.fold_type == 'SKF':
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.fold_type == 'KF':
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.fold_type == 'GKF':
            kfold = GroupKFold(n_splits=self.n_splits)
        elif self.fold_type == 'RKF':
            kfold = RepeatedKFold(n_splits=self.n_splits, n_repeats = self.n_repeats, random_state=self.seed)
        else:
            raise NotImplementedError("Select the Given Cv Statergy")
        return kfold
        

    def train(self, X, X_enc, test, test_enc):
        
        folds = self.getCVScheme()
        

        for model_name, [model, params] in tqdm(self.models.items()):
            print('='*5, f'Training : {model_name}', '='*5)

            if any(name in model_name for name in ["XGB", "CAT", "LGBM"]):
                self.X = X
                self.test = test
            else :
                self.X = X_enc
                self.test = test_enc
                
            self.cat_features_indices = [self.X.columns.get_loc(col) for col in self.cat_features] if 'CAT' in model_name and self.cat_features != None else None
            
            for n_fold, (train_idx, val_idx) in enumerate(tqdm(folds.split(self.X, self.y), desc = "Training Folds", total = self.n_splits)): 
                X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
                X_val, y_val = self.X.iloc[val_idx], self.y.iloc[val_idx]

                oof_preds = pd.DataFrame(columns = [model_name], index = X_val.index)
                test_preds = pd.DataFrame(columns = [model_name], index = self.test.index)

                model = self.model_train_decision(model_name, params, model, X_train, y_train, X_val, y_val)

                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                test_pred = model.predict(self.test)

                oof_preds[model_name] = y_val_pred
                test_preds[model_name] = test_pred

                train_score = mean_absolute_percentage_error(y_train, y_train_pred)
                val_score = mean_absolute_percentage_error(y_val, y_val_pred)

                print(f"Fold {n_fold+1} - Train RMSLE: {train_score:.4f}, Validation RMSLE: {val_score:.4f}")

                self.scores_df.loc[f'{model_name}', f'{n_fold + 1}'] = val_score
                self.OOF_preds = pd.concat([self.OOF_preds, oof_preds], axis = 0, ignore_index = False)
                self.TEST_preds = pd.concat([self.TEST_preds, test_preds], axis = 0, ignore_index = False)

            self.OOF_preds = self.OOF_preds.groupby(level = 0).mean()
            self.TEST_preds = self.TEST_preds.groupby(level = 0).mean()
            
            self.scores_df.loc[f'{model_name}', 'Score'] = self.scores_df.loc[f'{model_name}'][1:].mean()
            self.scores_df.sort_values('Score')
            
        return self.OOF_preds, self.TEST_preds, self.scores_df


    def model_train_decision(self, model_name, params, model, X_train, y_train, X_val, y_val):
        if "LGBM" in model_name:
                callbacks = [lgb.early_stopping(stopping_rounds = e_stop, verbose = False)]
                model = lgb.LGBMRegressor(**params, random_state = self.seed, verbose = -1, njobs = -1, device = 'cpu')
                model.fit(X_train, y_train, eval_set = [(X_val, y_val)],#eval_metric = '', # change error metric!
                          callbacks = callbacks) 

        elif "CAT" in model_name:
            # model = CatBoostRegressor(**params, random_state = SEED, verbose = 0)
            train_pool = Pool(data=X_train, label=y_train, cat_features = self.cat_features_indices)
            val_pool = Pool(data=X_val, label=y_val, cat_features = self.cat_features_indices)
            
            model = CatBoostRegressor(**params, random_state=self.seed, verbose=0, task_type='CPU')
            model.fit(X_train, y_train, 
                      eval_set = (X_val, y_val),
                      cat_features = self.cat_c,
                      early_stopping_rounds=100,
                      verbose = 0)
            
        elif "XGB" in model_name:
            model = XGBRegressor(**params,random_state = self.seed, objective= "reg:squarederror", enable_categorical=True, verbosity = 0)
            model.fit(X_train, y_train,
                     eval_set = [(X_val, y_val)],
                     early_stopping_rounds = 100,
                     verbose = 0)
        else :
            model.fit(X_train, y_train)

        return model



e_stop = 50


xgb_params = {'n_estimators': 779, 'learning_rate': 0.05040967684293959, 
                  'max_depth': 15, 'min_child_weight': 8, 'subsample': 0.7862803701842613, 
                  'colsample_bytree': 0.826716757679502, 'gamma': 0.030960361476628846, 
                  'reg_alpha': 0.035778578798813854, 'reg_lambda': 2.4687457931229737}

lgbm_params = {'n_estimators': 719, 'learning_rate': 0.2547321215651838, 'max_depth': -1, 
                  'num_leaves': 93, 'min_child_samples': 12, 'subsample': 0.7469457510257034, 
                  'colsample_bytree': 0.8574364208874832, 'reg_alpha': 0.4432890229534274, 
                  'reg_lambda': 0.5722929283639382, "verbosity" : -1}


models = {
    'LGBM_1' : ['', lgbm_params],
    'XGB_1' : ['', xgb_params]
}




training_object = Model(X = X, X_enc = X, y = y, 
                        test = test, test_enc = test, 
                        models = models)

oof_preds_df, test_preds_df, final_scores_df = training_object.train(X = X, X_enc = X, test = test, test_enc = test)


final_scores_df


sample['num_sold'] = (
    (0.61803)*test_preds_df['LGBM_1'] + 
    (0.38196)*test_preds_df['XGB_1']
)

sample.to_csv('submission.csv', index = False)

sample.head()




