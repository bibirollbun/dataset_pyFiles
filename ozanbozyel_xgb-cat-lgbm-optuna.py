import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
pd.set_option('mode.use_inf_as_na', False)

import random

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from lightgbm import early_stopping, log_evaluation

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
set_seed(42)


def visualize_data(df1, df2):
    numeric_cols = list(set(df1.select_dtypes(include=['int64', 'float64']).columns) & 
                         set(df2.select_dtypes(include=['int64', 'float64']).columns))
    categorical_cols = list(set(df1.select_dtypes(include=['object', 'category']).columns) & 
                            set(df2.select_dtypes(include=['object', 'category']).columns))
    
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            sns.histplot(df1[col], bins=30, kde=True, ax=axes[0, 0], color='blue')
            axes[0, 0].set_title(f'{col} Histogram (Dataset 1)')
            sns.boxplot(x=df1[col], ax=axes[1, 0], color='blue')
            axes[1, 0].set_title(f'{col} Boxplot (Dataset 1)')
            
            sns.histplot(df2[col], bins=30, kde=True, ax=axes[0, 1], color='red')
            axes[0, 1].set_title(f'{col} Histogram (Dataset 2)')
            sns.boxplot(x=df2[col], ax=axes[1, 1], color='red')
            axes[1, 1].set_title(f'{col} Boxplot (Dataset 2)')
            
            plt.tight_layout()
            plt.show()
    else:
        print("No numeric columns!!")
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            df1[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='Blues', ax=axes[0])
            axes[0].set_ylabel('')
            axes[0].set_title(f'Pie Chart of {col} (Dataset 1)')
            
            df2[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='Reds', ax=axes[1])
            axes[1].set_ylabel('')
            axes[1].set_title(f'Pie Chart of {col} (Dataset 2)')
            
            plt.tight_layout()
            plt.show()
    else:
        print("No categorical columns!!")


def visualize_missing_data(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap='gray', cbar=False, yticklabels=False)
    plt.title('Missing Values Heatmap')
    plt.show()

def missing_values_summary(df):
    missing_summary = df.isnull().sum()
    total_values = df.shape[0]
    missing_summary = missing_summary[missing_summary > 0]
    
    if missing_summary.empty:
        print("No missing values in the dataset.")
    else:
        missing_percentage = (missing_summary / total_values) * 100
        missing_df = pd.DataFrame({'Missing Values': missing_summary, 'Percentage': missing_percentage})
        print("Missing values per column:")
        print(missing_df)


def apply_label_encoding(df_1, df_2):
    cat_columns = df_1.select_dtypes(include=['object', 'category']).columns
    num_columns = df_1.select_dtypes(include=['float', 'int']).columns

    for col in cat_columns:
        mode_value = df_2[col].mode()[0] if not df_2[col].mode().empty else None
        df_2[col].fillna(mode_value, inplace=True)

    for col in num_columns:
        median_value = df_2[col].median()
        df_2[col].fillna(median_value, inplace=True)
        
    for col in cat_columns:
        le = LabelEncoder()
        df_1[col] = le.fit_transform(df_1[col])
        df_2[col] = le.transform(df_2[col]) 

    scaler = StandardScaler()
    df_1[num_columns] = scaler.fit_transform(df_1[num_columns])
    df_2[num_columns] = scaler.transform(df_2[num_columns]) 
    
    return df_1, df_2


class XGBoost_Model_Optimizer:
    def __init__(self, n_trials, seed=42, use_gpu=False):
        self.n_trials = n_trials
        self.best_model = None
        self.study = None
        self.seed = seed
        self.use_gpu = use_gpu
        self.model_path = '/kaggle/working/xgb_best_model.json'
        self.rmse_history = []
        np.random.seed(seed)

    def objective_xgb(self, trial, X_train, X_test, y_train, y_test):
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),  
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'n_jobs': -1, 
            'random_state': self.seed
        }

        if self.use_gpu:
            param['tree_method'] = 'gpu_hist'  
        else:
            param['tree_method'] = 'hist'  

        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, verbose=False)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        self.rmse_history.append(rmse)
        return rmse

    def train_xgb(self, X_train, X_test, y_train, y_test):
        print("ğŸ“Œ Optimization process started...")
        self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        self.study.optimize(lambda trial: self.objective_xgb(trial, X_train, X_test, y_train, y_test), n_trials=self.n_trials)

        best_params = self.study.best_params
        print(f'ğŸ�† Best parameters found: {best_params}')

        self.best_model = xgb.XGBRegressor(**best_params)
        self.best_model.fit(X_train, y_train, verbose=False)

        best_rmse = min(self.rmse_history)        
        self.best_model.save_model(self.model_path)
        print(f'\nğŸ’¾ Best model saved at -> {self.model_path}')
        print(f'âœ… Best RMSE: {best_rmse:.4f}')

    def plt_learning_curve_xgb(self):
        if not self.rmse_history:
            print('â�Œ You need to train a model first!')
            return

        plt.plot(self.rmse_history, marker='x', linestyle='-')
        plt.xlabel('Trials')
        plt.ylabel('RMSE')
        plt.title('RMSE Learning Curve')
        plt.show()

    def predict_xgb(self, p_data):
        try: 
            pred_model = xgb.XGBRegressor()
            pred_model.load_model(self.model_path)
            return pred_model.predict(p_data)    
        except FileNotFoundError:
            print(f'ğŸš¨ Error: Model file "{self.model_path}" not found. You need to train a model first!')
        except Exception as e:
            print(f'âš ï¸� Unexpected error: {e}')


class CatBoost_Model_Optimizer:
    def __init__(self, n_trials=50, seed=42, use_gpu=False):
        self.n_trials = n_trials
        self.model_path = '/kaggle/working/catboost_best_model.cbm'
        self.seed = seed
        self.use_gpu = use_gpu
        self.best_model = None
        self.study = None
        self.rmse_history = []
        np.random.seed(seed)
    
    def objective_cat(self, trial, X_train, X_test, y_train, y_test):
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'RMSE',
            'random_seed': self.seed,
            'verbose': 0
        }

        if self.use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = '0'  

        model = cb.CatBoostRegressor(**params, early_stopping_rounds=10)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        self.rmse_history.append(rmse)
        return rmse               
    
    def train_cat(self, X_train, X_test, y_train, y_test):
        print("ğŸ“Œ Optimization process started...")
        self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        self.study.optimize(lambda trial: self.objective_cat(trial, X_train, X_test, y_train, y_test), n_trials=self.n_trials)
        
        best_params = self.study.best_params
        print(f'ğŸ�† Best Parameters: {best_params}')
        
        self.best_model = cb.CatBoostRegressor(**best_params)
        self.best_model.fit(X_train, y_train, verbose=False)
        self.best_model.save_model(self.model_path)

        best_rmse = min(self.rmse_history)
        print(f'\nğŸ’¾ Best model saved at -> {self.model_path}')
        print(f'âœ… Best RMSE: {best_rmse:.4f}')

    def plt_learning_curve_cat(self):
        if not self.rmse_history:
            print('â�Œ You need to train a model first!')
            return

        plt.plot(self.rmse_history, marker='x', linestyle='-')
        plt.xlabel('Trials')
        plt.ylabel('RMSE')
        plt.title('RMSE Learning Curve')
        plt.show()
        
    def predict_cat(self, p_data):
        try: 
            pred_model = cb.CatBoostRegressor()
            pred_model.load_model(self.model_path)
            return pred_model.predict(p_data)    
        except FileNotFoundError:
            print(f'ğŸš¨ Error: Model file "{self.model_path}" not found. You need to train a model first!')
        except Exception as e:
            print(f'âš ï¸� Unexpected error: {e}')



class LightGBM_Model_Optimizer:
    def __init__(self, n_trials=50, seed=42, use_gpu=False):
        self.n_trials = n_trials
        self.model_path = '/kaggle/working/lightgbm_best_model.txt'
        self.seed = seed
        self.use_gpu = use_gpu
        self.best_model = None
        self.study = None
        self.rmse_history = []
        np.random.seed(seed)
    
    def objective_lgb(self, trial, X_train, X_test, y_train, y_test):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),  
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),  
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),  
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'verbose': -1
        }

        if self.use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        self.rmse_history.append(rmse)
        return rmse
    
    def train_lgb(self, X_train, X_test, y_train, y_test):
        print("ğŸ“Œ Optimization process started...")
        self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        self.study.optimize(lambda trial: self.objective_lgb(trial, X_train, X_test, y_train, y_test), n_trials=self.n_trials)
        
        best_params = self.study.best_params
        print(f'ğŸ�† Best Parameters: {best_params}')
        
        self.best_model = lgb.LGBMRegressor(**best_params)
        self.best_model.fit(X_train, y_train)
        self.best_model.booster_.save_model(self.model_path)

        best_rmse = min(self.rmse_history)
        print(f'\nğŸ’¾ Best model saved at -> {self.model_path}')
        print(f'âœ… Best RMSE: {best_rmse:.4f}')

    def plt_learning_curve_lgb(self):
        if not self.rmse_history:
            print('â�Œ You need to train a model first!')
            return

        plt.plot(self.rmse_history, marker='x', linestyle='-')
        plt.xlabel('Trials')
        plt.ylabel('RMSE')
        plt.title('RMSE Learning Curve')
        plt.show()
        
    def predict_lgb(self, p_data):
        try: 
            pred_model = lgb.Booster(model_file=self.model_path)
            return pred_model.predict(p_data)    
        except FileNotFoundError:
            print(f'ğŸš¨ Error: Model file "{self.model_path}" not found. You need to train a model first!')
        except Exception as e:
            print(f'âš ï¸� Unexpected error: {e}')



df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
df_train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

df_train["Laptop Compartment"] = df_train["Laptop Compartment"].astype("category")
df_train_extra["Laptop Compartment"] = df_train_extra["Laptop Compartment"].astype("category")
df_test["Laptop Compartment"] = df_test["Laptop Compartment"].astype("category")


visualize_data(df_train, df_train_extra)


missing_values_summary(df_train)
missing_values_summary(df_train_extra)


visualize_missing_data(df_train)
visualize_missing_data(df_train_extra)


df_train = df_train.dropna()
df_train_extra = df_train_extra.dropna()
missing_values_summary(df_train)
missing_values_summary(df_train_extra)


df_tr = pd.concat([df_train, df_train_extra], ignore_index=True)

y = df_tr['Price']
test_id = df_test['id']

df_tr = df_tr.drop(columns=['Price','id'])
df_test = df_test.drop(columns=['id'])

df_train, df_test = apply_label_encoding(df_tr, df_test)


X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.2, random_state=42)


xgb_model = XGBoost_Model_Optimizer(n_trials=50)
xgb_model.train_xgb(X_train, X_test, y_train, y_test)
xgb_model.plt_learning_curve_xgb()


cat_model = CatBoost_Model_Optimizer(n_trials=50)
cat_model.train_cat(X_train, X_test, y_train, y_test)
cat_model.plt_learning_curve_cat()


lgb_model = LightGBM_Model_Optimizer(n_trials = 50)
lgb_model.train_lgb(X_train, X_test, y_train, y_test)
lgb_model.plt_learning_curve_lgb()


#predict = xgb_model.predict_xgb(df_test)
prediction = cat_model.predict_cat(df_test)
#lgb_model.predict_lgb(df_test)


prediction


output = pd.DataFrame({
    'Id': test_id, 
    'SalePrice': prediction
})

output


output.to_csv('submission.csv', index=False)

