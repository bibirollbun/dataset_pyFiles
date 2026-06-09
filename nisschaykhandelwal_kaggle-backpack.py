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
import re
import time
from functools import partial
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from IPython.display import Image

from scipy.optimize import minimize
from scipy.stats import mstats
from scipy import stats

from sklearn.linear_model import (SGDOneClassSVM, LinearRegression, Ridge, 
                                 Lasso, ElasticNet)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                            mean_absolute_percentage_error)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import (KNNImputer, SimpleImputer)
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, IsolationForest, BaggingRegressor,
                              RandomForestRegressor)
from sklearn.model_selection import (StratifiedKFold, KFold, StratifiedGroupKFold,
                                     RepeatedStratifiedKFold, RepeatedKFold, cross_validate,
                                     train_test_split, TimeSeriesSplit)
from sklearn.preprocessing import (LabelEncoder, QuantileTransformer, StandardScaler,
                                   PowerTransformer, MaxAbsScaler, MinMaxScaler,
                                   RobustScaler, PolynomialFeatures, OrdinalEncoder, 
                                    OneHotEncoder,FunctionTransformer, KBinsDiscretizer)
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn import preprocessing
from sklearn.feature_selection import (VarianceThreshold, SequentialFeatureSelector, f_regression)
from sklearn.compose import ColumnTransformer

import requests
import holidays

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import optuna
from optuna.samplers import CmaEsSampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis
from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
%matplotlib inline

sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")




train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
original_data = pd.read_csv('/kaggle/input/og-data/Noisy_Student_Bag_Price_Prediction_Dataset.csv')




train.info()


train.columns


train.describe


train.dtypes[train.dtypes != 'object']


plt.scatter(x='Compartments', y='Price', data=train)


plt.scatter(x='Weight Capacity (kg)', y='Price', data=train)


train.shape


train.isnull().sum()


categorical_columns = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





class DataVisualizer:
    def __init__(self, df):
        self.df = df
        
    def plot_numerical_features(self, figsize=(14, 6)):
        num_features = self.df.select_dtypes(include=[np.number]).columns
        ncols = 2
        nrows = (len(num_features) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                               figsize=(figsize[0], figsize[1] * nrows))
        axes = axes.flatten()
        
        for i, feature in enumerate(num_features):
            sns.histplot(self.df[feature], bins=30, kde=True, 
                        ax=axes[i], color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribution of {feature}', fontsize=12)
            axes[i].set_xlabel(feature, fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()
        
    
    def plot_numerical_boxplots(self, figsize=(14, 6)):
        num_features = self.df.select_dtypes(include=[np.number]).columns
        ncols = 2
        nrows = (len(num_features) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                               figsize=(figsize[0], figsize[1] * nrows))
        axes = axes.flatten()
        
        for i, feature in enumerate(num_features):
            sns.boxplot(x=self.df[feature], ax=axes[i], color='lightgreen')
            axes[i].set_title(f'Boxplot of {feature}', fontsize=12)
            axes[i].set_xlabel(feature, fontsize=10)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, method='spearman'):
        num_df = self.df.select_dtypes(include=[np.number])
        corr = num_df.corr(method=method)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                    square=True, linewidths=0.5)
        plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=14)
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()
        
    def plot_qq_plot(self, figsize=(14, 6)):
        num_features = self.df.select_dtypes(include=[np.number]).columns
        ncols = 2
        nrows = (len(num_features) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                               figsize=(figsize[0], figsize[1] * nrows))
        axes = axes.flatten()
        
        for i, feature in enumerate(num_features):
            stats.probplot(self.df[feature], dist="norm", plot=axes[i])
            axes[i].set_title(f'QQ Plot of {feature}', fontsize=12)
            axes[i].set_xlabel('Theoretical Quantiles', fontsize=10)
            axes[i].set_ylabel('Sample Quantiles', fontsize=10)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()
        
    def plot_pairplot(self):
        num_features = self.df.select_dtypes(include=[np.number]).columns
        sns.pairplot(self.df[num_features], diag_kind='kde', 
                    plot_kws={'alpha': 0.6, 'edgecolor': 'k'}, height=2.5)
        plt.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=14)
        plt.show()
        
    def plot_categorical_features(self, ncols=2, top_n=None):
        cat_features = self.df.select_dtypes(include=['object']).columns
        nrows = (len(cat_features) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                               figsize=(14, 6 * nrows))
        axes = axes.flatten()
        
        for i, feature in enumerate(cat_features):
            if top_n:
                top_categories = self.df[feature].value_counts().nlargest(top_n).index
                data = self.df[self.df[feature].isin(top_categories)]
                sns.countplot(data=data, y=feature, ax=axes[i], 
                            palette='viridis', order=top_categories)
            else:
                sns.countplot(data=self.df, y=feature, ax=axes[i], 
                            palette='viridis')
            
            axes[i].set_title(f'Count of {feature}', fontsize=12)
            axes[i].set_xlabel('Count', fontsize=10)
            axes[i].set_ylabel(feature, fontsize=10)
            axes[i].tick_params(axis='y', rotation=0)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()


class DataPreprocessor:
    @staticmethod
    def PolynomialFeatures_labeled(input_df, power):
        poly = preprocessing.PolynomialFeatures(power)
        output_nparray = poly.fit_transform(input_df)
        powers_nparray = poly.powers_
        
        input_feature_names = list(input_df.columns)
        target_feature_names = ["Constant Term"]
        
        for feature_distillation in powers_nparray[1:]:
            intermediary_label = ""
            final_label = ""
            for i in range(len(input_feature_names)):
                if feature_distillation[i] == 0:
                    continue
                else:
                    variable = input_feature_names[i]
                    power = feature_distillation[i]
                    intermediary_label = "%s+%d" % (variable, power)
                    if final_label == "":
                        final_label = intermediary_label
                    else:
                        final_label = final_label + "x" + intermediary_label
            target_feature_names.append(final_label)
            
        output_df = pd.DataFrame(output_nparray, columns=target_feature_names)
        return output_df
    
    @staticmethod
    def variance_threshold(df, th):
        var_thres = VarianceThreshold(threshold=th)
        var_thres.fit(df)
        new_cols = var_thres.get_support()
        return df.iloc[:, new_cols]
    
    @staticmethod
    def optimize_memory_usage(df, print_size=True):
        start_memory = df.memory_usage().sum() / 1024**2
        
        for col in df.select_dtypes(include=['int', 'float']).columns:
            col_type = df[col].dtype
            
            try:
                if col_type.kind == 'i':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='float')
            except Exception as e:
                logger.warning(f"Could not optimize column {col}: {str(e)}")
        
        end_memory = df.memory_usage().sum() / 1024**2
        savings = (start_memory - end_memory) / start_memory * 100
        
        if print_size:
            logger.info(f"Memory usage reduced from {start_memory:.2f}MB to {end_memory:.2f}MB ({savings:.1f}% savings)")
        
        return df




original_data = original_data.dropna()
train = pd.concat([train, original_data], axis=0).reset_index(drop=True)

train.shape, test.shape




train.describe().T




duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

train = train.drop_duplicates()




visualizer = DataVisualizer(train)



visualizer.plot_numerical_features()


visualizer.plot_numerical_boxplots()


visualizer.plot_correlation_matrix()


visualizer.plot_categorical_features()


visualizer.plot_pairplot()


train = DataPreprocessor.optimize_memory_usage(train)
test  = DataPreprocessor.optimize_memory_usage(test)
original_data = DataPreprocessor.optimize_memory_usage(original_data)



def preproc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    weight_col = "Weight Capacity (kg)"
    if weight_col not in df.columns:
        raise KeyError(f"Required column '{weight_col}' not found in DataFrame")
    
    WEIGHT_CLASSES: Dict[str, Dict[str, Union[float, str]]] = {
        'Light': {'max': 5, 'min': float('-inf')},
        'Middle': {'max': 15, 'min': 5},
        'Light_heavy': {'max': 20, 'min': 15},
        'Middel_heavy': {'max': 25, 'min': 20},
        'Heavy': {'max': float('inf'), 'min': 25}
    }
    
    median_weight = df[weight_col].median()
    df[weight_col] = df[weight_col].fillna(median_weight)
    
    if (df[weight_col] < 0).any():
        raise ValueError("Negative weight capacity values found")
    
    conditions = [
        (df[weight_col] > class_info['min']) & (df[weight_col] <= class_info['max'])
        for class_info in WEIGHT_CLASSES.values()
    ]
    choices = list(WEIGHT_CLASSES.keys())
    
    df['Weight_Class'] = np.select(conditions, choices, default='Unknown')
    
    df[weight_col] = df[weight_col].astype('float64')
    df['Weight_Class'] = df['Weight_Class'].astype('category')
    
    object_columns = df.select_dtypes(include=[object]).columns
    if len(object_columns) > 0:
        df[object_columns] = df[object_columns].fillna("None").astype('category')
    
    return df


preproc(train)
preproc(test)

train.shape, test.shape


num_f = ['Compartments', 'Weight Capacity (kg)']           # Numerical features
ohe_f = ['Brand', 'Material', 'Size', 'Color', 'Style', 'Waterproof']  # Categorical features for one-hot encoding
ord_f = ['Laptop Compartment'] #ordinal encoding


# Pipeline for categorical features (one-hot encoding)
ohe_pipe = Pipeline([
    ('imputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),  
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),  
    ('imputer_ohe_after', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))  
])

# Pipeline for ordinal features
ord_pipe = Pipeline([
    ('imputer_before', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),  
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),  # Convert to ordered numbers
    ('simpleImputer_after', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))  # Handle any remaining missing values
])

# Pipeline for numerical features
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
    ('scaler', StandardScaler())  
])


preprocessor = ColumnTransformer(
    [
        ('ohe', ohe_pipe, ohe_f),    
        ('ord', ord_pipe, ord_f),     
        ('num', num_pipe, num_f),    
    ], 
    remainder='passthrough'          
)


X=train.drop(columns=['Price'])
y=train['Price']

X_transformed=preprocessor.fit_transform(X,y)
test_transformed=preprocessor.transform(test)

X=pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
test=pd.DataFrame(test_transformed, columns=preprocessor.get_feature_names_out())

X = DataPreprocessor.variance_threshold(X,0.01)
list_name = (X.columns)
test = test[list_name]

X.shape, y.shape, test.shape


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def objective(trial):
    depth = trial.suggest_int('depth', 4, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    iterations = trial.suggest_int('iterations', 100, 2000)
    l2_leaf_reg = trial.suggest_int('l2_leaf_reg', 1, 10)
    bagging_temperature = trial.suggest_uniform('bagging_temperature', 0, 1)
    border_count = trial.suggest_int('border_count', 1, 255)
    random_strength = trial.suggest_int('random_strength', 1, 10)
    early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 10, 50)

    model = CatBoostRegressor(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        bagging_temperature=bagging_temperature,
        border_count=border_count,
        random_strength=random_strength,
        early_stopping_rounds=early_stopping_rounds,
        silent=True
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        preds = model.predict(X_test)
        score = rmse(y_test, preds)
        scores.append(score)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=100)

cat_param = study.best_params

print("Best parameters found: ", cat_param)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e2),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e2),
    }

    model = xgb.XGBRegressor(**params, silent=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        score = rmse(y_test, preds)
        scores.append(score)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=100)

xgb_param = study.best_params

print("Best parameters found: ", xgb_param)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', -1, 10),  
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e2),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e2),
    }

    model = LGBMRegressor(**params, verbose=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = model.predict(X_test)
        score = rmse(y_test, preds)
        scores.append(score)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=100)

lgb_param = study.best_params

print("Best parameters found: ", lgb_param)


def objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'l2_regularization': trial.suggest_loguniform('l2_regularization', 1e-4, 1e2),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
    }

    model = HistGradientBoostingRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = rmse(y_test, preds)
        scores.append(score)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=50)

hgb_param = study.best_params

print("Best parameters found: ", hgb_param)


fold = 5
FOLDs = KFold(n_splits=fold, shuffle=True)

# Initialize arrays for predictions
oof_cat, predictions_cat = np.zeros(len(X)), np.zeros(len(test))
oof_xgb, predictions_xgb = np.zeros(len(X)), np.zeros(len(test))
oof_lgb, predictions_lgb = np.zeros(len(X)), np.zeros(len(test))
oof_hgb, predictions_hgb = np.zeros(len(X)), np.zeros(len(test))


for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(X, y)):
    print(f'Fold {fold_+1}/{fold}')
    
    # Properly assign train and validation sets
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    # CatBoostRegressor
    cat_model = CatBoostRegressor(**cat_param, 
                                 random_state=42, 
                                 verbose=0)
    cat_model.fit(X_train, y_train)
    oof_cat[val_idx] = cat_model.predict(X_valid)
    predictions_cat += cat_model.predict(test) / FOLDs.n_splits
    cat_score = mean_squared_error(y_valid, oof_cat[val_idx], squared=False)
    print(f'Fold {fold_+1} CatBoostRegressor oof RMSE is --- {cat_score}')

    # XGBRegressor
    xgb_model = xgb.XGBRegressor(**xgb_param,
                                random_state=42)
    xgb_model.fit(X_train, y_train)
    oof_xgb[val_idx] = xgb_model.predict(X_valid)
    predictions_xgb += xgb_model.predict(test) / FOLDs.n_splits
    xgb_score = mean_squared_error(y_valid, oof_xgb[val_idx], squared=False)
    print(f'Fold {fold_+1} XGBRegressor oof RMSE is --- {xgb_score}')

    # LGBMRegressor
    lgb_model = LGBMRegressor(**lgb_param,
                             random_state=42,
                             verbose=-1)
    lgb_model.fit(X_train, y_train)
    oof_lgb[val_idx] = lgb_model.predict(X_valid)
    predictions_lgb += lgb_model.predict(test) / FOLDs.n_splits
    lgb_score = mean_squared_error(y_valid, oof_lgb[val_idx], squared=False)
    print(f'Fold {fold_+1} LGBMRegressor oof RMSE is --- {lgb_score}')

    # HistGradientBoostingRegressor
    hgb_model = HistGradientBoostingRegressor(**hgb_param,
                                             random_state=42)
    hgb_model.fit(X_train, y_train)
    oof_hgb[val_idx] = hgb_model.predict(X_valid)
    predictions_hgb += hgb_model.predict(test) / FOLDs.n_splits
    hgb_score = mean_squared_error(y_valid, oof_hgb[val_idx], squared=False)
    print(f'Fold {fold_+1} HistGradientBoostingRegressor oof RMSE is --- {hgb_score}')

   



blend_df = pd.DataFrame({'1': oof_cat,
                         '2': oof_xgb,
                         '3': oof_lgb,
                         '4': oof_hgb,
                         
                         })

blend_test_df = pd.DataFrame({  '1': predictions_cat,  
                                '2': predictions_xgb, 
                                '3': predictions_lgb, 
                                '4': predictions_hgb,  
                                
                        })

def calculate_rmse(weights, blend_df, y_):
    weighted_predictions = np.dot(blend_df, weights)
    return np.sqrt(mean_squared_error(y, weighted_predictions))

def constraint(weights):
    return np.sum(weights) - 1 

initial_weights = np.array([0.2] * blend_df.shape[1])  

constraints = {'type': 'eq', 'fun': constraint}
bounds = [(0, 1) for _ in range(blend_df.shape[1])]  

result = minimize(calculate_rmse, initial_weights, args=(blend_df, y), 
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
optimal_rmse = result.fun

print(f"Optimal weights: {optimal_weights}")
print(f"Best RMSE: {optimal_rmse:.4f}")


sample = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sample['Price'] = np.dot(blend_test_df, optimal_weights)
sample.to_csv('submission.csv', index=False)
sample.shape




