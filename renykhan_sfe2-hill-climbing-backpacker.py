import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from tqdm import tqdm
from colorama import Fore, Style, init
from IPython.display import clear_output, display, HTML

import warnings
warnings.filterwarnings('ignore')

# sns.set_style('darkgrid')


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, label_binarize, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping, LGBMRegressor, Dataset
from xgboost import DMatrix, XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class Config:
    state = 42
    early_stop = 100

    train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col = 'id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col = 'id')
    train_org = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')
    sample = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
    
    target = 'Price'

    fold_type = 'RKF'
    n_splits = 10
    n_repeats = 1

    original_data = 'Y'
    feature_eng = 'N' 
    missing = 'Y'      # To impute the rest of the values
    outliers = 'N'     # To remove outliers 
    log_trf = 'N'
    scaler_trf = 'N'


class EDA(Config):
    def __init__(self):
        super().__init__()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object']).columns.tolist()
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object']).columns.tolist()

        self.display_info()
        self.num_feature_plots()
        self.cat_feature_plots()
        self.target_distribution()


    def display_info(self):
        
        for data, label in zip([self.train, self.train_org, self.test], ['Train', 'Original', 'Test']):
            print(Style.BRIGHT+Fore.BLUE+f'\n Length of {label} : {len(data)}\n'+Style.RESET_ALL)
            
            print(Style.BRIGHT+Fore.BLUE+f'\n{label} head\n'+Style.RESET_ALL)
            display(data.head())
                           
            print(Style.BRIGHT+Fore.BLUE+f'\n{label} info\n'+Style.RESET_ALL)               
            display(data.info())
                           
            print(Style.BRIGHT+Fore.BLUE+f'\n{label} describe\n'+Style.RESET_ALL)
            display(data.describe().drop(index='count', columns=self.target, errors = 'ignore').T)
            
            print(Style.BRIGHT+Fore.BLUE+f'\n{label} missing values\n'+Style.RESET_ALL)               
            display(data.isna().sum())
            print('-'*100)
        return self

    def num_feature_plots(self):

        print(Style.BRIGHT+Fore.GREEN+f'\n Numerical Feature Distribution \n'+Style.RESET_ALL)

        # Define a custom color palette
        custom_palette = ['#3498db', '#e74c3c','#2ecc71']
        
        train_data, test_data, original_data = self.train.copy(), self.test.copy(), self.train_org.copy()
        
        # Add 'Dataset' column to distinguish between train and test data
        train_data['Dataset'] = 'Train'
        test_data['Dataset'] = 'Test'
        original_data['Dataset'] = 'Original'
        
        variables = self.num_features
        
        # Function to create and display a row of plots for a single variable
        def create_variable_plots(variable):
            sns.set_style('whitegrid')
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
            # Box plot
            plt.subplot(1, 2, 1)
            sns.boxplot(data=pd.concat([train_data, test_data,original_data.dropna()]), x=variable, y="Dataset", palette=custom_palette)
            plt.xlabel(variable)
            plt.title(f"Box Plot for {variable}")
        
            # Separate Histograms
            plt.subplot(1, 2, 2)
            sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
            sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
            sns.histplot(data=original_data.dropna(), x=variable, color=custom_palette[2], kde=True, bins=30, label="Original")
            plt.xlabel(variable)
            plt.ylabel("Frequency")
            plt.title(f"Histogram for Numerical Feature : {variable} [TRAIN, TEST & ORIGINAL]")
            plt.legend()
        
            # Adjust spacing between subplots
            plt.tight_layout()
        
            # Show the plots
            plt.show()
        
        # Perform univariate analysis for each variable
        for variable in variables:
            create_variable_plots(variable)
        
        # Drop the 'Dataset' column after analysis
        train_data.drop('Dataset', axis=1, inplace=True)
        test_data.drop('Dataset', axis=1, inplace=True)
        original_data.drop('Dataset', axis=1, inplace=True)
        
        print('-'*100)


    def cat_feature_plots(self):
        print(Style.BRIGHT+Fore.BLUE+f'\n Categorical Feature Distribution \n'+Style.RESET_ALL)
        
        pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']
        countplot_color = '#5C67A3'

        train_data, test_data, original_data = self.train.copy(), self.test.copy(), self.train_org.copy()
        
        # Function to create and display a row of plots for a single categorical variable
        def create_categorical_plots(variable):
            sns.set_style('whitegrid')
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
            # Pie Chart
            plt.subplot(1, 2, 1)
            train_data[variable].value_counts().plot.pie(
                autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3), startangle=140
            )
            plt.title(f"Pie Chart for {variable}")
        
            # Bar Graph
            plt.subplot(1, 2, 2)
            sns.countplot(
                data=pd.concat([train_data, test_data, original_data.dropna()]), 
                x=variable, 
                color=countplot_color,  # Using a single color for the countplot
                alpha=0.8  # Setting 80% opacity
            )
            plt.xlabel(variable)
            plt.ylabel("Count")
            plt.title(f"Bar Graph for {variable} [TRAIN, TEST & ORIGINAL Combined]")
        
            # Adjust spacing between subplots
            plt.tight_layout()
            
            # Show the plots
            plt.show()
        
        # Perform univariate analysis for each categorical variable
        for variable in self.cat_features:
            create_categorical_plots(variable)

        print('-'*100)


    def target_distribution(self):
        print(Style.BRIGHT+Fore.GREEN+f'\n Target Feature Distribution \n'+Style.RESET_ALL)
        
        # Define a custom color palette
        target_palette = ['#3498db', '#e74c3c']
        
        train_data, test_data, original_data = self.train.copy(), self.test.copy(), self.train_org.copy()
        
        # Add 'Dataset' column to distinguish between Train and Original data
        train_data['Dataset'] = 'Train'
        original_data['Dataset'] = 'Original'
        
        # Function to create and display a row of plots for the target variable
        def create_target_plots(target_variable):
            sns.set_style('whitegrid')
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
            # Box Plot
            plt.subplot(1, 2, 1)
            sns.boxplot(data=pd.concat([train_data, original_data.dropna()]), x=target_variable, y="Dataset", palette=target_palette)
            plt.xlabel(target_variable)
            plt.title(f"Box Plot for Target Feature '{target_variable}'")
        
            # Histogram
            plt.subplot(1, 2, 2)
            sns.histplot(data=train_data, x=target_variable, color=target_palette[0], kde=True, bins=30, label="Train")
            sns.histplot(data=original_data.dropna(), x=target_variable, color=target_palette[1], kde=True, bins=30, label="Original")
            plt.xlabel(target_variable)
            plt.ylabel("Frequency")
            plt.title(f"Histogram for Target Feature '{target_variable}' [TRAIN & ORIGINAL]")
            plt.legend()
        
            # Adjust spacing between subplots
            plt.tight_layout()
        
            # Show the plots
            plt.show()
        
        # Perform univariate analysis for the target variable
        create_target_plots(self.target)
        
        # Drop the 'Dataset' column after analysis
        train_data.drop('Dataset', axis=1, inplace=True)
        original_data.drop('Dataset', axis=1, inplace=True)
    
        print('-'*100)
        
        


e = EDA()


class Transform(Config):
    
    def __init__(self):
        super().__init__()

        
        if self.original_data == 'Y':
            self.train = pd.concat([self.train, self.train_org], ignore_index=True).drop_duplicates()
            self.train.dropna(subset = [self.target], inplace = True)
            self.train.reset_index(drop=True, inplace=True)

        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()

        self.train_raw = self.train.copy()
        self.test_raw = self.test.copy()
        
        if self.missing == 'Y':
            self.missing_values()
        
        if self.feature_eng == 'Y':
            self.train = self.new_features(self.train)
            self.test = self.new_features(self.test)
            self.train_raw = self.new_features(self.train_raw)
            
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
            
        if self.outliers == 'Y':    
            self.remove_outliers()
            
        if self.log_trf == 'Y':
            self.log_transformation()
            
        if self.scaler_trf == 'Y':
            self.scaler()
            
        self.train_enc = self.train.copy()
        self.test_enc = self.test.copy()
        self.transform()
        self.encode()

        
    def __call__(self):

        self.train[self.cat_features] = self.train[self.cat_features].astype('category')
        self.test[self.cat_features] = self.test[self.cat_features].astype('category')

        self.cat_features_card = []
        for f in self.cat_features:
            self.cat_features_card.append(self.train[f].nunique())

        self.train = self.reduce_mem(self.train)
        self.test = self.reduce_mem(self.test)
        
        self.y = self.train[self.target]
        self.train = self.train.drop(self.target, axis=1)
        self.train_enc = self.train_enc.drop(self.target, axis=1)
        
        return self.train, self.train_enc, self.y, self.test, self.test_enc, self.cat_features
    
    def transform(self):
        
        '''
        Threshold for skewness : 
        
        -0.5 to 0.5 => approx. symmetric
        -1 to -0.5 or 0.5 to 1 => Moderetly skewed
        <-1 or >1 => Highly skewed
        '''
        
        # For numerical columns we use mean if skewness is b/w -1 to 1 else we use median
        # For categorical columns we will use most frequent

        def data_imputation_pipeline(df : pd.DataFrame):

            # seperate numerical and categorical columns
            numerical_cols = df.select_dtypes(include = ["number"]).columns
            categorical_cols = df.select_dtypes(include = ["object", "category"]).columns
        
            # define cols to use mean and those on which to use median
            mean_numerical_cols = [col for col in numerical_cols if abs(df[col].skew()) <= 1]
            median_numerical_cols = [col for col in numerical_cols if abs(df[col].skew()) > 1]
        
            # define transformers for numerical and categorical data
            mean_numerical_transformer = SimpleImputer(strategy = "mean")
            median_numerical_transformer = SimpleImputer(strategy = "median")
            categorical_transformer = SimpleImputer(strategy = "most_frequent")
        
            # Combine transformers using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num1", mean_numerical_transformer, mean_numerical_cols),
                    ("num2", median_numerical_transformer, median_numerical_cols),
                    ("cat", categorical_transformer, categorical_cols),
                ]
            )
        
            # create a pipeline
            pipeline = Pipeline(steps = [("preprocessor", preprocessor)])
        
            return pipeline, mean_numerical_cols, median_numerical_cols, categorical_cols

        def update_df(train_df, X, test_df):
    
            pipeline, mean_cols, median_cols, cat_cols = data_imputation_pipeline(X)
        
            # Fit-transform the training data
            transformed_X = pipeline.fit_transform(X)
            transformed_test_df = pipeline.fit_transform(test_df)
            
            # Convert back to DataFrame with proper column names
            column_order = mean_cols + median_cols + list(cat_cols)
            
            X = pd.DataFrame(transformed_X, columns=column_order)
            test_df = pd.DataFrame(transformed_test_df, columns=column_order)
            
            # Restore original data types
            for col in mean_cols + median_cols:
                X[col] = pd.to_numeric(X[col])
                test_df[col] = pd.to_numeric(test_df[col])
            
            for col in cat_cols:
                X[col] = X[col].astype(train_df[col].dtype)
                test_df[col] = test_df[col].astype(train_df[col].dtype)
                
            # Convert object to category type
            X = X.apply(lambda x: x.astype('category') if x.dtype == 'object' else x)
            test_df = test_df.apply(lambda x: x.astype('category') if x.dtype == 'object' else x)

            return X, test_df

        
        X = self.train_raw.drop(self.target, axis = 1)
        X_trf, test_trf = update_df(self.train_raw, X, self.test_raw)

        self.train = pd.concat([X_trf, self.train_raw[self.target]], axis = 1)
        self.test = test_trf

    def encode(self):
        data = pd.concat([self.test, self.train])
        oe = OrdinalEncoder()
        data[self.cat_features] = oe.fit_transform(data[self.cat_features]).astype('int')
        
        scaler = StandardScaler()
        data[self.num_features + [self.target]] = scaler.fit_transform(data[self.num_features + [self.target]])
        
        self.train_enc = data[~data[self.target].isna()]
        self.test_enc = data[data[self.target].isna()].drop(self.target, axis=1)
        
       
    def new_features(self, data): 
        return data

    def log_transformation(self):
        self.train[self.target] = np.log1p(self.train[self.target]) 
        
        return self
        
    def remove_outliers(self):
        Q1 = self.train[self.target].quantile(0.25)
        Q3 = self.train[self.target].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5*IQR
        upper_limit = Q3 + 1.5*IQR
        self.train = self.train[(self.train[self.target] >= lower_limit) & (self.train[self.target] <= upper_limit)]
        self.train.reset_index(drop=True, inplace=True) 
        
    def scaler(self):
        scaler = StandardScaler()
        self.train[self.num_features] = scaler.fit_transform(self.train[self.num_features])
        self.test[self.num_features] = scaler.transform(self.test[self.num_features])
        return self
    
    def missing_values(self):
        self.train[self.num_features] = self.train[self.num_features].fillna(self.train[self.num_features].median())
        self.test[self.num_features] = self.test[self.num_features].fillna(self.test[self.num_features].median())
        self.train[self.cat_features] = self.train[self.cat_features].fillna('None')
        self.test[self.cat_features] = self.test[self.cat_features].fillna('None')
        return self

    def reduce_mem(self, df):

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', "uint16", "uint32", "uint64"]
        
        for col in df.columns:
            col_type = df[col].dtypes
            
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()

                if "int" in str(col_type):
                    if c_min >= np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min >= np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min >= np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    if c_min >= np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)  

        return df
    


t = Transform()
X, X_enc, y, test, test_enc, cat_features = t()


# defining the error
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


from sklearn.base import BaseEstimator, RegressorMixin
import contextlib, io
import ydf; ydf.verbose(2)
from ydf import GradientBoostedTreesLearner

def YDFRegressor(learner_class):

    class YDFXRegressor(BaseEstimator, RegressorMixin):

        def __init__(self, params={}):
            self.params = params

        def fit(self, X, y):
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            target = y.name
            params = self.params.copy()
            params['label'] = target
            params['task'] = ydf.Task.REGRESSION
            X = pd.concat([X, y], axis=1)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.model = learner_class(**params).train(X)
            return self

        def predict(self, X):
            assert isinstance(X, pd.DataFrame)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                return self.model.predict(X)

    return YDFXRegressor


## ADD Model training here

class Model(Config):
    
    def __init__(self, X, X_enc, y, test, test_enc, models):
        super().__init__()
        
        self.y = y
        self.models = models
        self.cat_c = list(X.select_dtypes(exclude = ['number']).columns)

        self.OOF_preds = pd.DataFrame()
        self.TEST_preds = pd.DataFrame()
        self.scores_df = pd.DataFrame(columns = ['Score'])

    def getCVScheme(self):
        if self.fold_type == 'SKF':
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.state)
        elif self.fold_type == 'KF':
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.state)
        elif self.fold_type == 'GKF':
            kfold = GroupKFold(n_splits=self.n_splits)
        elif self.fold_type == 'RKF':
            kfold = RepeatedKFold(n_splits=self.n_splits, n_repeats = self.n_repeats, random_state=self.state)
        elif self.fold_type == 'TSS' :
            kfold = TimeSeriesSplit(n_splits = self.n_splits)
        else:
            raise NotImplementedError("Select the Given CV Strategy")
            
        return kfold
        

    def train(self, X, X_enc, test, test_enc):
        
        folds = self.getCVScheme()
        

        for model_name, [model, params, take_log] in tqdm(self.models.items()):
            print('='*5, f'Training : {model_name}', '='*5)

            if any(name in model_name for name in ["XGB", "CAT", "LGBM"]):
                self.X = X
                self.test = test
                
            else :
                self.X = X_enc
                self.test = test_enc

            for n_fold, (train_idx, val_idx) in enumerate(tqdm(folds.split(self.X, self.y), desc = "Training Folds", total = self.n_splits)): 
                X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
                X_val, y_val = self.X.iloc[val_idx], self.y.iloc[val_idx]

                oof_preds = pd.DataFrame(columns = [model_name], index = X_val.index)
                test_preds = pd.DataFrame(columns = [model_name], index = self.test.index)

                model = self.model_train_decision(model_name, params, model, X_train, np.log1p(y_train), X_val, np.log1p(y_val)) if take_log else self.model_train_decision(model_name, params, model, X_train, y_train, X_val, y_val) 

                y_train_pred = np.expm1(model.predict(X_train)) if take_log else model.predict(X_train)
                y_val_pred = np.expm1(model.predict(X_val)) if take_log else model.predict(X_val)
                test_pred = np.expm1(model.predict(self.test)) if take_log else model.predict(self.test)

                oof_preds[model_name] = y_val_pred
                test_preds[model_name] = test_pred

                train_score = rmse(y_train, y_train_pred)
                val_score = rmse(y_val, y_val_pred)

                print(f"Fold {n_fold+1} - Train RMSE: {train_score:.4f}, Validation RMSE: {val_score:.4f}")
                
                self.scores_df.loc[f'{model_name}', f'{n_fold + 1}'] = val_score
                self.OOF_preds = pd.concat([self.OOF_preds, oof_preds], axis = 0, ignore_index = False) 
                self.TEST_preds = pd.concat([self.TEST_preds, test_preds], axis = 0, ignore_index = False)
                
            self.OOF_preds = self.OOF_preds.groupby(level = 0).mean()
            self.TEST_preds = self.TEST_preds.groupby(level = 0).mean()

            self.scores_df.loc[f'{model_name}', 'Score'] = self.scores_df.loc[f'{model_name}'][1:].mean()
            overall_model_score = self.scores_df.loc[f'{model_name}', 'Score']
            print('='*5, f'Training complete of : {model_name}', '='*5, f'Overall Score : {overall_model_score:.4f}')
            
            clear_output(wait = True)
            
            
        self.scores_df.loc['Ensemble'], self.OOF_preds["Ensemble"], self.TEST_preds["Ensemble"] = self.ensemble(self.OOF_preds, self.y, self.TEST_preds)
        self.scores_df.sort_values('Score')

        self.result()
            
        return self.OOF_preds, self.TEST_preds, self.scores_df


    def model_train_decision(self, model_name, params, model, X_train, y_train, X_val, y_val):
        if "LGBM" in model_name:
                callbacks = [lgb.early_stopping(stopping_rounds = self.early_stop, verbose = False)]
                # model = lgb.LGBMRegressor(**params, random_state = self.state, verbose = -1, njobs = -1, device = 'cpu')
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set = [(X_val, y_val)],#eval_metric = '', # change error metric!
                          callbacks = callbacks) 

        elif "CAT" in model_name:
            # model = CatBoostRegressor(**params, random_state = SEED, verbose = 0)
            # train_pool = Pool(data=X_train, label=y_train, cat_features = self.cat_features_indices)
            # val_pool = Pool(data=X_val, label=y_val, cat_features = self.cat_features_indices)
            
            # model = CatBoostRegressor(**params, random_state=self.state, verbose=0, task_type='CPU')
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, 
                      eval_set = (X_val, y_val),
                      cat_features = self.cat_c,
                      early_stopping_rounds=100,
                      verbose = 0)
            
        elif "XGB" in model_name:
            # model = XGBRegressor(**params,random_state = self.state, objective= "reg:squarederror", enable_categorical=True, verbosity = 0)
            model = XGBRegressor(**params)
            model.fit(X_train, y_train,
                     eval_set = [(X_val, y_val)],
                     verbose = 0)
        else :
            model.fit(X_train, y_train)

        return model

    def ensemble(self, X, y, test):
        ensemble_model_name = 'Bayesian Ridge Regression'
        print(f'Ensembling with {ensemble_model_name} started')
        scores = []
        
        oof_pred = np.zeros(X.shape[0])
        test_pred = np.zeros(test.shape[0])
        
        model = BayesianRidge(tol=1e-2, n_iter=1000000)
        
        kf = self.getCVScheme()
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)
            oof_pred[val_idx] = y_pred_val
            test_pred += model.predict(test) / self.n_splits
            
            score = rmse(y_val, y_pred_val)
            scores.append(score)
        print(f'Ensembling Finished Successfully !')
                   
        return np.mean(scores), oof_pred, test_pred


    def result(self):
               
        plt.figure(figsize=(14, 6))
        colors = ['#3cb371' if i != 'Ensemble' else 'r' for i in self.scores_df.Score.index]
        hbars = plt.barh(self.scores_df.index, self.scores_df.Score, color=colors, height=0.8)
        plt.bar_label(hbars, fmt='%.4f')
        plt.ylabel('Models')
        plt.xlabel('Score')              
        plt.show()

        y = (self.y).sort_index()
        self.OOF_preds['Ensemble'] = (self.OOF_preds['Ensemble']).sort_index()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].scatter(y, self.OOF_preds['Ensemble'], alpha=0.5, s=15, edgecolors='#3cb371')
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Actual vs. Predicted')

        axes[1].scatter(self.OOF_preds['Ensemble'], y - self.OOF_preds['Ensemble'], alpha=0.5, s=15, edgecolors='#3cb371')
        axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')

        plt.tight_layout()
        plt.show()





models = {
    'YDF': [YDFRegressor(GradientBoostedTreesLearner)({'num_trees': 1000,
                                                       'max_depth': 13,
                                                       }),
            '', False],
    'CAT1': ['', {'verbose': 0,
                      'random_state': Config.state,
                      'cat_features': cat_features,
                      'early_stopping_rounds': Config.early_stop,
                      'eval_metric': "RMSE",
                      'n_estimators' : 2000,
                              },
            False],
    'CAT2' : ['', {'verbose' : 0,
                    'eval_metric': 'RMSE',
                    'cat_features': cat_features,
                    'early_stopping_rounds': Config.early_stop,
                    'learning_rate': 0.13944051481200972,
                    'iterations': 1447,
                    'depth': 3,
                    'random_strength': 17,
                    'l2_leaf_reg': 7.554047383325137,
                    'bagging_temperature': 0.5838770203329602,
                    'verbose': 100,
                    'random_seed': Config.state,
                              },
             False],
    'XGB1': ['', {'tree_method': 'hist',
                         'n_estimators': 2000,
                         'objective': 'reg:squarederror',
                         'random_state': Config.state,
                         'enable_categorical': True,
                         'verbosity': 0,
                         'early_stopping_rounds': Config.early_stop,
                         'eval_metric': 'rmse',
                           },
            False],
    'LGBM1': ['', {'random_state': Config.state,
                           'early_stopping_round': Config.early_stop,
                           'categorical_feature': cat_features,
                           'verbose': -1,
                           'boosting_type': 'gbdt',
                           'n_estimators': 3000,
                           'eval_metric': 'rmse',
                          'objective': 'regression_l2',
                              },
             False],
    'LinReg_1' : [LinearRegression(), '', False],
    'Ridge_1' : [Ridge(random_state=Config.state), '', False],
    'RF_1' : [RandomForestRegressor(random_state=Config.state), '', False],
    'ENet_1' : [ElasticNet(random_state=Config.state), '', False],
    'Lasso_1' : [Lasso(random_state=Config.state), '', False],
    'HGB1': [HistGradientBoostingRegressor(**{'max_iter': 5000,
                                             'random_state': Config.state,
                                             'early_stopping': Config.early_stop
                                             }),'',
             False],
}


training_object = Model(X = X, X_enc = X_enc, y = y, 
                        test = test, test_enc = test_enc, 
                        models = models)

oof_preds_df, test_preds_df, scores_df = training_object.train(X = X, X_enc = X_enc, 
                                               test = test, test_enc = test_enc)



!pip install hillclimbers


from hillclimbers import climb_hill, partial


hc_test_pred_probs, hc_oof_pred_probs = climb_hill(
    train = pd.concat([X_enc, y], axis=1),
    oof_pred_df=oof_preds_df, 
    test_pred_df=test_preds_df,
    target=Config.target,
    objective='minimize', 
    eval_metric=partial(rmse), 
    negative_weights=True, 
    precision=0.0001, 
    plot_hill=False, 
    plot_hist=False,
    return_oof_preds=True
)


sample = Config.sample
sample[Config.target] = hc_test_pred_probs


sample.to_csv('submssion.csv', index = False)
sample.head()




