# Uninstall old version and install the latest version of scikit-learn
!pip uninstall scikit-learn -y
!pip install scikit-learn

# Verify version
import sklearn
print(sklearn.__version__)  # Should output: 1.6.1
print(sklearn.__file__)     # Check the module path


### Import All Possible Libraries ###

import pandas as pd
import numpy as np
from datetime import datetime
import time
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats as sms
import statsmodels.stats.descriptivestats as smd
from statsmodels.stats.stattools import jarque_bera, durbin_watson 
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import root_mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor, plot_importance
from sklearn.base import BaseEstimator, RegressorMixin, clone
from copy import deepcopy


### Import All the Necessary Data ###

df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e5/test.csv')

# Overview of the data
print('----- Quick overview of the train data -----')
print(df_train.info())
print('\n----- Quick overview of the test data -----')
print(df_test.info())


### Drop Unused Column ###

# Drop "id" column
df_train.drop(columns='id', inplace=True)
df_test.drop(columns='id', inplace=True)

print('After column dropping: ')
print(df_train.columns)
print(df_test.columns)


### Summary Statistics of Numerical Columns ###

# Separate the numerical columns only
df_train_num = df_train.select_dtypes(include='number')

# Show the summary statistics
smd.describe(df_train_num)


### Distribution of Each Numerical Column using KDE Plot ###

axes = df_train_num.plot.kde(subplots=True, layout=(3, 3), figsize=(14, 10), sharex=False)

# Get figure and add a title
fig = plt.gcf()
fig.suptitle('Numerical Columns Distributions with KDE Plot', y=1.025, fontsize=20)

plt.tight_layout()

plt.show()


### Distribution of Each Numerical Column using Box Plot ###

axes = df_train_num.plot.box(subplots=True, layout=(3, 3), figsize=(14, 10), sharex=False)

# Get figure and add a title
fig = plt.gcf()
fig.suptitle('Numerical Columns Distributions with Box Plot', y=1.025, fontsize=20)

plt.tight_layout()

plt.show()


### Correlation between Each Numerical Features ###

warnings.filterwarnings('ignore')

# Create the correlation dataframe
df_train_corr = df_train_num.drop(columns='Calories').corr(method='pearson')

# Create correlation heatmap
plt.figure(figsize=(12, 10))  # Adjust size based on number of features
sns.heatmap(
    df_train_corr,
    annot=True,              # Show correlation values
    cmap='coolwarm',         # Color scheme (red-blue)
    vmin=-1, vmax=1,         # Correlation range
    center=0,                # Center at 0
    fmt='.2f',               # 2 decimal places
    square=True,             # Square cells
    linewidths=0.5,          # Line between cells
    cbar_kws={'label': 'Pearson Correlation Coefficient'},  # Colorbar label
    annot_kws={'size': 10}   # Font size for annotations
)

plt.title('Correlation Heatmap of Numerical Features', fontsize=16, pad=20)

plt.show()


### Compute VIF Values for Each Numerical Feature ###

# Create a dataframe without the target variable
df_train_features = df_train_num.drop(columns='Calories')

df_vif = pd.DataFrame()
df_vif['Feature'] = df_train_features.columns
df_vif['VIF'] = [vif(df_train_features.values, i) for i in range(df_train_features.shape[1])]
print("----- VIF Table -----")
print(df_vif)


### Separate the Data into Features and Target Variables ###

X_train = df_train.drop(columns='Calories')
X_test = df_test
y_train = df_train[['Calories']]

'''
Note:
There is no target variable in the test data
'''

# Shape of the data
print('Shape of each data:')
print(f'X_train: {X_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_train: {y_train.shape}')


### Separate the Features into Numerical and Categorical ###

X_train_num = X_train.select_dtypes(include='number')
X_train_cat = X_train.select_dtypes(include='object')
X_test_num = X_test.select_dtypes(include='number')
X_test_cat = X_test.select_dtypes(include='object')

# Shape of the data
print('Shape of each data:')
print(f'Train numerical features data: {X_train_num.shape}')
print(f'Train categorical feature data: {X_train_cat.shape}')
print(f'Test numerical features data: {X_test_num.shape}')
print(f'Test categorical feature data: {X_test_cat.shape}')


### Apply Feature Scaling to Numerical Features ###

'''
Feature scaling technique is necessary for this data because there are some
columns that have higher values compared to the other columns, like "Height".
If this column be kept as it is, it may outweighs the other columns and might
contribute more to a model compared to the other columns.
RobustScaler() function will be used because there are some potential outliers
in the data and this function is robust with the presence of outliers.
'''

# Define the scaling estimator
robustscaler = RobustScaler()
X_train_scaled = robustscaler.fit_transform(X_train_num)
X_test_scaled = robustscaler.transform(X_test_num)

# Encapsulate to new dataframes
X_train_scaled = pd.DataFrame(X_train_scaled, 
                              columns=X_train_num.columns,
                              index=X_train_num.index)
X_test_scaled = pd.DataFrame(X_test_scaled, 
                              columns=X_test_num.columns,
                              index=X_test_num.index)

# Preview
print(f'Training set numerical features scaled preview: \n{X_train_scaled.shape}')
display(X_train_scaled.head())
print(f'\nTesting set numerical features scaled preview: \n{X_test_scaled.shape}')
display(X_test_scaled.head())


### Create Dummy Variables for the Categorical Feature ###

X_train_dummy = pd.get_dummies(X_train_cat)
X_test_dummy = pd.get_dummies(X_test_cat)

# Preview
print(f'Training set dummy preview: \n{X_train_dummy.shape}')
display(X_train_dummy.head())
print(f'\nTesting set dummy preview: \n{X_test_dummy.shape}')
display(X_test_dummy.head())


### Combine the Numerical and Categorical Features into One Dataframe ###

X_train = pd.concat([X_train_scaled, X_train_dummy], axis=1)
X_test = pd.concat([X_test_scaled, X_test_dummy], axis=1)

# Preview
print(f"Combined train data's shape: {X_train.shape}")
print(f"Combined test data's shape: {X_test.shape}")


### Create an Initial OLS Model ###

# Drop the dummy that are going to be used as the baseline
X_train_ols = X_train.drop(columns='Sex_male')

# Add an intercept column to the train data
X_train_ols = sm.add_constant(X_train_ols)

ols_mod = sm.OLS(y_train, X_train_ols.astype(float)).fit()

print(ols_mod.summary())


### Define Function to Check Regression Assumptions ###

warnings.filterwarnings('ignore')

def residual_analysis(model, residuals, fitted, data_name="Dataset"):
    ### Using Visualizations
    plt.figure(figsize=(15, 10))
    
    # Residuals vs Fitted (Homoscedasticity Checking)
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=fitted, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Plot')
    
    # Q-Q Plot (Normality Checking)
    plt.subplot(2, 2, 2)
    sm.qqplot(residuals, line='s', fit=True, ax=plt.gca())
    plt.title('Q-Q Plot')
    
    # Histogram (Normality Checking)
    plt.subplot(2, 2, 3)
    sns.histplot(residuals, kde=True, color='blue')
    plt.xlabel('Residuals')
    plt.title('Residuals Histogram')
    
    # Cook's Distance (Influential Observation)
    plt.subplot(2, 2, 4)
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    cooks_threshold = 0.5   # define the threshold for cook's distance
    plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
    plt.axhline(y=cooks_threshold, xmin=min(np.arange(len(cooks_d))), xmax=max(np.arange(len(cooks_d))),
                linewidth=2, linestyle='--', color='red')
    plt.xlabel('Observation Index')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance Plot")

    # Give Major Title
    plt.suptitle(f'Residual Analysis for {data_name}', y=1.05)
    plt.tight_layout()
    plt.show()
    
    ### Using Statistical Tests 
    print(f"\nResidual Diagnostics for {data_name}:")
    
    # Normality Test
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    print("\n1) Jarque-Bera Test:")
    print(f"Statistic: {jb_stat:.4f}, P-value: {jb_pvalue:.4f}, Skew: {skew:.4f}, Kurtosis: {kurtosis:.4f}")
    
    # Homoscedasticity (White Test)
    white_test = het_white(residuals, model.model.exog)
    labels = ['LM Statistic', 'LM P-value', 'F-Statistic', 'F P-value']
    print("\n2) White Test:")
    for label, value in zip(labels, white_test):
        print(f"{label}: {value:.4f}")
    
    # Autocorrelation (Durbin Watson Test)
    dw_stat = durbin_watson(residuals)
    print(f"\n3) Durbin-Watson: {dw_stat:.4f}")
    
    # Maximum Cook's Distance Value
    print(f"\n4) Max Cook's Distance: {cooks_d.max():.4f}")


residual_analysis(ols_mod, ols_mod.resid, ols_mod.fittedvalues, data_name="Calories Burnt Data")


### Apply Transformation to the Target Variable ###

log_transformer = FunctionTransformer(np.log)
y_train_log = log_transformer.fit_transform(y_train)

print('Preview of the transformed target variable:')
display(y_train_log.head())


### Remodel using the Newly Transformed Target Variable ###

ols_remod = sm.OLS(y_train_log, X_train_ols.astype(float)).fit()

print(ols_remod.summary())


### Residual Analysis on the New Model ###

residual_analysis(ols_remod, ols_remod.resid, ols_remod.fittedvalues, data_name="Calories Burnt Data")


### Define a Wrapper for OLS Model ###

'''
The purpose of this wrapper is to allow the OLS() function from statsmodels library to be used 
within cross_val_score() function from scikit-learn. This wrapper will ensure no negative values
generated from the prediction results. This wrapper will also has optional L1 or L2 regularization
techniques that can be applied to the OLS model.
'''

class StatsmodelsOLS(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.0, L1_wt=0.0, fit_intercept=True):
        """
        Statsmodels OLS with optional L1 or L2 regularization.

        ---------- Parameters ----------
        alpha : float, default=0.0
            Regularization strength; must be non-negative.
            Larger values increase regularization.
        L1_wt : float, default=0.0
            Weight of L1 penalty (0.0 = Ridge, 1.0 = Lasso, 0 < L1_wt < 1 = Elastic Net).
        fit_intercept : bool, default=True
            Whether to add a constant (intercept) to the design matrix.
        """
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.fit_intercept = fit_intercept
        self.model = None

    def fit(self, X, y):
        """
        Fit the model using OLS with optional regularization.

        ---------- Parameters ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        ---------- Returns ----------
        self : object
            Fitted estimator.
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Add intercept if required
        if self.fit_intercept:
            X = sm.add_constant(X)
        
        # Initialize OLS model
        ols_model = sm.OLS(y, X)

        # Fit with regularization if alpha > 0, else use standard OLS
        if self.alpha > 0:
            self.model = ols_model.fit_regularized(
                method='elastic_net',
                alpha=self.alpha,
                L1_wt=self.L1_wt
            )
        else:
            self.model = ols_model.fit()

        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        ---------- Parameters ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        ---------- Returns ----------
        y_pred : array-like of shape (n_samples,)
            Predicted values, clipped to be non-negative.
        """
        X = np.asarray(X)

        # Add intercept if required
        if self.fit_intercept:
            X = sm.add_constant(X)
        
        # Get predictions
        y_pred = self.model.predict(X)

        # Ensure no negative values
        y_pred = np.maximum(y_pred, 0)
        return y_pred

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return {
            'alpha': self.alpha,
            'L1_wt': self.L1_wt,
            'fit_intercept': self.fit_intercept
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self


### Drop the Constant Column from "X_train_ols" ###

'''
The constant column will be dropped from "X_train_ols" to avoid having double
constant column to represent the intercept.
'''

X_train_ols_no_constant = X_train_ols.drop(columns='const')

display(X_train_ols_no_constant.head())


### Performance on the Train & Validation Sets for Model without Regularization ###

# Define the estimator and fit it

'''
Because the metric that will be used is RMSLE, the transformed target variable needs to be back-transform
using exponential so that the value of RMSLE is not misleading.
'''

smOLS = StatsmodelsOLS()
smOLS.fit(X_train_ols_no_constant.astype('float'), np.exp(y_train_log)) # back-transform the target variable

# Predict the training set
y_train_ols_pred = smOLS.predict(X_train_ols_no_constant)

# Calculate RMSLE score for training set
ols_train_rmsle = root_mean_squared_log_error(np.exp(y_train_log), y_train_ols_pred)

# Calculate RMSLE on validation set
ols_val_rmsle = cross_val_score(smOLS, X_train_ols_no_constant.astype('float'), np.exp(y_train_log), 
                                cv=5, scoring='neg_root_mean_squared_log_error')
ols_val_rmsle = abs(np.mean(ols_val_rmsle))

print(f'OLS model training set RMSLE: {ols_train_rmsle: .3f}')
print(f'OLS model validation set RMSLE: {ols_val_rmsle: .3f}')
print(f'RMSLE difference: {ols_val_rmsle-ols_train_rmsle: .3f} or {(ols_val_rmsle-ols_train_rmsle)/ols_train_rmsle*100: .2f}%')


### Performance on the Train & Validation Sets for Model with Regularization ###

# Define the estimator and fit it

'''
Because the metric that will be used is RMSLE, the transformed target variable needs to be back-transform
using exponential so that the value of RMSLE is not misleading.
'''

smOLS_reg = StatsmodelsOLS(alpha=0.25, 
                           L1_wt=0.0  # implement L2 regularization
                           )
smOLS_reg.fit(X_train_ols_no_constant.astype('float'), np.exp(y_train_log)) # back-transform the target variable

# Predict the training set
y_train_ols_pred_reg = smOLS_reg.predict(X_train_ols_no_constant)

# Calculate RMSLE score for training set
ols_train_rmsle_reg = root_mean_squared_log_error(np.exp(y_train_log), y_train_ols_pred_reg)

# Calculate RMSLE on validation set
ols_val_rmsle_reg = cross_val_score(smOLS_reg, X_train_ols_no_constant.astype('float'), np.exp(y_train_log), 
                                    cv=5, scoring='neg_root_mean_squared_log_error')
ols_val_rmsle_reg = abs(np.mean(ols_val_rmsle_reg))

print(f'OLS model training set RMSLE: {ols_train_rmsle_reg: .3f}')
print(f'OLS model validation set RMSLE: {ols_val_rmsle_reg: .3f}')
print(f'RMSLE difference: {ols_val_rmsle_reg-ols_train_rmsle_reg: .3f} or {(ols_val_rmsle_reg-ols_train_rmsle_reg)/ols_train_rmsle_reg*100: .2f}%')


### Create a Custom Wrapper for MLPRegressor() Function ###

'''
Custom wrapper will be created for MLPRegressor() to modify the prediction results
and prevent it from generating negative prediction values because negative values will
result in undefined RMSLE value. 
'''

class PositivesMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **mlp_params):
        self.mlp = MLPRegressor(**mlp_params)
        self.min_value = 0  # 0 value for clipping

    def fit(self, X, y):
        self.mlp.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.mlp.predict(X)
        return np.maximum(y_pred, self.min_value) # ensure no negative predictions


### Performance on the Training Set ###

warnings.filterwarnings('ignore')

mlp = PositivesMLPRegressor(random_state=123)

# Measure training model running time
start_time = time.time()
mlp.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time

# Measure training model prediction time
start_time = time.time()
y_train_mlp_pred = mlp.predict(X_train)
end_time = time.time()
prediction_time = end_time - start_time 

# Compute the RMSLE score
mlp_train_rmsle = root_mean_squared_log_error(y_train, y_train_mlp_pred)

print('MLP model training set performances: ')
print(f'RMSLE score: {mlp_train_rmsle: .3f}')

print(f'\nModel training time: {train_time:.3f} seconds or {train_time/60: .1f} minute(s)')
print(f'Model prediction time: {prediction_time:.3f} seconds or {prediction_time/60: .3f} minute(s)')


### Performance on the Validation Set ###

warnings.filterwarnings('ignore')

mlp_val_rmsle = cross_val_score(mlp, X_train, y_train,
                                cv=5, scoring='neg_root_mean_squared_log_error')
mlp_val_rmsle = abs(np.mean(mlp_val_rmsle))

print(f'MLP model validation set RMSLE score: {mlp_val_rmsle: .3f}')
print(f'Training and validation set difference: {mlp_val_rmsle-mlp_train_rmsle: .3f} or {(mlp_val_rmsle-mlp_train_rmsle)/mlp_train_rmsle*100: .2f}%')


### Create a Custom Wrapper for XGBRegressor() Function ###

'''
Custom wrapper will be created for XGBRegressor() to modify the prediction results
and prevent it from generating negative prediction values because negative values will
result in undefined RMSLE value.
'''

class PositivesXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        alpha=0.0,
        lambda_=1.0,
        random_state=None,
        extra_params=None
    ):
        """
        XGBoost Regressor wrapper ensuring non-negative predictions.

        ---------- Parameters ----------
        n_estimators : int, default=100
            Number of boosting rounds.
        learning_rate : float, default=0.1
            Step size shrinkage used in update to prevent overfitting.
        max_depth : int, default=3
            Maximum depth of a tree.
        subsample : float, default=1.0
            Fraction of samples used for fitting each tree.
        colsample_bytree : float, default=1.0
            Fraction of features used for each tree.
        alpha : float, default=0.0
            L1 regularization term on weights.
        lambda_ : float, default=1.0
            L2 regularization term on weights.
        random_state : int or None, default=None
            Random seed for reproducibility.
        extra_params : dict or None, default=None
            Additional parameters to pass to XGBRegressor.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.alpha = alpha
        self.lambda_ = lambda_
        self.random_state = random_state
        self.extra_params = extra_params if extra_params is not None else {}
        self.model = None

    def fit(self, X, y, **fit_params):
        """
        Fit the XGBoost model.

        ---------- Parameters ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params : dict
            Additional parameters for XGBRegressor's fit method.

        ---------- Returns ----------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Combine parameters for XGBRegressor
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'alpha': self.alpha,
            'lambda': self.lambda_,
            'random_state': self.random_state
        }
        # Use a deep copy of extra_params to avoid modification
        params.update(deepcopy(self.extra_params))

        # Initialize and fit model
        self.model = XGBRegressor(**params)
        self.model.fit(X, y, **fit_params)

        return self

    def predict(self, X):
        """
        Predict using the fitted XGBoost model, ensuring non-negative outputs.

        ---------- Parameters ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        ---------- Returns ----------
        y_pred : array-like of shape (n_samples,)
            Predicted values, clipped to be strictly positive.
        """
        X = np.asarray(X)
        y_pred = self.model.predict(X)
        y_pred = np.maximum(y_pred, 1e-10)  # Strict positivity for RMSLE
        return y_pred

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        ---------- Returns ----------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'alpha': self.alpha,
            'lambda_': self.lambda_,
            'random_state': self.random_state,
            'extra_params': deepcopy(self.extra_params) if deep else self.extra_params
        }
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        ---------- Parameters ----------
        **params : dict
            Estimator parameters.

        ---------- Returns ----------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            if key == 'extra_params':
                self.extra_params = deepcopy(value) if value is not None else {}
            else:
                setattr(self, key, value)
        return self


### Performance on the Training Set ###

warnings.filterwarnings('ignore')

xgb = PositivesXGBRegressor(random_state=123)

# Measure training model running time
start_time = time.time()
xgb.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time

# Measure training model prediction time
start_time = time.time()
y_train_xgb_pred = xgb.predict(X_train)
end_time = time.time()
prediction_time = end_time - start_time 

# Compute the RMSLE score
xgb_train_rmsle = root_mean_squared_log_error(y_train, y_train_xgb_pred)

print('XGBoost Regressor model training set performances: ')
print(f'RMSLE score: {xgb_train_rmsle: .3f}')

print(f'\nModel training time: {train_time:.3f} seconds or {train_time/60: .1f} minute(s)')
print(f'Model prediction time: {prediction_time:.3f} seconds or {prediction_time/60: .3f} minute(s)')


### Performance on the Validation Set ###

warnings.filterwarnings('ignore')

start_time = time.time() 
xgb_val_rmsle = cross_val_score(xgb, X_train, y_train,
                                cv=5, scoring='neg_root_mean_squared_log_error')
end_time = time.time()  # measure the CV running time

xgb_val_rmsle = abs(np.mean(xgb_val_rmsle))

print(f'MLP model validation set RMSLE score: {xgb_val_rmsle: .3f}')
print(f'Training and validation set difference: {xgb_val_rmsle-xgb_train_rmsle: .3f} or {(xgb_val_rmsle-xgb_train_rmsle)/xgb_train_rmsle*100: .2f}%')
print(f'\nCV running time: {end_time-start_time: .3f} seconds or {(end_time-start_time)/60: .3f} minute(s)')


### Hyperparameters Tuning ###

warnings.filterwarnings('ignore')

# Define the hyperparameters to be tuned
xgb_params = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 10],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'alpha': [0, 0.15, 1.0],
    'lambda_': [0.1, 1.0, 10.0]
}

# Perform Halving Grid Search CV
start_time = time.time()
xgb_halving_grid = HalvingGridSearchCV(
    xgb,
    xgb_params,
    cv=5,
    scoring='neg_root_mean_squared_log_error'
)
xgb_halving_grid.fit(X_train, y_train)
end_time = time.time()  # measure running time

print(f'Best hyperparameters: \n{xgb_halving_grid.best_params_}')
print(f'\nBest mean RMSLE score: \n{-xgb_halving_grid.best_score_: .3f}')
print(f'\nTuning running time: {end_time-start_time: .3f} seconds or {(end_time-start_time)/60: .2f} minute(s)')


print(f'Train to tuning difference: {(-xgb_halving_grid.best_score_)-xgb_train_rmsle: .3f} or {((-xgb_halving_grid.best_score_)-xgb_train_rmsle)/xgb_train_rmsle*100: .2f}%')
print(f'Validation to tuning difference: {(-xgb_halving_grid.best_score_)-xgb_val_rmsle: .3f} or {((-xgb_halving_grid.best_score_)-xgb_val_rmsle)/xgb_val_rmsle*100: .2f}%')


### Predicting the Test Set ###

y_test_pred = xgb_halving_grid.predict(X_test)
y_test_pred = pd.DataFrame(y_test_pred,
                           columns=['Y Test'])

print("Test set's predictions preview: ")
display(y_test_pred.head())


### Summary Statistics Comparison between Train and Test Target Variables ###

train_test_summaries = pd.concat([y_train.describe(), y_test_pred.describe()],
                                 axis=1).rename(columns={'Calories': 'Y Train'})

display(train_test_summaries)


### Boxplot Comparison between Train and Test Target Variables ###

plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
sns.boxplot(data=y_train, y='Calories', color='.8')
plt.title("Train Data Target Variable Distribution", fontsize=15, y=1.05)

plt.subplot(1, 2, 2)
sns.boxplot(data=y_test_pred, y='Y Test', color='.35')
plt.title("Test Data Target Variable Distribution", fontsize=15, y=1.05)
plt.ylabel(' ')

plt.suptitle('Train VS Test Data Target Variables Comparison', fontsize=20, y=1.025)
plt.tight_layout()
plt.show()


### Export Prediction Results as .csv file ###

y_test_submission = y_test_pred.reset_index().rename(columns={'index': 'id',
                                                              'Y Test': 'Calories'})
y_test_submission['id'] = np.arange(750000, 1000000)
y_test_submission.to_csv('submission.csv', index=False)

