target = 'Calories'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import r2_score, mean_squared_log_error, roc_auc_score, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold, TimeSeriesSplit as TSS)

import sklearn
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, plot_importance, cv
from sklearn.compose import make_column_transformer
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

import optuna

import warnings
warnings.filterwarnings('ignore')

# verify the versions of my tools
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'sklearn version: {sklearn.__version__}')
# print(f'optuna version : {optuna.__version__}')


def rmsle(y_true, y_pred):
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the logarithm of the true and predicted values
    log_true = np.log1p(y_true+1)
    log_pred = np.log1p(y_pred+1)
    
    # Calculate the squared differences
    squared_diff = np.square(log_true - log_pred)
    
    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Calculate the root of the mean squared differences
    rmsle_value = np.sqrt(mean_squared_diff)
    
    return rmsle_value


# Define RMSLE metric
def rmsle_metric(y_true, y_pred):
    y_true = np.maximum(y_true, 1e-6)  # Avoid log(0) issues
    y_pred = np.maximum(y_pred, 1e-6)

    log_true = np.log(y_true)
    log_pred = np.log(y_pred)

    rmsle = np.sqrt(np.mean((log_true - log_pred) ** 2))
    return rmsle

# Train model using a custom metric
model = CatBoostRegressor(
    verbose=250,
    custom_metric=rmsle_metric,  # This won't affect optimization but will be displayed
    eval_metric="RMSE"  # You must set an optimization metric separately
)


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test_raw = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
orig_raw = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv').drop(columns=['User_ID'])

orig_raw.columns = train_raw.columns
train_raw.head(3)


orig_raw['Body_Temp'].min()


class Feature_Eng(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()
        df['Max_heart_rate'] = 207 - 0.7*df['Age']
        df['%_heart_rate'] = df['Heart_Rate']/df['Max_heart_rate']
        df['heart_rate_/_weight'] = df['Heart_Rate']/df['Weight']
        df['Body_Temp - 37'] = df['Body_Temp'] - 37 # assuming 37 as body temp at rest
        df['heat'] = df['Weight']*df['Body_Temp - 37']
        df['work'] = df['heat']*df['Duration']/60
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=[1, 2, 3, 4, 5]).astype('int')
        df['Age_pred_max_heart_rate'] = pd.cut(df['Age'], 
                                               bins=[10, 30, 35, 40, 45, 50, 55, 60, 65, 70, 90], 
                                               labels=[200, 190, 185, 180, 175, 170, 165, 160, 155, 150]
                                              ).astype('int')
        df['diff_heart_rate'] = df['Max_heart_rate'] - df['Age_pred_max_heart_rate']
        # df = df.drop(columns=['Age'])
        df['BMI'] = df['Weight']/(0.01*df['Height'])**2
        # df['Temp_Weight_ratio'] = df['Body_Temp']/df['Weight']
        # df['HeartRate_BodyTemp_ratio'] = df['Heart_Rate']/df['Body_Temp']
        # df['is_male'] = df['Sex'].map({'male':1, 'female':0})
        df['is_male'] = df['Sex']=='male'
        df = df.drop(columns=['Sex'])
            
        return df


prep_X = train_raw.copy()
prep_X = Feature_Eng().fit_transform(prep_X)
prep_X.head(3)


X_ = prep_X.copy()
y_ = X_.pop(target)

# Calculate Mutual Informa detion
mi = mutual_info_classif(X_, y_)*100

mi_df = pd.DataFrame({'MI': mi})

mi_df.index = X_.columns

# barplot of the mutual info of features
ax = mi_df.sort_values(by='MI', ascending=True).plot.barh(figsize=(10,6), color='sandybrown')
for mi in ax.containers:
    ax.bar_label(mi)
plt.title('Mutual information of the features')
plt.show()


corr_with_Target = X_.corrwith(y_).sort_values()
corr_with_Target.plot.barh(figsize=(9, 5), color='peru')
ax = plt.title('Features Correlation with the Target', fontsize=14, color='peru')
plt.tight_layout()


corr_with_Target.sort_values(ascending=False).to_frame().style.background_gradient(cmap='RdBu', axis=0)


all_num_feat = [col for col in prep_X.select_dtypes(include='number').columns.tolist() if col not in [target, 'is_male']]

plt.figure(figsize=(12, 8))
for f, feat in enumerate(all_num_feat , start=1):
    plt.subplot(4, 4, f)
    sns.boxenplot(prep_X, y=feat, x='is_male', palette='copper')
plt.tight_layout()


all_num_feat = [col for col in prep_X.select_dtypes(include='number').columns.tolist() if col not in [target, 'is_male']]

plt.figure(figsize=(12, 8))
for f, feat in enumerate(all_num_feat , start=1):
    plt.subplot(4, 4, f)
    sns.kdeplot(prep_X, x=feat, hue='is_male', palette='copper', fill=True)
    if f != 1:
        plt.legend([])
        plt.ylabel('')
plt.tight_layout()


plt.figure(figsize=(12, 10))
for f, feat in enumerate(all_num_feat , start=1):
    plt.subplot(4, 4, f)
    sns.scatterplot(prep_X, x=feat, y=target, palette='Dark2')
plt.tight_layout()


plt.figure(figsize=(10, 9))
prep_corr = prep_X.corr()
sns.heatmap(prep_corr, annot=True, fmt='.2f', cmap='copper', cbar=False);


class DHCF(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df, threshold=0.98):
        # Calculate the correlation matrix
        corr_matrix = df.corr().abs()
        # Select the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Initialize a set to keep track of features to drop
        to_drop = set()
        
        # Iterate over the columns and find features with correlation greater than the threshold
        for column in upper.columns:
            if any(upper[column] > threshold):
                # Add the column to the set of features to drop
                to_drop.add(column)
                # Remove the correlated columns from further consideration
                correlated_columns = upper.index[upper[column] > threshold].tolist()
                to_drop.update(correlated_columns)
        
        # Drop the features from the dataset
        df.drop(to_drop, axis=1, inplace=True)
    
        return df


prep_X = train_raw.copy()
prep_X.pop(target)

prep_treator = make_pipeline(Feature_Eng(), DHCF())
prep_X = prep_treator.fit_transform(prep_X)

num_feats = prep_X.select_dtypes(include='number').columns.tolist()

prep_X.head()


lgbm_1 = LGBMRegressor(verbose=-1)
lgbm_1.fit(X_, y_)

lgbm_feat_importances = pd.DataFrame({'Features':X_.columns, 
                                     'Importances':lgbm_1.feature_importances_}
                                   ).sort_values(by='Importances', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(lgbm_feat_importances, y='Features', x='Importances', color='peru')
for imp in ax.containers:
    ax.bar_label(imp)
plt.title('Features importance for lgbm model')
plt.show()


xgb_ = XGBRegressor()
xgb_.fit(X_, y_)

xgb_feat_importances = pd.DataFrame({'Features':X_.columns, 
                                     'Importances':xgb_.feature_importances_}
                                   ).sort_values(by='Importances', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(xgb_feat_importances, y='Features', x='Importances', color='peru')
for imp in ax.containers:
    ax.bar_label(imp)
plt.title('Features importance for xgb model')
plt.show()


cat_ = CatBoostRegressor(verbose=False)
cat_.fit(X_, y_)

xgb_feat_importances = pd.DataFrame({'Features':X_.columns, 
                                     'Importances':cat_.feature_importances_}
                                   ).sort_values(by='Importances', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(xgb_feat_importances, y='Features', x='Importances', color='peru')
for imp in ax.containers:
    ax.bar_label(imp)
plt.title('Features importance for catboost model')
plt.show()


# # List of features needing scaling
# feat_to_scale = ['Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 
#                  'Max_heart_rate', 'Age_pred_max_heart_rate', 'BMI']

feat_to_scale = num_feats

# Build my feature transformer
column_trans = make_column_transformer(
    # (RobustScaler(), num_feats), 
    (MinMaxScaler(), feat_to_scale), 
    remainder='passthrough', 
    sparse_threshold=0)


train_data = pd.concat([train_raw, orig_raw], ignore_index=True, axis=0)

X = train_raw.copy()
y = X.pop(target)


# Ensure X and y are defined externally
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function
def objective_xgb(trial):
    xgb_param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),  # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),  # L2 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    # Data preprocessing and model initialization
    model = make_pipeline(Feature_Eng(), 
                          DHCF(), 
                          column_trans, 
                          XGBRegressor(**xgb_param_grid))

    # fit the model
    model.fit(X_train, y_train)
    
    # Evaluate the model using RMSE
    try:
        preds = np.clip(model.predict(X_val), 
                        orig_raw[target].min(), 
                        orig_raw[target].max())
        score = rmsle(y_val, preds)
        return score
    except:
        return 100

# Define the function to run the study
def Run_Pass_xgb_study(n_trials=1):
    if n_trials > 1:
        # Create and run the study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_xgb, n_trials=n_trials, 
                       show_progress_bar=True)
        best_study_params = study.best_params

        # Print results
        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")       
        best_study_params = {'n_estimators': 660, 
                             'learning_rate': 0.01322533638052107, 
                             'max_depth': 12, 
                             'subsample': 0.5805387519403564, 
                             'colsample_bytree': 0.9630304997861762, 
                             'reg_alpha': 8.557860166592468, 
                             'reg_lambda': 8.371581373953365, 
                             'min_child_weight': 4}

    print(f"Best parameters: {best_study_params}")
    return best_study_params

xgb_best_params = Run_Pass_xgb_study(100)


def objective(trial):
    cat_param_grid = {
        "iterations": trial.suggest_int("iterations", 100, 1000, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
        # "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1),
        "random_strength": trial.suggest_float("random_strength", 0.1, 0.7),
        "depth": trial.suggest_int("depth", 1, 10),
        # "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "boosting_type": "Plain",
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "used_ram_limit": "5gb",
    }

    if cat_param_grid["bootstrap_type"] == "Bayesian":
        cat_param_grid["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 1)
    elif cat_param_grid["bootstrap_type"] == "Bernoulli":
        cat_param_grid["subsample"] = trial.suggest_float("subsample", 0.5, 1)

    # Train the model
    cat_estimator = CatBoostRegressor(**cat_param_grid, verbose=0, eval_fraction=0.2)
    model = make_pipeline(Feature_Eng(), 
                          DHCF(), 
                          column_trans, 
                          cat_estimator)
    
    # fit the model
    model.fit(X_train, y_train)
    
    # Evaluate the model using RMSE
    try:
        preds = np.clip(model.predict(X_val), 
                        orig_raw[target].min(), 
                        orig_raw[target].max())
                        
        score = rmsle(y_val, preds)
        return score
    except:
        return 100

def Run_Pass_cat_study(n_trials=1):
    if n_trials>1:
        # Create and run the study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=36000, show_progress_bar=True)
        best_study_params = study.best_params
        # Print the best trial
        print('Number of finished trials: {}'.format(len(study.trials)))
        trial = study.best_trial
        print('Best trial auc_score: {:.6f}'.format(trial.value))

    else:
        print('No need to run optuna, we will use the parameters obtained earlier')
        best_study_params = {'iterations': 660, 
                             'learning_rate': 0.07203497789356407, 
                             'colsample_bylevel': 0.9339427810126282, 
                             'random_strength': 0.21917140744502742, 
                             'depth': 10, 
                             'bootstrap_type': 'MVS'}
    print('best params: {}'.format(best_study_params))
    return best_study_params

cat_best_params = Run_Pass_cat_study(100)


# Choice of the estimator
model = 'xgb'

if model=='cat':
    estimator = CatBoostRegressor(**cat_best_params,
                                  verbose=500, 
                                  eval_metric='R2', 
                                  eval_fraction=0.2, 
                                  early_stopping_rounds=600)
elif model=='xgb':
    estimator = XGBRegressor(**xgb_best_params)
elif model=='lgb':
    estimator = LGBMRegressor(verbose=-1)


pipeline_steps = [
                  ('eng', Feature_Eng()), 
                  ('high_corr_drop', DHCF()),
                  ('transformer', column_trans),
                  ('estimator', estimator)
               ]

reg = Pipeline(pipeline_steps)
reg


spliter = KFold(n_splits=6, shuffle=True, random_state=15)

plt.figure(figsize=(14,8))
for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y), start=1):
    X_tr, X_va = X.loc[tr_ind], X.loc[va_ind]
    y_tr, y_va = y.loc[tr_ind], y.loc[va_ind]
    # fit and predict for the fold
    reg.fit(X_tr, y_tr)
    y_va_hat = reg.predict(X_va)
    y_va_hat = np.clip(y_va_hat, orig_raw[target].min(), orig_raw[target].max())
    # score the prediction for the fold
    rmsle_score = rmsle(y_va, y_va_hat)
    r2 = r2_score(y_va, y_va_hat)
    # plot a scatter plot to compare predicted vs true values
    plt.subplot(2,3,f)
    sns.scatterplot(x=y_va, y=y_va_hat)
    plt.title("Fold_{} || RMSLE score: {:.4f} || r2: {:.4}".format(f, rmsle_score, r2), color='maroon')
    plt.xlabel('true_values')
    plt.ylabel('predicted')
plt.tight_layout()


reg_final = Pipeline(pipeline_steps)

reg_final.fit(X, y)


test_pred = np.clip(reg_final.predict(test_raw), 
                    orig_raw[target].min(), 
                    orig_raw[target].max())

submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission[target] = test_pred

submission.to_csv('submission.csv', index=False)

submission[target].plot.kde(color='green', title='Distribution of the test predictions')
plt.show()

# Display the submission file
display(submission.head(10))
print('The file is ready for submission!')

