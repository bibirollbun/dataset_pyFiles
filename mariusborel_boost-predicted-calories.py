import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import r2_score, mean_squared_log_error, mean_squared_error, roc_auc_score, roc_curve
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

import shap

import optuna

import warnings
warnings.filterwarnings('ignore')

# verify the versions of my tools
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'sklearn version: {sklearn.__version__}')
# print(f'optuna version : {optuna.__version__}')


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test_raw = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
orig_raw = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv').drop(columns=['User_ID'])
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

target = 'Calories'

orig_raw.columns = train_raw.columns
train_raw.head(3)


train_comb = pd.concat([train_raw, orig_raw], ignore_index=True)
train_comb.tail()


# Should we engeneer the features?
create_new_features = False

# Define function for features engeneering
class Feature_Eng(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()
        if create_new_features:
            df['Max_heart_rate'] = 207 - 0.7*df['Age']
            df['%_heart_rate'] = df['Heart_Rate']/df['Max_heart_rate']
            df['heart_rate_/_weight'] = df['Heart_Rate']/df['Weight']
            df['heart_rate_/_Age'] = df['Heart_Rate']/df['Age']
            df['Heart_Rate_x_Duration'] = df['Heart_Rate']*df['Duration']
            df['Heart_Rate_x_Duration__log'] = np.log1p(df['Heart_Rate']*df['Duration'])
            df['bmi'] = df['Weight']/(0.01*df['Height'])**2
            df['log_weight_x_height'] = np.log(df['Weight']*df['Height'])
            df['body_temp_x_weight'] = df['Body_Temp']*np.log(df['Weight'])
            df['Body_Temp - 37'] = df['Body_Temp'] - 37 # assuming 37 as body temp at rest
            df['heat'] = df['Weight']*df['Body_Temp - 37']
            df['work'] = df['heat']*df['Duration']/60
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=[1, 2, 3, 4, 5]).astype('int')
            df['Age_pred_max_heart_rate'] = pd.cut(df['Age'], 
                                                   bins=[10, 30, 35, 40, 45, 50, 55, 60, 65, 70, 90], 
                                                   labels=[200, 190, 185, 180, 175, 170, 165, 160, 155, 150]
                                                  ).astype('int')
            df['diff_heart_rate'] = df['Max_heart_rate'] - df['Age_pred_max_heart_rate']
            df['Sex'] = (df['Sex'] == 'female')*1
            df['Temp_Weight_ratio'] = df['Body_Temp']/df['Weight']
            df['HeartRate_BodyTemp_ratio'] = df['Heart_Rate']/df['Body_Temp']
            df = df.drop(columns=['Duration'])      
        else:
            df['Sex'] = (df['Sex'] == 'female')*1
            df['Heart_Rate_x_Duration'] = df['Heart_Rate']*df['Duration']
            df['bmi'] = df['Weight']/(0.01*df['Height'])**2
            # df = df.drop(columns=['Duration'])
        
        return df


def data_target_prep(df):
    data_ = df.copy()
    data_ = Feature_Eng().fit_transform(data_)
    try:
        target_ = data_.pop(target)
        target_log = np.log1p(target_)
    except:
        pass

    try:
        return data_, target_, target_log
    except:
        return data_


use_orig_in_train = True

if use_orig_in_train:
    X, y, y_log = data_target_prep(train_comb)
else: 
    X, y, y_log = data_target_prep(train_raw)

X_test = data_target_prep(test_raw)

X_or, y_or, y_log_or = data_target_prep(orig_raw)


X.head()


ax = X.skew().sort_values().plot.barh(title='Features Skewness', color='Burlywood', figsize=(10, 10))
for skewness in ax.containers:
    ax.bar_label(skewness)
plt.xlim([-1.5, 1.5])
plt.xlabel('Skewness')
plt.show()


# plt.figure(figsize=(14, 8))
# for f, feat in enumerate(X.columns.tolist(), start=0):
#     if feat!='Sex':
#         plt.subplot(10, 6, f)
#         sns.violinplot(X, y=feat, x='Sex', palette='copper')
# plt.tight_layout()


# plt.figure(figsize=(14, 8))
# for f, feat in enumerate(X.columns.tolist(), start=0):
#     if feat!='Sex':
#         plt.subplot(10, 6, f)
#         sns.kdeplot(X, x=feat, hue='Sex', palette='copper', fill=True)
#         if f!=1:
#             plt.legend([])
# plt.tight_layout()


X_tr, X_va, y_log_tr, y_log_va = train_test_split(X, y_log, test_size=0.1, random_state=36)

[d.shape for d in [X_tr, X_va, y_log_tr, y_log_va]]


# Create CatBoost Pools
X_tr_pool = Pool(X_tr, y_log_tr, cat_features=['Sex'])
X_va_pool = Pool(X_va, y_log_va, cat_features=['Sex'])



def objective(trial):
    cat_param_grid = {
        "iterations": trial.suggest_int("iterations", 500, 1000, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1),
        "random_strength": trial.suggest_float("random_strength", 0.1, 1),
        "depth": trial.suggest_int("depth", 1, 16),
        "boosting_type": "Plain",
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "used_ram_limit": "10gb"
    }

    if cat_param_grid["bootstrap_type"] == "Bayesian":
        cat_param_grid["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 1)
    elif cat_param_grid["bootstrap_type"] == "Bernoulli":
        cat_param_grid["subsample"] = trial.suggest_float("subsample", 0.5, 1)

    # Train the model
    model = CatBoostRegressor(**cat_param_grid, verbose=0, eval_fraction=0.2)
    
    # fit the model
    model.fit(X_tr_pool)
    
    # Evaluate the model using RMSE
    try:
        preds = model.predict(X_va_pool)
        
        score = np.sqrt(mean_squared_error(y_log_va, preds))
        
        # score = rmsle(y_val, preds)
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
        if create_new_features:
            best_study_params = {'iterations': 850, 
                                 'learning_rate': 0.0814964582458643, 
                                 'colsample_bylevel': 0.851518636960139, 
                                 'random_strength': 0.17857527776580062, 
                                 'depth': 9, 
                                 'bootstrap_type': 'MVS'}
        else:
            best_study_params = {'iterations': 730, 
                                 'learning_rate': 0.09969738378973637, 
                                 'colsample_bylevel': 0.8121910932765485, 
                                 'random_strength': 0.2315038504123904, 
                                 'depth': 14, 
                                 'bootstrap_type': 'Bayesian', 
                                 'bagging_temperature': 0.2878598947199843}
    print('best params: {}'.format(best_study_params))
    return best_study_params

cat_best_params = Run_Pass_cat_study(50)


# Define the KFold splitter
spliter = KFold(n_splits=6, shuffle=True, random_state=42)

# Define the model pipeline
model = CatBoostRegressor(**cat_best_params, eval_metric='RMSE')

# Initialize a list to store RMSE scores
scores = []

# Set up the plot
plt.figure(figsize=(12, 8))

# Perform K-Fold cross-validation
for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y_log), start=1):
    X_tr, X_va = X.loc[tr_ind], X.loc[va_ind]
    y_tr_log, y_va_log = y_log.loc[tr_ind], y_log.loc[va_ind]
    
    # Create CatBoost Pools
    X_tr_pool = Pool(X_tr, y_tr_log, cat_features=['Sex'])
    X_va_pool = Pool(X_va, y_va_log, cat_features=['Sex'])
    
    print(f'Fold_{f}')
    
    # Fit the model
    model.fit(X=X_tr_pool, 
              eval_set=X_va_pool,
              verbose=200, 
              early_stopping_rounds=100)
    
    # Predict on validation set
    y_va_hat_log = model.predict(X_va_pool)
    
    # Reverse log transformation
    y_va = np.expm1(y_va_log)
    y_va_hat = np.expm1(y_va_hat_log)
    
    # Calculate evaluation metrics
    r2 = r2_score(y_va, y_va_hat)
    score = np.sqrt(mean_squared_error(y_va_log, y_va_hat_log))
    scores.append(score)
    
    # Plot the results
    plt.subplot(2, 3, f)
    sns.scatterplot(x=y_va, y=y_va_hat, hue=X_va['Sex'])
    plt.title(f"Fold_{f} || RMSE: {score:.4f} ~ R2: {r2:.4f}", color='Burlywood')
    plt.xlabel('True Values')
    plt.ylabel('Predicted')
    plt.tight_layout()
    
    print(f'RMSE: {score:.8f}\n\n')

# Print average RMSE
print(f'\nAverage RMSE: {np.mean(scores):.8f} Â± {np.std(scores):.8f}\n')

# Show the plot
plt.show()


3 # Fit the model
final_model =  Pipeline([
    ('est', CatBoostRegressor(**cat_best_params, eval_metric='RMSE', eval_fraction=0.2, verbose=0))]
                 ) 

final_model.fit(X, y_log, est__cat_features=['Sex'], est__early_stopping_rounds=100)

# predict on test data
pred_log = final_model.predict(X_test)

# Convert the preds from log
preds = np.round(np.expm1(pred_log), 4)
# Assign the preds to the submission data
sample_submission[target] = preds

display(sample_submission.head(20))

# Save the submission file
sample_submission.to_csv('submission.csv', index=False)
print('Your predictions are ready for submission!')

