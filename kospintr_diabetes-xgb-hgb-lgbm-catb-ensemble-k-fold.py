## Import packages

# General purpose modules
import time
from copy import deepcopy
import warnings

# Data handling and visualization modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Skikit-learn preprocessing and evaluation modules
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Skikit-learn ML modules
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

# Further ML modules
import xgboost as xgboost
import lightgbm as lightgbm
from catboost import CatBoostClassifier


## Read csv files

ADD_EXTERN_DATA = False

trainval = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
if ADD_EXTERN_DATA:
    extern_data = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
    trainval = pd.concat([trainval[trainval.columns[1:]], extern_data[trainval.columns[1:]]]
                         ).reset_index(drop=True).reset_index().rename(columns={'index':'id'})
    trainval['physical_activity_minutes_per_week'] += 1 # Add 1 minute to avoid 0 values
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

## Spliting the train data into train, val and tes

EXTENDED_STRAT = True # Stratification is based on multiple columns

strat_cols = ['family_history_diabetes', 'cardiovascular_history', 'ethnicity', 'diagnosed_diabetes']
trainval['multicat'] = LabelEncoder().fit_transform(trainval[strat_cols].astype(str).agg('_'.join, axis=1))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(trainval, trainval['multicat'] if EXTENDED_STRAT else trainval['diagnosed_diabetes']))
train = trainval.iloc[train_idx].reset_index()
val = trainval.iloc[val_idx].reset_index()
trainval_labels = trainval.pop('diagnosed_diabetes')
train_labels = train.pop('diagnosed_diabetes')
val_labels = val.pop('diagnosed_diabetes')

# Verify sizes
print(f"Total rows:   {len(trainval)}")
print(f"Dev train:    {len(train)} ({len(train)/len(trainval):.2%})")
print(f"Dev valid:    {len(val)} ({len(val)/len(trainval):.2%})")
print(f"Number of unique elements in multicat column: {len(trainval['multicat'].unique())}")


## Explore train dataset

print('List of dataset columns including data types and number of non-zero elements: ', end='\n\n')
train.info()
print('-'*80, end='\n\n')

# Explore categorical features
cat_columns = make_column_selector(dtype_include='object')(train)
print('Number of unique elements of categorical features: ', end='\n\n')
for cat in cat_columns:
    print(train[cat].value_counts(), end='\n\n')
print('-'*80, end='\n\n')

# Explore boolean features
bool_columns = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
print('Number of unique elements of boolean features: ', end='\n\n')
for bool_col in bool_columns:
    print(train[bool_col].value_counts(), end='\n\n')

# Explore numerical features
num_columns = make_column_selector(dtype_exclude='object')(train)
for col in ['index', 'id', 'physical_activity_minutes_per_week', 'family_history_diabetes',
            'hypertension_history', 'cardiovascular_history', 'multicat']:
    num_columns.remove(col)
train[num_columns].hist(bins=100, figsize=(16,10))
plt.suptitle('Probability distribution of numerical features: ')

# Explore logaritmical features
log_columns = ['physical_activity_minutes_per_week']
pd.concat([train[log_columns], np.log(train[log_columns]).rename(columns={
    'physical_activity_minutes_per_week' : 'log_physical_activity_minutes_per_week'})],
          axis=1).hist(bins=100, figsize=(16,4))
plt.suptitle('Distribution of physical activity and logarithm of physical activity')
print('-'*80, end='\n\n')


## Compare probabilty distribution of numerical features between train and validation sets

df_plot = pd.concat([train[num_columns].assign(Set='Train'),
                     val[num_columns].assign(Set='Validation')])
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn") # Suppress the specific FutureWarning

n_cols = 4
n_rows = (len(num_columns) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
fig.suptitle('Feature distributions in train and validation sets')
axes = axes.flatten()
for i in range(len(num_columns), len(axes)):
    axes[i].axis('off')
for i, col in enumerate(num_columns):
    sns.kdeplot(data=df_plot, x=col, ax=axes[i], hue='Set', common_norm=False, fill=True)
plt.tight_layout()
plt.show()
del df_plot


## Define and fit preprocessing pipeline

PCA_active = False # Activate PCA for numerical features
num_columns.remove('alcohol_consumption_per_week')
minmax_columns = ['alcohol_consumption_per_week']

stdscale_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('std_scaling', RobustScaler()), ('pca', PCA(random_state=42))] if PCA_active else
                             [('imputer', SimpleImputer(strategy="median")), ('std_scaling', RobustScaler())])
minmaxscale_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                                 ('minmax_scaling', MinMaxScaler())])
log_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                         ('log_trans', FunctionTransformer(func=np.log, feature_names_out='one-to-one')),
                         ('std_scaler', RobustScaler())])
onehot_pipeline = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")),
                            ('onehot', OneHotEncoder())])
ordinal_pipeline = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")),
                             ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
bool_pipeline = Pipeline([('imputer', SimpleImputer(strategy="most_frequent"))])

# Preprocessing pipeline
preprocessing = ColumnTransformer([("std_scaling", stdscale_pipeline, num_columns),
                                   ("minmax_scaling", minmaxscale_pipeline, minmax_columns),
                                   ("logstd_scaling", log_pipeline, log_columns),
                                   ("ordinal", ordinal_pipeline, cat_columns),
                                   #("onehot", onehot_pipeline, cat_columns),
                                   ("bool", bool_pipeline, bool_columns)])

# Preprocess data
train_prepared = preprocessing.fit_transform(train)
trainval_prepared = preprocessing.transform(trainval)
val_prepared = preprocessing.transform(val)


## Helping function to create parameter grids

def make_param(param_dict, model='est'):
    for elem in param_dict.copy():
        if elem == 'n_components':
            param_dict['pca'+'__'+elem] = param_dict.pop(elem)
        else:
            param_dict[model+'__'+elem] = param_dict.pop(elem)
    return param_dict


## Machine learning models and their hyperparameter search space

# Models
svc = SVC(kernel='linear', class_weight='balanced')
rfc = RandomForestClassifier(random_state=42)
kneigh = KNeighborsClassifier()
gbc = GradientBoostingClassifier(random_state=42)
xgb = xgboost.XGBClassifier(objective='binary:logistic', scale_pos_weight=0.604, seed=0, device='gpu', eval_metric="auc")
ada = AdaBoostClassifier(random_state=42)
hgbc = HistGradientBoostingClassifier(scoring='roc_auc', class_weight='balanced', random_state=42)
lgbm = lightgbm.LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, random_state=42, device ='cpu', verbosity=-1)
catc = CatBoostClassifier(eval_metric='AUC', auto_class_weights='Balanced', random_state=123, task_type='CPU', verbose=False)
sgdc = SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42)

# Model space
EstimatorStr = {1: 'svc', 2: 'rfc', 3: 'kneigh', 4: 'gbc', 5: 'xgb', 6: 'ada', 7: 'hgbc', 8: 'lgbm', 9: 'catc', 10: 'sgdc'}
EstimatorMdl = {1: svc, 2: rfc, 3: kneigh, 4: gbc, 5: xgb, 6: ada, 7: hgbc, 8: lgbm, 9: catc, 10: sgdc}


## Training parameters

# Tuned hyperparameter sets
# svc parameter
param_single_svc = make_param({#'C': 107, 'gamma': 0.00082, 'kernel': 'rbf', #'n_components': 300
                               })
# rfc parameter
param_single_rfc = make_param({#'n_estimators': 400, 'max_depth': 12, 'max_leaf_nodes': 100, 'min_samples_split': 3,
                               }) # 1.0/0.6923/
# kneight parameter
param_single_kneigh = make_param({#'n_estimators': 400, 'max_depth': 12, 'max_leaf_nodes': 100, 'min_samples_split': 3,
                                  }) # 0.8010/0.5936/
# gbc parameter
param_single_gbc = make_param({#'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'loss': 'huber', 'min_samples_split': 3,
                               }) # 0.7083/0.7072/
# xgb parameter
param_single_xgb = make_param(#{'n_estimators': 10000, 'learning_rate': 0.01, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.5,
                            {'enable_categorical': True, 'early_stopping_rounds': 200,
                             'learning_rate': 0.3,
                             'lambda': 0,
                             'alpha': 10,
                             'colsample_bytree': 0.5,
                             'subsample': 0.9,
                             'max_depth': 6,
                             #'min_child_weight': 5,
                               }) # 0.7304/0.7230/0.6941
# ada parameter
param_single_ada = make_param({#'estimator': DecisionTreeRegressor(max_depth=180, criterion='squared_error', max_leaf_nodes=150, min_samples_split=4),
                               #'n_estimators': 300, 'learning_rate': 0.75, 'loss': 'square',
                               }) # 0.7010/0.7013/
# hgbc parameter
param_single_hgbc = make_param({'max_iter': 1000, 'n_iter_no_change': 100,
                                }) # 0.7274/0.7241/0.6941
# lgbm parameter
param_single_lgbm = make_param({'n_estimators': 2000, 'learning_rate': 0.04151567000333162, 'num_leaves': 93, 'max_depth': 3, 
                                'min_child_samples': 97, 'subsample': 0.8336810469662667, 'colsample_bytree': 0.5021699121748862, 
                                'reg_alpha': 0.015640727219830758, 'reg_lambda': 1.374990603296636e-06,
                                }) # 0.7478/0.7258/0.6954
# catc parameter
param_single_catc = make_param({'n_estimators': 12000, 'depth': 3, 'learning_rate': 0.01,
                                'use_best_model': True, 'early_stopping_rounds': 300,
                                }) # 0.7508/0.7262
# sgdc parameter
param_single_sgdc = make_param({#'learning_rate': 0.06, 'depth': 2, 'l2_leaf_reg': 0.3
                                }) # 0.6932/0.6935


## Hyperparameter sets for parameter tuning

# xgb parameter
param_grid_xgb = make_param({#'n_estimators': [140, 500, 1000, 1500],
                             'max_depth': [2, 4, 6, 8],
                             'learning_rate': [0.1, 0.3, 0.5, 0.7],
                             'subsample': [0.1, 0.5, 0.9],
                             'colsample_bytree': [0.1, 0.5, 0.9],
                             'reg_lambda': [0, 0.1, 1, 10],
                             'reg_alpha': [0, 0.1, 1, 10],
                             })
# hgbc parameter
param_grid_hgbc = make_param({#'max_depth': [3, 4, 5, 6, 9, 12],
                              #'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                              'n_components': [10,20,30,40,42],
                              })
# lgbm parameter
param_grid_lgbm = make_param({'num_leaves': [80,110,120,125,130,135,140,150,180],
                              'max_depth': [24,29,31,32,33,35,40],
                              'learning_rate': [0.05,0.1,0.2],
                              #'n_estimators': [100,200,300],
                              })
# catc parameter
param_grid_catc = make_param({#'learning_rate': 0.06, 'depth': 2, 'l2_leaf_reg': 0.3
                                })


## Single fitting with tuned parameters or grid search for machine learning methods

TUNING_ML = False # Choose between single fitting or parameter tuning
est_ids = [5,7,8,9] # Choose model(s) to tune {1: 'svc', 2: 'rfc', 3: 'kneigh', 4: 'gbc', 5: 'xgb', 6: 'ada', 7: 'hgbc', 8: 'lgbm', 9: 'cat', 10: 'sgdc'}
est_ids_w_earlystopping = [5,8,9]

for est_id in est_ids:
    start_time = time.time()
    # Define pipeline w/wo PCA
    pipeline = Pipeline([('est', EstimatorMdl[est_id])])

    # Cross-validation configuration w/wo extended stratification
    cv_gen = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        train_prepared, train['multicat'] if EXTENDED_STRAT else train_labels)
    
    # Fitting or tuning on train dataset with 5 fold cross-validation
    param = globals()[f'param_grid_{EstimatorStr[est_id]}' if TUNING_ML else f'param_single_{EstimatorStr[est_id]}']
    if TUNING_ML:
        grid = GridSearchCV(pipeline, param, scoring='roc_auc', verbose=1, cv=cv_gen)
        grid.fit(train_prepared, np.array(train_labels))
        print(grid.best_params_)
        pipeline_train = grid.best_estimator_
    else:
        for i, (train_index, eval_index) in enumerate(cv_gen):
            X_train, X_eval = train_prepared[train_index], train_prepared[eval_index]
            y_train, y_eval = train_labels[train_index], train_labels[eval_index]
            pipeline_train = deepcopy(pipeline)
            pipeline_train.set_params(**param)
            if est_id in est_ids_w_earlystopping:
                eval_set = {'est__eval_set': [(X_eval, np.array(y_eval))]}
                if est_id==5: eval_set['est__verbose'] = 0
                pipeline_train.fit(X_train, np.array(y_train), **eval_set)
            else:
                pipeline_train.fit(X_train, np.array(y_train))
            globals()[f'model{i+1}_{EstimatorStr[est_id]}'] = pipeline_train

    # Fitting on train set for evaluation purposes
    pipeline_val = deepcopy(pipeline)
    pipeline_val.set_params(**grid.best_params_ if TUNING_ML else param)
    if est_id in est_ids_w_earlystopping:
        pipeline_val.fit(train_prepared, np.array(train_labels),
                         est__eval_set=[(val_prepared, val_labels)])
    else:
        pipeline_val.fit(train_prepared, np.array(train_labels))
    globals()[f'model_val_{EstimatorStr[est_id]}'] = pipeline_val

    # Calculate and show scores
    train_score = roc_auc_score(np.array(train_labels), pipeline_val.predict_proba(train_prepared)[:,1])
    val_score = roc_auc_score(np.array(val_labels), pipeline_val.predict_proba(val_prepared)[:,1])
    print(f'Estimator: {EstimatorStr[est_id]}')
    print(f'Train ROC_AUC score: {train_score}')
    print(f'Val ROC_AUC score: {val_score}')
    print(f'Elapsed time: {int(time.time() - start_time)} [s]')
    print('-'*40)


## ROC AUC score on mean value of ensemble predictions on validation dataset

val_pred = pd.DataFrame()

for est_id in est_ids:
    model_val = globals()[f'model_val_{EstimatorStr[est_id]}']
    val_pred[f'pred_{EstimatorStr[est_id]}'] = model_val.predict_proba(val_prepared)[:,1]
val_pred_avg = val_pred[[f'pred_{EstimatorStr[est_id]}' for est_id in est_ids]].mean(axis=1)
val_score_avg = roc_auc_score(np.array(val_labels), np.array(val_pred_avg))
print(f'Average val ROC_AUC score of {[EstimatorStr[i] for i in est_ids]} estimators: {val_score_avg}')


## Show feature importances

if est_id in [5, 8]:
    # Sort feature names and their importances
    categories = pipeline_train[0].get_feature_names_out() if PCA_active else preprocessing.get_feature_names_out()
    values = pipeline_train[-1].feature_importances_+1e-4
    sorted_values, sorted_categories = zip(*sorted(zip(values,categories), reverse=False))
    
    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('viridis')
    rescale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    normalized_values = rescale(np.log(sorted_values))
    ax.barh(sorted_categories, sorted_values, color=cmap(normalized_values), log=True)
    plt.title(f'Feature Importances (Magnitude)')
    plt.xlabel('Logarithmic importance score')
    plt.ylabel('Features')
    plt.tight_layout()
    ax.tick_params(left=False, bottom=True)


## Test prediction & submission 

test_prepared = preprocessing.transform(test)
test_pred = pd.DataFrame()
submission_df = test[['id']].copy()

# Make predictions for all folds and all estimators and take the mean value as prediction
for i in range(5):
    for est_id in est_ids:
        model_test = globals()[f'model{i+1}_{EstimatorStr[est_id]}']
        test_pred[f'pred{i+1}_{EstimatorStr[est_id]}'] = model_test.predict_proba(test_prepared)[:,1]
submission_df['diagnosed_diabetes'] = test_pred[[f'pred{i+1}_{EstimatorStr[est_id]}' for est_id in est_ids for i in range(5)]].median(axis=1)

submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")
submission_df

