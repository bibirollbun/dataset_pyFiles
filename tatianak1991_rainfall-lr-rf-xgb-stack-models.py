import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

import optuna


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df  = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
df_original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')


df_original['rainfall'] = df_original['rainfall'].map(lambda x: 1 if x == 'yes' else 0)
df_original.columns = df_original.columns.str.replace(' ', '')


col_order = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
       'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed',
       'rainfall']

assert all(col in df_original.columns for col in col_order)
df_original = df_original[col_order]
df_original.describe()
df_original = df_original.fillna(df_original.median())


df = pd.concat([df, df_original], axis=0, ignore_index=True)


class WeatherFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create synthetic weather-related features
    
    This transformer calculates humidity-related metrics, temperature differences, 
    wind power, pressure changes, and cyclic features for seasonality.
    
    Methods:
    --------
    - fit(X, y=None): Returns self (no fitting needed).
    - transform(X): Adds new weather-related features to the DataFrame.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def safe_specific_humidity(dewpoint):
        if dewpoint is None or np.isnan(dewpoint):  
            return np.nan
        return (6.11 * 10**(7.5 * dewpoint / (237.3 + dewpoint))) if (237.3 + dewpoint) != 0 else np.nan

    @staticmethod
    def safe_saturation_point(temperature):
        if temperature is None or np.isnan(temperature):  
            return np.nan
        return (6.11 * 10**(7.5 * temperature / (237.3 + temperature))) if (237.3 + temperature) != 0 else np.nan

    def fit(self, X, y=None):
        """No fitting required, just return self."""
        return self

    def transform(self, X):
        """Applies weather feature engineering to a Pandas DataFrame."""
        X = X.copy()
        X.fillna(X.median(), inplace=True)
        
        # Feature engineering
        X['specific_humidity'] = X['dewpoint'].apply(self.safe_specific_humidity)
        X['saturation'] = X['temparature'].apply(self.safe_saturation_point)
        X['relative_humidity'] = (X['specific_humidity'] / X['saturation']) * 100
        X['temp_diff'] = X['maxtemp'] - X['mintemp']
        X['dew_point_depression'] = X['temparature'] - X['dewpoint']
        X['wet_bulb_temp'] = X['temparature'] - (X['dew_point_depression'] / 3)
        X['u_wind'] = X['windspeed'] * np.cos(np.radians(X['winddirection']))
        X['v_wind'] = X['windspeed'] * np.sin(np.radians(X['winddirection']))
        X['pressure_change'] = X['pressure'].diff()
        X['sunshine_ratio'] = X['sunshine'] / 24
        X['vpd'] = X['saturation'] - X['specific_humidity']
        X['humidity_temp_ratio'] = X['humidity'] / (X['temparature'] + 1)
        X['wind_humidity_interaction'] = X['windspeed'] * X['humidity']

        X.fillna(X.median(), inplace=True)
        
        return X



class AddClusterLabels(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)

    def fit(self, X, y=None):
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        # Add the cluster labels as an additional column to the data
        labels = self.kmeans.predict(X).reshape(-1, 1)
        return np.hstack([X, labels])


class AddPrincipalComponents(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        # Apply PCA and append the components as new columns to the data (in array form)
        pca_components = self.pca.transform(X)
        return np.hstack([X, pca_components])


preprocessor_simple = Pipeline(steps=[
                                        ('add_new_features', WeatherFeatureEngineer()),  
                                        ('imputer', SimpleImputer(strategy='median')),  
                                        ('scaler', StandardScaler())
                                     ])


preprocessor_with_PC_Clusters =  Pipeline(steps=[
                                           ('preprocessor', preprocessor_simple),
                                           ('clustering', AddClusterLabels(n_clusters=2)),
                                           ('pca', AddPrincipalComponents(n_components=15)),
                                           ])


preprocessor_to_PC =  Pipeline(steps=[
                                           ('preprocessor', preprocessor_simple),
                                           ('clustering', AddClusterLabels(n_clusters=2)),
                                           ('pca', PCA(n_components=15))])


X = df.drop(columns=['rainfall'], axis=1)
y = df['rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# smote = SMOTE(random_state = 42)
# X_train, y_train = smote.fit_resample(X_train, y_train)


X_dict = {'X_simple': [preprocessor_simple.fit_transform(X_train), preprocessor_simple.transform(X_test)],
          
          'X_PC_cluster': [preprocessor_with_PC_Clusters.fit_transform(X_train, y_train), preprocessor_with_PC_Clusters.transform(X_test)],
          
          'X_to_PC': [preprocessor_to_PC.fit_transform(X_train), preprocessor_to_PC.transform(X_test)]}


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING) 


def auc_roc_plot(y_true: object, y_prob: object, title: str) -> None:
    '''Plot auc_roc curve with thresholds'''
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = (tpr - fpr).argmax()  # Find best tradeoff between TPR and FPR
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal Threshold for ROC AUC: {optimal_threshold}")
    
    auc_score = roc_auc_score(y_true, y_prob)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', color='blue', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.7)  # Random classifier line
    
    # Annotate some key threshold points
    for i in range(len(thresholds)):
        if i % 4 == 0:  # annotate every 3rd threshold
            plt.annotate(f'{thresholds[i]:.2f}', (fpr[i], tpr[i]), 
                         textcoords="offset points", xytext=(5,-5), ha='left', fontsize=8)
    
    # Labels and title
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


from optuna.exceptions import TrialPruned

def objective(trial: object, X_train: object, y_train: object, params: dict, model: object) -> float:
    # Hyperparameter search
    try:
        model = model(**params)
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        auc = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc').mean()  
        return auc 

    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise TrialPruned()


from sklearn.utils.class_weight import compute_class_weight

class_weigths = compute_class_weight('balanced', classes=[0, 1], y=y_train)


class_weigths 


from xgboost import XGBClassifier


def define_XGB_params(trial: object) -> object:
        params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 350),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'random_state': 42
        }
        return params


# best_XGB_params = {}  

# for key, value in X_dict.items():
#     study_XGB = optuna.create_study(direction='maximize')
#     study_XGB.optimize(lambda trial: objective(trial, value[0], y_train, params=define_XGB_params(trial), model = XGBClassifier), n_trials=30)
        
#     # Store the best params for each transformation
#     best_XGB_params[key] = study_XGB.best_params



# Show best param 
best_XGB_params = {'X_simple': {'n_estimators': 105,
                  'max_depth': 4,
                  'learning_rate': 0.012588668701686493,
                  'subsample': 0.8164678815377,
                  'colsample_bytree': 0.8674494851803404},
                 'X_PC_cluster': {'n_estimators': 244,
                  'max_depth': 3,
                  'learning_rate': 0.01090945547502421,
                  'subsample': 0.7828604832408981,
                  'colsample_bytree': 0.6319690828350122},
                 'X_to_PC': {'n_estimators': 233,
                  'max_depth': 4,
                  'learning_rate': 0.016241665938486742,
                  'subsample': 0.5466428771711898,
                  'colsample_bytree': 0.7782554725571741}}


# Train models with best params

models_XGB = {}  
aucs = {}
    
for key, value in X_dict.items():
    # Initialize the model with the best parameters
    model = XGBClassifier(**best_XGB_params[key], scale_pos_weight = class_weigths[1] / class_weigths[0])      
    model.fit(value[0], y_train)
        
    models_XGB[key] = model   
    y_pred_proba = model.predict_proba(value[1])[:, 1]
        
    auc = roc_auc_score(y_test, y_pred_proba)
    aucs[key] = auc
    auc_roc_plot(y_test, y_pred_proba, key)

print(aucs)


from sklearn.ensemble import RandomForestClassifier


def define_rf_params(trial: object) -> object:

    rf_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 150),
        'random_state': 42 
    }

    return rf_params


# best_rf_params = {}

# # find best params 
# for key, value in X_dict.items():
#     study = optuna.create_study(direction='maximize')
#     study.optimize(lambda trial: objective(trial, value[0], y_train, params=define_rf_params(trial), model = RandomForestClassifier), n_trials=30)
        
#     # Store the best params for each transformation
#     best_rf_params[key] = study.best_params


best_rf_params = {'X_simple': {'n_estimators': 277,
                  'max_depth': 12,
                  'min_samples_split': 8,
                  'max_leaf_nodes': 42},
                 'X_PC_cluster': {'n_estimators': 395,
                  'max_depth': 11,
                  'min_samples_split': 7,
                  'max_leaf_nodes': 20},
                 'X_to_PC': {'n_estimators': 350,
                  'max_depth': 15,
                  'min_samples_split': 7,
                  'max_leaf_nodes': 34}}



aucs = {}
rf_models = {}

for key, value in X_dict.items():

    rf_model = RandomForestClassifier(**best_rf_params[key], class_weight = {0: class_weigths[0], 1: class_weigths[1]})
    rf_model.fit(value[0], y_train)
    rf_models[key] = rf_model
    
    y_pred = rf_model.predict(value[1])
    y_prob = rf_model.predict_proba(value[1])[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    aucs[key]=auc
    auc_roc_plot(y_test, y_prob, key)

print(aucs)


def check_vif(train):
  vif = pd.DataFrame()
  vif["feature"] = train.columns
  vif["VIF"] = [variance_inflation_factor(train.values, i) for i in range(len(train.columns))]
  return vif

    # vifs = check_vif(X_smote_scaled_df)
# vifs.sort_values(by='VIF', ascending=False)


def objective(trial, X_train, y_train):
    # Define hyperparameter search space
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    C = trial.suggest_loguniform('C', 1e-7, 1e7)
    max_iter = trial.suggest_int('max_iter', 100, 20000)
    
    if penalty == 'l1':
        solver = 'liblinear'  
        l1_ratio = None  
    elif penalty == 'elasticnet':
        solver = 'saga'  
        l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
    else:
        solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'saga'])
        l1_ratio = None 

    # Create the Logistic Regression model
    model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter, solver=solver, l1_ratio=l1_ratio)

    # Cross-validation to evaluate the model's performance
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc').mean()

    return auc


# best_lr_params = {}

# # find best params 
# for key, value in X_dict.items():
#     study = optuna.create_study(direction='maximize')
#     study.optimize(lambda trial: objective(trial, value[0], y_train), n_trials=30)
    
#     # Store the best parameters (including solver)
#     best_params = study.best_params.copy()
    
#     # Explicitly add solver to the dictionary if it's not in the params
#     if 'solver' not in best_params:
#         if best_params['penalty'] == 'l1':
#             best_params['solver'] = 'liblinear'
#         elif best_params['penalty'] == 'elasticnet':
#             best_params['solver'] = 'saga'
#         else:
#             best_params['solver'] = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'saga'])

#     best_lr_params[key] = best_params



best_lr_params ={'X_simple': {'penalty': 'l1',
                  'C': 0.10837905428666719,
                  'max_iter': 16495,
                  'solver': 'liblinear'},
                 'X_PC_cluster': {'penalty': 'l1',
                  'C': 0.14681933360698182,
                  'max_iter': 10092,
                  'solver': 'liblinear'},
                 'X_to_PC': {'penalty': 'l1',
                  'C': 0.22625210971440388,
                  'max_iter': 4540,
                  'solver': 'liblinear'}}


aucs = {}
lr_models = {}

for key, value in X_dict.items():

    lr_model = LogisticRegression(**best_lr_params[key], class_weight = {0: class_weigths[0], 1: class_weigths[1]})
    lr_model.fit(value[0], y_train)
    lr_models[key] = lr_model
    
    y_pred = lr_model.predict(value[1])
    y_prob = lr_model.predict_proba(value[1])[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    aucs[key]=auc
    auc_roc_plot(y_test, y_prob, key)

print(aucs)


from sklearn.ensemble import VotingClassifier


ensemble_models = {}


for key, value in X_dict.items(): 

    ensemble_model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(**best_lr_params[key],  class_weight = {0: class_weigths[0], 1: class_weigths[1]})),
        ('rf', RandomForestClassifier(**best_rf_params[key],  class_weight = {0: class_weigths[0], 1: class_weigths[1]})),
        ('xgb', XGBClassifier(**best_XGB_params[key], scale_pos_weight = class_weigths[1] / class_weigths[0]))
    ], voting='soft', weights=[6, 4, 1])


    
    en_model = ensemble_model.fit(value[0], y_train)
    ensemble_models[key] = en_model
    
    y_prob = en_model.predict_proba(value[1])[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f'For {key}  auc = {auc}')
    auc_roc_plot(y_test, y_prob, key)


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier


stack_models = {}



from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

for key, value in X_dict.items():
    # Base learners
    base_learners = [
        ('lr', LogisticRegression(**best_lr_params[key],  class_weight = {0: class_weigths[0], 1: class_weigths[1]})),
        ('rf', RandomForestClassifier(**best_rf_params[key],  class_weight = {0: class_weigths[0], 1: class_weigths[1]})),
        ('svm', SVC(probability=True)),
        ('xgb', XGBClassifier(**best_XGB_params[key], scale_pos_weight = class_weigths[1] / class_weigths[0])),
    ]

    # Meta-model (level 1)
    meta_model = LogisticRegression(**best_lr_params[key], class_weight = {0: class_weigths[0], 1: class_weigths[1]})
    
    # Create a StackingClassifier
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, passthrough=True, cv=6)
    st_model = stacking_model.fit(value[0], y_train)
    stack_models[key] =  st_model
    
    y_prob = st_model.predict_proba(value[1])[:, 1]
    auc_roc_plot(y_test, y_prob, key)
    


stack_models


X_test_dict = {'X_simple': preprocessor_simple.transform(test_df),
          
              'X_PC_cluster': preprocessor_with_PC_Clusters.transform(test_df),
          
              'X_to_PC': preprocessor_to_PC.transform(test_df)}


for mkey, model in stack_models.items():
    for dkey, data in X_test_dict.items():
        if mkey == dkey and mkey == 'X_to_PC':
            
            y_prob = model.predict_proba(data)[:, 1]
            final_df = pd.DataFrame({'rainfall': y_prob},index=test_df.index)

            final_df.to_csv(f'/kaggle/working/submission.csv')
            print(f'submission created')





