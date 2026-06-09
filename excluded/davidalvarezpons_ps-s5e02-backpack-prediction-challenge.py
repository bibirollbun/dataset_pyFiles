# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import imp 
try: 
    imp.find_module('dython')
except ImportError: 
    !pip install dython > /dev/null

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


trainDs = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
trainExtraDs = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col='id')
testDs = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')


trainDs


testDs


trainExtraDs


def printStatistics(ds): 
    print('Column classification')
    numeric_columns = [ c for c in ds.columns if ds[c].dtype in ['float64'] ]
    date_columns = [ c for c in ds.columns if ds[c].dtype == 'datetime64[ns]' ]
    categoric_columns = [ c for c in ds.columns if ds[c].dtype == 'object' ]
    
    print('numeric:', numeric_columns)
    print('date:', date_columns)
    print('categoric: ', categoric_columns)

    print('Numeric statistics')
    for c in numeric_columns: 
        print(c)
        print(f"\t > Max: {ds[c].max()}")
        print(f"\t > Min: {ds[c].min()}")
        print(f"\t > Mean: {ds[c].mean()}")
        print(f"\t > StdDev: {ds[c].std()}")
        print(f"\t > Num missing: {sum(ds[c].isna())}")
        print(f"\t > Num Values:", ds[c].nunique())

    print('Categoric statistics')
    for c in categoric_columns: 
        print(c)
        print(f"\t > Num values: {ds[c].nunique()}")
        print(f"\t > Num missing: {sum(ds[c].isna())}")
        print(f"\t > Values:", ds[c].unique())

    print('Datetime statistics')
    for c in date_columns: 
        print(c)
        print(f"\t > Most recent: {trainDs[c].max()}")
        print(f"\t > Oldest:", trainDs[c].min())
        print(f"\t > Num missing: {sum(trainDs[c].isna())}")


printStatistics(trainDs)


printStatistics(trainExtraDs)


printStatistics(testDs)


sns.pairplot(trainDs)


sns.pairplot(trainExtraDs)


trainAllDs = pd.concat([trainDs, trainExtraDs], axis=0)


print("Duplicated rows: ", sum(trainAllDs.duplicated()))


trainAllDs.columns


trainAllDs['Weight Capacity (kg)'].describe()


%%time
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

columns = list(trainAllDs.columns)
imputer = ColumnTransformer(transformers=[
    ('numeric', KNNImputer(), ['Weight Capacity (kg)']),
    ('unknown', SimpleImputer(strategy='constant', fill_value='Unknown'), ['Laptop Compartment', 'Waterproof']),
    ('other', SimpleImputer(strategy='constant', fill_value='Other'), ['Brand', 'Material', 'Size', 'Style', 'Color']),
], remainder='passthrough', verbose_feature_names_out=False)

imputedDs = pd.DataFrame(imputer.fit_transform(trainAllDs.iloc[:1000]), columns=imputer.get_feature_names_out())


# Check that NaN data has actually gone away
imputedDs[imputedDs['Weight Capacity (kg)'].isna()]


# Compare to previous describe
imputedDs['Weight Capacity (kg)'].astype('float64').describe()





def visualizeData(data):     
    for i, col in enumerate(data.columns):
        print("Visualizing data from column: ", col)
        plt.figure(i)
        if data[col].dtype == 'object':
            plot=sns.histplot(data[col], label=col)
            for index, item in enumerate(plot.get_xticklabels()):
                item.set_rotation(45)
        elif data[col].dtype == 'float64':
            plot=sns.displot(data[col], label=col)
        elif data[col].dtype == 'datetime64[ns]': 
            pass
        plt.show()


visualizeData(trainDs)


visualizeData(trainExtraDs)


visualizeData(trainAllDs)





target = trainAllDs['Price']


trainAllDs.describe()


%%time 

from dython.nominal import associations

def bivariateAnalysis(data, target): 
    # Integer values
    plt.figure(1)
    numericData = data.loc[:, ['Weight Capacity (kg)', 'Compartments']]
    sns.heatmap(numericData.assign(target=target).corr(), annot=True)
    associations(dataset=data.select_dtypes(include='object').assign(target=target), nominal_columns='all',plot=True)

bivariateAnalysis(trainAllDs, target)


from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

def makePreprocessingPipelines():
    weightPrep = Pipeline([
        ('weight_imputer', KNNImputer()), 
        ('scaler', StandardScaler()),
    ])
    compartmentsPrep = Pipeline([
        ('scaler', StandardScaler()),
    ])

    sizePrep = Pipeline([
        ('ordinal_imputer', SimpleImputer(strategy='constant', fill_value='Other')),
        ('ordinal_encoder', OrdinalEncoder()),
    ])

    booleanPrep = Pipeline([
        ('boolean_imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
        ('one_hot_encoder', OneHotEncoder()),
    ])

    categoricalPrep = Pipeline([
        ('category_imputer', SimpleImputer(strategy='constant', fill_value='Other')),
        ('one_hot_encoder', OneHotEncoder()),
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('weight', weightPrep, ['Weight Capacity (kg)']),
        ('compartments', compartmentsPrep, ['Compartments']),
        ('size', sizePrep, ['Size']),
        ('boolean', booleanPrep, ['Laptop Compartment', 'Waterproof']),
        ('categorical', categoricalPrep, ['Brand', 'Material', 'Style', 'Color'])
    ], remainder='passthrough', verbose_feature_names_out=False)
    
    trainPipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('pandarizer', FunctionTransformer(
            lambda x: pd.DataFrame(
                x.toarray(), 
                columns = preprocessor.get_feature_names_out()
            )
        )),
    ])
    
    targetPipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])
    
    return trainPipeline, targetPipeline 



trainDs.dtypes


pipeX = trainAllDs[:1000].drop('Price', axis=1)
pipeY = pd.DataFrame({'Price':trainAllDs.iloc[:1000]['Price']})
auxTrainPipe, auxTargetPipe = makePreprocessingPipelines()


ds = auxTrainPipe.fit_transform(pipeX)
ds.head()


ds.describe()


# Show differences in distribution
pd.concat(
    [
        pipeY.describe(), 
        pd.DataFrame(
            auxTargetPipe.fit_transform(pipeY), 
            columns=['Transformed']).describe()
    ], 
  axis=1)


modellingDs = trainAllDs #.sample(1000000)


from sklearn.model_selection import train_test_split

targetColumn = 'Price'

y_all = modellingDs[targetColumn]
X_all = modellingDs.drop(targetColumn, axis=1)
X_train, X_eval, y_train_raw, y_eval_raw = train_test_split(X_all, y_all, test_size=.20)


trainPreprocessor, targetPreprocessor = makePreprocessingPipelines()


y_train = targetPreprocessor.fit_transform(pd.DataFrame(y_train_raw)).reshape(-1)
y_eval = targetPreprocessor.transform(pd.DataFrame(y_eval_raw)).reshape(-1)


from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

estimators = [
    # Random Forest: https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    ('random_forest', RandomForestRegressor(criterion='squared_error', max_depth=5, n_estimators=10)),
    # Neural Networks: https://scikit-learn.org/1.5/modules/neural_networks_supervised.html
    ('neural-net_mlp', MLPRegressor(hidden_layer_sizes=(10,), max_iter=3, early_stopping=True)), 
    # XGBoost: https://xgboost.readthedocs.io/en/stable/parameter.html
    ('xgb', XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_jobs=None)),
]


from datetime import datetime

def makeModellingPipelines(preprocessor, estimators, X, y):
    pipelines = {}
    for model in estimators:
        pipeName = model[0]
        print(f"Starting training {pipeName}...")
        pipe = Pipeline(steps=[('preprocessor', preprocessor), model])
        t0 = datetime.now()    
        pipe.fit(X, y)
        pipelines[pipeName] = pipe
        t1 = datetime.now()
        duration = t1 - t0
        print(f"\tFinished {pipeName}, took: {duration.total_seconds()}s")
    return pipelines


%%time

pipelines = makeModellingPipelines(trainPreprocessor, estimators, X_train, y_train)


from sklearn.metrics import mean_squared_error
import math

def print_predictions(target, predictions):
    print(f"")

from sklearn.model_selection import KFold,cross_validate
import matplotlib.pyplot as plt
%matplotlib inline

def plot_estimators(pipelines, X, y, n_splits=5): 
    scorers = []
    labels = []
    for name, model in pipelines.items(): 
        print(f"Cross-validating model {name}...")
        labels.append(name)
        kf = KFold(n_splits)
        t0 = datetime.now()
        model_score = cross_validate(model, X, y, scoring={'rmse': 'neg_root_mean_squared_error'}, cv=kf)
        scorers.append(model_score)
        t1 = datetime.now()
        duration = t1 - t0
        print(f"\tFinished {name} in {duration.total_seconds()}s")

    score_lists = {'rmse': [ s['test_rmse'] for s in scorers] }
    for i, (title, _list) in enumerate(score_lists.items()): 
        plt.figure(i)
        positive_scores = [ -1 * l for l in _list ]
        plot = sns.boxplot(data=positive_scores).set_xticklabels(labels, rotation=45)
        plt.title(title)

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    '''Generate a simple plot of the test and training learning curve'''
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def evaluatePipelines(pipes, X_train, y_train, X_eval, y_eval, estimators=True, learning_curves=True):
    for name, model in pipes.items(): 
        y_train_org = targetPreprocessor.inverse_transform(pd.DataFrame(y_train)).reshape(-1)
        y_eval_org = targetPreprocessor.inverse_transform(pd.DataFrame(y_eval)).reshape(-1)

        y_train_pred_raw = model.predict(X_train)
        y_train_pred = targetPreprocessor.inverse_transform(pd.DataFrame(y_train_pred_raw)).reshape(-1)

        y_eval_pred_raw = model.predict(X_eval)
        y_eval_pred = targetPreprocessor.inverse_transform(pd.DataFrame(y_eval_pred_raw)).reshape(-1)
        
        print(f'[{name}] Train - rmse: {math.sqrt(mean_squared_error(y_train_org,y_train_pred))}')
        print(f'[{name}] Train(scaled) - rmse: {math.sqrt(mean_squared_error(y_train,y_train_pred_raw))}')
        print(f'[{name}] Eval - rmse: {math.sqrt(mean_squared_error(y_eval_org,y_eval_pred))}')
        print(f'[{name}] Eval(scaled) - rmse: {math.sqrt(mean_squared_error(y_eval,y_eval_pred_raw))}')
    if estimators:   
        plot_estimators(pipes, X_train, y_train)
    
    if learning_curves: 
        for name, model in pipes.items(): 
            g = plot_learning_curve(
                model,
                name + ' learning curves',
                X_train,
                y_train,
                cv=KFold(5),
                n_jobs=4
            )


%%time
evaluatePipelines(pipelines, X_train, y_train, X_eval, y_eval, estimators=False, learning_curves=False)





from sklearn.model_selection import GridSearchCV

class HyperParameterTuning: 

    def __init__(self, pipelines, X, y): 
        self.pipes = pipelines
        self.data = X 
        self.target = y 
        self._forceRun = False

    def forceRun(self): 
        self._forceRun = True 

    def getParameters(self, name): 
        if name not in self.pipes: 
            print(f"Could not find a pipeline named: '{name}'")
            return []
        return self.pipes[name].named_steps[name].get_params()

    def tuneParamGrid(self, name, param_grid, refit='rmse', cv=5):
        # Comment out the next line to allow skipping tuning
        # self._forceRun = True 
        if not self._forceRun:
            print('Skipping for performance issues. Call `tuner.forceRun` to activate it.')
            return 
        self._forceRun = False
        model = self.pipes[name]
        param_grid = { f"{name}__{key}" : param_grid[key] for key in param_grid.keys() }
        metrics = {'rmse': 'neg_root_mean_squared_error'}
        print(f"Fine-tuning model {name}...")
        t0 = datetime.now()
        xgbcv = GridSearchCV(model, param_grid, scoring=metrics, refit=refit, cv=cv, return_train_score=True, verbose=2)
        xgbcv.fit(self.data, self.target)
        t1 = datetime.now()
        duration = t1 - t0
        print(f"\tFinished {name} in {duration.total_seconds()}s")
    
        print('best score: ' + str(xgbcv.best_score_))
        print('best params: ' + str(xgbcv.best_params_))
        results = pd.DataFrame(xgbcv.cv_results_)

        if len(param_grid) == 1: 
            for i,param in enumerate(param_grid.keys()):
                param_col = 'param_'+param
                graph_data = results[[param_col, 'mean_test_'+refit, 'mean_train_'+refit]]
                graph_data[param_col] = [ self._tupleKey(v) if type(v) is tuple else v for v in graph_data[param_col] ]
                graph_data = graph_data.rename(columns={'mean_test_'+refit:'test', 'mean_train_'+refit:'train'})
                graph_data = graph_data.melt('param_'+param, var_name='type', value_name=refit)
                plt.figure(i)
                plot = sns.lineplot(x='param_'+param, y=refit, hue='type', data=graph_data)
        elif len(param_grid) == 2: 
            param1 = list(param_grid.keys())[0]
            param2 = list(param_grid.keys())[1]
            graph_data = results[['param_'+param1,'param_'+param2,'mean_test_'+refit]]
            graph_data = graph_data.pivot(index='param_'+param1, columns='param_'+param2, values='mean_test_'+refit)
            sns.heatmap(graph_data, annot=True, xticklabels=True, yticklabels=True).set(xlabel=param2, ylabel=param1)

    def _tupleKey(self, t): 
        return '-'.join([str(ti) for ti in t])

tuner = HyperParameterTuning(pipelines, X_train, y_train)


best_params = {}


tuner.getParameters('random_forest')


n_samples, n_features = X_train.shape


param_grid={'n_estimators': [10, 100, 1000]}
# tuner.forceRun()
tuner.tuneParamGrid('random_forest', param_grid, cv=5)


param_grid={'max_depth': list(range(5, n_features + 1, 2))}
# tuner.forceRun()
tuner.tuneParamGrid('random_forest', param_grid, cv=5)


import math
param_grid={'min_samples_split': list(range(1, int(math.log(n_samples) + 1)))}
# tuner.forceRun()
tuner.tuneParamGrid('random_forest', param_grid, cv=5)


param_grid={'min_samples_leaf': list(range(1, int(math.log(n_samples) + 1)))}
# tuner.forceRun()
tuner.tuneParamGrid('random_forest', param_grid, cv=5)


param_grid={'max_leaf_nodes': [ 10 ** e for e in range(5) ]}
# tuner.forceRun()
tuner.tuneParamGrid('random_forest', param_grid, cv=5)


param_grid={'ccp_alpha': [ x / 10 for x in range(0, 50, 5)]}
# tuner.forceRun()
tuner.tuneParamGrid('random_forest', param_grid, cv=5)


best_params['random_forest'] = {
    # Fine-tuned
    'n_estimators': 1000,
    'max_depth': 7,
    # Default
    'bootstrap': True,
    'ccp_alpha': 0.0,
    'criterion': 'squared_error',
    'max_features': 1.0,
    'max_leaf_nodes': None,
    'max_samples': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'n_jobs': None,
    'oob_score': False,
    'random_state': None,
    'verbose': 0,
    'warm_start': False
}


tuner.getParameters('neural-net_mlp')


param_grid={'hidden_layer_sizes': [ (20, s, 100, ) for s in [10, 100, 1000 ] ] }
# tuner.forceRun()
tuner.tuneParamGrid('neural-net_mlp', param_grid, cv=5)


param_grid={'alpha': [ 10**-e for e in range(1, 10, 2)]}
# tuner.forceRun()
tuner.tuneParamGrid('neural-net_mlp', param_grid, cv=5)


param_grid={'max_iter': [10, 100, 1000] }
# tuner.forceRun()
tuner.tuneParamGrid('neural-net_mlp', param_grid, cv=5)


param_grid={'early_stopping': [ True, False ]}
# tuner.forceRun()
tuner.tuneParamGrid('neural-net_mlp', param_grid, cv=5)


param_grid={'beta_1': [ 0, 0.3, 0.6, 0.9 ], 'beta_2': [ 0, 0.333, 0.666, 0.999 ]}
# tuner.forceRun()
tuner.tuneParamGrid('neural-net_mlp', param_grid, cv=5)


param_grid={'n_iter_no_change': list(range(10, 50, 10))}
# tuner.forceRun()
tuner.tuneParamGrid('neural-net_mlp', param_grid, cv=5)


param_grid={'hidden_layer_sizes': [ (s, ) for s in [100, 200, 500, 1000, 2000, 3000] ] }
# tuner.forceRun()
tuner.tuneParamGrid('neural-net_mlp', param_grid, cv=5)


best_params['neural-net_mlp'] = {
    # Fine-tuned
    'hidden_layer_sizes': (20, 1000, 100),
    'max_iter': 1000,
    # Default
    'activation': 'relu',
    'alpha': 0.0001,
    'batch_size': 'auto',
    'beta_1': 0.9,
    'beta_2': 0.999,
    'early_stopping': True,
    'epsilon': 1e-08,
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'max_fun': 15000,
    'momentum': 0.9,
    'n_iter_no_change': 10,
    'nesterovs_momentum': True,
    'power_t': 0.5,
    'random_state': None,
    'shuffle': True,
    'solver': 'adam',
    'tol': 0.0001,
    'validation_fraction': 0.1,
    'verbose': False,
    'warm_start': False
}


tuner.getParameters('xgb')


param_grid={'n_estimators': [10, 100, 1000]}
# tuner.forceRun()
tuner.tuneParamGrid('xgb', param_grid, cv=5)


param_grid={'max_depth': list(range(3, n_features + 1, 3))}
# tuner.forceRun()
tuner.tuneParamGrid('xgb', param_grid, cv=5)


param_grid={'learning_rate': [0.001, 0.01, 0.1]}
# tuner.forceRun()
tuner.tuneParamGrid('xgb', param_grid, cv=5)


param_grid={'reg_alpha': [0.001, 0.5, 1, 5, 10, 50, 100, 500]}
# tuner.forceRun()
tuner.tuneParamGrid('xgb', param_grid, cv=5)


param_grid={'reg_lambda': [0.001, 0.5, 1, 5, 10, 50, 100, 500]}
# tuner.forceRun()
tuner.tuneParamGrid('xgb', param_grid, cv=5)


best_params['xgb'] = {
    # Defined by the competition
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    # Fine-tuned
    'n_estimators': 10,
    'max_depth': 3,
    # Default
    'base_score': None,
    'booster': None,
    'callbacks': None,
    'colsample_bylevel': None,
    'colsample_bynode': None,
    'colsample_bytree': None,
    'device': None,
    'early_stopping_rounds': None,
    'enable_categorical': False,
    'feature_types': None,
    'gamma': None,
    'grow_policy': None,
    'importance_type': None,
    'interaction_constraints': None,
    'learning_rate': None,
    'max_bin': None,
    'max_cat_threshold': None,
    'max_cat_to_onehot': None,
    'max_delta_step': None,
    'max_leaves': None,
    'min_child_weight': None,
    'missing': -1,
    'monotone_constraints': None,
    'multi_strategy': None,
    'n_jobs': None,
    'num_parallel_tree': None,
    'random_state': None,
    'reg_alpha': None,
    'reg_lambda': None,
    'sampling_method': None,
    'scale_pos_weight': None,
    'subsample': None,
    'tree_method': None,
    'validate_parameters': None,
    'verbosity': None
}


%%time

fineTuned_estimators = [
    ('random_forest', RandomForestRegressor(**best_params['random_forest'])),
    ('neural-net_mlp', MLPRegressor(**best_params['neural-net_mlp'])),
    ('xgb', XGBRegressor(**best_params['xgb'])),
]

fineTuned_pipelines = makeModellingPipelines(
    trainPreprocessor, 
    fineTuned_estimators, 
    X_train, 
    y_train
)



%%time
evaluatePipelines(pipelines, X_train, y_train, X_eval, y_eval, estimators=False, learning_curves=False)





for name, model in fineTuned_pipelines.items():
    csv_name = f"submission_{name}.csv"
    print('Submission file:', csv_name)
    y_pred = model.predict(testDs)
    y_pred_submission = targetPreprocessor.inverse_transform(pd.DataFrame(y_pred))
    output = pd.DataFrame({'id': testDs.index,
                           'Price': y_pred_submission.reshape(-1)})
    output.to_csv(csv_name, index=False)
    print(pd.read_csv(csv_name).head())
    print()





