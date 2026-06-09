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


def install_files():  
#install files:
    !pip install -q /kaggle/input/scikit-learn-download/scikit_learn-1.6.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    !pip install -q /kaggle/input/scikit-learn-download/xgboost-3.0.0-py3-none-manylinux_2_28_x86_64.whl

    #!pip install -q --upgrade polars==1.24.0

install_files()


#check keras backend:
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['POLARS_MAX_THREADS'] = '1'
os.environ['PYTHONHASHSEED'] = str(42)
import random

from keras import backend as K
print(K.backend())
import polars as pl
import tensorflow as tf
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.feature_selection import *
from sklearn.neighbors import *
from sklearn.pipeline import make_pipeline, FeatureUnion, make_union, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.ensemble import HistGradientBoostingClassifier
import gc
import keras
#for version:
import sklearn
from sklearn.base import *
from sklearn.utils.validation import check_is_fitted


from sklearn.ensemble import *
from sklearn.svm import SVC
from sklearn.model_selection import *
from sklearn.metrics import *

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import *



# Set all below outside the loop for fixed predictions and loading
GLOBAL_SEED = 1000
KERAS_SEED = 1000
keras.utils.set_random_seed(KERAS_SEED)
tf.random.set_seed(GLOBAL_SEED)
pl.set_random_seed(GLOBAL_SEED)
tf.config.experimental.enable_op_determinism()

#for loading features:
import joblib
from tqdm.notebook import tqdm
from tqdm.keras import TqdmCallback
PREFIX = "/kaggle/input/playground-series-s5e4/"

TRAIN_LINK = os.path.join(PREFIX, 'train.csv')
TEST_LINK = os.path.join(PREFIX, 'test.csv')
ORIG_LINK = "/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv"

print(f'sklearn version: {sklearn.__version__}')
print(f'polars version: {pl.__version__}')

#Regressors
import xgboost as xgb
print(f'xgboost verions: {xgb.__version__}')
import lightgbm as lgb
print(f'lightgbm verions: {lgb.__version__}')
#set random seed:
pl.set_random_seed(GLOBAL_SEED)

#concurrency:
import concurrent.futures

from multiprocessing import cpu_count
n_cores = cpu_count()
print(f'Number of Logical CPU cores: {n_cores}')


from itertools import combinations, permutations
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)
import optuna

# Suppress all warnings
warnings.filterwarnings("ignore")


orig = pl.read_csv(ORIG_LINK)
print(orig.shape)

orig.head()


train = pl.read_csv(TRAIN_LINK)
train = train.with_columns(
    pl.col('Episode_Length_minutes').clip(0,120),
)
test = pl.read_csv(TEST_LINK)
test = test.with_columns(
    pl.col('Episode_Length_minutes').clip(0,120),
)
print(train.shape)
train = train.select(pl.exclude('id'))
train.head(3)


train['Genre'].value_counts()


orig = pl.read_csv(ORIG_LINK)
orig['Genre'].value_counts()



def plot_nominal(feature, train = None):
    if train is None:
        train = pl.read_csv(TRAIN_LINK).sort(feature)
    else:
        train = train.sort(feature)
    orig = pl.read_csv(ORIG_LINK).sort(feature)
    fig, ax = plt.subplots(figsize = (12, 4), ncols = 2)
    sns.boxplot(data = train.to_pandas() , x = feature, y = 'Listening_Time_minutes', ax = ax[0])
    ax[0].set_title('Train Dataset')
    ax[0].tick_params(axis='x', labelrotation=90)
    sns.boxplot(data = orig.to_pandas() , x = feature, y = 'Listening_Time_minutes', ax = ax[1])
    ax[1].set_title('Source Dataset')
    ax[1].tick_params(axis='x', labelrotation=90)
    
    plt.show()
    plt.close()

def plot_numeric(feature, train = None):
    if train is None:
        train = pl.read_csv(TRAIN_LINK).sort(feature)
    else:
        train = train.sort(feature)
    orig = pl.read_csv(ORIG_LINK).sort(feature)
    fig, ax = plt.subplots(figsize = (12, 4), ncols = 2)
    sns.scatterplot(data = train.to_pandas() , x = feature, y = 'Listening_Time_minutes', ax = ax[0])
    ax[0].set_title('Train Dataset')
    ax[0].tick_params(axis='x', labelrotation=90)
    sns.scatterplot(data = orig.to_pandas() , x = feature, y = 'Listening_Time_minutes', ax = ax[1])
    ax[1].set_title('Source Dataset')
    ax[1].tick_params(axis='x', labelrotation=90)
    
    plt.show()
    plt.close()
    
plot_numeric(feature = 'Host_Popularity_percentage')



#Check how to impute the Episode_Length_minutes
train1 = train.filter(pl.all_horizontal(pl.all().is_not_null()))
'''
train2 = train.with_columns(
    Listening_Time_minutes = pl.when(pl.col('Episode_Length_minutes').is_not_null())\
    .then(pl.when(pl.col('Listening_Time_minutes').gt(pl.col('Episode_Length_minutes')))\
    .then(pl.col('Episode_Length_minutes')).otherwise(pl.col('Listening_Time_minutes')))\
    .otherwise(pl.col('Listening_Time_minutes'))
)
'''
train2 = train.with_columns(
    Episode_Length_minutes = pl.when(pl.col('Episode_Length_minutes').is_not_null())\
    .then(pl.when(pl.col('Listening_Time_minutes').gt(pl.col('Episode_Length_minutes')))\
    .then(None).otherwise(pl.col('Episode_Length_minutes')))\
    .otherwise(pl.col('Episode_Length_minutes'))
)


for_plot = train2.filter(pl.all_horizontal(pl.all().is_not_null()))
sns.scatterplot(data = for_plot.to_pandas(), x = 'Episode_Length_minutes', y = 'Listening_Time_minutes')



train.describe()


print(train['Podcast_Name'].value_counts().shape)
print(test['Podcast_Name'].value_counts().shape)


test


test.describe()


class CONFIG:
    NUMERIC_FEATURES = ['Episode_Length_minutes','Host_Popularity_percentage', 'Guest_Popularity_percentage']
    #Episode_Title is removed
    NOMINAL_FEATURES = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    N_FEATURES = len(NUMERIC_FEATURES) + len(NOMINAL_FEATURES)
    EXCLUDE = ['id', ]
    

#Feature Engineering:

class NominalEncoder_1(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OrdinalEncoder()
        self.encoder.set_output(transform = 'polars')
        
    def fit(self, X, y):
        
        self.string_cols = X.select(pl.col(pl.String)).columns
        #print(self.string_cols)
        self.encoder.fit(X.select(self.string_cols))
       
        self.fitted_ = True
        return self

    def transform(self, X):
        X= X.with_columns(
            pl.col('Episode_Length_minutes').clip(0,120),
        )
        
        X1 = self.encoder.transform(X.select(self.string_cols))
        X2 = X.select(pl.exclude(CONFIG.EXCLUDE)).select(pl.exclude(self.string_cols))
        X3 = pl.concat([X1, X2], how = 'horizontal')
        #change the Episode_Length_minutes. Clean the outliers:
        
        return X3




class NominalEncoder_2(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OrdinalEncoder()
        self.encoder.set_output(transform = 'polars')
        self.selected_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
        self.combo_string = list(combinations(self.selected_features, 2))
        
        
    def fit(self, X, y):
        for combs in self.combo_string:
            X = X.with_columns(
                pl.concat_list(pl.col(combs[0]), pl.col(combs[1])).list.join(separator = "X").alias(f'{combs[0]}_X_{combs[1]}')
            )
        self.string_cols = X.select(pl.col(pl.String)).columns
        #print(self.string_cols)
        self.encoder.fit(X.select(self.string_cols))
       
        self.fitted_ = True
        return self

    def transform(self, X):
        X= X.with_columns(
            pl.col('Episode_Length_minutes').clip(0,120),
        )
        for combs in self.combo_string:
            X = X.with_columns(
                pl.concat_list(pl.col(combs[0]), pl.col(combs[1])).list.join(separator = "X").alias(f'{combs[0]}_X_{combs[1]}')
            )
        
        X1 = self.encoder.transform(X.select(self.string_cols))
        X2 = X.select(pl.exclude(CONFIG.EXCLUDE)).select(pl.exclude(self.string_cols))
        X3 = pl.concat([X1, X2], how = 'horizontal')
        #change the Episode_Length_minutes. Clean the outliers:
        
        return X3


class NullEncoder(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X,y):
        self.fitted_ = True
        return self
    def transform(self, X):
        X = X.with_columns(
            pl.col('Episode_Length_minutes').fill_null(-1),
            pl.col('Guest_Popularity_percentage').fill_null(-1),
            
        )
        X = X.filter(pl.all_horizontal(pl.all().is_not_null()))
        return X
    



class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y):
        
        self.fitted_ = True
        return self

    def transform(self, X):
       
        #interactions of numeric features:
        self.combo = combinations(CONFIG.NUMERIC_FEATURES, 2)
        for combs in self.combo:
            X = X.with_columns(
                (pl.col(combs[0]).pow(2)).add(pl.col(combs[1])).alias(f'{combs[0]}_POW_ADD_{combs[1]}'),#12.7866
                pl.col(combs[0]).sub(pl.col(combs[1]).pow(2)).alias(f'{combs[0]}_SUB_POW_{combs[1]}'),#12.78153687690494
            )
        self.n_features = X.shape[-1]
        self.feature_names = X.columns
        CONFIG.MAX = X.max()
        CONFIG.FEATURES = X.columns
        return X

class FeatureEngineering_2(BaseEstimator, TransformerMixin):
    def __init__(self, random_state = 0):
        self.random_state = random_state
        self.te = TargetEncoder(random_state = self.random_state ).set_output(transform = 'polars')
        
    def fit(self, X, y):
        self.columns = X.columns
        self.combo = combinations(self.columns, 2)
        for f in self.combo:
            X = X.with_columns(
                pl.col(f[0]).cast(pl.String).add(pl.col(f[1]).cast(pl.String)).alias(f'{f[0]}X{f[1]}')
            )

        self.te.fit(X,y)
        self.fitted_ = True
        return self

    def transform(self, X):
        self.c = combinations(self.columns, 2)
        for f in self.c:
            X = X.with_columns(
                pl.col(f[0]).cast(pl.String).add(pl.col(f[1]).cast(pl.String)).alias(f'{f[0]}X{f[1]}')
            )
            
        X = self.te.transform(X)

       
        #interactions of numeric features:
        self.combo = combinations(CONFIG.NUMERIC_FEATURES, 2)
        for combs in self.combo:
            X = X.with_columns(
                (pl.col(combs[0]).pow(2)).add(pl.col(combs[1])).alias(f'{combs[0]}_POW_ADD_{combs[1]}'),#12.7866
                pl.col(combs[0]).sub(pl.col(combs[1]).pow(2)).alias(f'{combs[0]}_SUB_POW_{combs[1]}'),#12.78153687690494
            )
        self.n_features = X.shape[-1]
        self.feature_names = X.columns
        CONFIG.MAX = X.max()
        CONFIG.FEATURES = X.columns
        return X





def generate_XYTEST():
    
    '''
    X, Y and TEST data generator and cleaner.
    import polars as pl
    import numpy as np
    returns: X (train features) , Y(train label) and test dataset
    '''
    
    #Load the dataset into memory
    train = pl.read_csv(TRAIN_LINK)
    test = pl.read_csv(TEST_LINK)
    orig = pl.read_csv(ORIG_LINK).with_columns(
        pl.col('Number_of_Ads').cast(pl.Float64)
    ).filter(pl.all_horizontal(pl.all().is_not_null()))

    #Combine train data with the original dataset:
    train_main = pl.concat([train.select(pl.exclude('id')), orig ], how = 'vertical')
    
    #Remove duplicates:
    train_main = train_main.unique()
    
    #Clip the Target Variable if needed:
    
    train_main = train_main.with_columns(
          Listening_Time_minutes = pl.when(pl.col('Episode_Length_minutes').is_not_null())\
        .then(pl.when(pl.col('Listening_Time_minutes').gt(pl.col('Episode_Length_minutes')))\
              .then(pl.col('Episode_Length_minutes')).otherwise(pl.col('Listening_Time_minutes')))\
        .otherwise(pl.col('Listening_Time_minutes'))
    )

           
    X = train_main.select(pl.exclude('Listening_Time_minutes'))
    Y = train_main.select('Listening_Time_minutes').to_numpy().ravel()
    
    return X, Y, test


    

scaler = StandardScaler()
scaler.set_output(transform = 'polars')

'''Best trial: Value: 12.49795294154597'''
xgb_params_9 = {'eta': 0.03865131219698339,'max_depth': 10, 'min_child_weight': 5, 'subsample': 0.9461240630218295, 
                'colsample_bytree': 0.6519596123727526, 'gamma': 0.5747908732518904, 'lambda': 0.6272319520767028,
                'alpha': 0.8911309969286797, 'n_estimators': 1874, 'booster': 'gbtree', 'max_bin': 359, 
                'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 167, 
                'tree_method': 'hist', 'device': 'cuda'}

#z.fit(X,Y)
trx = make_pipeline(NullEncoder(), NominalEncoder_1(), 
                    FeatureEngineering_2(), StandardScaler().set_output(transform = 'polars'))
print('Fitting')
X,Y, TEST = generate_XYTEST()
'''
trx.fit(X,Y)
train_shape = pl.read_csv(TRAIN_LINK).shape
test_shape = pl.read_csv(TEST_LINK).shape
print(f'Orig train n_observations: {train_shape[0]}')
print(f'New train n_observations: : {X.shape[0]}')

print(f'Orig test n_observations: {test_shape[0]}')
print(f'New test n_observations: {test.shape[0]}')
'''
"""
Observation:
Removing or modifying the outliers will reduce the LB score
Conclusion:
The test dataset has many outlier targets. 
Recommendation:
Concentrate on Feature Engineering

"""
print('complete')





#A = trx.transform(X)
#A.head(4)


#B = trx.transform(TEST)
#B.head(4)



'''
plot_nominal('Number_of_Ads')
plot_nominal('Number_of_Ads', train = pl.concat([X, pl.DataFrame({'Listening_Time_minutes': Y.ravel()})], how = 'horizontal'))
plot_numeric('Episode_Length_minutes')
plot_numeric('Episode_Length_minutes', train = pl.concat([X, pl.DataFrame({'Listening_Time_minutes': Y.ravel()})], how = 'horizontal'))
plot_numeric('Host_Popularity_percentage')
plot_numeric('Host_Popularity_percentage', train = pl.concat([X, pl.DataFrame({'Listening_Time_minutes': Y.ravel()})], how = 'horizontal'))
plot_numeric('Guest_Popularity_percentage')
plot_numeric('Guest_Popularity_percentage', train = pl.concat([X, pl.DataFrame({'Listening_Time_minutes': Y.ravel()})], how = 'horizontal'))
'''
print()


'''
dat = trx.named_steps['xgbregressor'].feature_importances_
features = list(trx.named_steps['standardscaler'].get_feature_names_out())

result = pl.DataFrame({'FeatureName': features, 'FeatureImportance': dat}).sort('FeatureImportance', descending = True)
CONFIG.SELECTED_FEATURES = list(result[:20,0].to_numpy())
print(CONFIG.SELECTED_FEATURES)
'''
print('complete')


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y):
        self.fited_ = True
        return self
    def transform(self, X):
        return X.select(self.selected_features)

class XGBModel(RegressorMixin, BaseEstimator):
    '''
    Model wrapper for XGBRegressor
    '''
    def __init__(self, params, sample_fraction = 0.8, sample_seed = 0):
        self.params = params
        self.sample_fraction = sample_fraction
        self.sample_seed = sample_seed
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X,y):
        #fit only on non_null data:
        Y = pl.DataFrame({'Listening_Time_minutes': y})
        X1 = pl.concat([X,Y], how = 'horizontal')
        #sample the data frame::
        X1 = X1.sample(fraction = self.sample_fraction, seed = self.sample_seed )
        X = X1.select(pl.exclude('Listening_Time_minutes'))
        Y = X1.select('Listening_Time_minutes').to_numpy().ravel()
        self.model.fit(X,Y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)





class AutoEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features = 22, seed = 0, batch_size = 32, name = 'autoencoder', test_seed = 60,
                test_size = 0.20, epochs = 10):
        self.n_features = n_features
        self.seed = seed
        self.batch_size = batch_size
        self.name = name
        self.test_seed = test_seed
        self.test_size = test_size
        self.epochs = epochs
        
        #training parameteris
        self.loss = keras.losses.MeanSquaredError(name = 'MSE')
        self.metric = keras.metrics.RootMeanSquaredError(name = 'RMSE')
        self.optimizer = keras.optimizers.Lion(learning_rate = 1e-4)
        #callbacks
        self.filename = f'{self.name}.keras'
        self.saver = keras.callbacks.ModelCheckpoint(filepath = self.filename, monitor = 'val_RMSE', 
                                                      save_best_only = True, mode = 'min')
        self.stopper = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min')
        self.scheduler = keras.callbacks.LearningRateScheduler(self.scheduler)

        # create the model
        self.model = self.generate_autoencoder(n_features = self.n_features, 
                                              seed = self.seed, batch_size = self.batch_size, 
                                              name = self.name)

    def fit(self, X, y):
        #callbacks
        self.monitor = TqdmCallback(verbose = 0) #to avoid initial display
        
        #train now:
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = self.test_seed, test_size = self.test_size)
        
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = [self.metric])
            
        self.model.fit(x = xtrain,y = ytrain, validation_data = [xtest, ytest], shuffle = True, epochs = self.epochs,
                      callbacks = [self.monitor, self.stopper, self.saver, self.scheduler], verbose = 1,
                      validation_batch_size = 1024)
        #define below for ClassifierMixin
        self.fitted_ = True #indicate that model is fitted.
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        
        self.model = keras.saving.load_model(self.filename)
        pred = self.model.predict(X, batch_size = 1024)
        
        return pred
        

    def generate_autoencoder(self, n_features = 22, batch_size = 32, seed = 0, name = 'autoencoder'):
        keras.utils.clear_session(free_memory=True)
        keras.utils.set_random_seed(seed)
    
        ins = keras.layers.Input((n_features,), batch_size = batch_size)
        encoder = keras.layers.Dropout(0.7)(ins)
        encoder = keras.layers.Dense(64, activation = 'selu')(encoder)
        encoder = keras.layers.Dense(32, activation = 'selu')(encoder)
        #bottleneck
        bottleneck = keras.layers.Dense(22)(encoder)
        decoder = keras.layers.Dense(32, activation = 'selu')(bottleneck)
        decoder = keras.layers.Dense(64, activation = 'selu')(decoder)
        outs = keras.layers.Dense(n_features)(decoder)
    
        model = keras.Model(inputs = ins, outputs = outs, name = name)
    
        model.compile(optimizer =self.optimizer,loss = self.loss, metrics = [self.metric])
        return model

    def scheduler(self, epoch, lr):
        output = lr
        if epoch == 0:
            output = 1e-4
        if epoch == 1:
            output = 1e-5
        if epoch == 3:
            output = 1e-6
        return output


#autoencoder = AutoEncoder(n_features = 22)
#autoencoder.summary()
'''
train = pl.read_csv(TRAIN_LINK)
test = pl.read_csv(TEST_LINK)
X = train.select(pl.exclude('Listening_Time_minutes'))
Y = train.select('Listening_Time_minutes').to_numpy().ravel()

scaler = StandardScaler()
scaler.set_output(transform = 'polars')
trx = make_pipeline(BaseTransformer(), FeatureEngineering(), scaler)
X1 = trx.fit_transform(X,Y)

AE = AutoEncoder(n_features = trx.named_steps['featureengineering'].n_features, epochs = 20)

#transform and fit:
X2 = X1.filter(pl.all_horizontal(pl.all().is_not_nan()))
print(X2.shape)
#fit now:
AE.fit(X2,X2)
'''
print('complete')
    


'''
Keras as Scikit-learn estimator
'''

class Estimator(RegressorMixin, BaseEstimator):
    def __init__(self,  batch_size = 32, model_seed = 0, name = 'baseline', 
                 epochs = 100, embedding_output = 2, test_seed = 0, test_size = 0.10, 
                verbose = 0):
        '''
        output_dim = Embedding output
        '''
        #continue:
        self.batch_size = batch_size
        self.model_seed = model_seed
        self.test_seed = test_seed
        self.test_size = test_size
        self.name = name
        self.epochs = epochs
        self.embedding_output = embedding_output
        #training parameteris
        self.loss = keras.losses.MeanSquaredError(name = 'MSE')
        self.metric = keras.metrics.RootMeanSquaredError(name = 'RMSE')
        self.optimizer = keras.optimizers.Lion(learning_rate = 1e-4)
        #callbacks
        self.filename = f'{self.name}.keras'
        self.saver = keras.callbacks.ModelCheckpoint(filepath = self.filename, monitor = 'val_RMSE', 
                                                      save_best_only = True, mode = 'min')
        self.stopper = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min')
        self.scheduler = keras.callbacks.LearningRateScheduler(self.scheduler)


        #get the size of X:
        self.model = self.generate_model()
        #other params:
        self.verbose = verbose

    def fit(self, X, y):  
        #callbacks
        self.monitor = TqdmCallback(verbose = 0) #to avoid initial display
        
        #train now:
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = self.test_seed, test_size = self.test_size)
        xtrain = xtrain.to_dict()
        xtest = xtest.to_dict()
        
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = [self.metric])
            
        self.model.fit(x = xtrain,y = ytrain, validation_data = [xtest, ytest], shuffle = True, epochs = self.epochs,
                      callbacks = [self.monitor, self.stopper, self.saver, self.scheduler], verbose = self.verbose,
                      validation_batch_size = 1024)
        #define below for ClassifierMixin
        self.fitted_ = True #indicate that model is fitted.
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        
        self.model = keras.saving.load_model(self.filename)
        pred = self.model.predict(X.to_dict(), batch_size = 1024)
        
        return pred

    def scheduler(self, epoch, lr):
        output = lr
        if epoch == 0:
            output = 1e-5
        if epoch == 8:
            output = 1e-6
        return output


    def generate_model(self):
        keras.utils.clear_session(free_memory=True)
        keras.utils.set_random_seed(self.model_seed)
    
        inx = dict()
        embs = []
        nums = []
        
        for f in CONFIG.FEATURES:
            inx[f] = keras.Input((1,), name = f, batch_size = self.batch_size)
            if f in CONFIG.NOMINAL_FEATURES:
                emb = keras.layers.Embedding(input_dim = int(CONFIG.MAX[f][0]) + 1, output_dim = self.embedding_output)(inx[f])
                embs.append(emb)
            else:
                nums.append(inx[f])
        
        x1 = keras.layers.Concatenate()(embs)
        x1 = keras.ops.stack(embs, -1)
        x1 = keras.layers.Dense(16, 'selu')(x1)
        x1 = keras.layers.Flatten()(x1)
        
        x2 = keras.layers.Concatenate()(nums)
        xa = keras.layers.Concatenate()([x1,x2])
       
        xa = keras.layers.Dense(128, activation = 'selu')(xa)
        xa = keras.layers.Dense(64, activation = 'selu')(xa)

        x = keras.layers.Dense(1, activation = 'linear', name = 'regressor', bias_initializer = keras.initializers.Constant(45.0))(xa)
    
        model = keras.Model(inputs = inx, outputs = x, name = self.name)
        
        return model


print('complete')




'''
Best trial: Value: 12.67929177876347,
model = make_pipeline(BaseTransformer(), FeatureEngineering(), StandardScaler(), xgb.XGBRegressor(**params))
'''

xgb_params_6 = {'eta': 0.06222329940196073, 'max_depth': 10, 'min_child_weight': 1, 'subsample': 0.9738539843188956, 
                'colsample_bytree': 0.5310831070553464, 'gamma': 0.7016867080665651, 'lambda': 0.5295882653475782, 
                'alpha': 0.16986065725958813, 'n_estimators': 984, 'booster': 'gbtree', 'max_bin': 279, 
                'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 138, 'tree_method': 'hist', 
                'device': 'cuda'}



'''Best trial: Value: 12.605214924977219'''
#with additional non-null data from original source.
xgb_params_7 = {'eta': 0.03894063421630174, 'max_depth': 10, 'min_child_weight': 1, 'subsample': 0.9578967363645963,
                'colsample_bytree': 0.6519416220473021, 'gamma': 0.5070611360422013, 'lambda': 0.1873349187562977,
                'alpha': 0.8345578703312051, 'n_estimators': 1517, 'booster': 'gbtree', 'max_bin': 402, 
                'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 126, 'tree_method': 'hist',
                'device': 'cuda'}



'''Best trial: Value: 12.568321545858891'''
xgb_params_8 = {'eta': 0.056358229623348036, 'max_depth': 10, 'min_child_weight': 2, 'subsample': 0.967961154493475, 
                'colsample_bytree': 0.5873756489625827, 'gamma': 0.3228165498865387, 'lambda': 0.8951408387626582, 
                'alpha': 0.40378587758120965, 'n_estimators': 1829, 'booster': 'gbtree', 'max_bin': 431,
                'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 349, 'tree_method': 'hist',
                'device': 'cuda'}


'''Best trial: Value: 12.49795294154597'''
xgb_params_9 = {'eta': 0.03865131219698339,'max_depth': 10, 'min_child_weight': 5, 'subsample': 0.9461240630218295, 
                'colsample_bytree': 0.6519596123727526, 'gamma': 0.5747908732518904, 'lambda': 0.6272319520767028,
                'alpha': 0.8911309969286797, 'n_estimators': 1874, 'booster': 'gbtree', 'max_bin': 359, 
                'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 167, 
                'tree_method': 'hist', 'device': 'cuda'}




'''No nans config'''
'''Best trial: Value: 10.106144490800926'''
xgb_params_non_null_1 = {'eta': 0.08351878963142216, 'max_depth': 10, 'min_child_weight': 1, 'subsample': 0.8087027523970118,
                      'colsample_bytree': 0.8300138124594958, 'gamma': 0.04477521209277783, 'lambda': 0.8919509137550635, 
                      'alpha': 0.17504511360295322, 'n_estimators': 477, 'booster': 'dart', 'max_bin': 365, 
                      'objective': 'reg:squarederror',  'eval_metric': 'rmse', 'random_state': 72, 'tree_method': 'hist', 
                      'device': 'cpu'}


'''Best trial: Value: 10.111100389723067'''
#model:

xgb_params_non_null_2 = {'eta': 0.04309058132883985, 'max_depth': 10, 'min_child_weight': 2, 'subsample': 0.8413901261187522,
                         'colsample_bytree': 0.8883864807231816, 'gamma': 0.028232821362615054, 'lambda': 0.028359972581466573,
                         'alpha': 0.18437226352199523, 'n_estimators': 998, 'booster': 'gbtree', 'max_bin': 297, 
                         'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 396, 
                         'tree_method': 'hist', 'device': 'cuda'}

'''Best trial:
  Value: 10.107773417974316
  model = make_pipeline(BaseTransformer(), FeatureEngineering(), StandardScaler(), xgb.XGBRegressor(**params))

'''
xgb_params_non_null_3 = {'eta': 0.050043129307250266, 'max_depth': 10, 'min_child_weight': 3, 'subsample': 0.6657764113870365,
                         'colsample_bytree': 0.6331040203711943, 'gamma': 0.19980529698966576, 'lambda': 0.4887939063876674, 
                         'alpha': 0.9734458302181486, 'n_estimators': 979, 'booster': 'gbtree', 'max_bin': 465,
                         'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 138, 'tree_method': 'hist',
                         'device': 'cuda'}

'''Best trial: Value: 10.077810519677218'''
xgb_params_non_null_4 = {'eta': 0.07683968591599709, 'max_depth': 10, 'min_child_weight': 3, 'subsample': 0.930589137290428,
                         'colsample_bytree': 0.8164441890813419, 'gamma': 0.17213098398502014, 'lambda': 0.8242883806193232,
                         'alpha': 0.890390711936256, 'n_estimators': 1000, 'booster': 'gbtree', 'max_bin': 342, 
                         'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 763, 'tree_method': 'hist', 
                         'device': 'cpu'}

'''WIth Original data no nulls:
Best trial: Value: 10.042274279130078 
  '''
xgb_params_non_null_5 = {'eta': 0.052344214836098066, 'max_depth': 10, 'min_child_weight': 1, 'subsample': 0.9069191511807905,
                         'colsample_bytree': 0.8326760399840132, 'gamma': 0.9209175697460817, 'lambda': 0.33387295171363196,
                         'alpha': 0.6159122479807136, 'n_estimators': 1900, 'booster': 'gbtree', 'max_bin': 431, 
                         'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 349, 'tree_method': 'hist', 
                         'device': 'cuda'}


'''Best trial: Value: 10.054874722413603'''
xgb_params_non_null_6 = {'eta': 0.058502564428419945, 'max_depth': 10, 'min_child_weight': 2, 'subsample': 0.8673733652502045, 
                         'colsample_bytree': 0.9091348914359767, 'gamma': 0.09233776533067081, 'lambda': 0.9766794355172744,
                         'alpha': 0.44001752316042425, 'n_estimators': 1998, 'booster': 'gbtree', 'max_bin': 395, 
                         'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 721, 'tree_method': 'hist',
                         'device': 'cuda'}

print('complete')


#Optuna settings
def objective_xgb(trial, random_state = 8):
    """Objective function for Optuna."""

    # Hyperparameter search space
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': trial.suggest_float('eta', 0.03, 0.20),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.4, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'lambda': trial.suggest_float('lambda', 0, 1.0),
        'alpha': trial.suggest_float('alpha', 0, 1.0),
        'random_state': random_state,
        'n_estimators' : trial.suggest_int('n_estimators', 100, 2000),
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        #'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist'])
        'tree_method': 'hist',
        'max_bin': trial.suggest_int('max_bin', 200, 500),
        'device':'cuda'
    }

    X,Y = generate_XY() #with orig not nulls
    #X1,Y1 = generate_not_null_with_orig_XY()
    X_train, X_val, y_train, y_val = train_test_split(X,Y, test_size = 0.2, random_state = random_state)


    model = make_pipeline(NominalEncoder(), FeatureEngineering(), StandardScaler().set_output(transform = 'polars'),
                         xgb.XGBRegressor(**params))
    

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    return rmse

def objective_lgbm(trial, train, random_state = 8):
    """Objective function for Optuna."""

    # Hyperparameter search space
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.8),
        'max_depth': trial.suggest_int('max_depth', 3, 14),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 2),
        'random_state': random_state,
        'device': 'gpu',
        'verbose': -1
    }
    params['num_leaves'] = 2**params['max_depth']
    
    X,Y = generate_XY()
    X_train, X_val, y_train, y_val = train_test_split(X,Y, test_size = 0.2, random_state = random_state)
    model = make_pipeline(BaseTransformer(), lgb.LGBMRegressor(**params))
    

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    return rmse

# Set logging level to suppress trial-by-trial output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def obtain_best_params(random_state = 8, n_trials = 30, model = 'xgb'):

    # Optuna study
    study = optuna.create_study(direction='minimize')  # Maximize RMSE
    if model == 'xgb':
        objective_with_params = lambda trial: objective_xgb(trial, random_state = random_state)
    if model == 'lgbm':
        objective_with_params = lambda trial: objective_lgbm(trial, random_state = random_state)
        
    study.optimize(objective_with_params, n_trials=n_trials, show_progress_bar=True)  # Number of trials
    
    # Best parameters and AUC
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    #print("  Params: ")
    #for key, value in trial.params.items():
    #    print(f"    {key}: {value}")
    
    # Retrain with best parameters
    if model == 'xgb':
        
        best_params = study.best_trial.params
        best_params['objective'] = 'reg:squarederror'
        best_params['eval_metric'] = 'rmse'
        best_params['random_state'] = random_state
        best_params['tree_method'] = 'hist'
        best_params['device'] = 'cuda'
    if model == 'lgbm':
        best_params = study.best_trial.params
        best_params['objective'] = 'regression'
        best_params['metric'] = 'rmse'
        best_params['random_state'] = random_state
        best_params['boosting_type'] = 'gdbt'
        best_params['device'] = 'gpu'
        best_params['verbose'] = -1
        best_params['num_leaves'] = 2 ** best_params['max_depth']
        

    return best_params
#new_params = obtain_best_params(random_state = 167, n_trials = 120, model = 'xgb')
print('complete')
#new_params


'''Cross Validation'''



X,Y, TEST = generate_XYTEST()
#X1, Y1 = generate_not_null_with_orig_XY()
#X1, Y1 = generate_not_null_XY()
#print(X1.columns)
'''
estimators_1 = [('0', XGBModel(xgb_params_non_null_4)), ('1', XGBModel(xgb_params_non_null_5)),
             ('2', XGBModel(xgb_params_non_null_6))]
estimators_2 = [('a', XGBModel(xgb_params_non_null_4)), ('b', XGBModel(xgb_params_non_null_5)),
             ('c', XGBModel(xgb_params_non_null_6))]
voter_1 = VotingRegressor(estimators = estimators_1)
voter_2 = VotingRegressor(estimators = estimators_2)

model_1 = make_pipeline(NominalEncoder_1(), FeatureEngineering(), StandardScaler().set_output(transform = 'polars'), voter_1)
model_2 = make_pipeline(NominalEncoder_2(), FeatureEngineering(), StandardScaler().set_output(transform = 'polars'), voter_2)

main_model =  VotingRegressor(estimators = [('m0',model_1),('m2', model_2)])
'''

estimators_1 = [('0', XGBModel(params = xgb_params_6, sample_fraction = 1, sample_seed = 0)),
              ('1', XGBModel(params = xgb_params_7, sample_fraction = 1, sample_seed = 1)),
               ('2', XGBModel(params = xgb_params_8,sample_fraction = 1, sample_seed = 2))]

estimators_2 = [('2', XGBModel(params = xgb_params_8,sample_fraction = 1, sample_seed = 3)),
              ('3', XGBModel(params = xgb_params_9,sample_fraction = 1, sample_seed = 4))]

estimators_3 = [('0', XGBModel(xgb_params_non_null_4,sample_fraction = 1, sample_seed = 5)), 
                ('1', XGBModel(xgb_params_non_null_5,sample_fraction = 1, sample_seed = 6)),
             ('2', XGBModel(xgb_params_non_null_6,sample_fraction = 1, sample_seed = 7))]


model_1 = make_pipeline(NominalEncoder_1(),FeatureEngineering(), StandardScaler().set_output(transform ='polars'), 
                      VotingRegressor(estimators = estimators_1))

model_2 = make_pipeline(NominalEncoder_2(),FeatureEngineering(), StandardScaler().set_output(transform ='polars'), 
                      VotingRegressor(estimators = estimators_2))

model_3 = make_pipeline(NominalEncoder_2(),FeatureEngineering(), StandardScaler().set_output(transform ='polars'), 
                      VotingRegressor(estimators = estimators_3))

main_model = VotingRegressor(estimators = [('m0',model_1),('m2', model_2), ('m3', model_3)])


n_splits  = 5
skf = KFold(n_splits = n_splits, shuffle = True, random_state = 60)

#cv_scores = cross_val_score(estimator = main_model, X=X, y = Y, cv = skf,  scoring='neg_root_mean_squared_error', verbose = 3)
#mu = np.mean(cv_scores)
#sd = np.std(cv_scores, ddof = 1)
#print(f'Cv mean = {mu}, sd = {sd}')

'''

estimators = [('1', xgb.XGBRegressor(**xgb_params_4)), ('2', xgb.XGBRegressor(**xgb_params_5)),
              ('3', xgb.XGBRegressor(**xgb_params_6))]
voter = VotingRegressor(estimators = estimators)
model = make_pipeline(BaseTransformer(),FeatureEngineering(), StandardScaler(), voter)

[CV] END .............................. score: (test=-12.640) total time= 2.8min
[CV] END .............................. score: (test=-12.669) total time= 2.7min
[CV] END .............................. score: (test=-12.743) total time= 2.7min
[CV] END .............................. score: (test=-12.677) total time= 2.7min
[CV] END .............................. score: (test=-12.711) total time= 2.7min
Cv mean = -12.687944221078663, sd = 0.039504599130373956

#with orig dataset:
[CV] END .............................. score: (test=-12.518) total time= 3.0min
[CV] END .............................. score: (test=-12.582) total time= 2.9min
[CV] END .............................. score: (test=-12.656) total time= 2.9min
[CV] END .............................. score: (test=-12.625) total time= 2.8min
[CV] END .............................. score: (test=-12.622) total time= 2.8min
Cv mean = -12.600555273827082, sd = 0.05290810475014326

X,Y = generate_XY()
estimators = [('1', xgb.XGBRegressor(**xgb_params_6)), ('2', xgb.XGBRegressor(**xgb_params_7)),
              ('3', xgb.XGBRegressor(**xgb_params_8))]
voter = VotingRegressor(estimators = estimators)
model = make_pipeline(BaseTransformer(),FeatureEngineering(), StandardScaler().set_output(transform ='polars'), voter)

[CV] END .............................. score: (test=-12.563) total time= 1.3min
[CV] END .............................. score: (test=-12.486) total time= 1.2min
[CV] END .............................. score: (test=-12.545) total time= 1.3min
[CV] END .............................. score: (test=-12.547) total time= 1.2min
[CV] END .............................. score: (test=-12.540) total time= 1.3min
Cv mean = -12.536285968963591, sd = 0.029433136608399843


Altered Y:
[CV] END .............................. score: (test=-12.533) total time= 1.3min
[CV] END .............................. score: (test=-12.534) total time= 1.2min
[CV] END .............................. score: (test=-12.528) total time= 1.2min
[CV] END .............................. score: (test=-12.527) total time= 1.2min
[CV] END .............................. score: (test=-12.483) total time= 1.3min
Cv mean = -12.521034710509014, sd = 0.021369551332467675

Nulled Episode_Length_minutes if it is greater than Listening_Time_minutes

estimators = [('2', xgb.XGBRegressor(**xgb_params_8)),
              ('3', xgb.XGBRegressor(**xgb_params_9))]
voter = VotingRegressor(estimators = estimators)
model = make_pipeline(NominalEncoder(),FeatureEngineering(), StandardScaler().set_output(transform ='polars'), voter)


#interaction of nominal data
[CV] END .............................. score: (test=-12.481) total time= 1.5min
[CV] END .............................. score: (test=-12.519) total time= 1.5min
[CV] END .............................. score: (test=-12.500) total time= 1.5min
[CV] END .............................. score: (test=-12.563) total time= 1.5min
[CV] END .............................. score: (test=-12.556) total time= 1.5min
Cv mean = -12.523958156940743, sd = 0.035464868027957494


*************************


#non null:


estimators = [('1', xgb.XGBRegressor(**xgb_params_non_null_4)),
             ('2', xgb.XGBRegressor(**xgb_params_non_null_5))]

[CV] END .............................. score: (test=-10.019) total time= 1.1min
[CV] END ............................... score: (test=-9.940) total time= 1.1min
[CV] END ............................... score: (test=-9.987) total time= 1.1min
[CV] END .............................. score: (test=-10.012) total time= 1.1min
[CV] END .............................. score: (test=-10.007) total time= 1.1min
Cv mean = -9.992955168745283, sd = 0.03192381157136366

X1, Y1 = generate_not_null_XY()
estimators = [('0', xgb.XGBRegressor(**xgb_params_non_null_3)), ('1', xgb.XGBRegressor(**xgb_params_non_null_4)),
             ('2', xgb.XGBRegressor(**xgb_params_non_null_5))]
voter = VotingRegressor(estimators = estimators)
model = make_pipeline(BaseTransformer(),FeatureEngineering(), StandardScaler(), voter)

[CV] END .............................. score: (test=-10.017) total time= 1.4min
[CV] END ............................... score: (test=-9.939) total time= 1.4min
[CV] END ............................... score: (test=-9.984) total time= 1.4min
[CV] END .............................. score: (test=-10.013) total time= 1.4min
[CV] END .............................. score: (test=-10.005) total time= 1.4min
Cv mean = -9.991580135165172, sd = 0.031723208718530585

X1, Y1 = generate_not_null_XY()

estimators = [('0', xgb.XGBRegressor(**xgb_params_non_null_4)), ('1', xgb.XGBRegressor(**xgb_params_non_null_5)),
             ('2', xgb.XGBRegressor(**xgb_params_non_null_6))]
voter = VotingRegressor(estimators = estimators)
model = make_pipeline(BaseTransformer(),FeatureEngineering(), StandardScaler(), voter)
LB = 12.57929
[CV] END ............................... score: (test=-9.989) total time= 1.6min
[CV] END ............................... score: (test=-9.911) total time= 1.6min
[CV] END ............................... score: (test=-9.954) total time= 1.6min
[CV] END ............................... score: (test=-9.979) total time= 1.6min
[CV] END ............................... score: (test=-9.978) total time= 1.6min
Cv mean = -9.962153329596884, sd = 0.03155029580458319


#Modified Y:
LB = 12.57557
[CV] END ............................... score: (test=-9.956) total time= 4.9min
[CV] END ............................... score: (test=-9.890) total time= 4.9min
[CV] END ............................... score: (test=-9.935) total time= 4.9min
[CV] END ............................... score: (test=-9.972) total time= 4.9min
[CV] END ............................... score: (test=-9.961) total time= 4.8min
Cv mean = -9.942754384568158, sd = 0.03251550165805112


'''


'''
model_1 :
Cv mean = -12.523035380149866, sd = 0.048872371648263295
Cv mean = -12.526519483497617, sd = 0.03575545464697479

model_2:
Cv mean = -12.522456591884495, sd = 0.03300660988172083
Cv mean = -12.529699777394836, sd = 0.03472308535520251

Combined model_1 and model_2:
Cv mean = -12.498561020653677, sd = 0.057882039530635046


#Using XGBModel and xgb_params on with null values:
Cv mean = -12.498558834311206, sd = 0.03664396039845211
#Using XGBModel and xgb_params_non_null . Combined model_1, model_2, model_3
LB = 12.54217
[CV] END .............................. score: (test=-12.437) total time= 5.3min
[CV] END .............................. score: (test=-12.501) total time= 5.3min
[CV] END .............................. score: (test=-12.523) total time= 5.3min
[CV] END .............................. score: (test=-12.477) total time= 5.3min
[CV] END .............................. score: (test=-12.474) total time= 5.3min
Cv mean = -12.482150738125318, sd = 0.03245447212775933


#outliers removed:
LB = 20.30693
[CV] END ............................... score: (test=-9.927) total time= 3.8min
[CV] END ............................... score: (test=-9.899) total time= 3.8min
[CV] END ............................... score: (test=-9.864) total time= 3.8min
[CV] END ............................... score: (test=-9.917) total time= 3.8min
[CV] END ............................... score: (test=-9.899) total time= 3.8min
Cv mean = -9.901302231158784, sd = 0.02416488972956407
'''
print('complete')


estimators_1 = [('0', XGBModel(params = xgb_params_6, sample_fraction = 1, sample_seed = 0)),
              ('1', XGBModel(params = xgb_params_7, sample_fraction = 1, sample_seed = 1)),
               ('2', XGBModel(params = xgb_params_8,sample_fraction = 1, sample_seed = 2))]

estimators_2 = [('2', XGBModel(params = xgb_params_8,sample_fraction = 1, sample_seed = 3)),
              ('3', XGBModel(params = xgb_params_9,sample_fraction = 1, sample_seed = 4))]

estimators_3 = [('0', XGBModel(xgb_params_non_null_4,sample_fraction = 1, sample_seed = 5)), 
                ('1', XGBModel(xgb_params_non_null_5,sample_fraction = 1, sample_seed = 6)),
             ('2', XGBModel(xgb_params_non_null_6,sample_fraction = 1, sample_seed = 7))]


model_1 = make_pipeline(NominalEncoder_1(),FeatureEngineering(), StandardScaler().set_output(transform ='polars'), 
                      VotingRegressor(estimators = estimators_1))

model_2 = make_pipeline(NominalEncoder_2(),FeatureEngineering(), StandardScaler().set_output(transform ='polars'), 
                      VotingRegressor(estimators = estimators_2))

model_3 = make_pipeline(NominalEncoder_2(),FeatureEngineering(), StandardScaler().set_output(transform ='polars'), 
                      VotingRegressor(estimators = estimators_3))

main_model = VotingRegressor(estimators = [('m0',model_1),('m2', model_2), ('m3', model_3)])




'''Generate the data: '''
print('Generate Data')
X,Y, test = generate_XYTEST() #with orig not null data

print('Fit the model')

main_model.fit(X,Y)

print('Predicting the test dataset')
pred = main_model.predict(test).ravel()

print('Writing the submission')
#val_RMSE: 13.2122, epoch 6
sub = pl.DataFrame({'id': test['id'], 'Listening_Time_minutes': pred})
sub.write_csv('submission.csv')
print('All steps completed!')


sub.head()


from IPython.display import FileLink
FileLink('submission.csv')

