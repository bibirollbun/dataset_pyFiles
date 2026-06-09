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


!pip install --upgrade scikit-learn > /dev/null


import polars as pl
from sklearn.ensemble import *
from sklearn.pipeline import * 
from sklearn.compose import ColumnTransformer
from sklearn.metrics import * 
from sklearn.preprocessing import *
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
import joblib
import matplotlib.pyplot as plt


base_link = "/kaggle/input/playground-series-s5e9"
get_train = lambda: pl.read_csv(os.path.join(base_link, 'train.csv'))
get_test = lambda: pl.read_csv(os.path.join(base_link, 'test.csv'))
get_sample_submission = lambda : pl.read_csv(os.path.join(base_link, 'sample_submission.csv'))

def get_original():
    train_orig = pl.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")
    return train_orig


train = get_train()
print(train.shape)
train.head()


train.describe()


get_original().describe()


get_test().describe()


class Singleton(type):
    instance_container = []
    def __call__(self, *args, **kwargs):
        if self not in self.instance_container:
            instance = super().__call__(*args, **kwargs)
            self.instance_container.append(instance)
        
        return self.instance_container[0]
    
class Config(metaclass = Singleton):
    def __init__(self):
        self.FEATURES = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 
                         'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 
                         'TrackDurationMs', 'Energy']
        self.TARGET = 'BeatsPerMinute'

CONFIG = Config()
print('complete')


baseline_correlation = get_train().select(pl.exclude('id')).sample(10000, seed = 0).corr().with_columns(
    pl.Series(name = 'Variables', values = CONFIG.FEATURES + [CONFIG.TARGET])
).select(['Variables', 'BeatsPerMinute'])
baseline_correlation


nrows = 3
ncols = 3
fig, ax = plt.subplots(figsize = (12, 12), nrows = nrows, ncols = ncols)

train_sample = get_train().select(pl.exclude('id')).sample(10000, seed = 0)
feature_counter = 0
for i in range(nrows):
    for j in range(ncols):
        ax[i,j].scatter(train_sample[CONFIG.FEATURES[feature_counter]], train_sample[CONFIG.TARGET] )
        ax[i,j].set_ylabel(CONFIG.TARGET)
        ax[i,j].set_xlabel(CONFIG.FEATURES[feature_counter])
        feature_counter +=1
plt.show()


pipeline = make_pipeline(MinMaxScaler(), FunctionTransformer(func = np.log1p, feature_names_out = 'one-to-one' ))
result = pipeline.fit_transform(train_sample)

result_correlation = pl.DataFrame(result, schema = list(pipeline.get_feature_names_out())).corr().with_columns(
    pl.Series(name = 'Variables', values = CONFIG.FEATURES + [CONFIG.TARGET])
).select(['Variables', 'BeatsPerMinute'])
result_correlation


baseline_correlation.join(other = result_correlation, on = 'Variables', how = 'left').with_columns(
    (pl.col('BeatsPerMinute_right') - pl.col('BeatsPerMinute')).alias('Change'),
    pl.col('BeatsPerMinute_right').gt(pl.col('BeatsPerMinute')).alias('IncreaseCorrelation')
)


#Update the MInMaxScaler part
MM = MinMaxScaler()
MM.set_output(transform = 'polars')
transformer = make_union(
   #make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False, order='C')),
    make_pipeline(QuantileTransformer(random_state = 0)),
    make_pipeline(MM, ColumnTransformer( transformers = [
        ('log1p', FunctionTransformer(func = np.log1p, feature_names_out = 'one-to-one' ), ['AudioLoudness', 'VocalContent', 'LivePerformanceLikelihood'] )
    ], remainder = 'passthrough')),
    make_pipeline(KBinsDiscretizer(n_bins = 10, random_state = 0, encode = 'onehot-dense', strategy = 'uniform')),
    make_pipeline(SplineTransformer(n_knots = 5, include_bias = True))
)

generate_model = lambda : make_pipeline(transformer, RidgeCV(alphas = (0.1, .5, 2, 10)))
generate_model()


import gc

Config().N_SPLITS = 5

kf = KFold(n_splits = CONFIG.N_SPLITS, shuffle = True, random_state = 0)

X = get_train().select(CONFIG.FEATURES)
Y = get_train().select(CONFIG.TARGET)

test = get_test().select(CONFIG.FEATURES)

oof = np.zeros(shape = (Y.shape[0]))

test_result = np.zeros(shape = (test.shape[0]))

scores = {}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train = X[train_idx]
    X_orig = get_original().select(CONFIG.FEATURES)
    X_combo = pl.concat([X_train, X_orig], how = 'vertical')
    X_val = X[val_idx]
    
    y_train = Y[train_idx]
    y_orig = get_original().select(CONFIG.TARGET)
    y_combo = pl.concat([y_train, y_orig], how = 'vertical')
    y_val = Y[val_idx]


    model = generate_model()

    model.fit(X_combo, y_combo)
    oof[val_idx] = model.predict(X_val)

    score = np.sqrt(mean_squared_error(y_val, oof[val_idx]))
    scores[f'F{fold}'] = score
    print(f'Fold {fold} Root MSE score: {score}')

    
    del X_train, X_val, y_train, y_val, X_combo, y_combo
    gc.collect()


pl.DataFrame(scores).with_columns(
    pl.concat_list(pl.all()).list.mean().alias('MEAN'),
    pl.concat_list(pl.all()).list.std(ddof = 1).alias('STD')
)

#26.464837 - 5 fold MEAN baseline


overall_score = np.sqrt(mean_squared_error(Y, oof))
print(f'Overall RMSE score: {overall_score}')


full_train = pl.concat([get_train().select(CONFIG.FEATURES), get_original().select(CONFIG.FEATURES)], how = 'vertical')
full_y = pl.concat([get_train().select(CONFIG.TARGET), get_original().select(CONFIG.TARGET)], how = 'vertical')

model = generate_model()
model.fit(full_train, full_y)

test_full = get_test().select(CONFIG.FEATURES)

test_predictions = model.predict(test_full)
print('complete')



sub = get_sample_submission().select('id')
sub = sub.with_columns(
    BeatsPerMinute = test_predictions.ravel()
)

sub.write_csv("submission.csv")
sub.head()


from IPython.display import FileLink
FileLink('submission.csv')

