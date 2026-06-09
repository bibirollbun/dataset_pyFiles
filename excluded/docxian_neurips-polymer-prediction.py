# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# stats
from scipy import stats

# missing values
import missingno as msno

# ML
import h2o
from h2o.estimators import H2OGradientBoostingEstimator


# configs

# aesthetics
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'

# random
my_random_seed = 12345

# warnings
import warnings
warnings.filterwarnings('ignore')


# show files
!ls -l '/kaggle/input/neurips-open-polymer-prediction-2025'


# load data
t1 = time.time()
df_train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
df_test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
df_sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1,4))


# preview
df_train.head()


# show structure of data - train
df_train.info(show_counts=True, verbose=True)


# visualize structure of missing values
msno.matrix(df_train)


# (preliminary) test set, just three observations
df_test


# very simple features
df_train['length'] = df_train.SMILES.apply(len)
df_test['length'] = df_test.SMILES.apply(len)

df_train['count_c'] = df_train.SMILES.apply(lambda x : x.count('c'))
df_test['count_c'] = df_test.SMILES.apply(lambda x : x.count('c'))

df_train['count_C'] = df_train.SMILES.apply(lambda x : x.count('C'))
df_test['count_C'] = df_test.SMILES.apply(lambda x : x.count('C'))

df_train['count_O'] = df_train.SMILES.apply(lambda x : x.count('O'))
df_test['count_O'] = df_test.SMILES.apply(lambda x : x.count('O'))

df_train['count_N'] = df_train.SMILES.apply(lambda x : x.count('N'))
df_test['count_N'] = df_test.SMILES.apply(lambda x : x.count('N'))

# list of features
features = ['length', 'count_c', 'count_C', 'count_O', 'count_N']


# plot features
for f in features:
    plt.figure(figsize=(8,3))
    df_train[f].plot(kind='hist', bins=50, color=default_color_1)
    plt.title(f)
    plt.grid()
    plt.show()


# targets
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


# filtered data frame for each target
df_train_Tg = df_train[~df_train['Tg'].isna()]
df_train_FFV = df_train[~df_train['FFV'].isna()]
df_train_Tc = df_train[~df_train['Tc'].isna()]
df_train_Density = df_train[~df_train['Density'].isna()]
df_train_Rg = df_train[~df_train['Rg'].isna()]


# target: Glass transition temperature
print(df_train_Tg['Tg'].describe())
plt.figure(figsize=(8,4))
df_train_Tg['Tg'].plot(kind='hist', bins=50, color=default_color_3)
plt.title('Tg')
plt.grid()
plt.show()


# target: Fractional free volume
print(df_train_FFV['FFV'].describe())
plt.figure(figsize=(8,4))
df_train_FFV['FFV'].plot(kind='hist', bins=100, color=default_color_3)
plt.title('FFV')
plt.grid()
plt.show()


# target: Thermal conductivity
print(df_train_Tc['Tc'].describe())
plt.figure(figsize=(8,4))
df_train_Tc['Tc'].plot(kind='hist', bins=50, color=default_color_3)
plt.title('Tc')
plt.grid()
plt.show()


# target: Polymer density
print(df_train_Density['Density'].describe())
plt.figure(figsize=(8,4))
df_train_Density['Density'].plot(kind='hist', bins=50, color=default_color_3)
plt.title('Density')
plt.grid()
plt.show()


# target: Radius of gyration
print(df_train_Rg['Rg'].describe())
plt.figure(figsize=(8,4))
df_train_Rg['Rg'].plot(kind='hist', bins=50, color=default_color_3)
plt.title('Rg')
plt.grid()
plt.show()


# scatterplot
sns.pairplot(data=df_train[targets],
             diag_kws = { 'color' : default_color_3},
             plot_kws = { 'color' : default_color_3,
                          'alpha' : 0.5,
                          's' : 15})
plt.show()


# correlation of missingness
msno.heatmap(df_train, cmap='RdBu')


# scatter plots
for t in targets:
    for f in features:
        c = np.round(df_train[f].corr(df_train[t]), 4)
        plt.scatter(df_train[f], df_train[t], color=default_color_3,
                    alpha=0.5)
        plt.title(t + ' vs ' + f + ' | corr = ' + str(c))
        plt.grid()
        plt.show()


# start H2O
h2o.init(max_mem_size='20G', nthreads=4)


train_hex = h2o.H2OFrame(df_train_Tg)
test_hex = h2o.H2OFrame(df_test)


# setup of model
fit_Tg = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                      nfolds=5,
                                      ntrees = 50,
                                      learn_rate = 0.1,
                                      max_depth = 6,
                                      col_sample_rate = 0.7,                                    
                                      stopping_rounds = 10,
                                      stopping_metric = 'MAE',
                                      score_each_iteration = True,                                          
                                      seed=my_random_seed)


# run training
fit_Tg.train(features, 'Tg', training_frame = train_hex);


# short summary
fit_Tg.summary()


# show cross validation results
fit_Tg.cross_validation_metrics_summary().as_data_frame()


# predict on training data
pred_train = fit_Tg.predict(train_hex)
pred_train = pred_train.as_data_frame().predict


# scatter plot of predictions vs actual on training data
plt.scatter(df_train_Tg.Tg, pred_train, color=default_color_1, alpha=0.5)
plt.grid()
plt.show()


# predict on test data
pred_test = fit_Tg.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.Tg = pred_test


train_hex = h2o.H2OFrame(df_train_FFV)
test_hex = h2o.H2OFrame(df_test)


# setup of model
fit_FFV = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                       nfolds=5,
                                       ntrees = 200,
                                       learn_rate = 0.1,
                                       max_depth = 6,
                                       col_sample_rate = 0.7,                                    
                                       stopping_rounds = 10,
                                       stopping_metric = 'MAE',
                                       score_each_iteration = True,                                          
                                       seed=my_random_seed)


# run training
fit_FFV.train(features, 'FFV', training_frame = train_hex);


# short summary
fit_FFV.summary()


# show cross validation results
fit_FFV.cross_validation_metrics_summary().as_data_frame()


# predict on training data
pred_train = fit_FFV.predict(train_hex)
pred_train = pred_train.as_data_frame().predict


# scatter plot of predictions vs actual on training data
plt.scatter(df_train_FFV.FFV, pred_train, color=default_color_1, alpha=0.25)
plt.grid()
plt.show()


# predict on test data
pred_test = fit_FFV.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.FFV = pred_test


train_hex = h2o.H2OFrame(df_train_Tc)
test_hex = h2o.H2OFrame(df_test)


# setup of model
fit_Tc = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                      nfolds=5,
                                      ntrees = 100,
                                      learn_rate = 0.1,
                                      max_depth = 4,
                                      col_sample_rate = 0.7,                                    
                                      stopping_rounds = 10,
                                      stopping_metric = 'MAE',
                                      score_each_iteration = True,                                          
                                      seed=my_random_seed)


# run training
fit_Tc.train(features, 'Tc', training_frame = train_hex);


# short summary
fit_Tc.summary()


# show cross validation results
fit_Tc.cross_validation_metrics_summary().as_data_frame()


# predict on training data
pred_train = fit_Tc.predict(train_hex)
pred_train = pred_train.as_data_frame().predict


# scatter plot of predictions vs actual on training data
plt.scatter(df_train_Tc.Tc, pred_train, color=default_color_1, alpha=0.25)
plt.grid()
plt.show()


# predict on test data
pred_test = fit_Tc.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.Tc = pred_test


train_hex = h2o.H2OFrame(df_train_Density)
test_hex = h2o.H2OFrame(df_test)


# setup of model
fit_Density = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                           nfolds=5,
                                           ntrees = 100,
                                           learn_rate = 0.1,
                                           max_depth = 6,
                                           col_sample_rate = 0.7,                                    
                                           stopping_rounds = 10,
                                           stopping_metric = 'MAE',
                                           score_each_iteration = True,                                          
                                           seed=my_random_seed)


# run training
fit_Density.train(features, 'Density', training_frame = train_hex);


# short summary
fit_Density.summary()


# show cross validation results
fit_Density.cross_validation_metrics_summary().as_data_frame()


# predict on training data
pred_train = fit_Density.predict(train_hex)
pred_train = pred_train.as_data_frame().predict


# scatter plot of predictions vs actual on training data
plt.scatter(df_train_Density.Density, pred_train, color=default_color_1, alpha=0.25)
plt.grid()
plt.show()


# predict on test data
pred_test = fit_Density.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.Density = pred_test


train_hex = h2o.H2OFrame(df_train_Rg)
test_hex = h2o.H2OFrame(df_test)


# setup of model
fit_Rg = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                      nfolds=5,
                                      ntrees = 50,
                                      learn_rate = 0.1,
                                      max_depth = 6,
                                      col_sample_rate = 0.7,                                    
                                      stopping_rounds = 10,
                                      stopping_metric = 'MAE',
                                      score_each_iteration = True,                                          
                                      seed=my_random_seed)


# run training
fit_Rg.train(features, 'Rg', training_frame = train_hex);


# short summary
fit_Rg.summary()


# show cross validation results
fit_Rg.cross_validation_metrics_summary().as_data_frame()


# predict on training data
pred_train = fit_Rg.predict(train_hex)
pred_train = pred_train.as_data_frame().predict


# scatter plot of predictions vs actual on training data
plt.scatter(df_train_Rg.Rg, pred_train, color=default_color_1, alpha=0.25)
plt.grid()
plt.show()


# predict on test data
pred_test = fit_Rg.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.Rg = pred_test


# show submission data
df_sub.head()


# save submission file
df_sub.to_csv('submission.csv', index=False)

