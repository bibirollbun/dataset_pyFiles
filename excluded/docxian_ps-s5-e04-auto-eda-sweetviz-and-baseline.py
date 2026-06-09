# install packages
!pip install sweetviz


# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt


# configs

pd.set_option('display.float_format', lambda x:'%f'%x) 
# see: https://www.kaggle.com/competitions/playground-series-s5e4/discussion/571034

# random seed
my_random_seed = 4242

# aesthetics
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'


# load data
t1 = time.time()
df_train = pd.read_csv('../input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('../input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('../input/playground-series-s5e4/sample_submission.csv')
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1,2))


# overview train
df_train.info()


# overview test
df_test.info()


# define target
target = 'Listening_Time_minutes'


# preview train
df_train.head()


# for automatic EDA
import sweetviz


# create EDA report
sv_report = sweetviz.analyze(df_train, target_feat = target)


# display report in notebook
sv_report.show_notebook()


# create report
sv_train_test = sweetviz.compare([df_train, 'Training'], [df_test, 'Test'], target_feat = target)


# and display
sv_train_test.show_notebook()


# machine learning tools
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator, H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML


# start H2O
h2o.init(max_mem_size='12G', nthreads=4) # Use maximum of 12 GB RAM and 4 cores


# upload data in H2O environment
t1 = time.time()
train_hex = h2o.H2OFrame(df_train)
test_hex = h2o.H2OFrame(df_test)
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1,4))


# pick predictors
predictors = ['Podcast_Name', 'Episode_Title', 'Episode_Length_minutes', 'Genre',
              'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time',
              'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment']


# define AutoML setup
max_minutes = 15
model = H2OAutoML(max_runtime_secs=max_minutes*60,
                  sort_metric='RMSE',
                  exclude_algos=['StackedEnsemble'],
                  # include_algos=['GBM'],
                  seed=my_random_seed)


# train model
t1 = time.time()
model.train(predictors, target, training_frame = train_hex)
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1,4))


# leaderboard
lb = model.leaderboard
lb.head(rows=lb.nrows) # show full leaderboard


fit = model.leader
fit


# features importance
fit.varimp_plot()
plt.show()


# predict on test data
pred_test = fit.predict(test_hex).as_data_frame(use_multi_thread=True)
pred_test = pred_test.predict
pred_test = pred_test.clip(0) # clip negative values
df_test['pred_AML'] = pred_test


# basic stats
df_test.pred_AML.describe()


# plot predictions
plt.figure(figsize=(8,3))
df_test.pred_AML.plot(kind='hist', bins=100,
                      color=default_color_2)
plt.title('Predictions Test')
plt.grid()
plt.show()


# prepare and save submission files
df_sub_AML = df_sub.copy()
df_sub_AML[target] = df_test.pred_AML
df_sub_AML.to_csv('submission_AML.csv', index=False)

