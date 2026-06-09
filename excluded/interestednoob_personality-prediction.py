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


import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score, jaccard_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types
from sklearn.datasets import fetch_openml
from sklearn import metrics
import numpy
from numpy import sqrt
from numpy import sum
from numpy import square
import seaborn
import matplotlib
import statsmodels
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import time
import keras
from sklearn.decomposition import PCA
import xgboost as xgb
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from mlxtend.regressor import StackingRegressor
import lightgbm as lgbm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

import gc
import joblib


df_train = pandas.read_csv('/kaggle/input/playground-series-s5e7/train.csv')



df_train.describe()


df_train.columns


df_train.shape


df_train


print(df_train.isna().sum(),
      '\n\n',
      df_train.isna().sum()/18524)


df_train = pandas.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_train = df_train.bfill()
df_train['Drained_after_socializing'] = df_train['Drained_after_socializing'].replace({'Yes':1, 'No':0})
df_train['Stage_fear'] = df_train['Stage_fear'].replace({'Yes':1, 'No':0})
df_train = df_train.drop(['id'], axis = 1)

X = df_train.loc[:, df_train.columns != 'Personality']
y = df_train['Personality'].replace({'Introvert':0, 'Extrovert':1}) ##convert into binary

y

del df_train


y.shape


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.regularizers import L1, L2

import gc
gc.collect()

input_dim = X.shape[1]

# define NN
input_layer = Input(shape=(input_dim,))
x = Dense(128, activation = 'relu')(input_layer)
x = Dense(64, activation = 'tanh', kernel_regularizer=L1(0.2))(x)
x = Dropout(0.3)(x)
x = Dense(16, activation = 'tanh')(x)
x = Dense(8, activation = 'relu')(x)
output_layer = Dense(1, activation = 'sigmoid')(x)

# create NN
dense = Model(inputs=input_layer, outputs=output_layer)

#train, evaluate and val NN
dense.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['auc', 'binary_accuracy', 'binary_crossentropy'])

history = dense.fit(X, y, epochs=200, batch_size=256, shuffle=True , validation_split=0.2)

dense.summary()
loss, auc, acc, crsentp = dense.evaluate(X, y)
print(f"Test loss: {loss:.4f} auc: {auc:.4f} acc: {acc:.4f} Cross Entropy: {crsentp:.4f}")

#predict and save
y_pred = dense.predict(X)
new = pandas.DataFrame(y_pred).round(0)
new.to_csv('/kaggle/working/y_pred.csv', index = False)

# save
dense.save("/kaggle/working/dense-focal.keras")  # or use .h5 for HDF5 format


train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(0, len(train_loss))

fig, ax = plt.subplots(figsize = (20,10))
ax.plot(epochs, train_loss, label='Training Loss', color = 'red')
ax.plot(epochs, val_loss, label='Val Loss', color='blue')
ax.grid()
fig.savefig('/kaggle/working/epoch2.png')
plt.show()


print('Accuracy %.4f \n\nPrecision %.4f' %(accuracy_score(y, new), precision_score(y, new) ), '\n\nConfusion Matrix:\n\n', confusion_matrix(y, new), '\n\nReport\n\n', classification_report(y, new) )

print('ROC', roc_auc_score(y,new))


print('Accuracy %.4f \n\nPrecision %.4f' %(accuracy_score(y, new), precision_score(y, new) ), '\n\nConfusion Matrix:\n\n', confusion_matrix(y, new), '\n\nReport\n\n', classification_report(y, new) )

print('ROC', roc_auc_score(y,new))


df_test = pandas.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test = df_test.bfill().ffill()
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].replace({'Yes':1, 'No':0})
df_test['Stage_fear'] = df_test['Stage_fear'].replace({'Yes':1, 'No':0})
df_test = df_test.drop(['id'], axis = 1)

X = df_test.loc[:, df_test.columns != 'Personality']

del df_test


y_submit = dense.predict(X)
new = pandas.DataFrame(y_submit).round(0).rename({0:'Personality'}, axis = 1)
new['id'] = new.index+18524
new = new[['id', 'Personality']]
new['Personality'] = new['Personality'].replace({0:'Introvert', 1:'Extrovert'}) ##convert into binary
new.to_csv('/kaggle/working/y_submit.csv', index = False)


##check
pandas.read_csv('/kaggle/working/y_submit.csv')


df_train = pandas.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_train = df_train.bfill()
df_train['Drained_after_socializing'] = df_train['Drained_after_socializing'].replace({'Yes':1, 'No':0})
df_train['Stage_fear'] = df_train['Stage_fear'].replace({'Yes':1, 'No':0})
df_train = df_train.drop(['id'], axis = 1)

X = df_train.loc[:, df_train.columns != 'Personality']
y = df_train['Personality'].replace({'Introvert':0, 'Extrovert':1}) ##convert into binary

del df_train
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)


from sklearn.linear_model import RidgeClassifier, LogisticRegression, Perceptron
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier

sgd = Pipeline([('std', StandardScaler()), ('sgd',SGDRegressor())])
pac = Pipeline([('std', StandardScaler()), ('par',PassiveAggressiveRegressor())])
hgbc =  Pipeline([('std', StandardScaler()), ('hgbr',HistGradientBoostingClassifier())])
pcpt =  Pipeline([('std', StandardScaler()), ('hgbr',Perceptron())])
log = Pipeline([('std', StandardScaler()), ('lr',LogisticRegression())])
rd = Pipeline([('std', StandardScaler()), ('rd',RidgeClassifier())])

models = [('SGD',sgd),('PAR',pac),('Hist GB Reg',hgbc), ('Perceptron', pcpt), ('Logit',log), ('Ridge',rd)]

for model_name, model in models:
    ave_r2, ave_mae, ave_mape, ave_rmse, ave_mdae, ave_time = [],[],[],[],[],[]
    for i in range(0,5):
        kfold = model_selection.KFold(n_splits=5, random_state=i, shuffle=True) 
	
    	# execute cross val to est skill of ML model (cross_validate faster)
        start = time.time()
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring = ('r2','neg_root_mean_squared_error','neg_mean_absolute_error','neg_mean_absolute_percentage_error','neg_median_absolute_error'),return_train_score=False)
        end = time.time() 
        duration =  end-start
        
        ave_r2.append(cv_results['test_r2'].mean())
        ave_mae.append(cv_results['test_neg_mean_absolute_error'].mean())
        ave_mape.append(cv_results['test_neg_mean_absolute_percentage_error'].mean())
        ave_rmse.append(cv_results['test_neg_root_mean_squared_error'].mean())
        ave_mdae.append(cv_results['test_neg_median_absolute_error'].mean())
        ave_time.append(duration)

    # print results
    import statistics
    msg1 = "%s: mean R2: %f \nMAE: %f \nMAPE: %f \nRMSE: %f \nMDAE: %f" % (model_name, statistics.mean(ave_r2), statistics.mean(ave_mae), statistics.mean(ave_mape), statistics.mean(ave_rmse), statistics.mean(ave_mdae))
    print(msg1, "\n Time ~", statistics.mean(ave_time))
    gc.collect()


start4 = time.time()
log = Pipeline([('std', StandardScaler()), ('lr',LogisticRegression())])
rd = Pipeline([('std', StandardScaler()), ('rd',RidgeClassifier())])
hgbc =  Pipeline([('std', StandardScaler()), ('hgbr',HistGradientBoostingClassifier())])
vt = VotingClassifier(estimators=[('rd', rd),('histo', hgbc), ('logit', log)])
vt.fit(X_train, y_train)
end4 = time.time()

print("VT \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,vt.predict(X_test))), r2_score(y_test,vt.predict(X_test)), mean_absolute_percentage_error(y_test,vt.predict(X_test)),median_absolute_error(y_test,vt.predict(X_test)),mean_absolute_error(y_test,vt.predict(X_test)),end4-start4 ))


df_test = pandas.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test = df_test.bfill().ffill()
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].replace({'Yes':1, 'No':0})
df_test['Stage_fear'] = df_test['Stage_fear'].replace({'Yes':1, 'No':0})
df_test = df_test.drop(['id'], axis = 1)

X = df_test.loc[:, df_test.columns != 'Personality']

del df_test

y_submit = vt.predict(X)
new = pandas.DataFrame(y_submit).round(0).rename({0:'Personality'}, axis = 1)
new['id'] = new.index+18524
new = new[['id', 'Personality']]
new['Personality'] = new['Personality'].replace({0:'Introvert', 1:'Extrovert'}) ##convert into binary
new.to_csv('/kaggle/working/y_submit.csv', index = False)


##check
pandas.read_csv('/kaggle/working/y_submit.csv')

