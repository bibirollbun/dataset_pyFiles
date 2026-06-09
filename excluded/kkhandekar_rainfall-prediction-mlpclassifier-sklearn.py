# Install Optuna
!pip install optuna -q


#
# Libraries
#

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, string, re, random, gc, pickle, math,warnings
import json
from datetime import date
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from scipy.sparse import csr_matrix

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objects as go
from plotly import figure_factory as ff

# NLTK
# from nltk.corpus import stopwords

# Transformers
# import transformers
# from transformers import pipeline

# Sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.metrics import pairwise
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.neural_network import *

# Stats
import scipy
from scipy.stats import *

# Optuna
import optuna

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

# Setting
pd.set_option('max_colwidth',None)
seed = 1248
warnings.simplefilter('ignore')
# sentiment_pipeline = pipeline("sentiment-analysis")
# stopw = pd.read_json('/kaggle/input/english-stopwords/stop_words_english.json')
# stopw = stopw.Stopwords.tolist()


#
# Data
#

# Path
train = '/kaggle/input/playground-series-s5e3/train.csv'
test = '/kaggle/input/playground-series-s5e3/test.csv'
sub = '/kaggle/input/playground-series-s5e3/sample_submission.csv'

# Load
train = pd.read_csv(train).set_index('id')
test = pd.read_csv(test).set_index('id')

# Prep main-dataset
train_ds = train.drop(columns=['day'],axis=1)
test_ds = test.drop(columns=['day'],axis=1)

# Impute Test Data
test_ds['winddirection'].fillna(test_ds['winddirection'].median(), inplace=True)

# View
train_ds.head()


#
# Quick EDA (Train Data)
#
train_ds.iloc[:,0:10:1].describe().T


#
# Correlation Matrix
#

# Compute the correlation matrix
corr = train_ds.iloc[:,0:11:1].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation Plot')
plt.show()


# 
# Hist / Distribution Plot 
#
f, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 10), sharex=False)
for ax, feature in zip(axes.flat, train_ds.iloc[:,0:10:1].columns):
    sns.distplot(train_ds.iloc[:,0:10:1][feature] , color="skyblue", ax=ax)
    


#
# Feature Engineering , Scaling & Split
#

# Feature Engineering
x_train = train_ds.loc[:, train_ds.columns != 'rainfall']
y_train = train_ds[['rainfall']]

# Scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
test_ds_scaled = scaler.transform(test_ds)

# Train & Validation Split (only training data)
train_x, val_x, train_y, val_y = train_test_split(x_train_scaled, y_train, test_size=0.1, random_state=seed)
print(f"Training Size: {train_x.shape} | Validation Size: {val_x.shape}")



#
# Train & Fit Model
#

# Initiate
clf = MLPClassifier(hidden_layer_sizes=(150,100,50),batch_size=16,random_state=seed,max_iter = 200)

# Fit
clf.fit(train_x, train_y)


#
# Evaluate & PLot
#
y_pred = clf.predict(val_x)
print('Accuracy Achieved : {:.2f}\n'.format(accuracy_score(val_y, y_pred)))

# Plot
plt.plot(clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


#
# Hyper Parameter Tuning (Optuna)
#

def objective(trial):

    # MLPClassifier Params
    # layer_size = trial.suggest_int('layer_size', 100,300,50) # layer size
    # hidden_layer_sizes = (layer_size,)
    hidden_layer_sizes = (trial.suggest_int('hidden_layer_sizes', 100,300,50),)
    activation = trial.suggest_categorical('activation', ['tanh','logistic','relu']) # activation
    solver = trial.suggest_categorical('solver', ['sgd','adam']) # solver
    alpha = trial.suggest_float('alpha', 1e-4, 5e-2, log=True) # alpha
    batch_size = trial.suggest_int('batch_size', 8,64,8) # batch size
    learning_rate = trial.suggest_categorical('learning_rate', ['constant','adaptive']) # learning rate
    max_iter = trial.suggest_int('max_iter', 200,300,10) # max iter

    # Build Model
    classifier_obj = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                   activation=activation,
                                   solver=solver,
                                   alpha=alpha,
                                   batch_size=batch_size,
                                   learning_rate=learning_rate,
                                   max_iter=max_iter,
                                   random_state=seed)

    score = cross_val_score(classifier_obj, train_x, train_y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


# Pruner
pruner = optuna.pruners.MedianPruner()

# Tune
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",pruner=pruner)
    study.optimize(objective, n_trials=50)
    print(study.best_trial)


# Best Params & Value
print("Best hyperparameters:", study.best_params)
print("Best Accuracy Value:", study.best_value)
best_params = study.best_trial.params


# Applying Params to Model
clf_tune = MLPClassifier(hidden_layer_sizes=(best_params['hidden_layer_sizes'],),
                         activation=best_params['activation'],
                         solver=best_params['solver'],
                         alpha=best_params['alpha'],
                         batch_size=best_params['batch_size'],
                         learning_rate=best_params['learning_rate'],
                         max_iter=best_params['max_iter'],
                         random_state=seed)

# Fit
clf_tune.fit(train_x, train_y)


# Prediction on "actual" Test Data 
y_test_pred = clf_tune.predict(test_ds_scaled)


#
# Create Submission File
#

df_sub = pd.read_csv(sub)
df_sub['rainfall'] = y_test_pred

df_sub.to_csv('submission.csv', index=False)

# View
df_sub.head()

