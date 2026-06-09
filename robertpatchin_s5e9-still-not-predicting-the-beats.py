### Update libraries
!pip install --upgrade scikit-learn
#!pip install --upgrade seaborn


### inport common libraries and toolkits
import sys
import os
import warnings

import pandas as pd
import numpy as np
import sklearn as skl
import lightgbm as lgb

from time import time
import random

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

### set global defaults
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 15)
pd.set_option('display.width', 130)
pd.set_option('display.precision', 4)

random.seed(80085)

my_palette = sns.xkcd_palette(['ocean blue', 'gold', 'dull green', 'rose', 'steel', 'periwinkle'])
sns.set_theme(context = 'paper', style = 'ticks', palette = my_palette)

#sns.palplot(my_palette) 


###load data
#import kagglehub
#path = kagglehub.dataset_download("/datasets/gauravduttakiit/bpm-prediction-challenge/data/")

def get_dataframe(path, extra_data = None):
    """
    -loads train, test and sample_submission csv files in base as Dataframes
    -sets id_feature as Dataframes index 
    -cleans feature names
    -adds extra training data (e.g original data) if provided
    Returns:
    1 merged DataFrame for EDA & feature engineering
    2 list of training features
    3 list of targets
    """
    df_train = pd.read_csv(f"{path}/train.csv")
    df_test = pd.read_csv(f"{path}/test.csv")
    df_submission = pd.read_csv(f"{path}/sample_submission.csv")
    
    targets = list(df_submission.columns)
    features = list(df_test.columns)
    id_feature = [feature for feature in features if feature in targets]
    targets = [feature for feature in targets if feature not in id_feature]
    features = [feature for feature in features if feature not in id_feature]
   
    df_test = df_test.merge(df_submission, how = 'left', on = id_feature)
    df = pd.concat([df_train.assign(train = 'train'), df_test.assign(train = 'test')])
    
    if extra_data != None:
        df_extra_training = pd.read_csv(f"{extra_data}", sep =";")
        #df_extra_training = pd.read_csv(f"{extra_data}")
        try: df_extra_training.drop(id_feature[0], inplace = True, axis = 1)
        except: pass
        df_extra_training.reset_index(inplace = True)
        df_extra_training['index'] = df_extra_training['index'].apply(lambda x: x + len(df))
        df_extra_training.rename(columns={'index' : id_feature[0]}, inplace=True)
        df = pd.concat([df, df_extra_training.assign(train = 'extra')])
    
    df.set_index(id_feature, inplace = True)  
    ### clean feature names
    clean_feature_names = {}
    for i, col in enumerate(features):
        clean_feature_names[col] = col.casefold().strip().replace(" ", "_").replace("(", "_").replace(")", "_")
        features[i] = clean_feature_names[col]
    df.rename(columns=clean_feature_names, inplace=True)
    
    return df, features, targets


### load data into XY DataFrame

path = "/kaggle/input/playground-series-s5e9/"

XY, features, targets = get_dataframe(path)


### view targets
def summarize_data(df, features):
    """prints data summary and descriptive stats"""
    print("=" * 69)
    print(df[features].info())
    print("=" * 69)
    print(df[features].head(8).T)  ## use tail to show testing data
    print("=" * 69)
    try: print(df[features].describe(include = ['float', 'int']).T)
    except: pass
    try: print(df[features].describe(include = ['object', 'category', 'bool']).T)
    except: pass
    pass


def plot_pairplot(df, features, sample = 250, title = "", **kwargs):
    """
    plots pairplot of featurees in df
    sample is used to avoid overloading scatter plots
    """
    print("=" * 69)
    plot_df = df[features].sample(n = min(sample, df.shape[0]), random_state=42)
    g = sns.pairplot(plot_df, diag_kind="kde", **kwargs)
    g.map_lower(sns.kdeplot, levels=4, color="DarkSlateGrey")
    g.figure.suptitle(title, x = 0.98, ha = 'right', y=1.01)
    plt.show()
    pass


def plot_continuous_eda(df, features, sample = 250, target = targets[0]):
    """
    for each feature:
      plots scatterplot (feature relationship to target)
      plots feature histogram (data distribution)
      plots feature boxplot (outliers)
    supports EDA of numeric/continuous data
    """
    for feature in features:
        print("=" * 6, f"{feature} EDA", "=" * 6)
        fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4), sharex = True)
        axs[0].set_title(f'{feature} by {target}')
        axs[0].set_ylabel(" ")
        #sns.boxplot(data = df, x = feature, y = target, ax = axs[0], hue = target, legend = False, order = df[target].value_counts().index)
        #sns.lineplot(data = df, x=feature, y = target, estimator = 'mean', ax = axs[0]) 
        sns.scatterplot(data = df.sample(n = min(sample, df.shape[0])), x=feature, y = target, ax = axs[0]) 
        axs[1].set_title(f'{feature} distribution')
        axs[1].set_yticks([])
        sns.histplot(df[feature], ax=axs[1], bins = 'rice')
        axs[2].set_title(f'{feature} outliers')
        sns.boxplot(x = df[feature], ax=axs[2])
        for ax in axs:
             ax.set_xlabel(" ")  
        sns.despine
        plt.show()
    pass

from scipy.stats.mstats import winsorize
def get_winsor(df, features, limits = [0.0025, 0.0025]):
    """
    reduce outlier impact by winsorizing features in df using limits
    returns df
    """
    for feature in features:
        winsorize(df[feature], limits=limits, inplace=True)
    return df
    
def get_transformed_targets(df, targets, TargetTransformer):
    """
    scales or transforms targets in df with scikit learn scalers / transformers
    returns 
    1. df with transformed targets
    2. TargetTransformer to support inverse transformation of predictions
    """
    for target in targets:
        y = df.query("train != 'test'")[target].values
        TargetTransformer.fit(y.reshape(-1,1))
        y = df[target].values
        df[target] = TargetTransformer.transform(y.reshape(-1,1))
    return df, TargetTransformer

def get_transformed_features(df, features, FeatureTransformer):
    """
    scales or transforms features in df with scikit learn scalers / transformers
    returns df with transformed targets
    """
    for feature in features:
        X = df[feature].values
        df[feature] = FeatureTransformer.fit_transform(X.reshape(-1,1))
    return df
    
def get_log_transform_features(df, features):
    """
    transforms features in df using np.log
    feature values <= 0 will be inf / NaN
    returns df with transformed features
    """
    for feature in features:
        X = df[feature].values
        df[feature] = np.log1p(X)
        df[feature] = df[feature].apply(lambda x: -1 if x < -1 else x)
    return df

def plot_transforms(df, features):
    """plots selected transforms to inform transformer selection"""
    for feature in features:
        print("=" * 69)
        df_plot = pd.DataFrame(df[feature])
        try:
            y = df[feature].values
            df_plot['StandardScaler'] = skl.preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
            #df_plot['standarize'] =  (df_plot[feature] - df_plot[feature].mean()) / df_plot[feature].std()
            df_plot['PowerTransformer']  = skl.preprocessing.PowerTransformer().fit_transform(y.reshape(-1,1))
            df_plot['y_logTransform'] = np.log1p(y)
            #df_plot['QuantileTransformer']  = QuantileTransformer().fit_transform(y.reshape(-1,1))
            df_plot['MinMaxScaler'] = skl.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(y.reshape(-1,1))
            columns = list(df_plot.columns)
            fig, axs = plt.subplots(nrows=1, ncols=len(columns), sharey=False, figsize=(15,3))
            for i, col in enumerate(columns):
                plt.subplot(1, len(columns), i+1)
                sns.histplot(data=df_plot, stat='percent', x=col, kde=False, bins=30, multiple="stack")
                sns.despine()
                plt.yticks(())
                plt.ylabel(None)
            plt.title(f"compare {feature} transformers", fontsize=10)
            plt.show()
        except Exception as e:
            print(f"Unable to transform {feature}")
    pass


### print and plot summary info on target

summarize_data(XY.query("train != 'test'"), targets)
plot_continuous_eda(XY.query("train != 'test'"), targets)


### print and plot summary info on features

summarize_data(XY, features[:])

#plot_continuous_eda(XY, features[:3])


plot_features = features.copy()
plot_features.append(targets[0])

plot_pairplot(XY.query("train != 'test'"), plot_features)


### Get X/Y 

def calculate_score(actual, predicted):
    return skl.metrics.root_mean_squared_error(actual, predicted)

def get_Xy(df, features, targets = targets, validation_size = 0):
    """
    returns X,y (train & test) values as dataframes based on selcted features and targets
    if validation_ siza > 0, returns train, validation, and test dataframes
    """
    df_t, df_test = df[df["train"].ne('test')], df[df["train"].eq('test')]
    X = df_t[features]
    y = df_t[targets]
    X_test = df_test[features]
    y_test = df_test[targets]
    
    if validation_size > 0:
        X_train, X_val, y_train, y_val  = skl.model_selection.train_test_split(X, y, test_size = validation_size, random_state = 69)
        return X_train, X_val, y_train, y_val, X_test, y_test
    
    else: return X, y, X_test, y_test


def train_and_score_model(X_train, X_val, y_train, y_val, model, targets = targets, verbose = True):
    """
    trains a model and returns teained model & score
    """
    tic = time()
    model.fit(X_train, y_train)
    toc = time()
    if verbose == True: print("=" * 6, f"time training: {(toc - tic):.2f} sec")
    y_predict = model.predict(X_val)
    score = calculate_score(y_val, y_predict)
    print("=" * 6, f"  score:  {score:.5f}")
    if verbose == True:
        skl.metrics.PredictionErrorDisplay.from_predictions(y_val, y_predict, kind = 'actual_vs_predicted')
        plt.show()
    return model, score




### select top features and train a model
training_features = features.copy()
X_train, X_val, y_train, y_val, X_test, y_test = get_Xy(XY, training_features, validation_size = 0.1)

model = lgb.LGBMRegressor(verbose = -1)
trained_model, _ = train_and_score_model(X_train, X_val, y_train, y_val, model, verbose = True)


def predict_test_data(model, X_test, path = path, target = targets[0]):
    SUBMISSION = pd.read_csv(f"{path}/sample_submission.csv")
    SUBMISSION[target] = model.predict(X_test)
    SUBMISSION.to_csv('/kaggle/working/submission.csv', index=False)
    print(SUBMISSION.tail(5).T)
    print("=" * 6, 'save success', "=" * 6, "\n")
    
    return 
    
predict_test_data(model, XY.query("train == 'test'")[training_features])

