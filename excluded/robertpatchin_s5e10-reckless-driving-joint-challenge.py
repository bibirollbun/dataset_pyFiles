### Update libraries
!pip install --upgrade scikit-learn
!pip install --upgrade seaborn


### import common libraries and toolkits
import sys
import os

import pandas as pd
import numpy as np

import sklearn as skl
import lightgbm as lgb
#import xgboost as xgb
from catboost import CatBoostRegressor


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import seaborn as sns

import math 

import joblib
from time import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

### set global defaults
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 15)
pd.set_option('display.width', 130)
pd.set_option('display.precision', 4)

from matplotlib.colors import ListedColormap
MY_PALETTE = sns.xkcd_palette(['ocean blue', 'gold', 'dull green', 'dusty rose', 'dark lavender', 'carolina blue', 'sunflower', 'lichen', 'blush pink', 'dusty lavender', 'steel grey'])
MY_CMAP = ListedColormap(MY_PALETTE)
sns.set_theme(context = 'paper', style = 'ticks', palette = MY_PALETTE, rc={"figure.figsize": (9, 3), "axes.spines.right": False, "axes.spines.top": False})
#sns.palplot(MY_PALETTE) 

import random

SEED = 67
random.seed(SEED)
np.random.seed(SEED)
#torch.manual_seed(SEED)
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#CORES = min(4, cpu_count())  # Limit cores to avoid memory issues
PATH = "/kaggle/input/playground-series-s5e10/"



### load data
def summarize_data(df, features):
    """prints data summary and descriptive stats"""
    print("=" * 69)
    print(df[features].info())
    print("=" * 69)
    print(df[features].head(8).T) 
    print("=" * 69)
    try: print(df[features].describe(include = ['float', 'int']).T)
    except: pass
    try: print(df[features].describe(include = ['object', 'category', 'bool']).T)
    except: pass

def get_target_cuts(df, target, targets, cuts = 10):
    df["qcut_label"] = cuts  - pd.qcut(df[df.target_mask.eq(True)][target], cuts, labels=False)
    df["cut_label"] = cuts  - pd.cut(df[df.target_mask.eq(True)][target], cuts, labels=False)
    df[["qcut_label", "cut_label"]] = df[["qcut_label", "cut_label"]].fillna(-1).astype('int16')
    targets.extend(["qcut_label", "cut_label"])
    return df, targets

def load_tabular_data(path, extra_data = None, verbose = True, csv_sep=","):
    """
    loads tabular data from path into single DataFrame
    assumes path contains train.csv, test.csv, sample_submission.csv
    if extra_data is provided, it is assumed to be a csv file with additional training data
    Returns:
    - merged DataFrame for EDA & feature engineering
    - list of training features
    - list of targets, including column "target_mask" for separating test data
    - target 
    """
    df_train = pd.read_csv(f"{path}/train.csv", sep=csv_sep)
    df_test = pd.read_csv(f"{path}/test.csv", sep=csv_sep)
    df_submission = pd.read_csv(f"{path}/sample_submission.csv", sep=csv_sep)
    
    targets = list(df_submission.columns)
    features = list(df_test.columns)
    id_feature = [feature for feature in features if feature in targets]
    assert len(id_feature) == 1, "Expected exactly one ID column"
    targets = [feature for feature in targets if feature not in id_feature]
    features = [feature for feature in features if feature not in id_feature]
   
    df_test = df_test.merge(df_submission, how = 'left', on = id_feature)
    df = pd.concat([df_train.assign(target_mask = True), df_test.assign(target_mask = False)], ignore_index=True)
    
    if extra_data != None:
        df_extra_training = pd.read_csv(f"{extra_data}", sep=csv_sep)
        missing = set(targets + features + id_feature) - set(df_extra_training.columns)
        assert not missing, f"Extra Data missing columns: {missing}"
        df_extra_training[id_feature[0]] = range(len(df), len(df) + len(df_extra_training))
        df = pd.concat([df, df_extra_training.assign(target_mask = True)])
    
    df.set_index(id_feature, inplace = True)  
    ### clean feature names
    clean_feature_names = {}
    for i, col in enumerate(features):
        clean_feature_names[col] = col.casefold().strip().replace(" ","_").replace("(","_").replace(")","_").replace("-","_")
        features[i] = clean_feature_names[col]
    df.rename(columns=clean_feature_names, inplace=True)
    if verbose:
        print("=" * 69)
        print(f"Loaded {df.target_mask.eq(True).sum()} training samples of {len(features)} predictive features and {len(targets)} target(s) in DataFrame.")
        print(f"Loaded {df.target_mask.eq(False).sum()} testing samples in DataFrame.")
        print(f"DataFrame shape: {df.shape}. Ready to engineer and predict!")
        print("=" * 69)
    targets.append('target_mask')

    return df, features, targets, targets[0]


def split_training_data(df, features, targets, validation_size = None):
    """
    returns X,y (train & test) values as dataframes based on selcted features and targets
    if validation_size provided, returns train, validation, and test dataframes using either
    percentage (float) or selected rows (pd.Index)
    """
    XY[XY.target_mask.eq(True)]
    X = df[df.target_mask.eq(True)][features].astype(np.float32)
    y = df[df.target_mask.eq(True)][targets].astype(np.float32)

    X_test = df[df.target_mask.eq(False)][features].astype(np.float32)
    y_test = df[df.target_mask.eq(False)][targets].astype(np.float32)
    
    if type(validation_size) is float:
        X_train, X_val, y_train, y_val  = skl.model_selection.train_test_split(X, y, test_size = validation_size, random_state = SEED)
        return X_train, y_train, X_val,  y_val, X_test, y_test
    elif type(validation_size) is pd.Index: 
        X_train, y_train = X[~X.index.isin(validation_size)], y[~y.index.isin(validation_size)]
        X_val, y_val = X[X.index.isin(validation_size)], y[y.index.isin(validation_size)]
        return X_train, y_train, X_val,  y_val, X_test, y_test
    elif validation_size == None: return X, y, X_test, y_test
    else:
        print("Slice Type not recognized")


### Visualizations

def plot_target_eda(df, target, sample = 1000):
    if df[target].dtype == float:
        sns.histplot(df[target], bins = 69, kde = True)
    else:
        sns.countplot(data=df, x=target)
    plt.title(f'{target} distribution')
    plt.yticks([])
    plt.show()


def plot_features_eda(df, features, target, label, sample = 1000):
    """ 
    supports feature EDA with regression (float) target
    """
    f = len(features)
    if len(features) > 20:
        print("Plotting 20 features")
        f = 20
        features = features[:20]

    fig = plt.figure(figsize=(10, f * 3))
    gs = gridspec.GridSpec(f, 3, figure=fig, hspace=0.4)
        
    def plot_num_distribution(ax, feature):
        sns.histplot(df[feature], ax=ax, bins = 50)
        ax.set_title(f'{feature} distribution')
        ax.set_yticks([])
        ax.set_ylabel("Count")
        ax.set_xlabel("")

    def plot_cat_distribution(ax, feature, order, color_map):
        sns.countplot(data=df, x=feature, order=order, ax=ax,
                      palette=[color_map[val] for val in order])
        ax.set_title(f'{feature} distribution')
        ax.set_yticks([])
        ax.set_ylabel("Count")
        ax.set_xlabel("")

    def plot_num_relationship(ax, feature,  y_min=0, y_max=1):
        df_sampled = df.sample(n=min(sample, df.shape[0]), random_state=SEED)
        sns.regplot(data=df_sampled, x=feature, y=target, ax=ax,
                    scatter_kws={'alpha': 0.5, 's': 20}, line_kws={'color': 'xkcd:gold', 'linestyle': "--", 'linewidth': 2})
        ax.set_title(f'{target} vs {feature}')
        ax.set_ylabel("")
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("")

    def plot_cat_relationship(ax, feature, order, color_map, y_min=0, y_max=1):
        grouped = df.groupby(feature)
        sampled_dfs = []
        for name, group in grouped:
            frac = min(1.0, sample / len(df))
            sampled_dfs.append(group.sample(n=max(1, int(frac * len(group))), random_state=SEED))
        df_sampled = pd.concat(sampled_dfs)
        sns.stripplot(data=df_sampled, x=feature, y=target, order=order, ax=ax, zorder = 1, 
                          palette=[color_map[val] for val in order], alpha=0.5, jitter=True)
        sns.pointplot(data=df, x=feature, y=target, order=order, ax=ax, zorder = 2, 
                      color=MY_PALETTE[-1], errorbar = None)
        for i, val in enumerate(order):
            subset = df[df[feature] == val][target].dropna()
            q25, q75 = subset.quantile([0.25, 0.75])
            ax.vlines(x=i, ymin=q25, ymax=q75, color=MY_PALETTE[-1], linewidth=2,  zorder = 3)
        ax.set_title(f'{target} vs {feature}')
        ax.set_ylabel("")
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("")
    
    def plot_num_boxplot(ax, feature, label = None, top_label="", bottom_label=""):        
        if label == None:
            sns.boxplot(x = df[feature], ax=ax)
            ax.set_title(f'{feature} outliers')
        else:
            sns.boxplot(x = df[feature], palette=MY_PALETTE , ax=ax, legend = False, gap = .1,
                        hue = df[label], hue_order = sorted(df[label].dropna().unique().tolist()))
            ax.set_title(f'{feature} by target cut')
            ax.text(0, -0.45, top_label, ha='left', va='center', fontsize=8, color = 'black')
            ax.text(0, 0.45, bottom_label, ha='left', va='center', fontsize=8, color = 'black')
        ax.set_yticks([])

    def plot_cat_donut(ax, feature, label, order, color_map, inner_label="", outer_label=""):
        cats = sorted(df[label].dropna().unique().tolist())
        ring_width = 0.7 / len(cats)
        for i, cat in enumerate(cats):
            value_counts = df[df[label] == cat][feature].value_counts()
            sorted_counts = value_counts.reindex(order).dropna()
            slice_colors = [color_map[val] for val in sorted_counts.index]
            radius = 1 - ring_width * i
            ax.pie(sorted_counts, radius=radius, colors=slice_colors,
                   wedgeprops=dict(width=ring_width, edgecolor='w'),
                   labels=sorted_counts.index if i == 0 else None)
            ax.set_title(f'{feature} by target cut')
            ax.text(0, 0, inner_label, ha='center', va='center', fontsize=8, color = 'xkcd:steel grey')
            ax.text(-1.3, -1.3, outer_label, ha='left', va='center', fontsize=8, color = 'xkcd:steel grey')

    row_anchors = []
    for i, feature in enumerate(features):
        is_cat = (df[feature].dtype == "O" or df[feature].dtype == bool or
                  (np.issubdtype(df[feature].dtype, np.integer) and len(df[feature].dropna().unique()) < 10))
        ax0 = fig.add_subplot(gs[i, 0])
        row_anchors.append(ax0)
        if is_cat:
            order = sorted(df[feature].dropna().unique().tolist())
            color_map = dict(zip(order, MY_PALETTE[:len(order)]))
            plot_cat_distribution(ax0, feature, order, color_map)
            plot_cat_relationship(fig.add_subplot(gs[i, 1]), feature, order, color_map)
            plot_cat_donut(fig.add_subplot(gs[i, 2]), feature, label, order, color_map,
                           inner_label="LOW\nrisk", outer_label="HIGH risk")
        else:
            plot_num_distribution(ax0, feature)
            plot_num_relationship(fig.add_subplot(gs[i, 1]), feature)
            plot_num_boxplot(fig.add_subplot(gs[i, 2]), feature, label, 
                            top_label="HIGH risk", bottom_label="LOW risk")

    for i in range(f - 1):
        bottom_y = row_anchors[i].get_position().y0
        top_y = row_anchors[i + 1].get_position().y1
        y_pos = (bottom_y + top_y) / 2

        line = Line2D([0.05, 0.95], [y_pos, y_pos], transform=fig.transFigure,
                      color='black', linewidth=0.5, linestyle='--')
        fig.add_artist(line)

    plt.show()

### for feature to feature comparisons
def plot_pairplot(df, features, sample = 250, title = "", **kwargs):
    """
    """
    print("=" * 69)
    plot_df = df[features].sample(n = min(sample, df.shape[0]), random_state=SEED)
    g = sns.pairplot(plot_df, diag_kind="kde", **kwargs)
    g.map_lower(sns.kdeplot, levels=4, color="xkcd:slate")
    g.figure.suptitle(title, x = 0.98, ha = 'right', y=1.01)
    plt.show()


### for plotting on completion of training
def plot_training_results(y_true, y_pred, TargetTransformer = None):
    if TargetTransformer != None:
        y_t = TargetTransformer.inverse_transform(np.array(y_true).reshape(-1, 1) )
        y_p = TargetTransformer.inverse_transform(np.array(y_pred).reshape(-1, 1) )
    else:
        y_t = np.array(y_true).reshape(-1, 1)
        y_p = np.array(y_pred).reshape(-1, 1)
    val_score = calculate_score(y_t, y_p)
    
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 3, figure=fig)
        
    def plot_predictions(ax):
        ax.plot([min(y_t), max(y_t)], [min(y_t), max(y_t)], color='xkcd:gold', linestyle='-')  # Ideal line
        ax.plot([min(y_t), max(y_t)], [min(y_t) + val_score, max(y_t) + val_score], color='xkcd:silver', linestyle='--')  # Ideal line
        ax.plot([min(y_t), max(y_t)], [min(y_t) - val_score, max(y_t) - val_score], color='xkcd:silver', linestyle='--')  # Ideal line
        ax.scatter(y_p[:1000], y_t[:1000], alpha=0.3)
        ax.set_title(f'Actual(y) v Predicted(x) Score: {val_score:.5f}')
        ax.axis('off')
        
    def plot_distribution(ax):
        ax.hist(y_t, bins=50, color='xkcd:silver', alpha=0.7, density = True)
        ax.hist(y_p, bins=50, color='xkcd:ocean blue', alpha=0.7, density = True)
        ax.set_title("Prediction Distribution vs Training Distribution")
        ax.set_yticks([])
        ax.set_ylabel("Probability Density")

    def plot_residuals(ax):
        residuals = y_p - y_t
        ax.hist(residuals, bins=50, color='xkcd:dull green', alpha=0.7)
        ax.set_title("Residual Distribution")
        ax.set_yticks([])
        ax.set_ylabel("Count")

    plot_predictions(fig.add_subplot(gs[:, :2]))
    plot_residuals(fig.add_subplot(gs[0, 2]))
    plot_distribution(fig.add_subplot(gs[1, 2]))

    plt.tight_layout()
    plt.show()


#sns.palplot(MY_PALETTE) 
#print(f"Using device: {DEVICE}")
#print(f"Using {CORES} CPU cores when multiprocessing")
print(f"Home path is: {PATH}")


### load data into single DataFrame for Feature Engineering 
XY, features, targets, target = load_tabular_data(PATH)


### original features saved for deployed model validation
validation_sample = XY.iloc[-10:,:]


###Target stats
summarize_data(XY, target)


### plot target distribution
plot_target_eda(XY[XY.target_mask.eq(True)], target)

### get target cuts and qcuts as labels
XY, targets = get_target_cuts(XY, target, targets, cuts=7)


###feature stats
summarize_data(XY, features)


### plot features
plot_features_eda(XY[XY.target_mask.eq(True)], features, target,  "qcut_label")


### Data Engineering
def get_feature_interactions(df):
    df['sum_speed_night_acc'] = df[['speed_limit_70', 'speed_limit_60', 'lighting_night',
                                'num_reported_accidents_3', 'num_reported_accidents_4']].sum(axis=1)
    df['sum_speed_night_acc_weax'] = df['sum_speed_night_acc'] + df['weather_rainy'] + df['weather_foggy']
    df['sum_night_weax'] = df[['lighting_night', 'weather_rainy', 'weather_foggy']].sum(axis=1)
    df['s_curve'] = np.sin(2 * df['curvature'])
    df['dead_man_curve'] = df['sum_speed_night_acc_weax'] * df['s_curve']
    df['curvature_sqrt'] = df['curvature'] ** 0.5
    df['curvature_sq'] = df['curvature'] ** 2
    return df


### clip num_reported_accidents
XY['num_reported_accidents'] = XY['num_reported_accidents'].clip(upper=4)

### One Hot Encode Integers and Categoricals
training_features = [f for f in XY.columns.tolist() if f not in targets and
                     XY[f].dtype == 'O' or 
                     XY[f].dtype == 'int']

XY = pd.get_dummies(XY, columns=training_features, drop_first=True)

### get new features based on feature interactions
XY = get_feature_interactions(XY)


### plot new features
plot_features = [f for f in XY.columns.tolist() if f not in targets]
plot_features_eda(XY[XY.target_mask.eq(True)], plot_features[-7:], target, "qcut_label")


### Training and Evaluations
def calculate_score(actual, predicted, metric='rmse'):
    if metric == 'rmse':
        return skl.metrics.root_mean_squared_error(actual, predicted)
    elif metric == 'mae':
        return skl.metrics.mean_absolute_error(actual, predicted)
    elif metric == 'r2':
        return skl.metrics.r2_score(actual, predicted)
    else:
        raise ValueError("Unsupported metric")

def train_and_score_model(X_train, X_val, y_train, y_val, model, target = target, verbose = False, TargetTransformer = None):
    """
    trains a model and returns teained model & score
    """
    tic = time()
    model.fit(X_train, y_train)
    toc = time()
    if verbose == True: print(f" ***  time training: {(toc - tic):.2f} sec  ***")
    y_predict = model.predict(X_val)
    if TargetTransformer != None:
        y_v = TargetTransformer.inverse_transform(y_val.values.reshape(-1, 1))
        y_p = TargetTransformer.inverse_transform(y_predict.reshape(-1, 1))
    else:
        y_v = np.array(y_val).reshape(-1, 1)
        y_p = np.array(y_predict).reshape(-1, 1)
    
    score = calculate_score(y_v, y_p)
    print(f"***  score:  {score:.4f}  ***")
    if verbose == True:
        skl.metrics.PredictionErrorDisplay.from_predictions(y_v, y_p, kind = 'actual_vs_predicted')
        plt.show()
    return model, score


def get_feature_importance(X_train, X_val, y_train, y_val, verbose = True):
    """
    trains a model and returns dataseries of feature impotance 
    """
    ModelFeatureImportance, score = train_and_score_model(X_train, X_val, y_train, y_val, model = lgb.LGBMRegressor(verbose = -1), verbose = False)
    df = pd.Series(ModelFeatureImportance.feature_importances_, name="importance", index=X_train.columns)
    df.sort_values(ascending=False, inplace = True)
    print("=" * 69)
    print(f"  ***  Top feature is: {df.index[0]}  *** \n")
    if verbose:
        print("=" * 69)
        print(f"  Top Features:")
        print(df.head(12))
        print("=" * 69)
        print(f"  Bottom Features:")
        print("=" * 69)
        print(df.tail(12))
        print("=" * 69)
        print(f"Zero importance features: {(df == 0).sum()} of {len(df.index)}")
    return df


### Select Training Features
training_features = [f for f in XY.columns.tolist() if f not in targets]
X_train, y_train, X_val,  y_val, X_test, y_test = split_training_data(XY, training_features, target, validation_size = 0.2)

### Train Model
model = CatBoostRegressor(n_estimators=500, learning_rate=0.042, loss_function='RMSE', verbose = 100)
trained_model, _ = train_and_score_model(X_train, X_val, y_train, y_val, model)

### Save Model for Deployment
joblib.dump(trained_model, f"reckless_driving_model.pkl")

### Plot Validation Data Predictions
y_preds = trained_model.predict(X_val)
plot_training_results(y_val, y_preds)


### make predictions and submit!

def make_predictions(X, y, model, path = PATH, target = target, verbose = True, TargetTransformer = None):
    
    y_test = model.predict(X) 

    if TargetTransformer == None:
        y_pred = np.array(y_test).reshape(-1, 1) 
    else:
        y_pred = TargetTransformer.inverse_transform(np.array(y_test).reshape(-1, 1))

    SUBMISSION = pd.read_csv(f"{path}/sample_submission.csv")
    SUBMISSION[target] = y_pred
    SUBMISSION.to_csv('/kaggle/working/submission.csv', index=False)

    if verbose:
        sns.histplot(SUBMISSION[target], bins = 69, kde = True)
        plt.title(f'{target} test predictions distribution')
        plt.yticks([])
        plt.show()

    print("=" * 6, 'save success', "=" * 6, "\n")
    print(f"Predicted target mean: {y_pred.mean():.4f} +/- {y_pred.std():.4f}")
    return y_pred


### make predictions on test data and save submission csv
y_submit = make_predictions(X_test, y_test, trained_model) 


### validate deployed model makes predictions properly
#a = pd.Series(dict(zip(training_features, list(trained_model.get_feature_importance())))).sort_values(ascending = False)
#validation_sample[target] = y_submit[-10:]
#validation_sample[:4].T

