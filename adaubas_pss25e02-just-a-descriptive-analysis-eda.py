!pip install --upgrade seaborn


end, bold, Red, LightBlue = "\033[0m", "\033[1m", "\033[31m", "\033[94m"
bold_blue, bold_red = bold + LightBlue, bold + Red


import os
path = "/kaggle/input" if os.path.isdir("/kaggle/input/") else "."

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier


train = pd.read_csv(f'{path}/playground-series-s5e2/train.csv', index_col = 'id')
train_extra = pd.read_csv(f'{path}/playground-series-s5e2/training_extra.csv', index_col = 'id')
test = pd.read_csv(f'{path}/playground-series-s5e2/test.csv', index_col = 'id')

origin = pd.read_csv(f"{path}/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
origin["id"] = np.arange(test.index.max() + 1, test.index.max() + 1 + origin.shape[0])
origin = origin.set_index("id")

target = [f for f in train.columns if f not in test.columns][0]
original_features = list(test.columns)

nans = [train[test.columns].isna().sum().sum(), origin[test.columns].isna().sum().sum(), test.isna().sum().sum()]
#nans = [train[test.columns].isna().sum().sum(), test.isna().sum().sum()]
is_nans = np.sum(nans) > 0
    
print(f"Shape for {LightBlue}Train{end} {train.shape} and {LightBlue}Test{end} {test.shape}")
print(f"Nan values in {LightBlue}Train{end} : {nans[0]} | in {LightBlue}Test{end} : {nans[2]}")

print(f"Shape for {LightBlue}Extra{end} {train_extra.shape}| Nan : {train_extra.isna().sum().sum()}")

print(f"Target name is {bold}{target}{end}")

print(f"Shape for {LightBlue}Origin{end} {origin.shape} | ", end='')
print(f"Nan values in {LightBlue}Origin{end} : {nans[1]}")

print(f"\nAvailable columns for training : \n  {list(test.columns)}")

display(train.head(5).style.set_caption("Train"))
display(origin.head(5).style.set_caption("Origin"))


fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5), tight_layout = True)#, sharey = True)
for i, (df, label) in enumerate(zip([origin, train, train_extra], ["Origin", "Train", "Extra"])):
    sns.histplot(df, x = target, ax = ax[i]).set(
        title = f"{target} in {label}. From {df[target].min():.2f} to {df[target].max():.2f}")
    ax[i].spines[["top", "right"]].set_visible(False)


for df in [train, origin, train_extra]:
    df[target] = np.log1p(df[target])
    
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5), tight_layout = True)#, sharey = True)
for i, (df, label) in enumerate(zip([origin, train, train_extra], ["Origin", "Train", "Extra"])):
    sns.histplot(df, x = target, ax = ax[i]).set(
        title = f"Log of {target} in {label}. From {df[target].min():.0f} to {df[target].max():.0f}")
    ax[i].spines[["top", "right"]].set_visible(False)
    
for df in [train, origin, train_extra]:
    df[target] = np.expm1(df[target])


fig, ax = plt.subplots(2, 2, figsize = (15, len(test.columns)), sharey = True)
ax = ax.flatten()
for i, (df, label) in enumerate(zip([train, test, origin, train_extra], ["train", "test", "origin", "Extra"])):
    df[[f for f in test.columns]].nunique().plot.barh(ax=ax[i]).set(title = f"Unique values per column in {label}")
    ax[i].bar_label(ax[i].containers[0], fmt='%.1d', padding = 2);
    ax[i].spines[["right", "bottom"]].set_visible(False)
    ax[i].xaxis.set_ticks_position("top")


df_nan = train[test.columns].isna().sum()
df_nan = df_nan[df_nan>0]
fig, ax = plt.subplots(1, 1, figsize=(10, df_nan.shape[0]))
df_nan.plot.barh(ax = ax).set_title("Nan values in train")
ax.bar_label(ax.containers[0], fmt = '%.1d', padding = 2);
ax.spines[["right", "bottom"]].set_visible(False)
ax.xaxis.set_ticks_position("top")        


def desc_continous_features(
    features, train = train, target_continous_threshold = 10,
    suptitle_fontsize = "x-large", suptitle_weight = "bold", 
    title_fontsize = "large", title_weight = "normal"):

    for f in features:
        
        fig = plt.figure(figsize = (15, 5), constrained_layout = True)
        fig.suptitle(f, fontweight = suptitle_weight, fontsize = suptitle_fontsize)
        gs = fig.add_gridspec(nrows = 1, ncols = 4)
        ax = [fig.add_subplot(gs[:, :1]), fig.add_subplot(gs[:, 1:])]
        
        sns.histplot(train[f], ax = ax[0]).set(title = f)
        
        if train[target].nunique() < target_continous_threshold:
    
            sns.boxplot(data = train, x = target, y = f, ax=ax[1]).set(title = target)
            ax[1].set_xlabel("")
        
        else:
            
            ax[1].hist2d(x = train.dropna(subset=f)[f], y = train.dropna(subset=f)[target], cmap = "Blues")
            ax[1].set_xlabel(f)
            ax[1].set_ylabel(target)
            ax[1].set_title(f"{f} vs {target}")
        
        for i in range(2):
            ax[i].title.set(fontsize = title_fontsize, fontweight = title_weight)
            ax[i].spines[["right", "bottom"]].set_visible(False)
            ax[i].xaxis.set_ticks_position("top")

def desc_categorical_features(
    features, train = train, sharey = False, coutinous_threshold = 20,
    suptitle_fontsize = "x-large", suptitle_weight = "bold", 
    title_fontsize = "large", title_weight = "normal"):
    
    for f in features:

        fig = plt.figure(figsize = (15, train[f].nunique()), constrained_layout = True)
        fig.suptitle(f, fontweight = suptitle_weight, fontsize = suptitle_fontsize)
        gs = fig.add_gridspec(nrows = 1, ncols = 4)
        ax = [fig.add_subplot(gs[:, :1]), fig.add_subplot(gs[:, 1:])]
    
        train[f].value_counts().sort_index(ascending = False).plot.barh(ax = ax[0], sharey = sharey).set(title = "N obs in Train")
        ax[0].bar_label(ax[0].containers[0], fmt = "{:,.0f}", padding = 2)
    
        if train[target].nunique() == 2:
    
            (train.loc[train[target] == 1, f].value_counts(dropna = False) / train[f].value_counts(dropna = False)).sort_index(ascending = False).plot.barh(ax = ax[1], sharey = sharey)
            ax[1].set(title = f"% of {target} == 1 in Train")
            ax[1].bar_label(ax[1].containers[0], fmt = "{:.1%}", padding = 2)
        
        
        elif train[target].nunique() <= coutinous_threshold:
            
            cm = pd.crosstab(train[f], train[target])
            sns.heatmap(
                data = (cm.transpose() / cm.sum(axis = 1)).transpose(), ax = ax[1],
                cmap = sns.dark_palette("#69d", reverse = True, as_cmap = True),
                cbar = False, lw = 0.25, annot = True, fmt = '.0%'
            ).set(title = f'{f} per Target (sum of each row = 100%)')
    
        else:
            
            sns.boxplot(data = train, x = target, y = f, orient = "h", ax=ax[1]).set(title = f"{target} vs {f}")
            ax[1].set_ylabel(target)
            
        for i in range(2):
            ax[i].title.set(fontsize = title_fontsize, fontweight = title_weight)
            ax[i].spines[["right", "bottom"]].set_visible(False)
            ax[i].xaxis.set_ticks_position("top")
            ax[i].set_xlabel("")


desc_continous_features(['Weight Capacity (kg)'])


fig = plt.subplots(figsize = (2, 2))
corr = train[[target, 'Weight Capacity (kg)']].corr(method = 'pearson')
sns.heatmap(corr, annot = True, fmt = '.2f',mask = np.triu(np.ones_like(corr)), cmap = 'coolwarm', cbar = None, linewidth = 1).set(title="Absolute Pearson correlation");


desc_categorical_features([f for f in test.columns if f not in ['Weight Capacity (kg)']], sharey = True)


train["q_weight"] = pd.qcut(train['Weight Capacity (kg)'], q = 10, labels = False)
feats = [f for f in test.columns if f not in ['Weight Capacity (kg)']] + ["q_weight"]


from scipy.stats import chi2_contingency

import psutil
__n_cores = psutil.cpu_count()     # Available CPU cores
print(f"N Cores : {__n_cores}")
from multiprocessing import Pool   # Multiprocess Runs

def df_parallelize_run(func, list_params):
    num_cores = np.min([__n_cores, len(list_params)])
    pool = Pool(num_cores)
    res = pool.map(func, list_params)
    pool.close()
    pool.join()    
    return res

def coef_vcramer(feats, train=train):
    contingency_df = pd.crosstab(train[feats[0]].astype("category"), train[feats[1]].astype("category"))
    chi2 = chi2_contingency(contingency_df)[0]
    n = contingency_df.sum().sum()
    r, k = contingency_df.shape
    if r == 1 or k == 1:
        return [feats[0], feats[1], 0]
    else:
        return [feats[0], feats[1], np.sqrt(chi2 / (n * min((r - 1), (k - 1))))]

list_params = []
for i, feat1 in enumerate(sorted(feats)):
    for j, feat2 in enumerate(sorted(feats)):
        if i > j - 1:
            list_params.append([feat2, feat1])

v_cramer = df_parallelize_run(coef_vcramer, list_params)

res = pd.DataFrame(v_cramer, columns=["features", "index", "v_cramer"]).pivot(columns = "features", index = "index", values = "v_cramer")

fig = plt.subplots(figsize = (10, 10))
sns.heatmap(res, annot = True, fmt = '.2f', cmap = 'coolwarm', cbar = None, mask = np.triu(res)).set(title="Cramer's V");


from functools import partial
fn = partial(coef_vcramer, train = train_extra)

train_extra["q_weight"] = pd.qcut(train_extra['Weight Capacity (kg)'], q = 10, labels = False)
feats = [f for f in test.columns if f not in ['Weight Capacity (kg)']] + ["q_weight"]

list_params = []
for i, feat1 in enumerate(sorted(feats)):
    for j, feat2 in enumerate(sorted(feats)):
        if i > j - 1:
            list_params.append([feat2, feat1])

v_cramer = df_parallelize_run(fn, list_params)

res = pd.DataFrame(v_cramer, columns=["features", "index", "v_cramer"]).pivot(columns = "features", index = "index", values = "v_cramer")

fig = plt.subplots(figsize = (10, 10))
sns.heatmap(res, annot = True, fmt = '.2f', cmap = 'coolwarm', cbar = None, mask = np.triu(res)).set(title="Cramer's V on extra train dataset");


df_temp = pd.concat([test, train[test.columns]], axis = 0)
df_temp["is_test"] = 0
df_temp.loc[test.index, "is_test"] = 1
display(df_temp["is_test"].value_counts())

X_trn, X_val, y_trn, y_val = train_test_split(df_temp[test.columns], df_temp["is_test"], test_size = .2, random_state = 0)
my_et = make_pipeline(
    SimpleImputer(strategy = 'most_frequent').set_output(transform = "pandas"),
    ColumnTransformer(transformers = [
            ("label_encoder", OrdinalEncoder(**{"handle_unknown":"use_encoded_value", 'unknown_value' : -1, 'encoded_missing_value':-1}), 
             [f for f in test.columns if f not in ['Weight Capacity (kg)']])
        ], remainder = "passthrough", verbose_feature_names_out = False).set_output(transform = "pandas"),
    ExtraTreesClassifier(n_estimators = 300, random_state = 666, n_jobs = -1, max_depth = 8)
).fit(X_trn, y_trn)
print(f"AUC : {roc_auc_score(y_val, my_et.predict_proba(X_val)[:, 1]):.2f}")


df_temp = pd.concat([origin, train], axis = 0)
df_temp["is_origin"] = 0
df_temp.loc[origin.index, "is_origin"] = 1
display(df_temp["is_origin"].value_counts())

X_trn, X_val, y_trn, y_val = train_test_split(df_temp[test.columns], df_temp["is_origin"], test_size = .2, random_state = 0)
my_et = make_pipeline(
    SimpleImputer(strategy = 'most_frequent').set_output(transform = "pandas"),
    ColumnTransformer(transformers = [
            ("label_encoder", OrdinalEncoder(**{"handle_unknown":"use_encoded_value", 'unknown_value' : -1, 'encoded_missing_value':-1}), 
             [f for f in test.columns if f not in ['Weight Capacity (kg)']])
        ], remainder = "passthrough", verbose_feature_names_out = False).set_output(transform = "pandas"),
    ExtraTreesClassifier(n_estimators = 300, random_state = 666, n_jobs = -1, max_depth = 8)
).fit(X_trn, y_trn)
print(f"AUC : {roc_auc_score(y_val, my_et.predict_proba(X_val)[:, 1]):.2f}")


for f in test.columns:
    if f not in ['Weight Capacity (kg)']:
        ll = [m for m in list(train[f].unique()) if m not in list(origin[f].unique())]
        if len(ll)>0: print(f"{f} : there are {ll} in train but not origin")
        ll = [m for m in list(origin[f].unique()) if m not in list(train[f].unique())]
        if len(ll)>0: print(f"{f} : there are {ll} in origin but not train")


origin["Compartments"].isna().sum()


mask = ~origin["Compartments"].isna()

df_temp = pd.concat([origin.loc[mask], train], axis = 0)
df_temp["is_origin"] = 0
df_temp.loc[origin.loc[mask].index, "is_origin"] = 1
display(df_temp["is_origin"].value_counts())

X_trn, X_val, y_trn, y_val = train_test_split(df_temp[test.columns], df_temp["is_origin"], test_size = .2, random_state = 0)
my_et = make_pipeline(
    SimpleImputer(strategy = 'most_frequent').set_output(transform = "pandas"),
    ColumnTransformer(transformers = [
            ("label_encoder", OrdinalEncoder(**{"handle_unknown":"use_encoded_value", 'unknown_value' : -1, 'encoded_missing_value':-1}), 
             [f for f in test.columns if f not in ['Weight Capacity (kg)']])
        ], remainder = "passthrough", verbose_feature_names_out = False).set_output(transform = "pandas"),
    ExtraTreesClassifier(n_estimators = 300, random_state = 666, n_jobs = -1, max_depth = 8)
).fit(X_trn, y_trn)
print(f"AUC : {roc_auc_score(y_val, my_et.predict_proba(X_val)[:, 1]):.2f}")


df_temp = pd.concat([train_extra.sample(frac=.10, random_state=0), train], axis = 0)
df_temp["is_extra"] = 0
df_temp.loc[train_extra.sample(frac=.10, random_state=0).index, "is_extra"] = 1
display(df_temp["is_extra"].value_counts())

X_trn, X_val, y_trn, y_val = train_test_split(df_temp[test.columns], df_temp["is_extra"], test_size = .2, random_state = 0)
my_et = make_pipeline(
    SimpleImputer(strategy = 'most_frequent').set_output(transform = "pandas"),
    ColumnTransformer(transformers = [
            ("label_encoder", OrdinalEncoder(**{"handle_unknown":"use_encoded_value", 'unknown_value' : -1, 'encoded_missing_value':-1}), 
             [f for f in test.columns if f not in ['Weight Capacity (kg)']])
        ], remainder = "passthrough", verbose_feature_names_out = False).set_output(transform = "pandas"),
    ExtraTreesClassifier(n_estimators = 300, random_state = 666, n_jobs = -1, max_depth = 8)
).fit(X_trn, y_trn)
print(f"AUC : {roc_auc_score(y_val, my_et.predict_proba(X_val)[:, 1]):.2f}")




