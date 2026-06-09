SUBMISSION = True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm.notebook import tqdm
import polars as pl
import os
import gc

#------------
#--- data ---
#------------
base_path = "/kaggle/input/mercor-cheating-detection"
TARGET = 'is_cheating'
df = pd.read_csv(os.path.join(base_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))
df_graph = pd.read_csv(os.path.join(base_path, 'social_graph.csv'))
df_sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))


#-----------------
#--- cosmetics ---
#-----------------
SNS_CMAP = 'bwr'
from rich.console import Console
rc = Console(force_jupyter=False, color_system="truecolor")

def rprint(*texts: str, sep: str = ' ') -> None:
    """Print like built-in print but through rc"""
    rc.print(sep.join(str(t) for t in texts))

import warnings
warnings.filterwarnings('ignore')
%load_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
sns.set_palette(SNS_CMAP)
plt.style.use("dark_background")
plt.rcParams['grid.color'] = '#444444'
colors = sns.palettes.color_palette(SNS_CMAP)
pd.options.mode.chained_assignment = None
BIN_SNS_CMAP = [colors[1], colors[4]]
    
def df_head_binary(df, target_col=None, palette="Set2", n=5, alpha=0.5):
    """
    Highlights rows based on a binary target column using transparent background colors.

    Parameters:
    - df: DataFrame to style
    - target_col: column name to base row color on (default: last column)
    - palette: Seaborn palette name or list of two RGB colors
    - n: number of rows to display
    - alpha: transparency level (0 = fully transparent, 1 = opaque)
    """
    if target_col is None:
        target_col = df.columns[-1]

    df_show = df.head(n)

    # Get RGB colors and add alpha
    palette_colors = sns.color_palette(palette, 2)
    rgba_0 = tuple(int(c * 255) for c in palette_colors[0]) + (alpha,)
    rgba_1 = tuple(int(c * 255) for c in palette_colors[1]) + (alpha,)

    def row_style(row):
        target = row[target_col]
        rgba = rgba_0 if target == 0 else rgba_1
        return [f'background-color: rgba{rgba}'] * len(row)

    return df_show.style.apply(row_style, axis=1)

df_head_binary(df[~df[TARGET].isna()], TARGET, SNS_CMAP, n=8)


if not SUBMISSION:
    agg_df = df.agg(["nunique", "unique", lambda x:x.isna().sum(), "dtypes"]).T
    agg_df['unique'] = agg_df['unique'].apply(lambda x: x if len(x)<10 else x[:10]) 
    agg_df.style.apply(lambda s: [f'background-color: rgba({colors[0][0]*255}, {colors[0][1]*255}, {colors[0][2]*255}, 0.3)' if i % 2 == 0 else f'background-color: rgba({colors[5][0]*255}, {colors[5][1]*255}, {colors[5][2]*255}, 0.3)' for i in range(len(s))])


def plot_histograms_binary(
    df: pd.DataFrame,
    target: str,
    cols: list[str] = None,
    ncols: int = 2,
    unique_threshold: int = 7,
    cat_unique_cutoff: int = 10,
    palette: str = "Set2"
) -> None:
    """
    Plots histograms of features with hue as binned target.

    Parameters:
    - df: pd.DataFrame â€“ the dataset
    - target: str â€“ the target variable used for binning
    - cols: list[str], optional â€“ list of columns to plot; if None, uses all columns
    - ncols: int â€“ number of plots per row (for regular features)
    - unique_threshold: int â€“ categorical features with > this many uniques get a full row
    - cat_unique_cutoff: int â€“ max unique values to consider a column categorical
    - palette: str â€“ seaborn color palette

    Returns:
    - None
    """
    df = df.copy()
    if cols is None:
        cols = df.columns.tolist()

    def is_categorical(col: str) -> bool:
        return (
            df[col].dtype == "object" or
            df[col].dtype.name == "string" or
            df[col].nunique() <= cat_unique_cutoff
        )

    # Only categorical features with > threshold unique values get full row
    full_row_cols = [
        col for col in cols
        if is_categorical(col) and df[col].nunique() > unique_threshold
    ]
    regular_cols = [col for col in cols if col not in full_row_cols]

    total_two_col_rows = (len(regular_cols) + (ncols - 1)) // ncols
    total_rows = len(full_row_cols) + total_two_col_rows

    fig = plt.figure(figsize=(7 * ncols, 4 * total_rows))
    gs = GridSpec(total_rows, ncols, figure=fig)

    plot_row = 0

    # Full-width plots
    for col in full_row_cols:
        ax = fig.add_subplot(gs[plot_row, :])
        sns.histplot(
            data=df,
            x=col,
            discrete=True,
            hue=target,
            multiple="stack",
            shrink=0.8,
            palette=palette,
            ax=ax
        )
        ax.set_title(f"{col} vs {target}")
        if ax.get_legend(): ax.get_legend().remove()
        plot_row += 1

    # Remaining features 2 per row
    for i in range(0, len(regular_cols), ncols):
        for j in range(ncols):
            if i + j >= len(regular_cols):
                break
            col = regular_cols[i + j]
            ax = fig.add_subplot(gs[plot_row, j])
            if is_categorical(col):
                sns.histplot(
                    data=df,
                    x=col,
                    bins=None,
                    discrete=True,
                    shrink=0.8,
                    hue=target,
                    multiple="stack",
                    palette=palette,
                    ax=ax
                )
            else:
               sns.histplot(
                    data=df,
                    x=col,
                    bins=20,
                    hue=target,
                    multiple="stack",
                    palette=palette,
                    ax=ax
                )
            ax.set_title(f"{col} vs {target}")
            if ax.get_legend(): ax.get_legend().remove()
        plot_row += 1

    plt.tight_layout()
    plt.show()

if not SUBMISSION:
    
    df.loc[df['high_conf_clean'].isna(), 'high_conf_clean'] = 0
    plot_histograms_binary(
        df.drop(['user_hash'], axis=1),
        target=TARGET,
        palette=BIN_SNS_CMAP,
        ncols=3
    )


def join_graph(df, df_graph):
    df_merge = df_graph.merge(df[["user_hash", TARGET]], left_on="user_b", right_on="user_hash")
    
    df_merge = df_merge.groupby("user_a")[TARGET].agg(
    nbr_count = "count",
    nbr_sum = "sum",
    nbr_mean = "mean").reset_index()
    
    return df.merge(df_merge, left_on="user_hash", right_on="user_a")

if not SUBMISSION:
    
    df = join_graph(df)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    sns.violinplot(df[df["nbr_count"]<10], y="nbr_count", x=TARGET, palette=SNS_CMAP, ax=ax[0])
    sns.violinplot(df, y="nbr_mean", x=TARGET, palette=SNS_CMAP, ax=ax[1])
    fig.suptitle("Connected Users on graph (1-Hop)")
    plt.show()


%%time
from typing import cast

FEATURE_COLS = [f"feature_{i:03}" for i in range(1, 19)]
OHE_COLS = [f"_ohe_{feat}" for feat in FEATURE_COLS if len(df[feat].unique()) <= 12]
NAN_COLS = [f'_is_na_feature_{idx:03}' for idx in range(1, 19)]

rprint(f"feature: {FEATURE_COLS}")
rprint(f"ohe-features: {OHE_COLS}")
rc.rule()
def base_preproc(df:pd.DataFrame,
                   df_test:pd.DataFrame,
                   df_graph:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]:
    train_users:set = set(df['user_hash'])
    test_users:set = set(df_test['user_hash'])
    users:set = train_users.union(test_users)
    labelled_users:set = set(df[~df[TARGET].isna()]['user_hash'])
    unlabelled_users: set = train_users - labelled_users
    
    rprint(f"Found {len(labelled_users)} labelled train users")
    rprint(f"Found {len(unlabelled_users)} unlabelled train users")
    total_edges = len(df_graph)
    df_graph = df_graph[(df_graph["user_a"].isin(users)) | (df_graph["user_b"].isin(users))]
    rprint(f"Found One-Hop graph edges to train/test users: {len(df_graph)}/{total_edges}")
    df_graph:pd.DataFrame = pd.concat(
        cast(list[pd.DataFrame],([
            df_graph[['user_a', 'user_b']].rename(columns={'user_a': 'src', 'user_b': 'dst'}),
            df_graph[['user_b', 'user_a']].rename(columns={'user_b': 'src', 'user_a': 'dst'}),
        ])),
        ignore_index=True
    ).drop_duplicates()

    edges = set(df_graph["src"])
    rprint(f"Found {df['user_hash'].isin(edges).sum()}/{len(train_users)} [green]TRAIN[/] users in graph")
    rprint(f"Found {df_test['user_hash'].isin(edges).sum()}/{len(test_users)} [yellow]TEST[/] users in graph")
    labelled_to_labelled = df_graph[df_graph["src"].isin(labelled_users) & df_graph["dst"].isin(labelled_users)]["src"].nunique()
    unlabelled_to_labelled = df_graph[df_graph["src"].isin(unlabelled_users) & df_graph["dst"].isin(labelled_users)]["src"].nunique()
    train_users_matching_labelled_user = df_graph[df_graph["src"].isin(train_users)&df_graph["dst"].isin(labelled_users)]["src"].nunique()
    train_users_matching_train_user = df_graph[df_graph["src"].isin(train_users)&df_graph["dst"].isin(train_users)]["src"].nunique()
    test_users_matching_labelled_train_user  = df_graph[df_graph["src"].isin(test_users)&df_graph["dst"].isin(labelled_users)]["src"].nunique()
    test_users_matching_train_user = df_graph[df_graph["src"].isin(test_users)&df_graph["dst"].isin(train_users)]["src"].nunique()
    test_users_matching_test_user = df_graph[df_graph["src"].isin(test_users)&df_graph["dst"].isin(test_users)]["src"].nunique()
    rprint(f"Found {labelled_to_labelled}/{len(labelled_users)} ",
           f"[purple]LABELLED[/] [green]train[/] users matching a [purple]LABELLED[/] [green]train[/] user")
    rprint(f"Found {unlabelled_to_labelled}/{len(unlabelled_users)} ",
           f"[dim]unlabelled[/] [green]train[/] users matching a [purple]LABELLED[/] [green]train[/] user")
    rprint(f"Found {train_users_matching_labelled_user}/{len(train_users)}",
           f"[green]train[/] users matching a [purple]LABELLED[/] [green]train[/] user")
    rprint(f"Found {train_users_matching_train_user}/{len(train_users)}",
           f"[green]train[/] users matching a [green]train[/] user")
    rprint(f"Found {test_users_matching_labelled_train_user}/{len(test_users)}",
           f"[yellow]test[/] users matching a [purple]LABELLED[/] [green]train[/] user")
    rprint(f"Found {test_users_matching_train_user}/{len(test_users)}",
           f"[yellow]test[/] users matching a [green]train[/] user")
    rprint(f"Found {test_users_matching_test_user}/{len(test_users)}",
           f"[yellow]test[/] users matching a [yellow]test[/] user")
    
    nbr_count_all = (df_graph
        .groupby("src")["src"]
        .agg(nbr_count_all="count")
        .reset_index())

    nbr_labeled = (df_graph
        .merge(
            df[["user_hash", TARGET]],
            left_on="dst",
            right_on="user_hash",
            how="inner"
        )
        .groupby("src")[TARGET]
        .agg(
            nbr_count_labeled="count",
            nbr_sum="sum",
            nbr_mean="mean"
        )
        .reset_index())
    
    df = df.merge(nbr_labeled, left_on="user_hash", right_on="src", how="left")
    df = df.merge(nbr_count_all, left_on="user_hash", right_on="src", how="left")
    df_test = df_test.merge(nbr_labeled, left_on="user_hash", right_on="src", how="left")
    df_test = df_test.merge(nbr_count_all, left_on="user_hash", right_on="src", how="left")

    for feat_idx in range(1, 19):
        df[f"_is_na_feature_{feat_idx:03}"] = df[f"feature_{feat_idx:03}"].isna().astype(int)
        df_test[f"_is_na_feature_{feat_idx:03}"] = df_test[f"feature_{feat_idx:03}"].isna().astype(int)

    for ohe_feat in OHE_COLS:
        df[ohe_feat] = df[ohe_feat[5:]].astype(str)
        df_test[ohe_feat] = df_test[ohe_feat[5:]].astype(str)
        if len(df[ohe_feat].unique())>50:
            rprint(f"[red]FOUND {len(df[ohe_feat].unique())} values in train col to ohe!!![/]")
        if len(df_test[ohe_feat].unique())>50:
            rprint(f"[red]FOUND {len(df_test[ohe_feat].unique())} values in test col to ohe!!![/]")

    df["total_nan_count"] = df[FEATURE_COLS].isna().sum(axis=1)
    df_test["total_nan_count"] = df[FEATURE_COLS].isna().sum(axis=1)
    
    df["high_conf_clean"] = df["high_conf_clean"].fillna(0)
    rprint(df.shape, df_test.shape)
    return df, df_test, df_graph


df = pd.read_csv(os.path.join(base_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))
df_graph = pd.read_csv(os.path.join(base_path, 'social_graph.csv'))
df_sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
df, df_test, df_graph = base_preproc(df, df_test, df_graph)
rprint(f"[red]removed {gc.collect()} items[/]")


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from collections import Counter


one_hop_nbr_cols = ["nbr_sum", "nbr_mean", "nbr_count_labeled", "nbr_count_all"]
num_cols = FEATURE_COLS + one_hop_nbr_cols + ["total_nan_count"]

vis_preproc = Pipeline([
    ('cat-select', ColumnTransformer(transformers=[
        ('pass', 'passthrough', num_cols+NAN_COLS),
    ], verbose_feature_names_out=False).set_output(transform='pandas')),
])

cat_preproc = Pipeline([
    ('cat-select', ColumnTransformer(transformers=[
        ('pass', 'passthrough', num_cols+NAN_COLS),
    ], verbose_feature_names_out=False).set_output(transform='pandas')),
])

linear_preproc = Pipeline([
    ('cat-select', ColumnTransformer(transformers=[
        ('pass', 'passthrough', NAN_COLS),
        ('scale', StandardScaler(), num_cols)
    ])),
    ('simp', SimpleImputer())
])

ohe_preproc = Pipeline([
    ('cat-select', ColumnTransformer(transformers=[
        ('scale', StandardScaler(), num_cols),
        ('ohe', OneHotEncoder(), OHE_COLS)
    ])),
    ('simp', SimpleImputer())
])

ft_preproc = Pipeline([
    ('cat-select', ColumnTransformer(transformers=[
        ('pass', 'passthrough', OHE_COLS+num_cols+[TARGET]),
    ], verbose_feature_names_out=False).set_output(transform='pandas')),
])

sspt_preproc = Pipeline([
    ('cat-select', ColumnTransformer(transformers=[
        ('pass', 'passthrough', OHE_COLS),
        ('impute', SimpleImputer().set_output(transform='pandas'), num_cols)
    ], verbose_feature_names_out=False).set_output(transform='pandas')),
])

#====================================
def calculate_imbalance_ratio(series):
    counter = Counter(series)
    zeros = counter[0]
    ones = counter[1]
    ratio =  max(zeros, ones) / (min(zeros, ones)+0.001)
    return ratio, counter

labelled_df = df[df["high_conf_clean"]==0]
unlabelled_df = df[df["high_conf_clean"]==1]
calculate_imbalance_ratio(labelled_df[TARGET])


TREES_ON_GPU = False

import torch
if torch.cuda.is_available() and TREES_ON_GPU:
    from cuml.svm import SVC                   
else:
    from sklearn.svm import SVC
    
import optuna
from sklearn.linear_model import LogisticRegression

if torch.cuda.is_available() and TREES_ON_GPU:
    !git clone --recursive https://github.com/Microsoft/LightGBM
    !cd LightGBM
    !sh ./build-python.sh install --cuda
else:
    from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier, RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier


import torch
if torch.cuda.is_available() and TREES_ON_GPU:
    from cuml.svm import SVC                     
else:
    from sklearn.svm import SVC
    

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, log_loss, mean_squared_error, matthews_corrcoef
import copy
from tqdm import tqdm


def train_and_evaluate_model(model, X, y, X_test=None, cv=5, name=None, model_name=None, stratify=False, 
                             retrain:bool = True, fold_iterator:object = None, verbose: bool = 1):
    """
    Train and evaluate a model using cross-validation.

    Args:
        model: The model to train and evaluate.
        X: Features for training and evaluation.
        y: Target labels.
        X_test: Optional test set for predictions after cross-validation.
        cv: Number of cross-validation folds.
        name: Optional name of the model for display.
        stratify: Whether to use stratified k-fold.

    Returns:
        metrics: Dictionary containing metrics for all folds.
    """ 
    if stratify:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42).split(X, y)
    else:
        folds = KFold(n_splits=cv, shuffle=True, random_state=42).split(X, y)

    metrics = {
        'accuracy': [],
        'f1-score': [],
        'auc-roc': [],
        'precision': [],
        'mse': [],
        'log-loss': [],
        'mcc': []
    }
    
    if model_name is None:
        if type(model) is Pipeline:
            model_name = model[-1].__class__.__name__
        else:
            model_name = model.__class__.__name__
    if name is None:
        name = model_name

    if fold_iterator is None:
        fold_iterator = folds
    if verbose:
        rprint(f"{name} ( [magenta] {model_name} [/] )")
        fold_iterator = tqdm(fold_iterator, desc=f"Evaluating {name}")
    for train_index, test_index in fold_iterator:
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        else:  
            X_train, X_valid = X[train_index], X[test_index]
        if isinstance(y, pd.Series):
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        else:  
            y_train, y_valid = y[train_index], y[test_index]

        model_fold = copy.deepcopy(model)
        model_fold.fit(X_train, y_train)

        y_pred = model_fold.predict(X_valid)
        try:
            y_pred_proba = model_fold.predict_proba(X_valid)[:, 1] if hasattr(model_fold, "predict_proba") else y_pred
        except:
            print(clrd('Unable to predict probas', 'warning'))
            y_pred_proba = y_pred*0

        metrics['accuracy'].append(accuracy_score(y_valid, y_pred))
        metrics['f1-score'].append(f1_score(y_valid, y_pred))
        metrics['precision'].append(precision_score(y_valid, y_pred))
        metrics['auc-roc'].append(roc_auc_score(y_valid, y_pred_proba))
        metrics['mse'].append(mean_squared_error(y_valid, y_pred_proba))
        metrics['log-loss'].append(log_loss(y_valid, y_pred_proba))
        metrics['mcc'].append(matthews_corrcoef(y_valid, y_pred))
    
    for k, v in metrics.items():
        metrics[k] = np.mean(v)

    if retrain:
        model.fit(X, y)
    if verbose:
        rprint(f"[green]accuracy[/] : {metrics['accuracy']:.4f}  f1-score: {metrics['f1-score']:.4f}   [green]auc-roc:[/] [bold green]{metrics['auc-roc']:.4f}[/]")
        rprint(f"precision: {metrics['precision']:.4f}  mcc     : {metrics['mcc']:.4f}   mse    : {metrics['mse']:.4f}")
        print('-'*50)

    return metrics


models = {
    "linear1": make_pipeline(linear_preproc, LogisticRegression()),
    "lgbm1": make_pipeline(cat_preproc,
                     LGBMClassifier(
                         verbose = 0,
                         objective = 'binary',
                     )),
    "hist1": make_pipeline(cat_preproc, HistGradientBoostingClassifier()),
    # "rf1": make_pipeline(cat_preproc, RandomForestClassifier()),
    "xgb1": make_pipeline(cat_preproc,
                         XGBClassifier(
                            verbosity=0,
                            objective='binary:logistic',
                            eval_metric="auc",
                         )),
}


res = {}
for model_name, model in models.items():
    metrics = train_and_evaluate_model(model, df[df["high_conf_clean"]==0], df[df["high_conf_clean"]==0][TARGET], stratify=True, cv=5, name = model_name)
    res[model_name] = metrics


import shap

labelled_df = df[df["high_conf_clean"]==0]
vis_preproc.fit(labelled_df)
X = vis_preproc.transform(labelled_df)
vis_model = XGBClassifier(
                            verbosity=0,
                            objective='binary:logistic',
                            eval_metric="auc",
                         )
vis_model.fit(X, labelled_df[TARGET])
explainer = shap.TreeExplainer(vis_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, max_display=50, show=False)
plt.gca().tick_params(axis='y', colors='white')
plt.gca().tick_params(axis='x', colors='white')
plt.show()


!pip install -q git+https://github.com/manujosephv/pytorch_tabular.git

from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
    CategoryEmbeddingModelConfig,
    # FTTransformerConfig,
    # TabNetModelConfig,
    # GANDALFConfig,
)
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular import model_sweep
from pytorch_tabular.ssl_models.dae import DenoisingAutoEncoderConfig

from sklearn.model_selection import train_test_split


from kaggle_secrets import UserSecretsClient
import wandb

wandb_key = UserSecretsClient().get_secret("WANDB")
wandb.login(key=wandb_key)


from sklearn.preprocessing import StandardScaler

def preprocess_for_torch_tabular(df: pd.DataFrame,
                                  df_test: pd.DataFrame,
                                  feature_cols: list[str],
                                  ohe_cols: list[str],
                                  nan_cols: list[str],
                                  graph_feat_cols: list[str],
                                  target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    
    df["labelled"] = (~(df[TARGET].isna())).astype(int)
    # === 1. ONE-HOT ENCODING ===
    new_ohe_cols = []
    
    for col in ohe_cols:
        train_categories = sorted(df[col].dropna().unique())
        rprint(f"  OHE [{col}]: {len(train_categories)} categories")
        
        for category in train_categories:
            new_col_name = f"{col}_{category}"
            new_ohe_cols.append(new_col_name)
            
            df[new_col_name] = (df[col] == category).astype(int)
            df_test[new_col_name] = (df_test[col] == category).astype(int)
        
        df = df.drop(columns=[col])
        df_test = df_test.drop(columns=[col])
    
    rprint(f"  Created {len(new_ohe_cols)} OHE columns")
    
    # === 2. SCALE FEATURE COLUMNS ===
    scaler = StandardScaler()
    
    # Fit on train, transform both
    df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].fillna(0))
    
    rprint(f"  Scaled {len(feature_cols)} feature columns")
    
    # === 3. FILL ALL REMAINING NaNs (EXCEPT TARGET) ===
    
    # Get all columns except target
    cols_to_fill = [col for col in df.columns if col != target_col]
    
    df[cols_to_fill] = df[cols_to_fill].fillna(0)
    df_test = df_test.fillna(0)  # Test has no target, fill everything
        
    # === 4. ASSEMBLE FEATURE LIST ===
    tab_features = graph_feat_cols + feature_cols + new_ohe_cols + nan_cols
    
    rprint(f"  Total features for torch-tabular: {len(tab_features)}")
    rprint(f"    - Graph features: {len(graph_feat_cols)}")
    rprint(f"    - Scaled features: {len(feature_cols)}")
    rprint(f"    - OHE features: {len(new_ohe_cols)}")
    rprint(f"    - NaN indicators: {len(nan_cols)}")
    
    # Check label preservation
    nan_labels = df[target_col].isna().sum()
    
    return df, df_test, tab_features


# === USAGE ===

GRAPH_FEAT_COLS = ["nbr_count_labeled", "nbr_sum", "nbr_mean", "nbr_count_all", "total_nan_count"]

# Run preprocessing
df, df_test, tab_features = preprocess_for_torch_tabular(
    df=df, 
    df_test=df_test,
    feature_cols=FEATURE_COLS,
    ohe_cols=OHE_COLS,
    nan_cols=NAN_COLS,
    graph_feat_cols=GRAPH_FEAT_COLS,
    target_col=TARGET
)

# Split labeled/unlabeled
labelled_df = df[df["labelled"]==1]
unlabelled_df = df[df["labelled"]==0]
labelled_df[TARGET] = labelled_df[TARGET].astype(int)

# Training config
batch_size = 512
steps_per_epoch = int(unlabelled_df.shape[0] / batch_size)
epochs = 25

rprint(f"\n[bold]Training Configuration:[/bold]")
rprint(f"  Labeled samples: {labelled_df.shape[0]}")
rprint(f"  Unlabeled samples: {unlabelled_df.shape[0]}")
rprint(f"  Batch size: {batch_size}")
rprint(f"  Steps per epoch: {steps_per_epoch}")
rprint(f"  Epochs: {epochs}")
rprint(f"  Feature columns: {len(tab_features)}")


ssl_data_config = DataConfig(
    target=None,
    continuous_cols=tab_features,
    # categorical_cols=cat_cols,
    normalize_continuous_features=False,
    handle_missing_values=False, #ssl
    handle_unknown_categories=False, #ssl
)

ssl_trainer_config = TrainerConfig(
    batch_size=batch_size,
    max_epochs=epochs,
    early_stopping="valid_loss", # Turning off Early Stopping
    checkpoints="valid_loss", # Save best checkpoint monitoring val_loss
    load_best=True, # After training, load the best checkpoint
)

# Setting OneCycleLR schedule
ssl_optimizer_config = OptimizerConfig(
    lr_scheduler="OneCycleLR",
    lr_scheduler_params={
        "max_lr":1e-2, 
        "epochs": epochs, 
        "steps_per_epoch":steps_per_epoch
    }
)

# Setting the encoder config
encoder_config = CategoryEmbeddingModelConfig(
    task="backbone",
    layers="4096-2048-1024-512",
    activation="ReLU",
    head=None, #ssl
)

# Setting the decoder config.
# NOTE: the last dimension in encoder layers should be first dimension in decoder layers
# i.e. last encoder layer dim = 512, first decoder layer dim = 512
decoder_config = CategoryEmbeddingModelConfig(
    task="backbone",
    layers="512-2048-4096",
    activation="ReLU",
    head=None, #ssl
)

# DAE Config. No need to set task because it is hardcoded to SSL
# Can't set any loss or metrics as well because for the SSL task
# (especially for DAE), the loss and metrics are fixed.
ssl_model_config = DenoisingAutoEncoderConfig(
    # noise_strategy="zero",
    noise_strategy="swap",
    default_noise_probability = 0.7,
    include_input_features_inference=True,
    encoder_config=encoder_config,
    decoder_config=decoder_config,
    learning_rate=1e-3)

experiment_config = ExperimentConfig(
    project_name="mercor_cheating",
    run_name="ssl_dae",
    # exp_watch="gradients",
    log_target="wandb",
    # log_logits=True,
)

ssl_tabular_model = TabularModel(
    data_config=ssl_data_config,
    model_config=ssl_model_config,
    optimizer_config=ssl_optimizer_config,
    trainer_config=ssl_trainer_config,
    experiment_config=experiment_config,
    verbose=False
)


ssl_train, ssl_val = train_test_split(unlabelled_df, test_size=0.2)

ssl_tabular_model.pretrain(train=ssl_train, validation=ssl_val)


!pip install -q torch_optimizer


from torch_optimizer import QHAdam
ft_train, ft_val = train_test_split(labelled_df, test_size=0.2)

ft_trainer_config = TrainerConfig(
    batch_size=512,
    max_epochs=50,
    early_stopping="valid_loss",
    checkpoints="valid_loss",
    load_best=True,
)

ft_optimizer_config = OptimizerConfig(
    lr_scheduler="OneCycleLR",
    lr_scheduler_params={
        "max_lr":1e-3,
        "epochs": epochs,
        "steps_per_epoch":steps_per_epoch
    }
)

finetune_model = ssl_tabular_model.create_finetune_model(
    task="classification",
    train=ft_train,
    validation=ft_val,
    target=[TARGET],
    head="LinearHead",
    head_config={
        "layers": "256-64",
        "activation": "ReLU",
    },
    trainer_config=ft_trainer_config,
    optimizer_config=ft_optimizer_config,
    optimizer=QHAdam,
    optimizer_params={"nus": (0.7, 1.0), "betas": (0.95, 0.998)}
)


import torch

try:
    assert torch.equal(ssl_tabular_model.model.encoder.linear_layers[0].weight, finetune_model.model._backbone.encoder.linear_layers[0].weight)
except Exception as e:
    rprint(f"[red]error: {e}[/]")


finetune_model.finetune(
    freeze_backbone=True)


pred = finetune_model.predict(df_test)
rprint(pred.shape)
pred_proba = pred["is_cheating_1_probability"].values


df_sub = df_sample.copy()
df_sub[TARGET] = pred_proba
df_sub.to_csv('submission.csv', index=False)

