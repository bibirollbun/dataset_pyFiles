import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm.notebook import tqdm

import os
base_path = "/kaggle/input/playground-series-s5e9"
TARGET = 'BeatsPerMinute'
df = pd.read_csv(os.path.join(base_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))
df_sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
df_original = pd.read_csv(r"/kaggle/input/bpm-prediction-challenge/Train.csv")

import warnings
warnings.filterwarnings('ignore')
%load_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
SNS_CMAP = 'GnBu'
sns.set_palette(SNS_CMAP)
# plt.style.use("dark_background")
# plt.rcParams['grid.color'] = '#444444'
colors = sns.palettes.color_palette(SNS_CMAP)
BIN_SNS_CMAP = [colors[1], colors[4]]
pd.options.mode.chained_assignment = None

def clrd(text: str, color: str = None, con: bool = None, c1:str = 'ok', c2:str = 'error')->str:
    text = str(text)
    color_codes = {
        'ok': '\033[1;92m',
        'error': '\033[91m',
        'warning': '\033[93m',
        'success': '\033[92m',
        'status': '\033[95m',
        'special': '\033[94m',
        'log': '\033[96m',
        'reset': '\033[0m',
    }
    if con is not None:
        color = c1 if con else c2
    color_code = color_codes.get(color, color_codes['reset'])
    return f"{color_code}{text}{color_codes['reset']}"
    
def df_head_regression(df, target_col=None, palette="viridis", n=5, alpha=0.5):
    """
    Highlights rows based on a continuous target column using gradient background colors.

    Parameters:
    - df: DataFrame to style
    - target_col: column name to base row color on (default: last column)
    - palette: Seaborn/Matplotlib palette name or list of colors
    - n: number of rows to display
    - alpha: transparency level (0 = fully transparent, 1 = opaque)
    """
    if target_col is None:
        target_col = df.columns[-1]

    df_show = df.head(n)

    # Normalize target values between 0 and 1
    target_vals = df_show[target_col].astype(float)
    norm = (target_vals - target_vals.min()) / (target_vals.max() - target_vals.min() + 1e-9)

    # Get color palette
    cmap = plt.get_cmap(palette)

    def row_style(row):
        val = row[target_col]
        # Normalize this rowâ€™s target value
        val_norm = (val - target_vals.min()) / (target_vals.max() - target_vals.min() + 1e-9)
        # Get RGBA
        r, g, b, _ = cmap(val_norm)
        rgba = (int(r * 255), int(g * 255), int(b * 255), alpha)
        return [f'background-color: rgba{rgba}'] * len(row)

    return df_show.style.apply(row_style, axis=1)

df_head_regression(df, TARGET, SNS_CMAP, n=8)


agg_df = df.agg(["nunique", "unique", lambda x:x.isna().sum(), "dtypes"]).T
agg_df['unique'] = agg_df['unique'].apply(lambda x: x if len(x)<10 else x[:10]) 
agg_df.style.apply(lambda s: [f'background-color: rgba({colors[0][0]*255}, {colors[0][1]*255}, {colors[0][2]*255}, 0.5)' if i % 2 == 0 else f'background-color: rgba({colors[5][0]*255}, {colors[5][1]*255}, {colors[5][2]*255}, 0.5)' for i in range(len(s))])


def plot_histograms_continous(
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

    df["TARGET_BINNED"] = pd.cut(df[target], bins=5)

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
            hue="TARGET_BINNED",
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
                    hue="TARGET_BINNED",
                    multiple="stack",
                    palette=palette,
                    ax=ax
                )
            else:
               sns.histplot(
                    data=df,
                    x=col,
                    bins=20,
                    hue="TARGET_BINNED",
                    multiple="stack",
                    palette=palette,
                    ax=ax
                )
            ax.set_title(f"{col} vs {target}")
            if ax.get_legend(): ax.get_legend().remove()
        plot_row += 1

    plt.tight_layout()
    plt.show()

# df = df.drop(['id'], axis=1)
plot_histograms_continous(df, TARGET, palette=SNS_CMAP, ncols=3)


def plot_numerical_distributions(df: pd.DataFrame,
                                 target: str,
                                 num_cols: list[str]=None,
                                 drop_cols:list[str]=None,
                                 ):
    def is_categorical(col: str, cat_unique_cutoff:int = 15) -> bool:
        return (
            df[col].dtype == "object" or
            df[col].dtype.name == "string" or
            df[col].nunique() <= cat_unique_cutoff
        )
    
    df = df.copy()
    if num_cols is None:
        num_cols = [x for x in df.columns if not is_categorical(x)]
    if drop_cols is not None:
        num_cols = [col for col in num_cols if col not in drop_cols]
    fig, ax = plt.subplots(len(num_cols), 1, figsize=(15, 6*len(num_cols)))
    for i, col in enumerate(num_cols):
        try:
            #do not use use np.ceil for float cols
            if df[col].dtype == 'float64':
                df[col] = pd.cut(df[col], bins=30).apply(lambda x: x.right if pd.notnull(x) else np.nan)
            else:
                df[col] = pd.cut(df[col], bins=30).apply(lambda x: np.ceil(x.right) if pd.notnull(x) else np.nan)
            sns.boxplot(data=df, x=col, y=target, palette=SNS_CMAP, ax=ax[i])
            num_ticks = max(1, len(df[col].unique())//10)
            x_ticks = ax[i].get_xticks()
            ax[i].set_xticks(x_ticks[::num_ticks])
            ax[i].set_xticklabels([int(tick) for tick in x_ticks[::num_ticks]])
        except Exception as e:
            print(f"{i}: exception for column {col}")
            raise e
    
    plt.tight_layout()
    fig.suptitle('Distribution of Target vs Numerical Features')
    fig.subplots_adjust(top=0.97) 
    plt.show()

plot_numerical_distributions(df, TARGET, drop_cols=[TARGET])


from sklearn.preprocessing import OrdinalEncoder

numerics = [bool, int, 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
pp_df = df.select_dtypes(include=numerics)
plt.figure(figsize=(15, 15))
sns.heatmap(pp_df.corr(method="kendall"), cmap=SNS_CMAP, annot=True, annot_kws={'fontsize':7}, fmt='.1g', vmin=-1, vmax=1, center= 0)
plt.title("Feature Correlation")
plt.show()


sns.pairplot(df[['VocalContent', 'LivePerformanceLikelihood', 'MoodScore', 'Energy']].head(5000), kind="hist", corner=True)
plt.show()


def removeOutliers(df, col, threshold=0.005):
    """
    Removes outliers from a DataFrame based on numerical IQR or
    low-frequency categorical values.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    col (str): Column to base outlier removal on.
    threshold (float): Minimum frequency proportion to retain categorical values.
        Default is 0.001 (0.1%).
    
    Returns:
    DataFrame: The filtered DataFrame.
    """
    
    if pd.api.types.is_numeric_dtype(df[col]):
        # Calculate IQR for the specified column
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
        # For categorical data, remove low-frequency categories
        freq = df[col].value_counts(normalize=True)
        common_categories = freq[freq >= threshold].index
        return df[df[col].isin(common_categories)]
    return df

def plot_violins_cont(df: pd.DataFrame, target: str = "TARGET",
                           cols:list[str]=None, ncols: int = 3, outliers: bool = False)->None:
    """
    Plot a series of histograms for a binary target
    
    Parameters: 
        df (Dataframe) : data to be visualized
        *params : any
    """
    def is_categorical(col: str, cat_unique_cutoff:int = 15) -> bool:
        return (
            df[col].dtype == "object" or
            df[col].dtype.name == "string" or
            df[col].nunique() <= cat_unique_cutoff
        )
    
    df = df.copy()
    if cols is None:
        cols = [x for x in df.columns if not is_categorical(x)]

    nrows = (len(cols)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    for i, col in enumerate(cols):
        if outliers:
            data = removeOutliers(df, col)
        sns.violinplot(df_cat, y=col, x='_source', ax=axes.flatten()[i])
    plt.show()

df['_source'] = 'train'
df_test['_source'] = 'test'
df_test[TARGET] = 0
df_original['_source'] = 'original'
df_cat = pd.concat([df, df_original, df_test])
df_cat = df_cat.drop(['id'], axis=1)

plot_violins_cont(df_cat, 
                       ncols=2,
                       outliers=True)


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor, StackingRegressor, VotingRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
if torch.cuda.is_available():
    from cuml.svm import SVR                
else:
    from sklearn.svm import SVR

from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
import copy
import os

def train_and_evaluate_regression_model(model, X, y, X_test=None, cv=5, name=None, model_name=None,
                                       retrain:bool = True, verbose:bool = True):
    """
    Train and evaluate a regression model using cross-validation.

    Args:
        model: The model to train and evaluate.
        X: Features for training and evaluation.
        y: Target labels.
        X_test: Optional test set for predictions after cross-validation.
        cv: Number of cross-validation folds.
        name: Optional name of the model for display.

    Returns:
        metrics: Dictionary containing metrics for all folds.
        oof_preds: Out-of-fold predictions for training data.
        test_preds: Predictions on X_test if provided.
    """
    folds = KFold(n_splits=cv, shuffle=True, random_state=42).split(X, y)

    metrics = {
        'rmse': [],
        'mae': [],
        'r2': [],
    }
    
    if model_name is None:
        if type(model) is Pipeline:
            model_name = model[-1].__class__.__name__
        else:
            model_name = model.__class__.__name__
    if name is None:
        name = model_name

    print(f"{name} ( {clrd(model_name, 'status')} )")
    for train_index, test_index in tqdm(folds, desc=f"Fitting {cv} folds on {model_name}"):
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

        metrics['rmse'].append(np.sqrt(mean_squared_error(y_valid, y_pred))  )
        metrics['mae'].append(mean_absolute_error(y_valid, y_pred)  )
        metrics['r2'].append(r2_score(y_valid, y_pred)  )

    for k, v in metrics.items():
        metrics[k] = np.mean(v)

    if retrain:
        model.fit(X, y)
    if verbose:
        print(f"{clrd('rmse ', 'ok')} : {  metrics['rmse']:.4f}   {clrd('mae', 'log')}: {metrics['mae']:.4f}   {clrd('r2', 'log')}: {metrics['r2']:.4f}")
        print('-'*50)

    return metrics

models = {}


models.update({
    "linear": LinearRegression(),
    "histgrad": HistGradientBoostingRegressor(),
    "lgbm": LGBMRegressor(verbose=0),
    "xgb": XGBRegressor(verbosity=0,),
    "cb": CatBoostRegressor(verbose=False,),
})

for model_name, model in models.items():
    _ = train_and_evaluate_regression_model(model, df.drop(['id', TARGET], axis=1), df[TARGET], cv=5)


# !pip install -q -U "pytorch_tabular[extra]"
#doesn't work as of `7/9/25` because the fix to torch-tabular addressing torch>2.6's new .load changes has not yet been added to the pypi release  
#instead clone the main branch directly for torch>2.6

!pip install -q git+https://github.com/manujosephv/pytorch_tabular.git


# !pip install -q -U "rich[jupyter]"
from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
    CategoryEmbeddingModelConfig,
    FTTransformerConfig,
    TabNetModelConfig,
    GANDALFConfig,
)
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular import model_sweep


num_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']

data_config = DataConfig(
    target=[TARGET],
    continuous_cols=num_cols,
    normalize_continuous_features=True,
    continuous_feature_transform="quantile_normal",
    # num_workers=10,
    # pin_memory=True
    #wandb stalls with pin_memory :)
)

trainer_config = TrainerConfig(
    auto_lr_find=True,
    batch_size=2048,
    max_epochs=30,
    accelerator='gpu',
    
    # fast_dev_run = True
)

optimizer_config = OptimizerConfig(
    optimizer="Adam",
    lr_scheduler="StepLR",
    lr_scheduler_params={"step_size": 10}
)
sweep_optimizer_config = OptimizerConfig()

head_config = LinearHeadConfig(
    layers="",
    dropout=0.1,
    initialization="kaiming",
).__dict__

model_config = CategoryEmbeddingModelConfig(
    task="regression",
    layers="512-256-64",
    activation="LeakyReLU",
    learning_rate=1e-4,
    target_range = [(0, 250)],
    initialization = "kaiming",
    use_batch_norm = False,
    
    head="LinearHead",
    head_config=head_config,
    
    metrics = ['mean_squared_error', 'r2_score'],
)

experiment_config = ExperimentConfig(
    project_name="s5e9",
    run_name="CategoryEmbeddingModel",
    # exp_watch="gradients",
    log_target="wandb",
    # log_logits=True,
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    experiment_config=experiment_config,
    verbose=False,
    suppress_lightning_logger=True,
)


from kaggle_secrets import UserSecretsClient
import wandb

wandb_key = UserSecretsClient().get_secret("WANDB")
wandb.login(key=wandb_key)


from rich import print as rprint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def _rmse(y_true, y_pred):
    metric = np.sqrt(mean_squared_error(y_true, y_pred["prediction"].values))
    
train, val = train_test_split(
    df, test_size=0.05, random_state=42, shuffle=True
)
rprint(df.shape, train.shape, val.shape)


#1. fit
tabular_model.fit(train=train, validation=val)
result = tabular_model.evaluate(val)

#2. cross-validate
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     cv_scores, oof_predictions = tabular_model.cross_validate(
#         cv=5, train=df, return_oof = True, reset_datamodule = False)

# rprint(f"cross val rmse: {cv_scores}")


wandb.finish()
rprint(result)


# pred = pd.concat(oof_predictions)
pred = tabular_model.predict(df_test)
rprint(pred.shape)


%%time
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sweep_df, best_model = model_sweep(
        task="regression",
        train=train,
        test=val,
        data_config=data_config,
        optimizer_config=sweep_optimizer_config,
        trainer_config=trainer_config,
        model_list="standard",
        progress_bar=True,
        verbose=True,
        suppress_lightning_logger=False,
    )


sweep_df.drop(columns=["params", "time_taken", "epochs"]).style.background_gradient(
    subset=["test_mean_squared_error", "test_loss"], cmap=SNS_CMAP
).background_gradient(subset=["time_taken_per_epoch", "test_loss"], cmap=SNS_CMAP)


from pytorch_tabular.tabular_model_tuner import TabularModelTuner

search_space = {
    "model_config__layers": ["512-256-64", "1024-512-256", "1024-512-64"],
    "model_config__use_batch_norm": [True, False],
    
    "model_config.head_config__dropout": [0.1, 0.2, 0.3],
    "model_config.head_config__activation": ['kaiming', 'xavier'],

    "optimizer_config__optimizer": ["RAdam", "AdamW"],
    
}

tuner = TabularModelTuner(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config
)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = tuner.tune(
        train=train,
        validation=val,
        search_space=search_space,
        # strategy="random_search",
        strategy="grid_search",
        cv=3,
        metric="mean_squared_error",
        mode="min",
        progress_bar=True,
        verbose=False
    )


df_sub = df_sample.copy()
df_sub[TARGET] = pred
df_sub.to_csv('submission.csv', index=False)

