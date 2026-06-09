import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm.notebook import tqdm

import os
base_path = "/kaggle/input/playground-series-s5e8"
TARGET = 'y'
df = pd.read_csv(os.path.join(base_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))
df_original = pd.read_csv(r"/kaggle/input/bank-marketing-dataset-full/bank-full.csv", delimiter=";")
df_original[TARGET] = df_original[TARGET].map({
    'yes': 1,
    'no': 0
})
df_sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))

import warnings
warnings.filterwarnings('ignore')
%load_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
SNS_CMAP = 'vlag'
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

df_head_binary(df, TARGET, SNS_CMAP, n=8)



agg_df = df.agg(["nunique", "unique", lambda x:x.isna().sum(), "dtypes"]).T
agg_df['unique'] = agg_df['unique'].apply(lambda x: x if len(x)<10 else x[:10])
agg_df.style.apply(lambda s: [f'background-color: rgba({colors[0][0]*255}, {colors[0][1]*255}, {colors[0][2]*255}, 0.5)' if i % 2 == 0 else f'background-color: rgba({colors[5][0]*255}, {colors[5][1]*255}, {colors[5][2]*255}, 0.5)' for i in range(len(s))])


def removeOutliers(df, col, threshold=0.001):
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

plot_histograms_binary(
    df.drop(['id'], axis=1),
    target=TARGET,
    palette=SNS_CMAP
)


def plot_binary_distributions(df, target, filter_outliers = False, cols=None):
    if cols is None:
        cols = [col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])
                and len(df[col].unique())>10]
    fig,axes = plt.subplots(len(cols), 2,figsize=(15, 5*len(cols)))

    for i, col in enumerate(cols):
        if filter_outliers:
            filtered_df = removeOutliers(df, col)
        else:
            filtered_df = df
        sns.kdeplot(data = filtered_df, x=col, hue=df[target], fill=True, ax=axes[i][0], palette={0: colors[0], 1: colors[5]})
        sns.boxplot(filtered_df, y=col, x=TARGET, ax=axes[i][1], palette={0: colors[0], 1: colors[4]})
    fig.tight_layout()
    plt.show()

df = df.drop(['id'], axis=1)
df['month'] = df['month'].map({
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
})
plot_binary_distributions(df, TARGET, filter_outliers = True)


from sklearn.preprocessing import OrdinalEncoder
df[['job', 'marital', 'education', 'contact', 'poutcome']] = OrdinalEncoder().fit_transform(
    df[['job', 'marital', 'education', 'contact', 'poutcome']]
)
numerics = [bool, int, 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
pp_df = df.select_dtypes(include=numerics)
plt.figure(figsize=(15, 15))
sns.heatmap(pp_df.corr(), cmap=SNS_CMAP, annot=True, annot_kws={'fontsize':7}, fmt='.1g', vmin=-1, vmax=1, center= 0)
plt.title("Feature Correlation")
plt.show()


sns.pairplot(df[['age', 'contact', 'duration', 'balance', 'pdays', 'previous', 'y']].sample(10000), hue=TARGET, plot_kws={'alpha': 0.5}, palette=SNS_CMAP)
plt.show()


import scipy.stats as stats

num_cols = cols = [col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])
                and len(df[col].unique())>10]

def make_probaplots(df, cols, ncols=3):
    nrows = (len(cols)+ncols-1)//ncols
    fig,ax = plt.subplots(nrows, ncols ,figsize=(15, 6*nrows))
    for i, col in enumerate(cols):
        stats.probplot(df[col], dist="norm", plot=ax.flatten()[i])
        ax.flatten()[i].set_title(col)
    plt.show()


make_probaplots(df.dropna(), num_cols)


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

def plot_violins_categorical(df: pd.DataFrame, target: str = "TARGET",
                           cols:list[str]=None, ncols: int = 3, outliers: bool = False)->None:
    """
    Plot a series of histograms for a binary target
    
    Parameters: 
        df (Dataframe) : data to be visualized
        *params : any
    """
    df = df.copy()
    if cols is None:
        cols = df.columns
    nrows = (len(cols)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    for i, col in enumerate(cols):
        if outliers:
            data = removeOutliers(df, col)
        sns.violinplot(df_cat, y=col, x='_source', hue=TARGET, palette={0: colors[1], 1: colors[-2]}, ax=axes.flatten()[i])
    plt.show()

df['_source'] = 'train'
df_test['_source'] = 'test'
df_test[TARGET] = 0
df_original['_source'] = 'original'
df_cat = pd.concat([df, df_original, df_test])
df_cat = df_cat.drop(['id'], axis=1)

plot_violins_categorical(df_cat, 
                       cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'],
                       ncols=2,
                       outliers=True)


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


def cyclic_periodicals(df):
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_before_month = np.cumsum([0] + days_in_month[:-1])
    days_before_month = pd.Series(days_before_month, index=np.arange(1, 13))
    df['day_of_year'] = df['month'].map(days_before_month) + df['day']
    return df

def base_preproc(df):
    df = df.drop(['id'], axis=1)
    df['month'] = df['month'].map({
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    })
    df = cyclic_periodicals(df)
    return df

df = pd.read_csv(os.path.join(base_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))
df_original = pd.read_csv(r"/kaggle/input/bank-marketing-dataset-full/bank-full.csv", delimiter=";")
df_original[TARGET] = df_original[TARGET].map({
    'yes': 1,
    'no': 0
})
df['_source'] = 'train'
df_original['_source'] = 'origin'
df = pd.concat([df, df_original])
df = base_preproc(df)
df_test = base_preproc(df_test)
df.shape, df_test.shape


num_cols = ['age', 'balance', 'day', 'duration',
            'campaign', 'pdays', 'previous',
             'month_sin', 'month_cos', 'day_sin', 'day_cos', 'day_of_year']

ord_cols = ['marital', 'education', 'default',
            'housing', 'loan', 'contact', 'month', 'poutcome']

ohe_cols = ['job']
cat_cols = ohe_cols + ord_cols

class ShuffleFactory:
    @staticmethod
    def getShuffler(cat_cols):
        def shuffle_columns(df):
            df = df.copy()
            other_cols = [c for c in df.columns if c not in cat_cols]
            return df[cat_cols + other_cols]
        return shuffle_columns
    
cat_preproc = Pipeline([
    ('cat-select', ColumnTransformer(transformers=[
        ('pass', 'passthrough', num_cols+cat_cols),
    ], verbose_feature_names_out=False).set_output(transform='pandas')),
])

tree_preproc = Pipeline([
    ('rearrange', FunctionTransformer(ShuffleFactory.getShuffler(ord_cols+ohe_cols))),
    ('encode', ColumnTransformer(transformers=[
        ('ordinal', OrdinalEncoder(), ord_cols+ohe_cols),
        ('pass', 'passthrough', num_cols),
    ])),
])

linear_preproc = Pipeline([
    ('encode', ColumnTransformer(transformers=[
        ('pass', 'passthrough', num_cols),
        ('ordinal', OrdinalEncoder(), cat_cols),
        ('ohe', OneHotEncoder(), ohe_cols)]),
    ),
    ('scale', RobustScaler())
])


from collections import Counter

def calculate_imbalance_ratio(series):
    counter = Counter(series)
    zeros = counter[0]
    ones = counter[1]
    ratio =  max(zeros, ones) / min(zeros, ones)
    return ratio, counter

imbalance_ratio, class_counts = calculate_imbalance_ratio(df[TARGET])
imbalance_ratio


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
from sklearn.ensemble import BaggingClassifier, StackingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


import torch
if torch.cuda.is_available() and TREES_ON_GPU:
    from cuml.svm import SVC                     
else:
    from sklearn.svm import SVC
    

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, log_loss, mean_squared_error, matthews_corrcoef
import copy
from tqdm import tqdm


def train_and_evaluate_model(model, X, y, X_test=None, cv=5, name=None, stratify=False, 
                             retrain:bool = True, verbose: bool = 1):
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
    
    if name is None:
        name = model.__class__.__name__
        if name == "Pipeline":
            # Get the final estimator's class name
            name = model.steps[-1][1].__class__.__name__

    fold_iterator = folds
    if verbose:
        print(clrd(name, 'status'))
        fold_iterator = tqdm(folds, desc=f"Evaluating {name}")
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
        print(f"{clrd('accuracy ', 'ok')} : {  metrics['accuracy']:.4f}   {clrd('f1-score', 'log')}: {metrics['f1-score']:.4f}   {clrd('auc-roc', 'log')}: {metrics['auc-roc']:.4f}")
        print(f"{clrd('precision', 'log')} : {metrics['precision']:.4f}   {clrd('mcc     ', 'log')}: {metrics['mcc']:.4f}   {clrd('mse', 'log')}: {metrics['mse']:.4f}")
        print('-'*50)

    return metrics


models = {
    "logistic": make_pipeline(linear_preproc, LogisticRegression()),
    "lgbm": make_pipeline(tree_preproc,
                     LGBMClassifier(
                         verbose = 0,
                         objective = 'binary',
                     )),
    "rf": make_pipeline(tree_preproc, RandomForestClassifier()),
    "xgb": make_pipeline(tree_preproc,
                         XGBClassifier(
                            verbosity=0,
                            objective='binary:logistic',
                            eval_metric="auc",
                         )),
}

for model_name, model in models.items():
    _ = train_and_evaluate_model(model, df, df[TARGET], stratify=True, cv=5)


cb = make_pipeline(cat_preproc,
                   CatBoostClassifier(scale_pos_weight=imbalance_ratio,
                                      cat_features= cat_cols,
                                      verbose=False,
                                      task_type="GPU",
                                      devices="0:1"
                                     ))

train_and_evaluate_model(cb, df, df[TARGET], stratify=True, cv=3)


cb_importance = cb[1].get_feature_importance(type='FeatureImportance')  

import shap

X = cb[0].transform(df)
explainer = shap.TreeExplainer(cb[1])
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, max_display=30)


from pandas.api.types import is_numeric_dtype

correlations = []
for col in X.columns:
    if is_numeric_dtype(X[col]):
        corr = X[col].corr(df[TARGET])
    else:
        corr = np.nan
    correlations.append(corr)

df_importance = pd.DataFrame({
    'Feature': X.columns.tolist(),
    'CB_Importance': cb_importance,
    'Mean_ABS_SHAP': np.abs(shap_values[0]),
    'Corr_with_Target': np.abs(correlations)
}).sort_values("CB_Importance", ascending=False).reset_index(drop=True)

df_importance.style.background_gradient(cmap=SNS_CMAP)


pred = cb.predict_proba(df_test)[:, 1]


df_sub = df_sample.copy()
df_sub[TARGET] = pred
df_sub.to_csv('submission.csv', index=False)


plt.figure(figsize=(15, 6))
sns.kdeplot(pred)
plt.show()

