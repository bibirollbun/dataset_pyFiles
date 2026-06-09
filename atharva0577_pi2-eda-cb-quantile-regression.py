import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm.notebook import tqdm

import os
base_path = "/kaggle/input/prediction-interval-competition-ii-house-price"
df = pd.read_csv(os.path.join(base_path, 'dataset.csv'))
df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))
df_sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
TARGET = 'sale_price'

import warnings
warnings.filterwarnings('ignore')
%load_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
SNS_CMAP = 'afmhot'
sns.set_palette(SNS_CMAP)
# plt.style.use("dark_background")
# plt.rcParams['grid.color'] = '#444444'
colors = sns.palettes.color_palette(SNS_CMAP)
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

df = df.drop('id', axis=1)
df.head().style.background_gradient(cmap=SNS_CMAP)


agg_df = df.agg(["nunique", "unique", lambda x:x.isna().sum(), "dtypes"]).T
agg_df['unique'] = agg_df['unique'].apply(lambda x: x if len(x)<10 else x[:10])
agg_df.style.apply(lambda s: [f'background-color: rgba({colors[0][0]*255}, {colors[0][1]*255}, {colors[0][2]*255}, 0.5)' if i % 2 == 0 else f'background-color: rgba({colors[3][0]*255}, {colors[3][1]*255}, {colors[3][2]*255}, 0.5)' for i in range(len(s))])


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

df['sale_date'] = pd.to_datetime(df['sale_date'])
df['year'] = df['sale_date'].dt.year
df['month'] = df['sale_date'].dt.month
df['day_of_week'] = df['sale_date'].dt.dayofweek
df['time_elapsed'] = (df['sale_date'] - df['sale_date'].min()).dt.days

num_cols = ['sale_price', 'sale_nbr', 'latitude', 'longitude', 'area', 'land_val', 'imp_val', 'year_built', 'year_reno', 'sqft_lot', 'sqft', 'sqft_1', 'sqft_fbsmt', 'grade']
plot_histograms_continous(
    df.drop(['sale_date', 'zoning', 'subdivision', 'sale_warning', 'city'], axis=1),
    target=TARGET,
    palette=SNS_CMAP
)


def plot_numerical_distributions(df: pd.DataFrame, num_cols: list[str], drop_cols:list[str], target: str):
    df = df.copy()
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
            ax[i].set_xticks(x_ticks[::num_ticks])  # Keep every 5th tick
            ax[i].set_xticklabels([int(tick) for tick in x_ticks[::num_ticks]])  # Convert tick labels to integers
        except Exception as e:
            print(f"{i}: exception for column {col}")
            raise e
    
    plt.tight_layout()
    fig.suptitle('Distribution of Target vs Numerical Features')
    fig.subplots_adjust(top=0.97) 
    plt.show()

plot_numerical_distributions(df, num_cols, [TARGET], TARGET)


def plot_against_time(df: pd.DataFrame, num_cols: list[str], target: str):
    df = df.copy()
    
    fig, ax = plt.subplots(len(num_cols), 1, figsize=(15, 6*len(num_cols)))

    df[target] = pd.cut(df[target], bins=30).apply(lambda x: np.ceil(x.right) if pd.notnull(x) else np.nan)
    for i, col in enumerate(num_cols):
        try:
            sns.boxplot(data=df, x=target, y=col, palette='autumn', ax=ax[i])
            num_ticks = max(1, len(df[col].unique())//10)
            x_ticks = ax[i].get_xticks()
            ax[i].set_xticks(x_ticks[::num_ticks])  # Keep every 5th tick
            ax[i].set_xticklabels([int(tick) for tick in x_ticks[::num_ticks]])  # Convert tick labels to integers
        except Exception as e:
            print(f"{i}: exception for column {col}")
            raise e
    plt.tight_layout()
    fig.suptitle('Distribution of Numerical Features with Time(t)')
    fig.subplots_adjust(top=0.97) 


    plt.show()

plot_against_time(df, ['sale_price', 'sale_nbr'], 'time_elapsed')


def plot_violinplots_categoricals(
    df: pd.DataFrame,
    target: str,
    cat_cols: list[str] = None,
    unique_cat_threshold: int = 7,
    max_unique_values: int = 20,
    palette: str = "Set2",
    drop_cols: list[str] = None,
) -> None:
    """
    Plots violinplots for categorical features against a numerical target.

    Parameters:
    - df: pd.DataFrame â€“ the dataset
    - target: str â€“ the numerical column to be plotted on the y-axis
    - cat_cols: list[str], optional â€“ categorical columns to consider; if None, selects columns with unique values <= max_unique_values
    - unique_cat_threshold: int â€“ if a categorical feature has > this many unique values, it takes a full row
    - max_unique_values: int â€“ threshold to auto-select categorical columns if not provided
    - palette: str â€“ seaborn color palette for hue
    - drop_cols: list[str], optional â€“ columns to drop from cat_cols if auto-selected

    Returns:
    - None
    """
    df = df.copy()
    debug = globals().get('DEBUG', False)

    if cat_cols is None:
        cat_cols = [col for col in df.columns if df[col].nunique() <= max_unique_values and col != target]
    if drop_cols is not None:
        cat_cols = [col for col in cat_cols if col not in drop_cols]

    full_row_plots = [col for col in cat_cols if df[col].nunique() > unique_cat_threshold]
    two_col_plots = [col for col in cat_cols if df[col].nunique() <= unique_cat_threshold]

    if debug:
        print(f"single ({len(full_row_plots)}): {full_row_plots} \ndouble ({len(two_col_plots)}): {two_col_plots}")

    total_two_col_rows = (len(two_col_plots) + 1) // 2
    total_rows = len(full_row_plots) + total_two_col_rows

    fig = plt.figure(figsize=(16, 5 * total_rows))
    gs = GridSpec(total_rows, 2, figure=fig)
    plot_row = 0

    # Plot full-row violins (span both columns)
    for col in full_row_plots:
        ax = fig.add_subplot(gs[plot_row, :])  # span both columns
        sns.violinplot(data=df, x=col, y=target, ax=ax, palette=palette)
        ax.set_title(f"{col} vs {target}")
        if ax.get_legend(): ax.get_legend().remove()
        plot_row += 1

    # Plot two-per-row violins
    for i in range(0, len(two_col_plots), 2):
        col1 = two_col_plots[i]
        ax1 = fig.add_subplot(gs[plot_row, 0])
        sns.violinplot(data=df, x=col1, y=target, ax=ax1, palette=palette)
        ax1.set_title(f"{col1} vs {target}")
        if ax1.get_legend(): ax1.get_legend().remove()

        if i + 1 < len(two_col_plots):
            col2 = two_col_plots[i + 1]
            ax2 = fig.add_subplot(gs[plot_row, 1])
            sns.violinplot(data=df, x=col2, y=target, ax=ax2, palette=palette)
            ax2.set_title(f"{col2} vs {target}")
            if ax2.get_legend(): ax2.get_legend().remove()
        plot_row += 1

    plt.tight_layout()
    plt.show()

plot_violinplots_categoricals(df, TARGET, palette=SNS_CMAP)


def plot_heatmap(df, cmap=SNS_CMAP):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    plt.figure(figsize=(15, 15))
    sns.heatmap(df.select_dtypes(include=numerics).corr(),
                cmap=cmap, annot=False, annot_kws={'fontsize':7}, fmt='.1g', vmin=-1, vmax=1, center= 0)
    plt.title("Feature Correlation")
    plt.show()

plot_heatmap(df)


from catboost import CatBoostRegressor, Pool
import shap

df['isna_sale_nbr'] = df['sale_nbr'].isna()
df['isna_subdivision'] = df['subdivision'].isna()

cat_cols = [col for col in df.columns if (df[col].dtype == "object" 
                                          or df[col].dtype.name == "string"
                                          or df[col].nunique() <= 7 )]

num_cols = [col for col in df.columns if (
        pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique() > 7
        and col not in cat_cols)]

X = df[num_cols+cat_cols].drop(columns=[TARGET], axis=1)
X = X.fillna(0)
y = df[TARGET]

model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        cat_features=cat_cols,
        random_seed=42
    )
model.fit(X, y)
cb_importance = model.get_feature_importance(type='FeatureImportance')  

explainer = shap.TreeExplainer(model)
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


def base_preproc(df):
    debug = globals().get('DEBUG', False) 
    df = df.copy()
    original_shape = df.shape
    #extract time-related feat
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    df['time_elapsed'] = (df['sale_date'] - df['sale_date'].min()).dt.days

    #concat views
    view_cols = [col for col in df.columns if col[:4]=='view']
    if not len(view_cols)==10:
        print(clrd(f"{len(view_cols)} `view cols` found instead of 10", 'warn'))
    df['view_combined'] = df[view_cols].sum(axis=1)
    
    #drop some cols
    drop_cols = ['id', 'sale_date']
    df = df.drop(drop_cols, axis=1)

    new_shape = df.shape
    print(clrd(f"{original_shape} ===> {new_shape}", 'status'))
    return df
    
df = pd.read_csv(os.path.join(base_path, 'dataset.csv'))
df = base_preproc(df)

df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))
df_test = base_preproc(df_test)


def freq_target_encode(df_train: pd.DataFrame, df_test: pd.DataFrame,
                      cols: list[str], target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate frequency-encoded and target-encoded features
    for given columns inplace.
    """

    train_shape, test_shape = df_train.shape, df_test.shape
    for col in cols:
        #frequency-encoding
        frequencies = df_train[col].value_counts()
        df_train[f"freq_{col}"] = df_train[col].map(frequencies)
        df_test[f"freq_{col}"] = df_test[col].map(frequencies)

        #target-encoding
        targs = df_train[[col, TARGET]].groupby(col)[TARGET].transform('mean')
        df_train[f"targ_{col}"] = targs
        df_test = pd.merge(df_test, df_train[[col, f"targ_{col}"]].drop_duplicates(), how='left', on=col)

        print(f"Found {clrd(len(frequencies), 'ok')} unique values for the column {clrd(col, 'log')}")

    print("="*50)
    print(f"train: {clrd(train_shape, 'ok')} ====> {clrd(df_train.shape, 'ok')}")
    print(f"test : {clrd(test_shape, 'ok')} ====> {clrd(df_test.shape, 'ok')}")

    return df_train, df_test

freq_cols = ['city', 'zoning', 'subdivision',]
df, df_test = freq_target_encode(df, df_test, target=TARGET, cols = freq_cols)


num_cols = ['sale_nbr', 'latitude', 'longitude', 'area', 'land_val', 'imp_val',
            'year_built', 'year_reno', 'sqft_lot', 'sqft', 'sqft_1', 'sqft_fbsmt', 'grade',
            'fbsmt_grade', 'stories', 'beds', 'bath_full', 'bath_3qtr', 'bath_half',
            'garb_sqft', 'gara_sqft', 'wfnt', 'year', 'month', 'time_elapsed', 'view_combined']

cat_cols = ['sale_warning', 'join_status', 'join_year', 'city', 'zoning', 'subdivision',
            'present_use', 'condition', 'golf', 'greenbelt', 'noise_traffic', 'view_rainier',
            'view_olympics', 'view_cascades', 'view_territorial', 'view_skyline', 'view_sound',
            'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other', 'submarket']

X = df[num_cols+cat_cols]
X_test = df_test[num_cols+cat_cols]
X = X.fillna(0)
X_test = X_test.fillna(0)
y = df[TARGET]
X.shape, y.shape, X_test.shape


from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold


def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """Compute the Winkler Interval Score for prediction intervals.

    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.

    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)


import copy

def train_on_k_folds(model,
                 X, y,
                 X_test = None,
                 cv:int = 5,
                 verbose: int = 1,
                 retrain_model: bool = False,
                 ):

    test_preds = []
    scores = {"winkler": [], "coverage": []}
    cv_winkler = 0.0
    folds = KFold(n_splits=cv, shuffle=True, random_state=42)
    model_fold = None
    for fold, (train_idx, val_idx) in tqdm(enumerate(folds.split(X, y)), total=cv):
      model_fold = copy.deepcopy(model)
      X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
      X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

      model.fit(X_train_fold, y_train_fold)
      y_pred = model.predict(X_val_fold)
      if X_test:
        y_pred_test = model.predict(X_test)

      fold_winkler, fold_coverage = winkler_score(
          y_val_fold, y_pred[:, 0], y_pred[:, 1],
          return_coverage=True
      )
      scores["winkler"].append(fold_winkler)
      scores["coverage"].append(fold_coverage)

      if X_test:
        test_preds.append(y_pred_test)

      if verbose:
        print(f"Fold {fold+1}: {clrd(fold_winkler, 'status')}  Coverage: {clrd(fold_coverage, 'special')} \n")
    
    var = np.var(scores["winkler"])
    scores["winkler"] = np.mean(scores["winkler"])
    scores["coverage"] = np.mean(scores["coverage"])
    
    if verbose:
      print(f"CV Winkler Score: {clrd(scores['winkler'], 'ok')} += {var}  Coverage: {clrd(scores['coverage'], 'special')}")
    
    if retrain_model:
        pass
        model.fit(X_train, y_train)
    else:
        model = model_fold
    
    if X_test:
      test_preds = np.mean(test_preds, axis=0)
    else:
      test_preds = None
    return model, scores, test_preds


alpha = 0.05
quantile_levels = [alpha, 1 - alpha]
quantile_str = str(quantile_levels).replace('[','').replace(']','')

model = CatBoostRegressor(
    loss_function=f'MultiQuantile:alpha={quantile_str}',
    thread_count= 4,
    cat_features= cat_cols,
    bootstrap_type =  "Bernoulli",
    sampling_frequency= 'PerTree',
    verbose = False,
)

trained_model, scores, test_preds = train_on_k_folds(model, X, y)


pred = trained_model.predict(X_test)


!pip install -q optuna-integration[catboost]

import optuna
from optuna.integration import CatBoostPruningCallback


def objective(trial: optuna.Trial) -> float:
    alpha = 0.05
    quantile_levels = [alpha, 1 - alpha]
    # quantile_str = str(quantile_levels).replace('[','').replace(']','')
    quantile_str = "0.05,0.95"    
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1500, log=True),        
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 0.75, log=True),
        'depth': trial.suggest_int('depth', 2, 10, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg',1e-8, 100, log=True),
        'model_size_reg': trial.suggest_float('model_size_reg',1e-8, 100, log=True),
        'random_strength': trial.suggest_float('random_strength',1e-8, 100, log=True),
        'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.1, 1),
        'subsample': trial.suggest_float("subsample", 0.1, 1),
        
        # "used_ram_limit": "3gb",
    }

    try:
        metrics = {"winkler": []}
        folds = KFold(n_splits=5, shuffle=True, random_state=42).split(X, y)
        for train_index, test_index in folds:        
            X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

            model_trial = CatBoostRegressor(
                loss_function=f'MultiQuantile:alpha={quantile_str}',
                # thread_count= 4,
                cat_features= cat_cols,
                bootstrap_type =  "Bernoulli",
                sampling_frequency= 'PerTree',
                verbose = False,
                **param
            )
            
            pruning_callback = CatBoostPruningCallback(trial, f"MultiQuantile:alpha={quantile_str}")
            model_trial.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=100,
                callbacks=[pruning_callback],
            )
            # evoke pruning manually.
            pruning_callback.check_pruned()

            y_pred =  model_trial.predict(X_valid)
            fold_winkler = winkler_score(y_valid, y_pred[:, 0], y_pred[:, 1])
            metrics["winkler"].append(fold_winkler)
                
        return np.mean(metrics["winkler"])
        
    except Exception as e:
        if globals().get('DEBUG', False):
            raise e
        return 0.01


study = optuna.create_study(direction='minimize')
study.optimize(objective,
               # n_trials=3,
               timeout=25000
              )


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

class ConformalRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, base_model, alpha=0.1, calib_size=0.2, random_state=42):
        """
        base_model: an sklearn-compatible regressor with quantile prediction
        alpha: 1 - desired coverage level
        calib_size: fraction of data used for calibration
        """
        self.base_model = base_model
        self.alpha = alpha
        self.calib_size = calib_size
        self.random_state = random_state
        self.q_hat_ = None


    def fit(self, X, y):
        # 1. Split into training and calibration sets
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=self.calib_size, random_state=self.random_state
        )

        # 2. Fit passed model on train set
        self.base_model.fit(X_train, y_train)

        # 3. Predict on calibration set
        if type(y_calib)!=np.ndarray:
            y_calib = y_calib.to_numpy()
        preds = self.base_model.predict(X_calib)
        assert (preds.shape[1] == 2 and len(preds.shape)==2), f"base_model prediction shape {preds.shape} does not match required shape (-1, 2)"
        L, U = preds[:, 0], preds[:, 1]

        # 4. Compute nonconformity scores
        # print(L.shape, U.shape, y_calib.shape)
        # print(L.dtype, U.dtype, y_calib.dtype)
        scores = np.maximum(np.maximum(L - y_calib, y_calib - U), 0)
        # 5. Compute correction quantile
        self.q_hat_ = np.quantile(scores, 1 - self.alpha)

        return self

    def predict(self, X, **predict_kwargs):
        preds = self.base_model.predict(X, **predict_kwargs)
        L, U = preds[:, 0], preds[:, 1]

        L_adj = L - self.q_hat_
        U_adj = U + self.q_hat_

        return np.vstack([L_adj, U_adj]).T

    def predict_interval(self, X, **predict_kwargs):
        return self.predict(X, **predict_kwargs)


alpha = 0.05
quantile_levels = [alpha, 1 - alpha]
quantile_str = str(quantile_levels).replace('[','').replace(']','')

cb = CatBoostRegressor(
    loss_function=f'MultiQuantile:alpha={quantile_str}',
    thread_count= 4,
    cat_features= cat_cols,
    bootstrap_type =  "Bernoulli",
    sampling_frequency= 'PerTree',
    verbose = False,
)

conformal_cb = ConformalRegressor(cb)
conformal_cb, scores, test_preds = train_on_k_folds(conformal_cb, X, y)


df_sub = df_sample.copy()
df_sub[["pi_lower", "pi_upper"]] = pred
df_sub.to_csv('submission.csv', index=False)

