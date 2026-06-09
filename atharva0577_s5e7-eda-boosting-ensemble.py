import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm.notebook import tqdm

import os
base_path = "/kaggle/input/playground-series-s5e7"
df = pd.read_csv(os.path.join(base_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))
df_sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
TARGET = 'Personality'

import warnings
warnings.filterwarnings('ignore')
%load_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
SNS_CMAP = 'PRGn'
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

df[TARGET] = df[TARGET].map({
    'Extrovert': 0,
    'Introvert': 1
})
df_head_binary(df, TARGET, SNS_CMAP)


df.describe().iloc[1:].style.background_gradient(cmap=SNS_CMAP)


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
    
def plot_histograms_binary(df: pd.DataFrame, target: str = "TARGET",
                           cols:list[str]=None, ncols: int = 3, outliers: bool = False,
                          palette:str|list=None)->None:
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    for i, col in enumerate(cols):
        if outliers:
            data = removeOutliers(df, col)
        else:
            data = df
        if len(data[col].unique())>10:
            sns.histplot(data=data, x=col, bins=20, hue=TARGET, ax=axes.flatten()[i], kde=True, palette=palette)
        else:
            sns.histplot(data, x=col, hue=TARGET, ax=axes.flatten()[i], discrete=True, palette=palette)
            axes.flatten()[i].tick_params(axis='x', labelsize=5)
    plt.show()

# df = df.drop(['id'], axis=1)
plot_histograms_binary(df, TARGET, outliers=False, palette=BIN_SNS_CMAP)


def plot_binary_distributions(df, target, filter_outliers = False, cols=None):
    if cols is None:
        cols = [col for col in df.columns if len(df[col].unique())>10]
    fig,axes = plt.subplots(len(cols), 2,figsize=(15, 5*len(cols)))

    for i, col in enumerate(cols):
        if filter_outliers:
            filtered_df = removeOutliers(df)
        else:
            filtered_df = df
        sns.kdeplot(data = filtered_df, x=col, hue=df[target], fill=True, ax=axes[i][0], palette={0: colors[0], 1: colors[5]})
        sns.boxplot(df, y=col, x=TARGET, ax=axes[i][1], palette={0: colors[0], 1: colors[4]})
    fig.tight_layout()
    plt.show()
    
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
plot_binary_distributions(df, TARGET, filter_outliers = False, cols=num_cols)


def plot_nans(df:pd.DataFrame, num_cols):
    nan_cols = [col for col in df.columns if df[col].isna().sum() > 100]
    
    fig, ax = plt.subplots(len(nan_cols), 3, width_ratios=[3, 3, 1], figsize=(16, 8*len(nan_cols)))
    for i, col in enumerate(nan_cols):
        targ_num_cols = [numcol for numcol in num_cols if numcol != col]
        df[f"{col}_is_nan"] = df[col].isna()
        sns.violinplot(data=df, y=targ_num_cols[0], x=f"{col}_is_nan", ax=ax.flatten()[3*i], palette=SNS_CMAP)
        sns.violinplot(data=df, y=targ_num_cols[1], x=f"{col}_is_nan", ax=ax.flatten()[3*i+1], palette=SNS_CMAP)
        b = df[col].isna().sum()*100/df.shape[0]  # Complement percentage for category B
        a = 100-b 
        data = pd.DataFrame({
            "NaN": ["%", "%"],  
            "Type": ["Non-NaN", "NaN"],  
            "Percentage": [a, b]  
        })
        
        sns.barplot(
            data=data,
            x="NaN",
            y="Percentage",
            hue="Type",
            dodge=False,  
            palette=[colors[0], colors[-1]],
            ax=ax.flatten()[3*i+2]
        )
        
        ax.flatten()[3*i+2].legend().set_visible(False)
        
    fig.suptitle("Significance of nan values ?")
    fig.subplots_adjust(top=0.95, hspace=0.4) 
    plt.show()

plot_nans(df, num_cols)


df = df.drop(['id'], axis=1)
df['Stage_fear'] = df['Stage_fear'].map({'No': 0, 'Yes': 1})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
numerics = [bool, int, 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
pp_df = df.select_dtypes(include=numerics)
plt.figure(figsize=(15, 15))
sns.heatmap(pp_df.corr(), cmap=SNS_CMAP, annot=True, annot_kws={'fontsize':7}, fmt='.1g', vmin=-1, vmax=1, center= 0)
plt.title("Feature Correlation")
plt.show()


df = df[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency', 'Personality']]

sns.pairplot(df, hue=TARGET, plot_kws={'alpha': 0.5}, palette=SNS_CMAP)
plt.show()


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


def base_preproc(df):
    original_shape = df.shape
    df = df.drop(['id'], axis=1)

    #encode nans
    nan_cols = [col for col in df.columns if df[col].isna().sum() > 100]
    for col in nan_cols:
        df[f"{col}_is_nan"] = df[col].isna().astype(int)
    
    new_shape = df.shape
    print(f"{clrd(original_shape, 'status')} ===> {clrd(new_shape, 'status')}")
    return df

df = pd.read_csv(os.path.join(base_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))

df[TARGET] = df[TARGET].map({
    "Extrovert": 0,
    "Introvert": 1
})
df = base_preproc(df)
df_test = base_preproc(df_test)


ord_cols = ['Stage_fear', 'Drained_after_socializing']
num_cols = ['Time_spent_Alone',  'Social_event_attendance','Going_outside',
            'Friends_circle_size', 'Post_frequency']
nan_encoders = ['Time_spent_Alone_is_nan',
       'Stage_fear_is_nan', 'Social_event_attendance_is_nan',
       'Going_outside_is_nan', 'Drained_after_socializing_is_nan',
       'Friends_circle_size_is_nan', 'Post_frequency_is_nan']

preproc = Pipeline([
    ('col-trans', ColumnTransformer(transformers=[
        ('encode', OrdinalEncoder().set_output(transform='pandas'), ord_cols),
        ('pass', 'passthrough', num_cols+nan_encoders),
    ], verbose_feature_names_out=False).set_output(transform='pandas')),
    ("impute", SimpleImputer(strategy='constant', fill_value=0).set_output(transform='pandas')),
])

preproc2 = Pipeline([
    ('col-trans', ColumnTransformer(transformers=[
        ('encode', OrdinalEncoder().set_output(transform='pandas'), ord_cols),
        ('pass', 'passthrough', num_cols+nan_encoders),
    ], verbose_feature_names_out=False).set_output(transform='pandas')),
    ("impute", SimpleImputer(strategy='constant', fill_value=0).set_output(transform='pandas')),
    ("scale", StandardScaler())
])

preproc3 = Pipeline([
    ('col-trans', ColumnTransformer(transformers=[
        ('encode', OrdinalEncoder().set_output(transform='pandas'), ord_cols),
        ('pass', 'passthrough', num_cols+nan_encoders),
    ], verbose_feature_names_out=False).set_output(transform='pandas')),
    ("impute", KNNImputer(n_neighbors=2).set_output(transform='pandas')),
])


from collections import Counter

def calculate_imbalance_ratio(series):
    counter = Counter(series)
    zeros = counter[0]
    ones = counter[1]
    ratio =  max(zeros, ones) / min(zeros, ones)
    return ratio, counter

imbalance_ratio, class_counts = calculate_imbalance_ratio(df[TARGET])


import optuna
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import torch
if torch.cuda.is_available():
    from cuml.svm import SVC                      
else:
    from sklearn.svm import SVC

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, log_loss, mean_squared_error
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

    for k, v in metrics.items():
        metrics[k] = np.mean(v)

    if retrain:
        model.fit(X, y)
    if verbose:
        print(f"{clrd('accuracy ', 'ok')} : {  metrics['accuracy']:.4f}   {clrd('f1-score', 'log')}: {metrics['f1-score']:.4f}   {clrd('auc-roc', 'log')}: {metrics['auc-roc']:.4f}")
        print(f"{clrd('precision', 'log')} : {metrics['precision']:.4f}   {clrd('log-loss', 'log')}: {metrics['log-loss']:.4f}   {clrd('mse', 'log')}: {metrics['mse']:.4f}")
        print('-'*50)

    return metrics


baseliners = [
    make_pipeline(preproc, CatBoostClassifier(scale_pos_weight=imbalance_ratio, verbose=False)),
    make_pipeline(preproc, XGBClassifier(verbose=0, scale_pos_weight=4.5, objective='binary:logistic')),
    make_pipeline(preproc, LGBMClassifier(verbosity=-1)),
    make_pipeline(preproc2, LogisticRegression()),
    make_pipeline(preproc2, SVC()),
]

for baseliner in baseliners:
    train_and_evaluate_model(baseliner, df, df[TARGET], stratify=True)


import shap
X, y = preproc.fit_transform(df), df[TARGET]
model = CatBoostClassifier(scale_pos_weight=imbalance_ratio, verbose=False)
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


# 1. lightgbm
def objective(trial):
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample_for_bin": trial.suggest_int("subsample_for_bin", 50000, 1000000),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.0001, 0.1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    _use_custom_depth = trial.suggest_categorical("_use_custom_depth", [False, True])
    _use_regularization = trial.suggest_categorical("_use_regularization", [False, True])
    
    if _use_custom_depth:
        params["max_depth"] = trial.suggest_int("max_depth", 1, 20)
        params["num_leaves"] = trial.suggest_int("num_leaves", 2, min(256, 2**params["max_depth"]))
    if _use_regularization:
        params["lambda_l1"] = trial.suggest_float("lambda_l1", 1e-9, 10.0, log=True)
        params["lambda_l2"] = trial.suggest_float("lambda_l2", 1e-9, 10.0, log=True)

    if params["bagging_freq"] > 0:
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)

    try:
        X, y = df, df[TARGET]
        model_trial = LGBMClassifier(
                verbosity=-1,
                **params,
        )
        cv = 5
        if globals().get('DEBUG', False):
            cv = 2
        trial_pipe = make_pipeline(preproc, model_trial)
        metrics = train_and_evaluate_model(
            trial_pipe,
            X, y,
            cv=cv,
            stratify=True,
            verbose=False,
            retrain=False
        )
        print(f"{clrd('accuracy ', 'ok')} : {  metrics['accuracy']:.4f}   {clrd('f1-score', 'log')}: {metrics['f1-score']:.4f}   {clrd('auc-roc', 'log')}: {metrics['auc-roc']:.4f}")
        return metrics["accuracy"]
    except Exception as e:
        return 0.001

study = optuna.create_study(
    study_name="lgbm_study1",
    direction='maximize',
    storage="sqlite:///lgbm_study1.db",
    load_if_exists=False,
)
# study.optimize(objective, n_trials=n_trials)


# 2. xgboost
def objective(trial):
    params = {
        "eta": trial.suggest_loguniform("eta", 1e-3, 1),
        "gamma": trial.suggest_loguniform("gamma", 1e-6, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
    }

    try:
        X, y = df, df[TARGET]
        model_trial = XGBClassifier(
                verbose=0,
                objective='binary:logistic',
                **params,
        )
        cv = 5
        if globals().get('DEBUG', False):
            cv = 2
        trial_pipe = make_pipeline(preproc, model_trial)
        metrics = train_and_evaluate_model(
            trial_pipe,
            X, y,
            cv=cv,
            stratify=True,
            verbose=False,
            retrain=False
        )
        print(f"{clrd('accuracy ', 'ok')} : {  metrics['accuracy']:.4f}   {clrd('f1-score', 'log')}: {metrics['f1-score']:.4f}   {clrd('auc-roc', 'log')}: {metrics['auc-roc']:.4f}")
        return metrics["auc-roc"]
    except Exception as e:
        return 0.001


lgbm_params1 = {'learning_rate': 0.18823296973902906, 'n_estimators': 480, 'feature_fraction': 0.9195232205693685, 'bagging_freq': 2, 'min_child_samples': 58, 'subsample_for_bin': 170228, 'min_split_gain': 0.0026891299470766516, 'min_child_weight': 0.0015001966633777393, 'subsample': 0.9153506423127097, 'subsample_freq': 4, 'colsample_bytree': 0.671364538299203, 'max_depth': 4, 'num_leaves': 3, 'bagging_fraction': 0.6873354053651382}
lgbm_params2 = {'learning_rate': 0.005628025681982839, 'n_estimators': 323, 'feature_fraction': 0.8492812780788751, 'bagging_freq': 4, 'min_child_samples': 8, 'subsample_for_bin': 766522, 'min_split_gain': 0.1308928379481489, 'min_child_weight': 0.0010429507046734907, 'subsample': 0.7331347621940543, 'subsample_freq': 0, 'colsample_bytree': 0.5019278326628371, '_use_custom_depth': False, '_use_regularization': True, 'lambda_l1': 0.5352857726111039, 'lambda_l2': 2.429878135856599e-09, 'bagging_fraction': 0.8825531336130766}
lgbm_params3 = {'learning_rate': 0.029715314089089912, 'n_estimators': 75, 'feature_fraction': 0.5433760586007859, 'bagging_freq': 4, 'min_child_samples': 9, 'subsample_for_bin': 607294, 'min_split_gain': 0.2610801872724405, 'min_child_weight': 0.004743164570631372, 'subsample': 0.6486514969205959, 'subsample_freq': 4, 'colsample_bytree': 0.6320604357612342, '_use_custom_depth': False, '_use_regularization': True, 'lambda_l1': 8.900959311334528e-08, 'lambda_l2': 6.402057358349848e-07, 'bagging_fraction': 0.886557169324159}
xgb_params1 = {'eta': 0.30947348375254524, 'gamma': 0.0003552544798729266, 'max_depth': 19, 'min_child_weight': 184, 'max_delta_step': 2, 'subsample': 0.9442597699312696, 'colsample_bytree': 0.3378681794507, 'colsample_bylevel': 0.7762598452753994, 'lambda': 0.7243559032273449, 'alpha': 0.5503731253406781, 'n_estimators': 3030}
xgb_params2 = {'eta': 0.0018274538755272414, 'gamma': 0.0021583234908928045, 'max_depth': 49, 'min_child_weight': 15, 'max_delta_step': 1, 'subsample': 0.9782244579676395, 'colsample_bytree': 0.5983500510704229, 'colsample_bylevel': 0.31661335570564, 'lambda': 2.093556869009121, 'alpha': 0.4836965594250549, 'n_estimators': 4845}
xgb_params3 = {'eta': 0.0022291812296088153, 'gamma': 0.0034626817097779703, 'max_depth': 50, 'min_child_weight': 29, 'max_delta_step': 1, 'subsample': 0.9527835269901493, 'colsample_bytree': 0.748945025545701, 'colsample_bylevel': 0.32896308942194097, 'lambda': 1.1193547088879745, 'alpha': 0.3487446798755839, 'n_estimators': 4976}


models = [
    make_pipeline(preproc, LGBMClassifier(verbosity=-1,**lgbm_params1,)),
    make_pipeline(preproc, LGBMClassifier(verbosity=-1,**lgbm_params2,)),
    make_pipeline(preproc3, LGBMClassifier(verbosity=-1,**lgbm_params3)),
    make_pipeline(preproc, XGBClassifier(verbose=0, objective='binary:logistic', **xgb_params1,)),
    make_pipeline(preproc, XGBClassifier(verbose=0, objective='binary:logistic', **xgb_params2,)),
    make_pipeline(preproc, XGBClassifier(verbose=0, objective='binary:logistic', **xgb_params3,)),
]

for model in models:
    train_and_evaluate_model(model, df, df[TARGET], stratify=True)


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
from torchsummary import summary

from torch.utils.data import TensorDataset, DataLoader


class Classifier(nn.Module):
    def __init__(self,
                 input_dim:int,
                 num_classes:int=2,
                 dropout:float=0.5,
                 embedding_dim:int=16,
                ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
        
model = Classifier(input_dim=12)
summary(model, (12,))


from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Callable, ClassVar
import re

class EarlyStopping:
    def __init__(self, patience=5, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else 0

    def early_stop(self, score)->bool:
        if self.mode == 'min':
          self.best_score = min(self.best_score, score)
          if self.best_score - score < -self.delta:
            self.counter += 1
            if self.counter >= self.patience:
              return True
        else:
          self.best_score = min(self.best_score, score)
          if self.best_score - score > self.delta:
            self.counter += 1
            if self.counter >= self.patience:
              return True
        return False

class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            return [
                base_lr * step / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        elif step < self.total_steps:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
        else:
            return [self.min_lr for _ in self.base_lrs]
            
@dataclass
class TableData:
    """
    TableData — utility for logging tabular data with per-cell coloring.
    It also acts as a data store (dictionary of lists).

    Args:
        headers: column names.
        display: whether to print rows.
        min_width: minimum column width.
        max_width: maximum column width.
        color_func: function(row_idx, col_idx, data) -> color string or None.
        raise_exceptions: 0=silent, 1=raise, 2=verbose raise.
    """
    headers: List[str]
    display: bool = True
    min_width: int = 6
    max_width: int = 30
    color_func: Optional[
        Callable[[int, int, Dict[str, List[Any]]], Optional[str]]
    ] = None
    raise_exceptions: int = 0

    _widths: List[int] = field(init=False)
    _data: Dict[str, List[Any]] = field(init=False)
    _header_printed: bool = field(init=False, default=False)

    _color_codes: ClassVar[Dict[str, str]] = {
        # regular colors
        'black': '\033[30m', 'k': '\033[30m',
        'red': '\033[31m', 'r': '\033[31m',
        'green': '\033[32m', 'g': '\033[32m',
        'yellow': '\033[33m', 'y': '\033[33m',
        'blue': '\033[34m', 'b': '\033[34m',
        'magenta': '\033[35m', 'm': '\033[35m',
        'cyan': '\033[36m', 'c': '\033[36m',
        'white': '\033[37m', 'w': '\033[37m',
        # bright colors
        'bright_black': '\033[90m', 'bk': '\033[90m',
        'bright_red': '\033[91m', 'br': '\033[91m',
        'bright_green': '\033[92m', 'bg': '\033[92m',
        'bright_yellow': '\033[93m', 'by': '\033[93m',
        'bright_blue': '\033[94m', 'bb': '\033[94m',
        'bright_magenta': '\033[95m', 'bm': '\033[95m',
        'bright_cyan': '\033[96m', 'bc': '\033[96m',
        'bright_white': '\033[97m', 'bw': '\033[97m',
        # reset
        'reset': '\033[0m',
    }

    def __post_init__(self):
        self._data = {h: [] for h in self.headers}
        self._widths = [
            max(self.min_width, min(len(h), self.max_width))
            for h in self.headers
        ]
        if self.display:
            try:
                self._print_header()
            except Exception as e:
                if self.raise_exceptions:
                    raise e

    @property
    def data(self) -> Dict[str, List[Any]]:
        return self._data

    @classmethod
    def color_text(cls, text: str, color: Optional[str]) -> str:
        try:
            if not color:
                return text
            return f"{cls._color_codes.get(color, '')}{text}{cls._color_codes['reset']}"
        except Exception as e:
            return text

    @staticmethod
    def _strip_ansi(text: str) -> str:
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', text)

    def _format_cell(self, text: Any, width: int) -> str:
        s = str(text)
        s_plain = self._strip_ansi(s)
        if len(s_plain) > width:
            s = s_plain[:width - 3] + '...'
        return s + ' ' * (width - len(self._strip_ansi(s)))

    def _print_header(self):
        line = " ".join(self._format_cell(h, w) for h, w in zip(self.headers, self._widths))
        print(line)
        print("-" * len(line))
        self._header_printed = True

    def add_row(self, row: List[Any]):
        """
        Add a row to the table, store data, and optionally print it.
        """
        if len(row) != len(self.headers):
            raise ValueError(f"Row has {len(row)} elements but expected {len(self.headers)}")

        # store
        for key, val in zip(self.headers, row):
            self._data[key].append(val)

        if self.display:
            try:
                row_idx = len(self._data[self.headers[0]]) - 1
                cells = []

                for col_idx, (val, width, col_name) in enumerate(zip(row, self._widths, self.headers)):
                    color = None
                    if self.color_func:
                        color = self.color_func(row_idx, self.headers[col_idx], self._data)

                    cell_text = self._format_cell(val, width)
                    cell_text = self.color_text(cell_text, color)
                    cells.append(cell_text)

                row_str = " ".join(cells)
                print(row_str)

            except Exception as e:
                if self.raise_exceptions:
                    raise e

def train_color(row, col, data):
    if col=='epoch':
        return 'w'
    if 'accuracy' in col:
        if data[col][row]!=data[col][row-1]:
            return 'g'
        else:
            return 'w'
    if data[col][row]<data[col][row-1]:
        return 'b'
    else:
        return 'c'

def plot_loss_history(history):
    train_loss = history.get('train-loss', [])
    val_loss = history.get('val-loss', [])
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(15, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()        


num_epochs = 30
batch_size = 256
lr = 0.5*1e-3
alpha = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X, y = preproc2.fit_transform(df), df[TARGET]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
in_dim = X.shape[1]

steps_per_epoch = len(train_loader)
warmup_steps = steps_per_epoch * 2
total_steps = steps_per_epoch * num_epochs

model = Classifier(input_dim=in_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=alpha)
scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps, min_lr=1e-6)
criterion = nn.CrossEntropyLoss()


def train(model,
          optimizer,
          criterion,
          train_data,
          val_data=None,
          lr_scheduler=None,
          early_stopping=False,
          device='cpu',
          epochs=100,
          verbose:bool=True,
         ):

    hist = TableData(['epoch', 'train-loss', 'val-loss', 'val-accuracy'],
                     display=verbose, min_width=20,
                     color_func=train_color)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (X, y) in enumerate(train_data):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_data)

        val_loss, val_acc = 0.0, 0.0
        if val_data is not None:
            for batch_idx, (X, y) in enumerate(val_data):
                model.eval()
                X, y = X.to(device), y.to(device)
                with torch.no_grad():
                    logits = model(X)
                    val_loss += criterion(logits, y).item()
                    val_acc += float((logits.argmax(dim=1) == y).float().mean())
            val_loss /= len(val_data)
            val_acc /= len(val_data)

        hist.add_row([epoch, train_loss, val_loss, val_acc])

    return hist.data

history = train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    scheduler,
    early_stopping=False,
    device=device,
    epochs=num_epochs,
    verbose=True,
)

plot_loss_history(history)


X_test = preproc2.transform(df_test)
X_test = torch.tensor(X_test, dtype=torch.float32)
model.eval()
threshold = 0.38
with torch.no_grad():
    logits = model(X_test)
    logits = F.softmax(logits, dim=1)
    pred = logits[:, 1]>threshold
    pred = pred.cpu().numpy().astype(int)
    print(np.unique(pred, return_counts=True))


from sklearn.ensemble import StackingClassifier

stk1 = StackingClassifier(estimators = [
    ('lgbm1', LGBMClassifier(verbosity=-1,**lgbm_params1,)),
    ('lgbm2', LGBMClassifier(verbosity=-1,**lgbm_params2,)),
    ('xgb1', XGBClassifier(verbose=0, objective='binary:logistic', **xgb_params1,)),
])

stk_pipe1 = make_pipeline(preproc, stk1)
_ = train_and_evaluate_model(stk_pipe1, df, df[TARGET], stratify=True)
pred = stk_pipe1.predict(df_test)


from sklearn.ensemble import StackingClassifier

stk2 = StackingClassifier(estimators = [
    ('cb', CatBoostClassifier(scale_pos_weight=imbalance_ratio, verbose=False)),
    ('xgb', XGBClassifier(verbose=0, scale_pos_weight=4.5, objective='binary:logistic')),
    ('lgbm', LGBMClassifier(verbosity=-1)),
])

stk_pipe2 = make_pipeline(preproc, stk2)
train_and_evaluate_model(stk_pipe1, df, df[TARGET], stratify=True)
pred = stk_pipe2.predict(df_test)


df_sub = df_sample.copy()
df_sub[TARGET] = pred
df_sub[TARGET] = df_sub[TARGET].map({0: "Extrovert", 1: "Introvert"})
df_sub.to_csv('submission.csv', index=False)

