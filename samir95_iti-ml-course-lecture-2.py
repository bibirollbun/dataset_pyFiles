import warnings
warnings.filterwarnings('ignore')


import os

IS_KAGGLE = False
DATA_DIR = ""
if os.getcwd().startswith("/kaggle"):
    IS_KAGGLE = True
    DATA_DIR = "/kaggle/working/"

print("Notebook running on Kaggle!")


if IS_KAGGLE:
    print("Extracting competition data...")
    !unzip -o /kaggle/input/bluebook-for-bulldozers/Train.zip

    TRAIN_PATH = os.path.join(DATA_DIR, "Train.csv")


import pandas as pd


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder


def prepare_dataframe_for_modeling(
    dataframe,
    target_column=None,
    columns_to_skip=None,
    columns_to_ignore=None,
    scale_features=False,
    sample_size=None,
    pipeline=None,
    copy_data=True,
    max_n_cat=None
):
    """
    Processes a dataframe for modeling. In addition to basic imputation, scaling,
    and one-hot encoding, this function:
      - Extracts any ignored columns immediately.
      - For each categorical column:
           * If max_n_cat is provided and the number of unique values is > max_n_cat,
             applies ordinal encoding (using mode imputation) with a missing indicator.
           * Otherwise, uses one-hot encoding with a dummy for missing values.
      - For numeric columns, missing values are imputed using the median (with a missing indicator).
      
    Parameters:
      dataframe (pd.DataFrame): Input DataFrame.
      target_column (str): Name of the target column.
      columns_to_skip (list): Columns to drop from transformation.
      columns_to_ignore (list): Columns to extract and ignore (will be reattached later).
      scale_features (bool): If True, numeric features are scaled.
      sample_size (int): If provided, sample this many rows.
      pipeline (dict): If provided, uses the stored pipeline (transform mode).
      copy_data (bool): If True, work on a (shallow) copy; if False, modifies in place.
      max_n_cat (int): If provided, any categorical column with more than max_n_cat unique 
                       values is ordinal-encoded (with a missing indicator) rather than one-hot encoded.
                       
    Returns:
      Tuple (X_processed, y, pipeline) where:
        - X_processed (pd.DataFrame): The transformed features with ignored columns reattached.
        - y (np.array or pd.Series): Processed target variable.
        - pipeline (dict): Dictionary storing the fitted preprocessor and additional info.
    """
    # Optionally work on a copy.
    df = dataframe.copy(deep=False) if copy_data else dataframe

    # --- Optional Sampling ---
    if sample_size is not None:
        idxs = np.random.choice(df.index, size=sample_size, replace=False)
        df = df.loc[idxs]

    # --- Extract Ignored Columns Immediately ---
    if columns_to_ignore:
        ignored_df = df[columns_to_ignore]
        df.drop(columns=columns_to_ignore, inplace=True, errors='ignore')
    else:
        ignored_df = None

    # --- Process the Target Column ---
    target_encoder = None
    y = None
    if target_column is not None and target_column in df.columns:
        y = df[target_column]
        if not is_numeric_dtype(y):
            # For training mode, fit a LabelEncoder.
            if pipeline is None:
                target_encoder = LabelEncoder().fit(y)
            else:
                target_encoder = pipeline.get("target_encoder", None)
            y = target_encoder.transform(y)
        else:
            y = y.values
        # Ensure the target column is not used in features.
        if columns_to_skip is None:
            columns_to_skip = [target_column]
        elif target_column not in columns_to_skip:
            columns_to_skip.append(target_column)

    # --- Drop Columns to Skip (in place) ---
    if columns_to_skip:
        df.drop(columns=columns_to_skip, inplace=True, errors='ignore')

    # --- Identify Numeric and Categorical Features ---
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- For categorical columns, split into low- and high-cardinality groups ---
    cat_low = []
    cat_high = []
    if max_n_cat is not None:
        for col in categorical_features:
            nunique = df[col].nunique(dropna=True)
            if nunique <= max_n_cat:
                cat_low.append(col)
            else:
                cat_high.append(col)
    else:
        cat_high = categorical_features  # all categorical get ordinal encoding

    # --- Build or Use the Pipeline ---
    if pipeline is None:
        # Numeric pipeline: impute (with missing indicator) and (optionally) scale.
        num_pipeline_steps = [
            ('imputer', SimpleImputer(strategy='median', add_indicator=True))
        ]
        if scale_features:
            num_pipeline_steps.append(('scaler', StandardScaler()))
        numeric_pipeline = Pipeline(steps=num_pipeline_steps)

        # Categorical pipeline for low-cardinality features: impute and one-hot encode.
        if cat_low:
            cat_low_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # TODO: change this to fill with negative values (e.g. -1 or -999)
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        else:
            cat_low_pipeline = None

        # Categorical pipeline for high-cardinality features: impute with mode then ordinal encode
        # with an extra missing indicator.
        if cat_high:
            ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
            imputer = SimpleImputer(strategy="constant", fill_value=-1)

            cat_high_pipeline = Pipeline(steps=[
                ("ordinal", ordinal),
                ("imputer", imputer )
            ])
        
        else:
            cat_high_pipeline = None

        # Build the column transformer.
        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_pipeline, numeric_features))
        if cat_low_pipeline is not None:
            transformers.append(('cat_low', cat_low_pipeline, cat_low))
        if cat_high_pipeline is not None:
            transformers.append(('cat_high', cat_high_pipeline, cat_high))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        # Fit and transform the dataframe in place.
        X_transformed = preprocessor.fit_transform(df)

        # Attempt to get feature names (scikit-learn ≥1.0).
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None

        # Convert transformed output into a DataFrame.
        if feature_names is not None:
            X_processed = pd.DataFrame(X_transformed, index=df.index, columns=feature_names)
        else:
            X_processed = pd.DataFrame(X_transformed, index=df.index)

        # Save the pipeline details.
        pipeline = {
            'preprocessor': preprocessor,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'cat_low': cat_low,
            'cat_high': cat_high,
            'ignored_columns': columns_to_ignore,
            'columns_to_skip': columns_to_skip,
            'target_encoder': target_encoder if (target_column is not None and 
                                                  not is_numeric_dtype(dataframe[target_column]))
                                                  else None,
            'feature_names': feature_names,
            'max_n_cat': max_n_cat
        }
    else:
        # --- Transform mode: use the existing pipeline ---
        preprocessor = pipeline['preprocessor']
        # (Ignored columns have already been dropped above.)
        df.drop(columns=pipeline.get("columns_to_skip", []), inplace=True, errors='ignore')
        X_transformed = preprocessor.transform(df)
        feature_names = pipeline.get("feature_names", None)
        if feature_names is not None:
            X_processed = pd.DataFrame(X_transformed, index=df.index, columns=feature_names)
        else:
            X_processed = pd.DataFrame(X_transformed, index=df.index)

    # --- Reattach the Ignored Columns ---
    if ignored_df is not None:
        X_processed = pd.concat([ignored_df, X_processed], axis=1)

    return X_processed, y, pipeline


def split_vals(a,n): return a[:n].copy(), a[n:].copy()


import math

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


import IPython
import graphviz
from sklearn.tree import export_graphviz

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """Draws a representation of a random forest in IPython."""
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


DATA_DIR = "/kaggle/input/iti-ml-course-lecture-1"
df_raw = pd.read_parquet(os.path.join(DATA_DIR, "tmp/df_raw.parquet"))


df_raw = df_raw.sort_values("sale_elapsed")


df_raw["SalePrice"] = np.log(df_raw.SalePrice)


n_valid = 12000
n_trn = len(df_raw)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)


X_train, y_train, proc_pipeline = prepare_dataframe_for_modeling(raw_train, 'SalePrice')
X_valid, y_valid, _ = prepare_dataframe_for_modeling(raw_valid, 'SalePrice', pipeline=proc_pipeline)


raw_train.shape, X_train.shape, X_valid.shape


def clean_col_names(df):
    df.columns = [c.split("__")[-1] for c in df.columns]
    return df


X_train = clean_col_names(X_train)
X_valid = clean_col_names(X_valid)


X_train.head()


from sklearn.ensemble import RandomForestRegressor


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, max_samples=50_000, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


%time preds = np.stack([t.predict(X_valid) for t in m.estimators_])
np.mean(preds[:,0]), np.std(preds[:,0])


preds.shape


preds[:, 0]


preds.mean(axis=0).shape


from concurrent.futures import ProcessPoolExecutor

def parallel_trees(m, fn, n_jobs=8):
    return list(ProcessPoolExecutor(n_jobs).map(fn, m.estimators_))


def get_preds(t): return t.predict(X_valid)
%time preds = np.stack(parallel_trees(m, get_preds))
np.mean(preds[:,0]), np.std(preds[:,0])


preds.shape[0]


x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)


x.head()


x.pred_std.describe()


x.Enclosure.value_counts().plot.barh();


flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']
enc_summ = x[flds].groupby('Enclosure', as_index=False).mean()
enc_summ


# flds = ['YearMade', 'SalePrice', 'pred', 'pred_std']
# enc_summ = x[flds].groupby('YearMade', as_index=False).mean()
# enc_summ.sort_values("YearMade")


enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0,11));


enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0,11));


flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']
summ = x[flds].groupby(flds[0]).mean()
summ.sort_values("pred_std", ascending=False)


raw_valid.ProductSize.value_counts().plot.barh();


(summ.pred_std/summ.pred).sort_values(ascending=False)


def rf_feat_importance(m, df):
    return pd.DataFrame ({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


fi = rf_feat_importance(m, X_train); fi[:10]


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


fi[:30]


plot_fi(fi[:30]);


0.2 / 0.003


to_keep = fi[fi.imp>0.005].cols; len(to_keep)


X_train_org = X_train.copy()
X_valid_org = X_valid.copy()


X_train = X_train_org[to_keep]
X_valid = X_valid_org[to_keep]


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5,
                          max_samples=50_000, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


fi = rf_feat_importance(m, X_train)
plot_fi(fi);


X_train, y_train, proc_pipeline = prepare_dataframe_for_modeling(raw_train, 'SalePrice', max_n_cat=7)
X_valid, y_valid, _ = prepare_dataframe_for_modeling(raw_valid, 'SalePrice', pipeline=proc_pipeline)


X_train = clean_col_names(X_train)
X_valid = clean_col_names(X_valid)


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, max_samples=50_000, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


fi = rf_feat_importance(m, X_train)
plot_fi(fi[:25]);


from scipy.cluster import hierarchy as hc
import scipy


X_train = X_train_org[to_keep].copy()
X_valid = X_valid_org[to_keep].copy()


corr_condensed.shape


corr.shape


X_train.head()


corr = np.round(scipy.stats.spearmanr(X_train).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X_train.columns, orientation='left', leaf_font_size=16)
plt.show()


def get_oob(df):
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, max_samples=50_000, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5,
                          max_samples=50_000, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


to_drop = [
    "Hydraulics_Flow",
    "Grouser_Tracks",
    "Coupler_System",
    "sale_year",
    "sale_elapsed",
    "ProductGroup",
    "ProductGroupDesc",
    "fiBaseModel",
    "fiModelDesc"
]


for c in to_drop:
    print(c, get_oob(X_train.drop(c, axis=1)))


to_drop = ['sale_year', 'fiBaseModel', 'Grouser_Tracks', 'ProductGroup']
get_oob(X_train.drop(to_drop, axis=1))


!mkdir tmp


np.save('tmp/keep_cols.npy', np.array(X_train.columns))


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


%pip install -qq scikit-misc


X_train, y_train, proc_pipeline = prepare_dataframe_for_modeling(raw_train, 'SalePrice', max_n_cat=7)
X_valid, y_valid, _ = prepare_dataframe_for_modeling(raw_valid, 'SalePrice', pipeline=proc_pipeline)


X_train = clean_col_names(X_train)
X_valid = clean_col_names(X_valid)


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, max_samples=50_000, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


df_raw.plot('YearMade', 'sale_elapsed', 'scatter', alpha=0.01, figsize=(10,8));


x_all = df_raw[df_raw.YearMade>1930].sample(n=500, replace=False, random_state=42)


from plotnine import *

ggplot(x_all, aes('YearMade', 'SalePrice'))+stat_smooth(se=True, method='loess')


import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.inspection import PartialDependenceDisplay

def plot_pdp_sklearn(model, X, feature, feature_name=None,
                     target=None, plot_lines=True, subsample=None,
                     figsize=(10, 8), dpi=300, ncols=2, show_marginal=True, **kwargs):
    """
    Generate partial dependence plots (PDPs) for a given feature or features using scikit-learn.
    
    When two features are provided (as a non-nested list of exactly two elements) and
    show_marginal is True, a composite figure is produced that includes:
      - A 2D PDP (joint effect of the two features).
      - The individual 1D PDPs for each feature.
    
    For other cases (a single feature or multiple features), the function plots the PDP(s)
    in a grid (using ncols for layout).
    
    Parameters:
        model : estimator object
            A trained model used for predictions.
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            Data used to compute the partial dependence.
        feature : int, str, or list
            The feature(s) for which to compute the PDP.
            For a 2D PDP, pass a non-nested list of two features (e.g. ["feature1", "feature2"]).
            For multiple 1D PDPs, pass a list of features.
        feature_name : str, optional
            A display name for the feature (used for a single PDP plot);
            if not provided, the value of `feature` is used.
        target : int, str, or None, default=None
            For multi-class classifiers, the target class to plot; otherwise, leave as None.
        plot_lines : bool, default True
            If True, plots individual ICE curves along with the average PDP (i.e. kind="both");
            otherwise, plots only the average PDP (i.e. kind="average").
        subsample : int, float, or None, default None
            Controls the number of ICE curves to plot:
              - If an int, at most that many ICE curves will be plotted.
              - If a float between 0 and 1, it represents the fraction of ICE curves to plot.
              - If None, all ICE curves are plotted.
        figsize : tuple, default (10, 8)
            Size of the matplotlib figure.
        dpi : int, default 300
            Resolution of the figure.
        ncols : int, default 2
            Number of columns in the grid when plotting multiple PDP plots.
        show_marginal : bool, default True
            When exactly two features are provided, if True, also display the individual
            1D PDP plots for each feature along with the 2D PDP.
        **kwargs : dict
            Additional keyword arguments passed to
            `PartialDependenceDisplay.from_estimator`.
    
    Returns:
        If composite (2D PDP with marginal 1D PDPs):
            A dictionary with keys:
              - 'fig': the matplotlib Figure.
              - 'axes': a tuple of (ax_1d_feature1, ax_1d_feature2, ax_2d).
              - 'pdp_1d_feature1': PDP display for the first feature.
              - 'pdp_1d_feature2': PDP display for the second feature.
              - 'pdp_2d': PDP display for the 2D plot.
        Otherwise:
            The PartialDependenceDisplay object returned by scikit-learn.
    """
    # Determine the plotting mode based on the input.
    if (isinstance(feature, list) and len(feature) == 2 and 
        not isinstance(feature[0], (list, tuple)) and show_marginal):
        # Composite plot: two 1D PDPs and one 2D PDP.
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # Create a gridspec with 2 rows and 2 columns:
        # - Top row: two 1D PDP plots (one per feature).
        # - Bottom row: one 2D PDP spanning both columns.
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        kind = 'both' if plot_lines else 'average'
        feat_names = X.columns if hasattr(X, 'columns') else None
        
        # Plot 1D PDP for the first feature.
        display_1d_1 = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[feature[0]],
            feature_names=feat_names,
            kind=kind,
            subsample=subsample,
            target=target,
            ax=ax1,
            **kwargs
        )
        ax1.set_title(f'PDP of {feature[0]}')
        
        # Plot 1D PDP for the second feature.
        display_1d_2 = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[feature[1]],
            feature_names=feat_names,
            kind=kind,
            subsample=subsample,
            target=target,
            ax=ax2,
            **kwargs
        )
        ax2.set_title(f'PDP of {feature[1]}')
        
        # Plot 2D PDP for the feature pair.
        display_2d = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[feature],  # Note: passing a list of two features produces a 2D plot.
            feature_names=feat_names,
            kind=kind,
            subsample=subsample,
            target=target,
            ax=ax3,
            **kwargs
        )
        ax3.set_title(f'2D PDP of {feature[0]} and {feature[1]}')
        
        fig.suptitle(f'Composite Partial Dependence Plots', fontsize=16)
        plt.tight_layout()
        return {
            'fig': fig,
            'axes': (ax1, ax2, ax3),
            'pdp_1d_feature1': display_1d_1,
            'pdp_1d_feature2': display_1d_2,
            'pdp_2d': display_2d
        }
    else:
        # For single features, or when not showing marginals for 2D.
        # Decide what features to plot.
        if isinstance(feature, list):
            if len(feature) == 1:
                features_to_plot = feature
                title = f'PDP of {feature[0]}'
            elif len(feature) == 2 and not isinstance(feature[0], (list, tuple)):
                # If two features are passed but show_marginal is False, plot only the 2D PDP.
                features_to_plot = [feature]
                title = f'2D PDP of {feature[0]} and {feature[1]}'
            else:
                features_to_plot = feature
                title = 'PDP for multiple features'
        else:
            features_to_plot = [feature]
            title = f'PDP of {feature}'
        
        kind = 'both' if plot_lines else 'average'
        feat_names = X.columns if hasattr(X, 'columns') else None
        
        # Determine the subplot layout.
        if len(features_to_plot) == 1:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            n_plots = len(features_to_plot)
            nrows = math.ceil(n_plots / ncols)
            fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
            if isinstance(ax, np.ndarray):
                ax = ax.flatten()
        
        display = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=features_to_plot,
            feature_names=feat_names,
            kind=kind,
            subsample=subsample,
            target=target,
            ax=ax,
            **kwargs
        )
        
        # Set titles appropriately.
        if len(features_to_plot) == 1:
            ax.set_title(title)
        else:
            fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return display


x = X_train[X_train.YearMade>1930].sample(n=500, replace=False, random_state=42)


plot_pdp_sklearn(m, x, 'YearMade', 'SalePrice', plot_lines=True)


plot_pdp_sklearn(m, x, ['sale_elapsed', 'YearMade'], 'SalePrice', plot_lines=False)


df_keep = df_raw[list(to_keep) + ['SalePrice']].copy()
df_keep = df_keep.drop(to_drop, axis=1)


df_keep.shape


df_raw.YearMade[df_raw.YearMade < 1950] = 1950
df_keep['age'] = df_raw['age'] = df_raw.sale_year - df_raw.YearMade


n_valid = 12000
n_trn = len(df_raw)-n_valid
keep_train, keep_valid = split_vals(df_keep, n_trn)


X_train, y_train, proc_pipeline = prepare_dataframe_for_modeling(keep_train, 'SalePrice')
X_valid, y_valid, _ = prepare_dataframe_for_modeling(keep_valid, 'SalePrice', pipeline=proc_pipeline)


X_train = clean_col_names(X_train)
X_valid = clean_col_names(X_valid)


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)


plot_fi(rf_feat_importance(m, X_train));


df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext.is_valid[:n_trn] = 0


X, y, proc_pipeline = prepare_dataframe_for_modeling(df_ext, 'is_valid')


X = clean_col_names(X)


from sklearn.ensemble import RandomForestClassifier

m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, max_samples=50_000, n_jobs=-1, oob_score=True)
m.fit(X, y);
m.oob_score_


fi = rf_feat_importance(m, X); fi[:10]


X_train['SalesID'].nunique() == X_train.shape[0]
X_valid['SalesID'].nunique() == X_valid.shape[0]


import matplotlib.pyplot as plt

n = 300
plt.plot(np.arange(n), X_train["SalesID"].iloc[:n]);


feats=['SalesID', 'sale_elapsed', 'MachineID']
(X_train[feats]/1000).describe()


(X_valid[feats]/1000).describe()


X.drop(feats, axis=1, inplace=True)


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, max_samples=50_000, n_jobs=-1, oob_score=True)
m.fit(X, y);
m.oob_score_


fi = rf_feat_importance(m, X); fi[:10]


X_train = X_train.drop(feats, axis=1)
X_valid = X_valid.drop(feats, axis=1)


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, max_samples=50_000, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train);


print_score(m)


to_drop = [
    "age",
    "YearMade",
    "sale_day_of_year",
]


def get_score(c):
    X_train_tmp = X_train.drop(c, axis=1)
    X_valid_tmp = X_valid.drop(c, axis=1)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, max_samples=50_000, n_jobs=-1, oob_score=True)
    m.fit(X_train_tmp, y_train);
    res = [rmse(m.predict(X_train_tmp), y_train), rmse(m.predict(X_valid_tmp), y_valid),
                m.score(X_train_tmp, y_train), m.score(X_valid_tmp, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    return res


for c in to_drop:
    print(c, get_score(c))


X_train = X_train.drop("sale_day_of_year", axis=1)
X_valid = X_valid.drop("sale_day_of_year", axis=1)


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train);


print_score(m)




