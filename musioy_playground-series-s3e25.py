# %load ../utils/config.py
!pip install -q kaleido
import glob
import operator
import os
import shutil
import subprocess
import sys
import warnings
from array import array
from collections import defaultdict, namedtuple
from copy import copy
from functools import partial, singledispatch
from itertools import chain, combinations, product
from pathlib import Path
from time import strftime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats
import seaborn as sns
import shap
from colorama import Fore, Style
from IPython.display import HTML, Image, display_html
from lightgbm import LGBMClassifier, LGBMRegressor
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MetaEstimatorMixin,
    OneToOneFeatureMixin,
    TransformerMixin,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression, SGDOneClassSVM
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.metrics import (
    confusion_matrix,
    median_absolute_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.pipeline import FunctionTransformer, make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils.validation import check_array, check_is_fitted
from xgboost import XGBClassifier

# Environment
ON_KAGGLE = os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None

# Colorama settings.
CLR = (Style.BRIGHT + Fore.BLACK) if ON_KAGGLE else (Style.BRIGHT + Fore.WHITE)
RED = Style.BRIGHT + Fore.RED
BLUE = Style.BRIGHT + Fore.BLUE
CYAN = Style.BRIGHT + Fore.CYAN
MAGENTA = Style.BRIGHT + Fore.MAGENTA
RESET = Style.RESET_ALL

# Data Frame and Plotly colors.
FONT_COLOR = "#000000"           # é»‘è‰²å­—ä½“ï¼Œé€šç”¨æ€§å¼º
BACKGROUND_COLOR = "#FFFFFF"     # ç™½è‰²èƒŒæ™¯
GRADIENT_COLOR = "#7F7F7F"       # ä¸­æ€§ç�°ï¼Œç”¨äº�æ¸�å�˜/è¾…åŠ©çº¿

# è“�çº¢å¯¹æ¯”è‰²ç³»ï¼ˆä¸»è‰² + å¼ºå¯¹æ¯” + è¾…åŠ©ï¼‰
COLOR_SCHEME = np.array((
    "#FF9896",   # è¾…åŠ©ç²‰çº¢ï¼ˆæ›´æŸ”å’Œçš„çº¢è‰²å±‚æ¬¡
    "#D62728",  # å¼ºå¯¹æ¯”çº¢ï¼ˆé†’ç›®ã€�å¼ºè°ƒï¼‰
    "#1F77B4"  # ä¸»è‰²è“�ï¼ˆå†·é�™ã€�ç§‘æŠ€æ„Ÿï¼‰
))

# Ticks size for plotly and matplotlib.
TICKSIZE = 11

# Set Plotly theme.
pio.templates["minimalist"] = go.layout.Template(
    layout=go.Layout(
        font_family="Open Sans",
        font_color=FONT_COLOR,
        title_font_size=20,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        xaxis=dict(tickfont_size=TICKSIZE, titlefont_size=TICKSIZE, showgrid=False),
        yaxis=dict(tickfont_size=TICKSIZE, titlefont_size=TICKSIZE, showgrid=False),
        width=840,
        height=540,
        legend=dict(yanchor="bottom", xanchor="right", orientation="h", title=""),
    ),
    layout_colorway=COLOR_SCHEME,
)
pio.templates.default = "plotly+minimalist"

MATPLOTLIB_THEME = {
    "axes.labelcolor": FONT_COLOR,
    "axes.labelsize": TICKSIZE,
    "axes.facecolor": BACKGROUND_COLOR,
    "axes.titlesize": 14,
    "axes.grid": False,
    "xtick.labelsize": TICKSIZE,
    "xtick.color": FONT_COLOR,
    "ytick.labelsize": TICKSIZE,
    "ytick.color": FONT_COLOR,
    "figure.facecolor": BACKGROUND_COLOR,
    "figure.edgecolor": BACKGROUND_COLOR,
    "figure.titlesize": 14,
    "figure.dpi": 72,  # Locally Seaborn uses 72, meanwhile Kaggle 96.
    "text.color": FONT_COLOR,
    "font.size": TICKSIZE,
    "font.family": "Serif",
}
sns.set_theme(rc=MATPLOTLIB_THEME)

# Define Data Frame theme.
CELL_HOVER = {  # for row hover use <tr> instead of <td>
    "selector": "td:hover",
    "props": f"background-color: {BACKGROUND_COLOR}",
}
TEXT_HIGHLIGHT = {
    "selector": "td",
    "props": f"color: {FONT_COLOR}; font-weight: bold",
}
INDEX_NAMES = {
    "selector": ".index_name",
    "props": f"font-weight: normal; background-color: {BACKGROUND_COLOR}; color: {FONT_COLOR};",
}
HEADERS = {
    "selector": "th:not(.index_name)",
    "props": f"font-weight: normal; background-color: {BACKGROUND_COLOR}; color: {FONT_COLOR};",
}
DF_STYLE = (INDEX_NAMES, HEADERS, TEXT_HIGHLIGHT)
DF_CMAP = sns.light_palette(GRADIENT_COLOR, as_cmap=True)

# Html style for table of contents, code highlight and url.
HTML_STYLE = """
    <style>
    code {
        background: rgba(42, 53, 125, 0.10) !important;
        border-radius: 4px !important;
    }
    a {
        color: rgba(123, 171, 237, 1.0) !important;
    }
    ol.numbered-list {
        counter-reset: item;
    }
    ol.numbered-list li {
        display: block;
    }
    ol.numbered-list li:before {
        content: counters(item, '.') '. ';
        counter-increment: item;
    }
    </style>
"""


# Utility functions.
def download_from_kaggle(expr, /, data_dir=None):
    """Download all files from the Kaggle competition/dataset.

    Args:
        expr: Match expression to be used by kaggle API, e.g.
            "kaggle competitions download -c competition" or
            "kaggle datasets download -d user/dataset".
        data_dir: Optional. Directory path where to save files. Default to `None`,
        which means that files will be downloaded to `data` directory.

    Notes:
        If the associated files already exists, then it does nothing.
    """

    if data_dir is None:
        data_dir = Path("data/")
    else:
        data_dir = Path(data_dir)

    match expr.split():
        case ["kaggle", _, "download", *args] if args:
            data_dir.mkdir(parents=True, exist_ok=True)
            filename = args[-1].split("/")[-1] + ".zip"
            if not (data_dir / filename).is_file():
                subprocess.run(expr)
                shutil.unpack_archive(filename, data_dir)
                shutil.move(filename, data_dir)
        case _:
            raise SyntaxError("Invalid expression!")


def get_interpolated_colors(color1, color2, /, n_colors=1):
    """Return `n_colors` colors in HEX format, interpolated beetwen `color1` and `color2`.

    Args:
        color1: Initial HEX color to be interpolated from.
        color2: Final HEX color to be interpolated from.
        n_colors: Optional. Number of colors to be interpolated between `color1`
            and `color2`. Default to 1.

    Returns:
        colors: List of colors interpolated between `color1` and `color2`.
    """

    def interpolate(color1, color2, t):
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02X}{g:02X}{b:02X}"

    return [interpolate(color1, color2, k / (n_colors + 1)) for k in range(1, n_colors + 1)]


def get_pretty_frame(frame, /, gradient=False, formatter=None, precision=3, repr_html=False):
    stylish_frame = frame.style.set_table_styles(DF_STYLE).format(
        formatter=formatter, precision=precision
    )
    if gradient:
        stylish_frame = stylish_frame.background_gradient(DF_CMAP)  # type: ignore
    if repr_html:
        stylish_frame = stylish_frame.set_table_attributes("style='display:inline'")._repr_html_()
    return stylish_frame


def numeric_descr(frame, /):
    return (
        frame.describe(percentiles=(0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99))
        .T.drop("count", axis=1)
        .rename(columns=str.title)
    )


def frame_summary(frame, /):
    missing_vals = frame.isna().sum()
    missing_vals_ratio = missing_vals / len(frame)
    unique_vals = frame.apply(lambda col: len(col.unique()))
    most_freq_count = frame.apply(lambda col: col.value_counts().iloc[0])
    most_freq_val = frame.mode().iloc[:1].T.squeeze()
    unique_ratio = unique_vals / len(frame)
    freq_count_ratio = most_freq_count / len(frame)

    return pd.DataFrame(
        {
            "Dtype": frame.dtypes,
            "MissingValues": missing_vals,
            "MissingValuesRatio": missing_vals_ratio,
            "UniqueValues": unique_vals,
            "UniqueValuesRatio": unique_ratio,
            "MostFreqValue": most_freq_val,
            "MostFreqValueCount": most_freq_count,
            "MostFreqValueCountRatio": freq_count_ratio,
        }
    )


def check_categories_alignment(frame1, frame2, /, out_color=BLUE):
    print(CLR + "The same categories in training and test datasets?\n")
    cat_features = frame2.select_dtypes(include="object").columns.to_list()

    for feature in cat_features:
        frame1_unique = set(frame1[feature].unique())
        frame2_unique = set(frame2[feature].unique())
        same = np.all(frame1_unique == frame2_unique)
        print(CLR + f"{feature:25s}", out_color + f"{same}")


def get_lower_triangular_frame(frame, /):
    if not frame.shape[0] == frame.shape[1]:
        raise ValueError(f"{type(frame)!r} is not square frame")
    lower_triu = np.triu(np.ones_like(frame, dtype=bool))
    frame = frame.mask(lower_triu)
    return frame.dropna(axis="index", how="all").dropna(axis="columns", how="all")


def save_and_show_fig(fig, filename, /, img_dir=None, format="png"):
    if img_dir is None:
        img_dir = Path("images")
    if not isinstance(img_dir, Path):
        raise TypeError("The `img_dir` argument must be `Path` instance!")

    img_dir.mkdir(parents=True, exist_ok=True)
    fig_path = img_dir / (filename + "." + format)
    fig.write_image(fig_path)

    return Image(fig.to_image(format=format))


def get_n_rows_and_axes(n_features, n_cols, /, start_at=1):
    n_rows = int(np.ceil(n_features / n_cols))
    current_col = range(start_at, n_cols + start_at)
    current_row = range(start_at, n_rows + start_at)
    return n_rows, tuple(product(current_row, current_col))


def get_kde_estimation(
    series,
    *,
    bw_method=None,
    weights=None,
    percentile_range=(0, 100),
    estimate_points_frac=0.1,
    space_extension_frac=0.01,
    cumulative=False,
):
    """Return pdf dictionary for set of points using gaussian kernel density estimation.

    Args:
        series: The dataset with which `stats.gaussian_kde` is initialized.
        bw_method: Optional. The method used to calculate the estimator bandwidth.
        This can be 'scott', 'silverman', a scalar constant or a callable. If a scalar,
        this will be used directly as `kde.factor`. If a callable, it should take
        a `stats.gaussian_kde` instance as only parameter and return a scalar.
        If `None` (default), 'scott' is used.
        weights: Optional. Weights of datapoints. This must be the same shape as dataset.
        If `None` (default), the samples are assumed to be equally weighted.
        percentile_range: Optional. Percentile range of the `series` to create estimated space.
        By default (0, 100) range is used.
        estimate_points_frac: Optional. Fraction of `series` length to create linspace for
        estimated points.
        space_extension_frac: Optional. Estimation space will be extended by
        `space_extension_frac * len(series)` for both edges.
        cumulative: Optional. Whether to calculate cdf. Default to `False`.

    Returns:
        Dictionary with kde space, values, and cumulative values if `cumulative` is `True`.
    """

    series = pd.Series(series).dropna()
    kde = stats.gaussian_kde(series, bw_method=bw_method, weights=weights)
    start, stop = np.percentile(series, percentile_range)

    n_points = int(estimate_points_frac * len(series))
    n_extend = int(space_extension_frac * len(series))

    if n_extend > 0:
        dx = (stop - start) / (n_points - 1)
        start, stop = start - n_extend * dx, stop + n_extend * dx

    kde_space = np.linspace(start, stop, n_points)
    kde_vals = kde.evaluate(kde_space)
    results = {"space": kde_space, "vals": kde_vals}

    if cumulative:
        kde_vals_cum = np.cumsum(kde_vals)
        return results | {"vals_cumulative": kde_vals_cum / kde_vals_cum.max()}

    return results


def unit_norm(x):
    return x / np.sum(x)


# Html highlight. Must be included at the end of all imports!
HTML(HTML_STYLE)


competition = "playground-series-s3e25"
expr = f"kaggle competitions download -c {competition}"

if not ON_KAGGLE:
    download_from_kaggle(expr)
    train_path = "data/train.csv"
    test_path = "data/test.csv"
else:
    train_path = f"/kaggle/input/{competition}/train.csv"
    test_path = f"/kaggle/input/{competition}/test.csv"

train = pd.read_csv(train_path, index_col="id")  # .rename(columns=str.title)
test = pd.read_csv(test_path, index_col="id")  # .rename(columns=str.title)


get_pretty_frame(train.head())


train.info(verbose=False)


test.info(verbose=False)


fig = px.histogram(
    train,
    x="Hardness",
    histnorm="probability",
    marginal="box",
    height=460,
    title="Distribution of Hardness - Target Variable<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "This feature seems like a quantized one (many repetitive values)</span>",
)
fig.update_yaxes(title="Probability", row=1)
save_and_show_fig(fig, "hardness_distribution")


print(CLR + "Training Dataset:")
train_summary = frame_summary(train)
get_pretty_frame(train_summary, gradient=True)


print(CLR + "Test Dataset:")
test_summary = frame_summary(test)
get_pretty_frame(test_summary, gradient=True)


print(CLR + "Training Dataset:")
train_num_descr = numeric_descr(train)
get_pretty_frame(train_num_descr, gradient=True)


print(CLR + "Test Dataset:")
test_num_descr = numeric_descr(test)
get_pretty_frame(test_num_descr, gradient=True)


train_av = train.drop("Hardness", axis=1).assign(AV=0)
test_av = test.assign(AV=1)

data_av = pd.concat((train_av, test_av), ignore_index=True)
data_av = data_av.sample(frac=1.0, random_state=42)

X = data_av.drop("AV", axis=1)
y = data_av.AV

y_proba = cross_val_predict(
    estimator=make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
    X=X,
    y=y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=19937),
    method="predict_proba",
)

av_scores = {
    "ConfusionMatrix": confusion_matrix(y, y_proba.argmax(axis=1)),
    "FPR-TPR-Threshold": roc_curve(y, y_proba[:, 1]),
    "ROC-AUC": roc_auc_score(y, y_proba[:, 1]),
}


fig = go.Figure()
fig.add_scatter(
    x=av_scores["FPR-TPR-Threshold"][0],
    y=av_scores["FPR-TPR-Threshold"][1],
    name="AV Result",
    mode="lines",
    line_color=COLOR_SCHEME[2],
)
fig.add_scatter(
    x=[0, 1],
    y=[0, 1],
    name="Random Guess",
    mode="lines",
    line=dict(dash="longdash", color=COLOR_SCHEME[0]),
)
fig.add_annotation(
    x=0.05,
    y=0.85,
    align="left",
    xanchor="left",
    text=f"<b>AV ROC-AUC: {av_scores['ROC-AUC']:.5f}<br>" "Conclusion? The same distribution.",
    showarrow=False,
    font_size=14,
)
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
    range=(-0.01, 1.01),
    title="True Positive Rate (Recall)",
)
fig.update_xaxes(
    scaleanchor="y",
    scaleratio=1,
    range=(-0.01, 1.01),
    title="False Positive Rate (Fall-Out)",
)
fig.update_layout(
    title="Adversarial Validation Results<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Training and test datasets are indistinguishable</span>",
    width=540,
    legend=dict(y=1.0, x=1.2),
)
save_and_show_fig(fig, "adversarial_validation")


features = test.columns.to_list()

n_cols = 3
n_rows, axes = get_n_rows_and_axes(len(features), n_cols)

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    y_title="Probability Density",
    horizontal_spacing=0.1,
    vertical_spacing=0.1,
).update_annotations(font_size=14)

for frame, color, group in zip((train, test), (COLOR_SCHEME[0], COLOR_SCHEME[2]), ("Train", "Test")):
    for k, (var, (row, col)) in enumerate(zip(features, axes), start=1):
        start, end = np.percentile(frame[var], (1, 99))
        fig.add_histogram(
            x=frame[var],
            xbins=go.histogram.XBins(start=start, end=end),
            histnorm="probability density",
            marker_color=color,
            marker_line_width=0,
            opacity=0.8,
            name=group,
            legendgroup=group,
            showlegend=k == 1,
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=f"<b>{var}</b>", row=row, col=col)

fig.update_layout(
    width=840,
    height=740,
    legend=dict(y=1, x=1),
    title="Training & Test Feature Histograms<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Restricted to (1, 99) percentile range to avoid showing extreme outliers</span>",
    bargap=0,
    bargroupgap=0,
)
save_and_show_fig(fig, "histograms")


n_cols = 3
n_rows, axes = get_n_rows_and_axes(len(features), n_cols)

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    y_title="Probability Density",
    horizontal_spacing=0.1,
    vertical_spacing=0.1,
).update_annotations(font_size=14)

for frame, color, group in zip((train, test), (COLOR_SCHEME[0], COLOR_SCHEME[2]), ("Train", "Test")):
    for k, (var, (row, col)) in enumerate(zip(features, axes), start=1):
        kde = get_kde_estimation(frame[var], percentile_range=(1, 99))
        fig.add_scatter(
            x=kde["space"],
            y=kde["vals"],
            line=dict(dash="solid", color=color, width=1),
            fill="tozeroy",
            name=group,
            legendgroup=group,
            showlegend=k == 1,
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=f"<b>{var}</b>", row=row, col=col)

fig.update_layout(
    width=840,
    height=740,
    legend=dict(y=1, x=1),
    title="Training & Test Feature KDEs<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Restricted to (1, 99) percentile range to avoid showing extreme outliers</span>",
)
save_and_show_fig(fig, "kdes")


pearson_corr = train.corr(method="pearson")
lower_triu_corr = get_lower_triangular_frame(pearson_corr)
colormap = tuple(zip((0, 0.5, 1), COLOR_SCHEME[[1, 0, 2]]))

heatmap = go.Heatmap(
    z=lower_triu_corr,
    x=lower_triu_corr.columns,
    y=lower_triu_corr.index,
    text=lower_triu_corr.fillna(""),
    texttemplate="%{text:.2f}",
    xgap=4,
    ygap=4,
    showscale=True,
    colorscale=colormap,
    colorbar_len=1.02,
    hoverinfo="none",
)
fig = go.Figure(heatmap)
fig.update_layout(
    title="Training Dataset - Lower Triangle of Correlation Matrix (Pearson)<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Here we have several strongly correlated features. "
    "Are they really correlated or it's impact of outliers?</span>",
    yaxis_autorange="reversed",
    width=840,
    height=840,
)
save_and_show_fig(fig, "pearson_corr_matrix")


abs_corr = (
    lower_triu_corr.abs()
    .unstack()
    .sort_values(ascending=False)  # type: ignore
    .rename("Absolute Pearson Correlation")
    .to_frame()
    .reset_index(names=["Feature 1", "Feature 2"])
    .dropna()
    .round(5)
)

with pd.option_context("display.max_rows", 10):
    print(abs_corr)


dissimilarity = 1 - np.abs(pearson_corr)

fig = ff.create_dendrogram(
    dissimilarity,
    labels=pearson_corr.columns,
    orientation="left",
    colorscale=px.colors.sequential.Greys[3:],
    # squareform() returns lower triangular in compressed form - as 1D array.
    linkagefun=lambda x: linkage(squareform(dissimilarity), method="complete"),
)
fig.update_xaxes(showline=False, title="Distance", ticks="", range=[-0.03, 1.05])
fig.update_yaxes(showline=False, ticks="")
fig.update_layout(
    title="Training Dataset - Hierarchical Clustering using Correlation Matrix (Pearson)<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Small distance for some features. PCA maybe?</span>",
    height=460,
    width=840,
)
fig.update_traces(line_width=1.5, opacity=1)
save_and_show_fig(fig, "hierarchical_clustering")


n_cols, n_features = 3, 6
n_rows, axes = get_n_rows_and_axes(n_features, n_cols)

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    horizontal_spacing=0.1,
    vertical_spacing=0.15,
)

for (row, col), (feature1, feature2, corr) in zip(axes, abs_corr[:n_features].to_numpy()):
    fig.add_scatter(
        x=train[feature1],
        y=train[feature2],
        mode="markers",
        name="",
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text=feature1, row=row, col=col)
    fig.update_yaxes(title_text=feature2, row=row, col=col)

fig.update_layout(
    title="Training Dataset - Highly Linear Correlated Pairs<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Actual high correlation corresponds to the first pair only (perhaps the second too)</span>",
    width=840,
    height=540,
    showlegend=False,
)
fig.update_traces(
    marker=dict(size=1, symbol="x-thin", line=dict(width=1.5, color=COLOR_SCHEME[0])),
)
save_and_show_fig(fig, "highly_correlated_scatter_plots")


n_cols = 3
n_rows, axes = get_n_rows_and_axes(len(features), n_cols)

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    y_title="Hardness - Target Variable",
    horizontal_spacing=0.07,
    vertical_spacing=0.1,
)
fig.update_annotations(font_size=14, yshift=-45)

for (row, col), feature in zip(axes, features):
    fig.add_scatter(
        x=train[feature],
        y=train.Hardness,
        mode="markers",
        name=feature,
        row=row,
        col=col,
    )
    fig.update_xaxes(
        title_text=f"<b>{feature}</b>",
        row=row,
        col=col,
    )
    if not col == 1:
        fig.update_yaxes(showticklabels=False, row=row, col=col)

fig.update_layout(
    title="Training Dataset - Hardness vs Remaining Features<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Lack of clear dependencies at first sight</span>",
    width=840,
    height=840,
    showlegend=False,
)
fig.update_traces(
    marker=dict(size=1, symbol="x-thin", line=dict(width=1.5, color=COLOR_SCHEME[0])),
)
save_and_show_fig(fig, "scatter_plots")


n_cols = 3
n_rows, axes = get_n_rows_and_axes(len(features), n_cols)

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    y_title="Observed Values",
    x_title="Theoretical Quantiles",
    horizontal_spacing=0.1,
    vertical_spacing=0.1,
)
fig.update_annotations(font_size=14, yshift=-45)

for (row, col), feature in zip(axes, features):
    (osm, osr), (slope, intercept, R) = stats.probplot(train[feature].dropna(), rvalue=True)
    x_theory = np.array([osm[0], osm[-1]])
    y_theory = intercept + slope * x_theory
    R2 = f"R\u00b2 = {R * R:.2f}"
    fig.add_scatter(x=osm, y=osr, mode="markers", row=row, col=col, name=feature)
    fig.add_scatter(x=x_theory, y=y_theory, mode="lines", row=row, col=col)
    fig.add_annotation(
        x=-1.25,
        y=osr[-1] * 0.95,
        text=R2,
        showarrow=False,
        row=row,
        col=col,
        font_size=11,
    )
    fig.update_xaxes(
        title_text=f"<b>{feature}</b>",
        row=row,
        col=col,
    )

fig.update_layout(
    title="Training Dataset - Probability Plots against Normal Distribution<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Results in 'allelectrons_Total' and 'density_Total' are perturbed by some outliers</span>",
    width=840,
    height=840,
    showlegend=False,
)
fig.update_traces(
    marker=dict(size=1, symbol="x-thin", line=dict(width=2, color=COLOR_SCHEME[2])),
    line_color=COLOR_SCHEME[0],
)
save_and_show_fig(fig, "probability_plots")


r2_scores = pd.DataFrame(index=("Original", "YeoJohnson", "Log", "Sqrt"))

for feature in features:
    orig = train[feature].dropna()
    _, (*_, R_orig) = stats.probplot(orig, rvalue=True)
    _, (*_, R_yeojohn) = stats.probplot(stats.yeojohnson(orig)[0], rvalue=True)
    _, (*_, R_log) = stats.probplot(np.log1p(orig), rvalue=True)
    _, (*_, R_sqrt) = stats.probplot(np.sqrt(orig), rvalue=True)

    r2_scores[feature] = (
        R_orig * R_orig,
        R_yeojohn * R_yeojohn,
        R_log * R_log,
        R_sqrt * R_sqrt,
    )

r2_scores = r2_scores.transpose()
r2_scores["Winner"] = r2_scores.idxmax(axis=1)
get_pretty_frame(r2_scores)


density_Total_transformed = stats.yeojohnson(train.density_Total.dropna())[0]
(osm, osr), (slope, intercept, R) = stats.probplot(density_Total_transformed, rvalue=True)
x_theory = np.array([osm[0], osm[-1]])
y_theory = intercept + slope * x_theory

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Probability Plot against Normal Distribution", "Distribution"],
    horizontal_spacing=0.15,
)

fig.add_scatter(x=osm, y=osr, mode="markers", row=1, col=1, name="YeoJohnson(density_Total)")
fig.add_scatter(x=x_theory, y=y_theory, mode="lines", row=1, col=1)
fig.add_annotation(
    x=-1.25,
    y=osr[-1] * 0.75,
    text=f"R\u00b2 = {R * R:.3f}",
    showarrow=False,
    row=1,
    col=1,
)
fig.update_yaxes(title_text="Observed Values", row=1, col=1)
fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=1)
fig.update_traces(
    marker=dict(size=1, symbol="x-thin", line=dict(width=2, color=COLOR_SCHEME[2])),
    line_color=COLOR_SCHEME[0],
)

fig.add_histogram(
    x=density_Total_transformed,
    xbins=go.histogram.XBins(size=0.1),
    marker_color=COLOR_SCHEME[0],
    name="YeoJohnson(density_Total)",
    histnorm="probability density",
    row=1,
    col=2,
)
fig.update_yaxes(title_text="Probability Density", row=1, col=2)
fig.update_xaxes(title_text="YeoJohnson(density_Total)", row=1, col=2)

fig.update_layout(
    title="Yeo-Johnson Transformation for 'density_Total' Feature",
    showlegend=False,
    width=840,
    height=460,
    bargap=0.2,
)
fig.update_annotations(font_size=14)
save_and_show_fig(fig, "density_Total_after_transform")


X = train.drop("Hardness", axis=1)
y = train.Hardness

DefaultDecisionTreeRegressor = partial(
    DecisionTreeRegressor,
    criterion="absolute_error",  # Watch out on learning time complexity.
    random_state=42,
    max_depth=3,
)

tree = DefaultDecisionTreeRegressor().fit(X, y)


plt.figure(figsize=(11.5, 5.5), tight_layout=True)
plot_tree(
    decision_tree=tree,
    feature_names=tree.feature_names_in_.tolist(),
    filled=False,
    rounded=True,
    impurity=False,
    proportion=True,
    node_ids=True,
    ax=plt.gca(),
    fontsize=11,
)
plt.title("Decision Process in Decision Tree (depth = 3)")
plt.savefig("images/decision_process_in_tree")
plt.show()


for depth in range(2, 7):
    tree.set_params(max_depth=depth).fit(X, y)
    considered_features = tree.tree_.feature[tree.tree_.feature != -2]  # type: ignore # -2 means a leaf
    used_features = np.unique(considered_features)
    used_features = X.columns[used_features].to_list()
    print(CLR + f"Features at depth {depth}: {RED}{len(used_features):<5}", end="")
    tree_cv_results = -cross_val_score(
        estimator=tree,
        X=X,
        y=y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_median_absolute_error",
        n_jobs=2,
    )
    mean, std = tree_cv_results.mean(), tree_cv_results.std()
    print(CLR + "MedAE:", RED + f"{mean:.2f} \u00b1 {std:.2f}")


DefaultLGBMRegressor = partial(
    LGBMRegressor,
    objective="regression_l1",
    random_state=42,
    verbose=-1,
)


np.random.seed(42)
seeds = np.random.randint(0, 19937, size=5)

X = train.drop("Hardness", axis=1)
y = train.Hardness

lgbm = DefaultLGBMRegressor()
importances = []

for seed in seeds:
    np.random.seed(seed)
    X["RANDOM_1"] = np.random.normal(size=len(X))
    X["RANDOM_2"] = np.random.normal(size=len(X))
    X["RANDOM_3"] = np.random.normal(size=len(X))
    X["RANDOM_4"] = np.random.normal(size=len(X))
    X["RANDOM_5"] = np.random.normal(size=len(X))

    lgbm.set_params(random_state=seed).fit(X, y)
    importances.append(unit_norm(lgbm.feature_importances_))

importances = (
    pd.DataFrame({"Feature": X.columns, "Importance": np.array(importances).mean(axis=0)})
    .sort_values(by="Importance", ascending=False)
    .reset_index(drop=True)
)


fig = px.bar(
    importances,
    x="Importance",
    y="Feature",
    height=460,
    width=840,
    title="Feature Importances in LGBM Regressor - Under Reduction in MAE Criterion<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Used a default version of LGBM with L1 objective. "
    "Most of features are more important than random ones</span>",
)
fig.update_yaxes(categoryorder="total ascending", title="")
fig.update_xaxes(range=(-0.002, 0.11))
fig.update_traces(width=0.7)
save_and_show_fig(fig, "importance_with_mae_reduction")


np.random.seed(42)
seeds = np.random.randint(0, 19937, size=5)

lgbm = DefaultLGBMRegressor()
permutation_medae = defaultdict(list)

for seed in seeds:
    np.random.seed(seed)
    X["RANDOM_1"] = np.random.normal(size=len(X))
    X["RANDOM_2"] = np.random.normal(size=len(X))
    X["RANDOM_3"] = np.random.normal(size=len(X))
    X["RANDOM_4"] = np.random.normal(size=len(X))
    X["RANDOM_5"] = np.random.normal(size=len(X))

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    lgbm.set_params(random_state=seed)

    for k, (train_ids, valid_ids) in enumerate(kfold.split(X, y)):
        X_train, y_train = X.iloc[train_ids], y[train_ids]  # type: ignore
        X_valid, y_valid = X.iloc[valid_ids], y[valid_ids]  # type: ignore

        lgbm.fit(X_train, y_train)
        medae = median_absolute_error(y_valid, lgbm.predict(X_valid))  # type: ignore

        for i, feature in enumerate(X_train.columns):
            X_shuffled = X_valid.copy()
            X_shuffled.iloc[:, i] = np.random.permutation(X_shuffled.iloc[:, i])
            medae_shuffled = median_absolute_error(y_valid, lgbm.predict(X_shuffled))  # type: ignore
            # I assume an increase in MedAE if the attribute is essential.
            permutation_medae[feature].append(((medae_shuffled - medae) / medae) * 100.0)


medae_increase = (
    pd.DataFrame(permutation_medae)
    .mean()
    .sort_values(ascending=False)
    .to_frame(name="Mean MedAE Increase (%)")
    .reset_index(names="Feature")
)

fig = px.bar(
    medae_increase,
    x="Mean MedAE Increase (%)",
    y="Feature",
    height=460,
    width=840,
    title="Mean MedAE Increase in LGBM Regressor within Samples Permutation<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Used a default version of LGBM with L1 objective. "
    "Permutation of the 'allelectrons_Average' severly punishes MedAE</span>",
)
fig.update_yaxes(categoryorder="total ascending", title="")
fig.update_xaxes(range=(-1, 55))
fig.update_traces(width=0.7)
save_and_show_fig(fig, "importance_with_feature_permutation")


np.random.seed(42)
seeds = np.random.randint(0, 19937, size=5)

scaler = StandardScaler()
mutual_info = []

for seed in seeds:
    np.random.seed(seed)
    X["RANDOM_1"] = np.random.normal(size=len(X))
    X["RANDOM_2"] = np.random.normal(size=len(X))
    X["RANDOM_3"] = np.random.normal(size=len(X))
    X["RANDOM_4"] = np.random.normal(size=len(X))
    X["RANDOM_5"] = np.random.normal(size=len(X))

    # Choose of neighbors is subjective.
    mi = mutual_info_regression(X=scaler.fit_transform(X), y=y, n_neighbors=50, random_state=seed)
    mutual_info.append(mi)

mi_importances = (
    pd.DataFrame({"Feature": X.columns, "Mutual Information": np.array(mutual_info).mean(axis=0)})
    .sort_values(by="Mutual Information", ascending=False)
    .reset_index(drop=True)
)


fig = px.bar(
    mi_importances,
    x="Mutual Information",
    y="Feature",
    height=460,
    width=840,
    title="Feature Importances via Mutual Information<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Used 5 different seeds. In this test all available features "
    "are significantly more important than random ones</span>",
)
fig.update_yaxes(categoryorder="total ascending", title="")
fig.update_xaxes(range=(-0.005, 0.3))
fig.update_traces(width=0.7)
save_and_show_fig(fig, "mutual_information")


np.random.seed(42)

X = train.drop("Hardness", axis=1).assign(RANDOM_1=np.random.normal(size=len(train)))
y = train.Hardness

lgbm = DefaultLGBMRegressor().fit(X, y)

fig, axes = plt.subplots(4, 3, figsize=(11.5, 10), tight_layout=True, sharey=True)
plt.suptitle("One-Variable Partial Dependence in LGBM Regressor")
PartialDependenceDisplay.from_estimator(
    estimator=lgbm,  # type: ignore
    X=X,
    features=X.columns.tolist(),
    feature_names=X.columns.tolist(),
    response_method="auto",  # In regression, the response is `predict()` output.
    kind="both",  # PDP and ICE.
    percentiles=(0.01, 0.99),
    subsample=0.5,
    random_state=42,
    n_jobs=-1,
    ice_lines_kw={"color": COLOR_SCHEME[0], "linewidth": 0.2, "alpha": 0.1, "linestyle": "--"},
    pd_line_kw={"color": COLOR_SCHEME[2], "linewidth": 2.0},
    ax=axes.ravel(),  # type: ignore
)

for ax in axes.ravel():
    ax.get_legend().remove()
    if ax not in (axes[0, 0], axes[1, 0], axes[2, 0], axes[3, 0]):
        ax.set_ylabel("")

plt.savefig("images/one_way_partial_dependence")
plt.show()


interaction_pair1 = ["allelectrons_Average", "atomicweight_Average"]
interaction_pair2 = ["zaratio_Average", "ionenergy_Average"]

X = train[np.union1d(interaction_pair1, interaction_pair2)]
y = train.Hardness
lgbm = DefaultLGBMRegressor().fit(X, y)  # type: ignore

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.5, 5.5), tight_layout=True)
plt.suptitle("Two-Variable PDP in LGBM Regressor")
PartialDependenceDisplay.from_estimator(
    estimator=lgbm,  # type: ignore
    X=X,
    features=[interaction_pair1, interaction_pair2],
    feature_names=X.columns.to_list(),
    response_method="auto",  # In regression, the response is `predict()` output.
    percentiles=(0.01, 0.99),
    random_state=42,
    n_jobs=-1,
    contour_kw={"cmap": "pink"},
    ax=axes,  # type: ignore
)

plt.savefig("images/two_way_partial_dependence")
plt.show()


X = train.drop("Hardness", axis=1)
y = train.Hardness

transformer = PowerTransformer(method="yeo-johnson", standardize=True)
X_rescaled = transformer.fit_transform(X)

pca_2d = PCA(n_components=2, random_state=42)
iso_2d = Isomap(n_components=2, n_neighbors=20, n_jobs=-1)
tsne_2d = TSNE(n_components=2, random_state=42, n_jobs=-1)

pca_2d_results = pd.DataFrame(pca_2d.fit_transform(X_rescaled), columns=("x1", "x2")).join(y)
iso_2d_results = pd.DataFrame(iso_2d.fit_transform(X_rescaled), columns=("x1", "x2")).join(y)
tsne_2d_results = pd.DataFrame(tsne_2d.fit_transform(X_rescaled), columns=("x1", "x2")).join(y)


n_cols, n_projections = 3, 3
n_rows, axes = get_n_rows_and_axes(n_projections, n_cols)
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=("PCA", "Isomap", "TSNE"),
    x_title="x1",
    y_title="x2",
    # horizontal_spacing=0.1,
    vertical_spacing=0.1,
)

for (row, col), projection in zip(axes, (pca_2d_results, iso_2d_results, tsne_2d_results)):
    fig.add_scatter(
        x=projection.x1,
        y=projection.x2,
        mode="markers",
        marker=dict(size=1, color=projection.Hardness, coloraxis="coloraxis"),
        row=row,
        col=col,
        showlegend=False,
    )

fig.update_annotations(font_size=14, yshift=-15)
fig.update_coloraxes(
    colorbar=dict(
        title_text="Hardness",
        ticklabelposition="outside bottom",
        orientation="h",
        title_side="bottom",
        yanchor="bottom",
        xanchor="center",
        len=1.02,
        y=-0.5,
        x=0.5,
    ),
    colorscale=colormap,
)
fig.update_layout(
    title="Training Dataset - Dimensionality Reduction with Different Algorithms<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "TSNE (as usual) seems to create the best visuals</span>",
    width=840,
    height=440,
)
save_and_show_fig(fig, "projections_2d")


tsne_3d = TSNE(n_components=3, random_state=42, n_jobs=-1)
tsne_3d_results = pd.DataFrame(tsne_3d.fit_transform(X_rescaled), columns=("x1", "x2", "x3")).join(y)

fig = px.scatter_3d(
    tsne_3d_results,
    x="x1",
    y="x2",
    z="x3",
    color="Hardness",
    color_continuous_scale=colormap,
    opacity=0.5,
    height=840,
    width=840,
    title="Training Dataset - 3D Projection with t-SNE<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Many samples overlap but these with low 'Hardness' seems to "
    "be separated pretty well</span>",
)
fig.update_traces(marker_size=2)
fig.update_coloraxes(colorbar=dict(ticklabelposition="outside bottom"))
fig.show()


def remove_outliers(data, detector):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"'data' must be {pd.DataFrame!r} instance")

    result = detector.fit_predict(data)
    outlier_ids = pd.Series(result == -1, index=data.index, dtype=bool)
    data_ids = pd.Series(np.ones_like(data.index), index=data.index, dtype=bool)
    
    return data[~(outlier_ids & data_ids)]


lgbm = DefaultLGBMRegressor()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
detector = make_pipeline(
    PowerTransformer(method="yeo-johnson", standardize=True),
    LocalOutlierFactor(),
)

hyperparameter = "localoutlierfactor__contamination"
hyperparameter_values = [None] + np.arange(0.01, 0.15, 0.01).tolist()
no_outliers_medae = {}

for k, (train_ids, valid_ids) in enumerate(kfold.split(X, y), start=1):
    X_train, y_train = X.iloc[train_ids], y.iloc[train_ids]
    X_valid, y_valid = X.iloc[valid_ids], y.iloc[valid_ids]

    lgbm.fit(X_train, y_train)
    default_medae = median_absolute_error(y_valid, lgbm.predict(X_valid))  # type:ignore

    for hp_value in hyperparameter_values:
        if hp_value is None:
            no_outliers_medae[f"0 - {k}"] = default_medae
            continue

        detector.set_params(**{hyperparameter: hp_value})
        X_no_outliers = remove_outliers(X_train, detector)
        y_no_outliers = y_train[X_no_outliers.index]

        lgbm.fit(X_no_outliers, y_no_outliers)
        clean_medae = median_absolute_error(y_valid, lgbm.predict(X_valid))  # type:ignore
        no_outliers_medae[f"{hp_value} - {k}"] = clean_medae


detector_medae = pd.DataFrame({"KEY": no_outliers_medae.keys(), "MedAE": no_outliers_medae.values()})
detector_medae[[hyperparameter, "Fold"]] = detector_medae.KEY.str.split("-", expand=True)
default_medae = detector_medae[detector_medae[hyperparameter].astype(float) == 0].MedAE

fig = px.line(
    detector_medae,
    x=hyperparameter,
    y="MedAE",
    facet_row="Fold",
    facet_row_spacing=0.07,
    color_discrete_sequence=COLOR_SCHEME[2:],
    height=640,
    width=840,
    title=f"Influence of '{hyperparameter}' Hyperparameter on MedAE in LGBM<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    f"None of the '{hyperparameter}' values causes "
    "a reduction in MedAE in all folds</span>",
)
for fold, fold_default_medae in enumerate(default_medae):
    fig.add_hline(
        fold_default_medae,
        annotation_text=f"<b>Default MedAE: {fold_default_medae:.3f}</b>",
        annotation_position="bottom left",
        annotation_font_size=12,
        line_width=1.5,
        opacity=0.75,
        line_dash="dot",
        line_color=COLOR_SCHEME[0],
        row=len(default_medae) - fold,  # type:ignore
    )
fig.update_traces(line_width=2)
fig.update_layout(margin_pad=10)
fig.update_xaxes(tickformat=".2f", type="linear")
save_and_show_fig(fig, "outlier_detection")


def clip_numeric_feature(X, lower_bound=None, upper_bound=None):
    X = np.array(X, copy=False)
    return np.clip(X[:, [0]], lower_bound, upper_bound)


def features_interaction(X, op="mul", eps=1e-9):
    ops = {
        "mul": operator.mul,
        "truediv": operator.truediv,
        "add": operator.add,
        "sub": operator.sub,
    }
    X = np.array(X, copy=False)
    return ops.get(op, operator.mul)(X[:, [0]], X[:, [1]] + eps)


def interaction_name(function_transformer, feature_names_in, name):
    return [name]  # feature names out


preprocessing = make_pipeline(
    make_column_transformer(
        (
            FunctionTransformer(
                features_interaction,
                feature_names_out=partial(
                    interaction_name, name="allelectrons_atomicweight_avg_mul"
                ),
                kw_args=dict(op="mul"),
            ),
            ["allelectrons_Average", "atomicweight_Average"],
        ),
        (
            FunctionTransformer(
                clip_numeric_feature,
                feature_names_out="one-to-one",
                kw_args=dict(lower_bound=0, upper_bound=1894),
            ),
            ["allelectrons_Total"],
        ),
        (
            FunctionTransformer(
                clip_numeric_feature,
                feature_names_out="one-to-one",
                kw_args=dict(lower_bound=0, upper_bound=236),
            ),
            ["density_Total"],
        ),
        remainder="passthrough",
        verbose_feature_names_out=True,
    ),
    PowerTransformer(method="yeo-johnson", standardize=True),
    # FeatureAgglomeration(n_clusters=3, linkage="complete"),
)


class FeatureFromModel(
    MetaEstimatorMixin,  # Accepts any regressor
    BaseEstimator,  # `set_params()` and `get_params()`
    TransformerMixin,  # `fit_transform()`
    ClassNamePrefixFeaturesOutMixin,  # `get_feature_names_out()`
):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        check_array(X)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        self._n_features_out = y.ndim
        if hasattr(self.estimator, "feature_names_in_"):
            self.feature_names_in_ = self.estimator.feature_names_in_
        return self

    def transform(self, X):
        check_is_fitted(self)
        check_array(X)
        y_pred = self.estimator_.predict(X)
        if y_pred.ndim == 1:
            return y_pred.reshape(-1, 1)
        return y_pred


def round_to_nearest(y, known_values=None, top_n=20):
    if known_values is None:
        known_values = y.value_counts().index.to_numpy()[:top_n]
    y_repeated = np.tile(y, reps=(len(known_values), 1)).transpose()
    lowest_diff_ids = np.abs(y_repeated - known_values).argmin(axis=1)
    return known_values[lowest_diff_ids]


X = train.drop("Hardness", axis=1)
y = train.Hardness
y_oof = np.zeros_like(y)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
knn = make_pipeline(
    FeatureFromModel(KNeighborsRegressor(n_neighbors=100)),
    MinMaxScaler(),
)
lgb = LGBMRegressor(
    objective="regression_l1",
    min_child_samples=100,
    reg_alpha=20,
    reg_lambda=20,
    verbose=-1,
)

for fold, (train_ids, valid_ids) in enumerate(kfold.split(X, y), start=1):
    X_train, y_train = X.iloc[train_ids], y.iloc[train_ids]
    X_valid, y_valid = X.iloc[valid_ids], y.iloc[valid_ids]

    X_train = preprocessing.fit_transform(X_train)
    X_valid = preprocessing.transform(X_valid)

    X_train = np.c_[X_train, knn.fit_transform(X_train, round_to_nearest(y_train, top_n=10))]
    X_valid = np.c_[X_valid, knn.transform(X_valid)]

    lgb.fit(X_train, round_to_nearest(y_train, top_n=30))
    y_pred = lgb.predict(X_valid).round(2)  # type: ignore
    y_oof[valid_ids] = y_pred
    medae = median_absolute_error(y_valid, y_pred)  # type: ignore

    print(CLR + f"Fold: {fold:2d}", CLR + "- MedAE:", RED + f"{medae:.3f}")


fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)

fig.add_histogram(
    x=np.abs(y - y_oof),
    xbins=go.histogram.XBins(size=0.2),
    showlegend=False,
    row=1,
    col=1,
)
fig.update_xaxes(title="Absolute Error", range=(-0.5, 6), row=1, col=1)
fig.update_yaxes(title="Count", row=1, col=1)
fig.add_scatter(
    x=y,
    y=y_oof,
    mode="markers",
    showlegend=True,
    name="Predictions",
    marker=dict(symbol="x", size=2, color=COLOR_SCHEME[0]),
    row=1,
    col=2,
)
fig.add_scatter(x=[1, 10], y=[1, 10], name="Perfectly Predicted", mode="lines", row=1, col=2)
fig.update_yaxes(title="Predicted 'Hardness'", row=1, col=2)
fig.update_xaxes(title="True 'Hardness'", row=1, col=2)
fig.update_layout(
    title="Training Dataset - Out-of-Fold Predictions via Regression Approach<br>"
    "<span style='font-size: 75%; font-weight: bold;'>"
    "Unfortunately nothing special</span>",
    height=460,
    width=840,
    bargap=0.2,
    legend=dict(y=1.02, x=1),
)
save_and_show_fig(fig, "out_of_fold_preds_reg")


y_train = train.Hardness
X_train = preprocessing.fit_transform(train.drop("Hardness", axis=1))
X_train = np.c_[X_train, knn.fit_transform(X_train, round_to_nearest(y_train, top_n=10))]

X_test = preprocessing.transform(test)
X_test = np.c_[X_test, knn.transform(X_test)]

lgb.fit(X_train, round_to_nearest(y_train, top_n=30))

submission = pd.DataFrame(
    {
        "id": test.index,
        "Hardness": lgb.predict(X_test).round(2),  # type:ignore
    }
).set_index("id")

submission.to_csv("submission_reg.csv")
get_pretty_frame(submission.head(), precision=2)


X = train.drop("Hardness", axis=1)
y = train.Hardness
y_oof = np.zeros_like(y)

cats = np.array([1.75, 2.55, 3.75, 4.75, 5.75, 6.55, 7.75, 8.75, 9.75])
encoder = LabelEncoder()
rskf_cats = encoder.fit_transform(round_to_nearest(y, cats))

lgb = LGBMClassifier(random_state=42, max_depth=3, verbose=-1)
xgb = XGBClassifier(random_state=42, max_depth=3)

rskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_ids, valid_ids) in enumerate(rskf.split(X, rskf_cats), start=1):
    X_train, y_train = X.iloc[train_ids], y.iloc[train_ids]
    X_valid, y_valid = X.iloc[valid_ids], y.iloc[valid_ids]

    y_train = encoder.fit_transform(round_to_nearest(y_train, cats))
    X_train = preprocessing.fit_transform(X_train)
    X_valid = preprocessing.transform(X_valid)

    lgb.fit(X_train, y_train)  # type: ignore
    xgb.fit(X_train, y_train)

    lgb_proba = lgb.predict_proba(X_valid)
    xgb_proba = xgb.predict_proba(X_valid)

    y_cat = np.argmax(lgb_proba + xgb_proba, axis=1)
    y_pred = encoder.inverse_transform(y_cat)
    y_oof[valid_ids] = y_pred

    medae = median_absolute_error(y_valid, y_pred)
    print(CLR + f"Fold: {fold:2d}", CLR + "- MedAE:", RED + f"{medae:.3f}")


fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)

fig.add_histogram(
    x=np.abs(y - y_oof),
    xbins=go.histogram.XBins(size=0.2),
    showlegend=False,
    row=1,
    col=1,
)
fig.update_xaxes(title="Absolute Error", range=(-0.5, 6), row=1, col=1)
fig.update_yaxes(title="Count", row=1, col=1)
fig.add_scatter(
    x=y,
    y=y_oof,
    mode="markers",
    showlegend=True,
    name="Predictions",
    marker=dict(symbol="x", size=3, color=COLOR_SCHEME[0]),
    row=1,
    col=2,
)
fig.add_scatter(x=[1, 10], y=[1, 10], name="Perfectly Predicted", mode="lines", row=1, col=2)
fig.update_yaxes(title="Predicted 'Hardness'", row=1, col=2)
fig.update_xaxes(title="True 'Hardness'", row=1, col=2)
fig.update_layout(
    title="Training Dataset - Out-of-Fold Predictions via Multiclass Classification Approach",
    height=460,
    width=840,
    bargap=0.2,
    legend=dict(y=1.02, x=1),
)
save_and_show_fig(fig, "out_of_fold_preds_classif")


cats = np.array([1.75, 2.55, 3.75, 4.75, 5.75, 6.55, 7.75, 8.75, 9.75])
encoder = LabelEncoder()

X_train = preprocessing.fit_transform(train.drop("Hardness", axis=1))
X_test = preprocessing.transform(test)
y_train = encoder.fit_transform(round_to_nearest(train.Hardness, cats))

lgb = LGBMClassifier(random_state=42, max_depth=3, verbose=-1)
xgb = XGBClassifier(random_state=42, max_depth=3)

lgb.fit(X_train, y_train)  # type: ignore
xgb.fit(X_train, y_train)

lgb_proba = lgb.predict_proba(X_test)
xgb_proba = xgb.predict_proba(X_test)

y_cat = np.argmax(lgb_proba + xgb_proba, axis=1)
y_pred = encoder.inverse_transform(y_cat)

submission = pd.DataFrame(
    {
        "id": test.index,
        "Hardness": y_pred,
    }
).set_index("id")

submission.to_csv("submission_classif.csv")
get_pretty_frame(submission.head(), precision=2)

