import os
import time
import random
import shutil
import warnings
import itertools
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
sns.set_style("darkgrid")

%matplotlib inline

from IPython.display import display, HTML

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)



df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


BLUE_BOLD = "\033[1;34m"
RESET = "\033[0m"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(f"{BLUE_BOLD} Shape of the DataFrame:{RESET} {df.shape}")
print(f"{BLUE_BOLD} Rows:{RESET} {df.shape[0]}  {BLUE_BOLD}| Columns:{RESET} {df.shape[1]}")


print(f"\n{BLUE_BOLD}âš ï¸� Number of Duplicate Rows:{RESET} {df.duplicated().sum()}")

missing_count = df.isnull().sum()
missing_percent = (df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_percent})

print(f"\n{BLUE_BOLD} Missing Values (Count & %) in Each Column:{RESET}")
print(missing_df)

print(f"\n{BLUE_BOLD} Unique Values in Each Column:{RESET}")
print(df.nunique().sort_values())

print(f"\n{BLUE_BOLD}â„¹ï¸� DataFrame Info:{RESET}")
df.info()

pd.reset_option('display.max_columns')
pd.reset_option('display.width')



df.head()


import numpy as np
import pandas as pd
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap, to_hex

gradient_map = LinearSegmentedColormap.from_list(
    "skyblue_pink",
    ["#87CEEB", "#FF69B4"]
)

def make_gradient_colors(n):
    return [to_hex(gradient_map(i / max(n-1,1))) for i in range(n)]

def ip_cat_univariat(df):

    feature_print_count = {}

    def is_categorical(col):
        return (
            (df[col].dtype == 'object' and df[col].nunique() <= 15) or
            df[col].dtype.name == 'category' or
            (np.issubdtype(df[col].dtype, np.number) and df[col].nunique() <= 15)
        )

    def show_header(feature):
        feature_print_count[feature] = feature_print_count.get(feature, 0) + 1
        font_size = 24 + 4 * feature_print_count[feature]

        gradient_css = """
            background: linear-gradient(to right, #87CEEB, #FF69B4);
            -webkit-background-clip: text;
            color: transparent;
        """

        display(HTML(f"<h2 style='text-align:center; font-size:{font_size}px; {gradient_css}'><b>{feature}</b></h2>"))

    def plot_categorical(series, feature):
        vc = series.value_counts(dropna=False).sort_values(ascending=True)
        pct = (vc / vc.sum()) * 100

        colors = make_gradient_colors(len(vc))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("", ""),
            specs=[[{"type": "bar"}, {"type": "domain"}]]
        )

        fig.add_trace(
            go.Bar(
                x=vc.values,
                y=vc.index.astype(str),
                marker_color=colors,
                text=vc.values,
                textposition="outside",
                orientation="h"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Pie(
                labels=vc.index.astype(str),
                values=pct,
                hole=0.55,
                marker_colors=colors,
                textinfo="percent",
            ),
            row=1, col=2
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            height=200,
            width=600,
            margin=dict(t=10, l=20, r=20, b=20),
            font=dict(color="black")
        )

        fig.show()

    for feature in df.columns:
        if feature.lower() == "id":
            continue
        if is_categorical(feature):
            show_header(feature)
            plot_categorical(df[feature], feature)



ip_cat_univariat(df[['alcohol_consumption_per_week']])


ip_cat_univariat(df[['diagnosed_diabetes']])


ip_cat_univariat(df[['cardiovascular_history']])


ip_cat_univariat(df[['ethnicity']])


ip_cat_univariat(df[['gender']])


ip_cat_univariat(df[['smoking_status']])


ip_cat_univariat(df[['education_level']])


ip_cat_univariat(df[['family_history_diabetes']])


ip_cat_univariat(df[['employment_status']])


ip_cat_univariat(df[['income_level']])


from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sp_num_univariate(df):
    df = df.drop(columns=['id'])
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    pink_color = "#FF69B4"
    skyblue_shadow = "#6CB6E844"
    
    cols_per_row = 2
    total_cols = len(numeric_columns)
    total_rows = (total_cols + cols_per_row - 1) // cols_per_row

    fig = plt.figure(figsize=[8 * cols_per_row, 3.5 * total_rows])
    fig.subplots_adjust(hspace=0.8, wspace=0.4)

    for i, col in enumerate(numeric_columns):
        ax = fig.add_subplot(total_rows, cols_per_row, i + 1)
        ax.set_facecolor(skyblue_shadow)

        sns.boxplot(
            data=df,
            x=col,
            color=pink_color,
            linewidth=1.5,
            fliersize=2,
            ax=ax
        )

        ax.set_title(
            col,
            fontsize=18,
            fontweight='bold',
            color="black",
            pad=10
        )

        ax.set_xlabel('')
        ax.grid(False)
        sns.despine(ax=ax, left=True, bottom=True)

    plt.show()



sp_num_univariate(df)


from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def plot_num_cat(df, numeric_col, category_col, order=None, figsize=(12,5)):
    sns.set_style("white")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

    gradient_map = LinearSegmentedColormap.from_list(
        "skyblue_pink",
        ["#87CEEB", "#FF69B4"]
    )

    df_plot = df.copy()
    df_plot[category_col] = df_plot[category_col].astype(str)

    
    if category_col == "diagnosed_diabetes":
        df_plot[category_col] = df_plot[category_col].map({
            "0.0": "Non-Diabetic",
            "1.0": "Diabetic",
            "0": "Non-Diabetic",
            "1": "Diabetic"
        })
    # --------------------------------

    if order is None:
        order = df_plot.groupby(category_col)[numeric_col].mean().sort_values(ascending=False).index.tolist()

    unique_vals = len(order)
    gradient_colors = [gradient_map(i / (unique_vals - 1)) for i in range(unique_vals)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    sns.barplot(
        x=numeric_col,
        y=category_col,
        data=df_plot,
        palette=gradient_colors,
        order=order,
        estimator=np.mean,
        ci=None,
        ax=ax1
    )

    for p in ax1.patches:
        ax1.annotate(
            f'{p.get_width():.2f}',
            (p.get_x() + p.get_width()/2., p.get_y() + p.get_height()),
            ha='center', va='center',
            xytext=(0, 40),
            textcoords='offset points',
            fontsize=10, color='black'
        )

    ax1.set_title(f"Mean {numeric_col} by {category_col}")
    ax1.set_xlabel(numeric_col.upper())
    ax1.set_ylabel("")
    sns.despine(left=True, bottom=True, ax=ax1)

    ax2 = axes[1]
    sns.violinplot(
        x=numeric_col,
        y=category_col,
        data=df_plot,
        palette=gradient_colors,
        order=order,
        cut=0,
        scale="width",
        linewidth=1,
        ax=ax2
    )

    ax2.set_title(f"Distribution of {numeric_col} by {category_col}")
    ax2.set_xlabel(numeric_col.upper())
    ax2.set_ylabel("")
    plt.yticks([])
    sns.despine(left=True, bottom=True, ax=ax2)

    plt.tight_layout()
    plt.show()



plot_num_cat(df, numeric_col="age", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="physical_activity_minutes_per_week", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="diet_score", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="sleep_hours_per_day", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="screen_time_hours_per_day", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="waist_to_hip_ratio", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="cholesterol_total", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="ldl_cholesterol", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="triglycerides", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="diastolic_bp", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="heart_rate", category_col="diagnosed_diabetes")



plot_num_cat(df, numeric_col="hdl_cholesterol", category_col="diagnosed_diabetes")



from matplotlib.colors import LinearSegmentedColormap
import itertools, colorsys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML, display

gradient_map = LinearSegmentedColormap.from_list(
    "skyblue_pink",
    ["#87CEEB", "#FF69B4"]
)

def get_recycled_colors(categories):
    cats = list(categories)
    n = len(cats)
    if n == 1:
        return {cats[0]: gradient_map(0.5)}
    colors = [gradient_map(i / (n - 1)) for i in range(n)]
    return {cats[i]: colors[i] for i in range(n)}

def _create_subplots(n_plots, figsize=(20, 25)):
    n_rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.8, wspace=0.8)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_plots == 1:
        axes = np.array([[axes]])
    return fig, axes.flatten()

def _plot_categorical_vs_categorical(df, x_col, color_col, ax):
    unique_categories = df[color_col].dropna().unique()
    color_map = get_recycled_colors(unique_categories)
    sns.countplot(
        data=df, x=x_col, hue=color_col,
        palette=list(color_map.values()), ax=ax
    )
    ax.set_xlabel(x_col, fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax.set_title(f'{x_col} vs {color_col}', fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(title=color_col, frameon=False)
    ax.set_facecolor('white')

def _plot_categorical_vs_numeric(df, cat_col, target_col, ax):
    unique_categories = df[cat_col].dropna().unique()
    color_map = get_recycled_colors(unique_categories)
    palette = {cat: color_map[cat] for cat in unique_categories}
    sns.boxplot(
        data=df, x=cat_col, y=target_col,
        palette=palette, ax=ax
    )
    ax.set_xlabel(cat_col, fontsize=16, fontweight='bold')
    ax.set_ylabel(target_col, fontsize=16, fontweight='bold')
    ax.set_title(f'{target_col} by {cat_col}', fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_facecolor('white')

def _plot_numeric_vs_numeric(df, x_col, y_col, category_col=None, ax=None):
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    strength = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.3 else "weak"
    direction = "positive" if corr > 0 else "negative" if corr < 0 else "no"
    title = f"{x_col} vs {y_col}"
    if category_col:
        title += f" by {category_col}"
    if category_col:
        unique_cats = df[category_col].dropna().unique()
        color_map = get_recycled_colors(unique_cats)
        sns.scatterplot(
            data=df, x=x_col, y=y_col,
            hue=category_col, palette=list(color_map.values()),
            edgecolor=None, ax=ax
        )
    else:
        sns.scatterplot(
            data=df, x=x_col, y=y_col,
            color=gradient_map(0.6),
            edgecolor=None, ax=ax
        )
    ax.set_xlabel(x_col, fontsize=16, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=16, fontweight='bold')
    ax.set_title(f"{title}\nCorr: {corr:.2f} ({strength} {direction})", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_facecolor('white')
    if category_col:
        ax.legend(title=category_col, frameon=False)

def feature_target_analysis(df, target_col, color_col=None, plot_type=3):
    heading_color = "#FF6347"
    if df[target_col].nunique() < 8:
        df[target_col] = df[target_col].astype("object")
    cat_vs_num = []
    cat_vs_cat = []
    num_vs_num = []
    id_cols = [col for col in df.columns if "id" in col.lower()]
    excluded_cols = set(id_cols + [target_col])
    bool_cols = [col for col in df.columns if df[col].dtype == bool]
    df[bool_cols] = df[bool_cols].astype(object)
    drop_cols = [col for col in df.columns if df[col].dtype == object and df[col].nunique() > 15]
    df = df.drop(columns=drop_cols)
    excluded_cols.update(drop_cols)
    num_target_is_cat = pd.api.types.is_categorical_dtype(df[target_col]) or df[target_col].dtype == object
    for feature in df.columns:
        if feature in excluded_cols:
            continue
        feature_is_cat = pd.api.types.is_categorical_dtype(df[feature]) or df[feature].dtype == object
        feature_is_num = pd.api.types.is_numeric_dtype(df[feature])
        if feature_is_num and df[feature].nunique() < 10:
            feature_is_cat = True
            feature_is_num = False
        if feature_is_num and pd.api.types.is_numeric_dtype(df[target_col]):
            num_vs_num.append(feature)
        elif feature_is_cat and not num_target_is_cat:
            cat_vs_num.append(feature)
        elif feature_is_cat and num_target_is_cat:
            cat_vs_cat.append(feature)
        elif feature_is_num and num_target_is_cat:
            cat_vs_num.append(feature)
    if plot_type == 3 and num_vs_num:
        display(HTML(f"<h1 style='color:{heading_color}; text-align:center'>Numeric Feature vs {target_col}</h1>"))
        fig, axes = _create_subplots(len(num_vs_num))
        for i, feature in enumerate(num_vs_num):
            _plot_numeric_vs_numeric(df, feature, target_col, category_col=color_col, ax=axes[i])
        plt.tight_layout()
        plt.show()
    if plot_type == 2 and cat_vs_num:
        display(HTML(f"<h1 style='color:{heading_color}; text-align:center'>Categorical Feature vs {target_col}</h1>"))
        fig, axes = _create_subplots(len(cat_vs_num))
        for i, feature in enumerate(cat_vs_num):
            if pd.api.types.is_numeric_dtype(df[target_col]):
                _plot_categorical_vs_numeric(df, feature, target_col, ax=axes[i])
            else:
                _plot_categorical_vs_numeric(df, target_col, feature, ax=axes[i])
        plt.tight_layout()
        plt.show()
    if plot_type == 1 and cat_vs_cat:
        
        gradient_css = """
            background: linear-gradient(to right, #87CEEB, #FF69B4);
            -webkit-background-clip: text;
            color: transparent;
        """
        
        display(HTML(
            f"<h1 style='text-align:center; font-size:30px; font-weight:bold; {gradient_css}'>"
            f"Categorical Feature vs {target_col}"
            f"</h1>"
        ))

        fig, axes = _create_subplots(len(cat_vs_cat))
        for i, feature in enumerate(cat_vs_cat):
            _plot_categorical_vs_categorical(df, feature, target_col, ax=axes[i])
        plt.tight_layout()
        plt.show()




feature_target_analysis(df, target_col='diagnosed_diabetes',plot_type=1)


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

gradient_map = LinearSegmentedColormap.from_list(
    "skyblue_pink",
    ["#87CEEB", "#FF69B4"]
)

def plot_heatmap(df, figsize=(15, 5), annot=False, title="Heatmap"):
    corr = df.corr(numeric_only=True)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=gradient_map,
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar=True
    )
    plt.title(title, fontsize=16)
    plt.show()

plot_heatmap(df, annot=False, title="Correlation Heatmap")



df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df_train.head()


cat_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    "alcohol_consumption_per_week"
]



train = df_train.drop(['id'], axis=1).drop_duplicates()
test = df_test.drop(['id'], axis=1)


for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")



X = train.drop(['diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']


X.shape



from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# import optuna
# from optuna.samplers import TPESampler
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# def objective(trial, X_train, y_train, X_valid, y_valid):

#     params = {
#         "objective": "binary",
#         "metric": "auc",
#         "boosting_type": "gbdt",
#         "device": "gpu",
#         "gpu_platform_id": 0,
#         "gpu_device_id": 0,

#         "verbosity": -1,
#         "random_state": 42,

#         # Hyperparameters to tune
#         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
#         "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
#         "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 0.5),
#         "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 0.5),
#         "max_depth": trial.suggest_int("max_depth", -1, 20),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
#     }

#     model = LGBMClassifier(**params)

#     # Fit model
#     model.fit(
#         X_train,
#         y_train,
#         eval_set=[(X_valid, y_valid)],
#         eval_metric="auc",
#         categorical_feature=cat_cols,
#     )

#     # Predict probabilities
#     y_pred = model.predict_proba(X_valid)[:, 1]

#     # ROC-AUC
#     auc = roc_auc_score(y_valid, y_pred)
#     return auc


# # -----------------------------
# # TRAIN-TEST SPLIT
# # -----------------------------
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )


# # -----------------------------
# # OPTUNA STUDY
# # -----------------------------
# sampler = TPESampler(seed=42)
# study = optuna.create_study(direction="maximize", sampler=sampler)

# study.optimize(
#     lambda trial: objective(trial, X_train, y_train, X_valid, y_valid),
#     n_trials=50
# )

# # -----------------------------
# # BEST PARAMETERS
# # -----------------------------
# print("="*60)
# print("Best parameters:")
# print(study.best_params)


best_params = {
    'learning_rate': 0.075533,
    'n_estimators': 1912,
    'lambda_l1': 0.365997,
    'lambda_l2': 0.299329,
    'max_depth': 2,
    'colsample_bytree': 0.409196,
    'subsample': 0.623233,
    'min_child_samples': 88
}



from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

lgbmmodel = LGBMClassifier(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",

    # GPU
    device="gpu",
    gpu_platform_id=0,
    gpu_device_id=0,

    random_state=42,
    verbosity=-1,

    **best_params
)

lgbmmodel.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    categorical_feature=cat_cols
)



test_pred = lgbmmodel.predict_proba(test)[:, 1]



submission = pd.DataFrame({
    "id": df_test["id"],
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission.csv", index=False)



s = pd.read_csv("/kaggle/working/submission.csv")


s




