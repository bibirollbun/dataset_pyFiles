import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from IPython.display import Markdown
import time

import glob
import numpy as np
import polars as pl
import pandas as pd

import seaborn as sns
import collections

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import shap


PATH_DATA_ROOT = "/kaggle/input/home-credit-credit-risk-model-stability/"

batches = ["train", "test"]

PATH_DATA = {
    batch: f"{PATH_DATA_ROOT}parquet_files/{batch}/"
    for batch in batches
}


def set_pl_dtypes(df):
    for col in df.columns:
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64))
        if col[-1] in ("M"):
            df = df.with_columns(pl.col(col).cast(pl.Categorical("lexical")))
        if col[-1] in ("D"):
            df = df.with_columns(pl.col(col).cast(pl.Date))
        if col[-1] in ("L", "T") and not df[col].is_numeric:
            df = df.with_columns(pl.col(col).cast(pl.Categorical("lexical")))
    return df


dt_data = {
    batch: {
        "base": (pl.read_parquet(f"{PATH_DATA[batch]}{batch}_base.parquet")\
                 .with_columns(pl.col("date_decision").cast(pl.Date))),
        "static": pl.concat([pl.read_parquet(PATH_DATA_STATIC)\
                             .pipe(set_pl_dtypes) for PATH_DATA_STATIC in
                             glob.glob(f"{PATH_DATA[batch]}{batch}_static_0*.parquet")],
                            how="vertical_relaxed"),
        "static_cb": (pl.read_parquet(f"{PATH_DATA[batch]}{batch}_static_cb_0.parquet")\
                      .pipe(set_pl_dtypes)),
        "person_1": (pl.read_parquet(f"{PATH_DATA[batch]}{batch}_person_1.parquet")\
                     .pipe(set_pl_dtypes)),
        "credit_bureau_b_2": (pl.read_parquet(f"{PATH_DATA[batch]}{batch}_credit_bureau_b_2.parquet")\
                              .pipe(set_pl_dtypes))

    }   
    for batch in batches
}


for batch in batches:

    df_person_1_1 = dt_data[batch]["person_1"].group_by("case_id").agg(
        pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_max_A").cast(pl.Float64),
        (pl.col("incometype_1044T") == "SELFEMPLOYED").any().alias("anyselfemployed_T").cast(pl.Boolean)
    )
    df_person_1_2 = dt_data[batch]["person_1"].drop(
        ["mainoccupationinc_384A", "incometype_1044T"])\
        .filter(pl.col("num_group1") == 0).drop("num_group1")
    df_person_1_2 = df_person_1_2.rename(
        {col: col.rsplit("_",1)[0] + "_applicant_" + col.rsplit("_",1)[1] for
         col in df_person_1_2.drop("case_id").columns}
    )
    dt_data[batch]["person_1"] = (df_person_1_1\
        .join(other=df_person_1_2,
              how="left",
              on="case_id")
    )
    
    dt_data[batch]["credit_bureau_b_2"] = dt_data[batch]["credit_bureau_b_2"]\
        .group_by("case_id").agg(
            pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_max_A").cast(pl.Float64),
            (pl.col("pmts_dpdvalue_108P") > 31).any().alias("pmts_dpdvalue_anyover31_P").cast(pl.Boolean)
    )


dt_data = {
    batch: dt_data[batch]["base"]\
    .join(dt_data[batch]["static"],
          how="left",
          on="case_id")\
    .join(dt_data[batch]["static_cb"],
          how="left",
          on="case_id")\
    .join(dt_data[batch]["person_1"],
          how="left",
          on="case_id")\
    .join(dt_data[batch]["credit_bureau_b_2"],
          how="left",
          on="case_id")
    for batch in batches
}


dt_data["train"]


dt_data["test"].head()


dt_N_y_train = dict(collections.Counter(dt_data["train"]["target"]))

dt_N_y_train["ratio"] = dt_N_y_train[0] / dt_N_y_train[1]

plt.figure(figsize=(6.4, 4.8))
ax = plt.axes()
plt.title("Number of counts per class", pad=20)

sns.countplot(
    ax=ax,
    x=dt_data["train"]["target"].to_numpy(),
    color="blue",
    alpha=0.5,
    edgecolor="black",
    linewidth=1.0,
    width=0.075,
    hatch="////",
    zorder=2
)

ax.set_xlabel(r"Class, $y$", fontdict={"fontsize": 10})
ax.set_ylabel(r"Counts", fontdict={"fontsize": 10})

ax.minorticks_on()

ax.grid(
    visible=True,
    which="major",
    color="lightgray",
    linestyle="solid",
    linewidth=0.5
)
ax.grid(
    visible=True,
    which="minor",
    color="lightgray",
    linestyle="dotted",
    linewidth=0.5
)

plt.show() 

print()
display(pd.DataFrame(data={"$n^-/n^+$": dt_N_y_train["ratio"]},
                     index=[0])\
        .style\
        .format({"$n^-/n^+$": "{:.2f}"})\
        .set_caption("Ratio between numbers of settled ($y=0$) and defaulted ($y=1$) credit contract cases")\
        .set_table_styles([
                {"selector": "th.col_heading,td",
                 "props": [("width", "300px")]
                 },
                {"selector": "caption",
                 "props": [("font-size", "16px"),
                           ("font-weight", "bold"),
                           ("font-style", "italic")]
                 }
            ]))


# ---> Add a sample weight column to the training dataframe
dt_data["train"] = dt_data["train"].with_columns(
    pl.when(pl.col("target") == 1).then(dt_N_y_train["ratio"]).otherwise(1)\
    .alias("sample_weight").cast(pl.Float64)
)


dt_emp = {
    "emptiness": {col: dt_data["train"][col].null_count() / len(dt_data["train"][col]) for
                  col in dt_data["train"].columns},
    "almost_empty": {col: dt_data["train"][col].null_count() / len(dt_data["train"][col]) > 0.995 for
                     col in dt_data["train"].columns}
}


plt.figure(figsize=(6.4, 4.8))
ax = plt.axes()
plt.title("Number of counts per emptiness range in the training dataset", pad=20)

sns.histplot(
    ax=ax,
    data=dt_emp["emptiness"].values(),
    stat="count",
    bins=20,
    binrange=(0, 1),
    palette=["blue"],
    alpha=0.5,
    edgecolor="black",
    linewidth=1.0,
    hatch="////",
    zorder=2,
    legend=False
)

plt.axvline(
    x=0.995,
    color="black",
    linewidth=2,
    alpha=1,
    linestyle="dashed",
    label="$\mathrm{emptiness}=0.995$"
)

ax.set_xlabel(r"Emptiness", fontdict={"fontsize": 10})
ax.set_ylabel(r"Counts", fontdict={"fontsize": 10})

ax.set_xlim(
    left=0,
    right=1
)
    
ax.minorticks_on()

ax.grid(
    visible=True,
    which="major",
    color="lightgray",
    linestyle="solid",
    linewidth=0.5
)
ax.grid(
    visible=True,
    which="minor",
    color="lightgray",
    linestyle="dotted",
    linewidth=0.5
)

ax.legend(fontsize=8)

plt.show() 

print()
display(pd.DataFrame(data={"Counts ($\mathrm{emptiness}>0.995$)":
                           sum(dt_emp["almost_empty"].values())},
                     index=[0])\
        .style\
        .format({"Counts ($\mathrm{emptiness}>0.995$)": "{:d}"})\
        .set_caption("Number of almost empty columns in the training dataset")\
        .set_table_styles([
                {"selector": "th.col_heading,td",
                 "props": [("width", "300px")]
                 },
                {"selector": "caption",
                 "props": [("font-size", "16px"),
                           ("font-weight", "bold"),
                           ("font-style", "italic")]
                 }
            ]))


cols_drop = [key for key in dt_emp["almost_empty"].keys() if dt_emp["almost_empty"][key] == True]
dt_data["train"] = dt_data["train"].drop(cols_drop)
dt_data["test"] = dt_data["test"].drop(cols_drop)


cols_cat = dt_data["train"].select(pl.col(pl.Categorical)).columns

dt_N_cat = {
    "N_cat": {col: len(dt_data["train"][col].cat.get_categories()) for
              col in cols_cat},
}

dt_N_cat.update({
    "N_cat==1": {col: dt_N_cat["N_cat"][col] == 1 for col in cols_cat},
    "N_cat>1 and N_cat<=1": {col: dt_N_cat["N_cat"][col] > 1 and
                             dt_N_cat["N_cat"][col] <= 1000 for col in cols_cat},
    "N_cat>1000": {col: dt_N_cat["N_cat"][col] > 1000 for col in cols_cat}
})


x = list(range(len(["N_cat==1", "N_cat>1 and N_cat<=1", "N_cat>1000"]) + 1))

i_x = list(range(len(x)))
x_bar_points = sum(
    [[x[i]] * 
     (2 if i in (0, i_x[-1]) else 4)
     for i in i_x],
    []
)

y_bar_points = sum(
    [[0, sum(dt_N_cat[key].values()), sum(dt_N_cat[key].values()), 0]
     for key in ["N_cat==1", "N_cat>1 and N_cat<=1", "N_cat>1000"]],
    []
)

x_ticks = [(x[i] + x[i + 1]) / 2 for i in i_x[:-1]]

plt.figure(figsize=(6.4, 4.8))
ax = plt.axes()
plt.title("Number of counts per number of categories range in the training dataset", pad=20)

ax.plot(
    x_bar_points,
    y_bar_points,
    color="black",
    linewidth=1.0,
    zorder=2
)

ax.fill_between(
    x=x_bar_points,
    y1=np.zeros(len(x_bar_points)),
    y2=y_bar_points,
    facecolor="blue",
    edgecolor="black",
    hatch="////",
    linewidth=1.0,
    alpha=0.5,
    zorder=2
)
                                      
ax.set_xlabel(r"Number of categories, $N_{\mathrm{cat}}$", fontdict={"fontsize": 10})
ax.set_ylabel(r"Counts", fontdict={"fontsize": 10})

ax.set_xlim(
    left=0,
    right=3
)
ax.set_ylim(
    bottom=0
)

ax.set_xticks(x_ticks, labels=["$1$", "$[1,\,1000]$", f"$>1000$"])

ax.grid(
    visible=True,
    which="major",
    color="lightgray",
    linestyle="solid",
    linewidth=0.5
)
ax.grid(
    visible=True,
    which="minor",
    color="lightgray",
    linestyle="dotted",
    linewidth=0.5
)

plt.show() 

print()
display(pd.DataFrame(data={"Counts ($N_{\mathrm{cat}}=1$)": sum(dt_N_cat["N_cat==1"].values()),
                           "Counts ($N_{\mathrm{cat}}>1000$)": sum(dt_N_cat["N_cat>1000"].values())},
                     index=[0])\
        .style\
        .format({"N": "{:d}"})\
        .set_caption("Numbers of columns of single and of too many categories in the training dataset")\
        .set_table_styles([
                {"selector": "th.col_heading,td",
                 "props": [("width", "300px")]
                 },
                {"selector": "caption",
                 "props": [("font-size", "16px"),
                           ("font-weight", "bold"),
                           ("font-style", "italic")]
                 }
            ]))


cols_drop = [col for col in cols_cat if
             True in (dt_N_cat["N_cat==1"][col], dt_N_cat["N_cat>1000"])]

dt_data["train"] = dt_data["train"].drop(cols_drop)
dt_data["test"] = dt_data["test"].drop(cols_drop)


def dist_plot(df, col, col_fancy, binning=True, N_bins=10, xticklabels_rotation=0):

    if binning == True:
        x_bin_limit = np.linspace(df[col].min(), df[col].max(), N_bins + 1)

        i_x_bin_limit = range(len(x_bin_limit))

        df_dist = pl.DataFrame(
            schema=[
                "Bin's $x_{\mathrm{min}}$",
                "Bin's $x_{\mathrm{av}}$",
                "Bin's $x_{\mathrm{max}}$",
                "Counts ($y=0$)",
                "Counts ($y=1$)",
                "$P(y=1)\,[\%]$"
            ]
        )

        df_dist_plot = pl.DataFrame(
            schema=[
                "x",
                "y"
            ]
        )

        for i in i_x_bin_limit[:-1]:
            df_dist_i = (
                df.select(col, "target")\
                    .groupby(
                        ((pl.col(col) >= x_bin_limit[i]) &
                         ((pl.col(col) < x_bin_limit[i + 1]) if 
                          i != i_x_bin_limit[-2] else 
                          (pl.col(col) <= x_bin_limit[i + 1])))
                    )\
                    .agg(
                        (pl.col("target") == 0).sum().alias("Counts ($y=0$)"),
                        (pl.col("target") == 1).sum().alias("Counts ($y=1$)")
                    )\
                    .with_columns(
                        (pl.col("Counts ($y=1$)") /
                         (pl.col("Counts ($y=0$)") + pl.col("Counts ($y=1$)")) *
                         100).alias("$P(y=1)\,[\%]$")
                    )\
                    .filter(pl.col(col) == True)\
                    .drop(col)
                    .with_columns(
                        pl.lit(x_bin_limit[i]).alias("Bin's $x_{\mathrm{min}}$"),
                        pl.lit((x_bin_limit[i] + x_bin_limit[i + 1]) / 2).alias("Bin's $x_{\mathrm{av}}$"),
                        pl.lit(x_bin_limit[i + 1]).alias("Bin's $x_{\mathrm{max}}$"),
                    )
                    .select(
                        "Bin's $x_{\mathrm{min}}$",
                        "Bin's $x_{\mathrm{av}}$",
                        "Bin's $x_{\mathrm{max}}$",
                        "Counts ($y=0$)",
                        "Counts ($y=1$)",
                        "$P(y=1)\,[\%]$"
                    )
            )

            df_dist = pl.concat([
                df_dist,
                df_dist_i
            ],
                how="vertical_relaxed"
            )

            df_dist_plot = pl.concat([
                df_dist_plot,
                df_dist_i.select(pl.col("Bin's $x_{\mathrm{min}}$").alias("x")).with_columns(pl.lit(0).alias("y")),
                df_dist_i.select(pl.col("Bin's $x_{\mathrm{min}}$").alias("x"), pl.col("$P(y=1)\,[\%]$").alias("y")),
                df_dist_i.select(pl.col("Bin's $x_{\mathrm{max}}$").alias("x"), pl.col("$P(y=1)\,[\%]$").alias("y")),
                df_dist_i.select(pl.col("Bin's $x_{\mathrm{max}}$").alias("x")).with_columns(pl.lit(0).alias("y"))
            ],
                how="vertical_relaxed"
            )
    else:
        df_dist = (df.select(col, "target")\
            .groupby(col)\
            .agg((pl.col("target") == 0).sum().alias("Counts ($y=0$)"),
                 (pl.col("target") == 1).sum().alias("Counts ($y=1$)"))\
            .with_columns(
            (pl.col("Counts ($y=1$)") /
             (pl.col("Counts ($y=0$)") + pl.col("Counts ($y=1$)")) *
             100).alias("$P(y=1)\,[\%]$"))\
            .rename({col: col_fancy.capitalize()})
        )

    print()
    display(Markdown('---'))
    print()

    plt.figure(figsize=(6.4, 4.8))
    ax = plt.axes()
    plt.title(f"Relation between probability of credit default and {col_fancy}", pad=20)

    if binning == True:
        # Plot bar lines
        ax.plot(
            df_dist_plot["x"],
            df_dist_plot["y"],
            color="blue",
            linewidth=1.0,
            zorder=2
        )

        # Fill bars
        ax.fill_between(
            x=df_dist_plot["x"],
            y1=np.zeros(len(df_dist_plot["x"])),
            y2=df_dist_plot["y"],
            facecolor="blue",
            edgecolor="black",
            hatch="////",
            linewidth=1.0,
            alpha=0.5,
            zorder=2
        )

        for row in df_dist.rows(named=True):
            ax.annotate(
                text="${:.2f}\,\%$".format(row["$P(y=1)\,[\%]$"]),
                xy=(row["Bin's $x_{\mathrm{av}}$"], 
                    row["$P(y=1)\,[\%]$"]),
                ha="left",
                va="bottom",
                size=9,
                xytext=(-7.5, 2),
                textcoords="offset points",
                rotation=70
            )
    else:
        sns.barplot(
            ax=ax,
            data=df_dist.to_pandas(),
            x=col_fancy.capitalize(),
            y="$P(y=1)\,[\%]$",
            color="blue",
            edgecolor="black",
            hatch="////",
            linewidth=1.0,
            alpha=0.5,
            width=0.200,
            zorder=2
        )

        for bar in ax.patches:
            ax.annotate(
                text="${:.2f}\,\%$".format(bar.get_height()),
                xy=(bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()),
                ha="left",
                va="bottom",
                size=9,
                xytext=(-5, 2),
                textcoords="offset points",
                rotation=70
            )
               
    ax.set_xlabel(col_fancy.capitalize(), fontdict={"fontsize": 10})
    ax.set_ylabel(r"$P(y=1)\,[\%]$", fontdict={"fontsize": 10})

    ax.tick_params(axis="x", rotation=xticklabels_rotation)

    if binning == True:
        ax.set_xlim(
            left=min(df_dist_plot["x"]),
            right=max(df_dist_plot["x"])
        )
    ax.set_ylim(bottom=0, top=1.125 * ax.get_ylim()[1])

    ax.minorticks_on()

    ax.grid(
        visible=True,
        which="major",
        color="lightgray",
        linestyle="solid",
        linewidth=0.5
    )
    ax.grid(
        visible=True,
        which="minor",
        color="lightgray",
        linestyle="dotted",
        linewidth=0.5
    )

    plt.show()

    print()
    display(df_dist.to_pandas()\
            .style\
            .format({
                "P_1": "{:.2f}",
                "Bin's $x_{\mathrm{min}}$": "{:.2f}",
                "Bin's $x_{\mathrm{av}}$": "{:.2f}",
                "Bin's $x_{\mathrm{max}}$": "{:.2f}"
            } if binning==True else
            {"P_1": "{:.2f}"})\
            .set_caption(f"{col_fancy.capitalize()} - counts and probabilities")\
            .set_table_styles([
                    {"selector": "th.col_heading,td",
                     "props": [("width", "150px")]
                     },
                    {"selector": "caption",
                     "props": [("font-size", "16px"),
                               ("font-weight", "bold"),
                               ("font-style", "italic")]
                     }
                ]))


dt_data = {
    batch: dt_data[batch].rename({"annuity_780A": "monthly_payment_A"})
    for batch in batches
}

# Plot
dist_plot(
    df=dt_data["train"],
    col="monthly_payment_A",
    col_fancy="contract's monthly payment",
    binning=True,
    N_bins=10,
    xticklabels_rotation=0
)

dt_data = {
    batch: dt_data[batch].with_columns(
        (pl.col("date_decision").sub(pl.col("birth_applicant_259D"))\
         .alias("age_applicant_A").dt.days() / 365.25).cast(pl.Float64)
    ).drop("birth_applicant_259D")
    for batch in batches
}

dist_plot(
    df=dt_data["train"],
    col="age_applicant_A",
    col_fancy="applicant's age at decision time",
    binning=True,
    N_bins=20,
    xticklabels_rotation=0
)

dt_data = {
    batch: dt_data[batch].with_columns(
        (pl.col("date_decision").sub(pl.col("empl_employedfrom_applicant_271D"))\
         .alias("employment_time_applicant_A").dt.days() / 365.25).cast(pl.Float64)
    ).drop("empl_employedfrom_applicant_271D")
    for batch in batches
}

dist_plot(
    df=dt_data["train"],
    col="employment_time_applicant_A",
    col_fancy="applicant's employment time (in years) at decision date",
    binning=True,
    N_bins=12,
    xticklabels_rotation=0
)

dt_data = {
    batch: dt_data[batch].with_columns(
        (25 - pl.col("cntpmts24_3658933L"))\
         .alias("N_missing_payments_24_L").cast(pl.Int64)
    ).drop("cntpmts24_3658933L")
    for batch in batches
}

dist_plot(
    df=dt_data["train"],
    col="N_missing_payments_24_L",
    col_fancy="number of missing payments in the last $24$ months (and current one)",
    binning=True,
    N_bins=10,
    xticklabels_rotation=0
)

dist_plot(
    df=dt_data["train"],
    col="sex_applicant_738L",
    col_fancy="applicant's gender",
    binning=False,
    xticklabels_rotation=0
)


dt_data = {
    batch: dt_data[batch].with_columns(
        pl.col("mobilephncnt_593L")\
        .alias("n_persons_same_phone_A").cast(pl.Int64)
    ).drop("mobilephncnt_593L")
    for batch in batches
}

dist_plot(
    df=dt_data["train"],
    col="n_persons_same_phone_A",
    col_fancy="number of persons using the same phone number",
    binning=True,
    N_bins=10,
    xticklabels_rotation=0
)

dt_data = {
    batch: dt_data[batch].with_columns(
        pl.col("date_decision").dt.weekday().cast(pl.Int64)\
        .alias("weekday_date_decision_A"),
        pl.col("date_decision").dt.month().cast(pl.Int64)\
        .alias("month_date_decision_A"),
        pl.col("date_decision").dt.month().cast(pl.Int64)\
        .alias("year_date_decision_A"),
    )
    for batch in batches
}

dist_plot(
    df=dt_data["train"],
    col="weekday_date_decision_A",
    col_fancy="weekday of contract's decision date",
    binning=False,
    xticklabels_rotation=45
)

dt_data = {
    batch: dt_data[batch].with_columns(
        pl.col("pmtnum_254L")\
        .alias("n_payments_A").cast(pl.Int64)
    ).drop("pmtnum_254L")
    for batch in batches
}

# Plot
dist_plot(
    df=dt_data["train"],
    col="n_payments_A",
    col_fancy="number of done payments",
    binning=True,
    N_bins=12,
    xticklabels_rotation=0
)


dt_data["train"] = dt_data["train"].sort("case_id")
dt_data["train"] = dt_data["train"].sample(fraction=1, shuffle=True, seed=42)


cols_x = []
for col in dt_data["train"].columns:
    if col[-1].isupper() and col[:-1].islower():
        cols_x.append(col)


def convert_cols_obj_to_cols_cat(*dfs):
    cols_object = list(set().union(*(df.select_dtypes(include=["object"]).columns for df in dfs)))
    for col in cols_object:
        for df in dfs:
            df[col] = df[col].astype("category")
            new_dtype = pd.CategoricalDtype(categories=df[col].cat.categories.to_list() +
                                            ["Unknown"],
                                            ordered=True)
            df[col] = df[col].astype(new_dtype)
    return dfs

def convert_cols_date_to_cols_ord(df):
    cols_date = df.select_dtypes(include=["datetime64"]).columns
    for col in cols_date:
        df[col] = df[col].apply(lambda x: x.toordinal() if not
                                pd.isnull else -1000)
    return df


def convert_cols_bool_to_cols_int(df):
    cols_bool = df.select_dtypes(include=["bool"]).columns
    for col in cols_bool:
        df[col] = df[col].astype("int64")
    return df

def make_cat_excl_unknown(df, df_ref):
    for col in df_ref.select_dtypes(include=["category"]).columns:
        cat_ref = df_ref[col].cat.categories.to_list()
        cat = df[col].cat.categories.to_list()
        cat_common = list(set(cat).intersection(cat_ref))
        cat_exc = list(set(cat).difference(cat_common))
        new_dtype = pd.CategoricalDtype(categories=cat_common,
                                        ordered=True)
        df[col] = df[col].replace(to_replace=cat_exc, value="Unknown")
        df[col] = df[col].astype(new_dtype)
    return df

def covert_cols_cat_to_cols_code(df):
    dt_map = {}
    
    for col in df.select_dtypes(include=["category"]).columns:
        
    
        dt_map.update({col:
            dict(enumerate(df[col].cat.categories))
        })
        
        df[col] = df[col].cat.codes

    return (df, dt_map)


dt_data["train"] = {
    "base": (dt_data["train"][["case_id", "date_decision", "WEEK_NUM", "target", "sample_weight"]]\
             .rename({"target": "y"}).to_pandas()),
    "x": dt_data["train"][cols_x].to_pandas(),
    "y": dt_data["train"]["target"].to_pandas(),
}
dt_data["test"] = {
    "base": dt_data["test"][["case_id", "date_decision", "WEEK_NUM"]].to_pandas(),
    "x": dt_data["test"][cols_x].to_pandas()
}
(dt_data["train"]["x"], dt_data["test"]["x"]) = convert_cols_obj_to_cols_cat(
    dt_data["train"]["x"], dt_data["test"]["x"]
)

dt_data["train"]["x"] = convert_cols_date_to_cols_ord(dt_data["train"]["x"])
dt_data["test"]["x"] = convert_cols_date_to_cols_ord(dt_data["test"]["x"])

dt_data["test"]["x"] = make_cat_excl_unknown(
    df=dt_data["test"]["x"], df_ref=dt_data["train"]["x"]
)

(dt_data["train"]["x"], dt_data["train"]["cat_map"]) = covert_cols_cat_to_cols_code(dt_data["train"]["x"])
(dt_data["test"]["x"], dt_data["test"]["cat_map"]) = covert_cols_cat_to_cols_code(dt_data["test"]["x"])

dt_data["train"]["x"] = convert_cols_bool_to_cols_int(dt_data["train"]["x"])
dt_data["test"]["x"] = convert_cols_bool_to_cols_int(dt_data["test"]["x"])


params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 1,
    "bagging_fraction": 1,
    "bagging_freq": 0,
    "n_estimators": 1200,
    "max_depth": 3,
    "n_jobs": -1,
    "random_state": 42,
    "device_type": "gpu",
    "gpu_use_dp": True,
    "verbose": -1,
}

lgb_estimator = lgb.LGBMClassifier(**params)


param_grid = {
    "n_estimators": [600, 800, 1000],
    "max_depth": [8, 9, 10]
}

gs = GridSearchCV(
    estimator=lgb_estimator, 
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    return_train_score=True,
    n_jobs=1,
    refit=True,
    verbose=0
)

t_i = time.perf_counter()
gs.fit(
    X=dt_data["train"]["x"],
    y=dt_data["train"]["y"],
    sample_weight=dt_data["train"]["base"]["sample_weight"]
)

t_f = time.perf_counter()

print()
print(f"Running time: {(t_f - t_i)/3600:.2f} h")


cv_results = pd.DataFrame(gs.cv_results_)

cv_results.columns = cv_results.columns.str.replace("score", "auc")

cv_results.to_csv("cv_results.csv", index=True)

print()
display(cv_results[cv_results.filter(like="param_", axis="columns").columns.values.tolist() + 
                   ["mean_train_auc", "std_train_auc", "mean_test_auc", "std_test_auc"]]\
        .style.set_caption("Mean AUCs and respective standard devations")\
        .set_table_styles(
            [
                {"selector": "th.col_heading,td",
                 "props": [("width", "100px")]
                 },
                {"selector": "caption",
                 "props": [("font-size", "16px"),
                           ("font-weight", "bold"),
                           ("font-style", "italic")]
                 }
            ]))

(cv_results_best_train_auc, cv_results_best_valid_auc) = (
    pd.DataFrame(
        data=(cv_results[cv_results.filter(like="param_", axis="columns").columns.values.tolist() + 
                         ["mean_train_auc", "std_train_auc", "mean_test_auc", "std_test_auc"]]\
              .loc[cv_results[f"mean_{batch}_auc"].idxmax()].to_dict()),
        index=[cv_results[f"mean_{batch}_auc"].idxmax()]
    ) for batch in ["train", "test"]
)

for (df, batch_fancy) in ((cv_results_best_train_auc, "training"),
                          (cv_results_best_valid_auc, "validation")):
    print()
    display(df\
        .style.set_caption(f"Mean AUCs and respective standard devations at best {batch_fancy} mean AUC")\
        .set_table_styles(
            [
                {"selector": "th.col_heading,td",
                 "props": [("width", "100px")]
                 },
                {"selector": "caption",
                 "props": [("font-size", "16px"),
                           ("font-weight", "bold"),
                           ("font-style", "italic")]
                 }
            ])
    )


n_estimators = cv_results["param_n_estimators"].unique()
max_depth = cv_results["param_max_depth"].unique()
mean_test_auc = (cv_results.groupby("param_n_estimators")\
                 .agg(list)["mean_test_auc"].to_list())

plt.figure(figsize=(6.4, 4.8))
ax = plt.axes()
plt.title(
    r"Dependence of $\mathrm{AUC}_{\mathrm{valid},\,\mathrm{av}}$ on $N_{\mathrm{trees}}$ and $\mathrm{d}_{\mathrm{max}}$",
    pad=20
)

X = n_estimators
Y = max_depth
Z = np.transpose(mean_test_auc)

cs = ax.contour(X, Y, Z, levels=10, colors="white", linestyles="solid", linewidths=0.5)

csf = ax.contourf(X, Y, Z, levels=10, cmap="cividis")

ax.clabel(CS=cs, fmt="%1.3f", inline=True, fontsize=8)
                   
ax.set_xlabel(r"Number of trees, $N_{\mathrm{trees}}$", fontdict={"fontsize": 10})
ax.set_ylabel(r"Maximum tree depth, $\mathrm{d}_{\mathrm{max}}$", fontdict={"fontsize": 10})

cbar = plt.colorbar(mappable=csf, label="$\mathrm{AUC}_{\mathrm{valid},\,\mathrm{av}}$", ax=ax)
cbar.add_lines(cs)

plt.show() 


best_estimator = gs.best_estimator_


dt_data["train"]["base"]["P_pred"] = best_estimator.predict_proba(
    X=dt_data["train"]["x"]
)[:, 1]

dt_data["train"]["base"]["y_pred"] = ((dt_data["train"]["base"]["P_pred"] >= 0.5)\
                                      .astype(dtype="int32"))


cm = confusion_matrix(
    y_true=dt_data["train"]["y"],
    y_pred=dt_data["train"]["base"]["y_pred"]
)

disp = ConfusionMatrixDisplay(cm).plot(cmap="cividis")
disp.ax_.set_title("Confusion matrix", pad=20);


cr = pd.DataFrame(
    classification_report(y_true=dt_data["train"]["y"],
                          y_pred=dt_data["train"]["base"]["y_pred"],
                          sample_weight=dt_data["train"]["base"]["sample_weight"],
                          output_dict=True)).transpose()

cr.loc["accuracy"] = ["---", "---", cr.loc["accuracy", "precision"], cr.loc["macro avg", "support"]]

cr["support"] = cr["support"].astype(int) 

dt_data["train"]["precision"] = cr["precision"].loc["1"]
dt_data["train"]["recall"] = cr["recall"].loc["1"]
dt_data["train"]["f1-score"] = cr["f1-score"].loc["1"]
dt_data["train"]["accuracy"] = cr["f1-score"].loc["accuracy"]

print()
display(cr.style.set_caption("Classification report")\
        .set_table_styles(
            [
                {"selector": "th.col_heading,td",
                 "props": [("width", "100px")]
                 },
                {"selector": "caption",
                 "props": [("font-size", "16px"),
                           ("font-weight", "bold"),
                           ("font-style", "italic")]
                 }
            ]))


# ---> Compute ROC curve, AUC and Gini coefficient for the traning dataset

# Add ROC's FPR and TPR to the dataset dictionary
(dt_data["train"]["FPR"], dt_data["train"]["TPR"], _) = roc_curve(
    y_true=dt_data["train"]["y"],
    y_score=dt_data["train"]["base"]["P_pred"]
)

# Add AUC to the dataset dictionary
dt_data["train"]["auc"] = roc_auc_score(
    y_true=dt_data["train"]["y"],
    y_score=dt_data["train"]["base"]["P_pred"]
)

# Add Gini coefficient to the dataset dictionary
dt_data["train"]["g"] = 2 * dt_data["train"]["auc"] - 1


plt.figure(figsize=(4.8, 4.8))
ax = plt.axes()
plt.title(rf"ROC curves, AUC and Gini coefficient", pad=20)

ax.plot(dt_data["train"]["FPR"],
        dt_data["train"]["TPR"],
        linestyle="solid",
        color="black",
        alpha=1,
        linewidth=1.5,
        label="Estimator's ROC")

ax.fill_between(
    x=dt_data["train"]["FPR"],
    y1=np.zeros(len(dt_data["train"]["FPR"])),
    y2=dt_data["train"]["TPR"],
    edgecolor="midnightblue",
    facecolor="midnightblue",
    linewidth=0,
    alpha=0.65,
    label=(r"Estimator's $\mathrm{AUC}$ ($\mathrm{AUC} = " +
           rf"{dt_data['train']['auc']:.3f}$)"))

ax.fill_between(
    x=dt_data["train"]["FPR"],
    y1=dt_data["train"]["FPR"],
    y2=dt_data["train"]["TPR"],
    edgecolor="yellow",
    facecolor="None",
    linewidth=0,
    hatch = "xxx",
    alpha=1,
    label=(r"Estimator's $1/2$ Gini ($G = " +
           rf"{dt_data['train']['g']:.3f}$)"))

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="dashed",
    color="black",
    alpha=1,
    linewidth=1.25,
    label="Random estimator's ROC")

ax.set_xlabel(r"$\mathrm{FPR}$", fontdict={"fontsize": 10})
ax.set_ylabel(r"$\mathrm{TPR}$", fontdict={"fontsize": 10})

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

ax.set_aspect("equal")

ax.legend(loc="upper left", fontsize=8)

plt.show() 


def get_stability_score(
    df_base,
    w_G_av=1,
    w_a=88.0,
    w_RMSD=-0.5
):
    G = df_base[["WEEK_NUM", "y", "P_pred"]]\
        .sort_values(by="WEEK_NUM")\
        .groupby(by="WEEK_NUM")[["y", "P_pred"]]\
        .apply(lambda x:
               2 * roc_auc_score(x["y"], x["P_pred"]) - 1).tolist()
    
    G_av = np.mean(G)

    i = np.arange(len(G))
    
    [a, b] = np.polyfit(x=i, y=G, deg=1)
    
    G_fit = a * i + b
    
    RMSD = np.sqrt(np.mean((G_fit - G)**2))

    stability_score = w_G_av * G_av + w_a * min(0, a) + w_RMSD * RMSD
    
    dt = {
        "g_week": G,
        "a": a,
        "b": b,
        "RMSD": RMSD,
        "stability_score": stability_score
    }
    
    return dt

dt_data["train"].update(get_stability_score(dt_data["train"]["base"]))


plt.figure(figsize=(6.4, 4.8))
ax = plt.axes()
plt.title(rf"Stability score's elements", pad=20)

ax.plot(range(len(dt_data["train"]["g_week"])),
        dt_data["train"]["g_week"],
        linestyle="None",
        marker="o",
        markeredgecolor="black",
        markerfacecolor="None",
        markersize=3,
        alpha=1,
        label="Points ($G_i$)")

i = np.array(range(len(dt_data["train"]["g_week"])))[[0, -1]]
ax.plot(i,
        dt_data["train"]["a"] * i + dt_data["train"]["b"],
        linestyle="solid",
        color="blue",
        alpha=1,
        linewidth=1,
        label=("Fitting line ($G_{\mathrm{fit}}(i) = a \cdot i +b$, with " +
               f"$a=${dt_data['train']['a']:.2e})"))
                
ax.set_xlabel(r"Week number, $i$", fontdict={"fontsize": 10})
ax.set_ylabel(r"Gini coefficient, $G$", fontdict={"fontsize": 10})

ax.minorticks_on()

ax.grid(visible=True, which="major", color="lightgray", linestyle="solid",
        linewidth=0.5)
ax.grid(visible=True, which="minor", color="lightgray", linestyle="dotted",
        linewidth=0.5)

# Legend
ax.legend(fontsize=8)

# Show plot
plt.show() 


dt_summary = {
    "dataset": "train",
    "precision": dt_data["train"]["precision"],
    "recall": dt_data["train"]["recall"],
    "f1-score": dt_data["train"]["f1-score"],
    "accuracy": dt_data["train"]["accuracy"],
    "auc": dt_data["train"]["auc"],
    "g": dt_data["train"]["g"],
    "stability_score": dt_data["train"]["stability_score"]
}

print()
display(pd.DataFrame(data=dt_summary, index=[0])\
        .style.set_caption("Performance metrics for the training data")\
        .set_table_styles(
            [
                {"selector": "th.col_heading,td",
                 "props": [("width", "100px")]
                 },
                {"selector": "caption",
                 "props": [("font-size", "16px"),
                           ("font-weight", "bold"),
                           ("font-style", "italic")]
                 }
            ]))


dt_data["test"]["base"]["P_pred"] = best_estimator.predict_proba(
    X=dt_data["test"]["x"]
)[:, 1]


submission = pd.DataFrame({
    "case_id": dt_data["test"]["base"]["case_id"].to_numpy(),
    "score": dt_data["test"]["base"]["P_pred"]
})

print()
display(submission.style.set_caption("Submission dataframe")\
        .set_table_styles(
            [
                {"selector": "th.col_heading,td",
                 "props": [("width", "100px")]
                 },
                {"selector": "caption",
                 "props": [("font-size", "16px"),
                           ("font-weight", "bold"),
                           ("font-style", "italic")]
                 }
            ]))


submission.to_csv("submission.csv", index=False)


masker = shap.maskers.Independent(
    data=dt_data["train"]["x"],
    max_samples=100)
explainer = shap.Explainer(model=best_estimator.predict_proba, masker=masker)


t_i = time.perf_counter()

shapley = explainer(dt_data["train"]["x"].sample(frac=0.005, random_state=42))

t_f = time.perf_counter()

print()
print(f"Running time: {(t_f - t_i)/3600:.2f} h")


shap.plots.bar(
    shap_values = shapley[:,:,1].abs.mean(0),
    max_display=11,
    show=False
)

plt.title(
    label="Feature importance, $I^{(j)}$\n (the mean absolute Shapley value, $|\phi^{(j)}|_{\mathrm{av}}$)",
    loc="center",
    pad=20,
    fontdict={
        "fontsize": 15,
        "verticalalignment": "baseline",
        "horizontalalignment": "center"})

plt.show()


shap.plots.beeswarm(
    shap_values = shapley[:,:,1], 
    max_display=11,
    color=shap.plots.colors.red_blue,
    show=False
)

plt.title(
    label="Features and respective Shapley values,\n sorted by their importance ($I^{(j)}$)",
    loc="center",
    pad=20,
    fontdict={
        "fontsize": 15,
        "verticalalignment": "baseline",
        "horizontalalignment": "center"})

plt.show()

