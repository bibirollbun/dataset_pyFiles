# We load the competition data

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from ydata_profiling import ProfileReport
from sklearn.preprocessing import (
    OrdinalEncoder, 
    OneHotEncoder,
    RobustScaler
)
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    accuracy_score, 
    roc_curve, 
    roc_auc_score
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold, 
    cross_val_score,
    RandomizedSearchCV
)
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# We load the data

bank_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col="id")


bank_train.shape


bank_train.head()


bank_train.describe()


bank_train.describe(exclude=np.number)


bank_train.info()


# We pass the data to the profiler

profile_bank_data = ProfileReport(bank_train, title="Bank Data Profiling Report")

# Get the full profile

profile_bank_data


# Establishing the seaborn aesthetic

sns.set_style("darkgrid")

# We establish the color palette

palette = sns.set_palette("Greens_r")


# Function for categorical variables

def plot_categories(data, figsize, sizes, labels, colors, explode, title):

    print(
    "\nNumber of null values: ", data.isnull().sum(),
    "\nUnique values: ", data.nunique(),
    "\nDistribution of values: \n", data.value_counts(), "\n\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors,
        wedgeprops={"edgecolor" : "k"},
        explode=explode,
        textprops={'fontsize': 12}
    )
    axes[0].set_ylabel("")
    
    sns.histplot(
        data=data,
        color="green",
        edgecolor="k",
        ax=axes[1]
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    
    # Add labels with the exact value above each bar
    
    for container in axes[1].containers:
        axes[1].bar_label(container, fontsize=12)
    
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Function for numeric variables

def plot_numeric(data, column, figsize, title):

    print(
    "\nNumber of null values: ", data[column].isnull().sum(),
    "\nUnique values: ", data[column].nunique(),
    "\nVariable range:", data[column].min(), "to", data[column].max(), "\n\n"
    )

    # We graph the distribution
    
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    sns.stripplot(
        data=data, 
        x=column,
        jitter=True,
        palette=palette,
        ax=axes[0]
    )
    sns.boxenplot(
        data=data, 
        x=column,
        palette=palette,
        ax=axes[1]
    )
    plt.suptitle(t=title)
    plt.tight_layout()
    plt.show()


# We create a variable for the analysis

subscribe_to = bank_train["y"].replace([0, 1], ["No", "Yes"]).astype("category")

# We print and graph the distribution

counts_t = subscribe_to.value_counts()
labels_t = counts_t.index
sizes_t = counts_t.values
title_t = "Target variable distribution of values"

plot_categories(subscribe_to, (14, 6), sizes_t, labels_t, ["#fd5565","#8bd33f"], (0.0, 0.1), title_t)


# We print and graph the distribution

plot_numeric(bank_train, "age", (12, 4), "Distribution of values of the age variable")


# We print the general information of the variable

print(
    "\nNumber of null values: ", bank_train["job"].isnull().sum(),
    "\nUnique values: ", bank_train["job"].nunique(),
    "\nDistribution of values: \n", bank_train["job"].value_counts(), "\n\n"
    )

# We analyze the distribution

counts_j = bank_train["job"].value_counts()
labels_j = counts_j.index
sizes_j = counts_j.values

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

explode = (0.1, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.2, 0.3, 0.2, 0.3)

axes[0].pie(
    sizes_j,
    autopct='%1.1f%%',
    pctdistance=1.20,
    explode=explode,
    startangle=90,
    colors=sns.color_palette("Paired"),
    wedgeprops={"edgecolor" : "k"}
)
axes[0].set_ylabel("")
axes[0].legend(
    labels_j,
    title="Jobs",
    loc="upper left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)
sns.histplot(
    data=bank_train["job"],
    color="green",
    edgecolor="k",
    ax=axes[1]
).tick_params(axis='x', labelrotation=45)
axes[1].set_xlabel("")
axes[1].set_ylabel("")

# Add labels with the exact value above each bar

for container in axes[1].containers:
    axes[1].bar_label(container, fontsize=12)

fig.suptitle("Job variable distribution of values", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# We print and graph the distribution

counts_m = bank_train["marital"].value_counts()
labels_m = counts_m.index
sizes_m = counts_m.values
color_m = ["#409e70", "#7b79c7", "#d7d684"]
explode_m = (0.0, 0.1, 0.2)
title_m = "Marital Status variable distribution of values"

plot_categories(bank_train["marital"], (14, 6), sizes_m, labels_m, color_m, explode_m, title_m)


# We print and graph the distribution

counts_e = bank_train["education"].value_counts()
labels_e = counts_e.index
sizes_e = counts_e.values
color_e = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db"]
explode_e = (0.1, 0.2, 0.1, 0.4)
title_e = "Education distribution of values"

plot_categories(bank_train["education"], (14, 6), sizes_e, labels_e, color_e, explode_e, title_e)


# We print and graph the distribution

counts_d = bank_train["default"].value_counts()
labels_d = counts_d.index
sizes_d = counts_d.values
title_d = "Default Variable Distribution"

plot_categories(bank_train["default"], (14, 6), sizes_d, labels_d, ["#fd5565","#8bd33f"], (0.0, 0.5), title_d)


# We print and graph the distribution

plot_numeric(bank_train, "balance", (12, 4), "Balance Distribution of Values")


# We print and graph the distribution

counts_h = bank_train["housing"].value_counts()
labels_h = counts_h.index
sizes_h = counts_h.values
title_h = "Housing Variable Distribution"

plot_categories(bank_train["housing"], (14, 6), sizes_h, labels_h, ["#fd5565","#8bd33f"], (0.0, 0.1), title_h)


# We print and graph the distribution

counts_l = bank_train["loan"].value_counts()
labels_l = counts_l.index
sizes_l = counts_l.values
title_l = "Loan Variable Distribution"

plot_categories(bank_train["loan"], (14, 6), sizes_l, labels_l, ["#fd5565","#8bd33f"], (0.0, 0.1), title_l)


# We print and graph the distribution

counts_c = bank_train["contact"].value_counts()
labels_c = counts_c.index
sizes_c = counts_c.values
color_c = ["#409e70", "#7b79c7", "#d7d684"]
explode_c = (0.1, 0.1, 0.2)
title_c = "Contact Information Distribution"

plot_categories(bank_train["contact"], (14, 6), sizes_c, labels_c, color_c, explode_c, title_c)


# We print the general information of the variable

print(
    "\nNumber of null values Day: ", bank_train["day"].isnull().sum(),
    "\nUnique values Day: ", bank_train["day"].nunique(),
    "\nVariable range Day:", bank_train["day"].min(), "to", bank_train["day"].max(), "\n\n"
    "\nNumber of null values Month: ", bank_train["month"].isnull().sum(),
    "\nUnique values Month: ", bank_train["month"].nunique(),
    "\nDistribution of values: \n", bank_train["month"].value_counts(), "\n\n"
    )

# We analyze the distribution

counts_mo = bank_train["month"].value_counts()
labels_mo = counts_mo.index
sizes_mo = counts_mo.values

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

explode = (0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2)

axes[0].pie(
    sizes_mo,
    autopct='%1.1f%%',
    pctdistance=1.20,
    explode=explode,
    startangle=90,
    colors=sns.color_palette("Paired"),
    wedgeprops={"edgecolor" : "k"}
)
axes[0].set_xlabel("Month of the Year")
axes[0].set_ylabel("")
axes[0].legend(
    labels_mo,
    title="Months",
    loc="upper left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)
sns.histplot(
    data=bank_train["day"],
    color="green",
    edgecolor="k",
    ax=axes[1]
)
axes[1].set_xlabel("Day of the Month")
axes[1].set_ylabel("")

fig.suptitle("We analyze the distribution of the variables day and month", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# We print and graph the distribution

plot_numeric(bank_train, "duration", (12, 4), "Duration Distribution of Values")


# We print and graph the distribution

plot_numeric(bank_train, "campaign", (12, 4), "Campaign Distribution of Values")


# We print and graph the distribution

plot_numeric(bank_train, "pdays", (12, 4), "Pdays Distribution of Values")


# We print and graph the distribution

plot_numeric(bank_train, "previous", (12, 4), "Previous Distribution of Values")


# We print and graph the distribution

counts_p = bank_train["poutcome"].value_counts()
labels_p = counts_p.index
sizes_p = counts_p.values
color_p = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db"]
explode_p = (0.1, 0.2, 0.3, 0.5)
title_p = "Previous Outcome distribution of values"

plot_categories(bank_train["poutcome"], (14, 6), sizes_p, labels_p, color_p, explode_p, title_p)


# We make a copy of the original dataset

bank_new = bank_train.copy()


# We confirm that there is no null values

null_values = pd.DataFrame(
        {"Null Data" : bank_new.isnull().sum(), 
         "Percentage" : (bank_new.isnull().sum()) / (len(bank_new)) * (100)})

null_values


# We check for duplicate data

print(f"Length: {len(bank_new.duplicated())}")
print(f"Duplicates: {bank_new.duplicated().sum()}")


# We set the target labels for analysis

bank_new["y"] = bank_new["y"].replace([0, 1], ["Will not subscribe", "Will subscribe"])


# We replace the values with categorical ones

bank_new["pdays"] = np.where(bank_new["pdays"] >= 0, "Yes", "No")
bank_new["previous"] = np.where(bank_new["previous"] > 0, "Yes", "No")


# We replace with an upper threshold(95%) and lower threshold(5%) approximate value

bank_new["age"] = bank_new["age"].clip(lower=18, upper=70).round(decimals=0)
bank_new["balance"] = bank_new["balance"].clip(lower=-287, upper=3400).round(decimals=0)
bank_new["duration"] = bank_new["duration"].clip(lower=1, upper=750).round(decimals=0)
bank_new["campaign"] = bank_new["campaign"].clip(lower=1, upper=6).round(decimals=0)


fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 4))

sns.boxplot(x=bank_new["age"], palette=palette, ax=axes[0,0])
sns.boxplot(x=bank_new["balance"], palette=palette, ax=axes[0,1])
sns.boxplot(x=bank_new["duration"], palette=palette, ax=axes[1,0])
sns.boxplot(x=bank_new["campaign"], palette=palette, ax=axes[1,1])

plt.suptitle(t="Outliers Analysis")
plt.tight_layout()
plt.show()


bank_new.describe()


bank_new.describe(exclude=np.number)


# We create the new column "year"

bank_new["year"] = 2024

# We map the "month" column to avoid pandas datetime errors

month_map = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

# We create a new column with the numeric values of the month

bank_new["month"] = bank_new["month"].str.lower().map(month_map)


# We combine the three columns into a new column "date" of type datetime

bank_new["date"] = pd.to_datetime(bank_new[["year", "month", "day"]], errors="coerce")

# We check all the values in the column

day_data = bank_new["date"].dt.day
month_data = bank_new["date"].dt.month
year_data = bank_new["date"].dt.year

print(
    "\nUnique day values: ", day_data.unique(),
    "\nUnique values of the month: ", month_data.unique(),
    "\nUnique values of year: ", year_data.unique()
)


# We separate the erroneous rows

rows_with_error = bank_new[bank_new["date"].isnull()]

# We show the problematic rows for inspection

rows_with_error


# We delete the rows with incorrect dates

bank_new.dropna(subset=["date"], inplace=True)


# We changed the format for more efficient memory usage

bank_new[bank_new.select_dtypes(["object"]).columns] = (
    bank_new.select_dtypes(["object"]).apply(
        lambda x: x.astype("category"))
)


bank_new.info()


age_y = bank_new.pivot(columns="y", values="age")

age_y.describe().T


campaign_age = bank_new.pivot(columns="age", values="campaign")

campaign_age.describe()


# We analyze the Age

fig, axes = plt.subplots(figsize=(12, 4))

sns.barplot(
    data=bank_new, 
    x="age", 
    y="campaign", 
    hue="y",
    edgecolor="k", 
    palette="Paired"
)
plt.legend().set_title("")
plt.suptitle(t="Impact of Age on the Campaign")
plt.tight_layout()
plt.show()


# We analyze the clients job by y

fig, axes = plt.subplots(figsize=(10, 4))

sns.histplot(
    data=bank_new, 
    x="job",
    hue="y", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
).tick_params(axis='x', labelrotation=45)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title("Clients Job by whether they will subscribe or Not")
plt.tight_layout()
plt.show()


# We analyze the marital status by target

fig, axes = plt.subplots(figsize=(10, 4))

sns.histplot(
    data=bank_new,
    x="marital",
    hue="y",
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)
plt.title("Marital Status by whether the client subscribe or Not")
plt.tight_layout()
plt.show()


# We analyze the housing situation by target

fig, axes = plt.subplots(figsize=(10, 4))

sns.histplot(
    data=bank_new, 
    x="housing",
    hue="y",
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)
plt.title("Housing Situation by whether the client subscribe or Not")
plt.tight_layout()
plt.show()


# We analyze the clients education by y

fig, axes = plt.subplots(figsize=(10, 4))

sns.histplot(
    data=bank_new, 
    x="education",
    hue="y", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
).tick_params(axis='x', labelrotation=45)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title("Education by whether they will subscribe or Not")
plt.tight_layout()
plt.show()


# We analyze the clients education by job

fig, axes = plt.subplots(figsize=(10, 4))

sns.histplot(
    data=bank_new, 
    x="job",
    hue="education", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
).tick_params(axis='x', labelrotation=45)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title("Education by Clients Jobs")
plt.tight_layout()
plt.show()


balance_y = bank_new.pivot(columns="y", values="balance")

balance_y.describe()


# We analyze the balance of clients

fig, axes = plt.subplots(figsize=(10, 4))

sns.histplot(
    data=bank_new, 
    x="balance",
    hue="y", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title("Clients Balance by whether they will subscribe or Not")
plt.tight_layout()
plt.show()


# We analyze the client balance in time

fig, axes = plt.subplots(figsize=(12, 4))

sns.lineplot(
    data=bank_new, 
    x="date",
    y="balance",
    hue="y",
    palette="Paired"
)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title(label="Evolution of clients balance over time(year)")
plt.tight_layout()
plt.show()


# We analyze the client balance in time

fig, axes = plt.subplots(nrows=2, figsize=(16, 6))

sns.lineplot(
    data=bank_new, 
    x="day",
    y="balance",
    hue="y",
    palette="Paired",
    ax=axes[0]
)
sns.lineplot(
    data=bank_new, 
    x="month",
    y="balance",
    hue="y",
    palette="Paired",
    ax=axes[1]
)
axes[0].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
axes[1].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
plt.title(label="Evolution of clients balance over time(day & month)")
plt.tight_layout()
plt.show()


contact_c = bank_new.pivot(columns="contact", values="campaign")

contact_c.describe()


contact_d = bank_new.pivot(columns="contact", values="duration")

contact_d.describe()


# We analyze the clients contact information by y and time

fig, axes = plt.subplots(nrows=2, figsize=(16, 6))

sns.barplot(
    data=bank_new, 
    x="contact", 
    y="campaign", 
    hue="y",
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.lineplot(
    data=bank_new, 
    x="month",
    y="campaign",
    hue="contact",
    palette="Paired",
    ax=axes[1]
)
axes[0].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
axes[1].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
plt.suptitle(t="Contact information by whether they will subscribe or Not")
plt.tight_layout()
plt.show()


# We analyze the duration of contacted clients

fig, axes = plt.subplots(figsize=(10, 4))

sns.histplot(
    data=bank_new, 
    x="duration",
    hue="y", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title("Impact of contact time during the campaign")
plt.tight_layout()
plt.show()


previous_campaign = bank_new[["pdays", "previous", "poutcome", "y"]]

previous_campaign.describe(exclude=np.number)


# We analyze the previous campaign and the outcome

fig, axes = plt.subplots(nrows=2, figsize=(16, 6))

sns.histplot(
    data=bank_new, 
    x="poutcome",
    hue="pdays", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes[0]
)
sns.histplot(
    data=bank_new, 
    x="poutcome",
    hue="y", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)

plt.suptitle(t="Previous campaigns and their results")
plt.tight_layout()
plt.show()


# We separate the target variable from the features

x_bank = bank_train.drop(columns="y")
y_bank = bank_train["y"]


# We review the balance of the target variable

y_balance = np.asarray(np.unique(y_bank, return_counts=True))

print(y_balance)


# We modify the variable month with its respective numerical values

x_bank["month"] = x_bank["month"].str.lower().map(month_map)


# We separate the Age variable into bins

bins_age = [0, 35, 50, 100]
labels_age = ["Young", "MiddleAge", "Old"]
x_bank["age_bins"] = pd.cut(x_bank["age"], bins_age, labels=labels_age)
x_bank["age_bins"].value_counts()


# We separate the categorical variables from the numerical ones

df_number = x_bank.select_dtypes(include="number")
df_ordinal = x_bank[["job", "marital", "education", "contact", "poutcome", "age_bins"]]
df_onehot = x_bank[["default", "housing", "loan"]]


# We define the encoders an fit de data

enc_ordinal = OrdinalEncoder(categories="auto").set_output(transform="pandas")
enc_ordinal.fit(df_ordinal)
enc_onehot = OneHotEncoder(sparse=False, drop="if_binary").set_output(transform="pandas")
enc_onehot.fit(df_onehot)

# We apply the transformations

enc_group_one = enc_ordinal.transform(df_ordinal)
enc_group_two = enc_onehot.transform(df_onehot)

# We changed the name of the encoded columns

enc_group_two.rename(
    columns={"default_yes" : "default", 
             "housing_yes" : "housing",
             "loan_yes" : "loan"}, 
    inplace=True)


# We concatenate the resulting data

x_bank_new = pd.concat([df_number, enc_group_one, enc_group_two], axis=1)

x_bank_new.info()


# We apply a logarithmic transformation before scaling

for col in ["balance", "duration", "campaign", "pdays", "previous"]:
    # Only apply to positive values in "pdays"
    if col == "pdays":
        x_bank_new.loc[x_bank_new["pdays"] != -1, "pdays"] = np.log1p(
            x_bank_new.loc[x_bank_new["pdays"] != -1, "pdays"])
    else:
         x_bank_new[col] = np.log1p(x_bank_new[col] - x_bank_new[col].min())


x_bank_new.describe().T


# We separate the data into training and validation sets

x_train, x_val, y_train, y_val = (
    train_test_split(x_bank_new, y_bank, test_size=0.2, random_state=42)
)


# We apply the scaler to the data

scaler = RobustScaler().set_output(transform="pandas")
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)


x_train_scaled.describe().T


x_val_scaled.describe().T


# We graph the correlation between the variables

matrix_bank = x_train_scaled.corr(numeric_only=True).round(2)

plt.figure(figsize=(10, 4))

sns.heatmap(
    matrix_bank, 
    annot=True,
    cmap=sns.cubehelix_palette(rot=-.2)
    )


# We analyze the mutual information between each feature and the target variable

bank_scores = mutual_info_classif(x_train_scaled, y_train)
bank_scores = pd.Series(bank_scores, name="Bank Data MI Scores", index=x_train_scaled.columns)
bank_scores = bank_scores.sort_values(ascending=False)
bank_scores


# We visualize the results

scores = bank_scores.sort_values()
width = np.arange(len(bank_scores))
ticks = list(bank_scores.index)
plt.barh(width, bank_scores, color="c", edgecolor="k")
plt.yticks(width, ticks)
plt.title("Mutual Information Scores")
plt.figure(dpi=100, figsize=(8, 5))
plt.show()


# We create a function to evaluate the Accuracy and F1 Scores

def score_evaluator(model, xtrain, ytrain, xval, yval):

    '''
    Steps:
        We calculate the precision score.
        We obtain prediction from the model.
        We calculate the f1 score.
        Train result and accuracy score test.
        Print train and test macro f1 score.
    '''

    train_ascore = model.score(xtrain, ytrain)
    val_ascore = model.score(xval, yval)

    y_train_pred = model.predict(xtrain)
    y_val_pred = model.predict(xval)

    train_fscore = f1_score(ytrain, y_train_pred, average="macro")
    val_fscore = f1_score(yval, y_val_pred, average="macro")

    print(f"Train -------> Accuracy score: {train_ascore}")
    print(f"Validation --> Accuracy score: {val_ascore}\n")
    print(f"Train -------> F1-Score: {train_fscore}")
    print(f"Validation --> F1-Score: {val_fscore}")


# We create a function to evaluate the confusion matrix and report

def matrix_evaluator(model, xval, yval, list_classes, color_map):
    '''
    Steps:
        We create the confusion matrix.
        We plot the confusion matrix.
        Report results.
    '''
    y_pred = model.predict(xval)
    cm_values = confusion_matrix(yval, y_pred)
    df_cm = pd.DataFrame(
        cm_values,
        columns=list_classes,
        index=list_classes
        )
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap=color_map)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print("\n")
    print(classification_report(
        yval, y_pred, target_names=list_classes
    ))


# Function to calculate the roc auc score

def auc_evaluator(model, x, y):

    probs = model.predict_proba(x)
    probs = probs[:, 1]
    model_auc = roc_auc_score(y, probs)
    print(f"\n {model}: ROC AUC = %.3f" % (model_auc))


# We create a function to evaluate the ROC Curve

def roc_curve_evaluator(y_true, x_data, models_dict):
    '''
    Args:
        y_true (array-like): True labels of the target variable.
        X_data (DataFrame): Data to generate predictions.
        models_dict (dict): Dictionary with {"model_name": trained_model_object}.
    Steps:
        Plot the baseline (untrained model).
        Iterate over each model in the dictionary.
        Predict probabilities for the positive class.
        Calculate AUC and ROC curve.
        Print the score and graph the curve.
    '''
    plt.figure(figsize=(10, 8))
    
    ns_probs = [0 for _ in range(len(y_true))]
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    
    plt.plot(ns_fpr, ns_tpr, linestyle="--", color="r", label="Untrained model (AUC = 0.500)")

    for name, model in models_dict.items():
        model_probs = model.predict_proba(x_data)[:, 1]
        model_auc = roc_auc_score(y_true, model_probs)
        model_fpr, model_tpr, _ = roc_curve(y_true, model_probs)
        print(f'{name}: ROC AUC = {model_auc:.4f}')
        plt.plot(model_fpr, model_tpr, marker=".", label=f"{name} (AUC = {model_auc:.4f})")

    plt.title("ROC Curve", fontsize=16)
    plt.ylabel("True Positive Rate (Recall)")
    plt.xlabel("False Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()


# We create the model instance

lrc = LogisticRegression(class_weight="balanced")

# Train the model with the data

lrc.fit(x_train_scaled, y_train)


# We evaluate the accuracy and the f1-score

score_evaluator(lrc, x_train_scaled, y_train, x_val_scaled, y_val)


# We graph the confusion matrix

matrix_evaluator(
    lrc, 
    x_val_scaled, 
    y_val, 
    ["0", "1"], 
    "Blues"
)


# We calculate the roc auc score

auc_evaluator(lrc, x_val_scaled, y_val)


# We create the model instance

lgbmc = LGBMClassifier(is_unbalance=True, objective="binary", verbose=-1)

# Train the model with the data

lgbmc.fit(x_train_scaled, y_train)


# We evaluate the accuracy and the f1-score

score_evaluator(lgbmc, x_train_scaled, y_train, x_val_scaled, y_val)


# We graph the confusion matrix

matrix_evaluator(
    lgbmc, 
    x_val_scaled, 
    y_val, 
    ["0", "1"], 
    "Blues"
)


# We calculate the roc auc score

auc_evaluator(lgbmc, x_val_scaled, y_val)


# We create the model instance

cbc = CatBoostClassifier(silent=True)

# Train the model with the data

cbc.fit(x_train_scaled, y_train)


# We evaluate the accuracy and the f1-score

score_evaluator(cbc, x_train_scaled, y_train, x_val_scaled, y_val)


# We graph the confusion matrix

matrix_evaluator(
    cbc, 
    x_val_scaled, 
    y_val, 
    ["0", "1"], 
    "Blues"
)


# We calculate the roc auc score

auc_evaluator(cbc, x_val_scaled, y_val)


# We apply the ROC Curve function

models_to_test = {
    "LogisticRegression" : lrc,
    "LGBMClassifier" : lgbmc,
    "CatBoostClassifier" : cbc
}

roc_curve_evaluator(y_val, x_val_scaled, models_to_test)


# Create the StratifiedKFold object with 5 divisions (k=5)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# We establish the parameters to test

params_grid = {
    "learning_rate" : uniform(0.01, 0.3),
    "iterations": randint(200, 1500),
    "depth" : randint(4, 10),
    "l2_leaf_reg" : randint(1, 10),
    "bagging_temperature": uniform(0.0, 1.0),
    "random_strength" : uniform(0.0, 1.0)
}

# We use RandomizedSearchCV and cv method to evaluate

cbc_grid = RandomizedSearchCV(
    CatBoostClassifier(silent=True),
    params_grid,
    cv=skf,
    scoring="roc_auc",
    return_train_score=True
)

cbc_search = cbc_grid.fit(x_train_scaled, y_train)
print(f"Parameters: {cbc_search.best_params_}\nScore: {cbc_search.best_score_}")


# We save the results within a dataframe

cbc_cv_results = pd.DataFrame(cbc_search.cv_results_)

cbc_cv_results.head().sort_values(by="mean_test_score", ascending=True).T


# We fit the best estimator

cbc_result = cbc_search.best_estimator_  
cbc_result.fit(x_train_scaled, y_train)


# We calculate the roc auc score

auc_evaluator(cbc_result, x_val_scaled, y_val)


# We create an explainer for the best estimator

explainer = shap.Explainer(cbc_result)
shap_values = explainer.shap_values(x_val_scaled)

# we visualize the importance

fig = shap.summary_plot(shap_values, x_val_scaled, show=False)
plt.title("Feature Importance", fontsize=20, color="b", loc="left")
plt.xlabel("Mean SHAP Values", fontsize=20)
plt.ylabel("Features", fontsize=20)
plt.show()


# We load the test data

df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# We check the shape and that no duplicate data is found

print(f"Length: {len(df_test.duplicated())}")
print(f"Duplicates: {df_test.duplicated().sum()}")
print(f"Shape: {df_test.shape}")


df_test.info()


# We start by removing the variables that we will not use

df_test_new = df_test.drop(columns=["id"])


# We confirm that there is no null values in test data

null_test = pd.DataFrame(
        {"Null Data" : df_test_new.isnull().sum(), 
         "Percentage" : (df_test_new.isnull().sum()) / (len(df_test_new)) * (100)})

null_test


# We create a new column with the numeric values of the month

df_test_new["month"] = df_test_new["month"].str.lower().map(month_map)

# We separate the Age variable into bins

df_test_new["age_bins"] = pd.cut(df_test_new["age"], bins_age, labels=labels_age)


# We separate the categorical variables from the numerical ones

df_number_test = df_test_new.select_dtypes(include="number")
df_ordinal_test = df_test_new[["job", "marital", "education", "contact", "poutcome", "age_bins"]]
df_onehot_test = df_test_new[["default", "housing", "loan"]]

# We apply the transformations

test_group_one = enc_ordinal.transform(df_ordinal_test)
test_group_two = enc_onehot.transform(df_onehot_test)

# We changed the name of the encoded columns

test_group_two.rename(
    columns={"default_yes" : "default", 
             "housing_yes" : "housing",
             "loan_yes" : "loan"}, 
    inplace=True)

x_test = pd.concat([df_number_test, test_group_one, test_group_two], axis=1)


# We apply a logarithmic transformation before scaling

for col in ["balance", "duration", "campaign", "pdays", "previous"]:
    if col == "pdays":
        x_test.loc[x_test["pdays"] != -1, "pdays"] = np.log1p(
            x_test.loc[x_test["pdays"] != -1, "pdays"])
    else:
         x_test[col] = np.log1p(x_test[col] - x_test[col].min())

# We apply a scaler to the data

x_test_scaled = scaler.transform(x_test)

x_test_scaled.describe().T


# We apply the trained model

bank_predictions = cbc_result.predict_proba(x_test_scaled)

# We review the result

print("Total predictions: ", len(bank_predictions), "\n")

# We create the dataframe

single_submission = pd.DataFrame({
    "id" : df_test["id"], 
    "y" : bank_predictions[:, 1]
})

single_submission.head()


# We load the submission sample data

bank_sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


# We compare the results with the sample

print(
    f"Shape Sample Submission: {bank_sample.shape}",
    f"\nShape Bank Submission: {single_submission.shape}"
)
print("\n", bank_sample.head())


# We convert the dataframe to a csv file

single_submission.to_csv("submission.csv", index=False)

