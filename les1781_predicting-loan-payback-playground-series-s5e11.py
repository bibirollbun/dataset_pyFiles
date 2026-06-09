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

from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_val_score
)
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier


# We load the data

lpb_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv", index_col="id")


print("Train data shape:", lpb_train.shape)


lpb_train.head()


lpb_train.describe()


lpb_train.describe(exclude=np.number)


lpb_train.info()


# Establishing the seaborn aesthetic

sns.set_style("darkgrid")

# We establish the color palette

palette = sns.set_palette("Greens_r")


# Function to analyze number distributions

def plot_number_analyzer(data, column, figsize, suptitle):

    print(
    "Variable: ", column,
    "\nFormat: ", data[column].dtype,
    "\nNumber of null values: ", data[column].isnull().sum(),
    "\nUnique values: ", data[column].nunique(),
    "\nVariable range:", data[column].min(), "to", data[column].max(), "\n\n"
    )

    # We graph the distribution
    
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    sns.histplot(
        data=data, 
        x=column, 
        palette=palette,
        edgecolor="k",
        ax=axes[0]
    )
    sns.boxplot(
        data=data, 
        x=column,
        palette=palette,
        ax=axes[1]
    )
    plt.suptitle(t=suptitle)
    plt.tight_layout()
    plt.show()


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


# We create a variable for the analysis

subscribe_to = lpb_train["loan_paid_back"].replace([0, 1], ["No", "Yes"]).astype("category")

# We print and graph the distribution

counts_t = subscribe_to.value_counts()
labels_t = counts_t.index
sizes_t = counts_t.values
color_t = ["#409e70", "#7b79c7"]
explode_t = (0.0, 0.2)
title_t = "Target variable distribution of values"

plot_categories(subscribe_to, (12, 4), sizes_t, labels_t, color_t, explode_t, title_t)


# We print and graph the distribution

plot_number_analyzer(
    lpb_train, 
    "annual_income", 
    (12, 4), 
    "Distribution of values of the 'annual_income' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    lpb_train, 
    "debt_to_income_ratio", 
    (12, 4), 
    "Distribution of values of the 'debt_to_income_ratio' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    lpb_train, 
    "credit_score", 
    (12, 4), 
    "Distribution of values of the 'credit_score' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    lpb_train, 
    "loan_amount", 
    (12, 4), 
    "Distribution of values of the 'loan_amount' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    lpb_train, 
    "interest_rate", 
    (12, 4), 
    "Distribution of values of the 'interest_rate' variable"
)


# We print and graph the distribution

counts_g = lpb_train["gender"].value_counts()
labels_g = counts_g.index
sizes_g = counts_g.values
color_g = ["#409e70", "#7b79c7", "#d7d684"]
explode_g = (0.0, 0.1, 0.2)
title_g = "Gender variable distribution of values"

plot_categories(lpb_train["gender"], (14, 6), sizes_g, labels_g, color_g, explode_g, title_g)


# We print and graph the distribution

counts_m = lpb_train["marital_status"].value_counts()
labels_m = counts_m.index
sizes_m = counts_m.values
color_m = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db"]
explode_m = (0.1, 0.2, 0.3, 0.5)
title_m = "Marital Status variable distribution of values"

plot_categories(lpb_train["marital_status"], (14, 6), sizes_m, labels_m, color_m, explode_m, title_m)


# We print and graph the distribution

counts_e = lpb_train["education_level"].value_counts()
labels_e = counts_e.index
sizes_e = counts_e.values
color_e = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db", "#8d0c34"]
explode_e = (0.1, 0.2, 0.3, 0.2, 0.4)
title_e = "Education Level variable distribution of values"

plot_categories(lpb_train["education_level"], (14, 6), sizes_e, labels_e, color_e, explode_e, title_e)


# We print and graph the distribution

counts_es = lpb_train["employment_status"].value_counts()
labels_es = counts_es.index
sizes_es = counts_es.values
color_es = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db", "#8d0c34"]
explode_es = (0.1, 0.2, 0.3, 0.2, 0.4)
title_es = "Employment Status variable distribution of values"

plot_categories(lpb_train["employment_status"], (14, 6), sizes_es, labels_es, color_es, explode_es, title_es)


# We print and graph the distribution

counts_l = lpb_train["loan_purpose"].value_counts()
labels_l = counts_l.index
sizes_l = counts_l.values
color_l = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db", "#8d0c34", "#ca8d7e", "#687f7a", "#227bc9"]
explode_l = (0.1, 0.2, 0.3, 0.2, 0.4, 0.3, 0.3, 0.5)
title_l = "Loan Purpose variable distribution of values"

plot_categories(lpb_train["loan_purpose"], (14, 6), sizes_l, labels_l, color_l, explode_l, title_l)


# We print the general information of the variable

print(
    "\nNumber of null values: ", lpb_train["grade_subgrade"].isnull().sum(),
    "\nUnique values: ", lpb_train["grade_subgrade"].nunique(), "\n\n"
    )

# We analyze the distribution

fig, axes = plt.subplots(figsize=(16, 6))

g = sns.histplot(
    data=lpb_train, 
    x="grade_subgrade", 
    color="green",
    edgecolor="k"
)

# Add labels with the exact value above each bar
    
for container in g.containers:
    g.bar_label(container, fontsize=12)

plt.title(label="We analyze the distribution of the variable Grade Subgrade", fontsize=18)
plt.tight_layout()
plt.show()


# We make a copy of the original dataset

lpb_new = lpb_train.copy()


# We confirm that there is no null values

null_values = pd.DataFrame(
        {"Null Data" : lpb_new.isnull().sum(), 
         "Percentage" : (lpb_new.isnull().sum()) / (len(lpb_new)) * (100)})

null_values


# We check for duplicate data

print(f"Length: {len(lpb_new.duplicated())}")
print(f"Duplicates: {lpb_new.duplicated().sum()}")


# Function to analyze outliers

def outlier_analyzer(df):

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    n_cols = len(numerical_cols)
    n_rows = int(np.ceil(n_cols / 3))
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(18, 9))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=df[col], ax=axes[i], color="g")
        axes[i].set_title(f"Boxplot de {col}", fontsize=12)
        axes[i].set_xlabel("Feature values", fontsize=10)
        axes[i].set_ylabel("")
    for i in range(n_cols, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle("Distribution Analysis with Boxplots", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()


# We analyze the outliers

numerical_cols = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]

outlier_analyzer(lpb_new[numerical_cols])


# We handle outliers

for col in numerical_cols:
    Q1 = lpb_new[col].quantile(0.25)
    Q3 = lpb_new[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    lpb_new[col] = lpb_new[col].clip(lower=lower_bound, upper=upper_bound)


outlier_analyzer(lpb_new[numerical_cols])


# We map the variable for analysis

lpb_new["loan_paid_back"].replace([0, 1], ["No", "Yes"], inplace=True)

# We transform the variable into binary using the mode

lpb_new["gender"].replace("Other", "Female", inplace=True)


ai_pay = lpb_new.pivot(columns="loan_paid_back", values="annual_income")

ai_pay.describe().T


# # We graph the relationship

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=lpb_new, 
    x="annual_income",
    hue="loan_paid_back", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)

sns.move_legend(
    axes, "upper left",
    bbox_to_anchor=(1, 1), 
    ncol=7, 
    title="Loan paid back", 
    frameon=True
)

plt.title("Annual income distribution by Loan paid back")
plt.tight_layout()
plt.show()


cs_pay = lpb_new.pivot(columns="loan_paid_back", values="credit_score")

cs_pay.describe().T


# # We graph the relationship

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=lpb_new, 
    x="credit_score",
    hue="loan_paid_back", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)

sns.move_legend(
    axes, "upper left",
    bbox_to_anchor=(1, 1), 
    ncol=7, 
    title="Loan paid back", 
    frameon=True
)

plt.title("Credit score distribution by Loan paid back")
plt.tight_layout()
plt.show()


ls_pay = lpb_new.pivot(columns="loan_paid_back", values="loan_amount")

ls_pay.describe().T


# We graph the relationship

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=lpb_new, 
    x="loan_amount",
    hue="loan_paid_back", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)

sns.move_legend(
    axes, "upper left",
    bbox_to_anchor=(1, 1), 
    ncol=7, 
    title="Loan paid back", 
    frameon=True
)

plt.title("Loan amount distribution by Loan paid back")
plt.tight_layout()
plt.show()


ir_pay = lpb_new.pivot(columns="loan_paid_back", values="interest_rate")

ir_pay.describe().T


# We graph the relationship

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=lpb_new, 
    x="interest_rate",
    hue="loan_paid_back", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)

sns.move_legend(
    axes, "upper left",
    bbox_to_anchor=(1, 1), 
    ncol=7, 
    title="Loan paid back", 
    frameon=True
)

plt.title("Interest rate distribution by Loan paid back")
plt.tight_layout()
plt.show()


# We graph the relationship

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=lpb_new, 
    x="employment_status",
    hue="loan_paid_back", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette=["green", "red"],
    ax=axes
)

sns.move_legend(
    axes, "upper left",
    bbox_to_anchor=(1, 1), 
    ncol=7, 
    title="Loan paid back", 
    frameon=True
)

plt.title("Employment status distribution by Loan paid back")
plt.tight_layout()
plt.show()


# We graph the relationship

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=lpb_new, 
    x="loan_purpose",
    hue="loan_paid_back", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette=["green", "red"],
    ax=axes
)

sns.move_legend(
    axes, "upper left",
    bbox_to_anchor=(1, 1), 
    ncol=7, 
    title="Loan paid back", 
    frameon=True
)

plt.title("Loan purpose distribution by Loan paid back")
plt.tight_layout()
plt.show()


# We separate the target variable from the features

x_lpb = lpb_train.drop(columns="loan_paid_back")
y_lpb = lpb_train["loan_paid_back"]


# We separate the data into training and validation sets

x_train, x_val, y_train, y_val = (
    train_test_split(x_lpb, y_lpb, test_size=0.2, random_state=42)
)


# We handle outliers

numerical_cols = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]

for col in numerical_cols:
    Q1 = x_train[col].quantile(0.25)
    Q3 = x_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    x_train[col] = x_train[col].clip(lower=lower_bound, upper=upper_bound)
    x_val[col] = x_val[col].clip(lower=lower_bound, upper=upper_bound)


# We separate the columns that we want to transform

cat_cols = x_train.select_dtypes(include=["object"])

# We initialize the Target Encoder

t_encoder = TargetEncoder(cols=cat_cols)
t_encoder.fit(x_train, y_train)

# We perform the transformation on the data

x_train_encoded = t_encoder.transform(x_train)
x_val_encoded = t_encoder.transform(x_val)


x_train_encoded.info()


x_val_encoded.info()


x_train_encoded.describe().T


# We apply a scaler to the data

scaler = StandardScaler().set_output(transform="pandas")
scaler.fit(x_train_encoded)

x_train_scaled = scaler.transform(x_train_encoded)
x_val_scaled = scaler.transform(x_val_encoded)


x_train_scaled.describe().T


x_val_scaled.describe().T


x_train_scaled.corr()


lpb_scores = mutual_info_classif(x_train_scaled, y_train)
lpb_scores = pd.Series(lpb_scores, name="Loan Payback MI Scores", index=x_train_scaled.columns)
lpb_scores = lpb_scores.sort_values(ascending=False)
lpb_scores


# Function to calculate the roc auc score

def auc_evaluator(model, x, y, name):

    probs = model.predict_proba(x)
    probs = probs[:, 1]
    model_auc = roc_auc_score(y, probs)
    print(f"\n {name}: ROC AUC = %.5f" % (model_auc))


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


# We calculate the roc auc score

auc_evaluator(lrc, x_val_scaled, y_val, "LogisticRegression")


# We create the model instance

lgbmc = LGBMClassifier(is_unbalance=True, objective="binary", verbose=-1)

# Train the model with the data

lgbmc.fit(x_train_scaled, y_train)


# We calculate the roc auc score

auc_evaluator(lgbmc, x_val_scaled, y_val, "LGBMClassifier")


# We create the model instance

cbc = CatBoostClassifier(silent=True)

# Train the model with the data

cbc.fit(x_train_scaled, y_train)


# We calculate the roc auc score

auc_evaluator(cbc, x_val_scaled, y_val, "CatBoostClassifier")


# We create the model instance

xgbc = XGBClassifier()

# Train the model with the data

xgbc.fit(x_train_scaled, y_train)


# We calculate the roc auc score

auc_evaluator(xgbc, x_val_scaled, y_val, "XGBClassifier")


# We apply the ROC Curve function

models_to_test = {
    "LogisticRegression" : lrc,
    "LGBMClassifier" : lgbmc,
    "CatBoostClassifier" : cbc,
    "XGBClassifier" : xgbc,
}

roc_curve_evaluator(y_val, x_val_scaled, models_to_test)


# We define the final model

final_model_1 = lrc
final_model_2 = lgbmc
final_model_3 = cbc
final_model_4 = xgbc


# Create Base Learners

base_learners = [
    ("LogisticRegression", final_model_1),
    ("LGBMClassifier", final_model_2),
    ("CatBoostClassifier", final_model_3),
    ("XGBClassifier", final_model_4),
]

# Initialize Stacking Classifier with the Meta Learner

sclf = StackingClassifier(estimators=base_learners)

# Train the model with the data

sclf.fit(x_train_scaled, y_train)


auc_evaluator(sclf, x_val_scaled, y_val, "StackingRegressor")


# We load the test data

df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


# We check the shape and that no duplicate data is found

print(f"Length: {len(df_test.duplicated())}")
print(f"Duplicates: {df_test.duplicated().sum()}")
print(f"Shape: {df_test.shape}")
print(f"Nulls:\n\n{df_test.isnull().sum()}\n")


df_test.info()


# We start by removing the variables that we will not use

df_test_new = df_test.drop(columns=["id"])


for col in numerical_cols:
    Q1 = df_test_new[col].quantile(0.25)
    Q3 = df_test_new[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_test_new[col] = df_test_new[col].clip(lower=lower_bound, upper=upper_bound)


# We apply the transformations

x_encoded_test = t_encoder.transform(df_test_new)

x_encoded_test.info()


# We apply a scaler to the data

x_test_scaled = scaler.transform(x_encoded_test)

x_test_scaled.describe().T


# We apply the trained model

lpb_predictions = sclf.predict_proba(x_test_scaled)

# We review the result

print("Total predictions: ", len(lpb_predictions), "\n")

# We create the dataframe

lpb_submission = pd.DataFrame({
    "id" : df_test["id"], 
    "loan_paid_back" : lpb_predictions[:, 1]
})

lpb_submission.head()


# We load the submission sample data

lpb_sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


# We compare the results with the sample

print(
    f"Shape Sample Submission: {lpb_sample.shape}",
    f"\nShape rar Submission: {lpb_submission.shape}"
)
print("\n", lpb_sample.head())


# We convert the dataframe to a csv file

lpb_submission.to_csv("submission.csv", index=False)

