# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score, make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
import missingno as msno
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


application_train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
application_test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")
bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bureau_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")
previous_application = pd.read_csv("/kaggle/input/home-credit-default-risk/previous_application.csv")
posh_cash = pd.read_csv("/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv")
installments_payments = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")
credit_card_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")


files_name = ["application_train", "application_test", "bureau", "bureau_balance", "previous_application", "posh_cash", "installments_payments",
             "credit_card_balance"]

files = [application_train, application_test, bureau, bureau_balance, previous_application, posh_cash, installments_payments, credit_card_balance]

def check_df(dataframe):
    print("#################### Shape ######################")
    print(dataframe.shape)
    print("#################### Info ######################")
    print(dataframe.info())

for i in range(len(files)):
    print(f"{files_name[i]}: \n")
    check_df(files[i])
    print("\n\n ************************** \n\n")


# Selecting variable types:
def grab_col_names(dataframe, cat_th = 10, car_th=20):
    dataframe.columns = [col.upper() for col in dataframe.columns]
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object", "category", "bool"]]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                  str(dataframe[col].dtypes) in ["int64", "float64", "int32"]]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                  str(dataframe[col].dtypes) in ["object", "category", "bool"]]

    cat_cols = cat_cols + num_but_cat
    
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64", "int32"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols_count: {len(cat_cols)}")
    print(f"num_cols_count: {len(num_cols)}")
    print(f"cat_but_car_count: {len(cat_but_car)}")
    print(f"num_but_cat_count: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


# Let's combine the training and test data:
app_df = application_train._append(application_test).reset_index().drop("index", axis=1)
app_df.head()


app_df.tail()


print(f"Train size: {app_df[app_df['TARGET'].notnull()].shape}")
print(f"Test size: {app_df[~app_df['TARGET'].notnull()].shape}")


def feature_extraction(df):
    # 1. Rates
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    
    # 2. Age and durations
    df["AGE"] = -df["DAYS_BIRTH"] / 365
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365
    df["YEARS_REGISTRATION"] = -df["DAYS_REGISTRATION"] / 365
    df["YEARS_ID_PUBLISH"] = -df["DAYS_ID_PUBLISH"] / 365
    df["EMPLOYED_TO_AGE_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE"]
    
    # 3. Social rates
    df["SOCIAL_DEF_RATIO_30"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
    df["SOCIAL_DEF_RATIO_60"] = df["DEF_60_CNT_SOCIAL_CIRCLE"] / (df["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
    bureau_cols = [col for col in df.columns if "AMT_REQ_CREDIT_BUREAU" in col]
    df["TOTAL_CREDIT_BUREAU_REQUESTS"] = df[bureau_cols].sum(axis=1)
    
    # 4. Regional
    df["IS_SMALL_POPULATION_REGION"] = (df["REGION_POPULATION_RELATIVE"] < 0.05).astype(int)
    df["REGION_RATING_DIFF"] = df["REGION_RATING_CLIENT"] - df["REGION_RATING_CLIENT_W_CITY"]
    
    # 5. Number of documents
    doc_flags = [col for col in df.columns if col.startswith("FLAG_DOCUMENT")]
    df["TOTAL_DOC_FLAGS"] = df[doc_flags].sum(axis=1)
    return df

app_df = feature_extraction(app_df)


cat_cols, num_cols, cat_but_car = grab_col_names(app_df)

cat_cols = [col for col in cat_cols if "TARGET" not in col]


def prep_bureau(bureau_balance, bureau, df):
    bb_agg = (
    bureau_balance
    .groupby("SK_ID_BUREAU")
    .agg(
        months_min=("MONTHS_BALANCE", "min"),
        months_max=("MONTHS_BALANCE", "max"),
        status_last=("STATUS", "last")
    ).reset_index())

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    cat_cols, num_cols, cat_but_car = grab_col_names(bureau)

    b_aggriagtions={"CREDIT_ACTIVE": lambda s: (s == "Active").sum()}

    for col in num_cols:
        b_aggriagtions[col] = ["mean", "sum", "max"]

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(b_aggriagtions).reset_index()
    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg.rename(columns={'SK_ID_CURR_': 'SK_ID_CURR'}, inplace=True)
    df = pd.merge(df, bureau_agg, on='SK_ID_CURR', how='left')
    return df

app_df = prep_bureau(bureau_balance, bureau, app_df)
app_df.head()


def prep_pa(previous_application, df):
    pa_agg = previous_application.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "count",
        "NAME_CONTRACT_STATUS": lambda s: (s == "Approved").sum(),
        "AMT_APPLICATION": "mean",
        "AMT_CREDIT": "mean",
        "AMT_GOODS_PRICE": "mean",
        "AMT_ANNUITY": "mean"
    }).reset_index()
    
    # Joining to application_df
    
    df = pd.merge(df, pa_agg, on='SK_ID_CURR', how='left')
    return df

app_df = prep_pa(previous_application ,app_df)
app_df.head()


def prep_ip(installments_payments, df):
    # Has the installment amount been paid in full?
    installments_payments["PAY_DIFF"] = installments_payments["AMT_PAYMENT"] - installments_payments["AMT_INSTALMENT"]
    
    # Was there a delay in payment?
    installments_payments["PAY_DELAY_DAYS"] = installments_payments["DAYS_ENTRY_PAYMENT"] - installments_payments["DAYS_INSTALMENT"]
    
    # Let's remove the variables used to create a new variable from the data set
    installments_payments.drop(["AMT_PAYMENT", "AMT_INSTALMENT", "DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"], axis=1, inplace=True)
    
    cat_cols, num_cols, cat_but_car = grab_col_names(installments_payments)
    
    ip_aggriagtions={"SK_ID_PREV": "count"}
    
    for col in num_cols:
        ip_aggriagtions[col] = ["mean", "max", "min"]
    
    ip_agg = installments_payments.groupby('SK_ID_CURR').agg(ip_aggriagtions).reset_index()
    
    ip_agg.columns = ['_'.join(col).strip() for col in ip_agg.columns.values]
    ip_agg.rename(columns={'SK_ID_CURR_': 'SK_ID_CURR'}, inplace=True)
    df = pd.merge(df, ip_agg, on='SK_ID_CURR', how='left')
    return df

app_df = prep_ip(installments_payments ,app_df)
app_df.head()


def prep_pc(posh_cash, df):
    # Remaining Debt Ratio:
    posh_cash["FEATURE_DEBT_RATIO"] = posh_cash["CNT_INSTALMENT_FUTURE"]  / (posh_cash["CNT_INSTALMENT"] + posh_cash["CNT_INSTALMENT_FUTURE"])
    posh_cash["MONTHS_BALANCE"] = posh_cash["MONTHS_BALANCE"].abs()
    posh_cash["MONTHLY_PAYMENT"] = posh_cash["CNT_INSTALMENT"] / posh_cash["MONTHS_BALANCE"]
    
    p_agg = posh_cash.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "nunique",
        "NAME_CONTRACT_STATUS": lambda s: (s == "Active").sum(),
        "FEATURE_DEBT_RATIO": "mean",
        "MONTHLY_PAYMENT": "mean",
        "CNT_INSTALMENT": "mean",
        "CNT_INSTALMENT_FUTURE": "mean"
    }).reset_index()
    
    # Joining to application_df
    
    df = pd.merge(df, p_agg, on='SK_ID_CURR', how='left')
    return df

app_df = prep_pc(posh_cash ,app_df)
app_df.head()


def prep_cc(credit_card_balance, df):
    cc_agg = credit_card_balance.groupby("SK_ID_CURR").agg({
        "AMT_BALANCE": "mean",  # Mean balance on credit card
        "SK_DPD": "mean"  # Mean delinquency period on credit card
    }).reset_index()
    
    df = pd.merge(df, cc_agg, on='SK_ID_CURR', how='left')
    df.columns = [col.upper() for col in df.columns]
    df = df.drop_duplicates(keep="first").reset_index().drop("index",axis=1)
    return df

app_df = prep_cc(credit_card_balance ,app_df)
app_df.head()


app_df = app_df.reset_index().drop("index", axis=1)
train_size = len(app_df[app_df["TARGET"].notnull()])
train_df = app_df[:train_size]
test_df = app_df[train_size:]


train_df.head()


test_df.head()


t=train_df["TARGET"].value_counts()
labels=t.index
colors=["green","red"]
explode=[0,0]
sizes=t.values

#visual
plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%")
plt.title("Pie Chart of TARGET")
plt.show()


# Let's select variable types
cat_cols, num_cols, cat_but_car = grab_col_names(train_df)
cat_cols = [col for col in cat_cols if "TARGET" not in col]


# Distribution of categorical variables within the data set:
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#####################################")
    if plot:
        sns.countplot(x = dataframe[col_name], data=dataframe)
        plt.xticks(rotation=90)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(train_df, col, plot=False)


# Distribution of numerical variables:
def num_summary(dataframe, numerical_col, plot = False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################")
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.xticks(rotation=90)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(train_df, col, plot=False)


# Examining the target variable with categorical variables
def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col, observed=True)[target].mean()}))
    if plot:
        g=sns.catplot(x=categorical_col, y="TARGET", data=dataframe, kind="bar")
        g.set_ylabels("Repaid Probability")
        g.set_xticklabels(rotation=90)
        plt.show(block=True)

for col in cat_cols:
    target_summary_with_cat(train_df, "TARGET", col, plot=False)


to_drop = []
for col in cat_cols:
    df = pd.DataFrame({
        "TARGET_MEAN": train_df.groupby(col, observed=True)["TARGET"].mean()
    }).sort_values(by="TARGET_MEAN", ascending=False)
    
    diff = df.iloc[0, 0] - df.iloc[-1, 0]
    if diff <= 0.010:
        to_drop.append(col)

to_drop


train_df.drop(to_drop, axis=1, inplace=True)
test_df.drop(to_drop, axis=1, inplace=True)

# Let's select variable types
cat_cols, num_cols, cat_but_car = grab_col_names(train_df)
cat_cols = [col for col in cat_cols if "TARGET" not in col]


def update_time_vars(dataframe):
    # We previously created the age variable from the day birth variable.
    dataframe.drop(columns=['DAYS_BIRTH'], inplace=True)
    #DAYS_EMPLOYED
    dataframe['DAYS_EMPLOYED'] = dataframe['DAYS_EMPLOYED'].abs()
    dataframe['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    dataframe['DAYS_EMPLOYED'].fillna(dataframe['DAYS_EMPLOYED'].median(), inplace=True)
    
    days_cols = dataframe.columns[dataframe.columns.str.contains("DAYS", case=False)].tolist()
    for col in days_cols:
        dataframe[col] = dataframe[col].apply(lambda x: abs(x) if x < 0 else x)
    return dataframe

train_df = update_time_vars(train_df)
test_df = update_time_vars(test_df)


# Let's select variable types
cat_cols, num_cols, cat_but_car = grab_col_names(train_df)
cat_cols = [col for col in cat_cols if "TARGET" not in col]


def find_columns_with_negative_values(dataframe):

    numerical_cols = grab_col_names(dataframe)[1]
    columns_with_negative_values = []

    for col in numerical_cols:
        if (dataframe[col] < 0).any():  # Check if any value in the column is negative
            columns_with_negative_values.append(col)  # Add to the list if negative values are found

    return columns_with_negative_values

columns_with_negative_values = find_columns_with_negative_values(train_df)

print(columns_with_negative_values)


train_df[columns_with_negative_values].describe().T


def handle_negative_values(dataframe):
    
    columns_with_negative_values_to_handle=['AMT_CREDIT_SUM_DEBT_MEAN','AMT_CREDIT_SUM_DEBT_SUM','AMT_CREDIT_SUM_DEBT_MAX',
                                            'AMT_CREDIT_SUM_LIMIT_MEAN', 'AMT_CREDIT_SUM_LIMIT_SUM', 'AMT_CREDIT_SUM_LIMIT_MAX',
                                            'AMT_BALANCE', 'YEARS_EMPLOYED', 'EMPLOYED_TO_AGE_RATIO']

    for col in columns_with_negative_values_to_handle:
        dataframe[col] = dataframe[col].apply(lambda x: x if x >= 0 else np.nan)

    return dataframe

train_df = handle_negative_values(train_df)
test_df = handle_negative_values(test_df)


# Examining the target variable with numerical variables
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target, observed=True).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(train_df, "TARGET", col)


# 1: HO: M1 = M2 H1:M1 != M2
# 2: Assumption check:
# Normality Assumption:
# H0: The normal distribution assumption is met.
# H1: The normal distribution assumption is not met.

def normality(dataframe, num_cols, target="TARGET"):
    df = pd.DataFrame()
    test_stat_1 = []
    pvalue_1 = []
    test_stat_0 = []
    pvalue_0 = []
    
    for col in num_cols:
        test_stat, pvalue = shapiro(dataframe.loc[dataframe[target] == 1, col].dropna())
        test_stat_1.append(test_stat)
        pvalue_1.append(pvalue)
        test_stat, pvalue = shapiro(dataframe.loc[dataframe[target] == 0, col].dropna())
        test_stat_0.append(test_stat)
        pvalue_0.append(pvalue)

    df["Variable"] = num_cols
    df["test_stat target=1"] = test_stat_1
    df["pvalue target=1"] = pvalue_1
    df["test_stat target=0"] = test_stat_0
    df["pvalue target=0"] = pvalue_0
    return df

df_normality = normality(train_df, num_cols, target="TARGET")
df_normality


mann_whitney_u = []

for col in num_cols:
    group1 = train_df.loc[train_df["TARGET"] == 1, col].dropna()
    group0 = train_df.loc[train_df["TARGET"] == 0, col].dropna()
    test_stat, pvalue = mannwhitneyu(group1, group0)
    if pvalue > 0.05:
        mann_whitney_u.append({"variable": col, "p_value": pvalue})

mann_whitney_u_df = pd.DataFrame(mann_whitney_u)
print(mann_whitney_u_df)


to_drop = mann_whitney_u_df["variable"].tolist()

to_drop_existing = [col for col in to_drop if col in train_df.columns]

train_df.drop(to_drop_existing, axis=1, inplace=True)
test_df.drop(to_drop_existing, axis=1, inplace=True)


# Let's select variable types
cat_cols, num_cols, cat_but_car = grab_col_names(train_df)
cat_cols = [col for col in cat_cols if "TARGET" not in col]


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

drop_list = high_correlated_cols(train_df[num_cols])


train_df.drop(drop_list, axis=1, inplace=True)
test_df.drop(drop_list, axis=1, inplace=True)

# Let's select variable types
cat_cols, num_cols, cat_but_car = grab_col_names(train_df)
cat_cols = [col for col in cat_cols if "TARGET" not in col]


# Let's calculate the number and proportion of missing observations:
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys = ["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

    return missing_df

missing_df = missing_values_table(train_df)


drop_cols = missing_df[missing_df["ratio"] > 45].index

train_df.drop(drop_cols, axis=1, inplace=True)
test_df.drop(drop_cols, axis=1, inplace=True)

# Let's select variable types
cat_cols, num_cols, cat_but_car = grab_col_names(train_df)
cat_cols = [col for col in cat_cols if "TARGET" not in col]


def missing_analysis(df, corr_threshold=1.0):
    # Missing data rate
    missing_percent = df.isnull().mean()*100
    missing_percent = missing_percent.sort_values(ascending = False)
    missing_df = pd.DataFrame({
        "Missing %": missing_percent
    })

    #Correlation of missing data
    corr_matrix = df.isnull().corr()

    high_corr_pairs = []
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range (i+1, len(cols)):
            if corr_matrix.iloc[i, j] >= corr_threshold:
                high_corr_pairs.append({
                    'Var1': cols[i],
                    'Var2': cols[j],
                    'Corr': corr_matrix.iloc[i, j],
                    'Var1_Missing_%': missing_percent.get(cols[i], 0),
                    'Var2_Missing_%': missing_percent.get(cols[j], 0)
                })
                
    high_corr_df = pd.DataFrame(high_corr_pairs)

    return missing_df, high_corr_df


high_corr_df = missing_analysis(train_df)[1]
high_corr_df.head(10)


nan_cols = pd.unique(high_corr_df.loc[:, "Var1"])

nan_cat_cols = [col for col in nan_cols if col in cat_cols]
nan_num_cols = [col for col in nan_cols if col in num_cols]


fig,axes=plt.subplots(round(len(nan_cat_cols) / 3), 3, figsize=(18,15))

for i,ax in enumerate(fig.axes):
    if i<(len(nan_cat_cols)):
        sns.countplot(data=train_df, x=nan_cat_cols[i], order=train_df[nan_cat_cols[i]].value_counts().sort_index().index, ax=ax)
fig.tight_layout()


def fill_na_cat_mod(dataframe, cols):
    for col in cols:
        dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)

    return dataframe

train_df = fill_na_cat_mod(train_df, nan_cat_cols)
test_df = fill_na_cat_mod(test_df, nan_cat_cols)


def cat_summary_with_num(dataframe, cat, numerical_col):
    print(dataframe.groupby(cat, observed=True).agg({numerical_col: "mean"}), end="\n\n\n")

for col in nan_num_cols:
    cat_summary_with_num(train_df, "NAME_CONTRACT_TYPE", col)


# Let's assign the medians in the NAME_CONTRACT_TYPE breakdown

def fill_na_median(dataframe, group_col, num_cols):
    for col in num_cols:
        dataframe[col] = dataframe[col].fillna(
            dataframe.groupby(group_col)[col].transform("median")
        )
    return dataframe

train_df = fill_na_median(train_df, "NAME_CONTRACT_TYPE", nan_num_cols)
test_df = fill_na_median(test_df, "NAME_CONTRACT_TYPE", nan_num_cols)


missing_values = missing_values_table(train_df)

missing_cols = missing_values.index.tolist()

missing_cat_cols = [col for col in missing_cols if col in cat_cols]
missing_num_cols = [col for col in missing_cols if col in num_cols]


# Distribution of categorical variables containing null values
fig,axes=plt.subplots(round(len(missing_cat_cols) / 3), 2, figsize=(18,15))

for i,ax in enumerate(fig.axes):
    if i<(len(missing_cat_cols)):
        sns.countplot(data=train_df, x=missing_cat_cols[i], order=train_df[missing_cat_cols[i]].value_counts().sort_index().index, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.tight_layout()


def fill_na_cats_random(dataframe, cols):
    for col in cols:
        non_nulss = dataframe[col].dropna().values
        dataframe[col] = dataframe[col].apply(
            lambda x : np.random.choice(non_nulss) if pd.isna(x) else x
        )
    return dataframe

train_df = fill_na_cats_random(train_df, ["OCCUPATION_TYPE"])
test_df = fill_na_cats_random(test_df, ["OCCUPATION_TYPE"])

train_df = fill_na_cat_mod(train_df, ["NAME_TYPE_SUITE"])
test_df = fill_na_cat_mod(test_df, ["NAME_TYPE_SUITE"])


for col in missing_num_cols:
    cat_summary_with_num(train_df, "NAME_CONTRACT_TYPE", col)


train_df = fill_na_median(train_df, "NAME_CONTRACT_TYPE", missing_num_cols)
test_df = fill_na_median(test_df, "NAME_CONTRACT_TYPE", missing_num_cols)


print(train_df.isnull().sum())
print("------------------")
print(test_df.isnull().sum())


train_df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
test_df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)

train_df = fill_na_cat_mod(train_df, ['CODE_GENDER'])
test_df = fill_na_cat_mod(test_df, ['CODE_GENDER'])


# Outlier Detection
def outlier_threshold(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interqunatile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interqunatile_range
    low_limit = quartile1 - 1.5 * interqunatile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


cat_cols, num_cols, cat_but_car = grab_col_names(train_df)
cat_cols = [col for col in cat_cols if "TARGET" not in col]


for col in num_cols:
    print(col, check_outlier(train_df, col))


for col in num_cols:
    replace_with_threshold(train_df, col)


for col in num_cols:
    replace_with_threshold(test_df, col)


df = train_df._append(test_df)
df.shape


cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if "TARGET" not in col]


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts()/len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)


def rare_encoder(dataframe, rare_prec, cat_cols):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if col in cat_cols
                    and (temp_df[col].value_counts() / len(temp_df) < rare_prec).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_prec].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01, cat_cols)


rare_analyser(new_df, "TARGET", cat_cols)


new_df_copy = new_df.copy()


def encode_all(df, target_col=None):
    df = df.copy()

    # Target sütununu ayır (encoding yapmamak için)
    if target_col and target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        target = None

    # 1️⃣ Binary encoding (nunique == 2)
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # 2️⃣ Low-cardinality (3 ≤ nunique ≤ 10) → OHE
    ohe_cols = [col for col in df.columns if 2 < df[col].nunique() <= 10]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)

    # 3️⃣ High-cardinality (nunique > 10) → Label Encoding
    high_card_cols = [col for col in df.columns if df[col].nunique() > 10]
    for col in high_card_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Target sütununu geri ekle
    if target is not None:
        df[target_col] = target

    # 4️⃣ Son kontrol
    still_object = df.select_dtypes(include=["object"]).columns
    if len(still_object) > 0:
        print("⚠️ There are still object type columns:", list(still_object))
    else:
        print("✅ All columns became numeric.")

    return high_card_cols, df

new_df = encode_all(new_df, "TARGET")[1]
high_card_cols = encode_all(new_df, "TARGET")[0]


cat_cols, num_cols, cat_but_car = grab_col_names(new_df)
cat_cols = [col for col in cat_cols if "TARGET" not in col]


# Train_test_split
train_size = len(new_df[new_df["TARGET"].notnull()])
train_df = new_df[:train_size]
test_df = new_df[train_size:]


# Scaling
sc = StandardScaler()
train_df[num_cols+high_card_cols] = sc.fit_transform(train_df[num_cols+high_card_cols])
test_df[num_cols+high_card_cols] = sc.transform(test_df[num_cols+high_card_cols])

# Split
X = train_df.drop("TARGET", axis=1)
y = train_df["TARGET"]
X_test = test_df.drop("TARGET", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)


from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k="all")
selector.fit_transform(X_train, y_train)

df=pd.DataFrame()
df["features_name"]=X.columns.to_list()
df["f_selector_scores"]=selector.scores_
df=df.sort_values(by="f_selector_scores",ascending=False)


df.head(40)


select_columns = df.loc[df["f_selector_scores"]>300, "features_name"].tolist()

X_train = X_train[select_columns]
X_val = X_val[select_columns]
X_test = X_test[select_columns]

X = X[select_columns]


# Make column names strings and strip special characters
X_train.columns = [str(col) for col in X_train.columns]   # sayıları stringe çevir
X_train.columns = X_train.columns.str.replace(r'[\[\]<]', '', regex=True)

# It would be good to do the same for X_test.
X_test.columns = [str(col) for col in X_test.columns]
X_test.columns = X_test.columns.str.replace(r'[\[\]<]', '', regex=True)

# It would be good to do the same for X_val.
X_val.columns = [str(col) for col in X_val.columns]
X_val.columns = X_val.columns.str.replace(r'[\[\]<]', '', regex=True)

# It would be good to do the same for X.
X.columns = [str(col) for col in X.columns]
X.columns = X.columns.str.replace(r'[\[\]<]', '', regex=True)



def base_models(X, y, scoring = "roc_auc"):
    print("Base Models....")
    classifiers = [("XGBM", XGBClassifier(random_state=42, class_weight="balanced")),
                   ("LightGBM", LGBMClassifier(random_state=42 ,verbosity=-1, class_weight="balanced")),
                   ]
    for name, classifier in classifiers:
        cv_results = cross_val_score(classifier, X, y, cv = 3, scoring=scoring)
        print(f"{scoring}: {round(cv_results.mean(), 4)} ({name})")

base_models(X_train, y_train)


# Automated Hyperparameter Optimization
lightgbm_params = {"learning_rate": [0.01, 0.1],
                  "n_estimators": [300, 500, 800],
                  "colsample_bytree": [0.7, 1, 1.2],
                  "class_weight": ["balanced"]}

classifiers = [("LightGBM", LGBMClassifier(verbosity=-1), lightgbm_params)]

def hyperparameter_optimization(X, y, cv = 3, scoring = "roc_auc"):
    print("Hyperparameter Optimization.....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########### {name} ###########")
        cv_results = cross_val_score(classifier, X, y, cv = cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results.mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv = cv, n_jobs=-1, verbose=False).fit(X,y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_val_score(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results.mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X_train,y_train)


final_model = best_models["LightGBM"].fit(X_train, y_train)

y_pred = final_model.predict(X_train)
y_prob = final_model.predict_proba(X_train)[:,1]
print(classification_report(y_train, y_pred))
print(roc_auc_score(y_train, y_prob))


y_pred = final_model.predict(X_val)
y_prob = final_model.predict_proba(X_val)[:,1]
print(classification_report(y_val, y_pred))
print(roc_auc_score(y_val, y_prob))


# Compute the confusion matrix
confusion_mtx=confusion_matrix(y_val,y_pred)
# plot confusion matrix
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt=".1f", ax=ax)
plt.xlabel("Predict Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()




