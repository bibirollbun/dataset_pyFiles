# !pip install "modin[all]"

import os
import numpy as np
import pandas as pd
# import modin.pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")


dir_path = '/kaggle/input/home-credit-default-risk'

paths_dict = {
    "app_train": f'{dir_path}/application_train.csv',
    "bureau": f'{dir_path}/bureau.csv',
    # "bureau_balance": f'{dir_path}/bureau_balance.csv', ## Will not be used as it is unusful data.
    "pos_cash": f'{dir_path}/POS_CASH_balance.csv',
    "credit_card": f'{dir_path}/credit_card_balance.csv',
    "previous_app": f'{dir_path}/previous_application.csv',
    "insta_payments": f'{dir_path}/installments_payments.csv'
}
app_test_path = f'{dir_path}/application_test.csv'


# Store all in one Dict
dfs = {name: pd.read_csv(path) for name, path in paths_dict.items()}

app_test = pd.read_csv(app_test_path)


for name, df in dfs.items():
    print(name, f"Shape: {df.shape}")
print('app_test', f"Shape: {app_test.shape}")


# Get counts and calculate percentages
counts = dfs["app_train"]['TARGET'].value_counts()
percentages = counts / counts.sum() * 100

# Plot
ax = counts.plot(kind='bar')
plt.title('Distribution of TARGET')
plt.xticks([0, 1], ['(0)', '(1)'], rotation=0)
plt.ylabel('Count')
plt.xlabel('TARGET')

# Annotate bars with percentage
for i, (count, percent) in enumerate(zip(counts, percentages)):
    plt.text(i, count + 5000, f'{percent:.2f}%', ha='center', fontsize=12)

plt.show()


## Drop DPD-related columns as DPD csv is not provided
dfs["credit_card"] = dfs["credit_card"].drop(columns=["SK_DPD", "SK_DPD_DEF"])
dfs["pos_cash"] = dfs["pos_cash"].drop(columns=["SK_DPD", "SK_DPD_DEF"])



# Function to calculate missing values
def missing_values_table(df, threshold=0.5, name="The DataFrame"):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1).rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    missing_values_threshold = mis_val_table[mis_val_table["% of Total Values"] > threshold]

    # Print some summary information
    text = "\n".join([
        f"`{name}` has {str(df.shape[1])} columns.",
        f"There are {str(mis_val_table.shape[0])} columns that have missing values.",
        f"{str(missing_values_threshold.shape[0])} of them are Above the threshold (>{threshold})",
        "-"*50
    ])
    print (text)
    
    
    # Return the dataframe with missing information and dropped df
    return mis_val_table, df.drop(columns=missing_values_threshold.index.to_list())


dropped_dfs = {name: missing_values_table(df, threshold=50.0, name=name)[1] for name, df in dfs.items()}
del dfs


## app_test columns must be same as app_train columns
app_test = app_test[dropped_dfs['app_train'].columns[dropped_dfs['app_train'].columns != 'TARGET']]
app_test.shape


from sklearn.impute import SimpleImputer

for name, df in dropped_dfs.items():

    # Separate columns
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = [col for col in df.columns if col not in cat_cols]
    if name == 'app_train':
        num_cols = [col for col in num_cols if col != 'TARGET']

    # === Impute Categorical (mode) ===
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        if name == 'app_train':
            app_test[cat_cols] = cat_imputer.transform(app_test[cat_cols])

    # === Impute Numerical (median) ===
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        if name == 'app_train':
            num_cols = [col for col in num_cols if col != 'TARGET']
            app_test[num_cols] = num_imputer.transform(app_test[num_cols])


## First we need to encode categorical features

from sklearn.preprocessing import LabelEncoder
import copy

encoded_dfs = copy.deepcopy(dropped_dfs)

for name, df in encoded_dfs.items():

    # Separate columns
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = [col for col in df.columns if col not in cat_cols]

    # Label encode
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


## Drop columns with constant Features or quasi constant Features

from sklearn.feature_selection import VarianceThreshold

columns_to_keep = {}
for name, df in encoded_dfs.items():

    ## Check quasi Constant Features
    sel_quasi = VarianceThreshold(threshold=0.01)
    sel_quasi.fit(df)

    columns_to_keep[name] = df.columns[sel_quasi.get_support()].to_list()

    print(f"{name}:",
          str(df.columns[~sel_quasi.get_support()].size),
          f'constant features out of {str(df.columns.size)} features')


## keep not constant features and drop others
for name, cols in columns_to_keep.items():
    encoded_dfs[name] = encoded_dfs[name][cols]

for name, df in encoded_dfs.items():
    print(name, f"Shape: {df.shape}")



## drop SK_ID_BUREAU as it relates to bureau_balance which is dropped
encoded_dfs["bureau"] = encoded_dfs["bureau"].drop(columns=["SK_ID_BUREAU"])



## Apply droped columns to dropped_dfs
for name, df in encoded_dfs.items():
    dropped_dfs[name] = dropped_dfs[name][df.columns]

for name, df in dropped_dfs.items():
    print(name, f"Shape: {df.shape}")

## app_test columns must be same as app_train columns
app_test = app_test[dropped_dfs['app_train'].columns[dropped_dfs['app_train'].columns != 'TARGET']]
print("app_test", f"Shape: {app_test.shape}")


def my_scatterplot(df, col, ax):
    sns.scatterplot(data=df, x='x1', y='x2', hue=col, ax=ax, palette='coolwarm')
    ax.set_title(f'Scatter Plot of {col}')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_features(df, plot_funcs, width_ratios, height_ratios, 
                  n_col=1, primary_cols=0, title=None):
    def plot_feature(cols):
        cols_len = len(cols) - primary_cols
        curr_width_ratios = width_ratios[:cols_len * len(plot_funcs)]
        n_charts = len(plot_funcs) * cols_len

        # Create a figure with specified size and gridspec layout
        fig = plt.figure(figsize=(sum(curr_width_ratios), max(height_ratios)))
        gs = fig.add_gridspec(1, n_charts, 
                              width_ratios=curr_width_ratios, height_ratios=height_ratios)
        axes = [0] * n_charts
        for i in range(cols_len):
            for j in range(len(plot_funcs)):
                k = i * len(plot_funcs) + j
                axes[k] = fig.add_subplot(gs[0, k])
                # Call the specified plotting function with df, col, and axis ax
                plot_funcs[j](df, cols[i + primary_cols], axes[k])
                if title:
                    fig.suptitle(title)

        plt.tight_layout()
        plt.show()

    for i in range(primary_cols, len(df.columns), n_col):
        plot_feature(list(df.columns[:primary_cols])+list(df.columns[i:i+n_col]))


def plot_datetime_feature(df, feature, palette='deep'):
    # Count values of the specified feature
    feature_counts = df[feature].value_counts().sort_index()
    plt.figure(figsize=(12, 4))
    sns.barplot(x=feature_counts.index, y=feature_counts.values, palette=palette)
    # Set title and labels
    plt.title(f'Count Plot for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    # Rotate x-axis labels for readability
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def my_histplot(df, col, ax):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Histogram Plot of {col}')
def my_kdeplot(df, col, ax):
    sns.kdeplot(df[col], ax=ax, fill=True)
    ax.set_title(f'KDE Plot of {col}')
def my_distplot(df, col, ax):
    sns.distplot(df[col], ax=ax)
    ax.set_title(f'Distribution Plot of {col}')
def my_boxplot(df, col, ax):
    sns.boxplot(y=df[col], ax=ax)
def my_violinplot(df, col, ax):
    sns.violinplot(y=df[col], ax=ax)


def my_pie_chart(df, col, ax):
    labels = df[col].value_counts()
    ax.pie(labels, labels=labels.index, autopct='%1.1f%%')
    ax.set_title(f'Pie Chart of {col}')
def my_barplot(df, col, ax):
    value_counts = df[col].value_counts().sort_values(ascending=False)
    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, 
                orient='h', order=value_counts.index)
    ax.set_title(f'Bar Plot of {col}')
    ax.set_xlabel('Count')
    ax.set_ylabel(col)


def plot_numerical_features(df, plot_funcs=[my_boxplot, my_violinplot, my_distplot], 
                            width_ratios=[2, 2, 8], height_ratios=[4], 
                            n_col=1, primary_cols=0, title=None):
    plot_features(df, plot_funcs, width_ratios * n_col, height_ratios, n_col, primary_cols, title)

def plot_categorical_features(df, plot_funcs=[my_pie_chart, my_barplot], 
                              width_ratios=[4, 8], height_ratios=[4], 
                              n_col=1, primary_cols=0, title=None):
    plot_features(df, plot_funcs, width_ratios * n_col, height_ratios, n_col, primary_cols, title)


flags = ['FLAG_EMP_PHONE',
         'FLAG_WORK_PHONE',
         'FLAG_PHONE',
         'FLAG_EMAIL',
         'REG_REGION_NOT_LIVE_REGION',
         'REG_REGION_NOT_WORK_REGION',
         'LIVE_REGION_NOT_WORK_REGION',
         'REG_CITY_NOT_LIVE_CITY',
         'REG_CITY_NOT_WORK_CITY',
         'LIVE_CITY_NOT_WORK_CITY']

eda_dfs = copy.deepcopy(dropped_dfs)

# Convert 1 -> 'Yes', 0 -> 'No'
for col in flags:
    eda_dfs["app_train"][col] = eda_dfs["app_train"][col].map({1: 'Yes', 0: 'No'}).astype('object')



df_name = "app_train"
cat_cols = eda_dfs[df_name].select_dtypes(include='object').columns
num_cols = [col for col in eda_dfs[df_name].columns if col not in cat_cols]

plot_numerical_features(eda_dfs[df_name][num_cols])
plot_categorical_features(eda_dfs[df_name][cat_cols])


df_name = "bureau"
cat_cols = eda_dfs[df_name].select_dtypes(include='object').columns
num_cols = [col for col in eda_dfs[df_name].columns if col not in cat_cols]

plot_numerical_features(eda_dfs[df_name][num_cols])
plot_categorical_features(eda_dfs[df_name][cat_cols])


df_name = "pos_cash"
cat_cols = eda_dfs[df_name].select_dtypes(include='object').columns
num_cols = [col for col in eda_dfs[df_name].columns if col not in cat_cols]

plot_numerical_features(eda_dfs[df_name][num_cols])
plot_categorical_features(eda_dfs[df_name][cat_cols])


df_name = "credit_card"
cat_cols = eda_dfs[df_name].select_dtypes(include='object').columns
num_cols = [col for col in eda_dfs[df_name].columns if col not in cat_cols]

plot_numerical_features(eda_dfs[df_name][num_cols])
plot_categorical_features(eda_dfs[df_name][cat_cols])


df_name = "previous_app"
cat_cols = eda_dfs[df_name].select_dtypes(include='object').columns
num_cols = [col for col in eda_dfs[df_name].columns if col not in cat_cols]

plot_numerical_features(eda_dfs[df_name][num_cols])
plot_categorical_features(eda_dfs[df_name][cat_cols])


df_name = "insta_payments"
cat_cols = eda_dfs[df_name].select_dtypes(include='object').columns
num_cols = [col for col in eda_dfs[df_name].columns if col not in cat_cols]

plot_numerical_features(eda_dfs[df_name][num_cols])
plot_categorical_features(eda_dfs[df_name][cat_cols])


for name, df in eda_dfs.items():

    correlations = df.corr(numeric_only=True)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(16, 12))
    
    # Create a heatmap
    sns.heatmap(correlations, 
                cmap='coolwarm', 
                annot=True,
                fmt=".2f", 
                linewidths=0.5, 
                cbar_kws={"shrink": 0.8})
    
    plt.title(f'{name} Feature Correlation Matrix', fontsize=18)
    plt.tight_layout()
    plt.show()


to_drop = {
    'app_train': ['CNT_CHILDREN', 'AMT_CREDIT', 'REGION_RATING_CLIENT_W_CITY',
                  'FLOORSMAX_MODE', 'FLOORSMAX_MEDI', 'OBS_60_CNT_SOCIAL_CIRCLE',
                  'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5',
                  'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 'AMT_REQ_CREDIT_BUREAU_DAY',
                  'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                  'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
                  'ORGANIZATION_TYPE'],
    'bureau': [],
    'pos_cash': ['CNT_INSTALMENT'],
    'credit_card': ['AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
                    'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_DRAWINGS_POS_CURRENT'],
    'previous_app': ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'DAYS_LAST_DUE'],
    'insta_payments': ['DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT']
}

for name, cols in to_drop.items():
    dropped_dfs[name] = dropped_dfs[name].drop(columns=cols)

## app_test columns must be same as app_train columns
app_test = app_test.drop(columns=to_drop['app_train'])



for name, df in dropped_dfs.items():
    print(name, f"Shape: {df.shape}")

print("app_test", f"Shape: {app_test.shape}")


# run functions and pre_settings
def one_hot_encoder(df, nan_as_category=False):
    original_columns = list(df.columns)
    cols_to_encode = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            # if df[col].nunique() > 3:
            cols_to_encode.append(col)
    df = pd.get_dummies(df, columns=cols_to_encode, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def app_cleaning(app_train, app_test):
    
    # general cleaning procedures
    app_train = app_train[app_train['CODE_GENDER'] != 'XNA'] ## no problem as 'XNA' appears only in training data
    app_train = app_train[app_train['AMT_INCOME_TOTAL'] < 20000000] # remove a outlier 117M
    app_train = app_train[app_train['NAME_FAMILY_STATUS'] != "Unknown"] # remove a outlier
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    app_train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True) # set null value
    app_train['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True) # set null value
    app_test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True) # set null value
    app_test['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True) # set null value

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        app_train[bin_feature], uniques = pd.factorize(app_train[bin_feature])
        app_test[bin_feature], uniques = pd.factorize(app_test[bin_feature])

    app_train, _ = one_hot_encoder(app_train)
    app_test, _ = one_hot_encoder(app_test)
    
    # fix app_test does not have NAME_INCOME_TYPE value of "Maternity leave"
    app_test["NAME_INCOME_TYPE_Maternity leave"] = False
    
    return app_train, app_test

dropped_dfs['app_train'], app_test = app_cleaning(app_train=dropped_dfs['app_train'], app_test=app_test)


def get_age_label(days_birth):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    if age_years < 27: return 1
    elif age_years < 40: return 2
    elif age_years < 50: return 3
    elif age_years < 65: return 4
    elif age_years < 99: return 5
    else: return 0


def app_engineering(app_train, app_test):
    # Categorical age - based on target=1 plot
    app_train['AGE_RANGE'] = app_train['DAYS_BIRTH'].apply(lambda x: get_age_label(x))
    app_test['AGE_RANGE'] = app_test['DAYS_BIRTH'].apply(lambda x: get_age_label(x))

    # Some simple new features (percentages)
    app_train['DAYS_EMPLOYED_PERC'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']
    app_train['INCOME_PER_PERSON'] = app_train['AMT_INCOME_TOTAL'] / app_train['CNT_FAM_MEMBERS']
    app_train['ANNUITY_INCOME_PERC'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
    
    app_test['DAYS_EMPLOYED_PERC'] = app_test['DAYS_EMPLOYED'] / app_test['DAYS_BIRTH']
    app_test['INCOME_PER_PERSON'] = app_test['AMT_INCOME_TOTAL'] / app_test['CNT_FAM_MEMBERS']
    app_test['ANNUITY_INCOME_PERC'] = app_test['AMT_ANNUITY'] / app_test['AMT_INCOME_TOTAL']

    # Income ratios
    app_train['INCOME_TO_EMPLOYED_RATIO'] = app_train['AMT_INCOME_TOTAL'] / app_train['DAYS_EMPLOYED']
    app_train['INCOME_TO_BIRTH_RATIO'] = app_train['AMT_INCOME_TOTAL'] / app_train['DAYS_BIRTH']
    
    app_test['INCOME_TO_EMPLOYED_RATIO'] = app_test['AMT_INCOME_TOTAL'] / app_test['DAYS_EMPLOYED']
    app_test['INCOME_TO_BIRTH_RATIO'] = app_test['AMT_INCOME_TOTAL'] / app_test['DAYS_BIRTH']

    # Time ratios
    app_train['ID_TO_BIRTH_RATIO'] = app_train['DAYS_ID_PUBLISH'] / app_train['DAYS_BIRTH']
    app_train['PHONE_TO_BIRTH_RATIO'] = app_train['DAYS_LAST_PHONE_CHANGE'] / app_train['DAYS_BIRTH']
    
    app_test['ID_TO_BIRTH_RATIO'] = app_test['DAYS_ID_PUBLISH'] / app_test['DAYS_BIRTH']
    app_test['PHONE_TO_BIRTH_RATIO'] = app_test['DAYS_LAST_PHONE_CHANGE'] / app_test['DAYS_BIRTH']

    app_train['APP_DAYS_EMPLOYED_DAYS_BIRTH_diff'] = app_train['DAYS_EMPLOYED'] - app_train['DAYS_BIRTH']
    
    app_test['APP_DAYS_EMPLOYED_DAYS_BIRTH_diff'] = app_test['DAYS_EMPLOYED'] - app_test['DAYS_BIRTH']
    
    return app_train, app_test

dropped_dfs['app_train'], app_test = app_engineering(app_train=dropped_dfs['app_train'], app_test=app_test)


# def bureau_engineering(bureau):
    
#     # Credit duration and credit/account end date difference
#     bureau['CREDIT_DURATION'] = -bureau['DAYS_CREDIT'] + bureau['DAYS_CREDIT_ENDDATE']
#     bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    
#     # Credit to debt ratio and difference
#     bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
#     bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
#     bureau['BUREAU_CREDIT_FACT_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
#     bureau['BUREAU_CREDIT_ENDDATE_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
#     bureau['BUREAU_CREDIT_DEBT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']

#     # CREDIT_DAY_OVERDUE :
#     bureau['BUREAU_IS_DPD'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 0 else 0)
#     bureau['BUREAU_IS_DPD_OVER120'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 120 else 0)

#     bureau, bureau_cat = one_hot_encoder(bureau)
    
#     return bureau, bureau_cat

# dropped_dfs['bureau'], bureau_cat = bureau_engineering(bureau=dropped_dfs['bureau'])


def aggregate(df):
    ## df
    df, df_cat = one_hot_encoder(df)

    # Step 1: Identify numerical and categorical (one-hot) columns
    all_cols = [col for col in df.columns if col not in  ['SK_ID_CURR', 'SK_ID_PREV']]
    numerical_cols = [col for col in all_cols if col not in df_cat]  # original numeric

    # Step 2: Define aggregations
    agg_dict = {col: ['min', 'max', 'mean', 'var'] for col in numerical_cols}
    agg_dict.update({col: ['mean'] for col in df_cat})

    # Step 3: Groupby and aggregate
    df_agg = df.groupby('SK_ID_CURR').agg(agg_dict)

    # Step 4: Flatten MultiIndex columns
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg.reset_index(inplace=True)
    return df_agg


bureau_agg = aggregate(df=dropped_dfs['bureau'])
print(f"bureau_agg shape: {bureau_agg.shape}")

previous_app_agg = aggregate(df=dropped_dfs['previous_app'])
print(f"previous_app_agg shape: {previous_app_agg.shape}")

pos_cash_agg = aggregate(df=dropped_dfs['pos_cash'])
print(f"pos_cash_agg shape: {pos_cash_agg.shape}")

insta_payments_agg = aggregate(df=dropped_dfs['insta_payments'])
print(f"insta_payments_agg shape: {insta_payments_agg.shape}")

credit_card_agg = aggregate(df=dropped_dfs['credit_card'])
print(f"credit_card_agg shape: {credit_card_agg.shape}")


application_train  = dropped_dfs['app_train']
application_test  = app_test

application_train = application_train.merge(bureau_agg, how='left', on='SK_ID_CURR')
print('--=> application_train after merge with bureau:', application_train.shape)
application_test = application_test.merge(bureau_agg, how='left', on='SK_ID_CURR')
print('--=> application_test after merge with bureau:', application_test.shape)

application_train = application_train.merge(previous_app_agg, how='left', on='SK_ID_CURR')
print('--=> application_train after merge with previous_app:', application_train.shape)
application_test = application_test.merge(previous_app_agg, how='left', on='SK_ID_CURR')
print('--=> application_test after merge with previous_app:', application_test.shape)

application_train = application_train.merge(pos_cash_agg, how='left', on='SK_ID_CURR')
print('--=> application_train after merge with pos_cash:', application_train.shape)
application_test = application_test.merge(pos_cash_agg, how='left', on='SK_ID_CURR')
print('--=> application_test after merge with pos_cash:', application_test.shape)

application_train = application_train.merge(insta_payments_agg, how='left', on='SK_ID_CURR')
print('--=> application_train after merge with insta_payments:', application_train.shape)
application_test = application_test.merge(insta_payments_agg, how='left', on='SK_ID_CURR')
print('--=> application_test after merge with insta_payments:', application_test.shape)

application_train = application_train.merge(credit_card_agg, how='left', on='SK_ID_CURR')
print('--=> application_train after merge with credit_card:', application_train.shape)
application_test = application_test.merge(credit_card_agg, how='left', on='SK_ID_CURR')
print('--=> application_test after merge with credit_card:', application_test.shape)



def reduce_mem_usage(dataframe):
    m_start = dataframe.memory_usage().sum() / 1024 ** 2
    for col in dataframe.columns:
        col_type = dataframe[col].dtype
        if col_type != object:
            c_min = dataframe[col].min()
            c_max = dataframe[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dataframe[col] = dataframe[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dataframe[col] = dataframe[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dataframe[col] = dataframe[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dataframe[col] = dataframe[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dataframe[col] = dataframe[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dataframe[col] = dataframe[col].astype(np.float32)
                else:
                    dataframe[col] = dataframe[col].astype(np.float64)

    m_end = dataframe.memory_usage().sum() / 1024 ** 2
    return dataframe


application_train = reduce_mem_usage(application_train)
application_test = reduce_mem_usage(application_test)


base_folder_id = '/kaggle/working/home-credit-default-risk-working_data'
import os; os.makedirs(base_folder_id, exist_ok=True)

application_train.to_csv(os.path.join(base_folder_id, 'application_train.csv'), index=False)
application_test.to_csv(os.path.join(base_folder_id, 'application_test.csv'), index=False)

