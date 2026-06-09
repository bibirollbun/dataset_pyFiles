import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os


import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from sklearn.preprocessing import LabelEncoder

import gc
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
import time



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data_path = '/kaggle/input/home-credit-default-risk/'




app_train = pd.read_csv(f'{data_path}/application_train.csv')
app_test=pd.read_csv(f'{data_path}/application_test.csv')
bureau = pd.read_csv(f'{data_path}/bureau.csv')
bureau_balance = pd.read_csv(f'{data_path}/bureau_balance.csv')
pos_cash_balance = pd.read_csv(f'{data_path}/POS_CASH_balance.csv')
credit_card_balance = pd.read_csv(f'{data_path}/credit_card_balance.csv')
previous_application = pd.read_csv(f'{data_path}/previous_application.csv')
installments_payments = pd.read_csv(f'{data_path}/installments_payments.csv')


bb_agg = (bureau_balance.groupby("SK_ID_BUREAU").agg(months_min=("MONTHS_BALANCE", "min"),months_max=("MONTHS_BALANCE", "max"),status_last=("STATUS", "last")).reset_index())


bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")


# Aggregate the bureau table with app_train by customer (SK_ID_CURR)
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'AMT_CREDIT_SUM': ['mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    "CREDIT_ACTIVE": lambda s: (s == "Active").sum(),
    "CREDIT_DAY_OVERDUE": ["max"],
    'AMT_ANNUITY': ['mean', 'sum'],
    'DAYS_CREDIT': ['mean', 'min', 'max'],
    'CREDIT_TYPE': 'nunique'  
}).reset_index()


# Flatten MultiIndex columns
bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
bureau_agg.rename(columns={'SK_ID_CURR_': 'SK_ID_CURR'}, inplace=True)


app_train = pd.merge(app_train, bureau_agg, on='SK_ID_CURR', how='left')
app_test = pd.merge(app_test, bureau_agg, on='SK_ID_CURR', how='left')


PA_agg = previous_application.groupby("SK_ID_CURR").agg({
    "SK_ID_PREV": "count",  # Count of previous applications (how many loans each customer has applied for)
    "NAME_CONTRACT_STATUS": lambda s: (s == "Approved").sum(),  # Count of approved applications
    "AMT_APPLICATION": "mean",  # Mean application amount
    "AMT_CREDIT": "mean",  # Mean credit amount
}).reset_index()


app_train = pd.merge(app_train, PA_agg, on='SK_ID_CURR', how='left')
app_test = pd.merge(app_test, PA_agg, on='SK_ID_CURR', how='left')


# Positive = paid late (bad)
# Negative = paid early (good)

installments_payments["PAY_DIFF"] = installments_payments["AMT_PAYMENT"] - installments_payments["AMT_INSTALMENT"]
installments_payments["PAY_DELAY_DAYS"] = installments_payments["DAYS_ENTRY_PAYMENT"] - installments_payments["DAYS_INSTALMENT"]


IP_agg = (
    installments_payments
    .groupby("SK_ID_CURR")
    .agg(
        INST_CNT=("SK_ID_PREV", "count"),
        INST_PAY_DIFF_MEAN=("PAY_DIFF", "mean"),
        INST_PAY_DIFF_MIN=("PAY_DIFF", "min"),
        INST_PAY_DIFF_MAX=("PAY_DIFF", "max"),
        INST_DELAY_MEAN=("PAY_DELAY_DAYS", "mean"),
        INST_DELAY_MAX=("PAY_DELAY_DAYS", "max"),
        INST_TOTAL_PAID=("AMT_PAYMENT", "sum"),
        INST_TOTAL_DUE=("AMT_INSTALMENT", "sum"),
    )
).reset_index()


app_train = pd.merge(app_train, IP_agg, on='SK_ID_CURR', how='left')
app_test = pd.merge(app_test, IP_agg, on='SK_ID_CURR', how='left')


pos_agg = pos_cash_balance.groupby("SK_ID_CURR").agg({
    "SK_ID_PREV": "nunique",  # Count unique POS loans per customer
    "SK_DPD": "max"  # Max of SK_DPD (maximum delinquency)
}).reset_index()


app_train = pd.merge(app_train, pos_agg, on='SK_ID_CURR', how='left')
app_test = pd.merge(app_test, pos_agg, on='SK_ID_CURR', how='left')


cc_agg = credit_card_balance.groupby("SK_ID_CURR").agg({
    "AMT_BALANCE": "mean",  # Mean balance on credit card
    "SK_DPD": "mean"  # Mean delinquency period on credit card
}).reset_index()


app_train = pd.merge(app_train, cc_agg, on='SK_ID_CURR', how='left')
app_test = pd.merge(app_test, cc_agg, on='SK_ID_CURR', how='left')


print(f'The applicatoin training dataset contains {app_train.shape[1]} columns and {app_train.shape[0]} raws')
print(f'The applicatoin test dataset contains {app_test.shape[1]} columns and {app_test.shape[0]} raws')


df_train_categorical=app_train.select_dtypes(include='object')
df_train_numerical=app_train.select_dtypes(exclude='object')

df_test_categorical=app_test.select_dtypes(include='object')
df_test_numerical=app_test.select_dtypes(exclude='object')


temp = app_train["TARGET"].value_counts()
df = pd.DataFrame({'labels': temp.index, 'values': temp.values})

colors = sns.color_palette("Set2", n_colors=len(df))

plt.figure(figsize=(7, 7))
plt.pie(df['values'], labels=df['labels'], autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.1, 0))

plt.title('Loan Repayed or Not', fontsize=16, weight='bold')

plt.axis('equal')  
plt.show()


plt.figure(figsize=(22, 28))
for i, col in enumerate(df_train_categorical.drop(columns=['ORGANIZATION_TYPE'])):
    
    plt.subplot(4, 4, i + 1)
    df_train_categorical[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    
    plt.title(f'Countplot of: {col}', fontsize=16, fontweight='bold')
    plt.xticks( fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 8))

df_train_categorical['ORGANIZATION_TYPE'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Countplot of: ORGANIZATION_TYPE', fontsize=16, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


def plot_loan_percentage(feature, df=app_train, colormap='cividis', rotation=0, figsize=(8, 6)):

    # Calculate the percentage of Target based on the given feature values
    loan_percentage = df.groupby([feature, 'TARGET']).size().unstack(fill_value=0)

    # Convert counts to percentages based on the given feature values
    loan_percentage = loan_percentage.div(loan_percentage.sum(axis=1), axis=0) * 100

    # Sort the values in ascending order based on Target percentage (ascending order for 'Loan_Statused')
    loan_percentage = loan_percentage[['Repaid', 'Not Repaid']].sort_values(by='Repaid', ascending= False)

    # Plot stacked bar chart for percentage comparison
    ax = loan_percentage.plot(kind='bar', stacked=True, figsize=figsize, colormap=colormap)

    # Add percentage labels on top of the bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # only label bars with non-zero height
            ax.text(p.get_x() + p.get_width() / 2, p.get_height() + p.get_y(),
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=12, color='black')

    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    plt.title(f'Percentage of Loan_Status for {feature.capitalize()}', fontsize=14)
    plt.xlabel(feature.capitalize(), fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.xticks(rotation=rotation,fontsize=14)
    plt.legend(title='Loan_Status', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    plt.show()


app_train['TARGET']=app_train['TARGET'].map({0: 'Repaid', 1: 'Not Repaid'})


plot_loan_percentage('NAME_CONTRACT_TYPE')


plot_loan_percentage('CODE_GENDER')


# purpose of the loan
plot_loan_percentage('FLAG_OWN_CAR'),plot_loan_percentage('FLAG_OWN_REALTY')


plot_loan_percentage('NAME_INCOME_TYPE', colormap='rainbow', rotation=45, figsize=(16,12))


plot_loan_percentage('NAME_FAMILY_STATUS', colormap='rainbow', rotation=45, figsize=(16,12))


plot_loan_percentage('NAME_EDUCATION_TYPE', colormap='rainbow', rotation=45, figsize=(16,12))


plot_loan_percentage('ORGANIZATION_TYPE',colormap='inferno',rotation=45, figsize=(32,18))


# Plot distribution of one feature
def plot_distribution(feature, color):
    plt.figure(figsize=(10, 6))
    plt.title(f"Distribution of {feature}", fontsize=16, fontweight='bold')
    sns.histplot(app_train[feature].dropna(), color=color, kde=True, bins=100, edgecolor='black')
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


plot_distribution('AMT_GOODS_PRICE',color='red')


plot_distribution('AMT_CREDIT_x',color='red')


app_train['TARGET']=app_train['TARGET'].map({'Repaid': 0 , 'Not Repaid':1 })


def plot_distribution_continous(df, var):
    nrow = len(var)
    t1 = df.loc[df['TARGET'] != 0]
    t0 = df.loc[df['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure(figsize=(18, 14 * nrow))

    for i, feature in enumerate(var, 1):
        # Distribution plot
        plt.subplot(nrow, 2, i)
        plt.title(f"Distribution of {feature}", fontsize=16, fontweight='bold')
        sns.histplot(df[feature].dropna(), kde=True, bins=100, edgecolor='black', color='skyblue')
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Density plot by TARGET
        i += 1
        plt.subplot(nrow, 2, i)
        sns.kdeplot(t1[feature], bw_adjust=0.5, label="Not Repaid", shade=True, color='blue', alpha=0.7)
        sns.kdeplot(t0[feature], bw_adjust=0.5, label="Repaid", shade=True, color='red', alpha=0.7)
        plt.title(f"Density Plot of {feature}", fontsize=16, fontweight='bold')
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.show()


var = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT_x', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

plot_distribution_continous(app_train, var)


plt.figure(figsize=(10, 6))
sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_CREDIT_y', data=app_train)
plt.title('Income vs Credit Amount')
plt.xlabel('Total Income')
plt.ylabel('Requested Credit Amount')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_CREDIT_x', data=app_train)
plt.title('Income vs Credit Amount ')
plt.xlabel('Total Income')
plt.ylabel('Requested Credit Amount')
plt.show()


# Investigate Family Size and Housing Ownership
plt.figure(figsize=(10, 6))
sns.boxplot(x='NAME_HOUSING_TYPE', y='AMT_CREDIT_y', data=app_train)
plt.title('Housing Type vs Credit Amount')
plt.xlabel('Housing Type')
plt.ylabel('Requested Credit Amount')
plt.show()

# Family Size Impact vs Default Risk (scatter plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CNT_CHILDREN', y='AMT_CREDIT_y', data=app_train)
plt.title('Family Size vs Credit Amount')
plt.xlabel('Number of Children')
plt.ylabel('Requested Credit Amount')
plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(x='DAYS_EMPLOYED', y='AMT_CREDIT_y', data=app_train)
plt.title('Employment Duration vs Credit Amount')
plt.xlabel('Employment Duration (in days)')
plt.ylabel('Requested Credit Amount')
plt.show()

# Occupation vs Credit Amount
plt.figure(figsize=(12, 8))
sns.boxplot(x='OCCUPATION_TYPE', y='AMT_CREDIT_y', data=app_train)
plt.title('Occupation vs Credit Amount')
plt.xlabel('Occupation Type')
plt.ylabel('Requested Credit Amount')
plt.xticks(rotation=45)
plt.show()



day_columns = []
for col in app_train.columns:
    if 'day' in col.lower(): 
        day_columns.append(col)


app_train[day_columns]


# DAYS_BIRTH
app_train['DAYS_BIRTH'] = app_train['DAYS_BIRTH'].abs()  
app_train['AGE'] = (app_train['DAYS_BIRTH'].abs() / 365.25).astype(int)  
app_train.drop(columns=['DAYS_BIRTH'], inplace=True)

app_test['DAYS_BIRTH'] = app_test['DAYS_BIRTH'].abs()  
app_test['AGE'] = (app_test['DAYS_BIRTH'].abs() / 365.25).astype(int)  
app_test.drop(columns=['DAYS_BIRTH'], inplace=True)


#DAYS_EMPLOYED
app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].abs()  
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
app_train['DAYS_EMPLOYED'].fillna(app_train['DAYS_EMPLOYED'].median(), inplace=True)

app_test['DAYS_EMPLOYED'] = app_test['DAYS_EMPLOYED'].abs()  
app_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
app_test['DAYS_EMPLOYED'].fillna(app_test['DAYS_EMPLOYED'].median(), inplace=True)


day_columns = ['DAYS_REGISTRATION','DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE','DAYS_CREDIT_mean','DAYS_CREDIT_min','DAYS_CREDIT_max']

def handle_anomalies_days(df):
    for col in day_columns:
        if col in df.columns:
            # Convert negative values to positive
            df[col] = df[col].apply(lambda x: abs(x) if x < 0 else x)
            
    return df


app_train = handle_anomalies_days(app_train)
app_test = handle_anomalies_days(app_test)


def find_columns_with_negative_values(df):
    
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns_with_negative_values = []

    for col in numerical_columns:
        if (df[col] < 0).any():  # Check if any value in the column is negative
            columns_with_negative_values.append(col)  # Add to the list if negative values are found

    return columns_with_negative_values


columns_with_negative_values = find_columns_with_negative_values(app_train)

print(columns_with_negative_values)


app_train[columns_with_negative_values]


def handle_negative_values(df):

    columns_with_negative_values_to_handle=['AMT_CREDIT_SUM_DEBT_mean','AMT_CREDIT_SUM_DEBT_sum','INST_DELAY_MEAN','INST_DELAY_MAX','AMT_BALANCE']

    for col in columns_with_negative_values_to_handle:
        if col in df.columns:
            # Set negative values to NaN
            df[col] = df[col].apply(lambda x: x if x >= 0 else np.nan)

    return df



app_train = handle_negative_values(app_train)
app_test = handle_negative_values(app_test)



app_train['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
app_test['CODE_GENDER'].replace('XNA', np.nan, inplace=True)


def missing_values_table(df):

    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table.columns = ['No. of Missing Values', '% of the Missing Values']

    mis_val_table = mis_val_table[mis_val_table['% of the Missing Values'] > 0].sort_values(
        '% of the Missing Values', ascending=False).round(1)

    print(f"The dataframe has {df.shape[1]} columns.\n"
          f"There are {mis_val_table.shape[0]} columns that have missing values.")
    
    return mis_val_table


missing_values_table(app_train)


missing_values_table(app_test)


df_train = app_train.copy()
df_test = app_test.copy()


df_train_categorical=df_train.select_dtypes(include='object')
df_train_numerical=df_train.select_dtypes(exclude='object')

df_test_categorical=df_test.select_dtypes(include='object')
df_test_numerical=df_test.select_dtypes(exclude='object')


missing_values_table(df_train_categorical)


missing_values_table(df_test_categorical)


# Delete Features with more than 50 % missing

df_train.drop(columns=['FONDKAPREMONT_MODE','WALLSMATERIAL_MODE','HOUSETYPE_MODE','EMERGENCYSTATE_MODE'], inplace=True)
df_test.drop(columns=['FONDKAPREMONT_MODE','WALLSMATERIAL_MODE','HOUSETYPE_MODE','EMERGENCYSTATE_MODE'], inplace=True)


# impute other catergorical columns

# 1- impute the OCCUPATION_TYPE with random imputation (18 different job)

missing = df_train['OCCUPATION_TYPE'].isna()
categories = df_train['OCCUPATION_TYPE'].dropna().unique()

imputed_values = np.random.choice(categories, size=missing.sum(), replace=True)
df_train.loc[missing, 'OCCUPATION_TYPE'] = imputed_values
# test date
df_test['OCCUPATION_TYPE']=df_test['OCCUPATION_TYPE'].fillna(np.random.choice(categories))

# 2- impute the NAME_TYPE_SUITE with the mode

mode_value = df_train['NAME_TYPE_SUITE'].mode()[0]
df_train['NAME_TYPE_SUITE'].fillna(mode_value, inplace=True)
# test date
df_test['NAME_TYPE_SUITE'].fillna(mode_value, inplace=True)

# 3- impute the CODE_GENDER with the mode

mode_value = df_train['CODE_GENDER'].mode()[0]
df_train['CODE_GENDER'].fillna(mode_value, inplace=True)


df_train


df_test


threshold = 0.8

train_corr = df_train_numerical.corr().abs()
train_corr.head()


to_drop = []

for i in range(len(train_corr.columns)):
    for j in range(i):
        if train_corr.iloc[i, j] > threshold:
            colname = train_corr.columns[i]
            if colname not in to_drop:  # Only add once
                to_drop.append(colname)

print('There are %d columns to remove.' % len(to_drop))


df_train.drop(columns=to_drop, inplace=True)
df_test.drop(columns=to_drop, inplace=True)


missing_values_table(df_train)


missing_values_table(df_test)


train_missing = (df_train.isnull().sum() / len(df_train)).sort_values(ascending = False)
test_missing = (df_test.isnull().sum() / len(df_test)).sort_values(ascending = False)


train_missing = train_missing.index[train_missing > 0.45]
test_missing = test_missing.index[test_missing > 0.45]

all_missing = list(set(set(train_missing) | set(test_missing)))
print('There are %d columns with more than 50%% missing values' % len(all_missing))



df_train.drop(columns=all_missing, inplace=True)
df_test.drop(columns=all_missing, inplace=True)



missing_values_table(df_train)


missing_values_table(df_test)


missing_train_cols = [col for col in df_train.columns if df_train[col].isnull().any()]
missing_test_cols = [col for col in df_test.columns if df_test[col].isnull().any()]


len(missing_train_cols), len(missing_test_cols)


common_missing_cols = list(set(missing_train_cols) & set(missing_test_cols))

# Print the results
print("Columns with missing values in both datasets:")
print(common_missing_cols)

print("\nColumns with missing values in train dataset:")
print(list(set(missing_train_cols) - set(missing_test_cols)))

print("\nColumns with missing values in test dataset:")
print(list(set(missing_test_cols) - set(missing_train_cols)))


def impute_data(df):

    # 0-imputation for count or flag features
    zero_fill_features = [
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'CREDIT_ACTIVE_<lambda>',
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_YEAR',
        'SK_ID_PREV_x',
        'SK_DPD_x',
        'OBS_30_CNT_SOCIAL_CIRCLE'
    ]
    for col in zero_fill_features:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    # Median imputation for monetary and delay/time-based features
    median_fill_features = [
        'CREDIT_DAY_OVERDUE_max',
        'AMT_CREDIT_SUM_sum',
        'INST_TOTAL_PAID',
        'AMT_APPLICATION',
        'DAYS_CREDIT_mean',
        'DAYS_CREDIT_max',
        'AMT_CREDIT_SUM_OVERDUE_mean',
        'AMT_CREDIT_SUM_OVERDUE_sum',
        'AMT_CREDIT_SUM_DEBT_sum',
        'AMT_CREDIT_SUM_DEBT_mean',
        'INST_DELAY_MAX',
        'INST_PAY_DIFF_MEAN',
        'INST_PAY_DIFF_MIN',
        'INST_PAY_DIFF_MAX',
        'AMT_CREDIT_SUM_mean',
        'AMT_ANNUITY_sum',
        'AMT_ANNUITY',
        'INST_CNT',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3'
    ]
    for col in median_fill_features:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Mode imputation for categorical/ordinal features
    if 'NAME_CONTRACT_STATUS' in df.columns:
        df['NAME_CONTRACT_STATUS'].fillna(df['NAME_CONTRACT_STATUS'].mode()[0], inplace=True)

    if 'CREDIT_TYPE_nunique' in df.columns:
        df['CREDIT_TYPE_nunique'].fillna(df['CREDIT_TYPE_nunique'].mode()[0], inplace=True)

    return df



df_train['DAYS_LAST_PHONE_CHANGE'].fillna(df_train['DAYS_LAST_PHONE_CHANGE'].median(), inplace=True)


df_train = impute_data(df_train)



df_test = impute_data(df_test)



missing_values_table(df_train)


missing_values_table(df_test)


df_train.columns = df_train.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
df_test.columns = df_test.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)


le = LabelEncoder()
counter = 0

for col in df_train:
    if df_train[col].dtype == 'object':
        if len(list(df_train[col].unique())) <= 2:
            le.fit(df_train[col])
            df_train[col] = le.transform(df_train[col])
            df_test[col] = le.transform(df_test[col])
            
            counter += 1
            
print('%d columns were label encoded.' % counter)


# one-hot-encoding

df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)


train_labels = df_train['TARGET']

df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)

df_train['TARGET'] = train_labels

print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)


TARGET = 'TARGET'
ID = 'SK_ID_CURR'


features = [col for col in df_train.columns if col not in [TARGET, ID]]
X = df_train[features]
y = df_train[TARGET]
X_test = df_test[features]


X.columns = X.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
X_test.columns = X_test.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)


num_folds = 5
stratified = True  
debug = False
submission_file_name = "submission_lgbm.csv"


print(f"Training data shape: {X.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of features: {len(features)}")
print(f"Target distribution: {y.value_counts().sort_index()}")


if stratified:
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    print(f"Using Stratified {num_folds}-fold cross-validation")
else:
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    print(f"Using Regular {num_folds}-fold cross-validation")

# Initialize result containers
oof_preds = np.zeros(X.shape[0])  # Out-of-fold predictions
test_preds = np.zeros(X_test.shape[0])  # Test set predictions
feature_importance_df = pd.DataFrame()

print("Cross-validation setup completed")


print("Starting cross-validation training...")
cv_start_time = time.time()

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    print(f"\n=== Fold {n_fold + 1}/{num_folds} ===")
    fold_start_time = time.time()
    
    # Split data for current fold
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")
    print(f"Train target distribution: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Valid target distribution: {y_valid.value_counts().sort_index().to_dict()}")

    # Initialize LightGBM parameters
    clf = LGBMClassifier(
        n_jobs=4,                     # Use 4 CPU threads for parallel processing
        n_estimators=10000,           # Maximum number of boosting rounds
        learning_rate=0.02,           # Learning rate (small for better performance)
        num_leaves=34,                # Number of leaves in each tree
        colsample_bytree=0.9497036,   # Fraction of features used per tree
        subsample=0.8715623,          # Fraction of samples used per tree
        max_depth=8,                  # Maximum tree depth
        reg_alpha=0.041545473,        # L1 regularization
        reg_lambda=0.0735294,         # L2 regularization
        min_split_gain=0.0222415,     # Minimum gain to make split
        min_child_weight=39.3259775,  # Minimum sum of instance weight in child
        verbosity=-1,                 # Updated from 'verbose' - controls LightGBM's own verbosity
        random_state=1001,            # For reproducibility
        force_col_wise=True           # Forces the algorithm to work with column-wise data for better performance
    )

    # Setup callbacks for training 
    callbacks = [
        log_evaluation(period=200),   
        early_stopping(stopping_rounds=200)  
    ]

    # Train model with callbacks
    clf.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], 
            eval_metric='auc',           # Use AUC as evaluation metric, as asked in the competition
            callbacks=callbacks)         # Use callbacks instead of verbose and early_stopping_rounds

    # Generate out-of-fold predictions
    oof_preds[valid_idx] = clf.predict_proba(X_valid)[:, 1]
    
    # Generate test predictions and accumulate
    test_preds += clf.predict_proba(X_test)[:, 1] / folds.n_splits

    # Store feature importance for this fold
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    # Calculate and print fold performance
    fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
    print(f'Fold {n_fold + 1:2d} AUC: {fold_auc:.6f}')
    print(f'Best iteration: {clf.best_iteration_}')
    print(f'Fold {n_fold + 1} completed in {time.time() - fold_start_time:.0f}s')
    
    # Clean up memory
    del clf, X_train, X_valid, y_train, y_valid
    gc.collect()

print(f"\nCross-validation completed in {time.time() - cv_start_time:.0f}s")


overall_auc = roc_auc_score(y, oof_preds)
print(f'\n{"="*50}')
print(f'FINAL RESULTS')
print(f'{"="*50}')
print(f'Overall AUC Score: {overall_auc:.6f}')

# Calculate fold-wise AUC scores for detailed analysis
fold_aucs = []
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    fold_auc = roc_auc_score(y.iloc[valid_idx], oof_preds[valid_idx])
    fold_aucs.append(fold_auc)
    print(f'Fold {n_fold + 1:2d} AUC: {fold_auc:.6f}')

print(f'\nAUC Statistics:')
print(f'Mean: {np.mean(fold_aucs):.6f}')
print(f'Std:  {np.std(fold_aucs):.6f}')
print(f'Min:  {np.min(fold_aucs):.6f}')
print(f'Max:  {np.max(fold_aucs):.6f}')


# Plot ROC curve for the final model (using all predictions)

fpr, tpr, thresholds = roc_curve(y, oof_preds)
roc_auc = roc_auc_score(y, oof_preds)

plt.style.use('seaborn-darkgrid')

# Plot the ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='teal', lw=3, label=f'ROC curve (AUC = {roc_auc:.2f})', alpha=0.8)

plt.xlabel('False Positive Rate', fontsize=14, weight='bold')
plt.ylabel('True Positive Rate', fontsize=14, weight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, weight='bold')
plt.legend(loc='lower right', fontsize=12)

plt.grid(True, linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()


print("Feature importance...")
avg_importance = (feature_importance_df[["feature", "importance"]]
                 .groupby("feature")
                 .mean()
                 .sort_values(by="importance", ascending=False))

print(f"\nTop 20 Most Important Features:")
print(avg_importance.head(20))


top_features = avg_importance.head(40).index

# Filter data for top features only
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(top_features)]

plt.figure(figsize=(10, 12))
sns.barplot(x="importance", y="feature", 
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Feature Importance (Average across folds)', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)

plt.tight_layout()
plt.show()


print("Creating submission file...")

submission_df = pd.DataFrame({
    ID: df_test[ID].values,
    TARGET: test_preds
})

# Save submission file
submission_df.to_csv(submission_file_name, index=False)
print(f"Submission file saved as '{submission_file_name}'")


df_train_fet = df_train.copy()
df_test_fet = df_test.copy()


# 1. Credit-to-Income Ratio (CIR)
# Formula: CIR = AMT_CREDIT_x / AMT_INCOME_TOTAL
df_train_fet['CREDIT_TO_INCOME_RATIO'] = df_train_fet['AMT_CREDIT_x'] / df_train_fet['AMT_INCOME_TOTAL']
df_test_fet['CREDIT_TO_INCOME_RATIO'] = df_test_fet['AMT_CREDIT_x'] / df_test_fet['AMT_INCOME_TOTAL']



# 2. Debt-to-Income Ratio (DTI)
# Formula: DTI = (AMT_CREDIT_x + AMT_CREDIT_SUM_DEBT_sum) / AMT_INCOME_TOTAL
df_train_fet['DEBT_TO_INCOME_RATIO'] = (df_train_fet['AMT_CREDIT_x'] + df_train_fet['AMT_CREDIT_SUM_DEBT_sum']) / df_train_fet['AMT_INCOME_TOTAL']
df_test_fet['DEBT_TO_INCOME_RATIO'] = (df_test_fet['AMT_CREDIT_x'] + df_test_fet['AMT_CREDIT_SUM_DEBT_sum']) / df_test_fet['AMT_INCOME_TOTAL']



# 3. Loan Amount Overlap
# Formula: Loan Amount Overlap = AMT_CREDIT_x / AMT_APPLICATION
df_train_fet['LOAN_AMOUNT_OVERLAP'] = df_train_fet['AMT_CREDIT_x'] / df_train_fet['AMT_APPLICATION']
df_test_fet['LOAN_AMOUNT_OVERLAP'] = df_test_fet['AMT_CREDIT_x'] / df_test_fet['AMT_APPLICATION']




# 4. Credit Utilization
# Formula: Credit Utilization = AMT_CREDIT_SUM_DEBT_sum / AMT_CREDIT_x
df_train_fet['CREDIT_UTILIZATION'] = df_train_fet['AMT_CREDIT_SUM_DEBT_sum'] / df_train_fet['AMT_CREDIT_x']
df_test_fet['CREDIT_UTILIZATION'] = df_test_fet['AMT_CREDIT_SUM_DEBT_sum'] / df_test_fet['AMT_CREDIT_x']



# 1. Average Payment Delay
# Formula: INST_DELAY_MAX / INST_CNT (max delay divided by number of installments)
df_train_fet['AVG_PAYMENT_DELAY'] = df_train_fet['INST_DELAY_MAX'] / df_train_fet['INST_CNT']
df_test_fet['AVG_PAYMENT_DELAY'] = df_test_fet['INST_DELAY_MAX'] / df_test_fet['INST_CNT']



# 2. Late Payment Intensity
# Formula: Ratio of late payments to total installments
# We use INST_DELAY_MAX here to reflect the maximum delay (it will indicate intensity of being late).
df_train_fet['LATE_PAYMENT_INTENSITY'] = df_train_fet['INST_DELAY_MAX'] / df_train_fet['INST_CNT']
df_test_fet['LATE_PAYMENT_INTENSITY'] = df_test_fet['INST_DELAY_MAX'] / df_test_fet['INST_CNT']



# 3. Overdue Debt Ratio
# Formula: AMT_CREDIT_SUM_OVERDUE_sum / AMT_CREDIT_SUM_sum
df_train_fet['OVERDUE_DEBT_RATIO'] = df_train_fet['AMT_CREDIT_SUM_OVERDUE_sum'] / df_train_fet['AMT_CREDIT_SUM_sum']
df_test_fet['OVERDUE_DEBT_RATIO'] = df_test_fet['AMT_CREDIT_SUM_OVERDUE_sum'] / df_test_fet['AMT_CREDIT_SUM_sum']




# 4. Instalment Payment Consistency
# Formula: Standard Deviation of payment differences (INST_PAY_DIFF_MIN, INST_PAY_DIFF_MAX)
df_train_fet['INST_PAYMENT_CONSISTENCY'] = df_train_fet[['INST_PAY_DIFF_MIN', 'INST_PAY_DIFF_MAX']].std(axis=1)
df_test_fet['INST_PAYMENT_CONSISTENCY'] = df_test_fet[['INST_PAY_DIFF_MIN', 'INST_PAY_DIFF_MAX']].std(axis=1)




df_train_fet


le = LabelEncoder()
counter = 0

for col in df_train_fet:
    if df_train_fet[col].dtype == 'object':
        if len(list(df_train_fet[col].unique())) <= 2:
            le.fit(df_train_fet[col])
            df_train_fet[col] = le.transform(df_train_fet[col])
            df_test_fet_to_test[col] = le.transform(df_test_fet[col])
            
            counter += 1
            
print('%d columns were label encoded.' % counter)


# one-hot-encoding

df_train_fet = pd.get_dummies(df_train_fet)
df_test_fet = pd.get_dummies(df_test_fet)

print('Training Features shape: ', df_train_fet.shape)
print('Testing Features shape: ', df_test_fet.shape)


train_labels = df_train_fet['TARGET']

df_train_fet, df_test_fet = df_train_fet.align(df_test_fet, join = 'inner', axis = 1)

df_train_fet['TARGET'] = train_labels

print('Training Features shape: ', df_train_fet.shape)
print('Testing Features shape: ', df_test_fet.shape)


features = [col for col in df_train_fet.columns if col not in [TARGET, ID]]
X = df_train_fet[features]
y = df_train_fet[TARGET]
X_test  = df_test_fet[features]


X.columns = X.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
X_test.columns = X_test.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)


print(f"Training data shape: {X.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of features: {len(features)}")
print(f"Target distribution: {y.value_counts().sort_index()}")


num_folds = 10


if stratified:
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    print(f"Using Stratified {num_folds}-fold cross-validation")
else:
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    print(f"Using Regular {num_folds}-fold cross-validation")


# Initialize result containers
oof_preds = np.zeros(X.shape[0])  # Out-of-fold predictions
test_preds = np.zeros(X_test.shape[0])  # Test set predictions
feature_importance_df = pd.DataFrame()

print("Cross-validation setup completed")

print("Starting cross-validation training...")
cv_start_time = time.time()

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    print(f"\n=== Fold {n_fold + 1}/{num_folds} ===")
    fold_start_time = time.time()
    
    # Split data for current fold
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")
    print(f"Train target distribution: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Valid target distribution: {y_valid.value_counts().sort_index().to_dict()}")

    # Initialize LightGBM parameters
    clf = LGBMClassifier(
        n_jobs=4,                     # Use 4 CPU threads for parallel processing
        n_estimators=10000,           # Maximum number of boosting rounds
        learning_rate=0.02,           # Learning rate (small for better performance)
        num_leaves=34,                # Number of leaves in each tree
        colsample_bytree=0.9497036,   # Fraction of features used per tree
        subsample=0.8715623,          # Fraction of samples used per tree
        max_depth=8,                  # Maximum tree depth
        reg_alpha=0.041545473,        # L1 regularization
        reg_lambda=0.0735294,         # L2 regularization
        min_split_gain=0.0222415,     # Minimum gain to make split
        min_child_weight=39.3259775,  # Minimum sum of instance weight in child
        verbosity=-1,                 # Updated from 'verbose' - controls LightGBM's own verbosity
        random_state=1001,            # For reproducibility
        force_col_wise=True           # Forces the algorithm to work with column-wise data for better performance
    )

    # Setup callbacks for training 
    callbacks = [
        log_evaluation(period=200),   
        early_stopping(stopping_rounds=200)  
    ]

    # Train model with callbacks
    clf.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], 
            eval_metric='auc',           # Use AUC as evaluation metric, as asked in the competition
            callbacks=callbacks)         # Use callbacks instead of verbose and early_stopping_rounds

    # Generate out-of-fold predictions
    oof_preds[valid_idx] = clf.predict_proba(X_valid)[:, 1]
    
    # Generate test predictions and accumulate
    test_preds += clf.predict_proba(X_test)[:, 1] / folds.n_splits

    # Store feature importance for this fold
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    # Calculate and print fold performance
    fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
    print(f'Fold {n_fold + 1:2d} AUC: {fold_auc:.6f}')
    print(f'Best iteration: {clf.best_iteration_}')
    print(f'Fold {n_fold + 1} completed in {time.time() - fold_start_time:.0f}s')
    
    # Clean up memory
    del clf, X_train, X_valid, y_train, y_valid
    gc.collect()

print(f"\nCross-validation completed in {time.time() - cv_start_time:.0f}s")


overall_auc = roc_auc_score(y, oof_preds)
print(f'\n{"="*50}')
print(f'FINAL RESULTS')
print(f'{"="*50}')
print(f'Overall AUC Score: {overall_auc:.6f}')

# Calculate fold-wise AUC scores for detailed analysis
fold_aucs = []
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    fold_auc = roc_auc_score(y.iloc[valid_idx], oof_preds[valid_idx])
    fold_aucs.append(fold_auc)
    print(f'Fold {n_fold + 1:2d} AUC: {fold_auc:.6f}')

print(f'\nAUC Statistics:')
print(f'Mean: {np.mean(fold_aucs):.6f}')
print(f'Std:  {np.std(fold_aucs):.6f}')
print(f'Min:  {np.min(fold_aucs):.6f}')
print(f'Max:  {np.max(fold_aucs):.6f}')


# Plot ROC curve for the final model (using all predictions)

fpr, tpr, thresholds = roc_curve(y, oof_preds)
roc_auc = roc_auc_score(y, oof_preds)

plt.style.use('seaborn-darkgrid')

# Plot the ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='teal', lw=3, label=f'ROC curve (AUC = {roc_auc:.2f})', alpha=0.8)

plt.xlabel('False Positive Rate', fontsize=14, weight='bold')
plt.ylabel('True Positive Rate', fontsize=14, weight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, weight='bold')
plt.legend(loc='lower right', fontsize=12)

plt.grid(True, linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()


print("Creating submission file...")

submission_file_name = "submission.csv"
submission_df = pd.DataFrame({
    ID: df_test[ID].values,
    TARGET: test_preds
})

# Save submission file
submission_df.to_csv(submission_file_name, index=False)
print(f"Submission file saved as '{submission_file_name}'")

