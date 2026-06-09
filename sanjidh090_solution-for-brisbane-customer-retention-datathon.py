# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


TEST_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/kaggle_test_data.csv"
TRAIN_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/train_data.csv"
MEMBERS_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/members.csv"
USER_LOGS_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/user_logs.csv"
TRANSACTIONS_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/transactions.csv"


test = pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/kaggle_test_data.csv')
test


members = pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/members.csv')
members


train_data = pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/members.csv')
train_data


transactions = pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/transactions.csv')
transactions


user_logs = pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/user_logs.csv')
user_logs


# Add these to your existing imports cell
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool

# Set a nice style for Seaborn plots
sns.set_theme(style="whitegrid")

# Initialize Bokeh for notebook output
output_notebook()


# This cell is from your original notebook. Run it before the ID guessing cell.

def guess_id_column(df):
    candidates = ["msno", "member_id", "customer_id", "user_id", "userid", "id"]
    for c in candidates:
        for col in df.columns:
            if col.lower() == c:
                return col
    # Fallback: look for columns ending in _id
    for col in df.columns:
        if col.lower().endswith("_id"):
            return col
    return None

def guess_date_columns(df):
    # Common names
    date_like = []
    for col in df.columns:
        cl = col.lower()
        if ("date" in cl) or ("_dt" in cl) or ("time" in cl) or ("_at" in cl):
            date_like.append(col)
    # Also test columns that parse cleanly as dates for a small sample
    for col in df.columns:
        if col in date_like:
            continue
        try:
            pd.to_datetime(df[col].dropna().sample(min(20, df[col].dropna().shape[0])), errors="raise", infer_datetime_format=True)
            date_like.append(col)
        except Exception:
            pass
    return list(dict.fromkeys(date_like))

def parse_dates_inplace(df, date_cols):
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        except Exception:
            df[c] = pd.to_datetime(df[c], errors="coerce")

def num_cat_cols(df, max_cardinality=30):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    # Split low-cardinality categoricals (for summaries) vs free text
    low_card = [c for c in categorical if df[c].nunique(dropna=False) <= max_cardinality]
    return numeric, categorical, low_card

def describe_categoricals(df, cols, topn=10):
    out = {}
    for c in cols:
        vc = df[c].value_counts(dropna=False).head(topn)
        out[c] = vc
    return out

def missing_report(df):
    m = df.isna().sum().sort_values(ascending=False).to_frame("missing")
    m["pct_missing"] = (m["missing"] / len(df)).round(4)
    return m

def duplicate_report(df):
    return df.duplicated().sum()

def basic_outlier_bounds(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return lo, hi


# This cell is from your original notebook. Run it before the new plotting code.

MEMBER_ID = guess_id_column(members) or "msno"
TRANS_ID = guess_id_column(transactions) or MEMBER_ID

member_date_cols = guess_date_columns(members)
trans_date_cols = guess_date_columns(transactions)
test_date_cols = guess_date_columns(test) if test is not None else []

print("Guessed MEMBER_ID:", MEMBER_ID)
print("Guessed TRANS_ID:", TRANS_ID)
print("Member date-like cols:", member_date_cols)
print("Trans date-like cols:", trans_date_cols)
print("Test date-like cols:", test_date_cols)

parse_dates_inplace(members, member_date_cols)
parse_dates_inplace(transactions, trans_date_cols)
if test is not None:
    parse_dates_inplace(test, test_date_cols)

# Ensure ID columns are strings for joins
members[MEMBER_ID] = members[MEMBER_ID].astype(str)
transactions[TRANS_ID] = transactions[TRANS_ID].astype(str)
if test is not None and MEMBER_ID in test.columns:
    test[MEMBER_ID] = test[MEMBER_ID].astype(str)


## This code enhances the '# Members Feature Audit' section

# Use the variables already defined in your notebook
num_cols = members.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in members.columns if c not in num_cols and c != MEMBER_ID]

print("--- Enhanced Numeric Histograms (Seaborn) ---")
for col in num_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(members[col], kde=True, bins=40)
    plt.title(f"Distribution for {col}", fontsize=16)
    plt.show()

print("\\n--- Enhanced Categorical Bar Charts (Plotly) ---")
for col in cat_cols[:10]: # Limit to top 10 categoricals
    # Get value counts
    counts = members[col].value_counts(dropna=False).head(20).reset_index()
    counts.columns = [col, 'count'] # Rename columns for Plotly

    # Create interactive bar chart
    fig = px.bar(
        counts,
        x=col,
        y='count',
        title=f"Top 20 Categories for {col}",
        labels={col: col, 'count': 'Number of Members'},
        text='count' # Display count on bars
    )
    fig.update_traces(textposition='outside')
    fig.show()


## This code enhances the '# Members Feature Audit' section

# Use the variables already defined in your notebook
num_cols = members.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in members.columns if c not in num_cols and c != MEMBER_ID]

print("--- Enhanced Numeric Histograms (Seaborn) ---")
for col in num_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(members[col], kde=True, bins=40)
    plt.title(f"Distribution for {col}", fontsize=16)
    plt.show()

print("\\n--- Enhanced Categorical Bar Charts (Plotly) ---")
for col in cat_cols[:10]: # Limit to top 10 categoricals
    # Get value counts
    counts = members[col].value_counts(dropna=False).head(20).reset_index()
    counts.columns = [col, 'count'] # Rename columns for Plotly

    # Create interactive bar chart
    fig = px.bar(
        counts,
        x=col,
        y='count',
        title=f"Top 20 Categories for {col}",
        labels={col: col, 'count': 'Number of Members'},
        text='count' # Display count on bars
    )
    fig.update_traces(textposition='outside')
    fig.show()


## This function is a better replacement for plot_corr_heatmap

def plot_corr_heatmap_seaborn(df, title):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        print(f"{title}: Not enough numeric columns for correlation heatmap.")
        return

    corr = num.corr()

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr,
        annot=True,      # Show the correlation values
        fmt=".2f",       # Format to two decimal places
        cmap="coolwarm", # Use a diverging colormap
        linewidths=.5
    )
    plt.title(title, fontsize=16)
    plt.show()


# How to use it (in the '# Correlation Heatmap' section)
plot_corr_heatmap_seaborn(members, "Members: Numeric Correlations (Seaborn)")
plot_corr_heatmap_seaborn(transactions, "Transactions: Numeric Correlations (Seaborn)")
# Also great for RFM features!
if 'rfm_agg' in locals():
    plot_corr_heatmap_seaborn(rfm_agg, "RFM Features Correlation (Seaborn)")


## This code enhances the '# Cohort Analysis' section visualizations

# Assuming 'retention' DataFrame is calculated as in your notebook
if 'retention' in locals():
    retention_percent = retention * 100

    print("--- Cohort Retention Heatmap (Seaborn) ---")
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        retention_percent,
        annot=True,
        fmt=".1f", # Format to one decimal place
        cmap="viridis",
        cbar_kws={'label': 'Retention Rate (%)'}
    )
    plt.title("Monthly Cohort Retention (%)", fontsize=16)
    plt.xlabel("Months Since First Purchase")
    plt.ylabel("Cohort (YYYY-MM)")
    plt.show()


    print("\\n--- Interactive Cohort Retention Heatmap (Plotly) ---")
    fig = go.Figure(data=go.Heatmap(
        z=retention_percent.values,
        x=retention_percent.columns,
        y=retention_percent.index.astype(str),
        hoverongaps=False,
        colorscale='Viridis',
        colorbar={'title': 'Retention Rate (%)'}
    ))

    fig.update_layout(
        title='Interactive Monthly Cohort Retention',
        xaxis_title="Months Since First Purchase",
        yaxis_title="Cohort (YYYY-MM)"
    )
    fig.show()


## This is a new section demonstrating Bokeh for RFM analysis

if 'rfm_agg' in locals():
    print("--- Interactive RFM Scatter Plot (Bokeh) ---")

    # Define the tooltips to show on hover
    # The @ symbol refers to columns in the data source
    tooltips = [
        ("Member ID", f"@{TRANS_ID}"),
        ("Frequency", "@frequency"),
        ("Monetary", "@monetary{$0,0.00}"), # Format as currency
        ("Recency (days)", "@recency_days")
    ]

    # Create a figure
    p = figure(
        height=400,
        width=700,
        title="RFM: Frequency vs. Monetary Value",
        x_axis_label="Frequency (Number of Transactions)",
        y_axis_label="Monetary Value (Total Spent)",
        tools="pan,wheel_zoom,box_zoom,reset,save" # Interactive tools
    )

    # Add a scatter plot glyph
    p.scatter(
        x='frequency',
        y='monetary',
        source=rfm_agg,
        size=8,
        alpha=0.6,
        legend_label="Customer"
    )

    # Add the hover tool
    p.add_tools(HoverTool(tooltips=tooltips))

    # Show the plot
    show(p)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set plotting styles for better aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# --- 1. Load Data (Corrected) ---
# Define file paths
TEST_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/kaggle_test_data.csv"
TRAIN_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/train_data.csv"
MEMBERS_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/members.csv"
USER_LOGS_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/user_logs.csv"
TRANSACTIONS_PATH = "/kaggle/input/customer-retention-datathon-brisbane-edition/transactions.csv"

# Read data into pandas DataFrames
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
# Added low_memory=False to handle the DtypeWarning
members = pd.read_csv(MEMBERS_PATH, low_memory=False)
transactions = pd.read_csv(TRANSACTIONS_PATH)
user_logs = pd.read_csv(USER_LOGS_PATH)

# --- Convert Date Columns AFTER loading ---
# This is the correct place to specify the date format
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'], format='%Y%m%d')
transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'], format='%Y%m%d')

print("Data loading and date conversion complete.")


# --- 2. Initial Data Inspection ---
print("\n--- Data Overview ---")
def inspect_df(df, name):
    print(f"\n--- {name} DataFrame ---")
    print(f"Shape: {df.shape}")
    print("\nInfo:")
    df.info()
    print("\nMissing Values:")
    print(df.isna().sum() / len(df) * 100) # Show missing values as a percentage
    print(f"\nDuplicates: {df.duplicated().sum()}")

inspect_df(train, "Train")
inspect_df(members, "Members")
inspect_df(transactions, "Transactions")
inspect_df(user_logs, "User Logs")

# --- 3. Members Analysis ---
print("\n--- Visualizing Members Data ---")

## Age ('bd') Distribution
# We'll filter for a reasonable age range (1 to 100) to get a clean plot
cleaned_bd = members[(members['bd'] > 0) & (members['bd'] < 100)]['bd']
sns.histplot(cleaned_bd, bins=50, kde=True)
plt.title('Distribution of Member Age (1-100)', fontsize=16)
plt.xlabel('Age (bd)')
plt.ylabel('Count')
plt.show()

## Gender Distribution (Interactive)
gender_counts = members['gender'].value_counts().reset_index()
fig = px.pie(gender_counts, values='count', names='gender',
             title='Member Gender Distribution',
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

# --- 4. Transactions Analysis ---
print("\n--- Visualizing Transactions Data ---")

## Daily Transaction Count
daily_transactions = transactions.groupby('transaction_date').size().reset_index(name='count')
fig = px.line(daily_transactions, x='transaction_date', y='count',
              title='Daily Transaction Volume')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

## Payment Plan Days Distribution
sns.countplot(data=transactions, x='payment_plan_days', order=transactions['payment_plan_days'].value_counts().index[:10])
plt.title('Top 10 Most Common Payment Plan Durations (in Days)', fontsize=16)
plt.xlabel('Payment Plan (Days)')
plt.ylabel('Count')
plt.show()

## Price vs. Amount Paid
sns.scatterplot(data=transactions.sample(10000), x='plan_list_price', y='actual_amount_paid', alpha=0.5)
plt.title('Plan List Price vs. Actual Amount Paid (Sample of 10k)', fontsize=16)
plt.xlabel('Plan List Price')
plt.ylabel('Actual Amount Paid')
plt.plot([0, 2000], [0, 2000], 'r--', label='Price = Paid') # Add a line for reference
plt.legend()
plt.show()

# --- 5. User Logs Analysis ---
print("\n--- Visualizing User Logs Data ---")

## Listening Habits (log scale)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.histplot(user_logs['total_secs'], ax=axes[0], bins=100, log_scale=True, color='skyblue')
axes[0].set_title('Distribution of Total Seconds Listened Per Day (Log Scale)')
sns.histplot(user_logs['num_unq'], ax=axes[1], bins=100, log_scale=True, color='salmon')
axes[1].set_title('Distribution of Unique Songs Played Per Day (Log Scale)')
plt.tight_layout()
plt.show()

## Correlation Heatmap of Listening Patterns
log_features = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']
corr = user_logs[log_features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis')
plt.title('Correlation Matrix of User Listening Habits', fontsize=16)
plt.show()

print("\nEDA script finished.")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import gc # Garbage Collector interface

print("Starting the script...")

# --- 1. Memory Optimization (Helper Function) ---
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# --- 2. Load Data ---
print("Loading data...")
train = pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/train_data.csv')
test = pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/kaggle_test_data.csv')
members = reduce_mem_usage(pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/members.csv'))
transactions = reduce_mem_usage(pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/transactions.csv'))
user_logs = reduce_mem_usage(pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/user_logs.csv'))

# --- 3. Feature Engineering ---
print("Performing feature engineering...")

# Transactions Features
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'], format='%Y%m%d')
transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'], format='%Y%m%d')
transactions['discount'] = transactions['plan_list_price'] - transactions['actual_amount_paid']
transactions['days_to_expire'] = (transactions['membership_expire_date'] - transactions['transaction_date']).dt.days
transactions['is_discount'] = (transactions['plan_list_price'] > transactions['actual_amount_paid']).astype(int)

transactions_agg = transactions.groupby('msno').agg({
    'payment_plan_days': ['mean', 'last', 'std'],
    'plan_list_price': ['mean', 'last'],
    'actual_amount_paid': ['mean', 'last'],
    'is_auto_renew': ['last'],
    'is_cancel': ['sum', 'last'],
    'discount': ['mean', 'last'],
    'days_to_expire': ['mean', 'std'],
    'is_discount': ['mean', 'sum']
})
transactions_agg.columns = ['_'.join(col).strip() for col in transactions_agg.columns.values]
transactions_agg.rename(columns={'msno_': 'msno'}, inplace=True)

# User Logs Features
user_logs['total_songs'] = user_logs['num_25'] + user_logs['num_50'] + user_logs['num_75'] + user_logs['num_985'] + user_logs['num_100']
user_logs['completion_rate'] = user_logs['num_100'] / user_logs['total_songs']
user_logs['uniqueness_rate'] = user_logs['num_unq'] / user_logs['total_songs']

user_logs_agg = user_logs.groupby('msno').agg({
    'num_25': ['mean', 'sum'], 'num_50': ['mean', 'sum'], 'num_75': ['mean', 'sum'],
    'num_985': ['mean', 'sum'], 'num_100': ['mean', 'sum'], 'num_unq': ['mean', 'sum'],
    'total_secs': ['mean', 'sum', 'std'], 'completion_rate': ['mean', 'std'], 'uniqueness_rate': ['mean']
})
user_logs_agg.columns = ['_'.join(col).strip() for col in user_logs_agg.columns.values]
user_logs_agg.rename(columns={'msno_': 'msno'}, inplace=True)

# --- 4. Merge Data ---
print("Merging all data sources...")
train = pd.merge(train, members, on='msno', how='left')
train = pd.merge(train, transactions_agg, on='msno', how='left')
train = pd.merge(train, user_logs_agg, on='msno', how='left')

test = pd.merge(test, members, on='msno', how='left')
test = pd.merge(test, transactions_agg, on='msno', how='left')
test = pd.merge(test, user_logs_agg, on='msno', how='left')

del members, transactions, user_logs, transactions_agg, user_logs_agg; gc.collect()

# --- 5. Data Cleaning & Preparation ---
print("Cleaning and preparing final dataframes...")
def clean_and_prepare(df):
    df['bd'] = df['bd'].apply(lambda x: np.nan if x <= 0 or x >= 100 else x)
    df['bd'] = df['bd'].fillna(df['bd'].median())
    df['gender'].fillna('unknown', inplace=True)
    df['registration_init_time'] = pd.to_datetime(df['registration_init_time'], format='%Y%m%d', errors='coerce')
    df['account_tenure'] = (pd.to_datetime('2017-04-30') - df['registration_init_time']).dt.days
    df.fillna(0, inplace=True)
    return df

train = clean_and_prepare(train)
test = clean_and_prepare(test)

# --- 6. Label Encoding ---
print("Encoding categorical features...")
y = train['is_churn']
train = train.drop(['is_churn', 'msno', 'registration_init_time'], axis=1)
test = test.drop(['msno', 'registration_init_time'], axis=1)

categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    # Handle new categories in test set
    test[col] = test[col].map(lambda s: s if s in le.classes_ else 'unknown')
    le_classes = le.classes_.tolist()
    if 'unknown' not in le_classes: le_classes.append('unknown')
    le.classes_ = np.array(le_classes)
    test[col] = le.transform(test[col])

# --- 7. Model Training (Cross-Validation) ---
print("Starting final model training with cross-validation...")

# These parameters are a strong starting point. Replace them with your own Optuna results.
params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'n_estimators': 2000, 'learning_rate': 0.02, 'num_leaves': 80,
    'max_depth': 7, 'min_child_samples': 100, 'subsample': 0.9,
    'colsample_bytree': 0.7, 'random_state': 42, 'n_jobs': -1
}

N_SPLITS = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
    print(f"===== FOLD {fold+1}/{N_SPLITS} =====")
    X_train, y_train = train.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = train.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(100, verbose=True)])

    test_preds += model.predict_proba(test)[:, 1] / N_SPLITS
    gc.collect()

# --- 8. Create Submission File ---
print("Creating submission file...")
submission = pd.read_csv('/kaggle/input/customer-retention-datathon-brisbane-edition/kaggle_test_data.csv')
submission['is_churn'] = test_preds
submission.to_csv('submission_final.csv', index=False)

print("Script finished successfully! `submission_final(l).csv` is ready.")

