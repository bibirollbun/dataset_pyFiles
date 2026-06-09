import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        file_size = os.path.getsize(path)
        path_len = (len(path)+20)//20*20
        if file_size >= int(1e9):
            print(path.ljust(path_len), file_size // int(1e9), 'GB')
        elif file_size >= int(1e6):
            print(path.ljust(path_len), file_size // int(1e6), 'MB')
        elif file_size >= int(1e3):
            print(path.ljust(path_len), file_size // int(1e3), 'KB')
        else:
            print(path.ljust(path_len), file_size // int(1e0), 'B')


dir_path = '/kaggle/input/home-credit-default-risk'
train  = pd.read_csv(f'{dir_path}/application_train.csv')
test   = pd.read_csv(f'{dir_path}/application_test.csv')
bureau = pd.read_csv(f'{dir_path}/bureau.csv')
bureau_balance = pd.read_csv(f'{dir_path}/bureau_balance.csv')
previous_app   = pd.read_csv(f'{dir_path}/previous_application.csv')
pos_cash       = pd.read_csv(f'{dir_path}/POS_CASH_balance.csv')
credit_card    = pd.read_csv(f'{dir_path}/credit_card_balance.csv')
insta_payments = pd.read_csv(f'{dir_path}/installments_payments.csv')


def df_dtypes(df):
    pd.set_option('display.max_colwidth', None)
    # Group columns by their data types
    df_dtypes = df.columns.to_series().groupby(df.dtypes.astype(str)).apply(list).reset_index()
    df_dtypes.columns = ['dtype', 'columns']
    # Add a column for the number of columns
    df_dtypes['# columns'] = df_dtypes['columns'].apply(len)
    # Reorder columns
    df_dtypes = df_dtypes[['dtype', '# columns', 'columns']]
    # Apply styling
    df_dtypes = df_dtypes.style.set_properties(subset=['columns'], **{'text-align': 'left'})
    return df_dtypes


N_UNIQUE_THRESHOLD = 60

def get_categorical_features(df, nunique_threshold=N_UNIQUE_THRESHOLD):
    return [feature for feature in df.columns 
            if df[feature].nunique() < nunique_threshold]

def get_numerical_features(df, nunique_threshold=N_UNIQUE_THRESHOLD):
    return [feature for feature in df.select_dtypes(include=[np.number]).columns 
            if df[feature].nunique() >= nunique_threshold]

def build_my_info_table(df, nunique_threshold=N_UNIQUE_THRESHOLD):
    # Check for an empty DataFrame
    if df is None or df.empty:
        return None
    # Convert boolean columns to integer inplace
    boolean_columns = df.select_dtypes(include='bool').columns
    df[boolean_columns] = df[boolean_columns].astype(int)
    # Select numerical columns
    numerical_features = get_numerical_features(df)
    # Initialize list to store feature-wise metrics
    metrics = []
    for idx, col in enumerate(df.columns):
        column_data = df[col]
        dtype   = column_data.dtypes
        count   = column_data.count()
        mean    = column_data.mean()   if col in numerical_features else ''
        std     = column_data.std()    if col in numerical_features else ''
        min_val = column_data.min()    if col in numerical_features else ''
        q25     = column_data.quantile(0.25) if col in numerical_features else ''
        median  = column_data.median() if col in numerical_features else ''
        q75     = column_data.quantile(0.75) if col in numerical_features else ''
        max_val = column_data.max()    if col in numerical_features else ''
        iqr     = max_val - min_val    if col in numerical_features else ''
        nunique = column_data.nunique()
        unique_values   = column_data.unique() if nunique < nunique_threshold else ''
        mode    = column_data.mode().iloc[0] if not column_data.mode().empty else ''
        mode_count      = column_data.value_counts().max() \
                                             if not column_data.value_counts().empty else ''
        mode_percentage = (round(mode_count * 100 / len(column_data), 1) 
                                             if mode_count not in ['', None] else '')
        null_count      = column_data.isnull().sum()
        null_percentage = round(column_data.isnull().mean() * 100, 1)
        # Append the calculated metrics to the list
        metrics.append({
            "#": idx,
            "column": col,
            "dtype": dtype,
            "count": count,
            "mean": round(mean, 1)   if mean    not in ['', None] else '',
            "std": round(std, 1)     if std     not in ['', None] else '',
            "min": round(min_val, 1) if min_val not in ['', None] else '',
            "25%": round(q25, 1)     if q25     not in ['', None] else '',
            "50%": round(median, 1)  if median  not in ['', None] else '',
            "75%": round(q75, 1)     if q75     not in ['', None] else '',
            "max": round(max_val, 1) if max_val not in ['', None] else '',
            "IQR": round(iqr, 1)     if iqr     not in ['', None] else '',
            "nunique": nunique,
            "unique": unique_values,
            "mode": mode,
            "mode #": mode_count,
            "mode %": mode_percentage,
            "null #": null_count,
            "null %": null_percentage,
        })
    # Convert metrics list to DataFrame
    df_info = pd.DataFrame(metrics)
    # Ensure sorting by dtype is stable
    df_info = df_info.sort_values(by='dtype').reset_index(drop=True)
    return df_info


def fillna_and_replace_inf(df):
    # Select numerical and categorical columns once
    numerical_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(exclude=[np.number]).columns
    # Fill missing values and replace infinities for numerical features
    for feature in numerical_features:
        df[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
        median = df[feature].median()
        df[feature].fillna(median, inplace=True)
    # Fill missing values for categorical features
    for feature in categorical_features:
        if list(df[feature].mode()):
            mode = df[feature].mode()[0]
            df[feature].fillna(mode, inplace=True)
    return df

from sklearn.preprocessing import LabelEncoder

def encode_str_features(df):
    # Get list of categorical features
    categorical_features = get_categorical_features(df)
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Iterate over each categorical feature
    for feature in categorical_features:
        # Convert feature values to string (if not already) and encode with LabelEncoder
        df[feature] = label_encoder.fit_transform(df[feature].astype(str)).astype(np.int8)
    return df

def split_data_X_y(df, target_feature):
    y = df[target_feature]
    X = df.drop(columns=[target_feature])
    return X, y

def drop_id_feature(df, id_col='id'):
    df_id = df[id_col]
    df = df.drop(columns=[id_col])
    return df, df_id


def plot_bar_chart(df, x, y, xlabel, ylabel, title, xmin=None, xmax=None, palette='deep'):
    if df.shape[0] == 0:
        return
    size = (12, df.shape[0] / 4 + 1)
    plt.figure(figsize=size)
    sns.barplot(y=df[y], x=df[x], palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xmin is None or xmax is None:
        xrange = (df[x].max() - df[x].min()) * .1
        if xmin is None:
            xmin = max(0, df[x].min() - xrange) if df[x].min() >= 0 else df[x].min() - xrange
        if xmax is None:
            xmax = min(0, df[x].max() + xrange) if df[x].max() <= 0 else df[x].max() + xrange
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.show()

def plot_line_chart(df, x, y, xlabel, ylabel, title, figsize=(12, 4)):
    if df.shape[0] == 0:
        return
    plt.figure(figsize=figsize)
    plt.plot(df[x], df[y], marker='o', label=ylabel)
    plt.xticks(df[x])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def extract_date_features(df, date_columns):
    for feature in date_columns:
        df[feature] = pd.to_datetime(df[feature], errors='coerce')
        df[feature + '_year']         = df[feature].dt.year.astype('int64')
        df[feature + '_month']        = df[feature].dt.month.astype('int64')
        df[feature + '_day']          = df[feature].dt.day.astype('int64')
        df[feature + '_weekday']      = df[feature].dt.weekday.astype('int64')
        df[feature + '_weekofyear']   = df[feature].dt.isocalendar().week.astype('int64')
        df[feature + '_quarter']      = df[feature].dt.quarter.astype('int64')
        df[feature + '_isweekend']    = df[feature + '_weekday'].isin([5, 6]).astype('int64')
        df[feature + '_isleapyear']   = df[feature].dt.is_leap_year.astype('int64')
        df[feature + '_ismonthend']   = df[feature].dt.is_month_end.astype('int64')
        df[feature + '_ismonthstart'] = df[feature].dt.is_month_start.astype('int64')
        df[feature + '_season']       = (df[feature].dt.month % 12 // 3 + 1).map({
                                          1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'
                                         }).astype('object')
    return df

def extract_time_features(df, time_columns):
    for feature in time_columns:
        df[feature] = pd.to_datetime(df[feature], errors='coerce')
        df[feature + '_hour']         = df[feature].dt.hour.astype('int64')
        df[feature + '_minute']       = df[feature].dt.minute.astype('int64')
        df[feature + '_second']       = df[feature].dt.second.astype('int64')
        df[feature + '_timeofday']    = pd.cut(df[feature].dt.hour, 
                                               bins=[0, 6, 12, 18, 24], 
                                               labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                               right=False).astype('object')
        df[feature + '_isnoon']       = (df[feature].dt.hour == 12).astype('int64')
        df[feature + '_ismidnight']   = (df[feature].dt.hour == 0).astype('int64')
        df[feature + '_minutegroup']  = pd.cut(df[feature].dt.minute, 
                                               bins=[0, 15, 30, 45, 60], 
                                               labels=['Early', 'Mid', 'Late', 'End'],
                                               right=False).astype('object')
    return df


train.head()


test.head()


bureau.head()


bureau_balance.head()


previous_app.head()


pos_cash.head()


credit_card.head()


insta_payments.head()


# Aggregate bureau_balance
bureau_balance_agg = bureau_balance.groupby("SK_ID_BUREAU").agg({
    "MONTHS_BALANCE": ["count", "min", "max"]
}).reset_index()
bureau_balance_agg.columns = ["_".join(col) for col in bureau_balance_agg.columns]

# Merge bureau_balance with bureau
bureau = bureau.merge(bureau_balance_agg, left_on="SK_ID_BUREAU", right_on="SK_ID_BUREAU_", how="left").drop("SK_ID_BUREAU_", axis=1)


# Aggregate bureau
bureau_agg = bureau.groupby("SK_ID_CURR").agg({
    "SK_ID_BUREAU": "count",
    "CREDIT_ACTIVE": "nunique",
    "AMT_CREDIT_SUM": "sum"
}).reset_index()
bureau_agg.columns = ["SK_ID_CURR", "bureau_count", "credit_active_unique", "total_credit_sum"]

# Merge bureau with main datasets
train = train.merge(bureau_agg, on="SK_ID_CURR", how="left")
test  = test.merge(bureau_agg,  on="SK_ID_CURR", how="left")


# Load and process previous_application
previous_app_agg = previous_app.groupby("SK_ID_CURR").agg({
    "SK_ID_PREV": "count",
    "AMT_APPLICATION": "mean",
    "AMT_CREDIT": "sum"
}).reset_index()
previous_app_agg.columns = ["SK_ID_CURR", "prev_app_count", "avg_amt_app", "total_amt_credit"]

# Merge previous_application with main datasets
train = train.merge(previous_app_agg, on="SK_ID_CURR", how="left")
test  = test.merge(previous_app_agg, on="SK_ID_CURR", how="left")


# Load and process POS_CASH_balance
pos_cash_agg = pos_cash.groupby("SK_ID_CURR").agg({
    "SK_ID_PREV": "count",
    "MONTHS_BALANCE": "mean"
}).reset_index()
pos_cash_agg.columns = ["SK_ID_CURR", "pos_cash_count", "avg_months_balance"]

# Merge POS_CASH_balance with main datasets
train = train.merge(pos_cash_agg, on="SK_ID_CURR", how="left")
test  = test.merge(pos_cash_agg, on="SK_ID_CURR", how="left")


# Load and process credit_card_balance
credit_card_agg = credit_card.groupby("SK_ID_CURR").agg({
    "SK_ID_PREV": "count",
    "AMT_BALANCE": "mean"
}).reset_index()
credit_card_agg.columns = ["SK_ID_CURR", "credit_card_count", "avg_credit_balance"]

# Merge credit_card_balance with main datasets
train = train.merge(credit_card_agg, on="SK_ID_CURR", how="left")
test  = test.merge(credit_card_agg, on="SK_ID_CURR", how="left")


# Load and process installments_payments
insta_payments_agg = insta_payments.groupby("SK_ID_CURR").agg({
    "SK_ID_PREV": "count",
    "AMT_PAYMENT": "sum",
    "DAYS_ENTRY_PAYMENT": "mean"
}).reset_index()
insta_payments_agg.columns = ["SK_ID_CURR", "installments_count", "total_amt_payment", "avg_days_entry_payment"]

# Merge installments_payments with main datasets
train = train.merge(insta_payments_agg, on="SK_ID_CURR", how="left")
test  = test.merge(insta_payments_agg, on="SK_ID_CURR", how="left")


print(f'train.shape: {train.shape}')
print(f'test.shape : {test.shape}')


print(set(train) - set(test))
print(set(test) - set(train))


target_feature = list(set(train.columns)-set(test.columns))[0]
target_feature


df_dtypes(train)


categorical_features = get_categorical_features(train)
numerical_features   = get_numerical_features(train)
print(f'categorical_features: {len(categorical_features)}\n{categorical_features}')
print(f'numerical_features:   {len(numerical_features)}\n{numerical_features}')


info_table = build_my_info_table(train)


info_table[info_table['dtype'].astype(str).str.startswith(('object', 'datetime'))]


info_table[info_table['dtype'].astype(str).str.startswith(('float'))]


info_table[info_table['dtype'].astype(str).str.startswith(('int'))]


nan_df = info_table[info_table['null %'] >= 10][['column', 'null %']].sort_values(by='null %')
plot_bar_chart(nan_df, x='null %', y='column', 
               xlabel='Null Percentage %', ylabel='Feature', 
               title='Null Percentage in each Feature', 
               xmin=0, xmax=100, palette='coolwarm')


dropped_nan = set(nan_df[nan_df['null %'] > 20]['column'])
train = train.drop(columns=list(dropped_nan - set([target_feature])))
test  = test.drop(columns=list(dropped_nan - set([target_feature])))


train = fillna_and_replace_inf(train)
test  = fillna_and_replace_inf(test)


for feature in []:
    train = extract_date_features(train, [feature])
    test = extract_date_features(test, [feature])
    train = extract_time_features(train, [feature])
    test = extract_time_features(test, [feature])
    train[feature] = train[feature].astype(int) / 10**18
    test[feature] = test[feature].astype(int) / 10**18


df_dtypes(train)


df_dtypes(test)


TUNE_DATASET_LEN = int(1e3)
n_samples = min(int(1e6), train.shape[0])
original_train = train.sample(n_samples)
original_train, train = train, original_train


info_table = build_my_info_table(test)
#info_table


nan_df = info_table[info_table['null %'] >= 10][['column', 'null %']].sort_values(by='null %')
plot_bar_chart(nan_df, x='null %', y='column', 
               xlabel='Null Percentage %', ylabel='Feature', 
               title='Null Percentage in each Feature', 
               xmin=0, xmax=100, palette='coolwarm')


dropped_nan = set(nan_df[nan_df['null %'] > 20]['column'])
train = train.drop(columns=list(dropped_nan - set([target_feature])))
test  = test.drop(columns=list(dropped_nan - set([target_feature])))


train = encode_str_features(train)
test  = encode_str_features(test)


train, _ = drop_id_feature(train, 'SK_ID_CURR')
test, test_id = drop_id_feature(test, 'SK_ID_CURR')


train = train.drop(columns=train.select_dtypes(exclude=[np.number]).columns)
test  = test.drop(columns=test.select_dtypes(exclude=[np.number]).columns)


info_table = build_my_info_table(train)
mode_df = info_table[info_table['mode %'] >= 90][['column', 'mode %']].sort_values(by='mode %')
plot_bar_chart(mode_df, x='mode %', y='column', 
               xlabel='Mode Percentage %', ylabel='Feature', 
               title='Mode Percentage in each Feature', 
               xmin=90, xmax=100, palette='coolwarm')


dropped_mode = set(mode_df[mode_df['mode %'] > 94]['column'])
train = train.drop(columns=list(dropped_mode - set([target_feature])))
test = test.drop(columns=list(dropped_mode - set([target_feature])))


categorical_features = get_categorical_features(train)
numerical_features   = get_numerical_features(train)
print(f'categorical_features: {len(categorical_features)}\n{categorical_features}')
print(f'numerical_features:   {len(numerical_features)}\n{numerical_features}')


print(f'train.shape: {train.shape}')
print(f'test.shape : {test.shape}')


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


from matplotlib.colors import LinearSegmentedColormap

def my_heatmap(df, size, cmap, cbar_kws, font_size):
    plt.figure(figsize=size)
    sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap=cmap, center=0, 
                cbar_kws=cbar_kws, annot_kws={"size": font_size})
    plt.title('Correlation Heatmap')
    plt.show()
    
def plot_heatmap(df, size_factor=1/2):
    df = df.select_dtypes(include=[np.number])
    height = int(len(df.columns) * size_factor)
    if not height:
        return
    font_size = max(min(12, 119 // height), 8)
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        ['blue', 'lightblue', 'white', 'lightcoral', 'red'],
        N=5
    )
    cbar_kws = {'ticks': [-1, -.5, 0, .5, 1]}
    my_heatmap(df, size=(height+1, height+1), cmap=cmap, cbar_kws=cbar_kws, font_size=font_size)


from sklearn.decomposition          import PCA

def plot_labeled_data(X, labels, figsize=(12, 4), n_components=2, n_col=1):
    # Check if n_components is greater than the number of columns in X
    if n_components > X.shape[1]:
        for i in range(X.shape[1], n_components):
            X[f'x{i+1}'] = 1  # Add dummy columns if needed to match n_components

    # Perform PCA to reduce the dimensionality of X to 2D
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Create a DataFrame with the PCA-transformed data and the labels
    df = pd.DataFrame(X_pca, columns=[f'x{i+1}' for i in range(n_components)])
    for name in labels:
        df[f'Applying\n{name}'] = labels[name]  # Add labels as new columns for visualization

    # Plot the PCA-transformed data with labels using plot_features function
    plot_features(df,
                  plot_funcs=[my_scatterplot],  # Assuming my_scatterplot is defined elsewhere
                  width_ratios=[figsize[0]] * n_col,
                  height_ratios=[figsize[1]],
                  n_col=n_col,
                  primary_cols=n_components)


def replace_rare_categories(df, feature, class_name, threshold_percent):
    # Calculate the threshold count based on the percentage
    threshold_count = len(df) * (threshold_percent / 100)
    # Get the value counts of the column
    value_counts = df[feature].value_counts()
    # Find values that are below the threshold count
    rare_categories = value_counts[value_counts < threshold_count].index
    # Replace rare categories with 'Other'
    df[feature] = df[feature].apply(lambda x: class_name if x in rare_categories else x)
    return df


for feature in ['CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']:
    plot_categorical_features(original_train[[feature]])
    original_train = replace_rare_categories(original_train, feature, -1, threshold_percent=2.0)
    train          = replace_rare_categories(train, feature, -1, threshold_percent=2.0)
    plot_categorical_features(original_train[[feature]])


import torchmetrics

def evaluate_categorical_model(y_test, y_pred, n_classes=2):
    accuracy  = torchmetrics.functional.accuracy      (y_pred, y_test, task="multiclass", 
                                                       num_classes=n_classes)
    precision = torchmetrics.functional.precision     (y_pred, y_test, task="multiclass", 
                                                       num_classes=n_classes, average="macro")
    recall    = torchmetrics.functional.recall        (y_pred, y_test, task="multiclass", 
                                                       num_classes=n_classes, average="macro")
    f1        = torchmetrics.functional.f1_score      (y_pred, y_test, task="multiclass", 
                                                       num_classes=n_classes, average="macro")
    cohen     = torchmetrics.functional.cohen_kappa   (y_pred, y_test, task="multiclass", 
                                                       num_classes=n_classes)
    jaccard   = torchmetrics.functional.jaccard_index (y_pred, y_test, task="multiclass", 
                                                       num_classes=n_classes, average="macro")
    fbeta     = torchmetrics.functional.fbeta_score   (y_pred, y_test, beta=0.5, task="multiclass", 
                                                       num_classes=n_classes, average="macro")
    
    result = {
        'Accuracy': accuracy.item(),
        'Precision': precision.item(),
        'Recall': recall.item(),
        'F1': f1.item(),
        'Cohen': cohen.item(),
        'Jaccard': jaccard.item(),
        'FBeta': fbeta.item(),
    }
    return result


import torch.nn as nn

# Define a simple DNN
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_size):
        super(SimpleDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.output_size = output_size

    def forward(self, x):
        x = self.model(x)
        if self.output_size == 1:
            return nn.functional.sigmoid(x)
        return nn.functional.softmax(x, dim=1)

# Define a DNN with BatchNorm and Dropout
class AdvancedDNN(nn.Module):
    def __init__(self, input_dim, output_size):
        super(AdvancedDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )
        self.output_size = output_size

    def forward(self, x):
        x = self.model(x)
        if self.output_size == 1:
            return nn.functional.sigmoid(x)
        return nn.functional.softmax(x, dim=1)

# Define a simple 1D CNN model for tabular data
class CNN1D(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CNN1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_dim // 4), 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.output_size = output_size

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.output_size == 1:
            return nn.functional.sigmoid(x)
        return nn.functional.softmax(x, dim=1)

# Define a Hybrid CNN-DNN model
class HybridCNN_DNN(nn.Module):
    def __init__(self, input_dim, output_size):
        super(HybridCNN_DNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_dim // 2), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.output_size = output_size

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.output_size == 1:
            return nn.functional.sigmoid(x)
        return nn.functional.softmax(x, dim=1)

# Function to create categorical models
def get_categorical_models(input_size, output_size):
    categorical_models = {
        "SimpleDNN":     SimpleDNN(input_size, output_size),
        "AdvancedDNN":   AdvancedDNN(input_size, output_size),
        "CNN1D":         CNN1D(input_size, output_size),
        "HybridCNN_DNN": HybridCNN_DNN(input_size, output_size)
    }
    return categorical_models


import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, optimizer_method, 
                epochs, lr=0.001, val_loader=None, clip_value=1.0):

    criterion = nn.CrossEntropyLoss()
    optimizer = {
        'SGD':     optim.SGD(model.parameters(),     lr=lr),
        'Adam':    optim.Adam(model.parameters(),    lr=lr),
        'RMSprop': optim.RMSprop(model.parameters(), lr=lr),
        'NAdam':   optim.NAdam(model.parameters(),   lr=lr),
    }[optimizer_method]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(train_loader)
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.5f}")
        
        # Validation Step
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.5f}")

        scheduler.step()

def predict(model, X_test):
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        AL = model(X_test)
        pred = torch.argmax(AL, dim=1)
    return pred

def run_models(models, evaluate_model, X_train, X_test, y_train, y_test, batch_size=128, epochs=8):

    evaluation_results, y_predictions = {}, {}
    n_classes = len(set(y_train.unique()) | set(y_test.unique()))

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test.to_numpy(),  dtype=torch.float32)
    
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test.to_numpy(),  dtype=torch.long)

    y_train_one_hot = nn.functional.one_hot(y_train_tensor, num_classes=n_classes).float()
    y_test_one_hot  = nn.functional.one_hot(y_test_tensor,  num_classes=n_classes).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_one_hot)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for model_name, model in models.items():
        print(f'Training Model: {model_name}...')
        print(model)
        begin_time = time.time()
        train_model(model, train_loader, epochs=epochs, optimizer_method='Adam')
        y_pred = predict(model, X_test)        
        try:
            evaluation_results[model_name] = evaluate_model(y_test_tensor, y_pred, n_classes)
            y_predictions[model_name] = y_pred
        except Exception as e:
            print(f'Error occurred while running {model_name}: {e}')
        end_time = time.time()

        duration = round((end_time - begin_time) / 60, 2)
        print(f'{model_name} Model ran in {duration} minutes')
        print('------------------------------------')
    # Convert evaluation results to a DataFrame and reset index
    evaluation_results = pd.DataFrame(evaluation_results).T.reset_index()
    evaluation_results.rename(columns={'index': 'Model'}, inplace=True)
    return evaluation_results, y_predictions


from sklearn.model_selection import train_test_split

def split_data_train_test(df, target_feature, test_size=None):
    if test_size is None:
        test_size = (1 / np.log10(len(df)) / np.log2(np.log10(len(df))))
        test_size = max(0.01, min(0.5, test_size))
    y = df[target_feature]
    X = df.drop(columns=[target_feature])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def get_best_model(evaluation_results, models, metric, ascending=True):
    # Sort the evaluation results DataFrame based on the specified metric
    sorted_results = evaluation_results.sort_values(by=[metric], ascending=ascending)
    # Get the top result (best model) based on the sorted evaluation results
    top_result = sorted_results.head(1)
    # Extract the name of the best model
    best_model_name = top_result['Model'].iloc[0]
    # Retrieve the best model instance from the models dictionary
    best_model = models[best_model_name]
    return best_model


def plot_models_with_evaluation_metrics(models, models_result, metrics, title=''):
    # List of metrics where lower values indicate better performance
    ascending_metrics = ['Mean Absolute Error', 'Mean Squared Error', 'Max Error', 
                         'Mean Absolute Percentage Error', 'Median Absolute Error']
    # Iterate over each metric
    for metric in metrics:
        # Plot a bar chart comparing models based on the current metric
        plot_bar_chart(models_result, x=metric, y='Model', xlabel=metric, ylabel='Model', 
                       title=f'{title}Models Comparison using {metric} metric')
        # Determine the best model for the current metric
        best_model = get_best_model(models_result, models, metric, 
                                    ascending=(metric in ascending_metrics))
        # Print the name of the best model for the current metric
        best_model_name = best_model.__class__.__name__
        print(f'Best Model of the Models using {metric} is: {best_model_name} with value '
              f'{models_result[models_result["Model"] == best_model_name][metric].values}')


from sklearn.decomposition   import PCA

def plot_predictions(X, y_predictions, figsize=(6, 3), n_components=2, n_col=1):
    if n_components > X.shape[1]:
        for i in range(X.shape[1], n_components):
            X[f'x{i+1}'] = 1
    # Use PCA to reduce the dimensionality of the data to 2D for visualization
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    # Create a DataFrame with the PCA-transformed data and the labels
    df = pd.DataFrame(X_pca, columns=[f'x{i+1}' for i in range(n_components)])
    for name in y_predictions:
        df[f'y_prediction using \n{name}'] = y_predictions[name]
    plot_features(df, 
                  plot_funcs=[my_scatterplot], 
                  width_ratios=[figsize[0]]*n_col, 
                  height_ratios=[figsize[1]], 
                  n_col=n_col, 
                  primary_cols=n_components)


train_model1 = train.copy()
df_dtypes(train_model1)


baseline_models = get_categorical_models(input_size=train_model1.shape[1]-1, output_size=2)
evaluate_model_func = evaluate_categorical_model
#for i, j in baseline_models.items(): print(i, j)


X_train, X_test, y_train, y_test = split_data_train_test(train_model1, target_feature)
evaluation_results1, y_predictions = run_models(baseline_models, evaluate_model_func, X_train, X_test, y_train, y_test)


evaluation_results1


plot_predictions(X_test, y_predictions, figsize=(8, 4), n_col=1)


metrics = sorted(set(evaluation_results1.columns)-set(['Model']))
plot_models_with_evaluation_metrics(baseline_models, evaluation_results1, metrics, title='Baseline ')


best_model1 = get_best_model(evaluation_results1, baseline_models, 'F1', ascending=False)
print('Best Model of Baseline Models is:', best_model1.__class__.__name__)


del train_model1, X_train, X_test, y_train, y_test


from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(df):
    # Identify the categorical features in the dataframe
    categorical_features = get_categorical_features(df)
    # If the target feature is among the categorical features, remove it from the list
    if target_feature in categorical_features:
        categorical_features.remove(target_feature)
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse=False, drop='first')
    # Fit the encoder on the categorical features and transform them
    encoded_features = encoder.fit_transform(df[categorical_features])
    # Get the new column names for the encoded features
    new_columns = encoder.get_feature_names_out(categorical_features)
    # Create a new dataframe with the encoded features, using the original dataframe's index
    encoded_df = pd.DataFrame(encoded_features, columns=new_columns, index=df.index)
    # Drop the original categorical columns from the dataframe
    df.drop(columns=categorical_features, inplace=True)
    # Add the encoded features to the original dataframe
    for feature in encoded_df.columns:
        df[feature] = encoded_df[feature]
    return df


from itertools import combinations

def create_interaction_features(df, numerical_features=None):
    # If numerical_features is not provided, get all numerical features from the dataframe
    if numerical_features is None:
        numerical_features = get_numerical_features(df)
    # Generate all possible combinations of numerical features (taken 2 at a time)
    for combo in combinations(numerical_features, 2):
        # Create a new feature by multiplying the two features in each combination
        df[f'{combo[0]}_{combo[1]}_interaction'] = df[combo[0]] * df[combo[1]]
    return df


from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_features(df, degree=2, numerical_features=None):
    # If numerical_features is not provided, get all numerical features from the dataframe
    if numerical_features is None:
        numerical_features = get_numerical_features(df)
    # Initialize the PolynomialFeatures transformer with the specified degree
    poly = PolynomialFeatures(degree, include_bias=False)
    # Fit and transform the numerical features to create polynomial features
    poly_features = poly.fit_transform(df[numerical_features])
    # Get the names of the new polynomial features
    new_columns = poly.get_feature_names_out(numerical_features)
    # Create a DataFrame for the polynomial features with the new column names
    poly_df = pd.DataFrame(poly_features, columns=new_columns)
    # Add the polynomial features to the original dataframe
    for feature in poly_df.columns:
        df[feature] = poly_df[feature]
    return df


def bin_numerical_features(df, bins=10, numerical_features=None):
    # If numerical_features is not provided, get all numerical features from the dataframe
    if numerical_features is None:
        numerical_features = get_numerical_features(df)
    # Iterate through each numerical feature and bin it
    for feature in numerical_features:
        # Create a new column with '_binned' suffix to store the binned values
        df[feature + '_binned'] = pd.cut(df[feature], bins=bins, labels=False)
    return df


def target_encode_features(df, target_feature, categorical_features=None):
    # If categorical_features is not provided, get all categorical features from the dataframe
    if categorical_features is None:
        categorical_features = get_categorical_features(df)
    # Iterate through each categorical feature and perform target encoding
    for feature in categorical_features:
        # Calculate the mean of the target variable for each category in the current feature
        target_mean = df.groupby(feature)[target_feature].mean()
        # Map the target mean values back to the dataframe based on the current feature
        df[feature + '_target_enc'] = df[feature].map(target_mean)
    return df


from sklearn.preprocessing import StandardScaler

def scale_features(df, numerical_features=None):
    # If numerical_features is not provided, get all numerical features from the dataframe
    if numerical_features is None:
        numerical_features = get_numerical_features(df)
    # Initialize StandardScaler
    scaler = StandardScaler()
    # Iterate through each numerical feature and scale it
    for feature in numerical_features:
        # Fit and transform the current numerical feature using StandardScaler
        df[feature] = scaler.fit_transform(df[[feature]])
    return df


def log_transform_features(df, numerical_features=None):
    # If numerical_features is not provided, get all numerical features from the dataframe
    if numerical_features is None:
        numerical_features = get_numerical_features(df)
    # Iterate through each numerical feature and apply log transformation
    for feature in numerical_features:
        # Identify indices where the feature values are negative
        negative_indices = df[feature] < 0
        # Apply log transformation to the absolute values of the feature
        df[feature] = np.log1p(np.abs(df[feature]))
        # Restore negative signs to the transformed values
        df[feature] = np.where(negative_indices, -df[feature], df[feature])
    return df


def get_skewed_features(df, threshold=0.25):
    # Get all numerical features from the dataframe
    numerical_features = get_numerical_features(df)
    # Calculate skewness factor for each numerical feature
    skew_df = df[numerical_features].apply(lambda x: x.skew())
    # Sort skewness values in descending order
    skew_df = skew_df.sort_values(ascending=False)
    # Reset index to get a DataFrame with Feature and SkewFactor columns
    skew_df = skew_df.reset_index()
    skew_df.columns = ['Feature', 'SkewFactor']
    # Identify skewed features based on the absolute value of skewness factor
    skewed_features = list(skew_df[abs(skew_df['SkewFactor']) > threshold]['Feature'])
    # Identify non-skewed features as the complement of skewed features
    non_skewed_features = list(set(numerical_features) - set(skewed_features))
    # Return lists of skewed and non-skewed features, and the DataFrame with skewness factors
    return skewed_features, non_skewed_features, skew_df


def plot_features_correlation(df, features, target_feature, plot_kinds, 
                              title='', step=6, height=3, aspect=1):
    # Sample the dataframe to a maximum of 1000 rows for efficient plotting
    df = df.sample(min(1000, df.shape[0]))
    # Iterate over features in steps of 'step'
    for i in range(0, len(features), step):
        # Iterate over each plot type in plot_kinds
        for plot_kind in plot_kinds:
            # Create a pairplot for the current set of features and target feature
            pairplot = sns.pairplot(df, x_vars=features[i:i+step], y_vars=[target_feature], 
                                    kind=plot_kind, height=height, aspect=aspect)            
            # Set title for the pairplot
            pairplot.fig.suptitle(title + f' using {plot_kind} Plot', y=1.025)
            # Adjust layout for better presentation
            pairplot.fig.tight_layout()


def get_target_correlations(df, target_feature):
    numerical_features  = df.select_dtypes(include=[np.number]).columns
    correlation_matrix  = df[numerical_features].corr()
    target_correlations = correlation_matrix[target_feature].sort_values(ascending=False)
    target_correlations = target_correlations.reset_index()
    target_correlations.columns = ['Feature', 'TargetCorrelation']
    return target_correlations


train_model2 = train.copy()
df_dtypes(train_model2)


#df_cpy = create_interaction_features(train_model2.copy())
#df_dtypes(df_cpy)


#df_cpy = create_polynomial_features(train_model2.copy())
#df_dtypes(df_cpy)


df_cpy = target_encode_features(train_model2.copy(), target_feature)
df_dtypes(df_cpy)


df_cpy = one_hot_encoding(train_model2.copy())
df_dtypes(df_cpy)


df_cpy = bin_numerical_features(train_model2.copy())
df_dtypes(df_cpy)


skewed_features, non_skewed_features, skew_df = get_skewed_features(train_model2)
print(f'skewed_features:     {len(skewed_features)}\n{skewed_features}')
print(f'non_skewed_features: {len(non_skewed_features)}\n{non_skewed_features}')


plot_bar_chart(skew_df, x='SkewFactor', y='Feature', 
               xlabel='Skew Factor', ylabel='Feature', 
               title='Skew Factor in each Feature', palette='coolwarm')


n_col = 7
for i in range(0, len(skewed_features), n_col):
    j = min(i+n_col, len(skewed_features))
    plot_features_correlation(train_model2,
                              skewed_features[i:j],
                              target_feature,
                              title=f'Correlation of Skewed Features with {target_feature}',
                              step = n_col,
                              plot_kinds=['scatter', 'hist'])
    plot_features_correlation(log_transform_features(train_model2.copy(), skewed_features[i:j]),
                              skewed_features[i:j],
                              target_feature,
                              title=f'Correlation of Transformed Skewed Features with {target_feature}',
                              step = n_col,
                              plot_kinds=['scatter', 'hist'])


n_col = 7
for i in range(0, len(non_skewed_features), n_col):
    j = min(i+n_col, len(non_skewed_features))
    plot_features_correlation(train_model2,
                              non_skewed_features[i:j],
                              target_feature,
                              title=f'Correlation of Non-Skewed Features with {target_feature}',
                              step = n_col,
                              plot_kinds=['scatter', 'hist'])
    plot_features_correlation(scale_features(train_model2.copy(), non_skewed_features[i:j]),
                              non_skewed_features[i:j],
                              target_feature,
                              title=f'Correlation of Transformed Non-Skewed Features with {target_feature}',
                              step = n_col,
                              plot_kinds=['scatter', 'hist'])


train_model2 = log_transform_features(train_model2, skewed_features)
train_model2 = scale_features(train_model2, non_skewed_features)
#train_model2 = create_interaction_features(train_model2)
#train_model2 = create_polynomial_features(train_model2)
#train_model2 = target_encode_features(train_model2, target_feature)
train_model2 = one_hot_encoding(train_model2)
train_model2 = bin_numerical_features(train_model2)


df_dtypes(train_model2)


baseline_models = get_categorical_models(input_size=train_model2.shape[1]-1, output_size=2)
evaluate_model_func = evaluate_categorical_model
#for i, j in baseline_models.items(): print(i, j)


X_train, X_test, y_train, y_test = split_data_train_test(train_model2, target_feature)
evaluation_results2, y_predictions = run_models(baseline_models, evaluate_model_func, X_train, X_test, y_train, y_test)


evaluation_results2


plot_predictions(X_test, y_predictions, figsize=(8, 4), n_col=1)


metrics = sorted(set(evaluation_results2.columns)-set(['Model']))
plot_models_with_evaluation_metrics(baseline_models, evaluation_results2, metrics, title='Enhanced Features ')


best_model2 = get_best_model(evaluation_results2, baseline_models, 'F1', ascending=False)
print('Best Model of Enhanced Features Models is:', best_model2.__class__.__name__)


del train_model2, X_train, X_test, y_train, y_test


from sklearn.feature_selection   import VarianceThreshold

def variance_threshold_selector(X, threshold=None):
    # If threshold is not provided, set a default
    if threshold is None:
        threshold = 0.01
    # Capture the column names of X
    feature_names = list(X.columns)
    # Apply VarianceThreshold
    selector = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    X_selected = selector.fit_transform(X)
    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)
    # Filter the feature names based on selected indices
    X_selected_features = [feature_names[idx] for idx in selected_indices]
    return X_selected_features


from sklearn.feature_selection   import SelectKBest, SelectPercentile, \
                                        f_classif, f_regression, \
                                        mutual_info_classif, mutual_info_regression

def select_k_best_features(X, y, score_func, selection_method):
    if min(X.shape) < 2:
        return X.columns, [1] * len(X.columns)

    valid_score_funcs = [f_classif, f_regression, mutual_info_classif, mutual_info_regression]
    valid_selection_methods = [SelectKBest, SelectPercentile]

    # Validate score_func
    if score_func not in valid_score_funcs:
        raise ValueError('Invalid score_func parameter value')
    # Validate selection_method
    if selection_method not in valid_selection_methods:
        raise ValueError('Invalid selection_method parameter value')

    # Instantiate selector
    if selection_method is SelectPercentile:
        selector = selection_method(score_func, percentile=100)
    else:
        selector = selection_method(score_func, k=X.shape[1])
    # Fit selector and transform X
    X_selected = selector.fit_transform(X, y)
    # Filter selected features based on scores
    selected_features_mask = (selector.scores_ > 0) & (selector.scores_ < np.inf)
    X_selected_features = X.columns[selected_features_mask]
    return X_selected_features, selector.scores_[selected_features_mask]


from sklearn.feature_selection   import SelectFpr, SelectFdr, SelectFwe, \
                                        f_classif, f_regression

def select_features_by_significance(X, y, score_func, selection_method):
    if min(X.shape) < 2:
        return X.columns, [1] * len(X.columns)

    valid_score_funcs = [f_classif, f_regression]
    valid_selection_methods = [SelectFpr, SelectFdr, SelectFwe]

    # Validate score_func
    if score_func not in valid_score_funcs:
        raise ValueError('Invalid score_func parameter value')
    # Validate selection_method
    if selection_method not in valid_selection_methods:
        raise ValueError('Invalid selection_method parameter value')

    # Instantiate selector
    selector = selection_method(score_func, alpha=0.99)
    # Fit selector and transform X
    X_selected = selector.fit_transform(X, y)
    # Compute inverse of p-values
    pvalues_inverse = 1 / (selector.pvalues_ + 0.0001)
    # Filter selected features based on inverse p-values
    selected_features_mask = (pvalues_inverse > 0) & (pvalues_inverse < np.inf)
    X_selected_features = X.columns[selected_features_mask]
    return X_selected_features, pvalues_inverse[selected_features_mask]


from sklearn.linear_model        import LinearRegression, LogisticRegression, \
                                        Ridge, RidgeClassifier
from sklearn.tree                import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble            import RandomForestClassifier, RandomForestRegressor, \
                                        GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection   import RFE, SelectFromModel


def select_features_by_model(X, y, model_type, selection_method):
    if min(X.shape) < 2:
        return X.columns, [1] * len(X.columns)

    valid_model_types = [LinearRegression, LogisticRegression, 
                         RidgeClassifier, Ridge,
                         DecisionTreeClassifier, DecisionTreeRegressor,
                         RandomForestClassifier, RandomForestRegressor,
                         GradientBoostingClassifier, GradientBoostingRegressor]
    valid_selection_methods = [RFE, SelectFromModel]

    # Validate model_type
    if model_type not in valid_model_types:
        raise ValueError('Invalid model_type parameter value')
    # Validate selection_method
    if selection_method not in valid_selection_methods:
        raise ValueError('Invalid selection_method parameter value')

    feature_scores = np.zeros(len(X.columns))

    # Perform feature selection using RFE
    if selection_method == RFE:
        # Instantiate RFE with the given model and set to select all features initially
        rfe = RFE(estimator=model_type(), n_features_to_select=1)
        # Fit RFE to the data and transform X
        X_selected = rfe.fit_transform(X, y)
        # Get feature rankings
        feature_rankings = rfe.ranking_
        # Convert rankings to importance scores (higher score means more important)
        feature_scores = max(feature_rankings) - feature_rankings + 1
    
    # Perform feature selection using SelectFromModel
    elif selection_method == SelectFromModel:
        # Instantiate and fit the model
        model = model_type()
        model.fit(X, y)
        # Instantiate SelectFromModel with the trained model
        selector = SelectFromModel(model, prefit=True)
        # Transform X using the selector
        X_selected = selector.transform(X)

        # Check for feature importances in the model
        if hasattr(model, 'feature_importances_'):
            feature_scores = model.feature_importances_
        # Check for coefficients in the model
        elif hasattr(model, 'coef_'):
            feature_scores = model.coef_

        # Flatten the coefficients if they are in 2D
        if len(feature_scores.shape) == 2:
            feature_scores = np.mean(np.abs(feature_scores), axis=0)

    # Filter selected features based on scores
    selected_features_mask = (feature_scores >= 0) & (feature_scores < np.inf)
    X_selected_features = X.columns[selected_features_mask]
    return X_selected_features, feature_scores[selected_features_mask]


def combine_features_and_importance(features, importance_factor, top_n=None):
    # If top_n is not provided
    if top_n is None:
        top_n = features.shape[0]
    # Create a DataFrame to store feature names and importance factors
    importance_df = pd.DataFrame({
        'Feature': features,
        'ImportanceFactor': importance_factor
    })
    # Sort the DataFrame by ImportanceFactor in descending order
    importance_df = importance_df.sort_values(by='ImportanceFactor', ascending=False)
    # Select the top N features
    importance_df = importance_df.head(top_n)
    return importance_df
    
def apply_feature_selection(X, y, score_funcs, selection_methods, feature_selection_method):
    # Initialize dictionaries to store importance dataframes
    all_importance_dfs = {score_func.__name__: {} for score_func in score_funcs}
    for score_func in score_funcs:
        for selection_method in selection_methods:
            # Apply the feature selection method
            X_selected, scores = feature_selection_method(X, y, score_func, selection_method)
            # Create a DataFrame of feature importances
            importance_df = combine_features_and_importance(X_selected, scores)
            # Store the DataFrame in the dictionary
            all_importance_dfs[score_func.__name__][selection_method.__name__] = importance_df
    return all_importance_dfs


from sklearn.linear_model        import LinearRegression, LogisticRegression, \
                                        Ridge, RidgeClassifier
from sklearn.tree                import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble            import RandomForestClassifier, RandomForestRegressor, \
                                        GradientBoostingClassifier, GradientBoostingRegressor
import sklearn.metrics
from sklearn.metrics             import get_scorer, make_scorer
from sklearn.model_selection     import cross_val_score


def evaluate_feature_selection(importance_df, X, y, model_types, metrics, 
                               max_selected_features=20, cv=5):
    # Initialize list to store evaluation results
    evaluation_results = []
    # Iterate over different numbers of selected features (k)
    for k in range(1, min(max_selected_features, X.shape[1]) + 1):
        for model_type in model_types:
            for metric in metrics:
                # Get top k features
                selected_features = importance_df.head(k)['Feature']
                # Instantiate the model
                model = model_type()
                metric_name = metric

                # Check if the target is multilabel
                if 2 < y.nunique() < N_UNIQUE_THRESHOLD:
                    # Use appropriate scorer for multilabel
                    metric_func = getattr(sklearn.metrics, metric + '_score')
                    if metric in ['f1', 'precision', 'recall']:
                        metric = make_scorer(metric_func, average='macro')
                    else:
                        metric = make_scorer(metric_func)
                else:
                    metric = get_scorer(metric)

                # Perform cross-validation
                scores = cross_val_score(model, X[selected_features], y, cv=cv, scoring=metric)
                # Store evaluation result for the current k
                evaluation_results.append({
                    'K': k,
                    'score': np.mean(scores),
                    'model': model_type.__name__,
                    'metric': metric_name,
                    'selected_features': list(selected_features)
                })

    # Convert evaluation results to DataFrame and format columns
    evaluation_results = pd.DataFrame(evaluation_results)

    return evaluation_results


def evaluate_feature_selection_methods(all_importance_dfs, X, y, 
                                       score_funcs, selection_methods, 
                                       model_types, metrics, max_selected_features):
    # Initialize dictionary to store evaluation results
    all_evaluation_results = {score_func.__name__: {} for score_func in score_funcs}
    # Iterate over each score function
    for score_func in score_funcs:
        # Iterate over each selection method
        for selection_method in selection_methods:
            curr_score_func = score_func.__name__
            curr_selection_method = selection_method.__name__
            # Retrieve importance dataframe from all_importance_dfs
            importance_df = all_importance_dfs[curr_score_func][curr_selection_method]
            # Evaluate all features based on their importance
            evaluation_results = evaluate_feature_selection(importance_df, X, y, 
                                                model_types, metrics, max_selected_features)
            # Store evaluation results for the current score function and selection method
            all_evaluation_results[curr_score_func][curr_selection_method] = evaluation_results
    return all_evaluation_results


def plot_features_importance(all_importance_dfs, all_evaluation_results, 
                             score_funcs, selection_methods, top_n, figsize):
    for score_func in score_funcs:
        for selection_method in selection_methods:
            curr_score_func = score_func.__name__
            curr_selection_method = selection_method.__name__

            # Retrieve the importance dataframe for the current score function and selection method
            importance_df = all_importance_dfs[curr_score_func][curr_selection_method]
            
            # Plot a bar chart of the top N feature importances
            plot_bar_chart(importance_df.head(top_n), 
                            x='ImportanceFactor', y='Feature', 
                            xlabel='Importance Factor', ylabel='Feature', 
                            title=f'Features Importance Factor using '
                                  f'{curr_score_func} and {curr_selection_method}', 
                            xmin=0, palette='coolwarm')
            
            # Retrieve the evaluation results for the current score function and selection method
            evaluation_results = all_evaluation_results[curr_score_func][curr_selection_method]
            
            for model_type in evaluation_results['model'].unique():
                for metric in evaluation_results['metric'].unique():
                    # Filter the evaluation results for the specified model type and metric
                    curr_evaluation_results = evaluation_results[
                        (evaluation_results['model'] == model_type) & 
                        (evaluation_results['metric'] == metric)
                    ]
                    # Plot a line chart of the evaluation results
                    plot_line_chart(curr_evaluation_results, figsize=figsize,
                                    x='K', y='score', xlabel='# Features', ylabel=f'{metric}', 
                                    title=f'# Features vs. {metric} Scores using {model_type}')


def analyse_feature_selection(X, y, score_funcs, selection_methods, 
                              feature_selection_method, model_types, metrics, 
                              top_n=50, figsize=(12, 3)):
    all_importance_dfs = \
        apply_feature_selection(X, y, score_funcs, selection_methods,
                                feature_selection_method)

    all_evaluation_results = \
        evaluate_feature_selection_methods(all_importance_dfs, X, y, 
                                           score_funcs, selection_methods, 
                                           model_types, metrics, top_n)

    plot_features_importance(all_importance_dfs, all_evaluation_results, 
                             score_funcs, selection_methods, top_n, figsize)

    return all_importance_dfs, all_evaluation_results


train_model3 = train.copy()
df_dtypes(train_model3)


selected_features_by_variance_threshold = variance_threshold_selector(train_model3, 0.1)
print(f'selected_features_by_variance_threshold are {len(selected_features_by_variance_threshold)} features\n{selected_features_by_variance_threshold}')


X, y = split_data_X_y(train_model3.sample(TUNE_DATASET_LEN), target_feature)
score_funcs, selection_methods = [], [SelectKBest, SelectPercentile]

score_funcs = [f_classif, mutual_info_classif]
all_importance_dfs, all_evaluation_results = \
    analyse_feature_selection(X, y, score_funcs, selection_methods, 
                              select_k_best_features,
                              [LogisticRegression], 
                              ['f1', 'accuracy'])


n_features = 50
selected_features_by_k_best = set(X.columns)
for key1 in all_importance_dfs.keys():
    for key2 in all_importance_dfs[key1].keys():
        selected_features_by_k_best &= set(all_importance_dfs[key1][key2].head(n_features)['Feature'].tolist())
print(f'selected_features_by_k_best are {len(selected_features_by_k_best)} features\n{selected_features_by_k_best}')


X, y = split_data_X_y(train_model3.sample(TUNE_DATASET_LEN), target_feature)
score_funcs, selection_methods = [], [SelectFpr, SelectFdr, SelectFwe]

score_funcs = [f_classif]
all_importance_dfs, all_evaluation_results = \
    analyse_feature_selection(X, y, score_funcs, selection_methods, 
                              select_features_by_significance,
                              [LogisticRegression], 
                              ['f1', 'accuracy'])


n_features = 50
selected_features_by_significance = set(X.columns)
for key1 in all_importance_dfs.keys():
    for key2 in all_importance_dfs[key1].keys():
        selected_features_by_significance &= set(all_importance_dfs[key1][key2].head(n_features)['Feature'].tolist())
print(f'selected_features_by_significance are {len(selected_features_by_significance)} features\n{selected_features_by_significance}')


X, y = split_data_X_y(train_model3.sample(TUNE_DATASET_LEN), target_feature)
score_funcs, selection_methods = [], [RFE, SelectFromModel]

score_funcs = [RandomForestClassifier]
all_importance_dfs, all_evaluation_results = \
    analyse_feature_selection(X, y, score_funcs, selection_methods, 
                              select_features_by_model,
                              [LogisticRegression], 
                              ['f1', 'accuracy'])


n_features = 50
selected_features_by_model = set(X.columns)
for key1 in all_importance_dfs.keys():
    for key2 in all_importance_dfs[key1].keys():
        selected_features_by_model &= set(all_importance_dfs[key1][key2].head(n_features)['Feature'].tolist())
print(f'selected_features_by_model are {len(selected_features_by_model)} features\n{selected_features_by_model}')


print(f'selected_features_by_variance_threshold are {len(selected_features_by_variance_threshold)} features\n{selected_features_by_variance_threshold}')
print(f'selected_features_by_k_best are {len(selected_features_by_k_best)} features\n{selected_features_by_k_best}')
print(f'selected_features_by_significance are {len(selected_features_by_significance)} features\n{selected_features_by_significance}')
print(f'selected_features_by_model are {len(selected_features_by_model)} features\n{selected_features_by_model}')


selected_features = set(selected_features_by_variance_threshold) & selected_features_by_model
print(f'selected_features are {len(selected_features)} features\n{selected_features}')


correlations_df = get_target_correlations(train_model3[list(selected_features)+[target_feature]], target_feature)
plot_bar_chart(correlations_df, x='TargetCorrelation', y='Feature', 
               xlabel='Target Correlation', ylabel='Feature', 
               title='Target Correlation in each Feature', palette='coolwarm')


plot_heatmap(train_model3[list(selected_features)+[target_feature]])


df_dtypes(train_model3)


train_model3 = train_model3[list(selected_features)+[target_feature]]


df_dtypes(train_model3)


baseline_models = get_categorical_models(input_size=train_model3.shape[1]-1, output_size=2)
evaluate_model_func = evaluate_categorical_model
#for i, j in baseline_models.items(): print(i, j)


X_train, X_test, y_train, y_test = split_data_train_test(train_model3, target_feature)
evaluation_results3, y_predictions = run_models(baseline_models, evaluate_model_func, X_train, X_test, y_train, y_test)


evaluation_results3


plot_predictions(X_test, y_predictions, figsize=(8, 4), n_col=1)


metrics = sorted(set(evaluation_results3.columns)-set(['Model']))
plot_models_with_evaluation_metrics(baseline_models, evaluation_results3, metrics, title='Selected Features ')


best_model3 = get_best_model(evaluation_results3, baseline_models, 'F1', ascending=False)
print('Best Model of Selected Features Models is:', best_model3.__class__.__name__)


del train_model3, X_train, X_test, y_train, y_test


def boxplot_outlier_detection(df, col):
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_indices = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)].tolist()
    return outlier_indices

def modified_zscore_outlier_detection(df, col, threshold=3.5):
    median = np.median(df[col])
    mad = np.median(np.abs(df[col] - median))
    modified_z_scores = 0.6745 * (df[col] - median) / mad
    outlier_indices = df.index[np.abs(modified_z_scores) > threshold].tolist()
    return outlier_indices

from sklearn.ensemble import IsolationForest

def isolation_forest_outlier_detection(df, col, contamination=0.05):
    model = IsolationForest(contamination=contamination)
    model.fit(df[[col]])
    outlier_predictions = model.predict(df[[col]])
    outlier_indices = df.index[outlier_predictions == -1].tolist()
    return outlier_indices

from sklearn.neighbors import LocalOutlierFactor

def local_factor_outlier_detection(df, col, n_neighbors=10):
    model = LocalOutlierFactor(n_neighbors=n_neighbors)
    outlier_predictions = model.fit_predict(df[[col]])
    outlier_indices = df.index[outlier_predictions == -1].tolist()
    return outlier_indices

from sklearn.cluster import DBSCAN

def dbscan_outlier_detection(df, col, eps=0.36, min_samples=2):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    outlier_predictions = model.fit_predict(df[[col]])
    outlier_indices = df.index[outlier_predictions == -1].tolist()
    return outlier_indices

from sklearn.cluster import KMeans

def kmeans_outlier_detection(df, col, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(df[[col]])
    centers = kmeans.cluster_centers_
    distances = np.abs(df[col] - centers[clusters].ravel())
    threshold = np.percentile(distances, 95)
    outlier_indices = df.index[distances > threshold].tolist()
    return outlier_indices


outlier_detection_methods = [
    boxplot_outlier_detection,
    modified_zscore_outlier_detection,
    isolation_forest_outlier_detection,
    local_factor_outlier_detection,
    dbscan_outlier_detection,
    kmeans_outlier_detection,
]


def label_removed_points(df, df_cleaned):
    # Add a helper column to identify the source of each row
    df['point_type'] = 'original point'
    df_cleaned['point_type'] = 'remaining point'
    # Concatenate both DataFrames
    combined_df = pd.concat([df, df_cleaned], ignore_index=True)
    # Create a DataFrame to identify which rows are duplicates
    duplicates = combined_df.duplicated(subset=df.columns.difference(['point_type']), keep=False)
    # Create masks for original points that are not duplicates and are non-duplicates
    original_points_mask = (combined_df['point_type'] == 'original point')
    non_duplicate_mask = ~duplicates
    mask = non_duplicate_mask & original_points_mask
    # Mark rows that are not duplicates in df as 'removed point'
    combined_df.loc[mask, 'point_type'] = 'removed point'
    # Extract the original DataFrame with the correct labels
    df['point_type'] = combined_df.loc[combined_df.index < len(df), 'point_type'].values
    # Delete the df_cleaned DataFrame to free up memory
    del df_cleaned
    return df


def plot_outlier_feature(df, df_cleaned, feature, outlier_method):
    # Check if outlier cleaning reduced the dataset size
    if df.shape[0] <= df_cleaned.shape[0]:
        return
    # Calculate percentage decrease in dataset size after outlier removal
    decrease = ((df.shape[0] - df_cleaned.shape[0]) / df.shape[0]) * 100
    # Print information about the effect of outlier detection
    print(f'Applying Outlier Detection on Feature {feature} '
          f'will decrease the Dataset by {decrease:.2f}% using {outlier_method}')
    print(f'From df.shape: {df.shape} to df_cleaned.shape: {df_cleaned.shape}')
    # Plot numerical features before outlier detection
    plot_numerical_features(df[[feature]], 
                            title='Data Before Outlier Detection')
    # Plot numerical features after outlier detection
    plot_numerical_features(df_cleaned[[feature]],
                            title=f'Data After Outlier Detection')
    # Label removed points in the original DataFrame
    df_removed = label_removed_points(df.copy(), df_cleaned.copy())
    # Create labels dictionary for plot_labeled_data function
    labels = {outlier_method: df_removed['point_type']}
    # Plot labeled data points
    plot_labeled_data(df_removed.drop(columns=['point_type']), labels)


def remove_outlier_data(df, features, outlier_detection_methods, enable_plot=True):
    # Set to collect all outlier indices
    all_outlier_indices = set()
    outlier_detection_methods_names = str([method.__name__ for method in outlier_detection_methods])
    for feature in features:
        # Initialize with all indices
        curr_feature_outlier_indices = set(np.arange(df.shape[0]))
        for outlier_detection_method in outlier_detection_methods:
            # Detect outliers
            outlier_indices = outlier_detection_method(df, feature)
            # Intersection to find common outliers
            curr_feature_outlier_indices &= set(outlier_indices)
        # Union to combine outliers from all features
        all_outlier_indices |= curr_feature_outlier_indices
        # Drop outliers from the DataFrame
        df_cleaned = df.drop(list(curr_feature_outlier_indices))
        if enable_plot:
            # Plot before and after outlier removal
            plot_outlier_feature(df, df_cleaned, feature, outlier_detection_methods_names)
    # Drop all collected outlier indices
    df_cleaned = df.drop(list(all_outlier_indices))
    # Calculate reduction in data size
    decrease = ((df.shape[0] - df_cleaned.shape[0]) / df.shape[0]) * 100
    print(f'After Applying Outlier Detection on All Features, '
          f'the Dataset decreased by {decrease:.2f}%')
    print(f'From df.shape: {df.shape} to df_cleaned.shape: {df_cleaned.shape}')
    return df_cleaned


train_model4 = train.copy()
df_dtypes(train_model4)


numerical_features = get_numerical_features(train_model4)
selected_numerical_features = set(numerical_features) & (set(correlations_df.head(2)['Feature']) | set(correlations_df.tail(2)['Feature']))
print(f'selected_numerical_features: {selected_numerical_features}')


_ = remove_outlier_data(train_model4, selected_numerical_features, [boxplot_outlier_detection])


_ = remove_outlier_data(train_model4, selected_numerical_features, [modified_zscore_outlier_detection])


_ = remove_outlier_data(train_model4, selected_numerical_features, [isolation_forest_outlier_detection])


#_ = remove_outlier_data(train_model4, selected_numerical_features, [local_factor_outlier_detection])


#_ = remove_outlier_data(train_model4, selected_numerical_features, [dbscan_outlier_detection])


#_ = remove_outlier_data(train_model4, selected_numerical_features, [kmeans_outlier_detection])


_ = remove_outlier_data(train_model4, 
                        selected_numerical_features, 
                        [boxplot_outlier_detection, modified_zscore_outlier_detection, isolation_forest_outlier_detection], 
                        enable_plot=False)


df_dtypes(train_model4)


train_model4 = remove_outlier_data(train_model4, 
                                   selected_numerical_features, 
                                   [boxplot_outlier_detection, modified_zscore_outlier_detection, isolation_forest_outlier_detection], 
                                   enable_plot=False)


df_dtypes(train_model4)


baseline_models = get_categorical_models(input_size=train_model4.shape[1]-1, output_size=2)
evaluate_model_func = evaluate_categorical_model
#for i, j in baseline_models.items(): print(i, j)


X_train, X_test, y_train, y_test = split_data_train_test(train_model4, target_feature)
evaluation_results4, y_predictions = run_models(baseline_models, evaluate_model_func, X_train, X_test, y_train, y_test)


evaluation_results4


plot_predictions(X_test, y_predictions, figsize=(8, 4), n_col=1)


metrics = sorted(set(evaluation_results4.columns)-set(['Model']))
plot_models_with_evaluation_metrics(baseline_models, evaluation_results4, metrics, title='Outlier Treatment ')


best_model4 = get_best_model(evaluation_results4, baseline_models, 'F1', ascending=False)
print('Best Model of Outlier Treatment Models is:', best_model4.__class__.__name__)


del train_model4, X_train, X_test, y_train, y_test


from imblearn.over_sampling import SMOTE

def smote_oversampling(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

from imblearn.over_sampling import BorderlineSMOTE

def borderline_smote_oversampling(X, y):
    borderline_smote = BorderlineSMOTE()
    X_resampled, y_resampled = borderline_smote.fit_resample(X, y)
    return X_resampled, y_resampled

from imblearn.combine import SMOTEENN

def smote_enn_oversampling(X, y):
    smote_enn = SMOTEENN()
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled

from imblearn.combine import SMOTETomek

def smote_tomek_oversampling(X, y):
    smote_tomek = SMOTETomek()
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    return X_resampled, y_resampled


oversampling_methods = [
    smote_oversampling, 
    borderline_smote_oversampling, 
    smote_enn_oversampling, 
    smote_tomek_oversampling
]


def label_added_points(df, df_resampled):
    # Add a helper column to identify the source of each row
    df['point_type'] = 'original point'
    df_resampled['point_type'] = 'added point'
    # Concatenate both DataFrames
    combined_df = pd.concat([df, df_resampled], ignore_index=True)
    # Create a DataFrame to identify which rows are duplicates
    duplicates = combined_df.duplicated(subset=df.columns.difference(['point_type']), keep=False)
    # Mark rows that are not duplicates in df_resampled as 'new point'
    combined_df.loc[duplicates, 'point_type'] = 'original point'
    # Extract the resampled DataFrame with the correct labels
    df_resampled['point_type'] = \
        combined_df.loc[combined_df.index >= len(df), 'point_type'].values
    # Delete the df DataFrame to free up memory
    del df
    return df_resampled


def plot_imbalanced_feature(df, df_resampled, feature, oversampling_method):
    # Ensure that oversampling has increased the number of samples
    if df.shape[0] >= df_resampled.shape[0]:
        return
    # Calculate the percentage increase in dataset size due to oversampling
    increase = ((df_resampled.shape[0] - df.shape[0]) / df.shape[0]) * 100
    print(f'Applying Oversampling on Feature {feature} '
          f'will increase the Dataset by {increase:.2f}% using {oversampling_method}')
    print(f'From df.shape: {df.shape} to df_resampled.shape: {df_resampled.shape}')
    # Plot the feature distribution before oversampling
    plot_categorical_features(df[[feature]],
                              title=f'Data Before Oversampling')
    # Plot the feature distribution after oversampling
    plot_categorical_features(df_resampled[[feature]],
                              title=f'Data After Oversampling')
    # Label the newly added points in the resampled DataFrame
    df_resampled = label_added_points(df.copy(), df_resampled.copy())
    # Create a dictionary of labels for the plot_labeled_data function
    labels = {oversampling_method: df_resampled['point_type']}
    # Plot the labeled data points
    plot_labeled_data(df_resampled.drop(columns=['point_type']), labels)


def oversampling_imbalanced_data(df, feature, oversampling_method):
    # Separate features (X) and target (y)
    X, y = split_data_X_y(df, feature)
    # Perform oversampling using the specified method
    X_resampled, y_resampled = oversampling_method(X, y)
    # Concatenate resampled features and target into a single DataFrame
    df_resampled = pd.concat([
        pd.DataFrame(X_resampled, columns=X.columns), 
        pd.DataFrame(y_resampled, columns=[feature])
    ], axis=1)
    return df_resampled


from collections import Counter

def oversampling_data(df, features, oversampling_method, enable_plot=True):
    df_resampled = df.copy()
    for feature in features:
        # Skip features with only one unique value or with very few samples in the smallest class
        if df[feature].nunique() == 1 or min(Counter(df[feature]).values()) < 6:
            continue
        if enable_plot:
            # Perform oversampling and plot the feature distribution after oversampling
            df_resampled = oversampling_imbalanced_data(df, feature, oversampling_method)
            # Plot before and after outlier removal
            plot_imbalanced_feature(df, df_resampled, feature, oversampling_method.__name__)
        else:
            # Apply the first oversampling method in the list to the DataFrame in place
            df_resampled = oversampling_imbalanced_data(df_resampled, feature, oversampling_method)
    if not enable_plot:
        # Calculate the percentage increase in dataset size due to oversampling
        increase = ((df_resampled.shape[0] - df.shape[0]) / df.shape[0]) * 100
        print(f'After Applying Oversampling on All Features, '
              f'the Dataset increased by {increase:.2f}%')
        print(f'From df.shape: {df.shape} to df_resampled.shape: {df_resampled.shape}')
    return df_resampled


train_model5 = train.copy()
df_dtypes(train_model5)


oversamplying_features = get_categorical_features(train_model5.select_dtypes(exclude=['float']))
info_table = build_my_info_table(train_model5[oversamplying_features])


selected_oversamplying_features = set(oversamplying_features) & (set(correlations_df.head(1)['Feature']) | set(correlations_df.tail(1)['Feature'])) & set(['TARGET'])
print(f'selected_oversamplying_features: {selected_oversamplying_features}')


_ = oversampling_data(train_model5, selected_oversamplying_features, smote_oversampling)


#_ = oversampling_data(train_model5, selected_oversamplying_features, borderline_smote_oversampling)


#_ = oversampling_data(train_model5, selected_oversamplying_features, smote_enn_oversampling)


#_ = oversampling_data(train_model5, selected_oversamplying_features, smote_tomek_oversampling)


df_dtypes(train_model5)


train_model5 = oversampling_data(train_model5, selected_oversamplying_features, smote_oversampling, enable_plot=False)


df_dtypes(train_model5)


baseline_models = get_categorical_models(input_size=train_model5.shape[1]-1, output_size=2)
evaluate_model_func = evaluate_categorical_model
#for i, j in baseline_models.items(): print(i, j)


X_train, X_test, y_train, y_test = split_data_train_test(train_model5, target_feature)
evaluation_results5, y_predictions = run_models(baseline_models, evaluate_model_func, X_train, X_test, y_train, y_test)


evaluation_results5


plot_predictions(X_test, y_predictions, figsize=(8, 4), n_col=1)


metrics = sorted(set(evaluation_results5.columns)-set(['Model']))
plot_models_with_evaluation_metrics(baseline_models, evaluation_results5, metrics, title='Imbalanced Data Treatment ')


best_model6 = get_best_model(evaluation_results5, baseline_models, 'F1', ascending=False)
print('Best Model of Imbalanced Data Treatment Models is:', best_model6.__class__.__name__)


del train_model5, X_train, X_test, y_train, y_test


train_model6 = train.copy()
df_dtypes(train_model6)


train_model6 = log_transform_features(train_model6, skewed_features)
train_model6 = scale_features(train_model6, non_skewed_features)
#train_model6 = create_interaction_features(train_model6)
#train_model6 = create_polynomial_features(train_model6)
#train_model6 = target_encode_features(train_model6, target_feature)
train_model6 = one_hot_encoding(train_model6)
train_model6 = bin_numerical_features(train_model6)


df_dtypes(train_model6)


selected_features_by_variance_threshold = variance_threshold_selector(train_model6, 0.1)
print(f'selected_features_by_variance_threshold are {len(selected_features_by_variance_threshold)} features\n{selected_features_by_variance_threshold}')


X, y = split_data_X_y(train_model6.sample(TUNE_DATASET_LEN), target_feature)
score_funcs, selection_methods = [], [SelectKBest, SelectPercentile]

score_funcs = [f_classif, mutual_info_classif]
all_importance_dfs, all_evaluation_results = \
    analyse_feature_selection(X, y, score_funcs, selection_methods, 
                              select_k_best_features,
                              [LogisticRegression], 
                              ['f1', 'accuracy'])


n_features = 50
selected_features_by_k_best = set(X.columns)
for key1 in all_importance_dfs.keys():
    for key2 in all_importance_dfs[key1].keys():
        selected_features_by_k_best &= set(all_importance_dfs[key1][key2].head(n_features)['Feature'].tolist())
print(f'selected_features_by_k_best are {len(selected_features_by_k_best)} features\n{selected_features_by_k_best}')


X, y = split_data_X_y(train_model6.sample(TUNE_DATASET_LEN), target_feature)
score_funcs, selection_methods = [], [SelectFpr, SelectFdr, SelectFwe]

score_funcs = [f_classif]
all_importance_dfs, all_evaluation_results = \
    analyse_feature_selection(X, y, score_funcs, selection_methods, 
                              select_features_by_significance,
                              [LogisticRegression], 
                              ['f1', 'accuracy'])


n_features = 50
selected_features_by_significance = set(X.columns)
for key1 in all_importance_dfs.keys():
    for key2 in all_importance_dfs[key1].keys():
        selected_features_by_significance &= set(all_importance_dfs[key1][key2].head(n_features)['Feature'].tolist())
print(f'selected_features_by_significance are {len(selected_features_by_significance)} features\n{selected_features_by_significance}')


X, y = split_data_X_y(train_model6.sample(TUNE_DATASET_LEN), target_feature)
score_funcs, selection_methods = [], [RFE, SelectFromModel]

score_funcs = [LogisticRegression]
all_importance_dfs, all_evaluation_results = \
    analyse_feature_selection(X, y, score_funcs, selection_methods, 
                              select_features_by_model,
                              [LogisticRegression], 
                              ['f1', 'accuracy'])


n_features = 50
selected_features_by_model = set(X.columns)
for key1 in all_importance_dfs.keys():
    for key2 in all_importance_dfs[key1].keys():
        selected_features_by_model &= set(all_importance_dfs[key1][key2].head(n_features)['Feature'].tolist())
print(f'selected_features_by_model are {len(selected_features_by_model)} features\n{selected_features_by_model}')


print(f'selected_features_by_variance_threshold are {len(selected_features_by_variance_threshold)} features\n{selected_features_by_variance_threshold}')
print(f'selected_features_by_k_best are {len(selected_features_by_k_best)} features\n{selected_features_by_k_best}')
print(f'selected_features_by_significance are {len(selected_features_by_significance)} features\n{selected_features_by_significance}')
print(f'selected_features_by_model are {len(selected_features_by_model)} features\n{selected_features_by_model}')


selected_features = set(selected_features_by_variance_threshold) & selected_features_by_model
print(f'selected_features are {len(selected_features)} features\n{selected_features}')


train_model6 = train_model6[list(selected_features)+[target_feature]]


correlations_df = get_target_correlations(train_model6[list(selected_features)+[target_feature]], target_feature)
plot_bar_chart(correlations_df, x='TargetCorrelation', y='Feature', 
               xlabel='Target Correlation', ylabel='Feature', 
               title='Target Correlation in each Feature', palette='coolwarm')


plot_heatmap(train_model6[list(selected_features)+[target_feature]])


numerical_features = get_numerical_features(train_model6)
selected_numerical_features = set(numerical_features) & (set(correlations_df.head(4)['Feature']) | set(correlations_df.tail(4)['Feature']))
print(f'selected_numerical_features: {selected_numerical_features}')


train_model6 = remove_outlier_data(train_model6, 
                                   selected_numerical_features, 
                                   [boxplot_outlier_detection, modified_zscore_outlier_detection, isolation_forest_outlier_detection], 
                                   enable_plot=False)


oversamplying_features = get_categorical_features(train_model6.select_dtypes(exclude=['float']))
build_my_info_table(train_model6[oversamplying_features])


selected_oversamplying_features = set(oversamplying_features) & (set(correlations_df.head(1)['Feature']) | set(correlations_df.tail(1)['Feature'])) & set(['TARGET'])
print(f'selected_oversamplying_features: {selected_oversamplying_features}')


train_model6 = oversampling_data(train_model6, selected_oversamplying_features, smote_oversampling, enable_plot=False)


df_dtypes(train_model6)


baseline_models = get_categorical_models(input_size=train_model6.shape[1]-1, output_size=2)
evaluate_model_func = evaluate_categorical_model
#for i, j in baseline_models.items(): print(i, j)


X_train, X_test, y_train, y_test = split_data_train_test(train_model6, target_feature)
evaluation_results6, y_predictions = run_models(baseline_models, evaluate_model_func, X_train, X_test, y_train, y_test)


evaluation_results6


evaluation_results5


evaluation_results4


evaluation_results3


evaluation_results2


plot_predictions(X_test, y_predictions, figsize=(8, 4), n_col=1)


metrics = sorted(set(evaluation_results6.columns)-set(['Model']))
plot_models_with_evaluation_metrics(baseline_models, evaluation_results6, metrics, title='Combined All Enhancements ')


best_model6 = get_best_model(evaluation_results6, baseline_models, 'F1', ascending=False)
print('Best Model of Combined All Enhancements Models is:', best_model6.__class__.__name__)


del train_model6, X_train, X_test, y_train, y_test


param_grids = {
    'Perceptron': {
        'penalty': [None, 'l2', 'l1', 'elasticnet'],
        'alpha': [0.01, 0.1, 1.0, 5.0, 10.0],
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000, 4000, 8000],
        'early_stopping': [True, False],
        'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5],
        'n_iter_no_change': [5, 10, 20, 40],
        'class_weight': [None, 'balanced'],
    },
    'RidgeClassifier': {
        'alpha': [0.01, 0.1, 1.0, 5.0, 10.0],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
        'class_weight': [None, 'balanced'],
        'max_iter': [1000, 2000, 4000, 8000],
    },
    'RidgeClassifierCV': {
        'alphas': [(0.01, 0.1, 1.0, 5.0, 10.0), (0.05, 0.5, 5.0, 25.0)],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced'],
    },
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga', 'newton-cg', 'sag'],
        'max_iter': [1000, 2000, 4000, 8000],
        'multi_class': ['auto', 'ovr', 'multinomial']
    },
    'LogisticRegressionCV': {
        'Cs': [10, 50, 75, 100],
        'solver': ['lbfgs', 'saga', 'newton-cg', 'sag'],
        'max_iter': [1000, 2000, 4000, 8000],
        'multi_class': ['auto', 'ovr', 'multinomial'],
    },
    'SGDClassifier': {
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'alpha': [0.01, 0.1, 1.0, 5.0, 10.0],
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000, 4000, 8000],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'early_stopping': [True, False],
        'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5],
        'n_iter_no_change': [5, 10, 20, 40],
        'average': [False, True],
        'eta0': [0.001, 0.01, 0.1, 1.0],
    },
    'PassiveAggressiveClassifier': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'fit_intercept': [True, False],
        'loss': ['squared_hinge', 'hinge'],
        'max_iter': [1000, 2000, 4000, 8000],
        'early_stopping': [True, False],
        'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5],
        'n_iter_no_change': [5, 10, 20, 40],
        'class_weight': [None, 'balanced'],
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [1, 2, 3, 5, 7, 10, 12],
        'min_samples_split': [3, 5, 7, 10],
        'min_samples_leaf': [2, 4, 6, 8],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    },
    'ExtraTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [1, 2, 3, 5, 7, 10, 12],
        'min_samples_split': [3, 5, 7, 10],
        'min_samples_leaf': [2, 4, 6, 8],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    },
}


from sklearn.metrics import make_scorer, f1_score, mean_squared_error, r2_score

f1_binary_scorer     = make_scorer(lambda y_true, y_pred : f1_score(y_true, y_pred, average='binary'), greater_is_better=True)
f1_macro_scorer      = make_scorer(lambda y_true, y_pred : f1_score(y_true, y_pred, average='macro'),  greater_is_better=True)


from sklearn.experimental    import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

def tune_hyperparameters(model, param_grid, X, y, scoring, 
                         cv=5, refit=True, factor=2):
    # Initialize HalvingRandomSearchCV with the specified parameters
    halving_search = HalvingRandomSearchCV(
        estimator=model, param_distributions=param_grid, cv=cv,
        scoring=scoring, refit=refit, factor=factor,
    )
    # Fit the HalvingGridSearchCV on the data
    halving_search.fit(X, y)

    # Retrieve the best estimator, best score, and best parameters
    best_estimator = halving_search.best_estimator_
    best_score = halving_search.best_score_
    best_params = halving_search.best_params_

    return best_estimator, best_score, best_params


import time

def tune_models(models, X, y, scoring):
    # Perform hyperparameter tuning for each model
    for model_name, model in models.items():
        print(f'Tuning hyperparameters for {model_name}...')
        begin_time = time.time()
        # Tune hyperparameters using HalvingGridSearchCV
        best_estimator, best_score, best_params = \
            tune_hyperparameters(model, param_grids[model_name], 
                                 X, y, scoring=scoring)
        end_time = time.time()
        duration = round((end_time - begin_time) / 60, 2)
        # Update the model with the best estimator found
        models[model_name] = best_estimator
        # Print the best score, parameters, and estimator for the model
        print(f'Best score      : {round(best_score, 2)}')
        print(f'Best parameters : {best_params}')
        print(f'Best estimator  : {best_estimator}')
        print(f'{model_name} Model tuned in'.ljust(50), f'{duration} minutes')
        print('------------------------------------')
    return models


#train_model7 = original_train.copy()
train_model7 = train.copy()
df_dtypes(train_model7)


test_model7 = test.copy()
df_dtypes(test_model7)


skewed_features, non_skewed_features, skew_df = get_skewed_features(train_model7)
print(f'skewed_features:     {len(skewed_features)}\n{skewed_features}')
print(f'non_skewed_features: {len(non_skewed_features)}\n{non_skewed_features}')


train_model7 = log_transform_features(train_model7, skewed_features)
train_model7 = scale_features(train_model7, non_skewed_features)
#train_model7 = create_interaction_features(train_model7)
#train_model7 = create_polynomial_features(train_model7)
#train_model7 = target_encode_features(train_model7, target_feature)
train_model7 = one_hot_encoding(train_model7)
train_model7 = bin_numerical_features(train_model7)
df_dtypes(train_model7)


skewed_features, non_skewed_features, skew_df = get_skewed_features(test_model7)
print(f'skewed_features:     {len(skewed_features)}\n{skewed_features}')
print(f'non_skewed_features: {len(non_skewed_features)}\n{non_skewed_features}')


test_model7 = log_transform_features(test_model7, skewed_features)
test_model7 = scale_features(test_model7, non_skewed_features)
#test_model7 = create_interaction_features(test_model7)
#test_model7 = create_polynomial_features(test_model7)
#test_model7 = target_encode_features(test_model7, target_feature)
test_model7 = one_hot_encoding(test_model7)
test_model7 = bin_numerical_features(test_model7)
df_dtypes(test_model7)


common_selected_features = list(set(selected_features) & set(train_model7.columns) & set(test_model7.columns))
print(f'common_selected_features: {common_selected_features}')


train_model7 = train_model7[common_selected_features+[target_feature]]
df_dtypes(train_model7)


test_model7 = test_model7[common_selected_features]
df_dtypes(test_model7)


print(set(train_model7) - set(test_model7))
print(set(test_model7) - set(train_model7))


numerical_features = get_numerical_features(train_model7)
selected_numerical_features = set(numerical_features) & (set(correlations_df.head(4)['Feature']) | set(correlations_df.tail(4)['Feature']))
print(f'selected_numerical_features: {selected_numerical_features}')


train_model7 = remove_outlier_data(train_model7, 
                                   selected_numerical_features, 
                                   [boxplot_outlier_detection, modified_zscore_outlier_detection, isolation_forest_outlier_detection], 
                                   enable_plot=False)


oversamplying_features = get_categorical_features(train_model7.select_dtypes(exclude=['float']))
build_my_info_table(train_model7[oversamplying_features])


selected_oversamplying_features = set(oversamplying_features) & (set(correlations_df.head(1)['Feature']) | set(correlations_df.tail(1)['Feature'])) & set(['TARGET'])
print(f'selected_oversamplying_features: {selected_oversamplying_features}')


train_model7 = oversampling_data(train_model7, selected_oversamplying_features, smote_oversampling, enable_plot=False)


baseline_models = get_categorical_models(input_size=train_model7.shape[1]-1, output_size=2)
evaluate_model_func = evaluate_categorical_model
#for i, j in baseline_models.items(): print(i, j)


X, y = split_data_X_y(train_model7.sample(TUNE_DATASET_LEN), target_feature)
tuned_baseline_models = baseline_models # tune_models(baseline_models, X, y, 'f1_macro')


X_train, X_test, y_train, y_test = split_data_train_test(train_model7, target_feature)
evaluation_results7, y_predictions = run_models(tuned_baseline_models, evaluate_model_func, X_train, X_test, y_train, y_test)


evaluation_results7


plot_predictions(X_test, y_predictions, figsize=(8, 4), n_col=1)


metrics = sorted(set(evaluation_results7.columns)-set(['Model']))
for metric in metrics:
    plot_models_with_evaluation_metrics(tuned_baseline_models, evaluation_results7, [metric], title='Other Predictive ')
    plot_models_with_evaluation_metrics(tuned_baseline_models, evaluation_results6, [metric], title='Combined All Enhancements ')
    plot_models_with_evaluation_metrics(tuned_baseline_models, evaluation_results2, [metric], title='Enhanced Features ')


best_model7 = get_best_model(evaluation_results7, tuned_baseline_models, 'F1', ascending=False)
print('Best Model of Other Predictive Models is:', best_model7.__class__.__name__)


#del train_model7, X_train, X_test, y_train, y_test


submission = pd.read_csv(f'{dir_path}/sample_submission.csv')
submission.head()


submission = pd.DataFrame({
    'SK_ID_CURR' : test_id,
    target_feature : predict(best_model7, test_model7[X_test.columns]),
})
submission.head()


submission.to_csv("submission.csv", index = False, header = True)

