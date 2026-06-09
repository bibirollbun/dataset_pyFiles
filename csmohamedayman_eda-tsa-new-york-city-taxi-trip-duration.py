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


!unzip -o -d /kaggle/working/ /kaggle/input/nyc-taxi-trip-duration/train.zip 
!unzip -o -d /kaggle/working/ /kaggle/input/nyc-taxi-trip-duration/test.zip 
!unzip -o -d /kaggle/working/ /kaggle/input/nyc-taxi-trip-duration/sample_submission.zip 


dir_path = '/kaggle/working'
train = pd.read_csv(f'{dir_path}/train.csv')
test  = pd.read_csv(f'{dir_path}/test.csv')


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


N_UNIQUE_THRESHOLD = 55

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


print(f'train.shape: {train.shape}')
print(f'test.shape : {test.shape}')


df_dtypes(train)


df_dtypes(test)


print(set(train) - set(test))
print(set(test) - set(train))


target_feature = list(set(train.columns)-set(test.columns))[1]
target_feature


categorical_features = get_categorical_features(train)
numerical_features   = get_numerical_features(train)
print(f'categorical_features: {len(categorical_features)}\n{categorical_features}')
print(f'numerical_features:   {len(numerical_features)}\n{numerical_features}')


info_table = build_my_info_table(train)
info_table


nan_df = info_table[info_table['null %'] >= 10][['column', 'null %']].sort_values(by='null %')
plot_bar_chart(nan_df, x='null %', y='column', 
               xlabel='Null Percentage %', ylabel='Feature', 
               title='Null Percentage in each Feature', 
               xmin=0, xmax=100, palette='coolwarm')


dropped_nan = set(nan_df[nan_df['null %'] > 25]['column'])
train = train.drop(columns=list(dropped_nan - set([target_feature])))
test  = test.drop(columns=list(dropped_nan - set([target_feature])))


train = fillna_and_replace_inf(train)
test  = fillna_and_replace_inf(test)


for feature in ['pickup_datetime', 'dropoff_datetime']:
    if feature in train.columns: 
        train = extract_date_features(train, [feature])
        train = extract_time_features(train, [feature])
        train[feature] = train[feature].astype(int) / 10**18
    if feature in test.columns: 
        test = extract_date_features(test, [feature])
        test = extract_time_features(test, [feature])
        test[feature] = test[feature].astype(int) / 10**18


TUNE_DATASET_LEN = int(1e3)
n_samples = min(int(1e6), train.shape[0])
original_train = train.sample(n_samples)
original_train, train = train, original_train


info_table = build_my_info_table(test)
info_table


nan_df = info_table[info_table['null %'] >= 10][['column', 'null %']].sort_values(by='null %')
plot_bar_chart(nan_df, x='null %', y='column', 
               xlabel='Null Percentage %', ylabel='Feature', 
               title='Null Percentage in each Feature', 
               xmin=0, xmax=100, palette='coolwarm')


dropped_nan = set(nan_df[nan_df['null %'] > 25]['column'])
train = train.drop(columns=list(dropped_nan - set([target_feature])))
test  = test.drop(columns=list(dropped_nan - set([target_feature])))


train = encode_str_features(train)
test  = encode_str_features(test)


train, _ = drop_id_feature(train, 'vendor_id')
train, _ = drop_id_feature(train, 'id')
test, test_id  = drop_id_feature(test, 'vendor_id')
test, test_id  = drop_id_feature(test, 'id')


train = train.drop(columns=train.select_dtypes(exclude=[np.number]).columns)
test = test.drop(columns=test.select_dtypes(exclude=[np.number]).columns)


info_table = build_my_info_table(train)
mode_df = info_table[info_table['mode %'] >= 98][['column', 'mode %']].sort_values(by='mode %')
plot_bar_chart(mode_df, x='mode %', y='column', 
               xlabel='Mode Percentage %', ylabel='Feature', 
               title='Mode Percentage in each Feature', 
               xmin=90, xmax=100, palette='coolwarm')


dropped_mode = set(mode_df[mode_df['mode %'] > 98]['column'])
train = train.drop(columns=list(dropped_mode - set([target_feature]) & set(train.columns)))
test  = test.drop(columns=list(dropped_mode - set([target_feature])  & set(test.columns)))


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


plot_numerical_features(train[numerical_features])


plot_categorical_features(train[categorical_features])


for feature in ['passenger_count']:
    plot_categorical_features(original_train[[feature]])
    original_train = replace_rare_categories(original_train, feature, 'Other', threshold_percent=2)
    train          = replace_rare_categories(train, feature, -1, threshold_percent=2)
    plot_categorical_features(original_train[[feature]])


# Filter out outliers in the numerical target_feature
Q1 = train[target_feature].quantile(0.25)
Q3 = train[target_feature].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = train[(train[target_feature] >= Q1 - 1.5 * IQR) & 
                       (train[target_feature] <= Q3 + 1.5 * IQR)]


for feature in categorical_features:
    if feature != target_feature:
        pass
        #sns.catplot(data=df_no_outliers, x=feature, y=target_feature, kind='box', height=3.75, aspect=2.75)


for feature in numerical_features:
    if feature != target_feature:
        pass
        #sns.catplot(data=df_no_outliers, x=feature, y=target_feature, kind='bar', height=3.75, aspect=2.75)


for feature1 in categorical_features:
    for feature2 in categorical_features:
        if feature1 != feature2:
            pass
            #sns.catplot(data=df_no_outliers, x=feature1, y=target_feature, hue=feature2, kind='box', height=4, aspect=3)


original_train.shape


n_samples = min(int(1e7), original_train.shape[0])
df = original_train.sample(n_samples)
print(f'df.shape:\n{df.shape}')
print(f'df.columns:\n{df.columns}')


# Calculating trip distance using haversine formula (approximate)
from math import radians, cos, sin, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

df['trip_distance'] = df.apply(lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'], 
                                                     row['dropoff_latitude'], row['dropoff_longitude']), axis=1)


# Calculate the number of trips per day for each vendor
daily_trip_count_by_vendor = df.groupby(['vendor_id', 'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day']).size().reset_index(name='trip_count')

# Rename columns to match 'year', 'month', and 'day' for `pd.to_datetime()`
daily_trip_count_by_vendor = daily_trip_count_by_vendor.rename(columns={
    'pickup_datetime_year': 'year', 
    'pickup_datetime_month': 'month', 
    'pickup_datetime_day': 'day'
})

# Create a Date column in 'YYYY-MM-DD' format
daily_trip_count_by_vendor['Date'] = pd.to_datetime(daily_trip_count_by_vendor[['year', 'month', 'day']])

# Plotting for each vendor
plt.figure(figsize=(18, 6))

# Loop through each vendor and plot their daily trip counts
for vendor in daily_trip_count_by_vendor['vendor_id'].unique():
    vendor_data = daily_trip_count_by_vendor[daily_trip_count_by_vendor['vendor_id'] == vendor]
    vendor_data.set_index('Date', inplace=True)  # Set Date as index for each vendor's data
    plt.plot(vendor_data.index, vendor_data['trip_count'], label=f'Vendor {vendor}', linewidth=1.25)

# Set x-ticks to show every 5th day with 'YYYY-MM-DD' format
tick_positions = daily_trip_count_by_vendor['Date'].unique()[::5]
plt.xticks(ticks=tick_positions, labels=pd.to_datetime(tick_positions).strftime('%Y-%m-%d'), rotation=90)

# Set the title, labels, and grid
plt.title('Number of Trips Per Day by Vendor')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.legend(title='Vendor ID')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Calculate the daily mean of trip duration for each vendor
daily_trip_duration_by_vendor = df.groupby(['vendor_id', 'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day'])['trip_duration'].mean().reset_index()

# Rename columns to match the required 'year', 'month', and 'day' for `pd.to_datetime()`
daily_trip_duration_by_vendor = daily_trip_duration_by_vendor.rename(columns={
    'pickup_datetime_year': 'year', 
    'pickup_datetime_month': 'month', 
    'pickup_datetime_day': 'day'
})

# Create a Date column in 'YYYY-MM-DD' format
daily_trip_duration_by_vendor['Date'] = pd.to_datetime(daily_trip_duration_by_vendor[['year', 'month', 'day']])

# Plotting for each vendor
plt.figure(figsize=(18, 6))

# Loop through each vendor and plot their daily mean trip durations
for vendor in daily_trip_duration_by_vendor['vendor_id'].unique():
    vendor_data = daily_trip_duration_by_vendor[daily_trip_duration_by_vendor['vendor_id'] == vendor]
    vendor_data.set_index('Date', inplace=True)  # Set Date as index for each vendor's data
    plt.plot(vendor_data.index, vendor_data['trip_duration'] / 60, label=f'Vendor {vendor}', linewidth=1.25)  # Convert seconds to minutes

# Set x-ticks to show every 5th day with 'YYYY-MM-DD' format
tick_positions = daily_trip_duration_by_vendor['Date'].unique()[::5]
plt.xticks(ticks=tick_positions, labels=pd.to_datetime(tick_positions).strftime('%Y-%m-%d'), rotation=90)

# Set the title, labels, and grid
plt.title('Mean Trip Duration Over Time by Vendor')
plt.xlabel('Date')
plt.ylabel('Mean Trip Duration (minutes)')
plt.legend(title='Vendor ID')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Calculate the daily mean of trip distance for each vendor
daily_trip_distance_by_vendor = df.groupby(['vendor_id', 'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day'])['trip_distance'].mean().reset_index()

# Rename columns to match the required 'year', 'month', and 'day' for `pd.to_datetime()`
daily_trip_distance_by_vendor = daily_trip_distance_by_vendor.rename(columns={
    'pickup_datetime_year': 'year', 
    'pickup_datetime_month': 'month', 
    'pickup_datetime_day': 'day'
})

# Create a Date column in 'YYYY-MM-DD' format
daily_trip_distance_by_vendor['Date'] = pd.to_datetime(daily_trip_distance_by_vendor[['year', 'month', 'day']])

# Plotting for each vendor
plt.figure(figsize=(18, 6))

# Loop through each vendor and plot their daily mean trip distances
for vendor in daily_trip_distance_by_vendor['vendor_id'].unique():
    vendor_data = daily_trip_distance_by_vendor[daily_trip_distance_by_vendor['vendor_id'] == vendor]
    vendor_data.set_index('Date', inplace=True)  # Set Date as index for each vendor's data
    plt.plot(vendor_data.index, vendor_data['trip_distance'], label=f'Vendor {vendor}', linewidth=1.25)  # Plot distance

# Set x-ticks to show every 5th day with 'YYYY-MM-DD' format
tick_positions = daily_trip_distance_by_vendor['Date'].unique()[::5]
plt.xticks(ticks=tick_positions, labels=pd.to_datetime(tick_positions).strftime('%Y-%m-%d'), rotation=90)

# Set the title, labels, and grid
plt.title('Average Trip Distance Over Time by Vendor')
plt.xlabel('Date')
plt.ylabel('Average Trip Distance (km)')
plt.legend(title='Vendor ID')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Calculate the number of trips per week for each vendor
weekly_trip_count_by_vendor = df.groupby(['vendor_id', 'pickup_datetime_year', 'pickup_datetime_weekofyear']).size().reset_index(name='trip_count')

# Create a 'Week' column to represent the week as a string (for labeling purposes)
weekly_trip_count_by_vendor['Week'] = weekly_trip_count_by_vendor['pickup_datetime_year'].astype(str) + '-W' + weekly_trip_count_by_vendor['pickup_datetime_weekofyear'].astype(str)

# Plotting for each vendor
plt.figure(figsize=(18, 6))

# Loop through each vendor and plot their weekly trip counts
for vendor in weekly_trip_count_by_vendor['vendor_id'].unique():
    vendor_data = weekly_trip_count_by_vendor[weekly_trip_count_by_vendor['vendor_id'] == vendor]
    plt.plot(vendor_data['Week'], vendor_data['trip_count'], label=f'Vendor {vendor}', linewidth=1.25)  # Plot trip counts

# Set x-ticks to show every week
plt.xticks(ticks=weekly_trip_count_by_vendor['Week'][::2], labels=weekly_trip_count_by_vendor['Week'][::2], rotation=90)

# Set the title, labels, and legend
plt.title('Number of Trips Per Week by Vendor')
plt.xlabel('Week (YYYY-WW)')
plt.ylabel('Number of Trips')
plt.legend(title='Vendor ID')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Calculate the mean trip duration per week for each vendor
weekly_trip_duration_by_vendor = df.groupby(['vendor_id', 'pickup_datetime_year', 'pickup_datetime_weekofyear'])['trip_duration'].mean().reset_index(name='mean_trip_duration')

# Create a 'Week' column to represent the week as a string (for labeling purposes)
weekly_trip_duration_by_vendor['Week'] = weekly_trip_duration_by_vendor['pickup_datetime_year'].astype(str) + '-W' + weekly_trip_duration_by_vendor['pickup_datetime_weekofyear'].astype(str)

# Plotting for each vendor
plt.figure(figsize=(18, 6))

# Loop through each vendor and plot their mean trip durations
for vendor in weekly_trip_duration_by_vendor['vendor_id'].unique():
    vendor_data = weekly_trip_duration_by_vendor[weekly_trip_duration_by_vendor['vendor_id'] == vendor]
    plt.plot(vendor_data['Week'], vendor_data['mean_trip_duration'] / 60, label=f'Vendor {vendor}', linewidth=1.25)  # Convert seconds to minutes

# Set x-ticks to show every week
plt.xticks(ticks=weekly_trip_duration_by_vendor['Week'][::2], labels=weekly_trip_duration_by_vendor['Week'][::2], rotation=90)

# Set the title, labels, and legend
plt.title('Average Trip Duration Per Week by Vendor')
plt.xlabel('Week (YYYY-WW)')
plt.ylabel('Average Trip Duration (minutes)')
plt.legend(title='Vendor ID')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Calculate the mean trip distance per week for each vendor
weekly_trip_distance_by_vendor = df.groupby(['vendor_id', 'pickup_datetime_year', 'pickup_datetime_weekofyear'])['trip_distance'].mean().reset_index(name='mean_trip_distance')

# Create a 'Week' column to represent the week as a string (for labeling purposes)
weekly_trip_distance_by_vendor['Week'] = weekly_trip_distance_by_vendor['pickup_datetime_year'].astype(str) + '-W' + weekly_trip_distance_by_vendor['pickup_datetime_weekofyear'].astype(str)

# Plotting for each vendor
plt.figure(figsize=(18, 6))

# Loop through each vendor and plot their mean trip distances
for vendor in weekly_trip_distance_by_vendor['vendor_id'].unique():
    vendor_data = weekly_trip_distance_by_vendor[weekly_trip_distance_by_vendor['vendor_id'] == vendor]
    plt.plot(vendor_data['Week'], vendor_data['mean_trip_distance'], label=f'Vendor {vendor}', linewidth=1.25)

# Set x-ticks to show every week
plt.xticks(ticks=weekly_trip_distance_by_vendor['Week'][::2], labels=weekly_trip_distance_by_vendor['Week'][::2], rotation=90)

# Set the title, labels, and legend
plt.title('Average Trip Distance Per Week by Vendor')
plt.xlabel('Week (YYYY-WW)')
plt.ylabel('Average Trip Distance (km)')
plt.legend(title='Vendor ID')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Calculate the number of trips by hour for each vendor
trip_count_by_hour_vendor = df.groupby(['vendor_id', 'pickup_datetime_hour']).size().reset_index(name='trip_count')

# Pivot the data to get vendor_id as columns and hours as index
trip_count_pivot = trip_count_by_hour_vendor.pivot(index='pickup_datetime_hour', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(trip_count_pivot))

# Plotting each vendor's trip count as bars
for i, vendor in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Number of Trips by Hour of the Day (by Vendor)')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], range(24), rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
trip_count_pivot


# Calculate the number of trips by hour for each vendor
trip_count_by_hour_vendor = df.groupby(['vendor_id', 'pickup_datetime_hour']).size().reset_index(name='trip_count')

# Pivot the data to ensure all 24 hours are represented for each vendor
trip_count_pivot = trip_count_by_hour_vendor.pivot(index='pickup_datetime_hour', columns='vendor_id', values='trip_count').fillna(0)

# Create subplots for each vendor
vendors = trip_count_pivot.columns
num_vendors = len(vendors)

fig, axs = plt.subplots(1, num_vendors, figsize=(18, 6))

# Loop through each vendor to create pie charts
for i, vendor in enumerate(vendors):
    # Get counts for the current vendor
    counts = trip_count_pivot[vendor]
    
    # Create pie chart for this vendor
    axs[i].pie(counts, labels=range(24), autopct='%1.1f%%', startangle=140)
    axs[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Add title with a line space
    axs[i].set_title(f'Vendor {vendor} - Trip Counts by Hour\n')

plt.tight_layout()
plt.show()
trip_count_pivot


# Group by pickup hour and vendor, calculating average trip duration
avg_trip_duration_by_hour_vendor = df.groupby(['vendor_id', 'pickup_datetime_hour'])['trip_duration'].mean().reset_index()

# Convert trip duration from seconds to minutes
avg_trip_duration_by_hour_vendor['trip_duration'] /= 60  # Convert to minutes

# Pivot the data to have hours as the index and vendor_id as columns
avg_duration_pivot = avg_trip_duration_by_hour_vendor.pivot(index='pickup_datetime_hour', columns='vendor_id', values='trip_duration').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_duration_pivot))

# Plotting each vendor's average trip duration as bars
for i, vendor in enumerate(avg_duration_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_duration_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Duration by Hour of the Day (by Vendor)')
plt.xlabel('Hour')
plt.ylabel('Average Trip Duration (minutes)')
plt.xticks([x + bar_width * (len(avg_duration_pivot.columns) - 1) / 2 for x in r], range(24), rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_duration_pivot


# Group by pickup hour and vendor, calculating average trip distance
avg_trip_distance_by_hour_vendor = df.groupby(['vendor_id', 'pickup_datetime_hour'])['trip_distance'].mean().reset_index()

# Pivot the data to have hours as the index and vendor_id as columns
avg_distance_pivot = avg_trip_distance_by_hour_vendor.pivot(index='pickup_datetime_hour', columns='vendor_id', values='trip_distance').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_distance_pivot))

# Plotting each vendor's average trip distance as bars
for i, vendor in enumerate(avg_distance_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_distance_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Distance by Hour of the Day (by Vendor)')
plt.xlabel('Hour')
plt.ylabel('Average Trip Distance (km)')
plt.xticks([x + bar_width * (len(avg_distance_pivot.columns) - 1) / 2 for x in r], range(24), rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_distance_pivot


# Group by weekday and vendor, counting the number of trips
trip_count_by_day_vendor = df.groupby(['vendor_id', 'pickup_datetime_weekday']).size().reset_index(name='trip_count')

# Pivot the data to have weekdays as the index and vendor_id as columns
trip_count_pivot = trip_count_by_day_vendor.pivot(index='pickup_datetime_weekday', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(trip_count_pivot))

# Plotting each vendor's trip count as bars
for i, vendor in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Number of Trips by Day of the Week (by Vendor)')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], range(7), rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by weekday and vendor, counting the number of trips
trip_count_by_day_vendor = df.groupby(['vendor_id', 'pickup_datetime_weekday']).size().reset_index(name='trip_count')

# Pivot the data to have weekdays as the index and vendor_id as columns
trip_count_pivot = trip_count_by_day_vendor.pivot(index='pickup_datetime_weekday', columns='vendor_id', values='trip_count').fillna(0)

# Create subplots for each vendor
vendors = trip_count_pivot.columns
num_vendors = len(vendors)

fig, axs = plt.subplots(1, num_vendors, figsize=(18, 6))

# Loop through each vendor to create pie charts
for i, vendor in enumerate(vendors):
    # Get counts for the current vendor
    counts = trip_count_pivot[vendor]
    
    # Create pie chart for this vendor
    axs[i].pie(counts, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], autopct='%1.1f%%', startangle=140)
    axs[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Add title with a line space
    axs[i].set_title(f'Vendor {vendor} - Trip Counts by Day of the Week\n')

plt.tight_layout()
plt.show()
trip_count_pivot


# Group by pickup weekday and vendor, calculating the average trip duration
avg_trip_duration_by_day_vendor = df.groupby(['vendor_id', 'pickup_datetime_weekday'])['trip_duration'].mean().reset_index()

# Pivot the data to have weekdays as the index and vendor_id as columns
avg_trip_duration_pivot = avg_trip_duration_by_day_vendor.pivot(index='pickup_datetime_weekday', columns='vendor_id', values='trip_duration').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_duration_pivot))

# Plotting each vendor's average trip duration as bars
for i, vendor in enumerate(avg_trip_duration_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_duration_pivot[vendor] / 60, width=bar_width, label=f'Vendor {vendor}', alpha=0.6)  # Convert seconds to minutes

# Set the title and labels
plt.title('Average Trip Duration by Day of the Week (by Vendor)')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Average Trip Duration (minutes)')
plt.xticks([x + bar_width * (len(avg_trip_duration_pivot.columns) - 1) / 2 for x in r], range(7), rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_duration_pivot


# Group by pickup weekday and vendor, calculating the average trip distance
avg_trip_distance_by_day_vendor = df.groupby(['vendor_id', 'pickup_datetime_weekday'])['trip_distance'].mean().reset_index()

# Pivot the data to have weekdays as the index and vendor_id as columns
avg_trip_distance_pivot = avg_trip_distance_by_day_vendor.pivot(index='pickup_datetime_weekday', columns='vendor_id', values='trip_distance').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_distance_pivot))

# Plotting each vendor's average trip distance as bars
for i, vendor in enumerate(avg_trip_distance_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_distance_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Distance by Day of the Week (by Vendor)')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Average Trip Distance (km)')
plt.xticks([x + bar_width * (len(avg_trip_distance_pivot.columns) - 1) / 2 for x in r], range(7), rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_distance_pivot


# Group by pickup season and vendor, counting the number of trips
trip_count_by_season_vendor = df.groupby(['vendor_id', 'pickup_datetime_season']).size().reset_index(name='trip_count')

# Pivot the data to have seasons as the index and vendor_id as columns
trip_count_pivot = trip_count_by_season_vendor.pivot(index='pickup_datetime_season', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(trip_count_pivot))

# Plotting each vendor's trip count as bars
for i, vendor in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Number of Trips by Season (by Vendor)')
plt.xlabel('Season')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], trip_count_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by pickup season and vendor, counting the number of trips
trip_count_by_season_vendor = df.groupby(['vendor_id', 'pickup_datetime_season']).size().reset_index(name='trip_count')

# Pivot the data to have seasons as the index and vendor_id as columns
trip_count_pivot = trip_count_by_season_vendor.pivot(index='pickup_datetime_season', columns='vendor_id', values='trip_count').fillna(0)

# Create subplots for each vendor
vendors = trip_count_pivot.columns
num_vendors = len(vendors)

fig, axs = plt.subplots(1, num_vendors, figsize=(18, 6))

# Loop through each vendor to create pie charts
for i, vendor in enumerate(vendors):
    # Get counts for the current vendor
    counts = trip_count_pivot[vendor]
    
    # Create pie chart for this vendor
    axs[i].pie(counts, labels=df['pickup_datetime_season'].unique(), autopct='%1.1f%%', startangle=140)
    axs[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Add title with a line space
    axs[i].set_title(f'Vendor {vendor} - Trip Counts by Season\n')

plt.tight_layout()
plt.show()
trip_count_pivot


# Group by season and vendor, calculating the average trip duration
avg_trip_duration_by_season_vendor = df.groupby(['vendor_id', 'pickup_datetime_season'])['trip_duration'].mean().reset_index()

# Convert duration from seconds to minutes
avg_trip_duration_by_season_vendor['trip_duration'] = avg_trip_duration_by_season_vendor['trip_duration'] / 60

# Pivot the data to have seasons as the index and vendor_id as columns
avg_duration_pivot = avg_trip_duration_by_season_vendor.pivot(index='pickup_datetime_season', columns='vendor_id', values='trip_duration').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_duration_pivot))

# Plotting each vendor's average trip duration as bars
for i, vendor in enumerate(avg_duration_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_duration_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Duration by Season (by Vendor)')
plt.xlabel('Season')
plt.ylabel('Average Trip Duration (minutes)')
plt.xticks([x + bar_width * (len(avg_duration_pivot.columns) - 1) / 2 for x in r], avg_duration_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_duration_pivot


# Group by season and vendor, calculating the average trip distance
avg_trip_distance_by_season_vendor = df.groupby(['vendor_id', 'pickup_datetime_season'])['trip_distance'].mean().reset_index()

# Pivot the data to have seasons as the index and vendor_id as columns
avg_distance_pivot = avg_trip_distance_by_season_vendor.pivot(index='pickup_datetime_season', columns='vendor_id', values='trip_distance').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_distance_pivot))

# Plotting each vendor's average trip distance as bars
for i, vendor in enumerate(avg_distance_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_distance_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Distance by Season (by Vendor)')
plt.xlabel('Season')
plt.ylabel('Average Trip Distance (km)')
plt.xticks([x + bar_width * (len(avg_distance_pivot.columns) - 1) / 2 for x in r], avg_distance_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_distance_pivot


# Group by pickup quarter and vendor, counting the number of trips
trip_count_by_quarter_vendor = df.groupby(['vendor_id', 'pickup_datetime_quarter']).size().reset_index(name='trip_count')

# Pivot the data to have quarters as the index and vendor_id as columns
trip_count_pivot = trip_count_by_quarter_vendor.pivot(index='pickup_datetime_quarter', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bars
bar_width = 0.35
# Set positions of bars on X axis
r = range(len(trip_count_pivot))

# Plotting each vendor's trip count as bars
for i, vendor in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Number of Trips by Quarter (by Vendor)')
plt.xlabel('Quarter')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], 
           [f'Q{i}' for i in df['pickup_datetime_quarter'].unique()], rotation=0)  # Center the ticks with quarter names
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by pickup quarter and vendor, counting the number of trips
trip_count_by_quarter_vendor = df.groupby(['vendor_id', 'pickup_datetime_quarter']).size().reset_index(name='trip_count')

# Pivot the data to have quarters as the index and vendor_id as columns
trip_count_pivot = trip_count_by_quarter_vendor.pivot(index='pickup_datetime_quarter', columns='vendor_id', values='trip_count').fillna(0)

# Create subplots for each vendor
vendors = trip_count_pivot.columns
num_vendors = len(vendors)

fig, axs = plt.subplots(1, num_vendors, figsize=(18, 6))

# Loop through each vendor to create pie charts
for i, vendor in enumerate(vendors):
    # Get counts for the current vendor
    counts = trip_count_pivot[vendor]
    
    # Create pie chart for this vendor
    axs[i].pie(counts, labels=[f'Q{i}' for i in df['pickup_datetime_quarter'].unique()], autopct='%1.1f%%', startangle=140)
    axs[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Add title with a line space
    axs[i].set_title(f'Vendor {vendor} - Trip Counts by Quarter\n')

plt.tight_layout()
plt.show()
trip_count_pivot


# Group by pickup quarter and vendor, calculating the average trip duration
avg_trip_duration_by_quarter_vendor = df.groupby(['vendor_id', 'pickup_datetime_quarter'])['trip_duration'].mean().reset_index()

# Convert duration from seconds to minutes
avg_trip_duration_by_quarter_vendor['trip_duration'] = avg_trip_duration_by_quarter_vendor['trip_duration'] / 60

# Pivot the data to have quarters as the index and vendor_id as columns
avg_duration_pivot = avg_trip_duration_by_quarter_vendor.pivot(index='pickup_datetime_quarter', columns='vendor_id', values='trip_duration').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bars on X axis
r = range(len(avg_duration_pivot))

# Plotting each vendor's average trip duration as bars
for i, vendor in enumerate(avg_duration_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_duration_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Duration by Quarter (by Vendor)')
plt.xlabel('Quarter')
plt.ylabel('Average Trip Duration (minutes)')
plt.xticks([x + bar_width * (len(avg_duration_pivot.columns) - 1) / 2 for x in r], [f'Q{i}' for i in df['pickup_datetime_quarter'].unique()], rotation=0)  # Center the ticks with quarter labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_duration_pivot


# Group by pickup quarter and vendor, calculating the average trip distance
avg_trip_distance_by_quarter_vendor = df.groupby(['vendor_id', 'pickup_datetime_quarter'])['trip_distance'].mean().reset_index()

# Pivot the data to have quarters as the index and vendor_id as columns
avg_distance_pivot = avg_trip_distance_by_quarter_vendor.pivot(index='pickup_datetime_quarter', columns='vendor_id', values='trip_distance').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bars on X axis
r = range(len(avg_distance_pivot))

# Plotting each vendor's average trip distance as bars
for i, vendor in enumerate(avg_distance_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_distance_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Distance by Quarter (by Vendor)')
plt.xlabel('Quarter')
plt.ylabel('Average Trip Distance (km)')
plt.xticks([x + bar_width * (len(avg_distance_pivot.columns) - 1) / 2 for x in r], [f'Q{i}' for i in df['pickup_datetime_quarter'].unique()], rotation=0)  # Center the ticks with quarter labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_distance_pivot


# Group by month and vendor to calculate the number of trips
trip_count_by_month_vendor = df.groupby(['vendor_id', 'pickup_datetime_month']).size().reset_index(name='trip_count')

# Pivot the data to have months as the index and vendor_id as columns
trip_count_pivot = trip_count_by_month_vendor.pivot(index='pickup_datetime_month', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(trip_count_pivot))

# Plotting each vendor's trip count as bars
for i, vendor in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Number of Trips by Month (by Vendor)')
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], trip_count_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by month and vendor to calculate the number of trips
trip_count_by_month_vendor = df.groupby(['vendor_id', 'pickup_datetime_month']).size().reset_index(name='trip_count')

# Pivot the data to have months as the index and vendor_id as columns
trip_count_pivot = trip_count_by_month_vendor.pivot(index='pickup_datetime_month', columns='vendor_id', values='trip_count').fillna(0)

# Create subplots for each vendor
vendors = trip_count_pivot.columns
num_vendors = len(vendors)

fig, axs = plt.subplots(1, num_vendors, figsize=(18, 6))

# Loop through each vendor to create pie charts
for i, vendor in enumerate(vendors):
    # Get counts for the current vendor
    counts = trip_count_pivot[vendor]
    
    # Create pie chart for this vendor
    axs[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    axs[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Add title with a line space
    axs[i].set_title(f'Vendor {vendor} - Trip Counts by Month\n')

plt.tight_layout()
plt.show()
trip_count_pivot


# Group by month and vendor to calculate average trip duration
avg_trip_duration_by_month_vendor = df.groupby(['vendor_id', 'pickup_datetime_month'])['trip_duration'].mean().reset_index()

# Convert trip duration from seconds to minutes
avg_trip_duration_by_month_vendor['trip_duration'] = avg_trip_duration_by_month_vendor['trip_duration'] / 60

# Pivot the data to have months as the index and vendor_id as columns
avg_trip_duration_pivot = avg_trip_duration_by_month_vendor.pivot(index='pickup_datetime_month', columns='vendor_id', values='trip_duration').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_duration_pivot))

# Plotting each vendor's average trip duration as bars
for i, vendor in enumerate(avg_trip_duration_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_duration_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Duration by Month (by Vendor)')
plt.xlabel('Month')
plt.ylabel('Average Trip Duration (minutes)')
plt.xticks([x + bar_width * (len(avg_trip_duration_pivot.columns) - 1) / 2 for x in r], avg_trip_duration_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_duration_pivot


# Group by month and vendor to calculate average trip distance
avg_trip_distance_by_month_vendor = df.groupby(['vendor_id', 'pickup_datetime_month'])['trip_distance'].mean().reset_index()

# Pivot the data to have months as the index and vendor_id as columns
avg_trip_distance_pivot = avg_trip_distance_by_month_vendor.pivot(index='pickup_datetime_month', columns='vendor_id', values='trip_distance').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_distance_pivot))

# Plotting each vendor's average trip distance as bars
for i, vendor in enumerate(avg_trip_distance_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_distance_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Distance by Month (by Vendor)')
plt.xlabel('Month')
plt.ylabel('Average Trip Distance (km)')
plt.xticks([x + bar_width * (len(avg_trip_distance_pivot.columns) - 1) / 2 for x in r], avg_trip_distance_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_distance_pivot


# Group by time of day and vendor to calculate the number of trips
trip_count_by_time_of_day_vendor = df.groupby(['vendor_id', 'pickup_datetime_timeofday']).size().reset_index(name='trip_count')

# Pivot the data to have time of day as the index and vendor_id as columns
trip_count_pivot = trip_count_by_time_of_day_vendor.pivot(index='pickup_datetime_timeofday', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(trip_count_pivot))

# Plotting each vendor's trip count as bars
for i, vendor in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Number of Trips by Time of Day (by Vendor)')
plt.xlabel('Time of Day')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], trip_count_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by time of day and vendor to calculate the number of trips
trip_count_by_time_of_day_vendor = df.groupby(['vendor_id', 'pickup_datetime_timeofday']).size().reset_index(name='trip_count')

# Pivot the data to have time of day as the index and vendor_id as columns
trip_count_pivot = trip_count_by_time_of_day_vendor.pivot(index='pickup_datetime_timeofday', columns='vendor_id', values='trip_count').fillna(0)

# Create subplots for each vendor
vendors = trip_count_pivot.columns
num_vendors = len(vendors)

fig, axs = plt.subplots(1, num_vendors, figsize=(18, 6))

# Loop through each vendor to create pie charts
for i, vendor in enumerate(vendors):
    # Get counts for the current vendor
    counts = trip_count_pivot[vendor]
    
    # Create pie chart for this vendor
    axs[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    axs[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Add title with a line space
    axs[i].set_title(f'Vendor {vendor} - Trip Counts by Time of Day\n')

plt.tight_layout()
plt.show()
trip_count_pivot


# Group by time of day and vendor to calculate average trip duration
avg_trip_duration_by_timeofday_vendor = df.groupby(['vendor_id', 'pickup_datetime_timeofday'])['trip_duration'].mean().reset_index()

# Convert seconds to minutes for better readability
avg_trip_duration_by_timeofday_vendor['trip_duration'] = avg_trip_duration_by_timeofday_vendor['trip_duration'] / 60

# Pivot the data to have time of day as the index and vendor_id as columns
avg_trip_duration_pivot = avg_trip_duration_by_timeofday_vendor.pivot(index='pickup_datetime_timeofday', columns='vendor_id', values='trip_duration').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_duration_pivot))

# Plotting each vendor's average trip duration as bars
for i, vendor in enumerate(avg_trip_duration_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_duration_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Duration by Time of Day (by Vendor)')
plt.xlabel('Time of Day')
plt.ylabel('Average Trip Duration (minutes)')
plt.xticks([x + bar_width * (len(avg_trip_duration_pivot.columns) - 1) / 2 for x in r], avg_trip_duration_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_duration_pivot


# Group by time of day and vendor to calculate average trip distance
avg_trip_distance_by_timeofday_vendor = df.groupby(['vendor_id', 'pickup_datetime_timeofday'])['trip_distance'].mean().reset_index()

# Pivot the data to have time of day as the index and vendor_id as columns
avg_trip_distance_pivot = avg_trip_distance_by_timeofday_vendor.pivot(index='pickup_datetime_timeofday', columns='vendor_id', values='trip_distance').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_distance_pivot))

# Plotting each vendor's average trip distance as bars
for i, vendor in enumerate(avg_trip_distance_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_distance_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Distance by Time of Day (by Vendor)')
plt.xlabel('Time of Day')
plt.ylabel('Average Trip Distance (km)')
plt.xticks([x + bar_width * (len(avg_trip_distance_pivot.columns) - 1) / 2 for x in r], avg_trip_distance_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_distance_pivot


# Group by minute group and vendor to calculate the number of trips
trip_count_by_minutegroup_vendor = df.groupby(['vendor_id', 'pickup_datetime_minutegroup']).size().reset_index(name='trip_count')

# Pivot the data to have minute group as the index and vendor_id as columns
trip_count_pivot = trip_count_by_minutegroup_vendor.pivot(index='pickup_datetime_minutegroup', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(trip_count_pivot))

# Plotting each vendor's trip count as bars
for i, vendor in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Number of Trips by Minute Group (by Vendor)')
plt.xlabel('Minute Group')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], trip_count_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by minute group and vendor to calculate the number of trips
trip_count_by_minutegroup_vendor = df.groupby(['vendor_id', 'pickup_datetime_minutegroup']).size().reset_index(name='trip_count')

# Pivot the data to have minute group as the index and vendor_id as columns
trip_count_pivot = trip_count_by_minutegroup_vendor.pivot(index='pickup_datetime_minutegroup', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
num_vendors = len(trip_count_pivot.columns)
fig, axes = plt.subplots(1, num_vendors, figsize=(18, 6), sharey=True)

# Plotting each vendor's trip count as a pie chart
for i, vendor in enumerate(trip_count_pivot.columns):
    axes[i].pie(trip_count_pivot[vendor], labels=trip_count_pivot.index, autopct='%1.1f%%', startangle=90)
    axes[i].axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    # Add title with a line space
    axes[i].set_title(f'Vendor {vendor} - Trip Counts by Minute Group (by Vendor)\n')

plt.tight_layout()
plt.show()
trip_count_pivot


# Group by minute group and vendor to calculate average trip duration
avg_trip_duration_by_minutegroup_vendor = df.groupby(['vendor_id', 'pickup_datetime_minutegroup'])['trip_duration'].mean().reset_index()

# Pivot the data to have minute group as the index and vendor_id as columns
avg_trip_duration_pivot = avg_trip_duration_by_minutegroup_vendor.pivot(index='pickup_datetime_minutegroup', columns='vendor_id', values='trip_duration').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_duration_pivot))

# Plotting each vendor's average trip duration as bars
for i, vendor in enumerate(avg_trip_duration_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_duration_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Duration by Minute Group (by Vendor)')
plt.xlabel('Minute Group')
plt.ylabel('Average Trip Duration (minutes)')
plt.xticks([x + bar_width * (len(avg_trip_duration_pivot.columns) - 1) / 2 for x in r], avg_trip_duration_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_duration_pivot


# Group by minute group and vendor to calculate average trip distance
avg_trip_distance_by_minutegroup_vendor = df.groupby(['vendor_id', 'pickup_datetime_minutegroup'])['trip_distance'].mean().reset_index()

# Pivot the data to have minute group as the index and vendor_id as columns
avg_trip_distance_pivot = avg_trip_distance_by_minutegroup_vendor.pivot(index='pickup_datetime_minutegroup', columns='vendor_id', values='trip_distance').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_distance_pivot))

# Plotting each vendor's average trip distance as bars
for i, vendor in enumerate(avg_trip_distance_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_distance_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Distance by Minute Group (by Vendor)')
plt.xlabel('Minute Group')
plt.ylabel('Average Trip Distance (km)')
plt.xticks([x + bar_width * (len(avg_trip_distance_pivot.columns) - 1) / 2 for x in r], avg_trip_distance_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_distance_pivot


# Group by passenger count and vendor to calculate the number of trips
trip_count_by_passenger_count_vendor = df.groupby(['vendor_id', 'passenger_count']).size().reset_index(name='trip_count')

# Pivot the data to have passenger_count as the index and vendor_id as columns
trip_count_pivot = trip_count_by_passenger_count_vendor.pivot(index='passenger_count', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(trip_count_pivot))

# Plotting each vendor's trip count as bars
for i, vendor in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Number of Trips by Passenger Count (by Vendor)')
plt.xlabel('Passenger Count')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], trip_count_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by passenger count and vendor to calculate the number of trips
trip_count_by_passenger_count_vendor = df.groupby(['vendor_id', 'passenger_count']).size().reset_index(name='trip_count')

# Pivot the data to have passenger_count as the index and vendor_id as columns
trip_count_pivot = trip_count_by_passenger_count_vendor.pivot(index='passenger_count', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
num_vendors = len(trip_count_pivot.columns)
fig, axes = plt.subplots(1, num_vendors, figsize=(18, 6), sharey=True)

# Plotting each vendor's trip count as a pie chart
for i, vendor in enumerate(trip_count_pivot.columns):
    axes[i].pie(trip_count_pivot[vendor], labels=trip_count_pivot.index, autopct='%1.1f%%', startangle=90)
    axes[i].axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    # Add title with a line space
    axes[i].set_title(f'Vendor {vendor} - Trip Counts by Passenger Count (by Vendor)\n')

plt.tight_layout()
plt.show()
trip_count_pivot


# Group by passenger count and vendor to calculate average trip duration
avg_trip_duration_by_passenger_count_vendor = df.groupby(['vendor_id', 'passenger_count'])['trip_duration'].mean().reset_index()

# Convert trip duration from seconds to minutes for easier interpretation
avg_trip_duration_by_passenger_count_vendor['trip_duration'] /= 60

# Pivot the data to have passenger_count as the index and vendor_id as columns
avg_trip_duration_pivot = avg_trip_duration_by_passenger_count_vendor.pivot(index='passenger_count', columns='vendor_id', values='trip_duration').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_duration_pivot))

# Plotting each vendor's average trip duration as bars
for i, vendor in enumerate(avg_trip_duration_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_duration_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Duration by Passenger Count (by Vendor)')
plt.xlabel('Passenger Count')
plt.ylabel('Average Trip Duration (minutes)')
plt.xticks([x + bar_width * (len(avg_trip_duration_pivot.columns) - 1) / 2 for x in r], avg_trip_duration_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_duration_pivot


# Group by passenger count and vendor to calculate average trip distance
avg_trip_distance_by_passenger_count_vendor = df.groupby(['vendor_id', 'passenger_count'])['trip_distance'].mean().reset_index()

# Pivot the data to have passenger_count as the index and vendor_id as columns
avg_trip_distance_pivot = avg_trip_distance_by_passenger_count_vendor.pivot(index='passenger_count', columns='vendor_id', values='trip_distance').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_distance_pivot))

# Plotting each vendor's average trip distance as bars
for i, vendor in enumerate(avg_trip_distance_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_distance_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Distance by Passenger Count (by Vendor)')
plt.xlabel('Passenger Count')
plt.ylabel('Average Trip Distance (km)')
plt.xticks([x + bar_width * (len(avg_trip_distance_pivot.columns) - 1) / 2 for x in r], avg_trip_distance_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_distance_pivot


# Group by weekend status and vendor to calculate the number of trips
trip_count_by_weekend_vendor = df.groupby(['vendor_id', 'pickup_datetime_isweekend']).size().reset_index(name='trip_count')

# Pivot the data to have isweekend as the index and vendor_id as columns
trip_count_pivot = trip_count_by_weekend_vendor.pivot(index='pickup_datetime_isweekend', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(trip_count_pivot))

# Plotting each vendor's trip count as bars
for i, vendor in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Number of Trips on Weekdays vs. Weekends (by Vendor)')
plt.xlabel('Is Weekend (0 = Weekday, 1 = Weekend)')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], ['Weekday', 'Weekend'], rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by weekend status and vendor to calculate the number of trips
trip_count_by_weekend_vendor = df.groupby(['vendor_id', 'pickup_datetime_isweekend']).size().reset_index(name='trip_count')

# Pivot the data to have isweekend as the index and vendor_id as columns
trip_count_pivot = trip_count_by_weekend_vendor.pivot(index='pickup_datetime_isweekend', columns='vendor_id', values='trip_count').fillna(0)

# Plotting
num_vendors = len(trip_count_pivot.columns)
fig, axes = plt.subplots(1, num_vendors, figsize=(18, 6), sharey=True)

# Set labels for the pie chart
labels = ['Weekday', 'Weekend']

# Plotting each vendor's trip count as a pie chart
for i, vendor in enumerate(trip_count_pivot.columns):
    axes[i].pie(trip_count_pivot[vendor], labels=labels, autopct='%1.1f%%', startangle=90)
    axes[i].axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    # Add title with a line space
    axes[i].set_title(f'Vendor {vendor} - Trip Counts on Weekdays vs. Weekends\n')

plt.tight_layout()
plt.show()
trip_count_pivot


# Group by weekend status and vendor to calculate average trip duration
avg_trip_duration_by_weekend_vendor = df.groupby(['vendor_id', 'pickup_datetime_isweekend'])['trip_duration'].mean().reset_index()

# Pivot the data to have isweekend as the index and vendor_id as columns
avg_trip_duration_pivot = avg_trip_duration_by_weekend_vendor.pivot(index='pickup_datetime_isweekend', columns='vendor_id', values='trip_duration').fillna(0) / 60  # Convert to minutes

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_duration_pivot))

# Plotting each vendor's average trip duration as bars
for i, vendor in enumerate(avg_trip_duration_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_duration_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Duration on Weekdays vs. Weekends (by Vendor)')
plt.xlabel('Is Weekend (0 = Weekday, 1 = Weekend)')
plt.ylabel('Average Trip Duration (minutes)')
plt.xticks([x + bar_width * (len(avg_trip_duration_pivot.columns) - 1) / 2 for x in r], ['Weekday', 'Weekend'], rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_duration_pivot


# Group by weekend status and vendor to calculate average trip distance
avg_trip_distance_by_weekend_vendor = df.groupby(['vendor_id', 'pickup_datetime_isweekend'])['trip_distance'].mean().reset_index()

# Pivot the data to have isweekend as the index and vendor_id as columns
avg_trip_distance_pivot = avg_trip_distance_by_weekend_vendor.pivot(index='pickup_datetime_isweekend', columns='vendor_id', values='trip_distance').fillna(0)  # No need to fill with 0 as the mean will handle it

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bar
bar_width = 0.35
# Set positions of bar on X axis
r = range(len(avg_trip_distance_pivot))

# Plotting each vendor's average trip distance as bars
for i, vendor in enumerate(avg_trip_distance_pivot.columns):
    plt.bar([x + bar_width * i for x in r], avg_trip_distance_pivot[vendor], width=bar_width, label=f'Vendor {vendor}', alpha=0.6)

# Set the title and labels
plt.title('Average Trip Distance on Weekdays vs. Weekends (by Vendor)')
plt.xlabel('Is Weekend (0 = Weekday, 1 = Weekend)')
plt.ylabel('Average Trip Distance (km)')
plt.xticks([x + bar_width * (len(avg_trip_distance_pivot.columns) - 1) / 2 for x in r], ['Weekday', 'Weekend'], rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Vendor ID')
plt.tight_layout()
plt.show()
avg_trip_distance_pivot


# Get unique vendor IDs
vendors = df['vendor_id'].unique()

# Create a heatmap for each vendor
for vendor in vendors:
    # Filter the data for the current vendor
    vendor_data = df[df['vendor_id'] == vendor]
    
    # Create the pivot table for average trip duration
    pivot_table = vendor_data.pivot_table(values='trip_duration', index='pickup_datetime_weekday', columns='pickup_datetime_hour', aggfunc='mean')

    # Create the heatmap
    plt.figure(figsize=(18, 6))
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".0f")
    plt.title(f"Average Trip Duration by Hour and Day of the Week for Vendor {vendor}")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Day of the Week")
    plt.xticks(ticks=range(24), labels=range(24))
    plt.yticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.tight_layout()
    plt.show()


# Calculate the number of trips per day for each passenger count
daily_trip_count_by_passenger = df.groupby(['passenger_count', 'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day']).size().reset_index(name='trip_count')

# Create a 'Date' column to represent the date as a datetime object for sorting purposes
daily_trip_count_by_passenger['Date'] = pd.to_datetime(
    daily_trip_count_by_passenger['pickup_datetime_year'].astype(str) + '-' + 
    daily_trip_count_by_passenger['pickup_datetime_month'].astype(str) + '-' + 
    daily_trip_count_by_passenger['pickup_datetime_day'].astype(str)
)

# Sort by date
daily_trip_count_by_passenger = daily_trip_count_by_passenger.sort_values('Date')

# Remove the last day from the data
daily_trip_count_by_passenger = daily_trip_count_by_passenger[daily_trip_count_by_passenger['Date'] != daily_trip_count_by_passenger['Date'].max()]

# Plotting for each passenger count
plt.figure(figsize=(18, 6))

# Loop through each passenger count to plot their daily trip counts
for passenger_count in daily_trip_count_by_passenger['passenger_count'].unique():
    passenger_data = daily_trip_count_by_passenger[daily_trip_count_by_passenger['passenger_count'] == passenger_count]
    
    # Plot trip counts for the current passenger count
    plt.plot(passenger_data['Date'], passenger_data['trip_count'], label=f'Passengers {passenger_count}', linewidth=1.25)

# Set x-ticks to show every day
all_dates = daily_trip_count_by_passenger['Date'].unique()  # Get unique dates for x-ticks
plt.xticks(ticks=all_dates[::5], labels=[date.strftime('%Y-%m-%d') for date in all_dates[::5]], rotation=90)  # Display every second day

# Set the title, labels, and legend
plt.title('Number of Trips Per Day by Passenger Count (All Vendors)')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.legend(title='Passenger Count', bbox_to_anchor=(1.05, 1), loc='upper left')  # Position legend outside the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Calculate the number of trips per week for each passenger count
weekly_trip_count_by_passenger = df.groupby(['passenger_count', 'pickup_datetime_year', 'pickup_datetime_weekofyear']).size().reset_index(name='trip_count')

# Create a 'Week' column to represent the week as a datetime object for sorting purposes
weekly_trip_count_by_passenger['Week'] = pd.to_datetime(
    weekly_trip_count_by_passenger['pickup_datetime_year'].astype(str) + '-W' + 
    weekly_trip_count_by_passenger['pickup_datetime_weekofyear'].astype(str) + '-1', 
    format="%Y-W%W-%w"
)

# Sort by week
weekly_trip_count_by_passenger = weekly_trip_count_by_passenger.sort_values('Week')

# Remove the last week from the data
weekly_trip_count_by_passenger = weekly_trip_count_by_passenger[weekly_trip_count_by_passenger['Week'] != weekly_trip_count_by_passenger['Week'].max()]

# Plotting for each passenger count
plt.figure(figsize=(18, 6))

# Loop through each passenger count to plot their weekly trip counts
for passenger_count in weekly_trip_count_by_passenger['passenger_count'].unique():
    passenger_data = weekly_trip_count_by_passenger[weekly_trip_count_by_passenger['passenger_count'] == passenger_count]
    
    # Plot trip counts for the current passenger count
    plt.plot(passenger_data['Week'], passenger_data['trip_count'], label=f'Passengers {passenger_count}', linewidth=1.25)

# Set x-ticks to show every week
all_weeks = weekly_trip_count_by_passenger['Week'].unique()  # Get unique weeks for x-ticks
plt.xticks(ticks=all_weeks, labels=[week.strftime('%Y-W%U') for week in all_weeks], rotation=90)  # Display every second week

# Set the title, labels, and legend
plt.title('Number of Trips Per Week by Passenger Count (All Vendors)')
plt.xlabel('Week (YYYY-WW)')
plt.ylabel('Number of Trips')
plt.legend(title='Passenger Count', bbox_to_anchor=(1.05, 1), loc='upper left')  # Position legend outside the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Calculate the number of trips per hour for each passenger count
hourly_trip_count_by_passenger = df.groupby(['passenger_count', 'pickup_datetime_hour']).size().reset_index(name='trip_count')

# Sort by hour
hourly_trip_count_by_passenger = hourly_trip_count_by_passenger.sort_values(['pickup_datetime_hour', 'passenger_count'])

# Plotting for each passenger count
plt.figure(figsize=(18, 6))

# Loop through each passenger count to plot their hourly trip counts
for passenger_count in hourly_trip_count_by_passenger['passenger_count'].unique():
    passenger_data = hourly_trip_count_by_passenger[hourly_trip_count_by_passenger['passenger_count'] == passenger_count]
    
    # Plot trip counts for the current passenger count
    plt.plot(passenger_data['pickup_datetime_hour'], passenger_data['trip_count'], label=f'Passengers {passenger_count}', linewidth=1.25)

# Set x-ticks to show every hour
plt.xticks(ticks=range(24), labels=range(24), rotation=0)  # Display each hour

# Set the title, labels, and legend
plt.title('Number of Trips Per Hour by Passenger Count (All Vendors)')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.legend(title='Passenger Count', bbox_to_anchor=(1.05, 1), loc='upper left')  # Position legend outside the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Group by month and passenger count to calculate the number of trips
trip_count_by_month_passenger = df.groupby(['pickup_datetime_month', 'passenger_count']).size().reset_index(name='trip_count')

# Pivot the data to have month as the index and passenger_count as columns
trip_count_pivot = trip_count_by_month_passenger.pivot(index='pickup_datetime_month', columns='passenger_count', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bars
bar_width = 0.15
# Set positions of bars on X axis
r = range(len(trip_count_pivot))

# Plotting each passenger count's trip count as bars
for i, passenger_count in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[passenger_count], width=bar_width, label=f'Passenger Count: {passenger_count}', alpha=0.6)

# Set title and labels
plt.title('Number of Trips by Month for Each Passenger Count')
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], trip_count_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Passenger Count')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by season and passenger count to calculate the number of trips
trip_count_by_season_passenger = df.groupby(['pickup_datetime_season', 'passenger_count']).size().reset_index(name='trip_count')

# Pivot the data to have season as the index and passenger_count as columns
trip_count_pivot = trip_count_by_season_passenger.pivot(index='pickup_datetime_season', columns='passenger_count', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bars
bar_width = 0.15
# Set positions of bars on X axis
r = range(len(trip_count_pivot))

# Plotting each passenger count's trip count as bars
for i, passenger_count in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[passenger_count], width=bar_width, label=f'Passenger Count: {passenger_count}', alpha=0.6)

# Set title and labels
plt.title('Number of Trips by Season for Each Passenger Count')
plt.xlabel('Season')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], trip_count_pivot.index, rotation=0)  # Center the ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Passenger Count')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by pickup quarter and passenger count to calculate the number of trips
trip_count_by_quarter_passenger = df.groupby(['pickup_datetime_quarter', 'passenger_count']).size().reset_index(name='trip_count')

# Pivot the data to have quarter as the index and passenger_count as columns
trip_count_pivot = trip_count_by_quarter_passenger.pivot(index='pickup_datetime_quarter', columns='passenger_count', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bars
bar_width = 0.15
# Set positions of bars on X axis
r = range(len(trip_count_pivot))

# Plotting each passenger count's trip count as bars
for i, passenger_count in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[passenger_count], width=bar_width, label=f'Passenger Count: {passenger_count}', alpha=0.6)

# Set title and labels
plt.title('Number of Trips by Quarter for Each Passenger Count')
plt.xlabel('Quarter')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], [f'Q{i}' for i in df['pickup_datetime_quarter'].unique()], rotation=0)  # Center the ticks with quarter labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Passenger Count')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by day of the week and passenger count to calculate the number of trips
trip_count_by_day_passenger = df.groupby(['pickup_datetime_weekday', 'passenger_count']).size().reset_index(name='trip_count')

# Pivot the data to have day of the week as the index and passenger_count as columns
trip_count_pivot = trip_count_by_day_passenger.pivot(index='pickup_datetime_weekday', columns='passenger_count', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bars
bar_width = 0.15
# Set positions of bars on X axis
r = range(len(trip_count_pivot))

# Plotting each passenger count's trip count as bars
for i, passenger_count in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[passenger_count], width=bar_width, 
            label=f'Passenger Count: {passenger_count}', alpha=0.6)

# Set title and labels
plt.title('Number of Trips by Day of the Week for Each Passenger Count')
plt.xlabel('Day of the Week (0=Mon, 6=Sun)')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], 
           ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)  # Center the ticks with day names
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Passenger Count')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by time of day (hour) and passenger count to calculate the number of trips
trip_count_by_time_passenger = df.groupby(['pickup_datetime_timeofday', 'passenger_count']).size().reset_index(name='trip_count')

# Pivot the data to have time of day (hour) as the index and passenger_count as columns
trip_count_pivot = trip_count_by_time_passenger.pivot(index='pickup_datetime_timeofday', columns='passenger_count', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bars
bar_width = 0.15
# Set positions of bars on X axis
r = range(len(trip_count_pivot))

# Plotting each passenger count's trip count as bars
for i, passenger_count in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[passenger_count], width=bar_width, 
            label=f'Passenger Count: {passenger_count}', alpha=0.6)

# Set title and labels
plt.title('Number of Trips by Time of Day for Each Passenger Count')
plt.xlabel('Time of Day (Hour)')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], 
           [hour for hour in trip_count_pivot.index], rotation=0)  # Center the ticks with hour labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Passenger Count')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by minute group and passenger count to calculate the number of trips
trip_count_by_minutegroup_passenger = df.groupby(['pickup_datetime_minutegroup', 'passenger_count']).size().reset_index(name='trip_count')

# Pivot the data to have minute group as the index and passenger_count as columns
trip_count_pivot = trip_count_by_minutegroup_passenger.pivot(index='pickup_datetime_minutegroup', columns='passenger_count', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bars
bar_width = 0.15
# Set positions of bars on X axis
r = range(len(trip_count_pivot))

# Plotting each passenger count's trip count as bars
for i, passenger_count in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[passenger_count], width=bar_width, 
            label=f'Passenger Count: {passenger_count}', alpha=0.6)

# Set title and labels
plt.title('Number of Trips by Minute Group for Each Passenger Count')
plt.xlabel('Minute Group')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], 
           [minute for minute in trip_count_pivot.index], rotation=0)  # Center the ticks with minute group labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Passenger Count')
plt.tight_layout()
plt.show()
trip_count_pivot


# Group by isweekend and passenger count to calculate the number of trips
trip_count_by_weekend_passenger = df.groupby(['pickup_datetime_isweekend', 'passenger_count']).size().reset_index(name='trip_count')

# Pivot the data to have isweekend as the index and passenger_count as columns
trip_count_pivot = trip_count_by_weekend_passenger.pivot(index='pickup_datetime_isweekend', columns='passenger_count', values='trip_count').fillna(0)

# Plotting
plt.figure(figsize=(18, 6))

# Set width of bars
bar_width = 0.15
# Set positions of bars on X axis
r = range(len(trip_count_pivot))

# Plotting each passenger count's trip count as bars
for i, passenger_count in enumerate(trip_count_pivot.columns):
    plt.bar([x + bar_width * i for x in r], trip_count_pivot[passenger_count], width=bar_width, 
            label=f'Passenger Count: {passenger_count}', alpha=0.6)

# Set title and labels
plt.title('Number of Trips by Weekend/Weekday for Each Passenger Count')
plt.xlabel('Weekend (True/False)')
plt.ylabel('Number of Trips')
plt.xticks([x + bar_width * (len(trip_count_pivot.columns) - 1) / 2 for x in r], 
           ['Weekday', 'Weekend'], rotation=0)  # Center the ticks with labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Passenger Count')
plt.tight_layout()
plt.show()
trip_count_pivot


# Set the plot size and style
plt.figure(figsize=(8, 8))
sns.set_style("whitegrid")

# Scatter plot of trip duration vs. trip distance, colored by vendor
sns.scatterplot(
    data=df,
    x=df['trip_distance'], 
    y=df['trip_duration'] / 60, 
    hue='vendor_id', 
    palette='viridis', 
    alpha=0.6, 
    edgecolor=None
)

# Set plot title and labels
plt.title('Relationship Between Trip Duration and Trip Distance by Vendor')
plt.xlabel('Trip Distance (km)')
plt.ylabel('Trip Duration (minutes)')
plt.legend(title='Vendor ID')
plt.xlim(0, 60)
plt.ylim(0, 180)
plt.tight_layout()
plt.show()


!pip install basemap basemap-data-hires


from mpl_toolkits.basemap import Basemap

# Set up the map, focusing on Japan region as an example
plt.figure(figsize=(8, 16))
lat, height, lon, width = 29, 30, -129, 60
m = Basemap(projection='merc', llcrnrlat=lat, urcrnrlat=lat+height, llcrnrlon=lon, urcrnrlon=lon+width, resolution='i')

# Draw coastlines, countries, and map boundaries
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgray', lake_color='lightblue')
m.drawparallels(range(lat, lat+height, 6), labels=[1, 0, 0, 0])
meridians = m.drawmeridians(range(lon, lon+width, 6), labels=[0, 0, 0, 1])

# Rotate longitude labels
for meridian in meridians:
    try:
        for label in meridians[meridian][1]:
            label.set_rotation(90)
    except:
        continue

# Convert lat and long to map projection coordinates
vendor_colors = {1: 'blue', 2: 'green'}  # Adjust based on your vendor IDs

for vendor_id, color in vendor_colors.items():
    vendor_data = df[df['vendor_id'] == vendor_id]
    x, y = m(vendor_data['pickup_longitude'].values, vendor_data['pickup_latitude'].values)
    m.scatter(x, y, color=color, label=f'Vendor {vendor_id}', s=10)

plt.title('Pickup Points by Vendor')
plt.legend(loc='upper right')
plt.xticks(rotation=90)
plt.show()


from mpl_toolkits.basemap import Basemap

# Set up the map, focusing on Japan region as an example
plt.figure(figsize=(8, 16))
lat, height, lon, width = 29, 30, -129, 60
m = Basemap(projection='merc', llcrnrlat=lat, urcrnrlat=lat+height, llcrnrlon=lon, urcrnrlon=lon+width, resolution='i')


# Draw coastlines, countries, and map boundaries
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgray', lake_color='lightblue')
m.drawparallels(range(lat, lat+height, 6), labels=[1, 0, 0, 0])
meridians = m.drawmeridians(range(lon, lon+width, 6), labels=[0, 0, 0, 1])

# Rotate longitude labels
for meridian in meridians:
    try:
        for label in meridians[meridian][1]:
            label.set_rotation(90)
    except:
        continue

# Plot drop-off points for each vendor
vendor_colors = {1: 'blue', 2: 'green'}  # Adjust based on your vendor IDs

for vendor_id, color in vendor_colors.items():
    vendor_data = df[df['vendor_id'] == vendor_id]
    x, y = m(vendor_data['dropoff_longitude'].values, vendor_data['dropoff_latitude'].values)
    m.scatter(x, y, color=color, label=f'Vendor {vendor_id}', s=10)

plt.title('Drop-off Points by Vendor')
plt.legend(loc='upper right')
plt.show()

