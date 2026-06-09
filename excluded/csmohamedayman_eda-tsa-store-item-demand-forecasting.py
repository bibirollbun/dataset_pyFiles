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


dir_path = '/kaggle/input/demand-forecasting-kernels-only'
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


target_feature = list(set(train.columns)-set(test.columns))[0]
target_feature


df_dtypes(train)


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
test = test.drop(columns=list(dropped_nan - set([target_feature])))


for feature in ['date']:
    train = extract_date_features(train, [feature])
    test = extract_date_features(test, [feature])
    train = extract_time_features(train, [feature])
    test = extract_time_features(test, [feature])
    train[feature] = train[feature].astype(int) / 10**18
    test[feature] = test[feature].astype(int) / 10**18


df_dtypes(train)


df_dtypes(test)


train = fillna_and_replace_inf(train)
test = fillna_and_replace_inf(test)


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


#train, _ = drop_id_feature(train, 'id')
test, test_id = drop_id_feature(test, 'id')


train = train.drop(columns=train.select_dtypes(exclude=[np.number]).columns)
test = test.drop(columns=test.select_dtypes(exclude=[np.number]).columns)


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


if 'item' in categorical_features: categorical_features.remove('item')
if 'item' in numerical_features:   numerical_features.remove('item')
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


plot_numerical_features(original_train[numerical_features])


plot_categorical_features(original_train[categorical_features])


for feature in []:
    plot_categorical_features(original_train[[feature]])
    original_train = replace_rare_categories(original_train, feature, 'Other', threshold_percent=2)
    #train          = replace_rare_categories(train, feature, -1, threshold_percent=2)
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


# Group sales by 'date_year', 'date_month', and 'store'
store_monthly_sales = df.groupby(['date_year', 'date_month', 'store'])['sales'].sum().unstack()
# Combine year and month into a proper datetime index
store_monthly_sales.index = pd.to_datetime(store_monthly_sales.index.map(lambda x: f"{x[0]}-{x[1]:02d}"))
# Plotting sales for each store separately
store_monthly_sales.plot(kind='line', figsize=(18, 6), marker='o')
# Get the positions for the x-ticks (every months)
tick_positions = store_monthly_sales.index
# Set the x-ticks and labels
plt.xticks(ticks=tick_positions, labels=tick_positions.strftime('%b %Y'), rotation=90)
# Set the title and labels
plt.title('Monthly Sales Trends by Store')
plt.xlabel('Month-Year')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Group sales by 'date_year' and 'store'
store_yearly_sales = df.groupby(['date_year', 'store'])['sales'].sum().unstack()
# Sum sales across all years for each store
total_sales_by_store = store_yearly_sales.sum(axis=0)
# Create a pie chart comparing total sales between stores
plt.figure(figsize=(6, 6))
plt.pie(total_sales_by_store, labels=total_sales_by_store.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Total Sales Distribution Across Stores')
plt.tight_layout()
plt.show()
total_sales_by_store


# Group sales by 'date_year' and 'store'
store_yearly_sales = df.groupby(['date_year', 'store'])['sales'].sum().unstack()
# Convert the index to a string for clean labeling
store_yearly_sales.index = store_yearly_sales.index.map(str)  # Convert year to string for clean labeling

# Plotting sales for each store as a bar chart
store_yearly_sales.plot(kind='bar', figsize=(18, 6))

# Set the x-ticks and labels
plt.xticks(ticks=range(len(store_yearly_sales.index)), 
           labels=store_yearly_sales.index, rotation=0)  # Keep labels horizontal

# Set the title and labels
plt.title('Yearly Sales Trends by Store')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
store_yearly_sales


# Aggregate sales data for all stores across years
total_sales_by_year = store_yearly_sales.sum(axis=1)  # Sum sales for all stores for each year
# Convert year to string for clean labeling
years = total_sales_by_year.index.map(str)
# Create a single pie chart for the total sales distribution across all stores
plt.figure(figsize=(6, 6))
plt.pie(total_sales_by_year, labels=years, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Total Sales Distribution Across Years')
plt.tight_layout()
plt.show()
total_sales_by_year


# Group sales by 'date_year', 'date_month', and 'item'
item_monthly_sales = df.groupby(['date_year', 'date_month', 'item'])['sales'].sum().unstack()
# Combine year and month into a proper datetime index
item_monthly_sales.index = pd.to_datetime(item_monthly_sales.index.map(lambda x: f"{x[0]}-{x[1]:02d}"))
# Sum sales across all years for each item
total_sales_per_item = item_monthly_sales.sum(axis=0)
# Get the top 10 items based on total sales
top_items = total_sales_per_item.nlargest(10).index
# Filter for the top 10 items
top_item_monthly_sales = item_monthly_sales[top_items]
# Plotting sales for the top 10 items separately as a horizontal bar chart
top_item_monthly_sales.plot(kind='line', figsize=(18, 6))
# Get the positions for the x-ticks (every month)
tick_positions = store_monthly_sales.index
# Set the x-ticks and labels
plt.xticks(ticks=tick_positions, labels=tick_positions.strftime('%b %Y'), rotation=90)
# Set the title and labels
plt.title('Monthly Sales Trends for Top 10 Items')
plt.xlabel('Month-Year')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Group sales by 'date_year', 'date_month', and 'item'
item_monthly_sales = df.groupby(['date_year', 'date_month', 'item'])['sales'].sum().unstack()
# Combine year and month into a proper datetime index
item_monthly_sales.index = pd.to_datetime(item_monthly_sales.index.map(lambda x: f"{x[0]}-{x[1]:02d}"))
# Get the top 200 items based on total sales
top_items = item_monthly_sales.sum(axis=0).nlargest(215).index
# Filter for the top 30 items
top_item_monthly_sales = item_monthly_sales[top_items]

# Group into three sets
grouped_item_sales = []
group_items = top_items[0:35]
group_sales = top_item_monthly_sales[group_items].sum(axis=1)
grouped_item_sales.append(group_sales)
group_items = top_items[35:215]
group_sales = top_item_monthly_sales[group_items].sum(axis=1)
grouped_item_sales.append(group_sales)

# Create a DataFrame to hold the grouped sales
grouped_item_sales_df = pd.DataFrame({
    'Top 35 Items': grouped_item_sales[0],
    'Last 180 Items': grouped_item_sales[1]
})
# Plotting the grouped sales as a bar chart
grouped_item_sales_df.plot(kind='bar', figsize=(18, 6), width=0.8)
# Set the x-ticks and labels
tick_positions = grouped_item_sales_df.index
plt.xticks(ticks=range(len(tick_positions)), 
           labels=tick_positions.strftime('%b %Y'), rotation=90)
# Set the title and labels
plt.title('Monthly Sales Trends for Top 35 vs. Last 180 Sold Items')
plt.xlabel('Month-Year')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Group sales by 'date_year' and 'item'
item_yearly_sales = df.groupby(['date_year', 'item'])['sales'].sum().unstack()
# Get the top 200 items based on total sales
top_items = item_yearly_sales.sum(axis=0).nlargest(215).index
# Filter for the top items
top_item_yearly_sales = item_yearly_sales[top_items]

# Group into three sets
grouped_item_sales = []
group_items = top_items[0:35]
group_sales = top_item_yearly_sales[group_items].sum(axis=1)
grouped_item_sales.append(group_sales)
group_items = top_items[35:215]
group_sales = top_item_yearly_sales[group_items].sum(axis=1)
grouped_item_sales.append(group_sales)

# Create a DataFrame to hold the grouped sales
grouped_item_sales_df = pd.DataFrame({
    'Top 35 Items': grouped_item_sales[0],
    'Last 180 Items': grouped_item_sales[1]
})
# Plotting the grouped sales as a bar chart
grouped_item_sales_df.plot(kind='bar', figsize=(18, 6))
# Set the x-ticks and labels
plt.xticks(ticks=range(len(grouped_item_sales_df.index)), 
           labels=grouped_item_sales_df.index, rotation=0)
# Set the title and labels
plt.title('Yearly Sales Trends for Top 30 vs. Last 180 Sold Items')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
grouped_item_sales_df


# Aggregate sales data for all years
total_sales_data = grouped_item_sales_df.sum(axis=0)  # Sum sales across all years
# Create a single pie chart for the total sales distribution
plt.figure(figsize=(6, 6))
plt.pie(total_sales_data, labels=total_sales_data.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Total Sales Distribution for Top 35 vs. Last 180 Sold Items')
plt.tight_layout()
plt.show()
total_sales_data


# Group sales by 'date_weekday' and 'store'
daily_sales_per_store = df.groupby(['date_weekday', 'store'])['sales'].sum().unstack()
# Plotting sales for each store as a separate bar chart
daily_sales_per_store.plot(kind='bar', figsize=(18, 6))
# Set the title and labels
plt.title('Sales by Day of the Week for Each Store')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Total Sales')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
daily_sales_per_store


# Sum the sales across all stores for each day of the week
total_sales_per_day = daily_sales_per_store.sum(axis=1)
# Days of the week labels
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Create a pie chart for all stores combined
plt.figure(figsize=(6, 6))
plt.pie(total_sales_per_day, labels=days_of_week, autopct='%1.1f%%', startangle=90, counterclock=False)
# Set the title
plt.title('Total Sales Distribution by Day of the Week for All Stores')
plt.tight_layout()
plt.show()
total_sales_per_day


# Group sales by 'date_month' and 'store'
monthly_sales_per_store = df.groupby(['date_month', 'store'])['sales'].sum().unstack()
# Plotting sales for each store as a separate bar chart
monthly_sales_per_store.plot(kind='bar', figsize=(18, 6))
# Set the title and labels
plt.title('Sales by Month of the Year for Each Store')
plt.xlabel('Month of the Year (1=January, 12=December)')
plt.ylabel('Total Sales')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
monthly_sales_per_store


# Sum the sales across all stores for each month
total_sales_per_month = monthly_sales_per_store.sum(axis=1)
# Months of the year labels
months_of_year = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
# Create a pie chart for all stores combined
plt.figure(figsize=(6, 6))
plt.pie(total_sales_per_month, labels=months_of_year, autopct='%1.1f%%', startangle=90, counterclock=False)
# Set the title
plt.title('Total Sales Distribution by Month for All Stores')
plt.tight_layout()
plt.show()
total_sales_per_month


# Group sales by 'date_quarter' and 'store'
quarterly_sales_per_store = df.groupby(['date_quarter', 'store'])['sales'].sum().unstack()
# Plotting sales for each store as a separate bar chart
quarterly_sales_per_store.plot(kind='bar', figsize=(18, 6))
# Set the x-ticks and labels
plt.xticks(ticks=range(len(quarterly_sales_per_store.index)), 
           labels=['Q1', 'Q2', 'Q3', 'Q4'], rotation=0)
# Set the title and labels
plt.title('Sales by Quarter of the Year for Each Store')
plt.xlabel('Quarter (Q1, Q2, Q3, Q4)')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
quarterly_sales_per_store


# Sum the sales across all stores for each quarter
total_sales_per_quarter = quarterly_sales_per_store.sum(axis=1)
# Quarters of the year labels
quarters_of_year = ['Q1', 'Q2', 'Q3', 'Q4']
# Create a pie chart for all stores combined
plt.figure(figsize=(6, 6))
plt.pie(total_sales_per_quarter, labels=quarters_of_year, autopct='%1.1f%%', startangle=90, counterclock=False)
# Set the title
plt.title('Total Sales Distribution by Quarter for All Stores')
plt.tight_layout()
plt.show()
total_sales_per_quarter


# Group sales by 'season' and 'store'
seasonal_sales_per_store = df.groupby(['date_season', 'store'])['sales'].sum().unstack()
# Plotting sales for each store as a separate bar chart
seasonal_sales_per_store.plot(kind='bar', figsize=(18, 6))
# Set the x-ticks and labels
plt.xticks(ticks=range(len(seasonal_sales_per_store.index)), 
           labels=seasonal_sales_per_store.index, rotation=0)
# Set the title and labels
plt.title('Sales by Season of the Year for Each Store')
plt.xlabel('Season')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
seasonal_sales_per_store


# Sum the sales across all stores for each season
total_sales_per_season = seasonal_sales_per_store.sum(axis=1)
# Seasons labels
seasons = seasonal_sales_per_store.index  # Assuming seasons are already defined (e.g., Winter, Spring, Summer, Fall)
# Create a pie chart for all stores combined
plt.figure(figsize=(6, 6))
plt.pie(total_sales_per_season, labels=seasons, autopct='%1.1f%%', startangle=90, counterclock=False)
# Set the title
plt.title('Total Sales Distribution by Season for All Stores')
plt.tight_layout()
plt.show()
total_sales_per_season


# Group by year and month to calculate monthly sales
monthly_sales = df.groupby(['date_year', 'date_month'])['sales'].sum()
# Calculate monthly percentage change to measure volatility
monthly_sales_volatility = monthly_sales.pct_change().fillna(0)
# Create a new index for Month-Year
monthly_sales_volatility.index = pd.to_datetime(monthly_sales_volatility.index.map(lambda x: f"{x[0]}-{x[1]:02d}-01"))
# Plot sales volatility on a monthly basis
plt.figure(figsize=(18, 6))
plt.plot(monthly_sales_volatility.index, monthly_sales_volatility, color='orange', marker='o')
# Set the x-ticks and labels
tick_positions = monthly_sales_volatility.index
plt.xticks(ticks=tick_positions, labels=tick_positions.strftime('%b %Y'), rotation=90)
# Set the title and labels
plt.title('Monthly Sales Volatility')
plt.ylabel('Percentage Change')
plt.xlabel('Month-Year')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Group by month-year, and weekend/weekday status to compare sales
weekend_sales = df.groupby(['date_year', 'date_month', 'date_isweekend'])['sales'].sum().unstack()
# Combine year and month into a proper datetime index for easier plotting
weekend_sales.index = pd.to_datetime(weekend_sales.index.map(lambda x: f"{x[0]}-{x[1]:02d}"))
# Plotting average sales on weekdays vs weekends
plt.figure(figsize=(18, 6))
for store in weekend_sales.columns:
    plt.plot(weekend_sales.index, weekend_sales[store], marker='o', label=f'Store {store}', linestyle='-')
# Set the x-ticks and labels
tick_positions = weekend_sales.index
plt.xticks(ticks=tick_positions, labels=tick_positions.strftime('%b %Y'), rotation=90)
# Set the title and labels
plt.legend(title='Weekend Status', labels=['Weekday (0)', 'Weekend (1)'], loc='upper left')
plt.title('Monthly Sales by Weekdays vs Weekends')
plt.xlabel('Month-Year')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Group by month and weekend/weekday status to compare sales for each store
weekend_sales = df.groupby(['date_month', 'date_isweekend'])['sales'].sum().unstack()
# Plotting sales on weekdays vs weekends as a bar chart
weekend_sales.plot(kind='bar', color=['navy', 'orange'], figsize=(18, 6))
# Set the title and labels
plt.legend(title='Weekend Status', labels=['Weekday (0)', 'Weekend (1)'], loc='upper left')
plt.title('Sales by Weekdays vs Weekends for Each Month of the Year')
plt.xlabel('Month of the Year (1=January, 12=December)')
plt.ylabel('Total Sales')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
weekend_sales


# Group by month and weekend/weekday status to get total sales
weekend_sales = df.groupby(['date_month', 'date_isweekend'])['sales'].sum().unstack()
# Define months for labeling
months_of_year = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# First pie chart for Weekdays
axes[0].pie(weekend_sales[0], labels=months_of_year, autopct='%1.1f%%', startangle=90, counterclock=False)
axes[0].set_title('Sales Distribution on Weekdays by Month')
# Second pie chart for Weekends
axes[1].pie(weekend_sales[1], labels=months_of_year, autopct='%1.1f%%', startangle=90, counterclock=False)
axes[1].set_title('Sales Distribution on Weekends by Month')
# Set overall title
plt.suptitle('Sales Distribution by Month: Weekdays vs Weekends')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
weekend_sales


# Group by quarter and weekend/weekday status
weekend_sales = df.groupby(['date_quarter', 'date_isweekend'])['sales'].sum().unstack()
# Plotting sales on weekdays vs weekends as a bar chart
weekend_sales.plot(kind='bar', color=['navy', 'orange'], figsize=(18, 6))
# Set the x-ticks and labels
plt.xticks(ticks=range(len(weekend_sales.index)), 
           labels=['Q1', 'Q2', 'Q3', 'Q4'], rotation=0)
# Set the title and labels
plt.legend(title='Weekend Status', labels=['Weekday (0)', 'Weekend (1)'], loc='upper left')
plt.title('Sales by Weekdays vs Weekends for Each Quarter of the Year')
plt.xlabel('Quarter of the Year (Q1, Q2, Q3, Q4)')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
weekend_sales


# Group by quarter and weekend/weekday status to get total sales
weekend_sales = df.groupby(['date_quarter', 'date_isweekend'])['sales'].sum().unstack()
# Define quarters for labeling
quarters_of_year = ['Q1', 'Q2', 'Q3', 'Q4']
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# First pie chart for Weekdays
axes[0].pie(weekend_sales[0], labels=quarters_of_year, autopct='%1.1f%%', startangle=90, counterclock=False)
axes[0].set_title('Sales Distribution on Weekdays by Quarter')
# Second pie chart for Weekends
axes[1].pie(weekend_sales[1], labels=quarters_of_year, autopct='%1.1f%%', startangle=90, counterclock=False)
axes[1].set_title('Sales Distribution on Weekends by Quarter')
# Set overall title
plt.suptitle('Sales Distribution by Quarter: Weekdays vs Weekends')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
weekend_sales


# Group by season and weekend/weekday status
weekend_sales = df.groupby(['date_season', 'date_isweekend'])['sales'].sum().unstack()
# Plotting sales on weekdays vs weekends as a bar chart
weekend_sales.plot(kind='bar', color=['navy', 'orange'], figsize=(18, 6))
# Set the x-ticks and labels
plt.xticks(ticks=range(len(weekend_sales.index)), 
           labels=weekend_sales.index, rotation=0)
# Set the title and labels
plt.legend(title='Weekend Status', labels=['Weekday (0)', 'Weekend (1)'], loc='upper left')
plt.title('Sales by Weekdays vs Weekends for Each Season of the Year')
plt.xlabel('Season of the Year')
plt.ylabel('Total Sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
weekend_sales


# Group by season and weekend/weekday status to get total sales
weekend_sales = df.groupby(['date_season', 'date_isweekend'])['sales'].sum().unstack()
# Define seasons for labeling
seasons = weekend_sales.index.tolist()  # e.g., ['Winter', 'Spring', 'Summer', 'Fall']
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# First pie chart for Weekdays
axes[0].pie(weekend_sales[0], labels=seasons, autopct='%1.1f%%', startangle=90, counterclock=False)
axes[0].set_title('Sales Distribution on Weekdays by Season')
# Second pie chart for Weekends
axes[1].pie(weekend_sales[1], labels=seasons, autopct='%1.1f%%', startangle=90, counterclock=False)
axes[1].set_title('Sales Distribution on Weekends by Season')
# Set overall title
plt.suptitle('Sales Distribution by Season: Weekdays vs Weekends')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
weekend_sales

