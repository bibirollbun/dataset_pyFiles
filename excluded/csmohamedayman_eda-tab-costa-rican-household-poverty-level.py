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


dir_path = '/kaggle/input/costa-rican-household-poverty-prediction'
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


info_table[info_table['dtype'].astype(str).str.startswith(('int'))].head(60)


info_table[info_table['dtype'].astype(str).str.startswith(('int'))].tail(60)


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


train, _ = drop_id_feature(train, 'Id')
test, test_id = drop_id_feature(test, 'Id')


train = train.drop(columns=['edjefe', 'edjefa'])
test  = test.drop(columns=['edjefe', 'edjefa'])
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


features = ['rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv', 'hhsize', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total', 'bedrooms', 'qmobilephone', 'SQBhogar_total', 'SQBhogar_nin']
for feature in features:
    plot_categorical_features(original_train[[feature]])
    original_train = replace_rare_categories(original_train, feature, -1, threshold_percent=3.5)
    train          = replace_rare_categories(train, feature, -1, threshold_percent=3.5)
    plot_categorical_features(original_train[[feature]])


#sns.pairplot(train.sample(TUNE_DATASET_LEN))
#plt.show()


housing_features = ['rooms', 'bedrooms', 'hacdor', 'hacapo', 'v14a']
plot_heatmap(original_train[housing_features+[target_feature]])


education_features = ['escolari', 'edjefe', 'edjefa', 'meaneduc']
plot_heatmap(original_train[education_features+[target_feature]])


asset_features = ['refrig', 'v18q', 'television', 'mobilephone', 'qmobilephone']
plot_heatmap(original_train[asset_features+[target_feature]])


region_features = ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']
plot_heatmap(original_train[region_features+[target_feature]])


sanitation_features = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']
plot_heatmap(original_train[sanitation_features+[target_feature]])


ownership_features = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
plot_heatmap(original_train[ownership_features+[target_feature]])


family_features = ['r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3']
plot_heatmap(original_train[family_features+[target_feature]])


device_ownership_features = ['refrig', 'television', 'mobilephone']
plot_heatmap(original_train[device_ownership_features+[target_feature]])


utilities_access_features = ['abastaguadentro', 'abastaguafuera', 'abastaguano', 'noelec', 'coopele']
plot_heatmap(original_train[utilities_access_features+[target_feature]])


infrastructure_quality_features = ['epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3']
plot_heatmap(original_train[infrastructure_quality_features+[target_feature]])


education_level_features = ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']
plot_heatmap(original_train[education_level_features+[target_feature]])


demographic_features = ['male', 'female', 'age', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total']
plot_heatmap(original_train[demographic_features+[target_feature]])


squared_features = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned']
plot_heatmap(original_train[squared_features+[target_feature]])


living_conditions_features = ['overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
plot_heatmap(original_train[living_conditions_features+[target_feature]])


gender_household_role_features = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
                                  'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 
                                  'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
plot_heatmap(original_train[gender_household_role_features+[target_feature]])


from sklearn.preprocessing import StandardScaler

# Standardizing the data
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train.sample(TUNE_DATASET_LEN)), columns=train.columns)
dm_train = train_scaled.copy()


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Applying PCA
pca = PCA(n_components=2, random_state=42)
dm_train[["PCA_feature1", "PCA_feature2"]] = pca.fit_transform(train_scaled)
# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
dm_train[["t-SNE_feature1", "t-SNE_feature2"]] = tsne.fit_transform(train_scaled)

# Plotting PCA and t-SNE
plt.figure(figsize=(6*2, 6*1))
for i, (x, y, title) in enumerate(zip(["PCA_feature1", "t-SNE_feature1"], ["PCA_feature2", "t-SNE_feature2"], ["PCA", "t-SNE"])):
    plt.subplot(1, 2, i + 1)
    sns.scatterplot(x=dm_train[x], y=dm_train[y], palette="coolwarm", alpha=0.6)
    plt.title(title)
plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train.sample(TUNE_DATASET_LEN)), columns=train.columns)
clust_train = train_scaled.copy()


from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, BisectingKMeans, MiniBatchKMeans, AffinityPropagation

# Applying clustering algorithms
kmeans = KMeans(n_clusters=2, random_state=42)
clust_train["KMeans"] = kmeans.fit_predict(train_scaled)

agglo = AgglomerativeClustering(n_clusters=2)
clust_train["Agglomerative"] = agglo.fit_predict(train_scaled)

dbscan = DBSCAN(eps=1, min_samples=5)
clust_train["DBSCAN"] = dbscan.fit_predict(train_scaled)

bisect_kmeans = BisectingKMeans(n_clusters=2, random_state=42)
clust_train["BisectingKMeans"] = bisect_kmeans.fit_predict(train_scaled)

minibatch_kmeans = MiniBatchKMeans(n_clusters=2, random_state=42)
clust_train["MiniBatchKMeans"] = minibatch_kmeans.fit_predict(train_scaled)

affinity_prop = AffinityPropagation(random_state=42)
clust_train["AffinityPropagation"] = affinity_prop.fit_predict(train_scaled)

cluster_methods = ["KMeans", "Agglomerative", "DBSCAN", "BisectingKMeans", "MiniBatchKMeans", "AffinityPropagation"]
titles = ["KMeans", "Agglomerative", "DBSCAN", "BisectingKMeans", "MiniBatchKMeans", "AffinityPropagation"]


plt.figure(figsize=(6*2, 6*3))
for i, (col, title) in enumerate(zip(cluster_methods, titles)):
    plt.subplot(3, 2, i + 1)
    sns.scatterplot(x=dm_train["PCA_feature1"], y=dm_train["PCA_feature2"], hue=clust_train[col], palette="coolwarm", alpha=0.6)
    plt.title(title)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6*2, 6*3))
for i, (col, title) in enumerate(zip(cluster_methods, titles)):
    plt.subplot(3, 2, i + 1)
    sns.scatterplot(x=dm_train["t-SNE_feature1"], y=dm_train["t-SNE_feature2"], hue=clust_train[col], palette="coolwarm", alpha=0.6)
    plt.title(title)
plt.tight_layout()
plt.show()


original_train.shape


n_samples = min(int(1e7), original_train.shape[0])
df = original_train.sample(n_samples)
print(f'df.shape:\n{df.shape}')
print(f'df.columns:\n{df.columns}')


def plot_bar_charts(df_features, chart_title, xlabel, ylabel, legend_title, feature_labels, figsize=(14, 4)):
    df_features.plot(kind='bar', figsize=figsize)
    plt.title(chart_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.legend(title=legend_title, 
               labels=feature_labels,
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Group 1: Housing Features
housing_features = ['rooms', 'bedrooms', 'hacdor', 'hacapo', 'v14a']
feature_labels = ['Rooms', 'Bedrooms', 'Overcrowding Bedrooms', 'Shared Bathroom', 'Toilet Availability']

df_housing_features = df.groupby('Target')[housing_features].sum()
plot_bar_charts(df_housing_features, chart_title='Housing Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Housing Features', feature_labels=feature_labels)
df_housing_features


# Group 1: Housing Features
df_housing_features = df.groupby('Target')[housing_features].mean()
plot_bar_charts(df_housing_features, chart_title='Housing Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Housing Features', feature_labels=feature_labels)
df_housing_features


# Group 2: Education Features
education_features = ['escolari', 'edjefe', 'edjefa', 'meaneduc']
for feature in education_features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
feature_labels=['Years of Schooling', 'Years Behind in School', 'Head of Household Education (Male)', 
                'Head of Household Education (Female)', 'Average Years of Education']

df_education_features = df.groupby('Target')[education_features].sum()
plot_bar_charts(df_education_features, chart_title='Education Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Education Features', feature_labels=feature_labels)
df_education_features


# Group 2: Education Features
df_education_features = df.groupby('Target')[education_features].mean()
plot_bar_charts(df_education_features, 'Education Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Education Features', feature_labels=feature_labels)
df_education_features


# Group 3: Asset Features
asset_features = ['refrig', 'v18q', 'television', 'mobilephone', 'qmobilephone']
feature_labels=['Refrigerator', 'Tablet Ownership', 'Television', 'Mobile Phone', 'Quantity of Mobile Phones']

df_asset_features = df.groupby('Target')[asset_features].sum()
plot_bar_charts(df_asset_features, chart_title='Asset Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Asset Features', feature_labels=feature_labels)
df_asset_features


# Group 3: Asset Features
df_asset_features = df.groupby('Target')[asset_features].mean()
plot_bar_charts(df_asset_features, chart_title='Asset Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Asset Features', feature_labels=feature_labels)
df_asset_features


# Group 4: Dependency Features
dependency_features = ['dependency', 'edjefe', 'edjefa']
for feature in dependency_features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
feature_labels=['Dependency Rate', 'Head of Household Education (Male)', 'Head of Household Education (Female)']

df_dependency_features = df.groupby('Target')[dependency_features].sum()
plot_bar_charts(df_dependency_features, chart_title='Dependency Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Dependency Features', feature_labels=feature_labels)
df_dependency_features


# Group 4: Dependency Features
df_dependency_features = df.groupby('Target')[dependency_features].mean()
plot_bar_charts(df_dependency_features, chart_title='Dependency Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Dependency Features', feature_labels=feature_labels)
df_dependency_features


# Group 5: Region Features
region_features = ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']
feature_labels=['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5', 'Region 6', 'Urban Area', 'Rural Area']

df_region_features = df.groupby('Target')[region_features].sum()
plot_bar_charts(df_region_features, chart_title='Region Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Region Features', feature_labels=feature_labels)
df_region_features


# Group 5: Region Features
df_region_features = df.groupby('Target')[region_features].mean()
plot_bar_charts(df_region_features, chart_title='Region Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Region Features', feature_labels=feature_labels)
df_region_features


# Group 6: Sanitation Features
sanitation_features = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']
feature_labels=['No Toilet', 'Pit Toilet', 'Flush Toilet', 'Connected to Sewer', 'Septic Tank']

df_sanitation_features = df.groupby('Target')[sanitation_features].sum()
plot_bar_charts(df_sanitation_features, chart_title='Sanitation Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Sanitation Features', feature_labels=feature_labels)
df_sanitation_features


# Group 6: Sanitation Features
df_sanitation_features = df.groupby('Target')[sanitation_features].mean()
plot_bar_charts(df_sanitation_features, chart_title='Sanitation Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Sanitation Features', feature_labels=feature_labels)
df_sanitation_features


# Group 7: Ownership Features
ownership_features = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
feature_labels=['Own, Fully Paid', 'Own, Paying', 'Rented', 'Precarious', 'Other']

df_ownership_features = df.groupby('Target')[ownership_features].sum()
plot_bar_charts(df_ownership_features, chart_title='Ownership Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Ownership Features', feature_labels=feature_labels)
df_ownership_features


# Group 7: Ownership Features
df_ownership_features = df.groupby('Target')[ownership_features].mean()
plot_bar_charts(df_ownership_features, chart_title='Ownership Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Ownership Features', feature_labels=feature_labels)
df_ownership_features


# Group 8: Family Structure Features
family_features = ['r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3']
feature_labels=['Males Under 12', 'Males 12+', 'Total Males', 'Females Under 12', 
                'Females 12+', 'Total Females', 'Total Under 12', 'Total 12+', 'Household Total']

df_family_features = df.groupby('Target')[family_features].sum()
plot_bar_charts(df_family_features, chart_title='Family Structure Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Family Structure Features', feature_labels=feature_labels)
df_family_features


# Group 8: Family Structure Features
df_family_features = df.groupby('Target')[family_features].mean()
plot_bar_charts(df_family_features, chart_title='Family Structure Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Family Structure Features', feature_labels=feature_labels)
df_family_features


# Group 9: Device Ownership Features
device_ownership_features = ['refrig', 'television', 'mobilephone']
feature_labels=['Refrigerator', 'Television', 'Mobile Phone']

df_device_ownership_features = df.groupby('Target')[device_ownership_features].sum()
plot_bar_charts(df_device_ownership_features, chart_title='Device Ownership Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Device Ownership Features', feature_labels=feature_labels)
df_device_ownership_features


# Group 9: Device Ownership Features
df_device_ownership_features = df.groupby('Target')[device_ownership_features].mean()
plot_bar_charts(df_device_ownership_features, chart_title='Device Ownership Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Device Ownership Features', feature_labels=feature_labels)
df_device_ownership_features


# Group 10: Utilities Access Features
utilities_access_features = ['abastaguadentro', 'abastaguafuera', 'abastaguano', 'noelec', 'coopele']
feature_labels=['Water Inside', 'Water Outside', 'No Water Service', 'No Electricity', 'Cooperative Electricity']

df_utilities_access_features = df.groupby('Target')[utilities_access_features].sum()
plot_bar_charts(df_utilities_access_features, chart_title='Utilities Access Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Utilities Access Features', feature_labels=feature_labels)
df_utilities_access_features


# Group 10: Utilities Access Features
df_utilities_access_features = df.groupby('Target')[utilities_access_features].mean()
plot_bar_charts(df_utilities_access_features, chart_title='Utilities Access Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Utilities Access Features', feature_labels=feature_labels)
df_utilities_access_features


# Group 11: Infrastructure Quality Features
infrastructure_quality_features = ['epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3']
feature_labels=['Bad Walls', 'Regular Walls', 'Good Walls', 'Bad Roof', 'Regular Roof', 'Good Roof', 'Bad Floor', 'Regular Floor', 'Good Floor']

df_infrastructure_quality_features = df.groupby('Target')[infrastructure_quality_features].sum()
plot_bar_charts(df_infrastructure_quality_features, chart_title='Infrastructure Quality Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Infrastructure Quality Features', feature_labels=feature_labels)
df_infrastructure_quality_features


# Group 11: Infrastructure Quality Features
df_infrastructure_quality_features = df.groupby('Target')[infrastructure_quality_features].mean()
plot_bar_charts(df_infrastructure_quality_features, chart_title='Infrastructure Quality Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Infrastructure Quality Features', feature_labels=feature_labels)
df_infrastructure_quality_features


# Group 12: Education Level Features
education_level_features = ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']
feature_labels=['No Level', 'Primary Incomplete', 'Primary Complete', 'Secondary Incomplete', 
                'Secondary Complete', 'Technical', 'University Incomplete', 'University Complete', 'Post-graduate']

df_education_level_features = df.groupby('Target')[education_level_features].sum()
plot_bar_charts(df_education_level_features, chart_title='Education Level Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Education Level Features', feature_labels=feature_labels)
df_education_level_features


# Group 12: Education Level Features
df_education_level_features = df.groupby('Target')[education_level_features].mean()
plot_bar_charts(df_education_level_features, chart_title='Education Level Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Education Level Features', feature_labels=feature_labels)
df_education_level_features


# Group 13: Demographic Features
demographic_features = ['male', 'female', 'age', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total']
feature_labels=['Male', 'Female', 'Age', 'Children', 'Adults', 'Elderly', 'Total Household']

df_demographic_features = df.groupby('Target')[demographic_features].sum()
plot_bar_charts(df_demographic_features, chart_title='Demographic Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Demographic Features', feature_labels=feature_labels)
df_demographic_features


# Group 13: Demographic Features
df_demographic_features = df.groupby('Target')[demographic_features].mean()
plot_bar_charts(df_demographic_features, chart_title='Demographic Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Demographic Features', feature_labels=feature_labels)
df_demographic_features


# Group 14: Squared Features (Squared Transformations)
squared_features = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned']
feature_labels=['Schooling Squared', 'Age Squared', 'Total Household Squared', 'Head of Household Education Squared', 
                'Children in Household Squared', 'Overcrowding Squared', 'Dependency Squared', 'Mean Education Squared']

df_squared_features = df.groupby('Target')[squared_features].sum()
plot_bar_charts(df_squared_features, chart_title='Squared Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Squared Features', feature_labels=feature_labels)
df_squared_features


# Group 14: Squared Features (Squared Transformations)
df_squared_features = df.groupby('Target')[squared_features].mean()
plot_bar_charts(df_squared_features, chart_title='Squared Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Squared Features', feature_labels=feature_labels)
df_squared_features


# Group 15: Living Conditions Features
living_conditions_features = ['overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
feature_labels=['Overcrowding', 'Owned, Fully Paid', 'Owned, Paying', 'Rented', 'Precarious', 'Other']

df_living_conditions_features = df.groupby('Target')[living_conditions_features].sum()
plot_bar_charts(df_living_conditions_features, chart_title='Living Conditions Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Living Conditions Features', feature_labels=feature_labels)
df_living_conditions_features


# Group 15: Living Conditions Features
df_living_conditions_features = df.groupby('Target')[living_conditions_features].mean()
plot_bar_charts(df_living_conditions_features, chart_title='Living Conditions Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Living Conditions Features', feature_labels=feature_labels)
df_living_conditions_features


# Group 16: Gender and Household Role Features
gender_household_role_features = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
                                  'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 
                                  'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
feature_labels=['Single', 'Married', 'Divorced', 'Separated', 'Widowed', 'In Union', 
                'Other Civil Status', 'Head of Household', 'Spouse/Partner', 'Child', 
                'Stepchild', 'Son/Daughter-in-Law', 'Grandchild', 'Parent', 'Parent-in-law', 
                'Other Relative', 'Domestic Help', 'Non-relative', 'Other']

df_gender_household_role_features = df.groupby('Target')[gender_household_role_features].sum()
plot_bar_charts(df_gender_household_role_features, chart_title='Gender and Household Role Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Count of Households', 
                legend_title='Gender and Household Role Features', feature_labels=feature_labels, figsize=(12, 6))
df_gender_household_role_features


# Group 16: Gender and Household Role Features
df_gender_household_role_features = df.groupby('Target')[gender_household_role_features].mean()
plot_bar_charts(df_gender_household_role_features, chart_title='Gender and Household Role Features by Poverty Level', 
                xlabel='Poverty Level (1=Extreme, 4=Non-Vulnerable)', ylabel='Average Value', 
                legend_title='Gender and Household Role Features', feature_labels=feature_labels, figsize=(12, 6))
df_gender_household_role_features


from sklearn.preprocessing import MinMaxScaler

def add_columns(df):
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Group 1: Housing Features
    housing_features = list(set(df.columns) & set(['rooms', 'bedrooms', 'hacdor', 'hacapo', 'v14a']))
    if housing_features:
        df[housing_features] = scaler.fit_transform(df[housing_features])
        df['housing_features_combined'] = df[housing_features].sum(axis=1)
    
    # Group 2: Education Features
    education_features = list(set(df.columns) & set(['escolari', 'rez_esc', 'edjefe', 'edjefa', 'meaneduc']))
    if education_features:
        df[education_features] = scaler.fit_transform(df[education_features])
        df['education_features_combined'] = df[education_features].sum(axis=1)
    
    # Group 3: Asset Features
    asset_features = list(set(df.columns) & set(['refrig', 'v18q', 'television', 'mobilephone', 'qmobilephone']))
    if asset_features:
        df[asset_features] = scaler.fit_transform(df[asset_features])
        df['asset_features_combined'] = df[asset_features].sum(axis=1)
    
    # Group 4: Dependency Features
    dependency_features = list(set(df.columns) & set(['dependency', 'edjefe', 'edjefa']))
    if dependency_features:
        df[dependency_features] = scaler.fit_transform(df[dependency_features])
        df['dependency_features_combined'] = df[dependency_features].sum(axis=1)
    
    # Group 5: Region Features
    region_features = list(set(df.columns) & set(['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']))
    if region_features:
        df[region_features] = scaler.fit_transform(df[region_features])
        df['region_features_combined'] = df[region_features].sum(axis=1)
    
    # Group 6: Sanitation Features
    sanitation_features = list(set(df.columns) & set(['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']))
    if sanitation_features:
        df[sanitation_features] = scaler.fit_transform(df[sanitation_features])
        df['sanitation_features_combined'] = df[sanitation_features].sum(axis=1)
    
    # Group 7: Ownership Features
    ownership_features = list(set(df.columns) & set(['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']))
    if ownership_features:
        df[ownership_features] = scaler.fit_transform(df[ownership_features])
        df['ownership_features_combined'] = df[ownership_features].sum(axis=1)
    
    # Group 8: Family Structure Features
    family_features = list(set(df.columns) & set(['r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3']))
    if family_features:
        df[family_features] = scaler.fit_transform(df[family_features])
        df['family_structure_features_combined'] = df[family_features].sum(axis=1)
    
    # Group 9: Device Ownership Features
    device_ownership_features = list(set(df.columns) & set(['refrig', 'television', 'mobilephone']))
    if device_ownership_features:
        df[device_ownership_features] = scaler.fit_transform(df[device_ownership_features])
        df['device_ownership_features_combined'] = df[device_ownership_features].sum(axis=1)
    
    # Group 10: Utilities Access Features
    utilities_access_features = list(set(df.columns) & set(['abastaguadentro', 'abastaguafuera', 'abastaguano', 'noelec', 'coopele']))
    if utilities_access_features:
        df[utilities_access_features] = scaler.fit_transform(df[utilities_access_features])
        df['utilities_access_features_combined'] = df[utilities_access_features].sum(axis=1)
    
    # Group 11: Infrastructure Quality Features
    infrastructure_quality_features = list(set(df.columns) & set(['epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3']))
    if infrastructure_quality_features:
        df[infrastructure_quality_features] = scaler.fit_transform(df[infrastructure_quality_features])
        df['infrastructure_quality_features_combined'] = df[infrastructure_quality_features].sum(axis=1)
    
    # Group 12: Education Level Features
    education_level_features = list(set(df.columns) & set(['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']))
    if education_level_features:
        df[education_level_features] = scaler.fit_transform(df[education_level_features])
        df['education_level_features_combined'] = df[education_level_features].sum(axis=1)

    # Group 13: Demographic Features
    demographic_features = list(set(df.columns) & set(['male', 'female', 'age', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total']))
    if demographic_features:
        df[demographic_features] = scaler.fit_transform(df[demographic_features])
        df['demographic_features_combined'] = df[demographic_features].sum(axis=1)
    
    # Group 14: Squared Features (Squared Transformations)
    squared_features = list(set(df.columns) & set(['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned']))
    if squared_features:
        df[squared_features] = scaler.fit_transform(df[squared_features])
        df['squared_features_combined'] = df[squared_features].sum(axis=1)

    # Group 15: Living Conditions Features (New group)
    living_conditions_features = list(set(df.columns) & set(['overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']))
    if living_conditions_features:
        df[living_conditions_features] = scaler.fit_transform(df[living_conditions_features])
        df['living_conditions_features_combined'] = df[living_conditions_features].sum(axis=1)

    # Group 16: Gender Household Role Features (New group)
    gender_household_role_features = list(set(df.columns) & set(['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
                                      'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 
                                      'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']))
    if gender_household_role_features:
        df[gender_household_role_features] = scaler.fit_transform(df[gender_household_role_features])
        df['gender_household_role_features_combined'] = df[gender_household_role_features].sum(axis=1)

    return df


df_transformed = add_columns(df)


# List of new feature columns created for each group after standardization and combination
group_features = [
    'housing_features_combined', 'education_features_combined', 'asset_features_combined', 
    'dependency_features_combined', 'region_features_combined', 'sanitation_features_combined', 
    'ownership_features_combined', 'family_structure_features_combined', 'device_ownership_features_combined', 
    'utilities_access_features_combined', 'infrastructure_quality_features_combined', 
    'education_level_features_combined', 'demographic_features_combined', 'squared_features_combined', 
    'living_conditions_features_combined', 'gender_household_role_features_combined'
]

# Plotting the combined mean values for each feature group across the classes
fig, ax = plt.subplots(figsize=(16, 8))  # Create the figure and axis object
bar_width = 0.2  # Bar width for each class
x = np.arange(len(group_features))  # x-ticks for each feature group

# Colors for each class
colors = ['blue', 'green', 'orange', 'red']

# Loop through each feature group and plot the mean values for all classes
for i, feature in enumerate(group_features):
    means = df_transformed.groupby('Target')[feature].mean().values  # Get means for the 4 target classes
    ax.bar(x[i] - 1.5 * bar_width, means[0], width=bar_width, color=colors[0], label='1=Extreme' if i == 0 else "")
    ax.bar(x[i] - 0.5 * bar_width, means[1], width=bar_width, color=colors[1], label='2=Moderate' if i == 0 else "")
    ax.bar(x[i] + 0.5 * bar_width, means[2], width=bar_width, color=colors[2], label='3=Vulnerable' if i == 0 else "")
    ax.bar(x[i] + 1.5 * bar_width, means[3], width=bar_width, color=colors[3], label='4=Non-Vulnerable' if i == 0 else "")

# Adding labels, title, and other formatting
ax.set_xticks(x)
ax.set_xticklabels(group_features, rotation=90)
ax.set_xlabel('Feature Groups')
ax.set_ylabel('Mean Values by Poverty Level')
ax.set_title('Comparison of Combined and Standardized Feature Groups by Poverty Level')

ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

