import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings


warnings.filterwarnings("ignore")


N_UNIQUE_THRESHOLD = 20


def df_dtypes(df):
    pd.set_option('display.max_colwidth', None)

    df_dtypes = df.columns.to_series().groupby(df.dtypes.astype(str)).apply(list).reset_index()
    df_dtypes.columns = ['Data Type', 'Columns']
    df_dtypes['Count'] = df_dtypes['Columns'].apply(len)

    df_dtypes = df_dtypes[['Data Type', 'Count', 'Columns']]
    df_dtypes['Columns'] = df_dtypes['Columns'].apply(lambda cols: ', '.join(cols))

    styled = df_dtypes.style.set_properties(
        subset=['Columns'],
        **{
            'text-align': 'left',
            'white-space': 'pre-wrap',
            'padding': '8px'
        }
    ).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

    return styled


def get_categorical_features(df, nunique_threshold=N_UNIQUE_THRESHOLD, verbose=False):
    cat_features = [
        feature for feature in df.columns
        if df[feature].nunique() < nunique_threshold
    ]
    
    if verbose:
        grouped = {}
        for f in cat_features:
            n_unique = df[f].nunique()
            if n_unique not in grouped:
                grouped[n_unique] = []
            grouped[n_unique].append(f)
        
        print(f"ðŸŸ¡ Categorical Features (nunique < {nunique_threshold}): {len(cat_features)}\n")
        for uniq_val in sorted(grouped):
            print(f"â€¢ {uniq_val} unique values: {', '.join(grouped[uniq_val])}")
        print()

    return cat_features


def get_numerical_features(df, nunique_threshold=N_UNIQUE_THRESHOLD, verbose=False):
    num_features = [
        feature for feature in df.select_dtypes(include=[np.number]).columns
        if df[feature].nunique() >= nunique_threshold
    ]
    
    if verbose:
        print(f"ðŸ”µ Numerical Features (nunique â‰¥ {nunique_threshold}): {len(num_features)}\n")
        for f in num_features:
            print(f"â€¢ {f}")
        print()

    return num_features


def build_my_info_table(df, nunique_threshold=N_UNIQUE_THRESHOLD):
    if df is None or df.empty:
        return None

    # Convert boolean columns to integers
    boolean_columns = df.select_dtypes(include='bool').columns
    df[boolean_columns] = df[boolean_columns].astype(int)

    numerical_features = get_numerical_features(df)

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
        unique_values = column_data.unique() if nunique < nunique_threshold else ''
        mode    = column_data.mode().iloc[0] if not column_data.mode().empty else ''
        mode_count = column_data.value_counts().max() if not column_data.value_counts().empty else ''
        mode_percentage = (round(mode_count * 100 / len(column_data), 1) 
                           if mode_count not in ['', None] else '')
        null_count = column_data.isnull().sum()
        null_percentage = round(column_data.isnull().mean() * 100, 1)

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

    df_info = pd.DataFrame(metrics)

    return df_info


def plot_bar_chart(
    df, x, y, xlabel="", ylabel="", title="",
    xmin=None, xmax=None, palette='Blues_d',
    sort=True, show_values=True
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty:
        return

    sns.set(style="whitegrid")

    # Sort data if requested
    data = df.copy()
    if sort:
        data = data.sort_values(by=x, ascending=True)

    # Dynamic figure size
    height = max(5, data.shape[0] * 0.4)
    plt.figure(figsize=(12, height))

    # Create the plot
    ax = sns.barplot(y=y, x=x, data=data, palette=palette)

    # Add value labels to bars
    if show_values:
        for p in ax.patches:
            width = p.get_width()
            ax.text(
                width + 0.01 * max(data[x].max(), 1),
                p.get_y() + p.get_height() / 2,
                f'{width:.1f}',
                va='center'
            )

    # Labels and title
    plt.title(title, fontsize=14, weight='bold', pad=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Axis limits
    if xmin is None or xmax is None:
        margin = (data[x].max() - data[x].min()) * 0.1 or 1
        if xmin is None:
            xmin = min(0, data[x].min() - margin)
        if xmax is None:
            xmax = data[x].max() + margin

    plt.xlim(xmin, xmax)

    # Style tweaks
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def fillna_and_replace_inf(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numerical_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().all():
            continue
        median = df[col].median()
        df[col] = df[col].fillna(median)

    for col in categorical_cols:
        if df[col].isnull().all():
            continue
        mode_series = df[col].mode()
        if not mode_series.empty:
            mode = mode_series[0]
            df[col] = df[col].fillna(mode)

    return df


def encode_str_features(df, return_encoders=False):
    categorical_features = get_categorical_features(df)
    encoders = {}

    for feature in categorical_features:
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature].astype(str)).astype(np.int16)
        encoders[feature] = encoder

    if return_encoders:
        return df, encoders
    return df


def drop_id_feature(df, id_col='id'):
    df_id = df[id_col]
    df = df.drop(columns=[id_col])
    return df, df_id


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


def my_histplot(df, col, ax):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Histogram Plot of {col}')
def my_kdeplot(df, col, ax):
    sns.kdeplot(df[col], ax=ax, fill=True)
    ax.set_title(f'KDE Plot of {col}')
def my_distplot(df, col, ax):
    sns.histplot(df[col], ax=ax, kde=True)
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


!unzip -o -d /kaggle/working/ /kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip
!unzip -o -d /kaggle/working/ /kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip
!unzip -o -d /kaggle/working/ /kaggle/input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip


train = pd.read_csv('/kaggle/working/train.csv')
test = pd.read_csv('/kaggle/working/test.csv')


train.head()


test.head()


target_feature = 'y'


df_dtypes(train)


categorical_features = get_categorical_features(train, verbose=True)
numerical_features   = get_numerical_features(train, verbose=True)


info_table = build_my_info_table(train)
info_table


info_table[info_table['dtype'] == 'object']


info_table[(info_table['dtype'] == 'float64') | (info_table['dtype'] == 'float32')]


info_table[(info_table['dtype'] == 'int64') | (info_table['dtype'] == 'int32') | 
           (info_table['dtype'] == 'int16') | (info_table['dtype'] == 'int8')]


nan_df = info_table[info_table['null %'] >= 10][['column', 'null %']].sort_values(by='null %')


plot_bar_chart(
    nan_df,
    xmin=0,
    xmax=100,
    x='null %', 
    y='column', 
    xlabel='Missing %', 
    ylabel='Feature', 
    title='Missing Data per Feature'
)


dropped_nan = set(nan_df[nan_df['null %'] > 25]['column'])
train = train.drop(columns=list(dropped_nan - set([target_feature])))
test  = test.drop(columns=list(dropped_nan - set([target_feature])))


train = fillna_and_replace_inf(train)
test  = fillna_and_replace_inf(test)


info_table = build_my_info_table(test)
info_table


nan_df = info_table[info_table['null %'] >= 10][['column', 'null %']].sort_values(by='null %')


plot_bar_chart(
    nan_df,
    xmin=0,
    xmax=100,
    x='null %', 
    y='column', 
    xlabel='Missing %', 
    ylabel='Feature', 
    title='Missing Data per Feature'
)


dropped_nan = set(nan_df[nan_df['null %'] > 25]['column'])
train = train.drop(columns=list(dropped_nan - set([target_feature])))
test  = test.drop(columns=list(dropped_nan - set([target_feature])))


train = encode_str_features(train)
test  = encode_str_features(test)


train, _ = drop_id_feature(train, 'ID')
test, test_id = drop_id_feature(test, 'ID')


cat_train = train.drop(columns=train.select_dtypes(exclude=[np.number]).columns)
cat_test = test.drop(columns=test.select_dtypes(exclude=[np.number]).columns)


info_table = build_my_info_table(cat_train)
mode_df = info_table[info_table['mode %'] >= 90][['column', 'mode %']].sort_values(by='mode %')


plot_bar_chart(mode_df, x='mode %', y='column', 
               xlabel='Mode Percentage %', ylabel='Feature', 
               title='Mode Percentage in each Feature', 
               xmin=90, xmax=100, palette='coolwarm')


dropped_mode = set(mode_df[mode_df['mode %'] > 94]['column'])
train = train.drop(columns=list(dropped_mode - set([target_feature])))
test = test.drop(columns=list(dropped_mode - set([target_feature])))


categorical_features = get_categorical_features(train, verbose=True)
numerical_features   = get_numerical_features(train, verbose=True)


print(f'train.shape: {train.shape}')
print(f'test.shape : {test.shape}')


plot_numerical_features(train[numerical_features])


# plot_categorical_features(train[categorical_features])


# for feature in []:
#     plot_categorical_features(original_train[[feature]])
#     original_train = replace_rare_categories(original_train, feature, 'Other', threshold_percent=2)
#     train          = replace_rare_categories(train, feature, -1, threshold_percent=2)
#     plot_categorical_features(original_train[[feature]])


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




