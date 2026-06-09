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


!unzip -o -d /kaggle/working/ /kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip
!unzip -o -d /kaggle/working/ /kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip


dir_path = '/kaggle/working'
train = pd.read_csv(f'{dir_path}/train.tsv', delimiter='\t')
test  = pd.read_csv(f'{dir_path}/test.tsv', delimiter='\t')


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


N_UNIQUE_THRESHOLD = 50

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


info_table[info_table['dtype'].astype(str).str.startswith(('int'))]


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


train, _ = drop_id_feature(train, 'SentenceId')
test, test_id = drop_id_feature(test, 'SentenceId')
train, _ = drop_id_feature(train, 'PhraseId')
test, test_id = drop_id_feature(test, 'PhraseId')


#train = train.drop(columns=train.select_dtypes(exclude=[np.number]).columns)
#test = test.drop(columns=test.select_dtypes(exclude=[np.number]).columns)


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


import re, nltk
from nltk                            import word_tokenize, pos_tag
from nltk.corpus                     import stopwords
from nltk.sentiment                  import SentimentIntensityAnalyzer
from textblob                        import TextBlob
from sklearn.feature_extraction.text import strip_accents_unicode, \
                                            strip_accents_ascii


# Compile regex patterns once for better performance
re_single_double_chars = re.compile(r'\b\w{1,2}\b')
re_isolated_numbers = re.compile(r'\b\d+\b')
re_punctuations = re.compile(r'[^\w\s]')

# 1. Clean Text
def clean_text(text):
    # Vectorized text cleaning with precompiled regex
    text = re_single_double_chars.sub('', text)
    text = re_isolated_numbers.sub('', text)
    text = re_punctuations.sub('', text)
    return text

# 2. Normalize Accents
def normalize_accents(text):
    # Strip accents using Unicode normalization (e.g., é -> e)
    text = strip_accents_unicode(text)
    # Strip accents using ASCII approximation (e.g., Ü -> U)
    text = strip_accents_ascii(text)
    return text

# 3. Remove Stopwords (cached stopwords for efficiency)
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


# Basic Text Features with Vectorization and Parallelization
def basic_text_features(df, text_feature):
    def process_text(text):
        # Clean and normalize text
        text = clean_text(text)
        text = normalize_accents(text)
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = remove_stopwords(tokens)
        # Apply stemming and lemmatization
        # POS Tagging
        pos_tags = pos_tag(tokens)
        pos_counts = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        # Count POS Categories
        noun_count = pos_counts.get('NN', 0) + pos_counts.get('NNS', 0)
        verb_count = pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0)
        adj_count = pos_counts.get('JJ', 0)
        adv_count = pos_counts.get('RB', 0)
        
        return noun_count, verb_count, adj_count, adv_count
    
    df[[f'{text_feature}_noun_count', 
        f'{text_feature}_verb_count', 
        f'{text_feature}_adj_count', 
        f'{text_feature}_adv_count']] = df[text_feature].apply(process_text).apply(pd.Series)
    return df

# Instantiate VADER once for efficiency
sia = SentimentIntensityAnalyzer()

# Lexicon-Based Features using VADER and TextBlob
def lexicon_features(df, text_feature):
    def process_text(text):
        # Clean and normalize text
        text = clean_text(text)
        text = normalize_accents(text)
        # VADER Sentiment Analysis
        vader_scores = sia.polarity_scores(text)
        vader_pos = vader_scores['pos']
        vader_neu = vader_scores['neu']
        vader_neg = vader_scores['neg']
        vader_compound = vader_scores['compound']
        # TextBlob Sentiment Analysis
        text_blob = TextBlob(text)
        textblob_polarity = text_blob.sentiment.polarity
        textblob_subjectivity = text_blob.sentiment.subjectivity
        return (vader_pos, vader_neu, vader_neg, vader_compound, textblob_polarity, textblob_subjectivity)

    df[[f'{text_feature}_vader_pos', 
        f'{text_feature}_vader_neu', 
        f'{text_feature}_vader_neg', 
        f'{text_feature}_vader_compound', 
        f'{text_feature}_textblob_polarity', 
        f'{text_feature}_textblob_subjectivity']] = df[text_feature].apply(process_text).apply(pd.Series)
    return df

def add_columns(df, text_feature):
    # Ensure column is of type string
    df[text_feature] = df[text_feature].astype(str)
    # Tokenize and split sentences once to avoid redundancy
    values = df[text_feature].values
    word_lists = [val.split() for val in values]
    sentence_lists = [val.split('.') for val in values]
    # Calculate lengths
    df[f'{text_feature}_length_in_words'] = [
        len(words) for words in word_lists
    ]
    df[f'{text_feature}_length_in_chars'] = [
        len(val) for val in values
    ]
    df[f'{text_feature}_length_in_sentence'] = [
        len(sentences) for sentences in sentence_lists
    ]
    # Calculate average sentence lengths
    df[f'{text_feature}_avg_sentence_length_in_words'] = [
        np.mean([len(sentence.split()) for sentence in sentences]) if len(sentences) > 0 else 0
        for sentences in sentence_lists
    ]
    df[f'{text_feature}_avg_sentence_length_in_chars'] = [
        np.mean([sum(len(word) for word in sentence.split()) for sentence in sentences]) if len(sentences) > 0 else 0
        for sentences in sentence_lists
    ]
    # Vectorized sentiment analysis using TextBlob
    df[f'{text_feature}_sentiment'] = [
        TextBlob(val).sentiment.polarity for val in values
    ]
    return df


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


train = basic_text_features(train, 'Phrase')
train = lexicon_features(train, 'Phrase')
train = add_columns(train, 'Phrase')


original_train = basic_text_features(original_train, 'Phrase')
original_train = lexicon_features(original_train, 'Phrase')
original_train = add_columns(original_train, 'Phrase')


info_table = build_my_info_table(train)
info_table


train = train.drop(columns=train.select_dtypes(exclude=[np.number]).columns)


categorical_features = get_categorical_features(train)
numerical_features   = get_numerical_features(train)
print(f'categorical_features: {len(categorical_features)}\n{categorical_features}')
print(f'numerical_features:   {len(numerical_features)}\n{numerical_features}')


plot_numerical_features(train[numerical_features])


plot_categorical_features(train[categorical_features])


for feature in ['Phrase_noun_count', 'Phrase_verb_count', 'Phrase_adj_count', 'Phrase_adv_count', 'Phrase_length_in_sentence']:
    plot_categorical_features(original_train[[feature]])
    original_train = replace_rare_categories(original_train, feature, -1, threshold_percent=2.0)
    train          = replace_rare_categories(train, feature, -1, threshold_percent=2.0)
    plot_categorical_features(original_train[[feature]])


#sns.pairplot(train[numerical_features])
#plt.show()


plot_heatmap(train)


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
kmeans = KMeans(n_clusters=5, random_state=42)
clust_train["KMeans"] = kmeans.fit_predict(train_scaled)

agglo = AgglomerativeClustering(n_clusters=5)
clust_train["Agglomerative"] = agglo.fit_predict(train_scaled)

dbscan = DBSCAN(eps=1, min_samples=5)
clust_train["DBSCAN"] = dbscan.fit_predict(train_scaled)

bisect_kmeans = BisectingKMeans(n_clusters=5, random_state=42)
clust_train["BisectingKMeans"] = bisect_kmeans.fit_predict(train_scaled)

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
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


#TODO

