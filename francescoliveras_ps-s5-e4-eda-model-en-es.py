!pip install -qq scikit-learn==1.6.1


import os 
import re
import gc
import sys
import math
import time
import random
import warnings
import catboost
import datetime
import numpy as np 
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
import missingno as msno
import plotly.express as px
import category_encoders as ce
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder


# Put theme of notebook 
from colorama import Fore, Style

# Colors
red = Fore.RED + Style.BRIGHT
mgta = Fore.MAGENTA + Style.BRIGHT
yllw = Fore.YELLOW + Style.BRIGHT
cyn = Fore.CYAN + Style.BRIGHT
blue = Fore.BLUE + Style.BRIGHT

# Reset
res = Style.RESET_ALL
plt.style.use({"figure.facecolor": "#282a36"})


# Colors
YELLOW = "#F7C53E"

CYAN_G = "#0CF7AF"
CYAB_DARK = "#11AB7C"

PURPLE = "#D826F8"
PURPLE_DARJ = "#9309AB"
PURPLE_L = "#b683d6"

BLUE = "#0C97FA"
RED = "#FA1D19"
ORANGE = "#FA9F19"
GREEN = "#0CFA58"
LIGTH_BLUE = "#01FADC"
S_BLUE = "#81c9e6"
DARK_BLUE = "#394be6"
# Palettes
PALETTE_2 = [CYAN_G, PURPLE]
PALETTE_3 = [YELLOW, CYAN_G, PURPLE]
PALETTE_4 = [YELLOW, ORANGE, PURPLE, LIGTH_BLUE]
PALETTE_5 = [PURPLE_DARJ, PURPLE_L, PURPLE, BLUE, LIGTH_BLUE]
PALETTE_6 = [BLUE, RED, ORANGE, GREEN, LIGTH_BLUE, PURPLE]

# Vaporwave palette by Francesc Oliveras
PALETTE_7 = [PURPLE_DARJ, PURPLE_L, PURPLE, BLUE, LIGTH_BLUE, DARK_BLUE, S_BLUE]
PALETTE_7_C = [PURPLE_DARJ, BLUE, PURPLE, LIGTH_BLUE, PURPLE_L, S_BLUE, DARK_BLUE]
sns.palplot(sns.color_palette(PALETTE_7))

# Set Style
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)

cmap = mcolors.LinearSegmentedColormap.from_list("", PALETTE_2)
cmap_2 = mcolors.LinearSegmentedColormap.from_list("", [S_BLUE, PURPLE_DARJ])

font_family = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=10), width=1000, height=500))

warnings.filterwarnings('ignore')


PATH = "/kaggle/input/playground-series-s5e4"
SUBMISSION_FILENAME = "sample_submission.csv"
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"
ORIGINAL_FILENAME = "/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv"

TARGET = "Listening_Time_minutes"

SUBMISSION_DIR = os.path.join(PATH, SUBMISSION_FILENAME)
TRAIN_DIR = os.path.join(PATH, TRAIN_FILENAME) 
TEST_DIR = os.path.join(PATH, TEST_FILENAME)

SEED = 250


def show_corr_heatmap(df, title):
    
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize = (15, 10))
    plt.title(title)
    # sns.heatmap(corr, annot = False, linewidths=.5, fmt=".2f", square=True, mask = mask, cmap=cmap_2)
    if df.shape[1] < 25:
        sns.heatmap(corr, annot=True, linewidths=.5, fmt=".2f", square=True, mask=mask, cmap=cmap_2)
    else:
        sns.heatmap(corr, annot=False, linewidths=.5, square=True, mask=mask, cmap=cmap_2)

    plt.show()


def data_description(df):
    print("Data description")
    print(f"Total number of records {df.shape[0]}")
    print(f'number of features {df.shape[1]}\n\n')
    columns = df.columns
    data_type = []
    
    # Get the datatype of features
    for col in df.columns:
        data_type.append(df[col].dtype)
        
    n_uni = df.nunique()
    # Number of NaN values
    n_miss = df.isna().sum()
    
    names = list(zip(columns, data_type, n_uni, n_miss))
    variable_desc = pd.DataFrame(names, columns=["Name","Type","Unique levels","Missing"])
    print(variable_desc)


def plot_cont(col, ax, color=PALETTE_7[0]):
    sns.histplot(data=comb_df, x=col,
                hue="set",ax=ax, hue_order=labels,
                common_norm=False, **histplot_hyperparams)
    
    ax_2 = ax.twinx()
    ax_2 = plot_cont_dot(
        comb_df.query('set=="train"'),
        col, TARGET, ax_2,
        color=color
    )
    
    ax_2 = plot_cont_dot(
        comb_df, col,
        TARGET, ax_2,
        color=color
    )


def show_pie_mult(dataframe, target = TARGET):
    target_counts = dataframe[target].sum()

    # Creando el grÃ¡fico de pastel con un agujero en el centro
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(target_counts, labels=target, autopct='%1.1f%%', startangle=140, colors=PALETTE_7_C)

    # Agregando un cÃ­rculo blanco en el centro para hacer un agujero
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Ajustando el aspecto para que sea un cÃ­rculo y mostrando el grÃ¡fico
    plt.title('DistribuciÃ³n de los Targets')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def show_pie_categorical(dataframe, target=TARGET):
    target_counts = dataframe[target].value_counts()

    # Creando el grÃ¡fico de pastel con un agujero en el centro
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=140, colors=[PALETTE_7_C[0],PALETTE_7_C[1],PALETTE_7_C[2],
                                                                                                                            PALETTE_7_C[3],PALETTE_7_C[4],PALETTE_7_C[5]])

    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Ajustando el aspecto para que sea un cÃ­rculo y mostrando el grÃ¡fico
    plt.title('DistribuciÃ³n de los Targets')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def show_box_plot(dataframe):
    # numerical_features_for_boxplot = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('id')
    numerical_features_for_boxplot = dataframe.select_dtypes(include=['int64', 'float64'])

    plt.figure(figsize=(20, 15))

    for i, feature in enumerate(numerical_features_for_boxplot, 1):
        plt.subplot(7, 5, i)
        sns.boxplot(y=train_df[feature], color=PALETTE_7_C[i % len(PALETTE_7_C)])
        plt.title(feature)

    plt.tight_layout()
    plt.show()



def show_hist(dataframe):
    # Filtrando las columnas numÃ©ricas para sus histogramas
    numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns

    # Configurando el tamaÃ±o de la figura
    plt.figure(figsize=(20, 15))

    # Creando un histograma para cada caracterÃ­stica numÃ©rica
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(7, 5, i) # Ajustar segÃºn el nÃºmero de caracterÃ­sticas numÃ©ricas
        dataframe[feature].hist(bins=20, color=PALETTE_7_C[int(i%7)])
        plt.title(feature)

    plt.tight_layout()
    plt.show()


def apply_one_hot_encoding(df, threshold=10):
    """
    Apply One-Hot Encoding to categorical columns with unique values below a given threshold.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    threshold (int): The maximum number of unique values for a column to be eligible for OHE.

    Returns:
    pd.DataFrame: The dataframe with One-Hot Encoding applied to selected columns.
    """
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Determine columns eligible for OHE
    ohe_columns = [col for col in categorical_columns if df[col].nunique() < threshold]
    
    # Apply One-Hot Encoding
    df_ohe = pd.get_dummies(df, columns=ohe_columns, drop_first=True)
    
    return df_ohe


def apply_label_encoding(df, columns):
    """
    Apply Label Encoding to specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to apply Label Encoding.

    Returns:
    pd.DataFrame: DataFrame with Label Encoding applied.
    """
    df_encoded = df.copy()
    for col in columns:
        label_encoder = LabelEncoder()
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    return df_encoded


def apply_frequency_encoding(df, columns):
    """
    Apply Frequency Encoding to specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to apply Frequency Encoding.

    Returns:
    pd.DataFrame: DataFrame with Frequency Encoding applied.
    """
    df_encoded = df.copy()
    for col in columns:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df_encoded[col] = df[col].map(freq_map)
    return df_encoded


def apply_hashing_encoding(df, columns, n_features=8):
    """
    Apply Hashing Encoding to specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to apply Hashing Encoding.
    n_features (int): Number of features to generate for each column.

    Returns:
    pd.DataFrame: DataFrame with Hashing Encoding applied.
    """
    df_encoded = df.copy()
    for col in columns:
        hasher = FeatureHasher(n_features=n_features, input_type='string')
        hashed_features = hasher.transform(df_encoded[col].astype(str))
        hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f"{col}_hash_{i}" for i in range(n_features)])
        df_encoded = df_encoded.drop(columns=[col]).join(hashed_df)
    return df_encoded


def apply_binary_encoding(df, columns):
    """
    Apply Binary Encoding to specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to apply Binary Encoding.

    Returns:
    pd.DataFrame: DataFrame with Binary Encoding applied.
    """
    binary_encoder = ce.BinaryEncoder(cols=columns)
    df_encoded = binary_encoder.fit_transform(df)
    return df_encoded


def feature_eng(df):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
    
    df['Genre'] = df['Genre'].replace(genr_dict)
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict)
    
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')
    
    df = df.drop(columns=['Episode_Title'])
    return df


train_df = pd.read_csv(TRAIN_DIR, index_col = "id")
test_df = pd.read_csv(TEST_DIR, index_col = "id")
original_df = pd.read_csv(ORIGINAL_FILENAME)
submission_df = pd.read_csv(SUBMISSION_DIR, index_col = "id")


numerical_features_ = test_df.select_dtypes(exclude='object')
categorical_features_ = test_df.select_dtypes(include='object')


data_description(train_df)
data_description(test_df)
data_description(original_df)


msno.matrix(df=train_df, figsize=(15,5), color=(0,0.6,0.5))


train_df.shape, original_df.shape


#train_nb = apply_one_hot_encoding(train_df)
#train_nb = apply_label_encoding(train_nb, numerical_features_)


# show_corr_heatmap(train_df, "test")


train_df.columns, original_df.columns


# train = pd.concat([train_df, original_df], axis=0, ignore_index=True)
train = train_df


train = train.drop_duplicates()
train.shape


data_description(train)


numerical_features = test_df.select_dtypes(exclude='object')
categorical_features = test_df.select_dtypes(include='object')


train.isnull().sum().sort_values(ascending=False)


test_df.isnull().sum().sort_values(ascending=False)


train = train.dropna()


train = feature_eng(train)
test_df = feature_eng(test_df)


encode_columns = ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
pair_size = [2, 3, 4]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = '_'.join(cols)
        
        train[new_col_name] = train[list(cols)].astype(str).agg('_'.join, axis=1)
        train[new_col_name] = train[new_col_name].astype('category')
        
        test_df[new_col_name] = test_df[list(cols)].astype(str).agg('_'.join, axis=1)
        test_df[new_col_name] = test_df[new_col_name].astype('category')


X = train.drop(columns=[TARGET])
y = train[TARGET]
cv = KFold(5, random_state=SEED, shuffle=True)
y_pred = np.zeros(len(submission_df))





for idx_train, idx_valid in cv.split(X, y):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    X_test = test_df[X.columns].copy()
    
    encoded_columns = train.columns[11:]
    encoder = TargetEncoder(random_state=42)
    
    X_train[encoded_columns] = encoder.fit_transform(X_train[encoded_columns], y_train)
    X_valid[encoded_columns] = encoder.transform(X_valid[encoded_columns])
    X_test[encoded_columns] = encoder.transform(X_test[encoded_columns])

    model = lgb.LGBMRegressor(
        n_iter=1000,
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.03,
        objective='l2',
        metric='rmse', 
        verbosity=-1,
        max_bin=1024,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.log_evaluation(100)],
    )
    
    y_pred += model.predict(X_test)


submission_df[TARGET] = y_pred / 5
submission_df.to_csv('submission.csv')
submission_df.head()

