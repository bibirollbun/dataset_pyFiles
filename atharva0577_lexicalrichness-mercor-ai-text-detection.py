DEBUG = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
from tqdm import tqdm as tqdm_slim
from tqdm import tqdm
import re
import sys
from rich.console import Console
rc = Console(force_jupyter=False, color_system="truecolor")

import warnings
warnings.filterwarnings('ignore')
%load_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
sns.set_palette('RdYlGn')
SNS_CMAP = 'RdYlGn'
BIN_CMAP = sns.color_palette(["#9AE6B4", "#FC8181"])
BIN_CMAP_INVERT = sns.color_palette(["#FC8181", "#9AE6B4"])
plt.style.use("dark_background")
plt.rcParams['grid.color'] = '#444444'
colors = sns.palettes.color_palette(SNS_CMAP)
pd.options.mode.chained_assignment = None

def rprint(text:str)->None:
    rc.print(text)

def df_head_binary(df, target_col=None, palette=SNS_CMAP, n=5, alpha=0.2, row_height=50):
    """
    Highlights rows based on a binary target column using transparent background colors.

    Parameters:
    - df: DataFrame to style
    - target_col: column name to base row color on (default: last column)
    - palette: Seaborn palette name or list of two RGB colors
    - n: number of rows to display
    - alpha: transparency level (0 = fully transparent, 1 = opaque)
    - row_height: height of each row in pixels for better text visibility
    """
    if target_col is None:
        target_col = df.columns[-1]

    df_show = df.head(n)

    # Get RGB colors and add alpha
    palette_colors = sns.color_palette(palette, 2)
    rgba_0 = tuple(int(c * 255) for c in palette_colors[0]) + (alpha,)
    rgba_1 = tuple(int(c * 255) for c in palette_colors[1]) + (alpha,)

    def row_style(row):
        target = row[target_col]
        rgba = rgba_0 if target == 0 else rgba_1
        return [f'background-color: rgba{rgba}'] * len(row)

    styler = df_show.style.apply(row_style, axis=1)

    # Add CSS for larger rows and wrapped text
    styler.set_table_styles([
        {'selector': 'td, th',
         'props': [
             ('white-space', 'pre-wrap'),
             ('line-height', '1.4'),
             ('height', f'{row_height}px'),
             ('font-size', '13px'),
         ]}
    ])

    return styler

df = pd.read_csv(r"/kaggle/input/mercor-ai-detection/train.csv")
df_test = pd.read_csv(r"/kaggle/input/mercor-ai-detection/test.csv")
df_sample = pd.read_csv(r"/kaggle/input/mercor-ai-detection/sample_submission.csv")
TARGET = "is_cheating"
df_head_binary(df, TARGET, BIN_CMAP, n=8, row_height=70)


df['length'] = df['answer'].str.len()
df_test['length'] = df_test['answer'].str.len()
df['words'] = df['answer'].apply(lambda x: len(x.split()))
df_test['words'] = df_test['answer'].apply(lambda x: len(x.split()))
df['mean_word_length'] = df['answer'].apply(lambda x: np.mean([len(word) for word in x.split()]))
df_test['mean_word_length'] = df_test['answer'].apply(lambda x: np.mean([len(word) for word in x.split()]))
df['std_word_length'] = df['answer'].apply(lambda x: np.std([len(word) for word in x.split()]))
df_test['std_word_length'] = df_test['answer'].apply(lambda x: np.std([len(word) for word in x.split()]))

## Plotting
fig, ax = plt.subplots(4, 2, figsize=(12, 20))

# Left: Histogram of text lengths
sns.histplot(data=df, x='length', hue=TARGET, ax=ax[0][0], bins=10, kde=True, palette='Pastel2', multiple='stack')
ax[0][0].set_title('Text Length Distribution')
ax[0][0].set_xlabel('Text Length')
ax[0][0].set_ylabel('Count')

#right: test
sns.histplot(data=df_test, x='length', ax=ax[0][1],bins=10, kde=True, color='gray', multiple='stack')

ax[0][1].set_title('Test Text Length Distribution')
ax[0][1].set_xlabel('Text Length')
ax[0][1].set_ylabel('Count')

# Left: Histogram of word counts
sns.histplot(data=df[df['words']<400], x='words', hue=TARGET, ax=ax[1][0], bins=10, kde=True, palette='Pastel2', multiple='stack')
ax[1][0].set_title('Train Word Count Distribution (truncated)')
ax[1][0].set_xlabel('Word Count')
ax[1][0].set_ylabel('Count')

#right: test
sns.histplot(data=df_test[df_test['words']<400], x='words', ax=ax[1][1],bins=10, kde=True, color='gray', multiple='stack')
 
ax[1][1].set_title('Test Word Count Distribution (truncated)')
ax[1][1].set_xlabel('Word Count')
ax[1][1].set_ylabel('Count')

# Left: Histogram of text lengths
sns.histplot(data=df, x='mean_word_length', hue=TARGET, ax=ax[2][0], bins=10, kde=True, palette='Pastel2', multiple='stack')
ax[2][0].set_title('Train Mean Word Length Distribution')
ax[2][0].set_xlabel('Mean Word Length')
ax[2][0].set_ylabel('Count')

# Right: boxplot of diffs
sns.histplot(data=df_test, x='mean_word_length', ax=ax[2][1],bins=10, kde=True, color='gray', multiple='stack')
ax[2][1].set_title('Test Mean Word Length Distribution')
ax[2][1].set_xlabel('Mean Word Length')
ax[2][1].set_ylabel('Count')

# Left: Histogram of text lengths
sns.histplot(data=df, x='std_word_length', hue=TARGET, ax=ax[3][0], bins=10, kde=True, palette='Pastel2', multiple='stack')
ax[3][0].set_title('Train Std Word Length Distribution')
ax[3][0].set_xlabel('Std Word Length')
ax[3][0].set_ylabel('Count')

# Right: boxplot of diffs
sns.histplot(data=df_test, x='std_word_length', ax=ax[3][1],bins=10, kde=True, color='gray', multiple='stack')
ax[3][1].set_title('Test Std Word Length Distribution')
ax[3][1].set_xlabel('Std Word Length')
ax[3][1].set_ylabel('Count')

plt.tight_layout()
plt.show()


from collections import Counter
import string

labels = sorted(df[TARGET].unique())
punct_counters = {}
for lab in labels:
    texts = df.loc[df[TARGET] == lab, "answer"].astype(str).tolist()
    # Count punctuation chars that are in string.punctuation
    counter = Counter()
    for t in texts:
        for ch in t:
            if ch in string.punctuation:
                counter[ch] += 1
    punct_counters[lab] = counter

# ---------- Create tidy DataFrame for plotting ----------
# Collect union of punctuation marks across all labels
all_puncts = sorted(set().union(*(c.keys() for c in punct_counters.values())),
                    key=lambda p: -sum(punct_counters[l].get(p, 0) for l in labels))  # sort by total desc

rows = []
for p in all_puncts:
    for lab in labels:
        rows.append({
            "Punctuation": p,
            "Source": str(lab),  # convert to str for hue labels
            "Count": punct_counters[lab].get(p, 0)
        })

plot_df = pd.DataFrame(rows)

label_name_map = {labels[0]: "not-cheated", labels[1] if len(labels) > 1 else labels[0]: "cheated"}
# Apply mapping only when it matches keys
plot_df["Source"] = plot_df["Source"].map(lambda s: label_name_map.get(type(labels[0])(s)) if s.isdigit() else s)
# The above mapping is conservative â€” if you want forced mapping, replace with:
# plot_df["Source"] = plot_df["Source"].map({str(labels[0]): "Student", str(labels[1]): "AI"})

# ---------- Plot with seaborn ----------
# sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
# preserve the punct order (all_puncts) to keep bars sorted by total frequency
order = all_puncts

ax = sns.barplot(
    data=plot_df,
    x="Punctuation",
    y="Count",
    hue="Source",
    order=order,
    palette=BIN_CMAP,
    dodge=True  # side-by-side bars
)

ax.set_title("Punctuation Usage", fontsize=14, weight="bold", pad=12)
ax.set_xlabel("Symbol", fontsize=12)
ax.set_ylabel("Total Count", fontsize=12)
plt.xticks(rotation=0, fontsize=11)
plt.yticks(fontsize=11)
ax.legend(title="Source", fontsize=10, title_fontsize=11, loc="upper right")
sns.despine(left=False, bottom=False)
plt.tight_layout()
plt.show()


import re

def get_puntuation_feature(df:pd.DataFrame):
    df['punct_ratio'] = df['answer'].apply(lambda x: len(re.findall(r"[^\w\s]", x))/(len(x)))
    df['common_punct_count'] = df['answer'].apply(lambda x: x.count(',')+x.count('"')+x.count('-'))
    return df

df = get_puntuation_feature(df)
df_test = get_puntuation_feature(df_test)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Left: Histogram of text lengths
sns.histplot(data=df[df['common_punct_count']<60], x='common_punct_count', hue=TARGET, ax=ax[0], bins=10, kde=True, palette='Pastel2', multiple='stack')
ax[0].set_title('Train Punctuation Count Distribution')
ax[0].set_xlabel('Text punc-count')
ax[0].set_ylabel('Count')

#right: test
sns.histplot(data=df_test, x='common_punct_count', ax=ax[1],bins=10, kde=True, color='gray', multiple='stack')

ax[1].set_title('Punctuation Count Distribution')
ax[1].set_xlabel('Text punc-count')
ax[1].set_ylabel('Count')

plt.show()


!pip install -q lexicalrichness 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from lexicalrichness import LexicalRichness

nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def make_lexical_feat(df_row):
    text = df_row['answer']
    words = word_tokenize(text)
    total_words = len(words)
    stopwords_in_text = [w for w in words if w.lower() in STOPWORDS]
    df_row["stopword_ratio"] = len(stopwords_in_text) / total_words if total_words else 0

    lex = LexicalRichness(text)
    df_row["ttr"] = lex.ttr
    df_row["mtld"] = lex.mtld()
    try:
        df_row["mattr"] = lex.mattr()
    except:
        df_row["mattr"] = 0
    df_row["yulek"] = lex.yulek
    return df_row

df = df.apply(lambda x: make_lexical_feat(x), axis=1)

metrics = ["stopword_ratio", "ttr", "mtld", "mattr", "yulek"]
df["is_cheating_ai"] = df[TARGET].map({
    0: "0",
    1: "1"
}) 

fig, axes = plt.subplots(
    nrows=len(metrics),
    ncols=1,
    figsize=(12, 2.2 * len(metrics)),
    sharex=False
)

for ax, col in tqdm(zip(axes, metrics)):
    sns.boxplot(
        data=df,
        y="is_cheating_ai",
        x=col,
        dodge=False,
        palette=BIN_CMAP_INVERT,
        ax=ax,
        linewidth=1,
        fliersize=2.5
    )

plt.tight_layout()
plt.show()


!pip install -q wordfreq textstat

import spacy
from wordfreq import zipf_frequency
from textstat import flesch_kincaid_grade
from math import log2

NLP = spacy.load("en_core_web_sm")

def _make_linguistic_feat(df_row):
    text = df_row["answer"]
    
    words = word_tokenize(text)
    words_lower = [w.lower() for w in words if w.isalpha()]
    total_words = len(words_lower)

    if total_words == 0:
        df_row["mean_log_freq"] = 0
        df_row["lexical_sophistication"] = 0
        df_row["pos_entropy"] = 0
        df_row["flesch_kincaid"] = 0
        return df_row

    freqs = [zipf_frequency(w, "en") for w in words_lower]
    df_row["mean_log_freq"] = np.mean(freqs)
    df_row["lexical_sophistication"] = np.mean([1 if zipf_frequency(w, "en") < 3 else 0 for w in words_lower])

    doc = NLP(text)
    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    if pos_tags:
        _, counts = np.unique(pos_tags, return_counts=True)
        probs = counts / counts.sum()
        df_row["pos_entropy"] = -np.sum(probs * np.log2(probs))
    else:
        df_row["pos_entropy"] = 0

    df_row["flesch_kincaid"] = flesch_kincaid_grade(text)
    return df_row

df = df.apply(_make_linguistic_feat, axis=1)

linguistic_feat = ["mean_log_freq", "lexical_sophistication", "pos_entropy", "flesch_kincaid"]

fig, axes = plt.subplots(
    nrows=len(linguistic_feat),
    ncols=1,
    figsize=(12, 2.5 * len(metrics)),
    sharex=False
)

for ax, col in tqdm(zip(axes, linguistic_feat)):
    sns.violinplot(
        data=df,
        y="is_cheating_ai",
        x=col,
        dodge=False,
        palette=BIN_CMAP_INVERT,
        ax=ax,
        linewidth=1,
        fliersize=2.5
    )

plt.tight_layout()
plt.show()


!pip install -q lexicalrichness wordfreq textstat

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from lexicalrichness import LexicalRichness
import spacy
from wordfreq import zipf_frequency
from textstat import flesch_kincaid_grade
from math import log2

nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
NLP = spacy.load("en_core_web_sm")


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


def generate_structural_feat(df:pd.DataFrame)->pd.DataFrame:
    original_shape = df.shape
    original_cols = df.columns
    #---
    df['length'] = df['answer'].str.len()
    df['words'] = df['answer'].apply(lambda x: len(x.split()))
    df['mean_word_length'] = df['answer'].apply(lambda x: np.mean([len(word) for word in x.split()]))
    df['std_word_length'] = df['answer'].apply(lambda x: np.std([len(word) for word in x.split()]))
    df['punct_ratio'] = df['answer'].apply(lambda x: len(re.findall(r"[^\w\s]", x))/(len(x)))
    df['common_punct_count'] = df['answer'].apply(lambda x: x.count(',')+x.count('"')+x.count('-'))
    df['human_punct_count'] = df['answer'].apply(lambda x: x.count("'")+x.count('"')+x.count('?'))
    df['comma_count'] = df['answer'].apply(lambda x: x.count(","))
    df['hyphen_count'] = df['answer'].apply(lambda x: x.count("-"))
    #---
    new_shape = df.shape
    added_feat = [col for col in df.columns if col not in original_cols]
    rprint(f"df: {original_shape} ===> {new_shape}")
    rprint(f"Added {len(added_feat)} new [purple]Structural features[/]: {added_feat}")
    return df

def generate_lexical_feat(df:pd.DataFrame)->pd.DataFrame:
    original_shape = df.shape
    original_cols = df.columns
    def _make_lexical(df_row:pd.Series)->pd.Series:
        text = df_row['answer']
        words = word_tokenize(text)
        total_words = len(words)
        stopwords_in_text = [w for w in words if w.lower() in STOPWORDS]
        df_row["stopword_ratio"] = len(stopwords_in_text) / total_words if total_words else 0

        lex = LexicalRichness(text)
        df_row["ttr"] = lex.ttr
        df_row["mtld"] = lex.mtld()
        try:
            df_row["mattr"] = lex.mattr()
        except:
            df_row["mattr"] = 0
        df_row["yulek"] = lex.yulek
        return df_row
    #---
    df = df.apply(_make_lexical, axis=1)
    #---
    new_shape = df.shape
    added_feat = [col for col in df.columns if col not in original_cols]
    rprint(f"df: {original_shape} ===> {new_shape}")
    rprint(f"Added {len(added_feat)} new [purple]Lexical features[/]: {added_feat}")
    return df

def generate_linguistic_feat(df:pd.DataFrame)->pd.DataFrame:
    original_shape = df.shape
    original_cols = df.columns
    def _make_linguistic(df_row:pd.Series)->pd.Series:
        text = df_row["answer"]
        
        words = word_tokenize(text)
        words_lower = [w.lower() for w in words if w.isalpha()]
        total_words = len(words_lower)

        if total_words == 0:
            df_row["mean_log_freq"] = 0
            df_row["lexical_sophistication"] = 0
            df_row["pos_entropy"] = 0
            df_row["flesch_kincaid"] = 0
            return df_row

        freqs = [zipf_frequency(w, "en") for w in words_lower]
        df_row["mean_log_freq"] = np.mean(freqs)
        df_row["lexical_sophistication"] = np.mean([1 if zipf_frequency(w, "en") < 3 else 0 for w in words_lower])

        doc = NLP(text)
        pos_tags = [token.pos_ for token in doc if token.is_alpha]
        if pos_tags:
            _, counts = np.unique(pos_tags, return_counts=True)
            probs = counts / counts.sum()
            df_row["pos_entropy"] = -np.sum(probs * np.log2(probs))
        else:
            df_row["pos_entropy"] = 0

        df_row["flesch_kincaid"] = flesch_kincaid_grade(text)
        return df_row
    #---
    df = df.apply(_make_linguistic, axis=1)
    #---
    new_shape = df.shape
    added_feat = [col for col in df.columns if col not in original_cols]
    rprint(f"df: {original_shape} ===> {new_shape}")
    rprint(f"Added {len(added_feat)} new [purple]Linguistic features[/]: {added_feat}")
    return df
 
def generate_additional_feat(df:pd.DataFrame)->pd.DataFrame:
    """#additional feat taken from https://www.kaggle.com/code/yinlongabc/aitext"""
    original_shape = df.shape
    original_cols = df.columns
    #---
    df['extra__subordinate_ratio'] = df['answer'].str.count(r'\b(that|which|who|when|where|while|although|because|if)\b') / (df['words'] + 1)
    df["digit_ratio"] = df['answer'].apply(lambda x: sum(1 for c in x if c.isdigit())/max(1,len(x))).astype(np.float32)
    df["punct_count"] = df['answer'].str.count(r'[^\w\s]').fillna(0).astype(np.float32)
    df["words_per_char"] = df["words"] / (df["length"] + 1e-9)

    #---
    new_shape = df.shape
    added_feat = [col for col in df.columns if col not in original_cols]
    rprint(f"df: {original_shape} ===> {new_shape}")
    rprint(f"Added {len(added_feat)} additonal features: {added_feat}")
    return df


def feat_engi(df:pd.DataFrame)->pd.DataFrame:    
    df = generate_structural_feat(df)
    df = generate_lexical_feat(df)
    df = generate_linguistic_feat(df)
    df = generate_additional_feat(df)
    rc.rule()
    return df


%%time
df = pd.read_csv(r"/kaggle/input/mercor-ai-detection/train.csv")
df_test = pd.read_csv(r"/kaggle/input/mercor-ai-detection/test.csv")
df_sample = pd.read_csv(r"/kaggle/input/mercor-ai-detection/sample_submission.csv")

df = feat_engi(df)
df_test = feat_engi(df_test)


struct_feat = ['length', 'words', 'mean_word_length', 'std_word_length', 'punct_ratio', 
'common_punct_count', 'human_punct_count', 'comma_count', 'hyphen_count']
lexical_feat = ['stopword_ratio', 'ttr', 'mtld', 'mattr', 'yulek']
linguistic_feat = ['mean_log_freq', 'lexical_sophistication', 'pos_entropy', 'flesch_kincaid']
additional_feat = ['extra__subordinate_ratio', 'digit_ratio', 
'punct_count', 'words_per_char']

base_preproc = ColumnTransformer(transformers=[
        ('selector', 'passthrough', struct_feat+lexical_feat+linguistic_feat+additional_feat),
    ],
    verbose_feature_names_out=False).set_output(transform='pandas')

linear_preproc = Pipeline([
    ('base', base_preproc),
    ('scaler', StandardScaler()),
])


TREES_ON_GPU = False

import torch
if torch.cuda.is_available() and TREES_ON_GPU:
    from cuml.svm import SVC                   
else:
    from sklearn.svm import SVC
    
import optuna
from sklearn.linear_model import LogisticRegression

if torch.cuda.is_available() and TREES_ON_GPU:
    !git clone --recursive https://github.com/Microsoft/LightGBM
    !cd LightGBM
    !sh ./build-python.sh install --cuda
else:
    from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier, RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


import torch
if torch.cuda.is_available() and TREES_ON_GPU:
    from cuml.svm import SVC                     
else:
    from sklearn.svm import SVC
    

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, log_loss, mean_squared_error, matthews_corrcoef
import copy
from tqdm import tqdm


def train_and_evaluate_model(model, X, y, X_test=None, cv=5, name=None, model_name=None, stratify=False, 
                             retrain:bool = True, fold_iterator:object = None, verbose: bool = 1):
    """
    Train and evaluate a model using cross-validation.

    Args:
        model: The model to train and evaluate.
        X: Features for training and evaluation.
        y: Target labels.
        X_test: Optional test set for predictions after cross-validation.
        cv: Number of cross-validation folds.
        name: Optional name of the model for display.
        stratify: Whether to use stratified k-fold.

    Returns:
        metrics: Dictionary containing metrics for all folds.
    """ 
    if stratify:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42).split(X, y)
    else:
        folds = KFold(n_splits=cv, shuffle=True, random_state=42).split(X, y)

    metrics = {
        'accuracy': [],
        'f1-score': [],
        'auc-roc': [],
        'precision': [],
        'mse': [],
        'log-loss': [],
        'mcc': []
    }
    
    if model_name is None:
        if type(model) is Pipeline:
            model_name = model[-1].__class__.__name__
        else:
            model_name = model.__class__.__name__
    if name is None:
        name = model_name

    if fold_iterator is None:
        fold_iterator = folds
    if verbose:
        rprint(f"{name} ( [magenta] {model_name} [/] )")
        fold_iterator = tqdm(fold_iterator, desc=f"Evaluating {name}")
    for train_index, test_index in fold_iterator:
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        else:  
            X_train, X_valid = X[train_index], X[test_index]
        if isinstance(y, pd.Series):
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        else:  
            y_train, y_valid = y[train_index], y[test_index]

        model_fold = copy.deepcopy(model)
        model_fold.fit(X_train, y_train)

        y_pred = model_fold.predict(X_valid)
        try:
            y_pred_proba = model_fold.predict_proba(X_valid)[:, 1] if hasattr(model_fold, "predict_proba") else y_pred
        except:
            print(clrd('Unable to predict probas', 'warning'))
            y_pred_proba = y_pred*0

        metrics['accuracy'].append(accuracy_score(y_valid, y_pred))
        metrics['f1-score'].append(f1_score(y_valid, y_pred))
        metrics['precision'].append(precision_score(y_valid, y_pred))
        metrics['auc-roc'].append(roc_auc_score(y_valid, y_pred_proba))
        metrics['mse'].append(mean_squared_error(y_valid, y_pred_proba))
        metrics['log-loss'].append(log_loss(y_valid, y_pred_proba))
        metrics['mcc'].append(matthews_corrcoef(y_valid, y_pred))
    
    for k, v in metrics.items():
        metrics[k] = np.mean(v)

    if retrain:
        model.fit(X, y)
    if verbose:
        rprint(f"[green]accuracy[/] : {metrics['accuracy']:.4f}  f1-score: {metrics['f1-score']:.4f}   [green]auc-roc:[/] [bold green]{metrics['auc-roc']:.4f}[/]")
        rprint(f"precision: {metrics['precision']:.4f}  mcc     : {metrics['mcc']:.4f}   mse    : {metrics['mse']:.4f}")
        print('-'*50)

    return metrics


models = {
    "linear-regression": make_pipeline(linear_preproc, LogisticRegression()),
    "svm": make_pipeline(linear_preproc, SVC(probability=True)),
    "lgbm": make_pipeline(base_preproc,
                 LGBMClassifier(
                     verbosity=-1,
                     objective = 'binary',
                 )),
    "hist": make_pipeline(base_preproc, HistGradientBoostingClassifier()),
    "rf": make_pipeline(base_preproc, RandomForestClassifier()),
    "xgb": make_pipeline(base_preproc,
                         XGBClassifier(
                            verbosity=0,
                            objective='binary:logistic',
                            eval_metric="auc",
                         )),
    "cb": make_pipeline(base_preproc, CatBoostClassifier(verbose=False)),
}

res = {}
for model_name, model in models.items():
    metrics = train_and_evaluate_model(model, df, df[TARGET], stratify=True, cv=15, name = model_name)
    res[model_name] = metrics


import shap

X = base_preproc.transform(df)
explainer = shap.TreeExplainer(models["xgb"][1])
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, max_display=30, show=False)
plt.gca().tick_params(axis='y', colors='white')
plt.gca().tick_params(axis='x', colors='white')
plt.show()


from pandas.api.types import is_numeric_dtype

#-------------------------------
#1. cb imp
cb_importance = models["cb"][1].get_feature_importance(type='FeatureImportance')

#-------------------------------
#2. correlation
correlations = []
for col in X.columns:
    if is_numeric_dtype(X[col]):
        corr = X[col].corr(df[TARGET])
    else:
        corr = np.nan
    correlations.append(corr)

#-------------------------------
#3. linear regression wts
logistic_coeffs = np.abs(models["linear-regression"][1].coef_[0])

#-------------------------------
#4. dropping feats
base_score = res["rf"]['auc-roc']
y = df[TARGET]
drops = []
for col in X.columns:
    drops.append(base_score - cross_val_score(models["rf"][1], X.drop(columns=[col]), y, cv=5, scoring="roc_auc").mean())


df_importance = pd.DataFrame({
    'Feature': X.columns.tolist(),
    'CB_Importance': cb_importance,
    'Mean_ABS_SHAP': np.abs(shap_values[0]),
    'logictic_cf': logistic_coeffs,
    'Corr_with_Target': np.abs(correlations),
    'dropped_feats': drops,
}).sort_values("CB_Importance", ascending=False).reset_index(drop=True)

df_importance.style.background_gradient(cmap=SNS_CMAP)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=300,
    ngram_range=(1,2),
    sublinear_tf=True,
    stop_words="english"
)

final_preproc = ColumnTransformer(
    transformers=[
        ("num", "passthrough", struct_feat + lexical_feat + linguistic_feat + additional_feat),
        ("tfidf", tfidf, "answer"),
    ],
    verbose_feature_names_out=False
)

final_linear_preproc = Pipeline([
    ('base', final_preproc),
    ('scaler', StandardScaler(with_mean=False)),
])


models = {
    "linear-regression": make_pipeline(final_linear_preproc, LogisticRegression()),
    "svm": make_pipeline(final_linear_preproc, SVC(probability=True)),
    "lgbm": make_pipeline(final_preproc,
                 LGBMClassifier(
                     verbosity=-1,
                     objective = 'binary',
                 )),
    "xgb": make_pipeline(final_preproc,
                         XGBClassifier(
                            verbosity=0,
                            objective='binary:logistic',
                            eval_metric="auc",
                         )),
}

res = {}
for model_name, model in models.items():
    metrics = train_and_evaluate_model(model, df, df[TARGET], stratify=True, cv=10, name = model_name)
    res[model_name] = metrics


pred = models["svm"].predict_proba(df_test)
plt.figure(figsize=(15, 6))
sns.kdeplot(pred)
plt.show()


df_sub = pd.read_csv(r"/kaggle/input/mercor-ai-detection/sample_submission.csv")
df_sub[TARGET] = pred[:, 1]
df_sub.to_csv('submission.csv', index=False)

