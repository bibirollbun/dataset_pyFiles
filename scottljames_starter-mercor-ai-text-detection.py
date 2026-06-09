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


# ============================================
# Mercor AI Text Detection â€“ MULTI-EMBEDDINGS ENHANCED
# Multiple Embedding Models + 100+ Features + 6 Models + Visualizations
# Embeddings: Sentence-BERT, DistilBERT, TF-IDF LSA, Doc2Vec
# ============================================

import os, gc, re, string, math, time, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.neural_network import MLPClassifier

from scipy.sparse import hstack, csr_matrix
from scipy.stats import skew, kurtosis

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'SEED': 42,
    'N_FOLDS': 5,
    
    # Embedding settings - Enable multiple embeddings for better performance
    'USE_SENTENCE_BERT': True,      # All-MiniLM-L6-v2 (fast, good quality)
    'USE_DISTILBERT': False,         # DistilBERT (slower but powerful) - Set False for speed
    'USE_TFIDF_SVD': True,          # TF-IDF + SVD embeddings (fast)
    'USE_WORD2VEC': True,           # Word2Vec aggregation (fast)
    'USE_GEMMA': False,             # Original Gemma embeddings (if available)
    
    # Embedding dimensions
    'SBERT_DIM': 50,                # Sentence-BERT reduced dim
    'DISTILBERT_DIM': 50,           # DistilBERT reduced dim
    'TFIDF_SVD_DIM': 30,            # TF-IDF SVD components
    'WORD2VEC_DIM': 100,            # Word2Vec dimension
    'GEMMA_DIM': 50,                # Gemma reduced dim
    
    # Model settings
    'USE_CATBOOST': True,
    'USE_MLP': True,
    
    # TF-IDF settings
    'CHAR_NGRAM_RANGE': (3, 6),
    'CHAR_MAX_FEATURES': 300_000,
    'WORD_NGRAM_RANGE': (1, 3),      # Increased to trigrams
    'WORD_MAX_FEATURES': 250_000,    # Increased features
    
    # Visualization
    'CREATE_PLOTS': True,
    'PLOT_DPI': 150,
    
    # Advanced features
    'USE_POS_TAGGING': False,        # Part-of-speech features (requires nltk, slower)
}

SEED = CONFIG['SEED']
N_FOLDS = CONFIG['N_FOLDS']
rng = np.random.RandomState(SEED)

DATA_DIR = "/kaggle/input/mercor-ai-detection"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH  = f"{DATA_DIR}/test.csv"
GEMMA_PATH = "/kaggle/input/embeddinggemma/transformers/embeddinggemma-300m/1"

print("="*70)
print(" "*10 + "MERCOR AI DETECTION - MULTI-EMBEDDINGS")
print("="*70)
print(f"\nâš™ï¸�  CONFIGURATION:")
print(f"  Sentence-BERT:  {'âœ“' if CONFIG['USE_SENTENCE_BERT'] else 'âœ—'}")
print(f"  DistilBERT:     {'âœ“' if CONFIG['USE_DISTILBERT'] else 'âœ—'}")
print(f"  TF-IDF SVD:     {'âœ“' if CONFIG['USE_TFIDF_SVD'] else 'âœ—'}")
print(f"  Word2Vec:       {'âœ“' if CONFIG['USE_WORD2VEC'] else 'âœ—'}")
print(f"  Gemma:          {'âœ“' if CONFIG['USE_GEMMA'] else 'âœ—'}")
print(f"  CatBoost:       {'âœ“' if CONFIG['USE_CATBOOST'] else 'âœ—'}")
print(f"  MLP:            {'âœ“' if CONFIG['USE_MLP'] else 'âœ—'}")
print(f"  Plots:          {'âœ“' if CONFIG['CREATE_PLOTS'] else 'âœ—'}")

# Install packages
print("\n[0/16] Installing packages...")
packages_to_install = []

if CONFIG['USE_SENTENCE_BERT'] or CONFIG['USE_DISTILBERT']:
    packages_to_install.append('sentence-transformers')
if CONFIG['USE_CATBOOST']:
    packages_to_install.append('catboost')
if CONFIG['USE_WORD2VEC']:
    packages_to_install.append('gensim')
if CONFIG['USE_POS_TAGGING']:
    packages_to_install.append('nltk')

if packages_to_install:
    os.system(f"pip install -q {' '.join(packages_to_install)}")
    
if CONFIG['USE_CATBOOST']:
    import catboost as cb
if CONFIG['USE_WORD2VEC']:
    from gensim.models import Word2Vec

# --------------------------
# 1) Load data
# --------------------------
print("\n[1/16] Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")
print(f"  Class balance: {dict(train['is_cheating'].value_counts())}")

train["topic"]  = train["topic"].fillna("")
train["answer"] = train["answer"].fillna("")
test["topic"]   = test["topic"].fillna("")
test["answer"]  = test["answer"].fillna("")

y = train["is_cheating"].astype(int).values

# --------------------------
# 2) Text cleaning
# --------------------------
print("\n[2/16] Text preprocessing...")
def basic_norm(s: str) -> str:
    s = s.replace("\u00A0", " ").replace("\t"," ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

for col in ["topic", "answer"]:
    train[col] = train[col].astype(str).map(basic_norm)
    test[col]  = test[col].astype(str).map(basic_norm)

print("  âœ“ Complete")

# --------------------------
# 3) Enhanced Feature Engineering
# --------------------------
print("\n[3/16] Engineering 100+ features...")

STOPWORDS = {
    "a","an","the","and","or","but","if","while","for","to","of","in","on","at","by","with","from","as",
    "is","am","are","was","were","be","been","being","it","this","that","these","those","i","you","he","she","we","they",
    "me","him","her","us","them","my","your","his","her","our","their",
    "will","would","can","could","should","may","might","do","does","did","done","doing",
    "not","no","so","than","then","there","here","when","where","why","how",
    "however","therefore","moreover","furthermore","additionally","overall","hence","thus"
}

def count_syllables_en(word: str) -> int:
    w = re.sub(r'[^a-z]', '', word.lower())
    if not w: return 0
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

def shannon_entropy_char(s: str) -> float:
    if not s: return 0.0
    counts = np.array([s.count(c) for c in set(s)], dtype=float)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))

def repeat_ngram_ratio(words, n=3):
    if len(words) < n: return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    if not ngrams: return 0.0
    return 1.0 - len(set(ngrams))/len(ngrams)

def ultra_style_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    texts = df["answer"].tolist()
    topics = df["topic"].tolist()

    # Initialize lists
    wc, char_count, avg_wlen, median_wlen, max_wlen, min_wlen, std_wlen = [], [], [], [], [], [], []
    word_len_skewness, word_len_kurtosis = [], []
    ttr, unique_word_ratio, stop_ratio = [], [], []
    long_word_ratio, short_word_ratio, medium_word_ratio, very_long_word_ratio = [], [], [], []
    punct_ratio, upper_ratio, lower_ratio, digit_ratio, space_ratio, alpha_ratio, alnum_ratio = [], [], [], [], [], [], []
    sent_cnt, avg_sent_len, median_sent_len, max_sent_len, min_sent_len, sent_len_std, avg_sentence_complexity = [], [], [], [], [], [], []
    
    # Specific character counts
    comma_count, period_count, exclamation_count, question_count = [], [], [], []
    semicolon_count, colon_count, dash_count, underscore_count = [], [], [], []
    slash_count, backslash_count, pipe_count, ampersand_count = [], [], [], []
    at_sign_count, hash_count, dollar_count, percent_count = [], [], [], []
    caret_count, asterisk_count, plus_count, equals_count = [], [], [], []
    tilde_count, backtick_count = [], []
    
    single_quote_count, double_quote_count, quote_ratio = [], [], []
    paren_count, square_bracket_count, curly_bracket_count, angle_bracket_count, bracket_ratio = [], [], [], [], []
    ellipsis_count, newline_count, tab_count, multiple_space_count, consecutive_punct_count = [], [], [], [], []
    comma_per_100w, period_per_100w, exclamation_per_100w, question_per_100w, semicolon_per_100w, colon_per_100w = [], [], [], [], [], []
    entropy_char, entropy_word = [], []
    unigram_repeat, bigram_repeat, trigram_repeat, fourgram_repeat = [], [], [], []
    flesch_kincaid, gunning_fog, smog_index = [], [], []
    special_char_ratio, math_symbol_ratio, number_count, number_ratio, contraction_count = [], [], [], [], []
    
    rhetorical_markers = ["however","therefore","moreover","furthermore","additionally","thus","hence",
                          "in conclusion","in summary","to summarize","firstly","secondly","finally",
                          "consequently","nevertheless"]
    transition_words = ["also","furthermore","moreover","additionally","besides","however","nevertheless",
                       "nonetheless","still","yet","therefore","thus","consequently","accordingly","hence"]
    first_person = ["i","me","my","mine","myself","we","us","our","ours","ourselves"]
    second_person = ["you","your","yours","yourself","yourselves"]
    third_person = ["he","him","his","himself","she","her","hers","herself","it","its","itself",
                    "they","them","their","theirs","themselves"]
    
    rhet_count, transition_count = [], []
    first_person_ratio, second_person_ratio, third_person_ratio = [], [], []
    lexical_diversity, word_uniqueness, uppercase_sentence_start_ratio, avg_word_rank = [], [], [], []
    
    # NEW: Additional statistical features
    word_length_variance = []
    hapax_legomena_ratio = []  # Words appearing once
    hapax_dislegomena_ratio = []  # Words appearing twice
    avg_chars_per_word = []
    
    for s in texts:
        tokens = re.findall(r"\b\w+\b", s.lower())
        words = [t for t in tokens if t.isalpha()]
        n_words = len(words)
        n_chars = len(s)
        
        wc.append(n_words)
        char_count.append(n_chars)
        
        if n_words > 0:
            word_lengths = [len(w) for w in words]
            avg_wlen.append(np.mean(word_lengths))
            median_wlen.append(np.median(word_lengths))
            max_wlen.append(np.max(word_lengths))
            min_wlen.append(np.min(word_lengths))
            std_wlen.append(np.std(word_lengths))
            word_len_skewness.append(skew(word_lengths))
            word_len_kurtosis.append(kurtosis(word_lengths))
            word_length_variance.append(np.var(word_lengths))
            avg_chars_per_word.append(n_chars / n_words)
            
            long_word_ratio.append(sum(1 for w in words if len(w) > 6)/n_words)
            short_word_ratio.append(sum(1 for w in words if len(w) <= 3)/n_words)
            medium_word_ratio.append(sum(1 for w in words if 4 <= len(w) <= 6)/n_words)
            very_long_word_ratio.append(sum(1 for w in words if len(w) > 10)/n_words)
            
            uniq_words = len(set(words))
            ttr.append(uniq_words / n_words)
            unique_word_ratio.append(uniq_words / n_words)
            stop_ratio.append(sum(w in STOPWORDS for w in words)/n_words)
            
            lexical_diversity.append(len(set(words)) / np.log(n_words) if n_words > 50 else len(set(words)) / (n_words + 1))
            word_freq = Counter(words)
            word_uniqueness.append(sum(1 for w in words if word_freq[w] == 1) / n_words)
            
            # Hapax legomena and dislegomena
            hapax_legomena_ratio.append(sum(1 for w in words if word_freq[w] == 1) / n_words)
            hapax_dislegomena_ratio.append(sum(1 for w in words if word_freq[w] == 2) / n_words)
            
            first_person_ratio.append(sum(1 for w in words if w in first_person) / n_words)
            second_person_ratio.append(sum(1 for w in words if w in second_person) / n_words)
            third_person_ratio.append(sum(1 for w in words if w in third_person) / n_words)
            
            comma_per_100w.append(s.count(",")/(n_words/100.0 + 1e-6))
            period_per_100w.append(s.count(".")/(n_words/100.0 + 1e-6))
            exclamation_per_100w.append(s.count("!")/(n_words/100.0 + 1e-6))
            question_per_100w.append(s.count("?")/(n_words/100.0 + 1e-6))
            semicolon_per_100w.append(s.count(";")/(n_words/100.0 + 1e-6))
            colon_per_100w.append(s.count(":")/(n_words/100.0 + 1e-6))
            
            avg_word_rank.append(np.mean([len(w)**1.5 for w in words]))
        else:
            for lst in [avg_wlen, median_wlen, max_wlen, min_wlen, std_wlen, word_len_skewness, word_len_kurtosis,
                       word_length_variance, avg_chars_per_word, ttr, unique_word_ratio, stop_ratio, long_word_ratio, 
                       short_word_ratio, medium_word_ratio, very_long_word_ratio, lexical_diversity, word_uniqueness,
                       hapax_legomena_ratio, hapax_dislegomena_ratio, first_person_ratio, second_person_ratio,
                       third_person_ratio, comma_per_100w, period_per_100w, exclamation_per_100w, question_per_100w,
                       semicolon_per_100w, colon_per_100w, avg_word_rank]:
                lst.append(0.0)

        total_chars = n_chars + 1e-6
        punct_ratio.append(sum(ch in string.punctuation for ch in s)/total_chars)
        upper_ratio.append(sum(1 for ch in s if ch.isupper())/total_chars)
        lower_ratio.append(sum(1 for ch in s if ch.islower())/total_chars)
        digit_ratio.append(sum(1 for ch in s if ch.isdigit())/total_chars)
        space_ratio.append(s.count(" ")/total_chars)
        alpha_ratio.append(sum(1 for ch in s if ch.isalpha())/total_chars)
        alnum_ratio.append(sum(1 for ch in s if ch.isalnum())/total_chars)
        
        # Specific characters
        comma_count.append(s.count(","))
        period_count.append(s.count("."))
        exclamation_count.append(s.count("!"))
        question_count.append(s.count("?"))
        semicolon_count.append(s.count(";"))
        colon_count.append(s.count(":"))
        dash_count.append(s.count("-") + s.count("â€”") + s.count("â€“"))
        underscore_count.append(s.count("_"))
        slash_count.append(s.count("/"))
        backslash_count.append(s.count("\\"))
        pipe_count.append(s.count("|"))
        ampersand_count.append(s.count("&"))
        at_sign_count.append(s.count("@"))
        hash_count.append(s.count("#"))
        dollar_count.append(s.count("$"))
        percent_count.append(s.count("%"))
        caret_count.append(s.count("^"))
        asterisk_count.append(s.count("*"))
        plus_count.append(s.count("+"))
        equals_count.append(s.count("="))
        tilde_count.append(s.count("~"))
        backtick_count.append(s.count("`"))
        
        single_quote_count.append(s.count("'"))
        double_quote_count.append(s.count('"'))
        quote_ratio.append((s.count('"') + s.count("'"))/total_chars)
        
        paren_count.append(s.count("(") + s.count(")"))
        square_bracket_count.append(s.count("[") + s.count("]"))
        curly_bracket_count.append(s.count("{") + s.count("}"))
        angle_bracket_count.append(s.count("<") + s.count(">"))
        bracket_ratio.append((paren_count[-1] + square_bracket_count[-1] + curly_bracket_count[-1] + angle_bracket_count[-1])/total_chars)
        
        ellipsis_count.append(s.count("...") + s.count("â€¦"))
        newline_count.append(s.count("\n"))
        tab_count.append(s.count("\t"))
        multiple_space_count.append(len(re.findall(r'\s{2,}', s)))
        consecutive_punct_count.append(len(re.findall(r'[' + re.escape(string.punctuation) + ']{2,}', s)))
        
        special_char_ratio.append(sum(1 for ch in s if ch in "!@#$%^&*()_+-=[]{}|;:',.<>?/~`")/total_chars)
        math_symbol_ratio.append(sum(1 for ch in s if ch in "+=Ã—Ã·Â±âˆšâˆ�âˆ‘âˆ«â‰¤â‰¥â‰ ")/total_chars)
        
        numbers = re.findall(r'\d+', s)
        number_count.append(len(numbers))
        number_ratio.append(len(numbers)/(n_words + 1e-6))
        
        contractions = ["'t", "'s", "'re", "'ve", "'ll", "'d", "'m", "n't"]
        contraction_count.append(sum(s.lower().count(c) for c in contractions))

        sentences = re.split(r"[.!?]+", s)
        sentences = [x.strip() for x in sentences if x.strip()]
        n_sentences = len(sentences)
        sent_cnt.append(n_sentences)
        
        if n_sentences > 0 and n_words > 0:
            sent_lens = [len(re.findall(r"\b\w+\b", sent)) for sent in sentences]
            avg_sent_len.append(np.mean(sent_lens))
            median_sent_len.append(np.median(sent_lens))
            max_sent_len.append(np.max(sent_lens))
            min_sent_len.append(np.min(sent_lens))
            sent_len_std.append(np.std(sent_lens))
            avg_sentence_complexity.append(np.mean(sent_lens))
            uppercase_starts = sum(1 for sent in sentences if sent and sent[0].isupper())
            uppercase_sentence_start_ratio.append(uppercase_starts / n_sentences)
        else:
            for lst in [avg_sent_len, median_sent_len, max_sent_len, min_sent_len, sent_len_std, 
                       avg_sentence_complexity, uppercase_sentence_start_ratio]:
                lst.append(0.0)

        entropy_char.append(shannon_entropy_char(s))
        if words:
            word_counts = np.array([words.count(w) for w in set(words)], dtype=float)
            p_word = word_counts / word_counts.sum()
            entropy_word.append(float(-np.sum(p_word * np.log2(p_word + 1e-12))))
        else:
            entropy_word.append(0.0)

        unigram_repeat.append(repeat_ngram_ratio(words, n=1))
        bigram_repeat.append(repeat_ngram_ratio(words, n=2))
        trigram_repeat.append(repeat_ngram_ratio(words, n=3))
        fourgram_repeat.append(repeat_ngram_ratio(words, n=4))

        if n_words > 0 and n_sentences > 0:
            syllables = sum(count_syllables_en(w) for w in words)
            fk = 206.835 - 1.015*(n_words/n_sentences) - 84.6*(syllables/n_words)
            flesch_kincaid.append(fk)
            complex_words = sum(1 for w in words if count_syllables_en(w) >= 3)
            fog = 0.4 * ((n_words/n_sentences) + 100*(complex_words/n_words))
            gunning_fog.append(fog)
            if n_sentences >= 3:
                smog = 1.0430 * np.sqrt(complex_words * (30/n_sentences)) + 3.1291
                smog_index.append(smog)
            else:
                smog_index.append(0.0)
        else:
            flesch_kincaid.append(0.0)
            gunning_fog.append(0.0)
            smog_index.append(0.0)

        rc = sum(s.lower().count(m) for m in rhetorical_markers)
        rhet_count.append(rc/(n_words/100.0 + 1e-6))
        tc = sum(1 for w in words if w in transition_words)
        transition_count.append(tc/(n_words/100.0 + 1e-6))

    # Assign features
    feats.update({
        "word_count": wc, "char_count": char_count,
        "avg_word_len": avg_wlen, "median_word_len": median_wlen, "max_word_len": max_wlen,
        "min_word_len": min_wlen, "std_word_len": std_wlen,
        "word_len_skewness": word_len_skewness, "word_len_kurtosis": word_len_kurtosis,
        "word_length_variance": word_length_variance, "avg_chars_per_word": avg_chars_per_word,
        "type_token_ratio": ttr, "unique_word_ratio": unique_word_ratio, "lexical_diversity": lexical_diversity,
        "word_uniqueness": word_uniqueness, "hapax_legomena_ratio": hapax_legomena_ratio,
        "hapax_dislegomena_ratio": hapax_dislegomena_ratio, "stopword_ratio": stop_ratio,
        "long_word_ratio": long_word_ratio, "short_word_ratio": short_word_ratio,
        "medium_word_ratio": medium_word_ratio, "very_long_word_ratio": very_long_word_ratio,
        "punct_ratio": punct_ratio, "upper_ratio": upper_ratio, "lower_ratio": lower_ratio,
        "digit_ratio": digit_ratio, "space_ratio": space_ratio, "alpha_ratio": alpha_ratio, "alnum_ratio": alnum_ratio,
        "comma_count": comma_count, "period_count": period_count, "exclamation_count": exclamation_count,
        "question_count": question_count, "semicolon_count": semicolon_count, "colon_count": colon_count,
        "dash_count": dash_count, "underscore_count": underscore_count, "slash_count": slash_count,
        "backslash_count": backslash_count, "pipe_count": pipe_count, "ampersand_count": ampersand_count,
        "at_sign_count": at_sign_count, "hash_count": hash_count, "dollar_count": dollar_count,
        "percent_count": percent_count, "caret_count": caret_count, "asterisk_count": asterisk_count,
        "plus_count": plus_count, "equals_count": equals_count, "tilde_count": tilde_count, "backtick_count": backtick_count,
        "single_quote_count": single_quote_count, "double_quote_count": double_quote_count, "quote_ratio": quote_ratio,
        "paren_count": paren_count, "square_bracket_count": square_bracket_count,
        "curly_bracket_count": curly_bracket_count, "angle_bracket_count": angle_bracket_count, "bracket_ratio": bracket_ratio,
        "ellipsis_count": ellipsis_count, "newline_count": newline_count, "tab_count": tab_count,
        "multiple_space_count": multiple_space_count, "consecutive_punct_count": consecutive_punct_count,
        "special_char_ratio": special_char_ratio, "math_symbol_ratio": math_symbol_ratio,
        "number_count": number_count, "number_ratio": number_ratio, "contraction_count": contraction_count,
        "sentence_count": sent_cnt, "avg_sent_len": avg_sent_len, "median_sent_len": median_sent_len,
        "max_sent_len": max_sent_len, "min_sent_len": min_sent_len, "sent_len_std": sent_len_std,
        "sent_complexity": avg_sentence_complexity, "uppercase_sent_start_ratio": uppercase_sentence_start_ratio,
        "comma_per_100w": comma_per_100w, "period_per_100w": period_per_100w,
        "exclamation_per_100w": exclamation_per_100w, "question_per_100w": question_per_100w,
        "semicolon_per_100w": semicolon_per_100w, "colon_per_100w": colon_per_100w,
        "char_entropy": entropy_char, "word_entropy": entropy_word,
        "unigram_repeat": unigram_repeat, "bigram_repeat": bigram_repeat,
        "trigram_repeat": trigram_repeat, "fourgram_repeat": fourgram_repeat,
        "flesch_kincaid": flesch_kincaid, "gunning_fog": gunning_fog, "smog_index": smog_index,
        "rhetorical_markers_per_100w": rhet_count, "transition_words_per_100w": transition_count,
        "first_person_ratio": first_person_ratio, "second_person_ratio": second_person_ratio,
        "third_person_ratio": third_person_ratio, "avg_word_rank": avg_word_rank,
        "topic_len": [len(t) for t in topics],
        "topic_word_count": [len(re.findall(r"\b\w+\b", t)) for t in topics]
    })
    return pd.DataFrame(feats)

train_feats = ultra_style_features(train)
test_feats  = ultra_style_features(test)
print(f"  âœ“ Created {len(train_feats.columns)} features")

# Standardize
scaler = StandardScaler()
train_feats_scaled = pd.DataFrame(scaler.fit_transform(train_feats), columns=train_feats.columns, index=train.index)
test_feats_scaled = pd.DataFrame(scaler.transform(test_feats), columns=test_feats.columns, index=test.index)

# --------------------------
# 4) MULTIPLE EMBEDDINGS
# --------------------------
print("\n[4/16] Generating multiple embeddings...")

embedding_arrays_train = []
embedding_arrays_test = []
embedding_names = []

# 4.1) Sentence-BERT embeddings
if CONFIG['USE_SENTENCE_BERT']:
    try:
        print("  [1/5] Sentence-BERT (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        
        sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("    Encoding train...")
        sbert_train = sbert_model.encode(train["answer"].tolist(), show_progress_bar=False, batch_size=16)
        print("    Encoding test...")
        sbert_test = sbert_model.encode(test["answer"].tolist(), show_progress_bar=False, batch_size=16)
        
        # Reduce dimensions
        svd_sbert = TruncatedSVD(n_components=CONFIG['SBERT_DIM'], random_state=SEED)
        sbert_train_reduced = svd_sbert.fit_transform(sbert_train)
        sbert_test_reduced = svd_sbert.transform(sbert_test)
        
        embedding_arrays_train.append(sbert_train_reduced)
        embedding_arrays_test.append(sbert_test_reduced)
        embedding_names.append(f"SBERT-{CONFIG['SBERT_DIM']}")
        
        print(f"    âœ“ Shape: {sbert_train_reduced.shape}, Var: {svd_sbert.explained_variance_ratio_.sum():.3f}")
        del sbert_model, sbert_train, sbert_test
        gc.collect()
    except Exception as e:
        print(f"    âœ— Error: {e}")

# 4.2) DistilBERT embeddings
if CONFIG['USE_DISTILBERT']:
    try:
        print("  [2/5] DistilBERT...")
        from sentence_transformers import SentenceTransformer
        
        distilbert_model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
        
        print("    Encoding train...")
        distilbert_train = distilbert_model.encode(train["answer"].tolist(), show_progress_bar=False, batch_size=8)
        print("    Encoding test...")
        distilbert_test = distilbert_model.encode(test["answer"].tolist(), show_progress_bar=False, batch_size=8)
        
        svd_distilbert = TruncatedSVD(n_components=CONFIG['DISTILBERT_DIM'], random_state=SEED)
        distilbert_train_reduced = svd_distilbert.fit_transform(distilbert_train)
        distilbert_test_reduced = svd_distilbert.transform(distilbert_test)
        
        embedding_arrays_train.append(distilbert_train_reduced)
        embedding_arrays_test.append(distilbert_test_reduced)
        embedding_names.append(f"DistilBERT-{CONFIG['DISTILBERT_DIM']}")
        
        print(f"    âœ“ Shape: {distilbert_train_reduced.shape}, Var: {svd_distilbert.explained_variance_ratio_.sum():.3f}")
        del distilbert_model, distilbert_train, distilbert_test
        gc.collect()
    except Exception as e:
        print(f"    âœ— Error: {e}")

# 4.3) TF-IDF + SVD embeddings
if CONFIG['USE_TFIDF_SVD']:
    try:
        print("  [3/5] TF-IDF + SVD...")
        tfidf_emb = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=2, 
                                    max_features=10000, sublinear_tf=True)
        tfidf_emb_train = tfidf_emb.fit_transform(train["answer"])
        tfidf_emb_test = tfidf_emb.transform(test["answer"])
        
        svd_tfidf = TruncatedSVD(n_components=CONFIG['TFIDF_SVD_DIM'], random_state=SEED)
        tfidf_svd_train = svd_tfidf.fit_transform(tfidf_emb_train)
        tfidf_svd_test = svd_tfidf.transform(tfidf_emb_test)
        
        embedding_arrays_train.append(tfidf_svd_train)
        embedding_arrays_test.append(tfidf_svd_test)
        embedding_names.append(f"TFIDF-SVD-{CONFIG['TFIDF_SVD_DIM']}")
        
        print(f"    âœ“ Shape: {tfidf_svd_train.shape}, Var: {svd_tfidf.explained_variance_ratio_.sum():.3f}")
        del tfidf_emb, tfidf_emb_train, tfidf_emb_test
        gc.collect()
    except Exception as e:
        print(f"    âœ— Error: {e}")

# 4.4) Word2Vec embeddings
if CONFIG['USE_WORD2VEC']:
    try:
        print("  [4/5] Word2Vec...")
        # Train Word2Vec on the corpus
        train_sentences = [re.findall(r"\b\w+\b", text.lower()) for text in train["answer"]]
        test_sentences = [re.findall(r"\b\w+\b", text.lower()) for text in test["answer"]]
        
        w2v_model = Word2Vec(sentences=train_sentences, vector_size=CONFIG['WORD2VEC_DIM'], 
                            window=5, min_count=1, workers=4, seed=SEED, epochs=10)
        
        def get_doc_embedding(sentences, model):
            embeddings = []
            for sent in sentences:
                if len(sent) > 0:
                    vectors = [model.wv[word] for word in sent if word in model.wv]
                    if vectors:
                        embeddings.append(np.mean(vectors, axis=0))
                    else:
                        embeddings.append(np.zeros(model.vector_size))
                else:
                    embeddings.append(np.zeros(model.vector_size))
            return np.array(embeddings)
        
        w2v_train = get_doc_embedding(train_sentences, w2v_model)
        w2v_test = get_doc_embedding(test_sentences, w2v_model)
        
        embedding_arrays_train.append(w2v_train)
        embedding_arrays_test.append(w2v_test)
        embedding_names.append(f"Word2Vec-{CONFIG['WORD2VEC_DIM']}")
        
        print(f"    âœ“ Shape: {w2v_train.shape}")
        del w2v_model, train_sentences, test_sentences
        gc.collect()
    except Exception as e:
        print(f"    âœ— Error: {e}")

# 4.5) Gemma embeddings (original)
if CONFIG['USE_GEMMA']:
    try:
        if os.path.exists(GEMMA_PATH):
            print("  [5/5] EmbeddingGemma...")
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
            gemma_model = AutoModel.from_pretrained(GEMMA_PATH)
            gemma_model.eval()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gemma_model = gemma_model.to(device)
            
            def get_gemma_embeddings(texts, batch_size=8):
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = gemma_model(**inputs)
                        batch_embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(batch_embs)
                return np.vstack(embeddings)
            
            print("    Encoding train...")
            gemma_train = get_gemma_embeddings(train["answer"].tolist())
            print("    Encoding test...")
            gemma_test = get_gemma_embeddings(test["answer"].tolist())
            
            svd_gemma = TruncatedSVD(n_components=CONFIG['GEMMA_DIM'], random_state=SEED)
            gemma_train_reduced = svd_gemma.fit_transform(gemma_train)
            gemma_test_reduced = svd_gemma.transform(gemma_test)
            
            embedding_arrays_train.append(gemma_train_reduced)
            embedding_arrays_test.append(gemma_test_reduced)
            embedding_names.append(f"Gemma-{CONFIG['GEMMA_DIM']}")
            
            print(f"    âœ“ Shape: {gemma_train_reduced.shape}, Var: {svd_gemma.explained_variance_ratio_.sum():.3f}")
            del gemma_model, tokenizer, gemma_train, gemma_test
            gc.collect()
        else:
            print("  [5/5] Gemma model not found, skipping...")
    except Exception as e:
        print(f"    âœ— Error: {e}")

# Combine all embeddings
if embedding_arrays_train:
    train_embeddings_combined = np.hstack(embedding_arrays_train)
    test_embeddings_combined = np.hstack(embedding_arrays_test)
    print(f"\n  âœ“ Total embeddings: {len(embedding_names)} types")
    print(f"  âœ“ Combined shape: Train={train_embeddings_combined.shape}, Test={test_embeddings_combined.shape}")
    print(f"  âœ“ Embedding types: {', '.join(embedding_names)}")
    USE_EMBEDDINGS = True
else:
    train_embeddings_combined = None
    test_embeddings_combined = None
    USE_EMBEDDINGS = False
    print("  âš  No embeddings generated")

# --------------------------
# 5) Visualizations
# --------------------------
if CONFIG['CREATE_PLOTS']:
    print("\n[5/16] Creating visualizations...")
    os.makedirs("/kaggle/working/plots", exist_ok=True)
    
    # Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    train['is_cheating'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'coral'])
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xticklabels(['Human', 'AI'], rotation=0)
    train['is_cheating'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['skyblue', 'coral'])
    axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    plt.tight_layout()
    plt.savefig('/kaggle/working/plots/01_class_distribution.png', dpi=CONFIG['PLOT_DPI'], bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap
    correlation_data = train_feats.copy()
    correlation_data['target'] = y
    correlations = correlation_data.corr()['target'].abs().sort_values(ascending=False)
    top_features = correlations.head(26).index.tolist()[1:]
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    sns.heatmap(correlation_data[top_features + ['target']].corr(), annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0, square=True, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title('Correlation Matrix - Top 25', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/kaggle/working/plots/02_correlation_matrix.png', dpi=CONFIG['PLOT_DPI'], bbox_inches='tight')
    plt.close()
    
    print("  âœ“ Created 2 plots")
else:
    print("\n[5/16] Skipping visualizations")

# --------------------------
# 6) Target Encoding
# --------------------------
print("\n[6/16] Target encoding...")

def target_encode(train_series, y, test_series, n_splits=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_te = np.zeros(train_series.shape[0], dtype=float)
    global_mean = y.mean()
    for tr_idx, val_idx in skf.split(np.zeros_like(y), y):
        tr_topics, tr_y = train_series.iloc[tr_idx], y[tr_idx]
        means = pd.DataFrame({"topic": tr_topics.values, "y": tr_y}).groupby("topic")["y"].mean()
        oof_te[val_idx] = train_series.iloc[val_idx].map(means).fillna(global_mean).values
    full_means = pd.DataFrame({"topic": train_series.values, "y": y}).groupby("topic")["y"].mean()
    test_te = test_series.map(full_means).fillna(global_mean).values
    return oof_te.reshape(-1,1), test_te.reshape(-1,1)

train_topic_te, test_topic_te = target_encode(train["topic"], y, test["topic"])
print("  âœ“ Complete")

# --------------------------
# 7) TF-IDF
# --------------------------
print("\n[7/16] Creating TF-IDF features...")

tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=CONFIG['CHAR_NGRAM_RANGE'], min_df=2,
                             max_features=CONFIG['CHAR_MAX_FEATURES'], sublinear_tf=True, lowercase=True)
tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=CONFIG['WORD_NGRAM_RANGE'], min_df=2,
                             max_features=CONFIG['WORD_MAX_FEATURES'], token_pattern=r"(?u)\b\w+\b",
                             sublinear_tf=True, lowercase=True)
tfidf_topic = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1, max_features=50_000, 
                              sublinear_tf=True, lowercase=True)

Xc_train = tfidf_char.fit_transform(train["answer"])
Xc_test  = tfidf_char.transform(test["answer"])
Xw_train = tfidf_word.fit_transform(train["answer"])
Xw_test  = tfidf_word.transform(test["answer"])
Xt_train = tfidf_topic.fit_transform(train["topic"])
Xt_test  = tfidf_topic.transform(test["topic"])

print(f"  âœ“ Char: {Xc_train.shape}, Word: {Xw_train.shape}, Topic: {Xt_train.shape}")

# --------------------------
# 8) Seed
# --------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(SEED)

# --------------------------
# 9) Model Config
# --------------------------
print("\n[8/16] Configuring models...")

xgb_params = dict(n_estimators=1000, max_depth=6, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
                 reg_lambda=1.5, reg_alpha=0.1, min_child_weight=1.0, objective="binary:logistic",
                 eval_metric="auc", tree_method="hist", random_state=SEED)

lgb_params = dict(n_estimators=1200, max_depth=7, learning_rate=0.025, subsample=0.85, colsample_bytree=0.85,
                 reg_lambda=1.5, reg_alpha=0.8, min_child_samples=15, objective="binary", 
                 metric="auc", verbose=-1, random_state=SEED)

if CONFIG['USE_CATBOOST']:
    cb_params = dict(iterations=1000, depth=7, learning_rate=0.04, l2_leaf_reg=4.0, random_strength=0.6,
                    bagging_temperature=0.3, border_count=128, loss_function='Logloss',
                    eval_metric='AUC', random_seed=SEED, verbose=False)

num_models = 4 + (1 if CONFIG['USE_CATBOOST'] else 0) + (1 if CONFIG['USE_MLP'] else 0)
print(f"  âœ“ {num_models} models configured")

# --------------------------
# 10) K-Fold Training
# --------------------------
print(f"\n[9/16] Training {num_models} models with {N_FOLDS}-Fold CV...")
print("="*70)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# OOF & Test predictions
oof_char, oof_word, oof_xgb, oof_lgb = [np.zeros(len(train)) for _ in range(4)]
test_char, test_word, test_xgb, test_lgb = [np.zeros(len(test)) for _ in range(4)]

if CONFIG['USE_CATBOOST']:
    oof_cb, test_cb = np.zeros(len(train)), np.zeros(len(test))
if CONFIG['USE_MLP']:
    oof_mlp, test_mlp = np.zeros(len(train)), np.zeros(len(test))

# Feature importance
n_numeric_features = len(train_feats.columns) + 1
if USE_EMBEDDINGS:
    n_numeric_features += train_embeddings_combined.shape[1]

feature_importance_xgb = np.zeros(n_numeric_features)
feature_importance_lgb = np.zeros(n_numeric_features)
if CONFIG['USE_CATBOOST']:
    feature_importance_cb = np.zeros(n_numeric_features)

for fold, (tr_idx, va_idx) in enumerate(skf.split(Xc_train, y), 1):
    print(f"\n{'='*70}")
    print(f"FOLD {fold}/{N_FOLDS}")
    print(f"{'='*70}")
    
    Xc_tr, Xc_va = Xc_train[tr_idx], Xc_train[va_idx]
    Xw_tr, Xw_va = Xw_train[tr_idx], Xw_train[va_idx]
    Xt_tr, Xt_va = Xt_train[tr_idx], Xt_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Model 1: Char-LR
    print(f"  [1/{num_models}] Char-LR...")
    X_char_tr, X_char_va = hstack([Xc_tr, Xt_tr]).tocsr(), hstack([Xc_va, Xt_va]).tocsr()
    lr_char = LogisticRegression(C=5.0, solver="saga", penalty="l2", max_iter=5000, random_state=SEED)
    lr_char.fit(X_char_tr, y_tr)
    oof_char[va_idx] = lr_char.predict_proba(X_char_va)[:,1]

    # Model 2: Word-LR
    print(f"  [2/{num_models}] Word-LR...")
    X_word_tr, X_word_va = hstack([Xw_tr, Xt_tr]).tocsr(), hstack([Xw_va, Xt_va]).tocsr()
    lr_word = LogisticRegression(C=4.0, solver="saga", penalty="l2", max_iter=5000, random_state=SEED)
    lr_word.fit(X_word_tr, y_tr)
    oof_word[va_idx] = lr_word.predict_proba(X_word_va)[:,1]

    # Build feature matrix
    if USE_EMBEDDINGS:
        F_tr = np.hstack([train_feats_scaled.iloc[tr_idx].values, train_topic_te[tr_idx], train_embeddings_combined[tr_idx]])
        F_va = np.hstack([train_feats_scaled.iloc[va_idx].values, train_topic_te[va_idx], train_embeddings_combined[va_idx]])
    else:
        F_tr = np.hstack([train_feats_scaled.iloc[tr_idx].values, train_topic_te[tr_idx]])
        F_va = np.hstack([train_feats_scaled.iloc[va_idx].values, train_topic_te[va_idx]])

    # Model 3: XGBoost
    print(f"  [3/{num_models}] XGBoost...")
    dtr, dva = xgb.DMatrix(F_tr, label=y_tr), xgb.DMatrix(F_va, label=y_va)
    xgb_model = xgb.train(params=xgb_params, dtrain=dtr, num_boost_round=5000,
                         evals=[(dtr,"train"), (dva,"valid")], early_stopping_rounds=150, verbose_eval=False)
    oof_xgb[va_idx] = xgb_model.predict(dva)
    importance_dict = xgb_model.get_score(importance_type='gain')
    for key, val in importance_dict.items():
        feat_idx = int(key.replace('f', ''))
        if feat_idx < len(feature_importance_xgb):
            feature_importance_xgb[feat_idx] += val / N_FOLDS

    # Model 4: LightGBM
    print(f"  [4/{num_models}] LightGBM...")
    lgb_train = lgb.Dataset(F_tr, label=y_tr)
    lgb_valid = lgb.Dataset(F_va, label=y_va, reference=lgb_train)
    lgb_model = lgb.train(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_train, lgb_valid],
                         valid_names=['train', 'valid'], num_boost_round=5000,
                         callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False), lgb.log_evaluation(period=0)])
    oof_lgb[va_idx] = lgb_model.predict(F_va, num_iteration=lgb_model.best_iteration)
    feat_imp = lgb_model.feature_importance(importance_type='gain')
    feature_importance_lgb[:len(feat_imp)] += feat_imp / N_FOLDS

    model_num = 5
    # Model 5: CatBoost
    if CONFIG['USE_CATBOOST']:
        print(f"  [{model_num}/{num_models}] CatBoost...")
        cb_train, cb_valid = cb.Pool(F_tr, label=y_tr), cb.Pool(F_va, label=y_va)
        cb_model = cb.CatBoostClassifier(**cb_params)
        cb_model.fit(cb_train, eval_set=cb_valid, early_stopping_rounds=150, verbose=False)
        oof_cb[va_idx] = cb_model.predict_proba(F_va)[:,1]
        feat_imp = cb_model.get_feature_importance()
        feature_importance_cb[:len(feat_imp)] += feat_imp / N_FOLDS
        model_num += 1

    # Model 6: MLP
    if CONFIG['USE_MLP']:
        print(f"  [{model_num}/{num_models}] MLP...")
        mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
                           max_iter=600, random_state=SEED, early_stopping=True, validation_fraction=0.15,
                           learning_rate_init=0.001)
        mlp.fit(F_tr, y_tr)
        oof_mlp[va_idx] = mlp.predict_proba(F_va)[:,1]

    # Test predictions
    X_char_test = hstack([Xc_test, Xt_test]).tocsr()
    X_word_test = hstack([Xw_test, Xt_test]).tocsr()
    
    if USE_EMBEDDINGS:
        Ft = np.hstack([test_feats_scaled.values, test_topic_te, test_embeddings_combined])
    else:
        Ft = np.hstack([test_feats_scaled.values, test_topic_te])

    test_char += lr_char.predict_proba(X_char_test)[:,1] / N_FOLDS
    test_word += lr_word.predict_proba(X_word_test)[:,1] / N_FOLDS
    dte = xgb.DMatrix(Ft)
    test_xgb += xgb_model.predict(dte) / N_FOLDS
    test_lgb += lgb_model.predict(Ft, num_iteration=lgb_model.best_iteration) / N_FOLDS
    
    if CONFIG['USE_CATBOOST']:
        test_cb += cb_model.predict_proba(Ft)[:,1] / N_FOLDS
    if CONFIG['USE_MLP']:
        test_mlp += mlp.predict_proba(Ft)[:,1] / N_FOLDS

    # Fold metrics
    print(f"\n  Fold {fold} AUCs:")
    print(f"    Char-LR:  {roc_auc_score(y_va, oof_char[va_idx]):.5f}")
    print(f"    Word-LR:  {roc_auc_score(y_va, oof_word[va_idx]):.5f}")
    print(f"    XGBoost:  {roc_auc_score(y_va, oof_xgb[va_idx]):.5f}")
    print(f"    LightGBM: {roc_auc_score(y_va, oof_lgb[va_idx]):.5f}")
    if CONFIG['USE_CATBOOST']:
        print(f"    CatBoost: {roc_auc_score(y_va, oof_cb[va_idx]):.5f}")
    if CONFIG['USE_MLP']:
        print(f"    MLP:      {roc_auc_score(y_va, oof_mlp[va_idx]):.5f}")

# Overall OOF AUCs
print("\n" + "="*70)
print("OVERALL OUT-OF-FOLD AUCs")
print("="*70)
auc_char = roc_auc_score(y, oof_char)
auc_word = roc_auc_score(y, oof_word)
auc_xgb  = roc_auc_score(y, oof_xgb)
auc_lgb  = roc_auc_score(y, oof_lgb)

print(f"  Char-LR:  {auc_char:.5f}")
print(f"  Word-LR:  {auc_word:.5f}")
print(f"  XGBoost:  {auc_xgb:.5f}")
print(f"  LightGBM: {auc_lgb:.5f}")

model_names = ['Char-LR', 'Word-LR', 'XGBoost', 'LightGBM']
model_aucs = [auc_char, auc_word, auc_xgb, auc_lgb]

if CONFIG['USE_CATBOOST']:
    auc_cb = roc_auc_score(y, oof_cb)
    print(f"  CatBoost: {auc_cb:.5f}")
    model_names.append('CatBoost')
    model_aucs.append(auc_cb)

if CONFIG['USE_MLP']:
    auc_mlp = roc_auc_score(y, oof_mlp)
    print(f"  MLP:      {auc_mlp:.5f}")
    model_names.append('MLP')
    model_aucs.append(auc_mlp)

# --------------------------
# 11) More Visualizations
# --------------------------
if CONFIG['CREATE_PLOTS']:
    print("\n[10/16] Creating visualizations...")
    
    # ROC curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    models_dict = {'Char-LR': oof_char, 'Word-LR': oof_word, 'XGBoost': oof_xgb, 'LightGBM': oof_lgb}
    if CONFIG['USE_CATBOOST']:
        models_dict['CatBoost'] = oof_cb
    if CONFIG['USE_MLP']:
        models_dict['MLP'] = oof_mlp
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    for (name, preds), color in zip(models_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y, preds)
        auc = roc_auc_score(y, preds)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2.5, color=color)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Base Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/kaggle/working/plots/03_roc_curves.png', dpi=CONFIG['PLOT_DPI'], bbox_inches='tight')
    plt.close()
    
    print("  âœ“ Created 1 plot")
else:
    print("\n[10/16] Skipping visualizations")

# --------------------------
# 12) Stacking Ensemble
# --------------------------
print("\n[11/16] Training stacking ensemble...")

meta_features = [oof_char.reshape(-1,1), oof_word.reshape(-1,1), oof_xgb.reshape(-1,1), oof_lgb.reshape(-1,1)]
test_features = [test_char.reshape(-1,1), test_word.reshape(-1,1), test_xgb.reshape(-1,1), test_lgb.reshape(-1,1)]

if CONFIG['USE_CATBOOST']:
    meta_features.append(oof_cb.reshape(-1,1))
    test_features.append(test_cb.reshape(-1,1))
if CONFIG['USE_MLP']:
    meta_features.append(oof_mlp.reshape(-1,1))
    test_features.append(test_mlp.reshape(-1,1))

meta_train = np.hstack(meta_features + [train_feats_scaled.values, train_topic_te])
meta_test = np.hstack(test_features + [test_feats_scaled.values, test_topic_te])

blender = LogisticRegression(C=3.0, solver="lbfgs", max_iter=3000, random_state=SEED)
blender.fit(meta_train, y)

oof_blend = blender.predict_proba(meta_train)[:,1]
auc_blend = roc_auc_score(y, oof_blend)

print(f"\n{'='*70}")
print(f"FINAL STACKING AUC: {auc_blend:.5f}")
print(f"{'='*70}")

# Final predictions
test_pred = blender.predict_proba(meta_test)[:,1]
test_pred = np.clip(test_pred, 1e-6, 1-1e-6)

# --------------------------
# 13) Save Submission
# --------------------------
print("\n[12/16] Saving submission...")

sub = pd.DataFrame({"id": test["id"], "is_cheating": test_pred})
save_path = "/kaggle/working/submission.csv"
sub.to_csv(save_path, index=False)

print(f"\n{'='*70}")
print("âœ… SUBMISSION SAVED")
print(f"{'='*70}")
print(f"\nFile: {save_path}\n")
print(sub.head(10).to_string(index=False))

print(f"\nPrediction Stats:")
print(f"  Mean:   {test_pred.mean():.4f}")
print(f"  Median: {np.median(test_pred):.4f}")
print(f"  Std:    {test_pred.std():.4f}")
print(f"  Min:    {test_pred.min():.6f}")
print(f"  Max:    {test_pred.max():.6f}")

# --------------------------
# 14) Final Summary
# --------------------------
print(f"\n[13/16] Final Summary...")
print(f"\n{'='*70}")
print("ğŸ�‰ MULTI-EMBEDDINGS PIPELINE COMPLETE")
print(f"{'='*70}")

print(f"\nğŸ“Š DATA:")
print(f"  Features: {len(train_feats.columns)}")
if USE_EMBEDDINGS:
    print(f"  Embeddings: {len(embedding_names)} types")
    print(f"    - {', '.join(embedding_names)}")
    print(f"    - Total dims: {train_embeddings_combined.shape[1]}")
else:
    print(f"  Embeddings: None")

print(f"\nğŸ¤– MODELS ({num_models + 1}):")
for i, name in enumerate(model_names, 1):
    print(f"  {i}. {name}")
print(f"  {num_models+1}. Stacking Ensemble")

print(f"\nğŸ“ˆ PERFORMANCE (OOF AUC):")
all_names = model_names + ['ENSEMBLE']
all_aucs = model_aucs + [auc_blend]
for name, auc in zip(all_names, all_aucs):
    print(f"  {name:12s}: {auc:.5f}")

best_base = max(model_aucs)
print(f"\nğŸš€ IMPROVEMENT:")
print(f"  Best base:  {best_base:.5f}")
print(f"  Ensemble:   {auc_blend:.5f}")
print(f"  Gain:       +{(auc_blend - best_base):.5f}")

print(f"\n[14/16] Done!")
print(f"{'='*70}")
print("âœ¨ Good luck with your submission! ğŸš€")
print(f"{'='*70}")

gc.collect()

