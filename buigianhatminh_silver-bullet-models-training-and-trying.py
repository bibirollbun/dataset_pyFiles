import polars as pl
import pandas as pd
import numpy as np
import re
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from scipy.stats import skew, kurtosis
import warnings
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings("ignore")


num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
text_changes = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']


def count_by_values(df, colname, values):
    fts = df.select(pl.col('id').unique(maintain_order=True))
    for i, value in enumerate(values):
        tmp_df = df.group_by('id').agg(pl.col(colname).is_in([value]).sum().alias(f'{colname}_{i}_cnt'))
        fts  = fts.join(tmp_df, on='id', how='left') 
    return fts


def dev_feats(df):
    
    print("< Count by values features >")
    
    feats = count_by_values(df, 'activity', activities)
    feats = feats.join(count_by_values(df, 'text_change', text_changes), on='id', how='left') 
    feats = feats.join(count_by_values(df, 'down_event', events), on='id', how='left') 
    feats = feats.join(count_by_values(df, 'up_event', events), on='id', how='left') 

    print("< Input words stats features >")

    temp = df.filter((~pl.col('text_change').str.contains('=>')) & (pl.col('text_change') != 'NoChange'))
    temp = temp.group_by('id').agg(pl.col('text_change').str.concat('').str.extract_all(r'q+'))
    temp = temp.with_columns(input_word_count = pl.col('text_change').list.lengths(),
                             input_word_length_mean = pl.col('text_change').apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_max = pl.col('text_change').apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_std = pl.col('text_change').apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_median = pl.col('text_change').apply(lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_skew = pl.col('text_change').apply(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0)))
    temp = temp.drop('text_change')
    feats = feats.join(temp, on='id', how='left') 


    
    print("< Numerical columns features >")

    temp = df.group_by("id").agg(pl.sum('action_time').suffix('_sum'), pl.mean(num_cols).suffix('_mean'), pl.std(num_cols).suffix('_std'),
                                 pl.median(num_cols).suffix('_median'), pl.min(num_cols).suffix('_min'), pl.max(num_cols).suffix('_max'),
                                 pl.quantile(num_cols, 0.5).suffix('_quantile'))
    feats = feats.join(temp, on='id', how='left') 


    print("< Categorical columns features >")
    
    temp  = df.group_by("id").agg(pl.n_unique(['activity', 'down_event', 'up_event', 'text_change']))
    feats = feats.join(temp, on='id', how='left') 


    
    print("< Idle time features >")

    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.group_by("id").agg(inter_key_largest_lantency = pl.max('time_diff'),
                                   inter_key_median_lantency = pl.median('time_diff'),
                                   mean_pause_time = pl.mean('time_diff'),
                                   std_pause_time = pl.std('time_diff'),
                                   total_pause_time = pl.sum('time_diff'),
                                   pauses_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).count(),
                                   pauses_1_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).count(),
                                   pauses_1_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).count(),
                                   pauses_2_sec = pl.col('time_diff').filter((pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).count(),
                                   pauses_3_sec = pl.col('time_diff').filter(pl.col('time_diff') > 3).count(),)
    feats = feats.join(temp, on='id', how='left') 
    
    print("< P-bursts features >")

    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('time_diff')<2)
    temp = temp.with_columns(pl.when(pl.col("time_diff") & pl.col("time_diff").is_last()).then(pl.count()).over(pl.col("time_diff").rle_id()).alias('P-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(pl.mean('P-bursts').suffix('_mean'), pl.std('P-bursts').suffix('_std'), pl.count('P-bursts').suffix('_count'),
                                   pl.median('P-bursts').suffix('_median'), pl.max('P-bursts').suffix('_max'),
                                   pl.first('P-bursts').suffix('_first'), pl.last('P-bursts').suffix('_last'))
    feats = feats.join(temp, on='id', how='left') 


    print("< R-bursts features >")

    temp = df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('activity').is_in(['Remove/Cut']))
    temp = temp.with_columns(pl.when(pl.col("activity") & pl.col("activity").is_last()).then(pl.count()).over(pl.col("activity").rle_id()).alias('R-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(pl.mean('R-bursts').suffix('_mean'), pl.std('R-bursts').suffix('_std'), 
                                   pl.median('R-bursts').suffix('_median'), pl.max('R-bursts').suffix('_max'),
                                   pl.first('R-bursts').suffix('_first'), pl.last('R-bursts').suffix('_last'))
    feats = feats.join(temp, on='id', how='left')
    
    return feats


def train_valid_split(data_x, data_y, train_idx, valid_idx):
    x_train = data_x.iloc[train_idx]
    y_train = data_y[train_idx]
    x_valid = data_x.iloc[valid_idx]
    y_valid = data_y[valid_idx]
    return x_train, y_train, x_valid, y_valid


from sklearn.metrics import mean_squared_error
def evaluate(data_x, data_y, model, random_state=42, n_splits=10, test_x=None, return_score=True):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    test_y = np.zeros(len(data_x)) if (test_x is None) else np.zeros((len(test_x), n_splits))
    fold_rmse = []

    for i, (train_index, valid_index) in enumerate(skf.split(data_x, data_y.astype(str))):
        print(f'<Cross Validation round {i}>')
        train_x, train_y, valid_x, valid_y = train_valid_split(data_x, data_y, train_index, valid_index)
        model.fit(train_x, train_y)

        preds_valid = model.predict(valid_x)
        rmse_i = np.sqrt(mean_squared_error(valid_y, preds_valid, squared=False))  # RMSE for this fold
        fold_rmse.append(rmse_i)

        if test_x is None:
            test_y[valid_index] = preds_valid
        else:
            test_y[:, i] = model.predict(test_x)

    avg_rmse = np.mean(fold_rmse)

    if test_x is None:
        if return_score:
            return test_y, avg_rmse
        else:
            return test_y
    else:
        preds_test = np.mean(test_y, axis=1)
        if return_score:
            return preds_test, avg_rmse
        else:
            return preds_test


def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']

def reconstruct_essay(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        if Input[0] == 'Replace':
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue
        if "M" in Input[0]:
            croppedTxt = Input[0][10:]
            splitTxt = croppedTxt.split(' To ')
            valueArr = [item.split(', ') for item in splitTxt]
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            if moveData[0] != moveData[2]:
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def get_essay_df(df):
    df       = df[df.activity != 'Nonproduction']
    temp     = df.groupby('id').apply(lambda x: reconstruct_essay(x[['activity', 'cursor_position', 'text_change']]))
    essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
    essay_df = essay_df.merge(temp.rename('essay'), on='id')
    return essay_df


def word_feats(df):
    essay_df = df
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]

    word_agg_df = df[['id','word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def sent_feats(df):
    df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    df['sent_len'] = df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))
    df = df[df.sent_len!=0].reset_index(drop=True)

    sent_agg_df = pd.concat([df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), 
                             df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df


def parag_feats(df):
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    # Number of characters in paragraphs
    df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x)) 
    # Number of words in paragraphs
    df['paragraph_word_count'] = df['paragraph'].apply(lambda x: len(x.split(' ')))
    df = df[df.paragraph_len!=0].reset_index(drop=True)
    
    paragraph_agg_df = pd.concat([df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS), 
                                  df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1) 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

def product_to_keys(logs, essays):
    essays['product_len'] = essays.essay.str.len()
    tmp_df = logs[logs.activity.isin(['Input', 'Remove/Cut'])].groupby(['id']).agg({'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
    essays = essays.merge(tmp_df, on='id', how='left')
    essays['product_to_keys'] = essays['product_len'] / essays['keys_pressed']
    return essays[['id', 'product_to_keys']]

def get_keys_pressed_per_second(logs):
    temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
    temp_df_2 = logs.groupby(['id']).agg(min_down_time=('down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
    temp_df = temp_df.merge(temp_df_2, on='id', how='left')
    temp_df['keys_per_second'] = temp_df['keys_pressed'] / ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
    return temp_df[['id', 'keys_per_second']]

def target_encoding(train_df, scores, feature):
    
    train_df['target'] = train_df['id'].map(dict(scores.values))
    
    down_event_counts = train_df[feature].value_counts()
    rare_down_events = down_event_counts[down_event_counts <= 3].index
    # Replace 'target' values with NaN for these rare events
    train_df.loc[train_df[feature].isin(rare_down_events), 'target'] = np.nan    

    # Step 2: Calculate the mean 'target' for each 'down_event'
    mean_target_by_down_event = train_df.groupby(feature)['target'].mean().reset_index(name=f'{feature}_mean_target')
    train_df.drop(columns=["target"], inplace=True)
    
    return mean_target_by_down_event



# Newly added essay feature
def essay_feats(df, df_test = None):
    
    TEXT_COL = "essay"

    df["no_comma"] = df[TEXT_COL].str.count(",")
    df["no_quotes"] = df[TEXT_COL].str.count('"') + df[TEXT_COL].str.count("'")
    df["no_spaces"] = df[TEXT_COL].str.count(" ")
    df["no_dot"] = df[TEXT_COL].str.count(r"\.")
    df["no_exclamation"] = df[TEXT_COL].str.count("!")
    df["no_question"] = df[TEXT_COL].str.count(r"\?")
    df["no_semicolon"] = df[TEXT_COL].str.count(":")
    df["no_dot_space"] = np.log1p(df["no_dot"]) * np.log1p(df["no_spaces"])

    # === Bag of Words (Words Level) ===
    word_vectorizer = CountVectorizer(max_features=25, ngram_range=(1, 1))
    bow_words = word_vectorizer.fit_transform(df[TEXT_COL].fillna(""))
    
    bow_words_df = pd.DataFrame(
        bow_words.toarray(), 
        columns=[f"bow_word_{w}" for w in word_vectorizer.get_feature_names_out()]
    )

    '''
    # === Bag of Words (Paragraph Level) ===
    # Treat each paragraph as a token separated by '\n\n'
    paragraph_vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split("\n\n"), 
        max_features=25,
        lowercase=False
    )
    bow_paragraphs = paragraph_vectorizer.fit_transform(df[TEXT_COL].fillna(""))
    
    bow_paragraphs_df = pd.DataFrame(
        bow_paragraphs.toarray(),
        columns=[f"bow_para_{w}" for w in range(len(paragraph_vectorizer.get_feature_names_out()))]
    )
    '''

    # Word-level TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=40, ngram_range=(1, 1))
    tfidf_words = tfidf_vectorizer.fit_transform(df[TEXT_COL].fillna(""))
    
    tfidf_df = pd.DataFrame(
        tfidf_words.toarray(),
        columns=[f"tfidf_{w}" for w in tfidf_vectorizer.get_feature_names_out()]
    )

    if df_test is not None:
        df_test["no_comma"] = df_test[TEXT_COL].str.count(",")
        df_test["no_quotes"] = df_test[TEXT_COL].str.count('"') + df[TEXT_COL].str.count("'")
        df_test["no_spaces"] = df_test[TEXT_COL].str.count(" ")
        df_test["no_dot"] = df_test[TEXT_COL].str.count(r"\.")
        df_test["no_exclamation"] = df_test[TEXT_COL].str.count("!")
        df_test["no_question"] = df_test[TEXT_COL].str.count(r"\?")
        df_test["no_semicolon"] = df_test[TEXT_COL].str.count(":")
        df_test["no_dot_space"] = np.log1p(df["no_dot"]) * np.log1p(df["no_spaces"])

        bow_words_test = word_vectorizer.transform(df_test[TEXT_COL].fillna(""))
        bow_words_df_test = pd.DataFrame(
            bow_words_test.toarray(), 
            columns=[f"bow_word_{w}" for w in word_vectorizer.get_feature_names_out()]
        )
        
        '''
        bow_paragraphs_test = paragraph_vectorizer.transform(df_test[TEXT_COL].fillna(""))
        bow_paragraphs_df_test = pd.DataFrame(
                bow_paragraphs_test.toarray(),
                columns=[f"bow_para_{w}" for w in range(len(paragraph_vectorizer.get_feature_names_out()))]
            )
        '''

        tfidf_words_test = tfidf_vectorizer.transform(df_test[TEXT_COL].fillna(""))
        tfidf_df_test = pd.DataFrame(
            tfidf_words_test.toarray(),
            columns=[f"tfidf_{w}" for w in tfidf_vectorizer.get_feature_names_out()]
        )

        #features_df_test = pd.concat([df_test, bow_words_df_test, bow_paragraphs_df_test, tfidf_df_test], axis=1)
        features_df_test = pd.concat([df_test, bow_words_df_test, tfidf_df_test], axis=1)
    
    # === Combine all features ===
    #features_df = pd.concat([df, bow_words_df, bow_paragraphs_df, tfidf_df], axis=1)
    features_df = pd.concat([df, bow_words_df, tfidf_df], axis=1)
    
    return features_df, features_df_test


from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def get_vector_representation(
    df, 
    df_test=None, 
    VOCAB_SIZE=5000, 
    LOWERCASE=True,
    analyzer_type='word',     # 'word' or 'char_wb'
    use_stopwords=True,        # applies only if analyzer_type='word'
    ngram_range = (1,3)
):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from tqdm import tqdm
    from datasets import Dataset
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    TEXT_COL = "essay"

    # Combine training & test texts for tokenizer training if test provided
    texts = df[TEXT_COL].fillna('').tolist()
    if df_test is not None:
        texts += df_test[TEXT_COL].fillna('').tolist()

    # --- Create BPE tokenizer ---
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC()] + ([normalizers.Lowercase()] if LOWERCASE else [])
    )
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

    dataset = Dataset.from_dict({"text": texts})
    def train_corp_iter():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]
    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # --- Tokenize and reconstruct texts ---
    def bpe_tokenize(text):
        return " ".join(tokenizer.tokenize(text))

    tqdm.pandas(desc="Tokenizing training essays")
    df["bpe_text"] = df[TEXT_COL].fillna("").progress_apply(bpe_tokenize)
    df["bpe_text"] = df["bpe_text"].apply(
        lambda x: x if x.strip().replace("Ġ", "").strip() != "" else "<EMPTY>"
    )

    if df_test is not None:
        tqdm.pandas(desc="Tokenizing test essays")
        df_test["bpe_text"] = df_test[TEXT_COL].fillna("").progress_apply(bpe_tokenize)
        df_test["bpe_text"] = df_test["bpe_text"].apply(
            lambda x: x if x.strip().replace("Ġ", "").strip() != "" else "<EMPTY>"
        )

    # --- Choose analyzer mode ---
    analyzer_type = analyzer_type.lower()
    if analyzer_type not in ['word', 'char_wb']:
        raise ValueError("analyzer_type must be either 'word' or 'char_wb'")

    stop_words = 'english' if (analyzer_type == 'word' and use_stopwords) else None

    # --- Vectorization ---
    count_vec = CountVectorizer(
        max_features=VOCAB_SIZE,
        ngram_range=ngram_range,
        analyzer=analyzer_type,
        stop_words=stop_words
    )

    tfidf_vec = TfidfVectorizer(
        max_features=VOCAB_SIZE,
        ngram_range=ngram_range,
        analyzer=analyzer_type,
        stop_words=stop_words
    )

    count_vec.fit(df["bpe_text"])
    tfidf_vec.fit(df["bpe_text"])

    X_count = count_vec.transform(df["bpe_text"]).astype(float)
    X_tfidf = tfidf_vec.transform(df["bpe_text"]).astype(float)

    if df_test is not None:
        X_count_test = count_vec.transform(df_test["bpe_text"]).astype(float)
        X_tfidf_test = tfidf_vec.transform(df_test["bpe_text"]).astype(float)
        return X_count, X_tfidf, X_count_test, X_tfidf_test

    return X_count, X_tfidf, None, None


# Get evaluation of essay from model as feature
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import pandas as pd

def pred_feats(df, df_test=None, data_path='/kaggle/input/linking-writing-processes-to-writing-quality/', n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    pred_feat = df[['id', 'essay']].copy()
    pred_feat["essay_binary_pred_count"] = 0
    pred_feat["essay_regression_pred_count"] = 0
    pred_feat["essay_binary_pred_tfidf"] = 0
    pred_feat["essay_regression_pred_tfidf"] = 0
    
    X_count, X_tfidf, X_count_test, X_tfidf_test = get_vector_representation(df, df_test)
    
    train_scores = pd.read_csv(data_path + 'train_scores.csv')
    y = train_scores['score']
    y_bin = (y > 3.5).astype(int)

    # --- Store out-of-fold predictions ---
    oof_pred_bin_count = np.zeros(len(df))
    oof_pred_reg_count = np.zeros(len(df))
    oof_pred_bin_tfidf = np.zeros(len(df))
    oof_pred_reg_tfidf = np.zeros(len(df))

    fold = 0
    for train_idx, val_idx in kf.split(X_count):
        print(f"<Fold {fold+1}/{n_splits}>")

        # --- CountVectorizer Binary ---
        lgbm_param = {'n_estimators': 988, 
                      'learning_rate': 0.008161152988053227, 
                      'num_leaves': 118, 
                      'max_depth': 13, 
                      'min_child_samples': 45, 
                      'subsample': 0.6470722392501642, 
                      'colsample_bytree': 0.8451083824286914, 
                      'reg_alpha': 0.23871825951780937, 
                      'reg_lambda': 0.57460632521018373252101837 ,
                      'objective':'binary', 
                      'metric':'binary_logloss', 
                      'random_state':42}
        clf = LGBMClassifier(**lgbm_param)
        clf.fit(X_count[train_idx], y_bin.iloc[train_idx])
        oof_pred_bin_count[val_idx] = (clf.predict_proba(X_count[val_idx])[:, 1] >= 0.5).astype(int)
        pred_feat.loc[val_idx, "essay_binary_pred_count"] = oof_pred_bin_count[val_idx]
        
        # --- CountVectorizer Regression ---
        reg = LGBMRegressor(objective='regression', metric='rmse', random_state=42)
        reg.fit(X_count[train_idx], y.iloc[train_idx])
        oof_pred_reg_count[val_idx] = reg.predict(X_count[val_idx])
        pred_feat.loc[val_idx, "essay_regression_pred_count"] = oof_pred_reg_count[val_idx]
        
        # --- TF-IDF Binary ---
        clf2 = LGBMClassifier(**lgbm_param)
        clf2.fit(X_tfidf[train_idx], y_bin.iloc[train_idx])
        oof_pred_bin_tfidf[val_idx] = (clf2.predict_proba(X_tfidf[val_idx])[:, 1] >= 0.5).astype(int)
        pred_feat.loc[val_idx, "essay_binary_pred_tfidf"] = oof_pred_bin_tfidf[val_idx]
        
        # --- TF-IDF Regression ---
        reg2 = LGBMRegressor(objective='regression', metric='rmse', random_state=42)
        reg2.fit(X_tfidf[train_idx], y.iloc[train_idx])
        oof_pred_reg_tfidf[val_idx] = reg2.predict(X_tfidf[val_idx])
        pred_feat.loc[val_idx, "essay_regression_pred_tfidf"] = oof_pred_reg_tfidf[val_idx]

        fold += 1

    # --- Compute Cross-Validation Scores ---
    acc_count = accuracy_score(y_bin, oof_pred_bin_count)
    rmse_count = mean_squared_error(y, oof_pred_reg_count, squared=False)
    acc_tfidf = accuracy_score(y_bin, oof_pred_bin_tfidf)
    rmse_tfidf = mean_squared_error(y, oof_pred_reg_tfidf, squared=False)

    print(f"\nCV Results over {n_splits} folds:")
    print(f"  CountVectorizer Binary Accuracy:  {acc_count:.4f}")
    print(f"  CountVectorizer Regression RMSE: {rmse_count:.4f}")
    print(f"  TFIDF Binary Accuracy:            {acc_tfidf:.4f}")
    print(f"  TFIDF Regression RMSE:           {rmse_tfidf:.4f}\n")

    # --- Test predictions (train full model on all data) ---
    pred_feat_test = None
    if df_test is not None:
        clf.fit(X_count, y_bin)
        reg.fit(X_count, y)
        clf2.fit(X_tfidf, y_bin)
        reg2.fit(X_tfidf, y)
        
        pred_feat_test = df_test[['id', 'essay']].copy()
        pred_feat_test["essay_binary_pred_count"] = clf.predict(X_count_test)
        pred_feat_test["essay_regression_pred_count"] = reg.predict(X_count_test)
        pred_feat_test["essay_binary_pred_tfidf"] = clf2.predict(X_tfidf_test)
        pred_feat_test["essay_regression_pred_tfidf"] = reg2.predict(X_tfidf_test)
        pred_feat_test = pred_feat_test.drop(['essay'], axis=1)

    pred_feat = pred_feat.drop(['essay'], axis=1)
    return pred_feat, pred_feat_test


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def build_density_features(X_train, y_train, X_test=None, df_train=None, df_test=None,
                           score_bins=(0, 2, 4, 6), n_components=4, method="gmm", method_id = 'word'):
   
    print("< Density Feature Extraction >")

    # Convert to dense if needed
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    if X_test is not None:
        X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    else:
        X_test_dense = None

    # --- Standardize & Reduce Dimensionality ---
    print("Applying PCA dimensionality reduction...")
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_dense)
    if X_test_dense is not None:
        X_test_scaled = scaler.transform(X_test_dense)

    pca = PCA(n_components=64, random_state=42)
    X_train_reduced = pca.fit_transform(X_train_scaled)
    if X_test_dense is not None:
        X_test_reduced = pca.transform(X_test_scaled)

    # --- Bin scores into groups (e.g. low/mid/high) ---
    y_bins = np.digitize(y_train, bins=score_bins)
    unique_bins = np.unique(y_bins)

    # --- Fit density models per score group ---
    models = {}
    for b in unique_bins:
        mask = y_bins == b
        X_group = X_train_reduced[mask]
        if len(X_group) < 5:
            continue

        if method == "gmm":
            model = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=42
            ).fit(X_group)
        elif method == "kde":
            model = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_group)
        else:
            raise ValueError("method must be 'gmm' or 'kde'")

        models[b] = model
        print(f"Fitted {method.upper()} for group {b} (n={len(X_group)})")

    # --- Helper: compute densities ---
    def compute_density_features(X_reduced):
        log_probs = np.zeros((len(X_reduced), len(unique_bins)))
        for i, b in enumerate(unique_bins):
            if b in models:
                log_probs[:, i] = models[b].score_samples(X_reduced)
            else:
                log_probs[:, i] = -np.inf

        # Ratio & softmax
        density_ratio = log_probs[:, -1] - log_probs[:, 0] if len(unique_bins) >= 2 else np.zeros(len(X_reduced))
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        feats = {}
        for group in range(len(score_bins)):
            feats[f"posterior_{group}_{method_id}"] = probs[:,group]

        return pd.DataFrame(feats)

    # --- Compute for train ---
    train_density = compute_density_features(X_train_reduced)
    df_train_out = pd.concat([df_train, train_density], axis=1)

    # --- Compute for test ---
    df_test_out = None
    if X_test_dense is not None and df_test is not None:
        test_density = compute_density_features(X_test_reduced)
        df_test_out = df_test.reset_index(drop=True).copy()
        df_test_out = pd.concat([df_test_out, test_density], axis=1)

    # --- Save fitted components for reuse ---
    fitted_components = {
        "scaler": scaler,
        "pca": pca,
        "models": models,
        "unique_bins": unique_bins
    }

    print("✅ Density-based features successfully added.")
    return df_train_out, df_test_out, fitted_components



import optuna
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, GroupKFold
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np

def objective(trial, df, task="classification"):
    # --- 1️⃣ Get vector representation ---
    X_count, X_tfidf, _, _ = get_vector_representation(df)
    
    train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')
    y = train_scores['score']
    y_bin = (y > 3.5).astype(int)
    
    # --- 2️⃣ Hyperparameters to tune ---
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.01, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 120),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'verbosity': -1
    }

    groups = df['id']
    gkf = GroupKFold(n_splits=5)
    scores = []

    # --- 3️⃣ Train + CV ---
    for train_idx, val_idx in gkf.split(X_count, y, groups):
        if task == "classification":
            model = LGBMClassifier(**params, objective='binary', metric='binary_logloss')
            model.fit(X_count[train_idx], y_bin.iloc[train_idx])
            preds = model.predict(X_count[val_idx])
            acc = accuracy_score(y_bin.iloc[val_idx], preds)
            scores.append(acc)
        else:
            model = LGBMRegressor(**params, objective='regression', metric='rmse')
            model.fit(X_count[train_idx], y.iloc[train_idx])
            preds = model.predict(X_count[val_idx])
            rmse = mean_squared_error(y.iloc[val_idx], preds, squared=False)
            scores.append(rmse)

    # --- 4️⃣ Return mean CV score ---
    return np.mean(scores) if task == "classification" else np.mean(scores)

# --- 5️⃣ Run study (example for classifier) ---
def tune_model(df, task="classification", n_trials=30):
    direction = "maximize" if task == "classification" else "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(lambda trial: objective(trial, df, task), n_trials=n_trials)
    
    print(f"Best params for {task}: {study.best_trial.params}")
    print(f"Best score: {study.best_value:.4f}")
    return study.best_trial.params


data_path     = '/kaggle/input/linking-writing-processes-to-writing-quality/'
train_logs    = pl.scan_csv(data_path + 'train_logs.csv')
train_feats   = dev_feats(train_logs)
train_feats   = train_feats.collect().to_pandas()

print('< Essay Reconstruction >')
train_logs             = train_logs.collect().to_pandas()
train_essays           = get_essay_df(train_logs)
train_feats            = train_feats.merge(word_feats(train_essays), on='id', how='left')
train_feats            = train_feats.merge(sent_feats(train_essays), on='id', how='left')
train_feats            = train_feats.merge(parag_feats(train_essays), on='id', how='left')
train_feats            = train_feats.merge(get_keys_pressed_per_second(train_logs), on='id', how='left')
train_feats            = train_feats.merge(product_to_keys(train_logs, train_essays), on='id', how='left')


print('< Testing Data >')
test_logs   = pl.scan_csv(data_path + 'test_logs.csv')
test_feats  = dev_feats(test_logs)
test_feats  = test_feats.collect().to_pandas()

test_logs             = test_logs.collect().to_pandas()
test_essays           = get_essay_df(test_logs)
test_feats            = test_feats.merge(word_feats(test_essays), on='id', how='left')
test_feats            = test_feats.merge(sent_feats(test_essays), on='id', how='left')
test_feats            = test_feats.merge(parag_feats(test_essays), on='id', how='left')
test_feats            = test_feats.merge(get_keys_pressed_per_second(test_logs), on='id', how='left')
test_feats            = test_feats.merge(product_to_keys(test_logs, test_essays), on='id', how='left')


X_count, X_tfidf, X_count_test, X_tfidf_test = get_vector_representation(train_essays, test_essays, analyzer_type = 'char_wb')
    
train_scores = pd.read_csv(data_path + 'train_scores.csv')
y = train_scores['score']
y_bin = (y > 3.5).astype(int)

train_feats, test_feats, density_model = build_density_features(
    X_tfidf, y, 
    X_test=X_tfidf_test, 
    df_train=train_feats, 
    df_test=test_feats,
    method="gmm",   # or 'kde'
    method_id = 'char_wb' #vectorization method id, to prevent duplicate features later
)

X_count, X_tfidf, X_count_test, X_tfidf_test = get_vector_representation(train_essays, test_essays, analyzer_type = 'word')

train_feats, test_feats, density_model = build_density_features(
    X_tfidf, y, 
    X_test=X_tfidf_test, 
    df_train=train_feats, 
    df_test=test_feats,
    method="gmm",   
    method_id = 'word' 
)

X_count, X_tfidf, X_count_test, X_tfidf_test = get_vector_representation(train_essays, test_essays, analyzer_type = 'char_wb', ngram_range = (5,6))

train_feats, test_feats, density_model = build_density_features(
    X_tfidf, y, 
    X_test=X_tfidf_test, 
    df_train=train_feats, 
    df_test=test_feats,
    method="gmm",  
    method_id = 'char_wb_56'
)


print('< Mapping >')
train_scores   = pd.read_csv(data_path + 'train_scores.csv')
data           = train_feats.merge(train_scores, on='id', how='left')
x              = data.drop(['id', 'score'], axis=1)
y              = data['score'].values
print(f'Number of features: {len(x.columns)}')

test_ids = test_feats['id'].values
testin_x = test_feats.drop(['id'], axis=1)

print(train_feats.shape, test_feats.shape)


train_feats


'''
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
print('< Learning and Evaluation >')
xgb_param = {'n_estimators' : 1024,
      'colsample_bytree' : 0.375,
      'colsample_bylevel' : 0.375,
      'colsample_bynode' : 0.375,
      'subsample' : 0.375,
      'eta' : 0.01,
      'max_depth' : 4,
      'min_child_weight' : 9}
xgb_reg = xgb.XGBRegressor(**xgb_param)
xgb_reg.fit(x,y)

booster = xgb_reg.get_booster()
importance = booster.get_score(importance_type='gain')

# Sort by importance value (descending)
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Get the first 100 feature names
top_100_features = [feat for feat, score in sorted_features[:100]]
top_100_features.extend(["id"])

print(top_100_features)
'''


'''
#train_feats = train_feats[top_100_features]
#test_feats = test_feats[top_100_features]

#essay_feats_train, essay_feats_test = essay_feats(train_essays, test_essays)
pred_feats_train, pred_feats_test = pred_feats(train_essays, test_essays)
#train_feats            = train_feats.merge(essay_feats_train, on='id', how='left')
train_feats            = train_feats.merge(pred_feats_train, on='id', how='left')
train_feats            = train_feats.drop(['essay', 'word', 'sent', 'paragraph','bpe_text'],axis = 1, errors = 'ignore')
#test_feats             = test_feats.merge(essay_feats_test, on='id', how='left')
test_feats             = test_feats.merge(pred_feats_test, on='id', how='left')
test_feats             = test_feats.drop(['essay', 'word', 'sent', 'paragraph','bpe_text'],axis = 1, errors = 'ignore')


print('< Mapping >')
train_scores   = pd.read_csv(data_path + 'train_scores.csv')
data           = train_feats.merge(train_scores, on='id', how='left')
x              = data.drop(['id', 'score'], axis=1)
y              = data['score'].values
print(f'Number of features: {len(x.columns)}')


test_ids = test_feats['id'].values
testin_x = test_feats.drop(['id'], axis=1)

print(train_feats.shape, test_feats.shape)
'''


print('< Learning and Evaluation >')
lgbm_param = {'n_estimators': 1024,
         'learning_rate': 0.005,
         'metric': 'rmse',
         'random_state': 42,
         'force_col_wise': True,
         'verbosity': 0,}
lgbm_reg = LGBMRegressor(**lgbm_param)
y_pred_lgbm, avg_rmse_lgbm = evaluate(x, y, lgbm_reg, random_state=42, n_splits=5, test_x=testin_x, return_score=True)



'''
import optuna
from lightgbm import LGBMRegressor

def objective(trial):
    # Define the search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 256, 2048),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'metric': 'rmse',
        'random_state': 42,
        'force_col_wise': True,
        'verbosity': -1,
    }

    # Build model
    model = LGBMRegressor(**param)
    
    # Evaluate using your existing function
    _, avg_rmse = evaluate(x, y, model, random_state=42, n_splits=5, test_x=None, return_score=True)
    
    # Optuna minimizes the objective, so we return the RMSE directly
    return avg_rmse

# Create the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Print results
print("Best RMSE:", study.best_value)
print("Best hyperparameters:", study.best_params)

# Re-train the model with the best parameters
best_model = LGBMRegressor(**study.best_params)
y_pred_lgbm, avg_rmse_lgbm = evaluate(x, y, best_model, random_state=42, n_splits=5, test_x=testin_x, return_score=True)
'''





'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
rf_param = {'n_estimators' : 1024,
           'max_depth' : 5,
           'random_state': 42}
rf_reg = RandomForestRegressor(**rf_param)

if data.isna().sum().sum() > 0:
    data = data.dropna()
if testin_x.isna().sum().sum() > 0:
    imputer = SimpleImputer(strategy="mean", keep_empty_features = True)
    testin_x = pd.DataFrame(
        imputer.fit_transform(testin_x),
        columns=testin_x.columns
    )
x              = data.drop(['id', 'score'], axis=1)
y              = data['score'].values
y_pred_rf, avg_rmse_rf = evaluate(x, y, rf_reg, random_state=42, n_splits=10, test_x=testin_x, return_score=True)
'''


sub = pd.DataFrame({'id': test_ids, 'score': y_pred_lgbm})
sub.to_csv('submission.csv', index=False)




