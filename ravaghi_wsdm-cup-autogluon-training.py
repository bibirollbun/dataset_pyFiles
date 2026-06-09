!pip install -q autogluon.tabular ray==2.10.0


from sklearn.model_selection import StratifiedKFold
from autogluon.tabular import TabularPredictor
import pandas as pd
import warnings
import joblib
import shutil

warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet'
    test_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet'
    sample_sub_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/sample_submission.csv'

    data_path = '/kaggle/input/wsdm-cup-gemma-2-9b-4-bit-qlora'

    target = 'winner'
    n_folds = 5
    seed = 42

    char_vectorizer_params = {
        'analyzer': "char",
        "lowercase": False,
        "max_df": 0.605,
        "max_features": 331,
        "min_df": 0.075,
        "ngram_range": (1, 3),
        "strip_accents": "unicode"
    }

    word_vectorizer_params = {
        "analyzer": "word",
        "lowercase": True,
        "max_df": 0.985,
        "max_features": 769,
        "min_df": 0.01,
        "ngram_range": (1, 2),
        "strip_accents": "unicode"
    }


def reduce_mem_usage(dataframe):
    print('--- Reducing memory usage')
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        if col_type.name in ['category', 'object']:
            continue

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('------ Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('------ Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('------ Decreased memory usage by {:.1f}%'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe


def add_word_features(df, column):
    # Basic word counts
    df[f'{column}_word_count'] = df[column].apply(lambda x: len(x.split()))
    df[f'{column}_unique_word_count'] = df[column].apply(lambda x: len(set(x.lower().split())))
    
    # Word length statistics
    def safe_word_stats(text):
        words = str(text).split()
        if not words:
            return 0, 0  # avg_length, max_length for empty text
        word_lengths = [len(word) for word in words]
        return np.mean(word_lengths), max(word_lengths)
    
    word_stats = df[column].apply(safe_word_stats)
    df[f'{column}_avg_word_length'] = word_stats.apply(lambda x: x[0])
    df[f'{column}_max_word_length'] = word_stats.apply(lambda x: x[1])
    
    # Lexical diversity (unique words / total words)
    df[f'{column}_lexical_diversity'] = df.apply(
        lambda x: x[f'{column}_unique_word_count'] / x[f'{column}_word_count'] 
        if x[f'{column}_word_count'] > 0 else 0, axis=1
    )
    
    # Count specific word types
    df[f'{column}_uppercase_word_count'] = df[column].apply(lambda x: sum(1 for word in x.split() if word.isupper()))
    df[f'{column}_title_case_word_count'] = df[column].apply(lambda x: sum(1 for word in x.split() if word.istitle()))
    
    return df

def add_char_features(df, column):
    # Basic character counts
    df[f'{column}_char_count'] = df[column].str.len()
    df[f'{column}_letter_count'] = df[column].apply(lambda x: sum(c.isalpha() for c in x))
    df[f'{column}_digit_count'] = df[column].apply(lambda x: sum(c.isdigit() for c in x))
    df[f'{column}_whitespace_count'] = df[column].apply(lambda x: sum(c.isspace() for c in x))
    
    # Punctuation counts
    df[f'{column}_punctuation_count'] = df[column].apply(lambda x: sum(c in '.,!?;:' for c in x))
    df[f'{column}_special_char_count'] = df[column].apply(lambda x: sum(not (c.isalnum() or c.isspace()) for c in x))
    
    # Character ratios with safe division
    df[f'{column}_uppercase_ratio'] = df[column].apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1))
    df[f'{column}_lowercase_ratio'] = df[column].apply(lambda x: sum(c.islower() for c in x) / max(len(x), 1))
    
    return df

def add_sentence_features(df, column):
    # Sentence counts
    df[f'{column}_sentence_count'] = df[column].apply(lambda x: len(sent_tokenize(x)))
    
    # Average sentence length with safe division
    df[f'{column}_avg_sentence_length'] = df[column].apply(
        lambda x: np.mean([len(sent.split()) for sent in sent_tokenize(x)])
        if len(sent_tokenize(x)) > 0 else 0
    )
    
    # Sentence length variation with safe handling
    df[f'{column}_sentence_length_std'] = df[column].apply(
        lambda x: np.std([len(sent.split()) for sent in sent_tokenize(x)]) 
        if len(sent_tokenize(x)) > 1 else 0
    )
    
    # Question and exclamation counts
    df[f'{column}_question_count'] = df[column].str.count('\?')
    df[f'{column}_exclamation_count'] = df[column].str.count('!')
    
    return df

def add_stats_features(df, column):
    # Readability metrics (simplified Flesch Reading Ease)
    def calculate_readability(text):
        sentences = sent_tokenize(text)
        words = text.split()
        if not words or not sentences:
            return 0
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = sum(count_syllables(word) for word in words) / len(words)
        return 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        return count
    
    df[f'{column}_readability_score'] = df[column].apply(calculate_readability)
    
    # Text complexity features with safe division
    df[f'{column}_avg_word_per_sentence'] = df.apply(
        lambda x: x[f'{column}_word_count'] / x[f'{column}_sentence_count']
        if x[f'{column}_sentence_count'] > 0 else 0, axis=1
    )
    
    df[f'{column}_char_per_word'] = df.apply(
        lambda x: x[f'{column}_char_count'] / x[f'{column}_word_count']
        if x[f'{column}_word_count'] > 0 else 0, axis=1
    )
    
    return df

def get_text_similarity(text1, text2):
    text1, text2 = str(text1), str(text2)
    
    chars1, chars2 = set(text1.lower()), set(text2.lower())
    char_similarity = len(chars1 & chars2) / max(len(chars1 | chars2), 1)
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    word_similarity = len(words1 & words2) / max(len(words1 | words2), 1)
    
    length_ratio = len(text1) / max(len(text2), 1)
    
    return char_similarity, word_similarity, length_ratio

def add_features(df, is_train):
    if is_train:
        features = joblib.load(f'{CFG.data_path}/features/train_features.pkl')
        df = df.merge(features, on='id', how='left')
        return df
    
    for column in ['prompt', 'response_a', 'response_b']:
        df = add_word_features(df, column)
        df = add_char_features(df, column)
        df = add_sentence_features(df, column)
        df = add_stats_features(df, column)
        
    # Add text similarity features between prompt and response_a/response_b
    similarities_a = df.apply(lambda row: get_text_similarity(row['prompt'], row['response_a']), axis=1)
    similarities_b = df.apply(lambda row: get_text_similarity(row['prompt'], row['response_b']), axis=1)
    df['prompt_response_a_char_sim'], df['prompt_response_a_word_sim'], df['prompt_response_a_length_ratio'] = zip(*similarities_a)
    df['prompt_response_b_char_sim'], df['prompt_response_b_word_sim'], df['prompt_response_b_length_ratio'] = zip(*similarities_b)
        
    # Add comparative features between response_a and response_b with safe division
    for feature in df.columns:
        if feature.startswith('response_a_'):
            corresponding_b = feature.replace('response_a_', 'response_b_')
            if corresponding_b in df.columns:
                df[f'diff_{feature.replace("response_a_", "")}'] = df[feature] - df[corresponding_b]
                df[f'ratio_{feature.replace("response_a_", "")}'] = df.apply(
                    lambda x: x[feature] / x[corresponding_b] 
                    if x[corresponding_b] != 0 else 0, axis=1
                )
    
    return df


def add_tfidf_features(df, is_train):
    if is_train:
        tfidf_features = joblib.load(f'{CFG.data_path}/features/train_tfidf_features.pkl')
        df = df.merge(tfidf_features, on='id', how='left')
        return df

    for column in ['prompt', 'response_a', 'response_b']:
        for params in [CFG.char_vectorizer_params, CFG.word_vectorizer_params]:
            
            tfidf_vectorizer = joblib.load(f'{column}_{params["analyzer"]}_tfidf_vectorizer.pkl')
            tfidf_matrix = tfidf_vectorizer.transform(df[column].fillna(''))
            tfidf_dense = tfidf_matrix.toarray()
            
            feature_names = tfidf_vectorizer.get_feature_names_out()
            for i in range(len(feature_names)):
                df[f'{column}_{params["analyzer"]}_tfidf_{i}'] = tfidf_dense[:, i]
            
            del tfidf_vectorizer, tfidf_matrix, tfidf_dense
            
            gc.collect()
            
    return df


train = pd.read_parquet(CFG.train_path)

train[CFG.target] = train[CFG.target].map({"model_a": 0, "model_b": 1})

train = train.drop(columns=['model_a', 'model_b', 'language'])


train = add_features(train, is_train=True)
train = add_tfidf_features(train, is_train=True)


train['tta_oof'] = joblib.load(f'{CFG.data_path}/features/tta_oof_pred_probs_acc_0.683437.pkl')


train = train.drop(columns=['id', 'prompt', 'response_a', 'response_b'])


kf = StratifiedKFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True)
split = kf.split(train, train[CFG.target])
for i, (train_index, val_index) in enumerate(split):
    train.loc[val_index, 'fold'] = i


predictor = TabularPredictor(
    path="/AutoGluonModels",
    problem_type='binary',
    eval_metric='accuracy',
    label=CFG.target,
    groups='fold',
    verbosity=2
)


predictor.fit(
    train_data=train,
    time_limit=300,
    presets='best_quality',
    excluded_model_types=['KNN'],
    save_space=True
)


predictor.leaderboard(silent=True).style.background_gradient(subset=['score_val'], cmap='RdYlGn')


shutil.make_archive(
    "/kaggle/working/autogluon", 
    "zip", 
    "/AutoGluonModels"
)

