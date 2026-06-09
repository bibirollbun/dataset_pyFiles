from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from concurrent.futures import ThreadPoolExecutor
from autogluon.tabular import TabularPredictor
from timeit import default_timer as timer
from nltk.tokenize import sent_tokenize
from peft import PeftModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import joblib
import torch
import gc

warnings.filterwarnings("ignore")


class CFG:
    train_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet'
    test_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet'
    sample_sub_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/sample_submission.csv'
    
    gemma_dir = "/kaggle/input/gemma-2-9b-4bit-it-unsloth/transformers/default/1/gemma-2-9b-it-4bit-unsloth_old"
    lora_dir = "/kaggle/input/wsdm-cup-gemma-2-9b-4-bit-qlora/gemma2-9b-4bit/fold-0/gemma-2-9b-it-bnb-4bit-3072-8-f0/checkpoint-2700"

    data_path = '/kaggle/input/wsdm-cup-gemma-2-9b-4-bit-qlora'

    target = 'winner'
    n_folds = 5
    seed = 42
    
    max_length = 3072
    batch_size = 4

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


test = pd.read_parquet(CFG.test_path).fillna('')


if len(test) > 10_000:
    time_limit = int(3600 * 12) 
else:
    time_limit = int(3600 * 4.75)


def tokenize(tokenizer, prompt, response_a, response_b, max_length=CFG.max_length):
    prompt = ["<prompt>: " + t for t in prompt]
    response_a = ["\n\n<response_a>: " + t for t in response_a]
    response_b = ["\n\n<response_b>: " + t for t in response_b]
    
    texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
    tokenized = tokenizer(texts, max_length=max_length, truncation=True)
    
    return tokenized['input_ids'], tokenized['attention_mask']


tokenizer = GemmaTokenizerFast.from_pretrained(CFG.gemma_dir)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"


for col in ['prompt', 'response_a', 'response_b']:
    test[col] = test[col].fillna('')
    text_list = []
    if col == "prompt":
        max_no = 512
        s_no = 255
        e_no = -256
    else:
        max_no = 3072
        s_no = 1535
        e_no = -1536
    for text in tqdm(test[col]):
        encoded = tokenizer(text, return_offsets_mapping=True)
        if len(encoded['input_ids']) > max_no:
            start_idx, end_idx = encoded['offset_mapping'][s_no]
            new_text = text[:end_idx]
            start_idx, end_idx = encoded['offset_mapping'][e_no]
            new_text = new_text + "\n(snip)\n" + text[start_idx:]
            text = new_text
        text_list.append(text)
    test[col] = text_list


data = pd.DataFrame()
data["id"] = test["id"]
data["input_ids"], data["attention_mask"] = tokenize(tokenizer, test["prompt"], test["response_a"], test["response_b"])
data["length"] = data["input_ids"].apply(len)

aug_data = pd.DataFrame()
aug_data["id"] = test["id"]
# swap response_a & response_b
aug_data['input_ids'], aug_data['attention_mask'] = tokenize(tokenizer, test["prompt"], test["response_b"], test["response_a"])
aug_data["length"] = aug_data["input_ids"].apply(len)


model_0 = Gemma2ForSequenceClassification.from_pretrained(
    CFG.gemma_dir,
    device_map=torch.device("cuda:0"),
    use_cache=False,
)

model_1 = Gemma2ForSequenceClassification.from_pretrained(
    CFG.gemma_dir,
    device_map=torch.device("cuda:1"),
    use_cache=False,
)


model_0 = PeftModel.from_pretrained(model_0, CFG.lora_dir)
model_1 = PeftModel.from_pretrained(model_1, CFG.lora_dir)


model_0.eval()
model_1.eval()


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size, max_length=CFG.max_length):
    winners = []
    
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        proba = outputs.logits.softmax(-1).cpu()
        
        winners.extend(proba[:, 1].tolist())
    
    df['winner'] = winners
    
    return df


global_timer = timer()


data['index'] = np.arange(len(data), dtype=np.int32)
data = data.sort_values("length", ascending=False)


data_dict = {}
data_dict[0] = data[data["length"] > 1024].reset_index(drop=True)
data_dict[1] = data[data["length"] <= 1024].reset_index(drop=True)


result_df = []
for i, batch_size in enumerate([CFG.batch_size, CFG.batch_size]):
    if len(data_dict[i]) == 0:
        continue
        
    sub_1 = data_dict[i].iloc[0::2].copy()
    sub_2 = data_dict[i].iloc[1::2].copy()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(
            inference, 
            (sub_1, sub_2), 
            (model_0, model_1), 
            (torch.device("cuda:0"), torch.device("cuda:1")), 
            (batch_size, batch_size)
        )
        
    result_df.append(pd.concat(list(results), axis=0))


result_df = pd.concat(result_df).sort_values('index').reset_index(drop=True)


aug_data['index'] = np.arange(len(aug_data), dtype=np.int32)
aug_data = aug_data.sort_values("length", ascending=False)


CONFIDENCE_THRESHOLD = 0.2
not_confident_mask = abs(result_df['winner'] - 0.5) < CONFIDENCE_THRESHOLD

aug_data = aug_data[aug_data['index'].isin(result_df[not_confident_mask]['index'])]


aug_data_dict = {}
aug_data_dict[0] = aug_data[aug_data["length"] > 1024].reset_index(drop=True)
aug_data_dict[1] = aug_data[aug_data["length"] <= 1024].reset_index(drop=True)


aug_result_df = []
for i, batch_size in enumerate([CFG.batch_size, CFG.batch_size]):
    if len(aug_data_dict[i]) == 0:
        continue

    if timer() - global_timer > (time_limit - 1800):
        break
        
    sub_1 = aug_data_dict[i].iloc[0::2].copy()
    sub_2 = aug_data_dict[i].iloc[1::2].copy()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(
            inference, 
            (sub_1, sub_2), 
            (model_0, model_1), 
            (torch.device("cuda:0"), torch.device("cuda:1")), 
            (batch_size, batch_size)
        )
        
    aug_result_df.append(pd.concat(list(results), axis=0))


time_taken = timer() - global_timer

print(f'time for 10_000 (max  4.5 hr): {10_000/3*time_taken/60/60:4.1f}')
print(f'time for 25_000 (max 12.0 hr): {25_000/3*time_taken/60/60:4.1f}')


if len(aug_result_df) > 0:
    aug_result_df = pd.concat(aug_result_df).sort_values('index').reset_index(drop=True)
    aug_result_df["winner"] = 1 - aug_result_df['winner']
    
    result_df = result_df.merge(
        aug_result_df[['index', 'winner']], 
        on='index', 
        how='left', 
        suffixes=('', '_aug')
    )
    
    mask = result_df['winner_aug'].notna()
    result_df.loc[mask, 'winner'] = (result_df.loc[mask, 'winner'] + result_df.loc[mask, 'winner_aug']) / 2
    
    result_df = result_df.drop('winner_aug', axis=1)


gemma_test_pred_probs = result_df["winner"].values


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
            
            tfidf_vectorizer = joblib.load(f'{CFG.data_path}/vectorizers/{column}_{params["analyzer"]}_tfidf_vectorizer.pkl')
            tfidf_matrix = tfidf_vectorizer.transform(df[column].fillna(''))
            tfidf_dense = tfidf_matrix.toarray()
            
            feature_names = tfidf_vectorizer.get_feature_names_out()
            for i in range(len(feature_names)):
                df[f'{column}_{params["analyzer"]}_tfidf_{i}'] = tfidf_dense[:, i]
            
            del tfidf_vectorizer, tfidf_matrix, tfidf_dense
            
            gc.collect()
            
    return df


test = pd.read_parquet(CFG.test_path)
test = test.drop(columns=['scored'])


test = add_features(test, is_train=False)
test = add_tfidf_features(test, is_train=False)
test['tta_oof'] = gemma_test_pred_probs


test = test.drop(columns=['id', 'prompt', 'response_a', 'response_b'])


predictor = TabularPredictor.load('/kaggle/input/wsdm-cup-gemma-2-9b-4-bit-qlora/autogluon')


ag_test_preds = predictor.predict(test).values


sub = result_df[['id', 'winner']].copy()
sub['winner'] = np.where(ag_test_preds == 0, 'model_a', 'model_b')
sub.to_csv('submission.csv', index=False)
sub.head()

