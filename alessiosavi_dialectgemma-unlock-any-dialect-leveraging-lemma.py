# %pip install --upgrade --no-cache-dir pip wheel setuptools black isort jupyterlab-code-formatter jupyterthemes jupyterlab_darkside_theme nvitop
!pip install unidecode langdetect spacy 
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth 'transformers==4.47.1'
%pip install --upgrade nltk

!python -m spacy download en_core_web_md
!python -m spacy download fr_core_news_md
!python -m spacy download de_core_news_md
!python -m spacy download it_core_news_md
!python -m spacy download pl_core_news_md
!python -m spacy download pt_core_news_md
!python -m spacy download es_core_news_md
print('Remember to restart the kernel!!')


import kagglehub
dialect_to_ita_eng_path = kagglehub.dataset_download('alessiosavi/dialect-to-ita-eng')


import gc
import gzip
import json
import os
import pickle
import random
import re
import string
from collections import defaultdict
from copy import copy
from glob import glob
from pprint import pprint
from typing import *

# langdetect is used to detect the most similar language of the dialect, by this way we can enhance the translation process
import langdetect
import matplotlib.pyplot as plt
import nltk.translate.bleu_score as bleu
import nltk.translate.gleu_score as gleu
import numpy as np
import pandas as pd
import requests
import seaborn as sns

# spacy is used to perform the lemmatization
import spacy
import torch
from datasets import Dataset, load_dataset
from IPython.display import Markdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import TrainingArguments, pipeline
from trl import SFTTrainer

# Some dialect has phonetic symbols, with `unidecode` library we aim to clean these phonetics symbols
from unidecode import unidecode
from unsloth import FastLanguageModel, load_correct_tokenizer

# Set SpaCy to prioritize GPU usage for faster NLP operations if available
spacy.prefer_gpu()
tqdm.pandas()
# Set a fixed seed for reproducibility across experiments
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Define sequence length constraints for text processing tasks
max_seq_length = 1024
min_seq_length = 0

os.environ["HF_TOKEN"] = "hf_YOUR_HF_TOKEN"
# Define a function to collect garbage and free up GPU memory if available
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Cache the spacy language models
nlps: Dict[str, spacy.Language] = {}

# Compile a regex pattern to split text into sentences by detecting punctuation followed by whitespace or newline
batch_regex = re.compile(r"(?<=[.!?\n])\s+")
# Create a set of all standard punctuation characters for efficient lookup during text processing
punctuation_set = set(string.punctuation)
# We want to build a text dataset, sometimes wikipedia (especially for dialect) tends to have a list of dates.
# This regexp aims to find (and count) the numbers of 'date/numbers' in the text. We want to remove these examples.
number_regex = re.compile(r"\d*\.\d+|\d+", re.MULTILINE)

def _collect():
    x = 0
    for i in range(3):
        x += gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return x


# Due to my memory limitation, I batch each wikipedia page in a set of fixed number of words.
#  The last sentence of the first batch is the first sentence of the next batch.
#  Example: 'This is an example sentence. This sentence has to be separated. This, because it's too much to be handled by my GPU!'
#  Output:
#   BATCH 1: 'This is an example sentence. This sentence has to be separated.'
#   BATCH 2: 'This sentence has to be separated. This, because it's too much to be handled by my GPU!'
# The idea here is to have "continuity" of the context.
def batch_text_with_overlap(text: str, max_words=256) -> List[str]:
    """Batches text into chunks with a maximum number of words, allowing overlap between batches.

    Args:
        text: The input text string.
        max_words: The maximum number of words per batch.

    Returns:
        :return:A list of string batches.
    """
    # Split the text into sentences using regex
    sentences = batch_regex.split(text)

    batches = []
    current_batch = []
    current_word_count = 0

    for i, sentence in enumerate(sentences):
        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count <= max_words:
            # Add sentence to current batch
            current_batch.append(sentence)
            current_word_count += sentence_word_count
        else:
            # The current batch is ready, add it to batches
            if current_batch:
                batches.append(" ".join(current_batch).strip())
                # Start a new batch, including the last sentence of the previous batch
                current_batch = [sentence]
                current_word_count = sentence_word_count
                # Overlap: add the last sentence of the previous batch to the new batch
                if i > 0:
                    current_batch.insert(0, sentences[i - 1])
                    current_word_count += len(sentences[i - 1].split())
            else:
                # Handle cases where a single sentence is longer than max_words
                batches.append(sentence.strip())
                current_batch = []
                current_word_count = 0

    # Add the last batch if it's not empty
    if current_batch:
        batches.append(" ".join(current_batch).strip())
    return batches


def find_outliers(data, n=2):
    """
    Identifies outliers in a given Pandas Series based on the Z-score.
    This method leverages Z-score to clean "dirty" entries.

       Args:
           :param data: A Pandas Series containing numerical data.
           :param n: The number of standard deviations to use as a threshold for outlier detection.

       Returns:
          :return: A boolean Pandas Series where True indicates an outlier and False indicates a non-outlier.
    """
    mean = data.mean()
    std = data.std()
    data_zscore = (data - mean) / std

    # Identify outliers (e.g., |Z| > 3)
    mask = abs(data_zscore) > n
    return mask


def count_punctuation_and_numbers(text: str):
    """
    Counts the number of punctuation marks, numbers, and newline characters in a given string.
    This method is used in combination with `find_outliers` to clean "dirty" entries.

       Args:
           :param text: The input string.

       Returns:
           :return: A dictionary containing the counts of punctuation marks, numbers, and newline characters, or a dictionary with zero counts if the input string is empty.
    """
    if not text or len(text) == 0:
        return {"puncts": 0, "numbers": 0, "new_lines": 0}

    punct_count = sum(1 for char in text if char in punctuation_set)
    number_count = len(number_regex.findall(text))
    new_lines = text.count("\n")

    return {"puncts": punct_count, "numbers": number_count, "new_lines": new_lines}


def translate(text, target_language="Italian", model="meta-llama-3.1-70B-bnb-4bit"):
    """
    This method translates a given text into its corresponding language model.
    I've used a local cluster, using `unsloth/Meta-Llama-3.1-70B-bnb-4bit`.
    The model servers exposes a JSON over HTTP API that follows the [OpenAI spec](https://platform.openai.com/docs/api-reference/chat).
    This code is omitted cause is only infra-related stuff
    Args:
        :param text: The text to be translated.
        :param target_language: The language name of the target language.
        :param model: The LLM to use.
    Return:
        :return: Translated text.
    """
    url = "YOUR_CLUSTER_URL"
    prompt = f"You are a translator that don't add any sentences that are not written in the input text. Don't add also any comments, just translate the text into plain {target_language}."
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    resp = requests.post(
        url,
        stream=False,
        timeout=(30, 300),
        json={
            "model": "local",
            "messages": messages,
            "stream": False,
        },
        headers={"x-service-id": model},
    )
    resp.raise_for_status()
    return resp


def translate_local(
    text, target_language="it", model_name="Helsinki-NLP/opus-mt-fr-en"
):
    """
    Translates text from one language to another using a specified LLM.
    NOTE: This method is not tested, is saved as reference in order to play/debug the solution without having a local GPU cluster

    Args:
        :param text: The text to be translated.
        :param target_language: The target language code (e.g., "en" for English, "fr" for French, "es" for Spanish).
        :param model_name: The name of the translation model from the Hugging Face Model Hub.

    Returns:
        :return: The translated text, or None if an error occurs.
    """
    try:
        translator = pipeline("translation", model=model_name)
        translated_text = translator(text, target_lang=target_language)[0][
            "translation_text"
        ]
        return translated_text
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None


def clean_dataset(dataset: Dict[str, List[str]], n: int = 2) -> Dict[str, List[str]]:
    """
    Cleans a dataset by removing entries that are outliers based on punctuation and number counts.

    Args:
        :param: dataset: A dictionary where keys are identifiers and values are lists of strings.
        :param: n: The number of standard deviations to use as a threshold for outlier detection.

    Returns:
        :return: A new dictionary containing only the entries that are not considered outliers.
    """
    puncts = {}
    for k, v in dataset.items():
        puncts[k] = count_punctuation_and_numbers("\n".join(v))
    puncts = pd.DataFrame(puncts).T.reset_index()
    mask1 = find_outliers(puncts["puncts"], n=n)
    mask2 = find_outliers(puncts["numbers"], n=n)
    mask = mask1 | mask2
    keys = set(puncts[mask]["index"].to_list())
    return {k: v for k, v in dataset.items() if k not in keys}


def remove_content_by_target(d: Dict[str, List[str]], targets: List[str], lower=True):
    """Removes content from a dictionary based on the presence of specified target strings.

    Args:
        :param: d: A dictionary where keys are identifiers and values are lists of strings.
        :param: targets: A list of target strings to search for within the content.
        :param: lower: A boolean indicating whether to convert the content to lowercase before searching for targets.

    Returns:
        :return: A new dictionary containing only the content that does not contain any of the target strings.
    """
    data = defaultdict(list)
    for k, vs in d.items():
        for v in vs:
            _v = copy(v)
            if lower:
                _v = _v.lower()
            if not any(target in _v for target in targets):
                data[k].append(v)
    return data


def find_topic_by_target(d: Dict[str, List[str]], targets: List[str], lower=True):
    """Return the entries from a dictionary based on the presence of specified target strings.

    Args:
        :param: d: A dictionary where keys are identifiers and values are lists of strings.
        :param: targets: A list of target strings to search for within the content.
        :param: lower: A boolean indicating whether to convert the content to lowercase before searching for targets.

    Returns:
        :return: The keys of the input dictionary that contains any of the target strings.
    """
    keys = []
    for k, vs in d.items():
        text = "\n".join(vs)
        if lower:
            text = text.lower()
        if any(target in text for target in targets):
            keys.append(k)
    return keys


def remove_similar_documents(df, text_column, similarity_threshold=0.2):
    """Removes similar documents from a DataFrame based on TF-IDF and cosine similarity.

    Args:
        :param: df: The input Pandas DataFrame.
        :param: text_column: The name of the column containing the text data.
        :param: similarity_threshold: The threshold above which two documents are considered similar.

    Returns:
        :return: A new Pandas DataFrame with similar documents removed. Returns the original DataFrame if it's empty.
    """
    if df.empty:
        return df

    # 1. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(df[text_column])

    # 2. Cosine Similarity Calculation
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 3. Identify Documents to Remove
    to_remove = set()
    num_docs = len(df)
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            if similarity_matrix[i, j] > similarity_threshold:
                # Prioritize keeping the earlier document encountered in the DataFrame
                to_remove.add(j)

    # 4. Create a new DataFrame
    indices_to_keep = list(set(range(num_docs)) - to_remove)
    filtered_df = df.iloc[list(indices_to_keep)].copy()
    duplicated_df = df.iloc[list(to_remove)]

    return filtered_df, duplicated_df



def load_raw_data(language: str, keys: Union[List[str], None] = None):
    """Loads raw Wikipedia data for a given language, applying filtering and preprocessing.

    Args:
        :param: language: The language code (e.g., "en", "it").
        :param: keys: An optional list of titles to filter the dataset. If None, all titles are used.

    Returns:
        :return: A dictionary where keys are titles and values are lists of text batches (with overlap).
    """
    # Load dataset
    df = load_dataset("wikimedia/wikipedia", f"20231101.{language}", num_proc=6)[
        "train"
    ].to_pandas()
    if keys is not None:
        df = df[df["title"].isin(keys)]
    df["size"] = df["text"].apply(lambda x: len(x.split()))
    # Impose some limit on the dataset in order to reduce the global size of text data
    df = df[(df["size"] >= 128) & (df["size"] <= 1500)]
    df["text"] = df["text"].str.strip()
    mask1 = df["text"].str.contains("{{", regex=False) | df["text"].str.contains("}}", regex=False)
    mask2 = df["text"].str.contains("[[", regex=False) | df["text"].str.contains("]]", regex=False)
    mask3 = df["text"].str.contains("{|", regex=False) | df["text"].str.contains("|}", regex=False)
    df = df[~(mask1 | mask2 | mask3)].sample(frac=1, random_state=seed)
    ds_overlap = defaultdict(list)
    df[["title", "text"]].apply(
        lambda x: ds_overlap[x["title"]].extend(batch_text_with_overlap(x["text"])),
        axis=1,
    )
    assert len(ds_overlap.keys()) == len(df)

    ds_overlap = clean_dataset(ds_overlap, 2)
    print(f"Len after removing puncts/numbers: {len(ds_overlap)}")
    return ds_overlap

# Load the raw dataset from Wikipedia, `ia` is the dataset for `interlingua`
ds_overlap = load_raw_data(language="ia")
# Example of input data
for k,v in ds_overlap.items():
    print(k,v)
    break


def detect_language(text: str) -> str | None:
    """
    Args:
        :param text: Input text language
    Returns:
        :return: 2-digit of the language code (e.g., "en" for English)
    """
    try:
        lang = langdetect.detect(text)
        return lang
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None


def preprocess_text(text: str, nlp: spacy.Language):
    """
    Preprocesses text using a given SpaCy NLP model for lemmatization.

    Args:
        :param: text: The input text string.
        :param: nlp: The SpaCy NLP model to use for processing.

    Returns:
        :return: The lemmatized text string. Returns the original text if an error occurs during preprocessing.
    """
    try:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc]).strip()
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text  # Return the original text if preprocessing fails


def lemmatize_data(text: str, force_language: str | None = None) -> str:
    """
     Lemmatizes the input text using SpaCy, handling language detection and model loading.

    Args:
        :param: text: The input text string.
        :param: force_language: An optional language code to force lemmatization in a specific language. If None, the language is detected automatically.

    Returns:
        :return: The lemmatized text string (or an exception if `langdetext` is not able to understand the language code).
    """
    decoded_text = unidecode(text)
    language: str = (
        detect_language(decoded_text) if not force_language else force_language
    )
    if language in nlps:
        nlp = nlps[language]
    else:
        if language.startswith("en"):
            nlp = spacy.load(f"{language}_core_web_md")
        else:
            nlp = spacy.load(f"{language}_core_news_md")
        nlps[language] = nlp
    return preprocess_text(decoded_text, nlp)


# Here is an example of the lemmatization process, leveraging the straightforward pipeline above
for vs in ds_overlap.values():
    for v in vs:
        print("Before\n", v, "\n", "---" * 30)
        print("After\n", lemmatize_data(v), "\n", "---" * 30)
    break


# # Dirty hack to avoid losing the already translated dataset.
# if "ds_overlap_translated" not in globals():
#     ds_overlap_translated = defaultdict(list)
#     # Load the latest file saved, instead of restart from 0
#     files = glob(f"{dialect_to_ita_eng_path}/ds_overlap_translated_ia_lemma_it*")
#     if len(files) > 0:
#         files.sort(key=lambda x: os.path.getmtime(x))
#         ds_overlap_translated = defaultdict(list, json.load(gzip.open(files[-1])))


# print(f"Len of already translated data: {len(ds_overlap_translated)}")
# n = 0
# for k, vs in tqdm(ds_overlap.items(), total=len(ds_overlap), position=0):
#     # Avoid translating the already managed topics
#     if k in ds_overlap_translated and len(vs) == len(ds_overlap_translated[k]):
#         continue
#     for idx, v in enumerate(tqdm(vs, leave=False, position=1, desc=k)):
#         if (idx + 1) <= len(ds_overlap_translated[k]):
#             continue
#         try:
#             lemmatized_v = lemmatize_data(v)
#             res = translate(lemmatized_v)
#             response = (
#                 res.json()["choices"][0]["message"]["content"]
#                 # This cleaning is related to the LLAMA model used.
#                 .removeprefix("<|start_header_id|>assistant<|end_header_id|>").strip()
#             )
#             ds_overlap_translated[k].append(response)
#         except Exception as e:
#             print(f"Missing idx: {idx} of {k} | Error: {e}")
#             del ds_overlap_translated[k]
#             break
#     n += 1
#     if n % 5 == 0:
#         # Save the dataset every 5 wikipedia webpages translated
#         fname = f"data/ds_overlap_translated_ia_lemma_it_{n}_{len(ds_overlap_translated)}.json"
#         with gzip.open(fname, "wt") as zipfile:
#             json.dump(ds_overlap_translated, zipfile)
# # Save the last version of the dataset
# json.dump(ds_overlap_translated, open("ds_overlap_translated_ia_lemma_it.json", "wt"))


# Load the dataset
# Here we have the template configuration in order to play with different (already managed languages)

# dataset_config = {
#     "starting_language": "sicilian",
#     "wikilanguage": 'scn',

#     "italian": f"{dialect_to_ita_eng_path}/ds_overlap_translated_scn_lemma_it.json",
#     "english": f"{dialect_to_ita_eng_path}/ds_overlap_translated_scn_lemma_en.json"
# }

# dataset_config = {
#     "starting_language": "napolitan",
#     "wikilanguage": 'nap',

#     "italian": f"{dialect_to_ita_eng_path}/ds_overlap_translated_nap_lemma_it.json",
#     "english": f"{dialect_to_ita_eng_path}/ds_overlap_translated_nap_lemma_en.json"
# }


# dataset_config = {
#     "starting_language": "venetian",
#     "wikilanguage": 'vec',

#     "italian": f"{dialect_to_ita_eng_path}/ds_overlap_translated_vec_lemma_it.json",
#     "english": f"{dialect_to_ita_eng_path}/ds_overlap_translated_vec_lemma_en.json"
# }

dataset_config = {
    "starting_language": "interlingua",
    "wikilanguage": 'ia',

    "italian": f"{dialect_to_ita_eng_path}/ds_overlap_translated_ia_lemma_it.json",
    "english": f"{dialect_to_ita_eng_path}/ds_overlap_translated_ia_lemma_en.json"
}


it_fname = dataset_config["italian"]
en_fname = dataset_config["english"]
starting_language = dataset_config["starting_language"]
wikilanguage = dataset_config["wikilanguage"]
# NOTE: On your local env, the dataset will be gzipped (remember to change from plain `open` to `gzip.open`)
ds_overlap_it: Dict[str, List[str]] = json.load(open(it_fname))
ds_overlap_en: Dict[str, List[str]] = json.load(open(en_fname))


def aggregate_dataset(
    raw_base: Dict[str, List[str]], raw_translated: Dict[str, List[str]]
) -> Dict[str, List[Dict[str, List[str]]]]:
    """Aggregates two raw datasets (base and translated) into a single structured dataset.

    Args:
        :param: raw_base: A dictionary where keys are topics and values are lists of strings (base data).
        :param: raw_translated: A dictionary where keys are topics and values are lists of strings (translated data).

    Returns:
        :return: A dictionary where keys are topics and values are lists of dictionaries.
            Each inner dictionary contains the "original" and "translated" strings for a given data point.
    """
    ds = {}
    for topic in raw_translated:
        base_data = raw_base[topic]
        translated_data = raw_translated[topic]
        d = []
        for v1, v2 in zip(base_data, translated_data):
            d.append({"original": v1, "translated": v2})
        ds[topic] = d
    return ds


def filter_dataset(language, ds_translated, n=2, ds_base=None):
    """Filters and cleans a translated dataset based on a base dataset.

    Args:
        :param: language: The language of the dataset.
        :param: ds_translate: A dictionary representing the translated dataset (topic -> list of strings).
        :param: n: A threshold for outlier removal.
        :param: ds_base: An optional dictionary representing the base dataset (topic -> list of strings). If None, it's loaded using load_raw_data.

    Returns:
        :return: A tuple containing the filtered base dataset, the filtered translated dataset, and the aggregated dataset.
    """
    if ds_base is None:
        ds_base = load_raw_data(language, list(ds_translated.keys()))
    # Be sure that we are dealing only with the same topics.
    common_keys = set(ds_translated.keys()).intersection(ds_base.keys())

    # Due to the "batched" dataset (a big chunk of text is divided in multiple parts)
    #  we need to be sure that, for each topic, we have the same number of entries.
    common_keys = [k for k in common_keys if len(ds_base[k]) == len(ds_translated[k])]

    # Filtering both dataset and removing noisy topics
    raw_base = {k: ds_base[k] for k in common_keys}
    # Don't trust the dataset. Removing entries for which we have wikipedia-related noisy text
    raw_base = remove_content_by_target(raw_base, ["{|", "|}", "[[", "]]", "{{", "}}"])

    # Remove topics that contain a high number of numbers/punctuation.
    # By this way, we can have a more clean dataset
    raw_base = clean_dataset(raw_base, n)
    raw_translated = {k: ds_translated[k] for k in raw_base}
    raw_translated = clean_dataset(raw_translated, n)
    raw_base = {k: ds_base[k] for k in raw_translated}

    # Verify that we are dealing with the same topics
    assert raw_base.keys() == raw_translated.keys()
    # Verify that for each topic we have the same number of "separated batches"
    assert (
        sum(
            [
                abs(len(ia) - len(it))
                for ia, it in zip(raw_translated.values(), raw_base.values())
            ]
        )
        == 0
    )

    print(
        f"[{language}] Raw entries: {len(ds_translated)} | Clean entries: {len(raw_base)}"
    )
    return raw_base, raw_translated, aggregate_dataset(raw_base, raw_translated)


raw_base_en, raw_translated_en, ds_aggregated_en = filter_dataset(
    wikilanguage, ds_overlap_en
)
raw_base_it, raw_translated_it, ds_aggregated_it = filter_dataset(
    wikilanguage, ds_overlap_it
)
DATASET_TYPE_TRANSLATE = "translate"
DATASET_TYPE_DESCRIBE = "describe"
DATASET_TYPES = [DATASET_TYPE_TRANSLATE, DATASET_TYPE_DESCRIBE]
# We let the possibility to the final user to train against multiple languages
all_datasets = [
    {
        "starting_language": starting_language,
        "translated_language": "english",
        "dataset": ds_aggregated_en,
        "type": DATASET_TYPES,
    },
    {
        "starting_language": starting_language,
        "translated_language": "italian",
        "dataset": ds_aggregated_it,
        "type": DATASET_TYPES,
    },
]


instruction_translate = "Provide a punctual translation of the following text from {} to {}, without any comments, explanations or interpretations."
instruction_describe = "Describe the following topic in the following language: {}."

alpaca_prompt_template = """<start_of_turn>user
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

<end_of_turn>
<start_of_turn>model
### Response:
{}
<end_of_turn>"""
_collect()
display(Markdown('### Example of "translating task"'))
print(
    alpaca_prompt_template.format(
        instruction_translate.format("Italian", "English"), "INPUT", "OUTPUT"
    )
)
print("---" * 30)
display(Markdown('### Example of "describing task"'))
print(
    alpaca_prompt_template.format(
        instruction_describe.format("Italian"), "INPUT", "OUTPUT"
    )
)


# Initialize an empty DataFrame to store all processed data.
ds = pd.DataFrame()

for dataset_conf in tqdm(all_datasets):
    # This loop iterates through all dataset configurations, extracting the starting and translated languages and the raw dataset.
    starting_language = dataset_conf["starting_language"]
    translated_language = dataset_conf["translated_language"]
    dataset = dataset_conf["dataset"]

    # Creating a dataframe from the given "raw" dataset.
    # The dataset will contain ["topic", "original_content", "translated_content"]
    raw_text = []
    _ = [
        [raw_text.append([k, v["original"], v["translated"]]) for v in vs]
        for k, vs in dataset.items()
    ]
    tdf = pd.DataFrame(raw_text)
    tdf.columns = ["topic", "original_content", "translated_content"]

    for dataset_type in dataset_conf["type"]:
        if dataset_type == DATASET_TYPE_DESCRIBE:
            _tdf = tdf.copy()
            # Using the original content (in native language of dataset)
            _tdf["prompt"] = _tdf.apply(
                lambda x: alpaca_prompt_template.format(
                    instruction_describe.format(starting_language),
                    x["topic"],
                    x["original_content"],
                ),
                axis=1,
            )
            _tdf["starting_language"] = translated_language
            _tdf["translated_language"] = starting_language
            _tdf["task_type"] = DATASET_TYPE_DESCRIBE
            ds = pd.concat([ds, _tdf])
        elif dataset_type == DATASET_TYPE_TRANSLATE:
            # I've simplified the code avoiding fancy method (this is memory inefficient). Also, I want the code to be "hackable" to be modified by the final user purpose.
            # Here I store the "original-to-translated" translation
            tdf1 = tdf.copy()
            # Here I store the "translated-to-original" translation
            tdf2 = tdf.copy()

            # Translate from ORIGINAL_LANGUAGE --> TRANSLATED_LANGUAGE
            tdf1["prompt"] = tdf1.apply(
                lambda x: alpaca_prompt_template.format(
                    instruction_translate.format(
                        starting_language, translated_language
                    ),
                    x["original_content"],
                    x["translated_content"],
                ),
                axis=1,
            )
            tdf1["starting_language"] = starting_language
            tdf1["translated_language"] = translated_language
            tdf1["task_type"] = DATASET_TYPE_TRANSLATE
            ds = pd.concat([ds, tdf1])

            # Translate from TRANSLATED_LANGUAGE --> ORIGINAL_LANGUAGE
            tdf2["prompt"] = tdf2.apply(
                lambda x: alpaca_prompt_template.format(
                    instruction_translate.format(
                        translated_language, starting_language
                    ),
                    x["translated_content"],
                    x["original_content"],
                ),
                axis=1,
            )
            tdf2["starting_language"] = translated_language
            tdf2["translated_language"] = starting_language
            original_content = tdf2.pop("original_content")
            translated_content = tdf2.pop("translated_content")
            tdf2["original_content"] = translated_content
            tdf2["translated_content"] = original_content
            tdf2["task_type"] = DATASET_TYPE_TRANSLATE
            ds = pd.concat([ds, tdf2])

# For the "describe" dataset, we are interested only in describing the topic in interlingua.
# By this way we remove the "similar prompt" in order to remove the duplicated topic between italian and english
df = ds.drop_duplicates("prompt").reset_index(drop=True).copy()
del ds
_collect()
model_id = "google/gemma-2-2b-it"

tokenizer = load_correct_tokenizer(model_id, max_seq_length)
EOS_TOKEN = tokenizer.eos_token


df.groupby(['starting_language', 'translated_language', 'task_type']).size()


df["prompt"] = df["prompt"] + EOS_TOKEN
df["new_lines_ratio"] = df["prompt"].apply(lambda x: x.count("\n") / len(x.split()))
df["n_tokens"] = df["prompt"].progress_apply(lambda x: len(tokenizer.encode(x)))
df["len_ratio"] = df.apply(
    lambda x: len(x["original_content"].split()) / len(x["translated_content"].split()),
    axis=1,
)
mask_ratio = ~find_outliers(df["len_ratio"], n=1)
mask_len = (df["n_tokens"] >= min_seq_length) & (df["n_tokens"] <= max_seq_length)
mask_lines = ~find_outliers(df["new_lines_ratio"])
mask = mask_len & mask_lines & mask_ratio
df = df[mask].reset_index(drop=True)
df["len"] = df["prompt"].apply(lambda x: len(x.split()))

print(
    f"Removing {(~mask).sum()} entries that are out of [{max_seq_length} - {min_seq_length}] tokens. {len(df)} entries remaining."
)
sns.histplot(df["len"], kde=True).set(title="Number of words")
plt.show()
sns.histplot(df["n_tokens"], kde=True).set(title="Number of tokens")
plt.show()
sns.histplot(df["n_tokens"] / df["len"], kde=True).set(title="Tokens/Word ratio")
plt.show()
sns.histplot(df["len_ratio"], kde=True).set(title="Len ratio")
plt.show()


# Finding similar documents that share the same languages and task_type (translate, describe)
filtered_df = pd.DataFrame()
duplicated_df = pd.DataFrame()
for group_name, _df in df.groupby(
    ["starting_language", "translated_language", "task_type"]
):
    new_df, dups = remove_similar_documents(
        _df, "original_content", similarity_threshold=0.2
    )
    print(
        f'Removed {len(_df) - len(new_df)} entries for {dict(zip(["starting_language", "translated_language", "task_type"], list(group_name)))}'
    )
    filtered_df = pd.concat([filtered_df, new_df], ignore_index=True)
    duplicated_df = pd.concat([duplicated_df, dups], ignore_index=True)
del df
df = filtered_df.reset_index(drop=True).copy()
del filtered_df
gc.collect()
print(f'Test size ratio:{len(duplicated_df) / len(df) * 100}')
display(df.sample(2))
df[
    ["starting_language", "translated_language", "task_type"]
].value_counts().sort_index()


for group, tdf in df.groupby(["starting_language", "translated_language", "task_type"]):
    print("=" * 30, group, "=" * 30)
    print(tdf["prompt"].sample(1).item())

ds_train = df.sample(frac=1, random_state=seed)
ds_test = duplicated_df.sample(frac=1, random_state=seed)
train_topics = set(ds_train["topic"].unique())
test_topics = set(ds_test["topic"].unique())
print(
    f"Train examples: {len(ds_train)} | Test Examples: {len(ds_test)} | Common topics: {len(train_topics.intersection(test_topics))} | Only train topics: {len(train_topics - test_topics)} | Only test topics: {len(test_topics - train_topics)}"
)
dataset_train = Dataset.from_pandas(ds_train.convert_dtypes())

# NOTE:
# The `describe` dataset will not be used during evaluation.
# However, we save for reference if we want to evaluate how the network is able to generate dialogues in the dialect language.
dataset_test = Dataset.from_pandas(ds_test.convert_dtypes())
dataset_train.save_to_disk(
    f"datasets/{starting_language}_{'_'.join(DATASET_TYPES)}_train.hf"
)
dataset_test.save_to_disk(
    f"datasets/{starting_language}_{'_'.join(DATASET_TYPES)}_test.hf"
)
for x in dataset_train:
    break
x


# Another "dummy" method for create the test set, is to use the latest batch for each topic. By this way, the dataset will contain some overlapped sentences.
# I've choosen to don't use this method but can be useful for play a bit with the dataset creation
# def create_dataset(dataset):
#     ds_train = []
#     ds_test = []
#     for topic, data in tqdm(
#         dataset.groupby(
#             ["starting_language", "translated_language", "task_type", "topic"]
#         )
#     ):
#         if len(data) == 1:
#             ds_train.extend(data.to_dict(orient="records"))
#         else:
#             ds_train.extend(data.iloc[:-1].to_dict(orient="records"))
#             ds_test.extend(data.iloc[-1:].to_dict(orient="records"))
#     return ds_train, ds_test


# df.reset_index(drop=True, inplace=True)
# df["id"] = df.index
# ds_train, ds_test = create_dataset(df)
# ds_train = pd.DataFrame(ds_train).sample(frac=1, random_state=seed)
# ds_test = pd.DataFrame(ds_test).sample(frac=1, random_state=seed)


def load_base_model(model_id, max_seq_length, device="sequential"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map=device,
    )
    return model, tokenizer


# training_config = "configs/config_3.3-candidate-2epochs-ia.json"
model_config = {
    "dataset_name": f"{dataset_config['starting_language']}_translate_describe",
    "model_id": "google/gemma-2-2b-it",
    "modules_to_save": [
        "lm_head"
    ],
    "r": 64,
    "lora_alpha": 16,
    "epochs": 2,
    "dropout":0.3,
    "learning_rate": 0.001,
    "lr_scheduler_type": "cosine"
}
# model_config = json.load(open('YourConfFileHere.json'))
pprint(model_config)

modules_to_save = model_config.get("modules_to_save")
dataset_name = model_config.get("dataset_name")
epochs = model_config.get("epochs")
model_id = model_config.get("model_id")
learning_rate = model_config.get("learning_rate")
lr_scheduler_type = model_config.get("lr_scheduler_type")
r = model_config.get("r")
dropout = model_config.get("dropout")
lora_alpha = model_config.get("lora_alpha")


model, tokenizer = load_base_model(model_id, max_seq_length)

dataset_train = Dataset.load_from_disk(f"datasets/{dataset_name}_train.hf")
dataset_test = Dataset.load_from_disk(f"datasets/{dataset_name}_test.hf")
# Removing the "describe" dataset from the test set.
ds = dataset_test.to_pandas()
ds.pop("__index_level_0__")
ds = ds[ds["task_type"] == "translate"]
dataset_test = Dataset.from_pandas(ds)
del ds
_collect()


model = FastLanguageModel.get_peft_model(
    model,
    r=r,
    lora_alpha=lora_alpha,
    lora_dropout=dropout,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=seed,
    use_rslora=False,
    loftq_config=None,
    modules_to_save=modules_to_save,
)

do_eval = True  #'9b' not in model_id
batch_size = 2  # if do_eval else 1
grad_acc_batch_size = 8  # if do_eval else 16
training_args = TrainingArguments(
    # auto_find_batch_size=not do_eval,
    do_train=True,
    do_eval=do_eval,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1 if do_eval else None,
    gradient_accumulation_steps=grad_acc_batch_size,
    eval_accumulation_steps=grad_acc_batch_size if do_eval else None,
    eval_strategy="steps" if do_eval else "no",
    # Perform eval at each 10% of training dataset
    eval_steps=0.1 if do_eval else 0,
    logging_steps=10,
    warmup_steps=20,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type=lr_scheduler_type,
    seed=seed,
    output_dir="outputs",
    report_to=["tensorboard"],
)

if not do_eval:
    dataset_test = None
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

dl = trainer.get_train_dataloader()
# Ensure that the tokenizer respects the max length
for batch in dl:
    assert len(batch["input_ids"][0]) <= max_seq_length
    print(tokenizer.decode(batch["input_ids"][0]))
    del batch
    del dl
    break


_collect()
trainer_stats = trainer.train()

new_model = f"models/{os.path.basename(model_id)}_unsloth_{dataset_name}_{os.path.basename(os.path.splitext(os.path.basename(training_config))[0])}"
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)
print(new_model)


def sentence_metric(translated: str, original: str, metric=gleu) -> float:
    """Calculates a sentence-level metric score between a translated and an original sentence.

    Args:
        :param: translated: The translated sentence (string).
        :param: original: The original sentence (string).
        :param: metric: The metric function to use {gleu, bleu}.

    Returns:
        :return: The sentence-level metric score (float).
    """
    # Get the name of the sentence-level scoring method from the metric object
    method_name = [name for name in metric.__dict__.keys() if "sentence" in name][0]
    score_method = getattr(metric, method_name)
    return score_method([translated.split()], original.split())


def corpus_metric(translated: List[str], original: List[str], metric=gleu) -> float:
    """Calculates a corpus-level metric score between a list of translated sentences and a list of original sentences.

    Args:
        :param: translated: A list of translated sentences (strings).
        :param: original: A list of original sentences (strings).
        :param: metric: The metric function to use {gleu, bleu}.

    Returns:
        :return: The corpus-level metric score (float).

    Raises:
        AssertionError: If the lengths of the translated and original lists are not equal.
    """
    assert len(translated) == len(original)
    translated = [[sentence.split()] for sentence in translated]
    original = [sentence.split() for sentence in original]

    # Get the name of the corpus-level scoring method from the metric object.
    method_name = [name for name in metric.__dict__.keys() if "corpus" in name][0]
    score_method = getattr(metric, method_name)
    return score_method(translated, original)


def calculate_metrics(references: List[str], hypothesis: List[str], metric=gleu):
    """Calculates sentence-level and corpus-level metrics between a list of reference sentences and a list of hypothesis (translated) sentences.

    Args:
        :param: references: A list of reference sentences (strings).
        :param: hypothesis: A list of hypothesis (translated) sentences (strings).
        :param: metric: The metric function to use {bleu, gleu}.

    Returns:
        :return: A tuple containing the mean sentence-level metric score (float) and the corpus-level metric score (float).
    """
    _results: List[Tuple[str, str, float]] = []
    for test_example, model_response in zip(references, hypothesis):
        m = sentence_metric(model_response, test_example, metric)
        _results.append((model_response, test_example, m))
    return np.mean([v[2] for v in _results]).item(), corpus_metric(
        [v[0] for v in _results], [v[1] for v in _results]
    )

def calculate_metrics_from_dataset(inference_datasets):
    for language in inference_datasets:
        ds_response = inference_datasets[language]
        detokenized_responses = []
        for i in range(len(ds_response["response"])):
            _input = ds_response["inputs"][i]
            answer = ds_response["response"][i]
            _res = tokenizer.decode(
                answer[0][len(_input[0]) :], skip_special_tokens=False
            ).strip()
            detokenized_responses.append(extract_response(_res))
        ds_response["answer"] = detokenized_responses
        import numpy as np
        print(f'language: {language} -> {calculate_metrics(ds_response["answer"], ds_response["outputs"])}')

hyp = "she read the book because she was interested in world history"
ref_a = "she read the book because she was interested in world history"
ref_b = "she was interested in world history because she read the book"
sentence_metric(ref_b, hyp), corpus_metric([ref_b, ref_b], [hyp, hyp]), sentence_metric(
    ref_b, hyp, bleu
), corpus_metric([ref_b, ref_b], [hyp, hyp], bleu)


# Using the already fine-tuned model:
new_model = '/kaggle/input/gemma-2-2b-it_unsloth_interlingua_translate/transformers/v0/1'
inference_config = {
    "dataset": f"datasets/{starting_language}_{'_'.join(DATASET_TYPES)}_test.hf",
    "model": new_model,
}
# The tokenizer is not loaded, this because we can fine-tune the non-instructed version of Gemma
model, _ = load_base_model(
    inference_config["model"],
    max_seq_length,
)
# No matter the model used, we use the same dataset-format (with standard chat template)
tokenizer = load_correct_tokenizer("google/gemma-2-2b-it")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "<eos>"
MODEL_START_TOKEN = "<start_of_turn>model"
FastLanguageModel.for_inference(model)
_collect()


def tokenize_for_inference(entries: List[str], tokenizer=tokenizer, to_gpu=False):
    tokenized = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": entry}], return_tensors="pt"
        )
        for entry in entries
    ]
    if to_gpu:
        tokenized = [entry.to("cuda") for entry in tokenized]
    return tokenized


def extract_response(model_response: str):
    n = model_response.find(MODEL_START_TOKEN)
    response = (
        model_response[n:]
        .replace("### Response:\n", "")
        .removeprefix(MODEL_START_TOKEN)
        .removesuffix("<end_of_turn>")
        .strip()
    )
    return response


def create_inference_dataset(original_content, translated_content, ids, to_gpu=True):
    return (
        tokenize_for_inference(original_content, to_gpu=to_gpu),
        translated_content,
        ids,
    )


df_test = Dataset.load_from_disk(inference_config["dataset"]).to_pandas()
# Using only the `translate-related` dataset
df_test = df_test[df_test["task_type"] == "translate"]
# pprint(df_test.sample(1).to_dict(orient="records"))
# Create an id based on the dataset columns
df_test["hash"] = df_test.apply(
    lambda x: hash(
        x["topic"]
        + "|"
        + x["original_content"]
        + "|"
        + x["translated_content"]
        + "|"
        + x["starting_language"]
        + "|"
        + x["translated_language"]
        + "|"
        + x["task_type"]
        + "|"
    ),
    axis=1,
)

# For the inference phase, I use the `apply_chat_template` for compatibility, so we clean the template from the manually set special tokens.
alpaca_prompt_template_inference = (
    alpaca_prompt_template.replace("<start_of_turn>user", "")
    .replace("<end_of_turn>", "")
)
alpaca_prompt_template_inference = alpaca_prompt_template_inference[:alpaca_prompt_template_inference.find('<start_of_turn>model')].strip()


df_test["question"] = df_test.apply(
    lambda row: alpaca_prompt_template_inference.format(
        instruction_translate.format(
            row["starting_language"], row["translated_language"]
        ),
        row["original_content"],
    ).strip(),
    axis=1,
)
df_test.rename(columns={"translated_content": "answer"}, inplace=True)
df_test = df_test[["topic", "question", "answer", "starting_language", "hash"]]

test_datasets = {}
for group, _df in df_test.groupby("starting_language"):
    test_datasets[group] = _df.copy()
for k, v in test_datasets.items():
    print(f"{k} -> {len(v)}")

example_in_sentence = (
    test_datasets["italian"].sample(1, random_state=43)["question"].item()
)
example_out_sentence = (
    test_datasets["italian"].sample(1, random_state=43)["answer"].item()
)
print(example_in_sentence, "\n-------\n", example_out_sentence)


example_tokenized_sentence = tokenizer.apply_chat_template(
    [{"role": "user", "content": example_in_sentence}], return_tensors="pt"
).to("cuda")
with torch.no_grad():
    example_model_res = model.generate(example_tokenized_sentence).detach().cpu()
    # print(tokenizer.decode(example_model_res[0], skip_special_tokens=True))
detokenized_example_model_res = tokenizer.decode(
    example_model_res[0][len(example_tokenized_sentence[0]) :], skip_special_tokens=True
).strip()
print(tokenizer.batch_decode(example_tokenized_sentence)[0])
print(detokenized_example_model_res)
print("---" * 30)
print(example_out_sentence)


inference_datasets = {}
for k, ds in test_datasets.items():
    inputs, outputs, ids = create_inference_dataset(
        ds["question"].to_list(),
        ds["answer"].to_list(),
        ds["hash"].to_list(),
    )
    inference_datasets[k] = {"inputs": inputs, "outputs": outputs, "ids": ids}
inputs, outputs = (
    inference_datasets["italian"]["inputs"],
    inference_datasets["italian"]["outputs"],
)
for input_entry, output_entry in zip(inputs, outputs):
    print(tokenizer.batch_decode(input_entry, skip_special_tokens=False)[0])
    print("---" * 30)
    print(output_entry)
    print("===" * 30)
    break
_collect()


generation_config = {"max_new_tokens": max_seq_length} | {
    # "do_sample": True,
    # "early_stopping": True,
    # "temperature": 0.1,
    # "top_k": 10,
    # "top_p": 0.1,
    # "max_new_tokens": 256,
    # "repetition_penalty": 1.3,
}
# with torch.inference_mode():
with torch.no_grad():
    for language in inference_datasets:
        inference_dataset = inference_datasets[language]
        inference_dataset["response"] = []
        for _input in tqdm(inference_dataset["inputs"], desc=language):
            inference_dataset["response"].append(
                model.generate(_input, **generation_config).detach().cpu()
            )

pickle.dump(
    inference_datasets,
    open(f"inference_{os.path.basename(inference_config['dataset'])}.dict.pkl", "wb"),
)

calculate_metrics_from_dataset(inference_datasets)


dialect_to_ita_eng_path = kagglehub.dataset_download('alessiosavi/asavi-inference-gemma')


import pickle
inference_dataset_napolitan = pickle.load(open(f"{dialect_to_ita_eng_path}/inference_napolitan_translate_describe_test.hf.dict.pkl",'rb'))
calculate_metrics_from_dataset(inference_dataset_napolitan)


inference_dataset_interlingua = pickle.load(open(f"{dialect_to_ita_eng_path}/inference_interlingua_translate_describe_test.hf.dict.pkl",'rb'))
calculate_metrics_from_dataset(inference_dataset_interlingua)


italian_sample_mask = df_test['starting_language'] == 'italian'
italian_sample = df_test[italian_sample_mask].sample(2,random_state=seed)
english_sample_mask = df_test['starting_language'] == 'english'
english_sample = df_test[english_sample_mask].sample(2,random_state=seed)
interlingua_sample_mask = (df_test['starting_language'] == 'interlingua')
interlingua_sample = df_test[interlingua_sample_mask].sample(2,random_state=seed)
test_sample = pd.concat([italian_sample,english_sample,interlingua_sample])


inputs, outputs, ids = create_inference_dataset(
        test_sample["question"].to_list(),
        test_sample["answer"].to_list(),
        test_sample["hash"].to_list(),
    )
example_responses = []
generation_config = {"max_new_tokens": max_seq_length} 
for _input,_output in zip(inputs,outputs):
    example_responses.append(model.generate(_input, **generation_config).detach().cpu())


for entry,reference in zip(example_responses,outputs):
    print('--- Model Response ---')
    print(tokenizer.batch_decode(entry)[0])
    print('---'*30)
    print('--- Reference ---')
    print(reference)

