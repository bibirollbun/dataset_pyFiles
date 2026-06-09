# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     # for filename in filenames:
#     #     print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





! mkdir -p /tmp/src
! mkdir -p  /kaggle/working/logs


! uv pip install --system --no-index --find-links='/kaggle/input/latest-mdc-whls/whls' 'pymupdf'


import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

# We will use PyMuPDF (fitz) to extract text from PDFs.
import fitz
import polars as pl

# -----------------------------------------------------------------------------
# 1. Configuration and Setup
# -----------------------------------------------------------------------------

# Set this to True for training mode, False for test mode
IS_TRAINING_MODE = True

# Define the directory for logging and other output
LOG_FILE_PATH = "logs/project.log"
LOG_DIR = Path(LOG_FILE_PATH).parent

# Create the log directory if it doesn't exist
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(f"Error creating log directory: {e}")

# Get a basic logger
def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        fh = logging.FileHandler(LOG_FILE_PATH)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.propagate = False
    return logger

l = get_logger()
DOI_LINK = 'https://doi.org/'

# Path to the main competition directory
COMP_DIR = Path('/kaggle/input/make-data-count-finding-data-references/')

# Dynamically set PDF directory based on mode
if IS_TRAINING_MODE:
    PDF_DIR = COMP_DIR / 'train' / 'PDF'
else:
    PDF_DIR = COMP_DIR / 'test' / 'PDF'

COMPILED_PATTERNS = {
    'ref_header_patterns': [re.compile(r'\b(R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S|BIBLIOGRAPHY|LITERATURE CITED|WORKS CITED|CITED WORKS|ACKNOWLEDGEMENTS)\b[:\s]*', re.IGNORECASE)],    
    'citation_pattern': re.compile(r'^\s*(\[\d+\]|\(\d+\)|\d+\.|\d+\)|\d+(?=\s|$))\s*'),
    'first_citation_patterns': [
        re.compile(r'^\s*\[1\]\s*'),
        re.compile(r'^\s*\(1\)\s*'),
        re.compile(r'^\s*1\.\s*'),
        re.compile(r'^\s*1\)\s*'),
        re.compile(r'^\s*1(?=\s|$)'),
    ],
}

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------

def string_normalization(name: str) -> pl.Expr:
    """Normalizes and cleans text for better matching."""
    return pl.col(name).str.normalize("NFKC").str.replace_all(r"[^\p{Ascii}]", '')

def get_df_from_pdfs(pdf_dir: Path) -> pl.DataFrame:
    """
    Reads all PDF files from a directory, extracts their text, and
    returns a Polars DataFrame.
    """
    records = []
    pdf_files = list(pdf_dir.glob('*.pdf'))
    if not pdf_files:
        l.warning(f"No PDF files found in directory: {pdf_dir}")
        return pl.DataFrame({"article_id": [], "text": []})

    for pdf_file in pdf_files:
        try:
            doc = fitz.open(pdf_file)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            article_id = pdf_file.stem
            records.append({'article_id': article_id, 'text': text})
        except Exception as e:
            l.error(f"Failed to process {pdf_file}: {e}")

    return pl.DataFrame(records).with_columns(
        string_normalization('text').alias('text')
    )

def is_doi_link_expr(col_name: str) -> pl.Expr:
    """Returns a Polars expression that checks for a valid DOI link."""
    return pl.col(col_name).str.starts_with(DOI_LINK) & pl.col(col_name).str.contains('10.')

def score(df, gt, on, tag='all'):
    """Calculates F1, TP, FP, and FN scores."""
    hits = gt.join(df, on=on)
    tp = hits.height
    fp = df.height - tp
    fn = gt.height - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    return f"{tag} - f1: {f1:.4f} [{tp}/{fp}/{fn}]"

def evaluate(df, on=['article_id', 'dataset_id']):
    """Performs the full evaluation of extracted citations."""
    gt = pl.read_csv(COMP_DIR/'train_labels.csv').filter(pl.col('type')!='Missing')
    return (
        score(df, gt, on),
        score(df.filter(is_doi_link_expr('dataset_id')), gt.filter(is_doi_link_expr('dataset_id')), on, 'doi'),
        score(df.filter(~is_doi_link_expr('dataset_id')), gt.filter(~is_doi_link_expr('dataset_id')), on, 'acc'),
    )

def find_last_reference_header(text: str, header_patterns: list[re.Pattern]) -> Optional[int]:
    """Finds the index of the last reference header in a given text."""
    last_match_idx = None
    for pattern in header_patterns:
        matches = list(pattern.finditer(text))
        if matches:
            last_match_idx = matches[-1].start()
    return last_match_idx

def find_last_first_citation(text: str) -> Optional[int]:
    """Finds the line number of the last line that looks like a first citation."""
    lines = text.splitlines()
    last_match_line = None
    for line_num, line in enumerate(lines):
        line = line.strip()
        for pattern in COMPILED_PATTERNS['first_citation_patterns']:
            if pattern.match(line):
                next_lines = lines[line_num:line_num+3]
                if any(COMPILED_PATTERNS['citation_pattern'].match(l.strip()) for l in next_lines[1:]):
                    last_match_line = line_num
                break
    return last_match_line

def find_reference_start(text: str) -> Optional[int]:
    """Finds the starting line of the references section based on number patterns."""
    lines = text.splitlines()
    last_first_citation = find_last_first_citation(text)
    if last_first_citation is not None:
        return last_first_citation
    start_search_idx = int(len(lines) * 0.5)
    for i in range(start_search_idx, len(lines)):
        line = lines[i].strip()
        if COMPILED_PATTERNS['citation_pattern'].match(line):
            next_lines = lines[i:i+3]
            if sum(1 for l in next_lines if COMPILED_PATTERNS['citation_pattern'].match(l.strip())) >= 2:
                for j in range(i, max(-1, i-10), -1):
                    if not COMPILED_PATTERNS['citation_pattern'].match(lines[j].strip()):
                        return j + 1
                return max(0, i-10)
    return None

def split_text_and_references(text: str) -> Tuple[str, str]:
    """Splits a document into the main body and the references section."""
    header_idx = find_last_reference_header(text, COMPILED_PATTERNS['ref_header_patterns'])
    if header_idx is not None:
        header_idx2 = find_last_reference_header(text[:header_idx].strip(), COMPILED_PATTERNS['ref_header_patterns'])
        if header_idx2 is not None:
            header_idx3 = find_last_reference_header(text[:header_idx2].strip(), COMPILED_PATTERNS['ref_header_patterns'])
            if header_idx3 is not None:
                return text[:header_idx3].strip(), text[header_idx3:].strip()
            return text[:header_idx2].strip(), text[header_idx2:].strip()
        return text[:header_idx].strip(), text[header_idx:].strip()
    ref_start_line = find_reference_start(text)
    if ref_start_line is not None:
        lines = text.splitlines()
        body = '\n'.join(lines[:ref_start_line])
        refs = '\n'.join(lines[ref_start_line:])
        return body.strip(), refs.strip()
    return text.strip(), ''

def get_splits(df: pl.DataFrame) -> pl.DataFrame:
    """Applies the split_text_and_references function to a DataFrame of documents."""
    bodies, refs = [], []
    for raw_text in df['text']:
        main, ref = split_text_and_references(raw_text)
        bodies.append(main)
        refs.append(ref)
    return df.with_columns(pl.Series('body', bodies), pl.Series('ref', refs))

def tidy_extraction_new(df) -> pl.DataFrame:
    """
    Extracts citations from the 'body' and 'ref' columns of the DataFrame,
    cleans them, and combines them.
    """
    bad_ids = [f'{DOI_LINK}{e}' for e in ['10.5061/dryad', '10.5281/zenodo', '10.6073/pasta']]

    doi_df = (
        df.with_columns(pl.col('body').str.extract_all(r'10\s*\.\s*\d{4,9}\s*/\s*\S+').alias('match'))
          .explode('match')
          .drop_nulls('match')
          .with_columns(
              pl.col('match').str.replace_all(r'\s+', '')
                             .str.replace(r'[^A-Za-z0-9]+$', '')
                             .str.to_lowercase()
                             .alias('dataset_id')
          )
          .group_by('article_id', 'dataset_id')
          .agg('match')
          .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
    )

    REGEX_IDS = (
        r"(?i)\b(?:"
        r"CHEMBL\d+|"
        r"E-GEOD-\d+|E-PROT-\d+|E-MTAB-\d+|E-MEXP-\d+|EMPIAR-\d+|"
        r"ENSBTAG\d+|ENSOARG\d+|"
        r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
        r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}|"
        r"NC_\d{6}\.\d{1}|NM_\d{9}|"
        r"PRJNA\d+|PRJEB\d+|PRJDB\d+|PXD\d+|SAMN\d+|"
        r"GSE\d+|GSM\d+|GPL\d+|"
        r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
        r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
        r"(?:SR[PX]|STH|ERR|DRR|DRX|DRP|ERP|ERX)\d+|"
        r"CVCL_[A-Z0-9]{4}"
        r")"
    )
    
    acc_df = (
        df.with_columns(
            pl.col('text').str.extract_all(REGEX_IDS).alias('match')
        )
        .explode('match')
        .drop_nulls('match')
        .with_columns(
            pl.col('match').str.replace_all(r'\s+', '')
                           .str.replace(r'[^A-Za-z0-9]+$', '')
                           .str.replace(r'(?i)^PDB', '')
                           .alias('dataset_id')
        )
        .group_by('article_id', 'dataset_id')
        .agg('match')
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('dryad.'))
              .then(f'{DOI_LINK}10.5061/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('pasta/'))
              .then(f'{DOI_LINK}10.6073/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
    )

    df = pl.concat([doi_df, acc_df])

    df = (
        df.unique(['article_id', 'dataset_id'])
          .filter(~pl.col('article_id').str.replace('_','/').str.contains(pl.col('dataset_id').str.split(DOI_LINK).list.last().str.escape_regex()))
          .filter(~pl.col('dataset_id').str.contains(pl.col('article_id').str.replace('_','/').str.escape_regex()))
          .filter(~pl.col('dataset_id').str.contains('figshare', literal=True))
          .filter(~pl.col('dataset_id').is_in(bad_ids))
          .filter(
              pl.when(is_doi_link_expr('dataset_id') &
                      (pl.col('dataset_id').str.split('/').list.last().str.len_chars() < 5))
               .then(False)
               .otherwise(True)
          )
          .with_columns(pl.col('match').list.unique())
    )
    return df

def assume_type(df: pl.DataFrame) -> pl.DataFrame:
    """Assigns 'Primary' or 'Secondary' type based on citation ID."""
    return (
        df.with_columns(pl.when(is_doi_link_expr('dataset_id').or_(pl.col('dataset_id').str.starts_with('SAMN'))).then(pl.lit('Primary')).otherwise(pl.lit('Secondary')).alias('type'))
    )

def get_context_window(text: str, substring: str, window: int = 100) -> str:
    """
    Extracts a context window around a substring in the given text.
    """
    try:
        idx = text.find(substring)
        if idx == -1:
            raise ValueError
        start = max(idx - window, 0)
        end = min(idx + len(substring) + window, len(text))
        return text[start:end]
    except ValueError:
        l.warning(f"Substring '{substring}' not found in text.")
        return ""
    except Exception as e:
        l.error(f"Error getting context window for '{substring}': {e}")
        return ""

def get_window_df(text_df, ids_df):
    """
    Generates a DataFrame with context windows for each extracted citation.
    """
    df = ids_df.join(text_df, on='article_id')
    windows = []
    for text, match_ids in df.select('text', 'match').rows():
        windows.append(get_context_window(text, match_ids[0]))
    return df.with_columns(pl.Series('window', windows)).select('article_id', 'dataset_id', 'window')

# -----------------------------------------------------------------------------
# 3. Main Function to Run the New Method
# -----------------------------------------------------------------------------

def main():
    """
    Main function to orchestrate the PDF reading and citation parsing
    with the new logic.
    """
    l.info("--- Starting Citation Extraction Process ---")
    l.info(f"Running in {'Training' if IS_TRAINING_MODE else 'Test'} mode.")
    
    text_df = get_df_from_pdfs(PDF_DIR)
    
    # Split the text into main body and references
    df_with_splits = get_splits(text_df)
    
    # Extract citations
    df_extracted = tidy_extraction_new(df_with_splits)
    
    # Get context windows for each citation
    df_with_windows = get_window_df(text_df, df_extracted)
    
    # Assume types and evaluate
    df_labeled = assume_type(df_with_windows)
    
    if IS_TRAINING_MODE:
        # Load labels and join for saving a labeled dataset
        train_labels_df = pl.read_csv(COMP_DIR/'train_labels.csv')
        df_with_labels = df_with_windows.join(train_labels_df, on=['article_id', 'dataset_id'], how='left')
        df_with_labels.write_parquet('/kaggle/working/labeled_citations.parquet')
        l.info("Labeled dataset saved to labeled_citations.parquet")
        
        # Evaluate the results and log them
        results = evaluate(df_labeled)
        l.info("Evaluation Scores:")
        for r in results: l.info(r)
    else:
        # Save a submission file for test mode
        df_labeled.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
        l.info("Submission file saved to submission.csv")

if __name__ == '__main__':
    main()



train_set_df=pl.read_parquet("/kaggle/working/labeled_citations.parquet")


train_set_df


IS_TRAINING_MODE = False


# Dynamically set PDF directory based on mode
if IS_TRAINING_MODE:
    PDF_DIR = COMP_DIR / 'train' / 'PDF'
else:
    PDF_DIR = COMP_DIR / 'test' / 'PDF'


def main():
    """
    Main function to orchestrate the PDF reading and citation parsing
    with the new logic.
    """
    l.info("--- Starting Citation Extraction Process ---")
    l.info(f"Running in {'Training' if IS_TRAINING_MODE else 'Test'} mode.")
    
    text_df = get_df_from_pdfs(PDF_DIR)
    
    # Split the text into main body and references
    df_with_splits = get_splits(text_df)
    
    # Extract citations
    df_extracted = tidy_extraction_new(df_with_splits)
    
    # Get context windows for each citation
    df_with_windows = get_window_df(text_df, df_extracted)
    
    # Assume types and evaluate
    df_labeled = assume_type(df_with_windows)
    
    if IS_TRAINING_MODE:
        # Load labels and join for saving a labeled dataset
        train_labels_df = pl.read_csv(COMP_DIR/'train_labels.csv')
        df_with_labels = df_with_windows.join(train_labels_df, on=['article_id', 'dataset_id'], how='left')
        df_with_labels.write_parquet('/kaggle/working/labeled_citations.parquet')
        l.info("Labeled dataset saved to labeled_citations.parquet")
        
        # Evaluate the results and log them
        results = evaluate(df_labeled)
        l.info("Evaluation Scores:")
        for r in results: l.info(r)
    else:
        # Save a submission file for test mode
        df_with_windows.write_parquet('/kaggle/working/unlabeled_train.parquet')
        df_labeled.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
        l.info("Submission file saved to submission.csv")

if __name__ == '__main__':
    main()


test_set_df=pl.read_parquet("/kaggle/working/unlabeled_train.parquet")


test_set_df


submission_df=pd.read_csv("/kaggle/working/submission.csv")


submission_df


# !pip install -U keras-nlp
# !pip install -U keras-hub
# !pip install -U keras


import os
import keras
import random
import warnings
import keras_nlp
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt


import keras
import keras_hub


warnings.filterwarnings("ignore")


os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
os.environ["JAX_PLATFORMS"] = ""


def build_tf_dataset_for_inference(df):
    """
    Builds a TensorFlow dataset for inference from a pandas DataFrame.
    """
    AUTO = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = True
    
    # We will process all records for inference, so we remove the slicing
    
    # Convert the DataFrame into a dictionary with only the "prompts" key
    # No need for "responses" as we are not training
    dataset_dict_list = []
    for i in range(len(df)):
        dataset_dict = dict()
        dataset_dict["prompts"] = template.format(window=df.iloc[i, 0])
        dataset_dict_list.append(dataset_dict)
    
    # Create the TensorFlow dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: (item for item in dataset_dict_list),
        output_signature={
            "prompts": tf.TensorSpec(shape=(), dtype=tf.string),
        }
    )
    
    # No need for shuffle or cache during inference
    dataset = dataset.with_options(options).batch(1).prefetch(AUTO)
    
    return dataset


gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_270m")
gemma_lm.generate("Keras is a", max_length=30)

# Generate with batched prompts.
gemma_lm.generate(["Keras is a", "I want to say"], max_length=30)


gemma_lm.preprocessor.sequence_length = 512

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])


# gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.backbone.load_lora_weights("/kaggle/input/mdc-lora-gemma/gemma_finetune.lora.h5")
# gemma_lm.compile(sampler=keras_nlp.samplers.TopKSampler(k=3, temperature=0.7))

# Model Compilation
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


def generate_inference(model, df):
    """
    Generates model inference for each 'window' in the DataFrame.

    Args:
        model: The trained model to use for inference (e.g., gemma_lm).
        df: The DataFrame containing the 'window' data for inference.

    Returns:
        A list of generated responses, one for each 'window' in the DataFrame.
    """
    
    responses = []
    # Loop through each row of the DataFrame
    for _, row in df.iterrows():
        # Get the 'window' text from the current row
        window_text = row['window']
        
        # Create the prompt using the provided template
        # The 'template' object must be defined outside this function
        prompt = template.format(window=window_text)
        
        # Define the maximum length for the generated response
        max_length = 300
        
        # Generate the response using the model
        response = model.generate(prompt, max_length=max_length)
        
        # Extract and clean the generated response
        # This assumes the model's output format is consistent
        try:
            cleaned_response = response.split("Type: \n\n")[-1].strip()
        except IndexError:
            cleaned_response = response.strip() # Fallback if split fails
        
        # Append the cleaned response to the list
        responses.append(cleaned_response)
        
    return responses

# Example usage:
# Assuming 'test_set_df' from the image is the DataFrame to use for inference
# and 'gemma_lm' is your pre-loaded model.
# template = "Your predefined template string with a '{window}' placeholder."
# inference_results = generate_inference(gemma_lm, test_set_df)
# print(inference_results)


template = """System: 

You are an expert at analyzing research data citations in academic papers.

Classify the data as:
A) Primary: if the data was generated specifically for this study
B) Secondary: if the data was reused or derived from prior work  
C) None: if the DOI is in references, doesn't refer to research data, or is unrelated


text: 
{window}


Type: \n\n
"""



test_set_pd=test_set_df.to_pandas()


inference_results = generate_inference(gemma_lm, test_set_pd)


print(inference_results)


len("test_set_pd")


test_set_pd['type']=inference_results


test_set_pd


# test_set_pd['calculated']=submission_df.type


df_to_save = test_set_pd[['article_id', 'dataset_id', 'type']].copy()

# Add a 'row_id' column that corresponds to the row index
# df_to_save['row_id'] = df_to_save.index
df_to_save = df_to_save[df_to_save['type'] != 'None']
df_to_save.reset_index(drop=True)
df_to_save['row_id'] = df_to_save.index
columns_order = ['row_id', 'article_id', 'dataset_id', 'type']

# Reorder the DataFrame columns
df_to_save = df_to_save[columns_order]
# Write the DataFrame to a CSV, overwriting if the file exists
df_to_save.to_csv('/kaggle/working/submission.csv', index=False)


df_to_save.columns


df_to_save


# Assuming test_set_pd is your original DataFrame
# 1. Select the columns and create a copy
df_to_save = test_set_pd[['article_id', 'dataset_id', 'type']].copy()

# 2. Filter out rows where the 'type' is "None"
df_to_save = df_to_save[df_to_save['type'] != 'None']

# 3. Reset the index of the filtered DataFrame.
# The 'drop=True' argument removes the old index.
# We must reassign the result back to df_to_save.
df_to_save = df_to_save.reset_index(drop=True)

# 4. Add the new 'row_id' column from the new sequential index
df_to_save['row_id'] = df_to_save.index

# 5. Define the new column order with 'row_id' first
columns_order = ['row_id', 'article_id', 'dataset_id', 'type']

# 6. Reorder the columns and save to CSV
df_to_save[columns_order].to_csv('/kaggle/working/submission.csv', index=False)


df_to_save.tail(20)


submission_df.tail(20)


test_set_pd['regex_tag']=submission_df.type


test_set_pd.head(20)




