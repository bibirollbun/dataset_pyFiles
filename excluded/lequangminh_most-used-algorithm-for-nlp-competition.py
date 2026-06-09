import os
import sys
import json
import re
import subprocess
import datetime
import platform
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nbformat
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import psutil
from PIL import Image
from itertools import combinations
from collections import Counter
import networkx as nx




# Input folders
META_KAGGLE_PATH = Path("/kaggle/input/meta-kaggle")
CODE_PATH        = Path("/kaggle/input/meta-kaggle-code")

# Output folder
LOCAL_ART = Path("/kaggle/working/output/cache_parts")
LOCAL_ART.mkdir(parents=True, exist_ok=True)
ART = LOCAL_ART    # backward compat for cells 15–17
META = META_KAGGLE_PATH          # alias for brevity

#print("META_KAGGLE_PATH:", META_KAGGLE_PATH.exists())
#print("CODE_PATH       :", CODE_PATH.exists())
#print("LOCAL_ART        created at:", LOCAL_ART)


category_names = ["nlp","tabular/classical ml","time-series and signal","recommender / ranking","audio/speech","reinforcement learning/game","graph / network","medical / bio","finance","other"]


# Cell 5 – Load core tables + team counts
# -----------------------------------------

INT_NA  = "Int32"
BOOL_NA = "boolean"

# 1 Kernels
kernels = pd.read_csv(
    META / "Kernels.csv",
    usecols=["Id","AuthorUserId","CurrentKernelVersionId",
             "Medal","MedalAwardDate","TotalVotes"],
    dtype={"Id":"int32","AuthorUserId":INT_NA,
           "CurrentKernelVersionId":INT_NA,
           "Medal":"string","TotalVotes":INT_NA},
    parse_dates=["MedalAwardDate"],
    low_memory=False,
)

# 2 KernelVersions
kernel_versions = pd.read_csv(
    META / "KernelVersions.csv",
    usecols=["Id","ScriptId","AuthorUserId","CreationDate","TotalLines"],
    dtype={"Id":"int32","ScriptId":INT_NA,"AuthorUserId":INT_NA,"TotalLines":"float32"},
    parse_dates=["CreationDate"], low_memory=False,
)

# 3 Tags + CompetitionTags  (unchanged)
tags       = pd.read_csv(META / "Tags.csv",
                         usecols=["Id","Name"],
                         dtype={"Id":"int32","Name":"string"})
comp_tags  = pd.read_csv(META / "CompetitionTags.csv",
                         usecols=["CompetitionId","TagId"],
                         dtype={"CompetitionId":"int32","TagId":"int32"})

# 4 KernelVersionCompetitionSources  (column-name detection intact)
kv_raw  = pd.read_csv(META / "KernelVersionCompetitionSources.csv", low_memory=False)
cand    = ["CompetitionId","competitionId","SourceCompetitionId"]
found   = next((c for c in cand if c in kv_raw.columns), None)
kv_comp = (kv_raw
           .rename(columns={found: "CompetitionId"})
           .loc[:, ["KernelVersionId", "CompetitionId"]]
           .astype({"KernelVersionId":"int32","CompetitionId":"int32"}))



# 1. Build TagId -> TagName lookup
tag_lookup = tags.set_index("Id")["Name"].to_dict()

# 2. CompetitionTags -> human tag names
comp_tags["TagName"] = comp_tags["TagId"].map(tag_lookup)

TAG2CAT = {
    # CV (Computer Vision)
    "image classification": "cv",
    "object detection": "cv",
    "image segmentation": "cv",
    "cnn": "cv",
    "resnet": "cv",
    "unet": "cv",
    "yolo": "cv",
    "deeplab": "cv",
    "vision transformer": "cv",
    "image super resolution": "cv",
    "image-to-image": "cv",
    "image style transfer": "cv",
    "image augmentation": "cv",
    "image depth estimation": "cv",

    # NLP (Natural Language Processing)
    "nlp": "nlp",
    "text classification": "nlp",
    "translation": "nlp",
    "summarization": "nlp",
    "bert": "nlp",
    "gpt2": "nlp",
    "t5": "nlp",
    "question answering": "nlp",
    "token classification": "nlp",
    "text generation": "nlp",
    "word2vec skip-gram": "nlp",
    "sentence similarity": "nlp",
    "text-to-text generation": "nlp",
    "text segmentation": "nlp",
    "retrieval question answering": "nlp",

    # Tabular / Classical ML
    "linear regression": "tabular/classical ml",
    "logistic regression": "tabular/classical ml",
    "decision tree": "tabular/classical ml",
    "random forest": "tabular/classical ml",
    "xgboost": "tabular/classical ml",
    "lightgbm": "tabular/classical ml",
    "naive bayes": "tabular/classical ml",
    "k-means": "tabular/classical ml",
    "pca": "tabular/classical ml",
    "classification": "tabular/classical ml",
    "clustering": "tabular/classical ml",
    "regression": "tabular/classical ml",
    "tabular": "tabular/classical ml",

    # Time-Series and Signal
    "time series analysis": "time-series and signal",
    "signal processing": "time-series and signal",
    "lstm": "time-series and signal",
    "rnn": "time-series and signal",
    "sequence modeling": "time-series and signal",

    # Recommender / Ranking
    "recommender systems": "recommender / ranking",
    "retrieval/ranking": "recommender / ranking",

    # Audio / Speech
    "speech-to-text": "audio/speech",
    "text-to-speech": "audio/speech",
    "audio classification": "audio/speech",
    "whisper": "audio/speech",
    "wav2vec2": "audio/speech",
    "tacotron 2": "audio/speech",
    "audio command detection": "audio/speech",
    "speech synthesis": "audio/speech",

    # Reinforcement Learning / Game
    "reinforcement learning": "reinforcement learning/game",
    "simulations": "reinforcement learning/game",
    "games": "reinforcement learning/game",

    # Graph / Network
    "graph neural network": "graph / network",
    "graph": "graph / network",

    # Medical / Bio
    "genetics": "medical / bio",
    "healthcare": "medical / bio",
    "medical": "medical / bio",
    "diseases": "medical / bio",
    "cancer": "medical / bio",
    "neuroscience": "medical / bio",

    # Finance
    "finance": "finance",
    "banking": "finance",
    "investment": "finance",
    "crowdfunding": "finance",

    # Catch-All
    "data cleaning": "other catch-alls",
    "data visualization": "other catch-alls",
    "exploratory data analysis": "other catch-alls",
    "dimensionality reduction": "other catch-alls",
    "data analytics": "other catch-alls",
    "feature engineering": "other catch-alls",
    "deep learning": "other catch-alls",
    "automl": "other catch-alls",
    "transfer learning": "other catch-alls",
    "optimization": "other catch-alls",
    "model explainability": "other catch-alls",
    "python": "other catch-alls",
    "pytorch": "other catch-alls",
    "tensorflow": "other catch-alls",
}




def tag_to_category(tag: str | None) -> str:
    if not isinstance(tag, str):
        return "other"
    tag_l = tag.lower()
    for key, cat in TAG2CAT.items():
        if key in tag_l:
            return cat
    return "other"

comp_tags["Category"] = comp_tags["TagName"].apply(tag_to_category)
# 3. One category per CompetitionId
comp_cat = (
    comp_tags.groupby("CompetitionId")["Category"]
             .first()
             .reset_index()
)

# 4. Merge category **onto kv_comp first**
kv_comp_cat = (
    kv_comp
      .merge(comp_cat, on="CompetitionId", how="left")
      .astype({"CompetitionId": "Int32"})
)


import pandas as pd
imports_df = pd.read_csv("/kaggle/input/kv-comp-cat/kv_comp_cat.csv")
nlp = imports_df[imports_df['Category'] == 'nlp']
nlp


import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import os

# Cached version of code path function
@lru_cache(maxsize=10000)
def code_path_for_version(kv_id: int, code_path_str: str) -> str | None:
    """
    Meta Kaggle Code layout with caching for repeated lookups
    """
    CODE_PATH = Path(code_path_str)
    top = f"{kv_id // 1_000_000:04d}"
    sub = f"{(kv_id // 1_000) % 1_000:03d}"
    py_path = CODE_PATH / top / sub / f"{kv_id}.py"
    ipynb_path = CODE_PATH / top / sub / f"{kv_id}.ipynb"
    
    if py_path.exists():
        return str(py_path)
    if ipynb_path.exists():
        return str(ipynb_path)
    return None
## Extract list of NLP related algorithms
def get_nlp_algorithms():
    """
    Extract NLP algorithms from torch, keras, nltk, and transformers libraries.
    Returns a list of algorithm names with their import paths. Such as ["keras.layers.LSTM","transformers.AutoTokenizer",...]
    """
    
    nlp_algorithms = []
    
    # Process PyTorch NLP-related modules
    try:
        import torch
        
        # Focus on NLP-relevant torch.nn modules
        torch_nn = torch.nn
        nlp_layers = []
        
        for item_name in dir(torch_nn):
            if not item_name.startswith('_'):
                # NLP-specific neural network components
                nlp_keywords = [
                    'rnn', 'lstm', 'gru', 'transformer', 'embedding', 'attention',
                    'multihead', 'encoder', 'decoder', 'positional'
                ]
                
                if any(keyword in item_name.lower() for keyword in nlp_keywords):
                    nlp_algorithms.append(f"torch.nn.{item_name}")
                    
        # Add some known PyTorch NLP components
        known_torch_nlp = [
            'RNN', 'LSTM', 'GRU', 'Embedding', 'EmbeddingBag',
            'MultiheadAttention', 'TransformerEncoder', 'TransformerDecoder',
            'TransformerEncoderLayer', 'TransformerDecoderLayer'
        ]
        
        for alg in known_torch_nlp:
            if hasattr(torch_nn, alg):
                item = f"torch.nn.{alg}"
                if item not in nlp_algorithms:
                    nlp_algorithms.append(item)
                    
    except ImportError:
        print("torch not available")
    
    # Process Keras NLP-related modules
    try:
        import tensorflow as tf
        
        # Keras layers for NLP
        keras_layers = tf.keras.layers
        for item_name in dir(keras_layers):
            if not item_name.startswith('_'):
                nlp_keywords = [
                    'rnn', 'lstm', 'gru', 'embedding', 'attention', 'multihead',
                    'text', 'tokeniz', 'preprocess'
                ]
                
                if any(keyword in item_name.lower() for keyword in nlp_keywords):
                    nlp_algorithms.append(f"keras.layers.{item_name}")
        
        # Known Keras NLP layers
        known_keras_nlp = [
            'Embedding', 'LSTM', 'GRU', 'SimpleRNN', 'Bidirectional',
            'MultiHeadAttention', 'TextVectorization', 'StringLookup',
            'CategoryEncoding'
        ]
        
        for alg in known_keras_nlp:
            if hasattr(keras_layers, alg):
                item = f"keras.layers.{alg}"
                if item not in nlp_algorithms:
                    nlp_algorithms.append(item)
        
        # Keras preprocessing for text
        try:
            keras_preprocessing = tf.keras.preprocessing
            if hasattr(keras_preprocessing, 'text'):
                text_module = keras_preprocessing.text
                for item_name in dir(text_module):
                    if not item_name.startswith('_') and item_name[0].isupper():
                        nlp_algorithms.append(f"keras.preprocessing.text.{item_name}")
        except AttributeError:
            pass
            
    except ImportError:
        print("tensorflow not available")
    
    # Process standalone Keras if available
    try:
        import keras
        
        keras_layers = keras.layers
        for item_name in dir(keras_layers):
            if not item_name.startswith('_'):
                nlp_keywords = [
                    'rnn', 'lstm', 'gru', 'embedding', 'attention', 'multihead',
                    'text', 'tokeniz', 'preprocess'
                ]
                
                if any(keyword in item_name.lower() for keyword in nlp_keywords):
                    item = f"keras.layers.{item_name}"
                    if item not in nlp_algorithms:
                        nlp_algorithms.append(item)
                        
    except ImportError:
        pass
    
    # Process NLTK (most relevant for NLP)
    try:
        import nltk
        
        # NLTK modules most relevant for NLP algorithms
        nltk_modules = {
            'classify': 'classifiers',
            'cluster': 'clustering algorithms', 
            'sentiment': 'sentiment analysis',
            'tag': 'POS taggers',
            'parse': 'parsers',
            'tokenize': 'tokenizers',
            'stem': 'stemmers',
            'chunk': 'chunkers',
            'translate': 'translation',
            'corpus': 'corpus readers'
        }
        
        for module_name in nltk_modules.keys():
            try:
                module = getattr(nltk, module_name)
                for item_name in dir(module):
                    if (not item_name.startswith('_') and 
                        item_name[0].isupper() and 
                        len(item_name) > 2):
                        nlp_algorithms.append(f"nltk.{module_name}.{item_name}")
            except (AttributeError, ImportError):
                continue
        
        # Add known NLTK NLP algorithms
        known_nltk_nlp = [
            ('classify', 'NaiveBayesClassifier'),
            ('classify', 'MaxentClassifier'),
            ('classify', 'DecisionTreeClassifier'),
            ('cluster', 'KMeansClusterer'),
            ('sentiment', 'SentimentIntensityAnalyzer'),
            ('tag', 'PerceptronTagger'),
            ('tag', 'UnigramTagger'),
            ('tag', 'BigramTagger'),
            ('tag', 'TrigramTagger'),
            ('stem', 'PorterStemmer'),
            ('stem', 'LancasterStemmer'),
            ('stem', 'SnowballStemmer'),
            ('tokenize', 'WordTokenizer'),
            ('tokenize', 'SentenceTokenizer'),
            ('tokenize', 'PunktSentenceTokenizer'),
            ('tokenize', 'WordPunctTokenizer'),
            ('tokenize', 'TreebankWordTokenizer'),
            ('chunk', 'RegexpParser'),
            ('parse', 'RecursiveDescentParser'),
            ('parse', 'ShiftReduceParser')
        ]
        
        for module_name, alg in known_nltk_nlp:
            try:
                module = getattr(nltk, module_name)
                if hasattr(module, alg):
                    item = f"nltk.{module_name}.{alg}"
                    if item not in nlp_algorithms:
                        nlp_algorithms.append(item)
            except AttributeError:
                continue
                
    except ImportError:
        print("nltk not available")
    
    # Process Transformers (Hugging Face) - highly relevant for NLP
    try:
        import transformers
        
        for item_name in dir(transformers):
            if not item_name.startswith('_'):
                # Transformers library items are almost all NLP-related
                nlp_patterns = [
                    'model', 'tokenizer', 'processor', 'pipeline', 'config',
                    'bert', 'gpt', 'roberta', 'distilbert', 't5', 'bart',
                    'electra', 'albert', 'xlm', 'xlnet', 'deberta'
                ]
                
                item_lower = item_name.lower()
                if (any(pattern in item_lower for pattern in nlp_patterns) and
                    item_name[0].isupper() and
                    len(item_name) > 3):
                    nlp_algorithms.append(f"transformers.{item_name}")
                    
    except ImportError:
        print("transformers not available")
    
    # Remove duplicates and sort
    nlp_algorithms = sorted(list(set(nlp_algorithms)))
    
    print(f"Found {len(nlp_algorithms)} NLP algorithms")
    return nlp_algorithms


## Functions to improve processing speed
def process_single_file(args):
    """
    Process a single file and return usage counts
    This function will be run in parallel
    """
    kv_id, code_base_path, compiled_patterns = args
    
    # Get code file path
    code_file_path = code_path_for_version(kv_id, code_base_path)
    
    if code_file_path is None:
        return None, f"File not found for kv_id: {kv_id}"
    
    try:
        # Read the code file
        with open(code_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code_content = f.read()
        
        # Count usage for this file
        file_usage = defaultdict(int)
        
        # Use pre-compiled regex patterns for better performance
        for item, pattern in compiled_patterns.items():
            matches = pattern.findall(code_content)
            if matches:
                file_usage[item] = len(matches)
        
        return dict(file_usage), None
        
    except Exception as e:
        return None, f"Error reading {code_file_path}: {str(e)}"

def compile_regex_patterns(library_items):
    """Pre-compile regex patterns for better performance"""
    compiled_patterns = {}
    for item in library_items:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(item) + r'\b'
        compiled_patterns[item] = re.compile(pattern)
    return compiled_patterns


def analyze_library_usage_optimized(csv_path: str, code_base_path: str, 
                                  max_workers: int = None, chunk_size: int = 100):
    """
    Optimized version with parallel processing and batching
    
    Args:
        csv_path: Path to your imports_df.csv file
        code_base_path: Base path where code files are stored
        max_workers: Number of parallel workers (default: CPU count)
        chunk_size: Size of chunks for progress reporting
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Don't use too many workers
    
    print(f"Using {max_workers} parallel workers")
    
    # Load your dataset
    imports_df = pd.read_csv(csv_path)
    nlp_data = imports_df[imports_df['Category'] == 'nlp']
    
    # Get all library items and compile regex patterns once
    all_library_items = get_nlp_algorithms()
    compiled_patterns = compile_regex_patterns(all_library_items)
    
    print(f"Analyzing {len(all_library_items)} library items...")
    print(f"Processing {len(nlp_data)} kernel versions...")
    
    # Prepare arguments for parallel processing
    process_args = [
        (row['KernelVersionId'], code_base_path, compiled_patterns)
        for _, row in nlp_data.iterrows()
    ]
    
    # Initialize counters
    total_usage_counter = Counter()
    processed_files = 0
    failed_files = 0
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Process in chunks for progress reporting
        for i in range(0, len(process_args), chunk_size):
            chunk = process_args[i:i + chunk_size]
            
            # Submit chunk to executor
            future_to_args = {
                executor.submit(process_single_file, args): args 
                for args in chunk
            }
            
            # Collect results
            for future in future_to_args:
                try:
                    file_usage, error = future.result()
                    
                    if file_usage is not None:
                        # Merge results
                        for item, count in file_usage.items():
                            total_usage_counter[item] += count
                        processed_files += 1
                    else:
                        failed_files += 1
                        if error:
                            print(f"Failed: {error}")
                            
                except Exception as e:
                    failed_files += 1
                    print(f"Processing error: {e}")
            
            # Progress report
            current_processed = processed_files + failed_files
            print(f"Processed {current_processed}/{len(nlp_data)} files "
                  f"({current_processed/len(nlp_data)*100:.1f}%) - "
                  f"Success: {processed_files}, Failed: {failed_files}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_files} files")
    print(f"Failed to process: {failed_files} files")
    print(f"Total unique items found: {len(total_usage_counter)}")
    
    # Sort by usage frequency (most used first)
    sorted_usage = total_usage_counter.most_common()
    
    # Display results
    print(f"\nTop 20 most used library items:")
    print("=" * 50)
    for item, count in sorted_usage[:20]:
        print(f"{item}: {count} times")
    
    # Save results to CSV for further analysis
    usage_df = pd.DataFrame(sorted_usage, columns=['Library_Item', 'Usage_Count'])
    usage_df.to_csv('/kaggle/working/library_usage_frequency_optimized.csv', index=False) #Path to save the file
    print(f"\nResults saved to 'library_usage_frequency_optimized.csv'")
    
    return total_usage_counter



# Example usage:
if __name__ == "__main__":
    CSV_PATH = "/kaggle/input/kv-comp-cat/kv_comp_cat.csv"
    CODE_BASE_PATH = "/kaggle/input/meta-kaggle-code"  # Replace with actual path
    
    print("Choose optimization approach:")
    print("1. Parallel processing (fastest for sufficient RAM)")
    print("2. Memory-efficient batch processing (for large datasets)")
    print()
    print("To run the analysis:")
    print("# For parallel processing:")
    usage_stats = analyze_library_usage_optimized(CSV_PATH, CODE_BASE_PATH)
    print()


library_usage = pd.read_csv("/kaggle/input/algorithms-usage-for-nlp-competitions/library_usage_frequency_optimized.csv")


library_usage.head(50)




