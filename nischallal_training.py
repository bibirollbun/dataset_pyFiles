#
import os
# CUDA_LAUNCH_BLOCKING = "1" # Debug environment variable to see proper traces.
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU to used. 0 means use GPU 1.
os.environ["WANDB_DISABLED"] = "true" # The primary switch
os.environ["WANDB_MODE"] = "offline"

import re
import random
import logging
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache

#
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve
#
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Adam

from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


# ignore warnings
from transformers import logging as trans_logging
trans_logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)


"""This is equivalent to config.py. On google colab this class is used to hold
all the configurations to be used throughout the project.
"""
class Config:
    VER = 1
    SEED = 3407
    CUDA_AVAILABLE = False #  gpu or cpu? to be set later
    DEVICE = None # based on CUDA_AVAILABLE
    DATA_DIR = Path("/kaggle/input/jigsaw-agile-community-rules")
    OUTPUT_DIR = Path("Output")
    LOGS_DIR = Path(OUTPUT_DIR, "logs")
    MODELS_DIR = Path(OUTPUT_DIR, f"Jigsaw_agile_community_rule_classifier_ver_{VER}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOAD_TOKENS_FROM = None
    LOAD_MODEL_FROM = None
    DOWNLOADED_MODEL_PATH = None


def initialize_logger(log_file: str = Path(Config.LOGS_DIR, "info.log")):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

Config.logger = initialize_logger()


Config.CUDA_AVAILABLE = torch.cuda.is_available()
Config.DEVICE = torch.device("cuda" if Config.CUDA_AVAILABLE else "cpu")
Config.logger.info(f"We are using {Config.DEVICE}")


# https://odsc.medium.com/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752

def set_seed(seed: int = Config.SEED) -> None:
    """Seed all random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator

    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    Config.logger.info(f"Using Seed Number: {seed}")


set_seed()


"""
The classes below tracks different parameters to be used through out the project.
The idea is to make a change only here and not all the part where these
variables can be used.
"""
class FilePaths:
    train_csv = Path(Config.DATA_DIR, "train.csv")
    test_csv = Path(Config.DATA_DIR, "test.csv")
    submit_csv = Path(Config.DATA_DIR, "sample_submission.csv")


class ModelParams:
    MAX_LEN = 512 # TODO: Verify this, 512
    model_name = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
    output_len = 2



class DataLoaderParams:
    TRAIN_BATCH_SIZE = 8 #8
    VALID_BATCH_SIZE = 8 #4
    train_loader = {
            "batch_size": TRAIN_BATCH_SIZE,
            "num_workers": 4,
            "pin_memory": False,
            "drop_last": True,
            "shuffle": True,
            "collate_fn": None
    }

    valid_loader = {
            "batch_size": VALID_BATCH_SIZE,
            "num_workers": 4,
            "pin_memory": False,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None
    }

    test_loader = {
            "batch_size": VALID_BATCH_SIZE,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None
    }


class GlobalTrainParams:
    debug: bool = False
    epochs: int = 5


class CriterionParams:
    loss_function_name = "CrossEntropyLoss"


class OptimizerParams:
    """A class to track optimizer parameters.
    """
    optimizer_name = "Adam"
    lr = [1e-5, 3e-5, 2e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7]
    lr_decay = 0.96
    weight_decay = 0.01



class TokenizerParams:
    LOAD_TOKENS_FROM = None
    tokenizer_name = ModelParams().model_name
    lower_case = False # for deberta-v3-large
    max_length = ModelParams().MAX_LEN
    truncation = True
    padding = True #"max_length"


FILES = FilePaths()
LOADER_PARAMS = DataLoaderParams()
TRAIN_PARAMS = GlobalTrainParams()
CRITERION_PARAMS = CriterionParams()
OPTIMIZER_PARAMS = OptimizerParams()
MODEL_PARAMS = ModelParams()
TOKENIZER_PARAMS = TokenizerParams()


# Download tokenizer if not already downloaded and saved.
if Config.DOWNLOADED_MODEL_PATH is None:
  TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PARAMS.tokenizer_name)



def url_to_semantics(text: str) -> str:
    """
    Input : A text string (possibly containing URLs)
    Output: String containing keywords like 'domain:reddit path:comment'
    Logic :
        - Find URLs with regex
        - Extract domain and path parts
        - Add as 'domain:' and 'path:' tokens
        - Helps model learn URL meaning without raw URL noise
    """
    if not isinstance(text, str):
        return ""

    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    urls = re.findall(url_pattern, text)
    
    if not urls:
        return "" 

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()
        
        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split('.')
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3: # Avoid short parts like 'www'
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        # 2. Extract path parts
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()] # Split by common delimiters

        for part in path_parts:
            # Clean up potential file extensions or query params
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    if not all_semantics:
        return ""

    return f"\nURL Keywords: {' '.join(all_semantics)}"




def get_dataframe_to_train(data_path):
    """
    Input : Folder path containing 'train.csv' and 'test.csv'
    Output: Cleaned Pandas DataFrame with ['body','rule','subreddit','rule_violation']
    Logic :
        - Read both train/test CSVs
        - Flatten positive and negative examples into a single frame
        - Label positive → 1, negative → 0
        - Drop NA/empty texts
        - Remove duplicates by ['body','rule','subreddit']
        - Shuffle dataframe with random_state=42
    """
    train_dataset = pd.read_csv(f"{data_path}") 
    test_dataset = pd.read_csv(f"{data_path}")

    flatten = []

    flatten.append(train_dataset[["body", "rule", "subreddit","rule_violation"]].copy())

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in train_dataset.columns:
                sub_dataset = train_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in test_dataset.columns:
                sub_dataset = test_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(subset=['body', 'rule', 'subreddit'], ignore_index=True)
    dataframe.drop_duplicates(subset=['body','rule'],keep='first',inplace=True)
    
    return dataframe.sample(frac=1, random_state=42).reset_index(drop=True)


class JigsawDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


training_data_df = get_dataframe_to_train(FILES.train_csv)
test_df_for_prediction = pd.read_csv(FILES.test_csv)

print(training_data_df.shape)
print(test_df_for_prediction.shape)


training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
training_data_df['input_text'] = training_data_df['rule'] + "[SEP]" + training_data_df['body_with_url']


# from torch.optim import AdamW
# from transformers import Trainer, TrainingArguments, get_scheduler

# def get_optimizer_grouped_parameters(model, base_lr, weight_decay, lr_decay):
#     """
#     Create optimizer parameter groups with layer-wise learning rate decay.
#     """
#     layers = [model.deberta.encoder.layer[i] for i in range(model.config.num_hidden_layers)]
#     layers = list(reversed(layers))  # Top layers first (get higher LR)

#     optimizer_parameters = []

#     # Embedding layer
#     optimizer_parameters += [{
#         "params": model.deberta.embeddings.parameters(),
#         "lr": base_lr * (lr_decay ** (len(layers) + 1)),
#         "weight_decay": weight_decay
#     }]

#     # Transformer layers
#     for i, layer in enumerate(layers):
#         lr = base_lr * (lr_decay ** i)
#         optimizer_parameters += [{
#             "params": layer.parameters(),
#             "lr": lr,
#             "weight_decay": weight_decay
#         }]

#     # Classifier head (final layer)
#     optimizer_parameters += [{
#         "params": model.classifier.parameters(),
#         "lr": base_lr,
#         "weight_decay": 0.0
#     }]

#     return optimizer_parameters




# Tokenize
train_encodings = TOKENIZER(
    training_data_df['input_text'].tolist(), 
    # truncation=TOKENIZER_PARAMS.truncation,
    padding=TOKENIZER_PARAMS.padding,
    max_length=TOKENIZER_PARAMS.max_length)


train_dataset = JigsawDataset(train_encodings, training_data_df['rule_violation'].tolist())

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PARAMS.model_name, num_labels=MODEL_PARAMS.output_len)
# optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, OPTIMIZER_PARAMS.lr[1], OPTIMIZER_PARAMS.weight_decay, OPTIMIZER_PARAMS.lr_decay)
# optimizer = AdamW(optimizer_grouped_parameters)

# num_training_steps = int(len(training_data_df) / LOADER_PARAMS.TRAIN_BATCH_SIZE * TRAIN_PARAMS.epochs)
# scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=int(0.1 * num_training_steps),
#     num_training_steps=num_training_steps,
# )

# ****************************************

# For deberta-v3-large
# model.gradient_checkpointing_enable()  # <-- saves memory

# Log the training loss at each epoch
# logging_steps = len(training_data_df["rule_violation"])//LOADER_PARAMS.TRAIN_BATCH_SIZE
logging_steps = max(1, len(training_data_df) // (LOADER_PARAMS.TRAIN_BATCH_SIZE * 10))



training_args = TrainingArguments(
    report_to="none",
    output_dir=Config.OUTPUT_DIR,
    num_train_epochs=TRAIN_PARAMS.epochs,
    per_device_train_batch_size=LOADER_PARAMS.TRAIN_BATCH_SIZE,
    # per_device_eval_batch_size=LOADER_PARAMS.VALID_BATCH_SIZE,
    learning_rate=OptimizerParams.lr[2],
    # load_best_model_at_end=True,
    # metric_for_best_model="f1 (macro)",
    weight_decay=0.01, # TODO: Also make this as a param
    warmup_ratio=0.1,
    # eval_strategy="steps",
    save_strategy="steps",
    eval_steps = 100, #100
    logging_steps=logging_steps,
    disable_tqdm=False,

    # For deberta-v3-large
    # fp16=True, # Make it train fast.
    # gradient_accumulation_steps = 4, # 4

    # Save strategy
    save_total_limit=1,

)

trainer = Trainer(
    model=model,
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
    tokenizer=TOKENIZER,
    # optimizers=(optimizer, scheduler)  # Custom optimizer & scheduler
)

# Train
trainer.train()



# Save the final model
trainer.save_model(Config.MODELS_DIR)
trainer.tokenizer.save_pretrained(Config.MODELS_DIR)

print(f"Model and tokenizer saved to {Config.MODELS_DIR}")





###################3
def predict(trainer, tokenizer):
    test_df = pd.read_csv(FILES.test_csv)
    test_df['body_with_url'] = test_df['body'].apply(lambda x: x + url_to_semantics(x))
    test_df['input_text'] = test_df['rule'] + "[SEP]" + test_df['body_with_url']
    test_encodings = TOKENIZER(
        test_df['input_text'].tolist(), 
        # truncation=TOKENIZER_PARAMS.truncation,
        padding=TOKENIZER_PARAMS.padding,
        max_length=TOKENIZER_PARAMS.max_length
    )
    test_dataset = JigsawDataset(test_encodings)
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    submission_df = pd.DataFrame({
        "row_id": test_df["row_id"],
        "rule_violation": probs
    })
    output_file = f"submission_og.csv"
    submission_df.to_csv(output_file, index=False)
    print(submission_df.head(10))
    

    


predict(trainer, TOKENIZER)



import pandas as pd
import numpy as np
import argparse
import os

def run_probe(input_file: str, output_file: str, num_blocks: int, invert_block: int = -1, invert_all: bool = False):

    print(f"Reading submission file from: {input_file}")
    df = pd.read_csv(input_file)
 
    df['rule_violation'] = pd.to_numeric(df['rule_violation'])
    
    n_rows = len(df)
    print(f"Total rows: {n_rows}")

    if invert_all:
        print("Mode: Invert All. Transforming p -> 1-p for all predictions.")
        df['rule_violation'] = 1 - df['rule_violation']
    
    elif invert_block > 0:
        if not (1 <= invert_block <= num_blocks):
            raise ValueError(f"invert_block must be between 1 and {num_blocks}")
            
        print(f"Mode: Invert Gain Probe. Testing block {invert_block}/{num_blocks}.")
        print("Step 1: Inverting all predictions (p -> 1-p).")
        df['rule_violation'] = 1 - df['rule_violation']
        
        # 计算区块的边界
        block_size = n_rows / num_blocks
        start_index = int((invert_block - 1) * block_size)
        end_index = int(invert_block * block_size) if invert_block < num_blocks else n_rows
        
        print(f"Step 2: Reverting block {invert_block} (indices {start_index} to {end_index-1}) back to original (1-p -> p).")
        
        df.iloc[start_index:end_index, df.columns.get_loc('rule_violation')] = 1 - df.iloc[start_index:end_index]['rule_violation']

    else:
        print("Mode: Standard copy. No transformation applied.")

    print(f"Saving probed submission to: {output_file}")
    df.to_csv(output_file, index=False)
    print(df.head())
    print("Done.")

    
run_probe("/kaggle/working/submission_og.csv", "/kaggle/working/submission.csv", 10, 10, 0)




