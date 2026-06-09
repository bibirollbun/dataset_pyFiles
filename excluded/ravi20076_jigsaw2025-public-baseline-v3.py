


%%writefile myimports.py

import os, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import ctypes
libc = ctypes.CDLL("libc.so.6")
from os import path, walk, getpid
from psutil import Process
from copy import deepcopy

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from IPython.display import display, Math, Latex, clear_output

import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset
from transformers import (
    DebertaTokenizer,
    DebertaV2Tokenizer,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    RobertaTokenizer,
    DataCollatorWithPadding ,
    logging as t_logging,
)

from sklearn.model_selection import *
from sklearn.metrics import *

import joblib, random
from colorama import Fore, Back, Style
from warnings import filterwarnings
from gc import collect
from tqdm.notebook import tqdm

t_logging.set_verbosity_error()
import logging
(
    logging.getLogger(
        "transformers.modeling_utils"
        ).
    setLevel(logging.ERROR)
)
filterwarnings("ignore")

def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)

class Utils:
    """
    This class creates and uses several utility methods to be used across the code
    """

    def __init__(self):
        pass

    def ScoreMetric(self, ytrue, ypreds):
        score = roc_auc_score(ytrue, ypreds)

    def pp_preds(self, ypreds : np.ndarray)-> np.ndarray :
        "Post-processes the predictions using min-max values from the training data"
        return np.clip( ypreds , a_min = 0, a_max = 1 )

    def CleanMemory(self):
        "This method cleans the memory off unused objects and displays the cleaned state RAM usage"

        collect();
        libc.malloc_trim(0)
        pid        = getpid()
        py         = Process(pid)
        memory_use = py.memory_info()[0] / 2. ** 30
        return f"\nRAM usage = {memory_use :.4} GB"



%%writefile ettinV1_1.py

PrintColor(
    f"\n---> ETINV1_1 CV = 0.8672208 inferencing\n"
)

ip_path   = f"/kaggle/input/jigsaw2025publicmodelsv1/ETTINV1_1"
MAX_LEN   = 512
n_classes = 2

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding     = "longest",
        truncation  = True,
        max_length  = MAX_LEN
    )

def format_input(row):
    return f"""
    Comment: {row['body']}
    Rule: {row['rule']}
    Subreddit: {row['subreddit']}
    Positive Examples: {row['positive_example_1']} || {row['positive_example_2']}
    Negative Examples: {row['negative_example_1']} || {row['negative_example_2']}
    """

Xtest  = pd.read_csv(f"/kaggle/input/jigsaw-agile-community-rules/test.csv", index_col = ["row_id"])
sub_fl = pd.read_csv(f"/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv", index_col = ["row_id"])
Xtest['text']  = Xtest.apply(format_input, axis=1)


tokenizer = AutoTokenizer.from_pretrained(ip_path)
model = AutoModelForSequenceClassification.from_pretrained(
    ip_path,
    num_labels = n_classes
)

training_args = TrainingArguments(
    output_dir                  = "Model",
    do_train                    = False,
    per_device_eval_batch_size  = 64,
    report_to                   = "none",
    dataloader_pin_memory       = False,
    logging_strategy            = "no",
    fp16                        = True,
    torch_compile               = True,
    seed                        = 42,
    data_seed                   = 42,
)

trainer = Trainer(
    model               = model,
    args                = training_args,
    processing_class    = tokenizer,
    data_collator       = DataCollatorWithPadding(tokenizer=tokenizer),
)

Xt         = Dataset.from_pandas( Xtest[[ "text" ]]).map(tokenize, batched=True)
test_preds = trainer.predict(Xt)
test_preds = torch.nn.functional.softmax(torch.tensor(test_preds.predictions), dim=1).numpy()[:,1]

sub_fl["rule_violation"] = test_preds
sub_fl.to_csv(f"submission.csv", index = True)


%%time 

exec(open( f"myimports.py", "r").read())
exec(open( f"ettinV1_1.py", "r").read())

!head submission.csv
print()

shutil.rmtree("Model")
!ls 

print()

