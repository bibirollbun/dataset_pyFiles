!pip install datasets transformers tqdm langid pycountry -q


import torch
import pandas as pd

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, ClassLabel

import random
import langid
import pycountry

from abc import abstractmethod, ABC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import pandas as pd
import wandb


def get_train_data(
    seed: int = 42,
    removeNaNs: bool = False,
    validation_proportion: float = 0.2,
    shuffle: bool = True,
):
    """
    Returns two dataframes
    """
    train_df = pd.read_csv("/kaggle/input/nlp-cs-2025/train_submission.csv")

    if shuffle:
        train_df = train_df.sample(frac=1, random_state=seed)

    if removeNaNs:
        train_df = train_df.dropna()

    validation_df = train_df.sample(frac=validation_proportion, random_state=seed)
    train_df = train_df.drop(validation_df.index)

    return train_df, validation_df


def get_test_data():
    """
    Returns a dataframe
    """
    test_df = pd.read_csv("/kaggle/input/nlp-cs-2025/test_without_labels.csv")

    return test_df



Unicode = str
Language = str


train_df_without_NaNs, _ = get_train_data(
    seed=1, removeNaNs=True, validation_proportion=0
)
LABELS: list[Language] = train_df_without_NaNs["Label"].unique().tolist()


def get_unicode(char):
    """
    Returns the Unicode code point of a given character.

    Args:
    char (str): A single character.

    Returns:
    str: The Unicode code point in the format 'U+XXXX'.
    """
    if len(char) != 1:
        raise ValueError("Input must be a single character.")

    return f"U+{ord(char):04X}"


def process_unicode(dataset, get_unicode):
    """
    Process the dataset to gather unique Unicode code points and
    associate each with the corresponding language.

    Args:
    dataset (DataFrame): The dataset containing language labels and text.
    get_unicode (function): The function to get Unicode from character.

    Returns:
    set, dict: A set of all unique Unicode code points and a dict mapping languages to their Unicode code points.
    """
    # Initialize a set for all unique Unicode values
    all_unicodes = set()

    # Initialize a dictionary to store unicodes for each language
    language_unicodes = {}

    # Iterate through each row in the dataset
    for index, row in dataset.iterrows():
        # Get the language and the text
        language = row["Label"]
        text = row["Text"]  # Assuming 'Text' column contains the actual text

        # Initialize a set to store unicodes for this specific language
        language_unicode_set = set()

        # Iterate over each character in the text
        for char in text:
            # Get the Unicode for the character
            unicode = get_unicode(char)

            # Add the Unicode to the global set
            all_unicodes.add(unicode)

            # Add the Unicode to the language-specific set
            language_unicode_set.add(unicode)

        # Add the language-specific Unicode set to the dictionary
        if language not in language_unicodes:
            language_unicodes[language] = language_unicode_set
        else:
            language_unicodes[language].update(language_unicode_set)

    return all_unicodes, language_unicodes


def inverse_dictionary(dictionary):
    """
    dictionnary : keys = languages, values = set of all unicodes seen in that language

    output : keys = unicodes, values = set of languages in which they appear
    """
    res = {}

    for language in dictionary:
        for unicode in dictionary[language]:
            if unicode not in res:
                res[unicode] = set()
            res[unicode].add(language)

    return res


# Set device
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb")
wandb.login(key=secret_value_0)
wandb.init(project="KaggleNLP")

device = torch.device("cuda:0")

# Custom Dataset class
class LanguageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = [text.replace("</s>", "").strip() for text in texts]  # Remove extra <eos> tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }



# Load and preprocess data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    texts = df["Text"].values

    # Create label mapping
    unique_labels = df["Label"].unique()
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    labels = [label_to_id[label] for label in df["Label"]]
    return texts, labels, label_to_id, id_to_label


# Compute metrics function for Trainer
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    metrics = {"accuracy": accuracy, "f1_weighted": f1}

    wandb.log(metrics)
    return metrics


# Main training function
def train_model():
    # Load data
    texts, labels, label_to_id, id_to_label = load_and_prepare_data(
        "/kaggle/input/nlp-cs-2025/train_submission.csv"
    )

    # Split into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    
    print("number of classes in train", len(label_to_id))
    print("number of samples in train", len(train_texts))

    # Initialize tokenizer and model
    model_name = "google/madlad400-3b-mt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_to_id)
    ).to(device)

    for name, param in model.named_parameters():
        if "classif" not in name:
            param.requires_grad = False
            
            
    # Create datasets
    train_dataset = LanguageDataset(train_texts, train_labels, tokenizer, max_length=64)
    val_dataset = LanguageDataset(val_texts, val_labels, tokenizer, max_length=64)

    # Training arguments optimized for RTX 4090
    training_args = TrainingArguments(
        output_dir="./results_2",
        num_train_epochs=2,
        per_device_train_batch_size=64,  # Adjust based on VRAM usage
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,  # Effective batch size: 128
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        fp16=True,
        load_best_model_at_end=True,
        #report_to="wandb",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")

    return trainer, id_to_label


# Inference function
def predict(text, model, tokenizer, id_to_label, max_length=128):
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    return id_to_label[prediction]


if __name__ == "__main__":
    # Train the model
    trainer, id_to_label = train_model()


