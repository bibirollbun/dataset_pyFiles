# Suppress warnings for cleaner output

import warnings
warnings.simplefilter('ignore')


# ğŸ§° Essential libraries for data handling and model training


# Python standard libraries
import gc    # Garbage collection
import json
import torch
import random
from itertools import chain
from functools import partial
from pathlib import Path

# Data preprocessing and visualization
import numpy as np
import pandas as pd
from datasets import Dataset, features

# Model evaluation
from sklearn.metrics import f1_score

# HuggingFace Transformers
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)


class InferenceConfig:
    """
    Configuration settings used during model inference (prediction on test data).
    """
    # ğŸ“‚ Path to test data
    TEST_DATA_PATH = '/kaggle/input/pii-detection-removal-from-educational-data/test.json'
    
    # ğŸ“¦ Path to saved model checkpoint from training
    MODEL_CHECKPOINT_PATH = '/kaggle/input/deberta_finetuned1/pytorch/default/1/output/checkpoint-1000'
    
    # ğŸ”§ Tokenization settings
    MAX_TOKEN_LENGTH = 3072
    TOKENIZER_STRIDE = 256
    NUM_WORKERS = 4
    
    # ğŸ”� Inference behavior
    CONFIDENCE_THRESHOLD = 0.84
    BATCH_SIZE = 4
    RANDOM_SEED = 457


def set_global_seed(seed: int):
    """
    Sets the seed across random, numpy, and torch for reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Set seed using the config value
set_global_seed(InferenceConfig.RANDOM_SEED)


gc.collect()


class InferencePreprocessor:
    """
    This class handles tokenization of input data for inference,
    while maintaining a map between original tokens and tokenized output.
    """
    
    def tokenize_with_mapping(self, example, tokenizer):
        """
        Tokenizes input text and builds a map from characters to original tokens.

        Args:
            example (dict): A sample from the dataset with keys "tokens" and "trailing_whitespace".
            tokenizer: A HuggingFace tokenizer compatible with the trained model.

        Returns:
            dict: A dictionary with tokenized output and a token-to-character map.
        """
        full_text = []
        token_map = []  # Keeps track of which character belongs to which original token
        current_token_index = 0

        for token, has_space in zip(example["tokens"], example["trailing_whitespace"]):
            full_text.append(token)
            token_map.extend([current_token_index] * len(token))

            if has_space:
                full_text.append(" ")
                token_map.append(-1)  # -1 means the space is not part of any token

            current_token_index += 1

        # Concatenate the full text and tokenize it
        joined_text = "".join(full_text)
        tokenized_output = tokenizer(
            joined_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=InferenceConfig.MAX_TOKEN_LENGTH,
            stride=InferenceConfig.TOKENIZER_STRIDE,
            return_overflowing_tokens=True
        )

        return {
            **tokenized_output,
            "token_map": token_map,
        }

# ğŸ“¦ Initialize the inference preprocessor
preprocessor = InferencePreprocessor()


# Step 1: Load the test data
# The test data is stored in JSON format and contains pre-tokenized documents.
with open(InferenceConfig.TEST_DATA_PATH, 'r') as file:
    test_data = json.load(file)

# Step 2: Convert the loaded data into a Hugging Face Dataset
# This structure makes it easier to apply transformations like tokenization.
test_dataset = Dataset.from_dict({
    "full_text": [item["full_text"] for item in test_data],
    "document": [item["document"] for item in test_data],
    "tokens": [item["tokens"] for item in test_data],
    "trailing_whitespace": [item["trailing_whitespace"] for item in test_data],
})

# Step 3: Initialize the tokenizer
# We use a tokenizer from a pre-trained transformer model (e.g., BERT, RoBERTa).
tokenizer = AutoTokenizer.from_pretrained(InferenceConfig.MODEL_CHECKPOINT_PATH)

# Step 4: Tokenize the dataset
# The custom function 'dp.tokenize' will prepare the dataset inputs for the model.
# The 'map' function applies the tokenize function to each sample.
# 'num_proc' allows multiprocessing to speed up the process.
tokenized_dataset = test_dataset.map(
    preprocessor.tokenize_with_mapping,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=InferenceConfig.NUM_WORKERS
)


from sklearn.metrics import fbeta_score

class ModelInference:
    def __init__(self):
        pass

    def trim_stride_predictions(self, sub_predictions, num_splits, config):
        if num_splits != 1:
            for i in range(num_splits):
                if i == 0:
                    sub_predictions = sub_predictions[:, :-1, :]
                elif i == num_splits - 1:
                    sub_predictions = sub_predictions[:, 1 + config.TOKENIZER_STRIDE:, :]
                else:
                    sub_predictions = sub_predictions[:, 1 + config.TOKENIZER_STRIDE:-1, :]
        return sub_predictions

    def trim_stride_offsets(self, offset_list, num_splits, config):
        if num_splits != 1:
            for i in range(num_splits):
                if i == 0:
                    offset_list = offset_list[:-1]
                elif i == num_splits - 1:
                    offset_list = offset_list[1 + config.TOKENIZER_STRIDE:]
                else:
                    offset_list = offset_list[1 + config.TOKENIZER_STRIDE:-1]
        return offset_list

    def run_inference_on_dataset(self, dataset, tokenizer, config):
        model = AutoModelForTokenClassification.from_pretrained(config.MODEL_CHECKPOINT_PATH)
        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

        eval_args = TrainingArguments(
            output_dir=".",
            per_device_eval_batch_size=config.BATCH_SIZE,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            data_collator=collator,
            tokenizer=tokenizer,
        )

        predictions_list = []
        data_dict = {
            "document": [],
            "token_map": [],
            "offset_mapping": [],
            "tokens": []
        }

        for row in dataset:
            split_predictions = []
            combined_offsets = []

            for i, offsets in enumerate(row["offset_mapping"]):
                split_input = Dataset.from_dict({
                    "input_ids": [row["input_ids"][i]],
                    "token_type_ids": [row["token_type_ids"][i]],
                    "attention_mask": [row["attention_mask"][i]],
                    "offset_mapping": [offsets],
                })

                prediction_logits = trainer.predict(split_input).predictions

                trimmed_preds = self.trim_stride_predictions(prediction_logits, len(row["offset_mapping"]), config)
                trimmed_offsets = self.trim_stride_offsets(offsets, len(row["offset_mapping"]), config)

                split_predictions.append(trimmed_preds)
                combined_offsets += trimmed_offsets

            data_dict["document"].append(row["document"])
            data_dict["tokens"].append(row["tokens"])
            data_dict["token_map"].append(row["token_map"])
            data_dict["offset_mapping"].append(combined_offsets)

            concatenated_preds = np.concatenate(split_predictions, axis=1)
            predictions_list.append(concatenated_preds)

        with open(Path(config.MODEL_CHECKPOINT_PATH) / "config.json", 'r') as f:
            id2label = json.load(f)["id2label"]

        final_predictions = []
        for logits in predictions_list:
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=2, keepdims=True)
            best_non_O = probs[:, :, :12].argmax(axis=-1)
            O_probs = probs[:, :, 12]
            labels = logits.argmax(-1)
            final = np.where(O_probs < config.CONFIDENCE_THRESHOLD, best_non_O, labels)
            final_predictions.append(final)

        processed_dataset = Dataset.from_dict(data_dict)

        unique_pairs = set()
        documents, tokens, labels, token_texts = [], [], [], []

        for pred, token_map, offsets, token_list, doc_id in zip(final_predictions, processed_dataset["token_map"],
                                                                processed_dataset["offset_mapping"],
                                                                processed_dataset["tokens"],
                                                                processed_dataset["document"]):

            for token_id, (start, end) in zip(pred[0], offsets):
                label_name = id2label[str(token_id)]
                if start + end == 0:
                    continue

                if token_map[start] == -1:
                    start += 1

                while start < len(token_map) and token_list[token_map[start]].isspace():
                    start += 1

                if start >= len(token_map):
                    break

                true_token_id = token_map[start]

                if label_name != "O" and true_token_id != -1:
                    pair = (doc_id, true_token_id)
                    if pair not in unique_pairs:
                        documents.append(doc_id)
                        tokens.append(true_token_id)
                        labels.append(label_name)
                        token_texts.append(token_list[true_token_id])
                        unique_pairs.add(pair)

        final_df = pd.DataFrame({
            "document": documents,
            "token": tokens,
            "label": labels,
            "token_str": token_texts,
        })

        final_df["row_id"] = range(len(final_df))
        return final_df

    def compute_fbeta(self, predictions_df, ground_truth_df, beta=1.0):
        """
        Compare predicted labels with ground-truth labels and compute F-beta score.
        """
        # PrÃ©parer les prÃ©dictions
        pred = predictions_df[["document", "token", "label"]].copy()
        pred["token"] = pred["token"].astype(int)
    
        # PrÃ©parer le ground-truth : explosion des tokens et labels
        exploded_true = []
    
        for _, row in ground_truth_df.iterrows():
            doc_id = row["document"]
            tokens = row["tokens"]
            labels = row["labels"]
            for idx, (tok, lab) in enumerate(zip(tokens, labels)):
                exploded_true.append({
                    "document": doc_id,
                    "token": idx,
                    "label": lab
                })
    
        true_df = pd.DataFrame(exploded_true)
    
        # Fusionner proprement
        merged = pd.merge(pred, true_df, on=["document", "token"], how="outer", suffixes=("_pred", "_true")).fillna("O")
    
        # Extraire les labels
        y_pred = merged["label_pred"].tolist()
        y_true = merged["label_true"].tolist()
    
    
        # Calcul du F-beta
        score = fbeta_score(
            y_true, y_pred, beta=beta, average="weighted",
            labels=list(set(y_true + y_pred))
        )
    
        print("F-beta score:", score)
        return {"fbeta": score}



mi = ModelInference()


# Process dataset and create DataFrame
df = mi.run_inference_on_dataset(tokenized_dataset, tokenizer, InferenceConfig)


df[["row_id", "document", "token", "label"]].to_csv("submission.csv", index=False)


df.head(10)




