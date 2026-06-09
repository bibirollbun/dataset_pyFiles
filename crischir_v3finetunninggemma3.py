# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import torch

graphic_card = '0' # choose which graphic card

os.environ["CUDA_VISIBLE_DEVICES"] = graphic_card
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
device       = torch.device(f"cuda:{graphic_card}" if torch.cuda.is_available() else "cpu")




if device.type == 'cuda':
    print("Using GPU.")
    torch.cuda.set_device(0)  
    print(torch.cuda.device_count())
    gpu_device   = 'cuda:0'
else:
    print("Using CPU.")
    gpu_device = torch.device('cpu')


print(f"Using device: {device}")


import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict, Features, ClassLabel, Value

from peft import (LoraConfig, 
                  PeftModel, 
                  prepare_model_for_kbit_training, 
                  get_peft_model,
                  PeftModelForSequenceClassification,
                  PeftConfig)

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.gemma3 import Gemma3ForConditionalGeneration, Gemma3Processor, Gemma3ForCausalLM
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoModelForCausalLM)

import bitsandbytes as bnb
import evaluate
import kagglehub


from datasets import Dataset, load_dataset, DatasetDict


test=False
if test:
    EPOCH=2
    print("number of epoch =2")
else:
    EPOCH=10
    print("number of epoch =10")


from sklearn.utils import shuffle


import random


# --- DATASET PREPARATION ---

# Load training data
add_on_df=pd.read_csv("/kaggle/input/dummy-data-jigsaw-2025/training examples2.csv", encoding='ISO-8859-1')
add_on_df=add_on_df.dropna()
train_df=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
train_df = pd.concat([train_df, add_on_df], axis=0)
train_df = shuffle(train_df).reset_index(drop=True)
train_df.rule_violation=train_df.rule_violation.astype(int)

# Apply random positive and negative example selection for training
def select_random_examples(df):
    df["positive_example"] = df.apply(
        lambda row: random.choice([row["positive_example_1"], row["positive_example_2"]]),
        axis=1
    )
    df["negative_example"] = df.apply(
        lambda row: random.choice([row["negative_example_1"], row["negative_example_2"]]),
        axis=1
    )
    df = df.drop(
        columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"],
        errors="ignore"
    )
    return df

train_df = select_random_examples(train_df)


if test:
    train_df=train_df[:300]


# Define the features with 'rule_violation' as a ClassLabel
features = Features({
    'row_id': Value(dtype='int64'),
    'body': Value(dtype='string'),
    'rule': Value(dtype='string'),
    'subreddit': Value(dtype='string'),
    'positive_example': Value(dtype='string'),
    'negative_example': Value(dtype='string'),
    'rule_violation': ClassLabel(names=['not_violation', 'violation']),
})
# print(train_df)
full_dataset = Dataset.from_pandas(train_df, features=features)
# Perform a single, stratified split for training and a combined validation/test set
# `stratify_by_column` will now work because 'rule_violation' is a ClassLabel
dataset_prompt = full_dataset.train_test_split(
    test_size=0.1, 
    seed=42, 
    stratify_by_column="rule_violation"
)

# Split the 10% test set into validation and final test sets
# Re-apply stratification to ensure both classes are present in both splits
test_valid = dataset_prompt['test'].train_test_split(
    test_size=0.5, # Split 50/50 for a more balanced split
    seed=42,
    stratify_by_column="rule_violation"
)

dataset_prompt = DatasetDict({
    'train': dataset_prompt['train'],
    'valid': test_valid['train'],
    'test': test_valid['test']
})

print(f"Dataset structure:\n{dataset_prompt}")

# Define labels
class_label = 'violation'
class2id = {f'not_{class_label}': 0, class_label: 1}
id2class = {v: k for k, v in class2id.items()}
num_labels = len(class2id)



# add_on_df=pd.read_csv("/kaggle/input/dummy-data-jigsaw-2025/training examples2.csv", encoding='ISO-8859-1')
# add_on_df=add_on_df.dropna()


# train_df=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
# train_df = pd.concat([train_df, add_on_df], axis=0)
# train_df = shuffle(train_df).reset_index(drop=True)




# train_df.describe()


# train_df.rule_violation=train_df.rule_violation.astype(int)


# train_df.tail()





# full_dataset = Dataset.from_pandas(train_df)


# # Split the dataset
# dataset_prompt = full_dataset.train_test_split(test_size=0.1, seed=42)
# test_valid = dataset_prompt['test'].train_test_split(test_size=0.3)


# dataset_prompt = DatasetDict({
#     'train': dataset_prompt['train'],
#     'valid': test_valid['train'],
#     'test': test_valid['test']
# })


# print(f"Dataset structure:\n{dataset_prompt}")


# # Define labels
# class_label = 'violation'
# class2id = {f'not_{class_label}': 0, class_label: 1}
# id2class = {v: k for k, v in class2id.items()}
# num_labels = len(class2id)


# num_labels



# import torch
# from transformers import AutoTokenizer


# --- 3. Model and Tokenizer Loading ---
# GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-4b-it")

GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")

# processor = Gemma3Processor.from_pretrained(GEMMA_PATH)

# Quantization configuration

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)


# model = Gemma3ForCausalLM.from_pretrained(GEMMA_PATH, 
#                                           torch_dtype=torch.bfloat16, 
#                                           # device_map=gpu_device,
#                                           attn_implementation='eager',
#                                           quantization_config=bnb_config  )

# model.lm_head = torch.nn.Linear(model.config.hidden_size, len(class2id.keys()),


# Load Model
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    quantization_config=bnb_config,
    device_map=gpu_device,
    attn_implementation='eager',
    torch_dtype=torch.bfloat16,
)


# # processor = Gemma3Processor.from_pretrained(GEMMA_PATH)
# processor = AutoTokenizer.from_pretrained(GEMMA_PATH)


# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
# **CORRECTION**: Set padding token correctly. The tokenizer object itself has the `eos_token`, not a `.tokenizer` attribute.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# # --- 4. Preprocessing ---
# def create_prompt(input_row):
#     """
#     Creates the formatted prompt text. Note: for classification, we do not include the answer in the input.
#     """
#     prompt_text = f"""<start_of_turn>user
# You are a really experienced moderator for the subreddit /r/{input_row['subreddit']}. Your job is to determine if the following reported comment violates the rule: {input_row['rule']}.

# Here are some examples:
# Positive Example 1: {str(input_row['positive_example_1'])} -> Decision: True
# Negative Example 1: {str(input_row['negative_example_1'])} -> Decision: False
# Negative Example 2: {str(input_row['negative_example_2'])} -> Decision: False
# Positive Example 2: {str(input_row['positive_example_2'])} -> Decision: True

# Now, evaluate the following comment:
# Comment: {str(input_row['body'])}
# <end_of_turn>
# <start_of_turn>model
# Decision:"""
#     return prompt_text


# # **CORRECTION**: This is the main fix. This function now correctly prepares data for a classification task.
# # It tokenizes only the input prompt and assigns the 'rule_violation' column as the label.
# def preprocess_function(sample):
#     """
#     Preprocessing function for sequence classification.
#     """
#     # 1. Create the prompt text which serves as the input to the model.
#     text = create_prompt(sample)
    
#     # 2. Tokenize the text.
#     tokenized_output = tokenizer(text, truncation=True, padding="max_length", max_length=900)
    
#     # 3. Assign the label for classification.
#     tokenized_output['labels'] = sample['rule_violation']
    
#     return tokenized_output



# --- Preprocessing Functions ---
def create_prompt(input_row):    
    """
    Creates the formatted prompt text. Note: for classification, we do not include the answer in the input.
    Now uses 'positive_example' and 'negative_example' columns which are pre-selected.
    """
    prompt_text = f"""<start_of_turn>userYou are a really experienced moderator for the subreddit /r/{input_row['subreddit']}. Your job is to determine if the following reported comment violates the rule: {input_row['rule']}.Here are some examples:Positive Example 1: {str(input_row['positive_example'])} -> Decision: TrueNegative Example 1: {str(input_row['negative_example'])} -> Decision: FalseNow, evaluate the following comment:Comment: {str(input_row['body'])}<end_of_turn><start_of_turn>modelDecision:"""
    return prompt_text

def preprocess_function(sample):    
    """
    Preprocessing function for sequence classification for training.
    """
    text = create_prompt(sample)
    tokenized_output = tokenizer(text, truncation=True, padding="max_length", max_length=550)
    tokenized_output['labels'] = sample['rule_violation']
    return tokenized_output

# Apply the preprocessing function to the dataset
dataset_prompt_tokenized = dataset_prompt.map(preprocess_function, batched=False, remove_columns=dataset_prompt['train'].column_names)
print(f"Tokenized dataset structure:\n{dataset_prompt_tokenized}")



# # Apply the preprocessing function to the dataset
# dataset_prompt_tokenized = dataset_prompt.map(preprocess_function, batched=False, remove_columns=dataset_prompt['train'].column_names)
# print(f"Tokenized dataset structure:\n{dataset_prompt_tokenized}")


# --- 5. Model Configuration for Classification ---
# Replace the model's classification head
model.lm_head = torch.nn.Linear(model.config.hidden_size, num_labels, bias=False, device=gpu_device)


# --- 6. Custom Model Wrapper for Sequence Classification ---
# This custom class is necessary to correctly handle the output of the CausalLM for a classification task.
class GemmaForSequenceClassification(torch.nn.Module):
    def __init__(self, model, num_labels):
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.config = model.config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get the outputs from the base Gemma model
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # The hidden state of the last token is used for classification
        hidden_states = outputs.last_hidden_state
        
        # Get logits from the modified lm_head
        logits = self.model.lm_head(hidden_states)

        # Extract the logits for the last token of each sequence
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            # Find the index of the last non-padding token
            sequence_lengths = (torch.eq(input_ids, self.model.config.pad_token_id).int().argmax(-1) - 1).to(logits.device)
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            last_token_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        else:
            # Fallback for cases without input_ids (e.g., during generation)
            last_token_logits = logits[:, -1, :]
            
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(last_token_logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=last_token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Wrap the PEFT model
wrapped_model = GemmaForSequenceClassification(model, num_labels)


# # --- Define the path to your cloned metrics repository (if necessary) ---
# # metrics_path = "path/to/your/cloned/evaluate/metrics"
# # metrics_path_accuracy = "/kaggle/input/evaluate-metrics/evaluate/metrics/accuracy"
# # metrics_path_f1 = "/kaggle/input/evaluate-metrics/evaluate/metrics/f1"
# # metrics_path_precision = "/kaggle/input/evaluate-metrics/evaluate/metrics/precision"
# # metrics_path_recall = "/kaggle/input/evaluate-metrics/evaluate/metrics/recall"
# # metrics_path_roc_auc = "/kaggle/input/evaluate-metrics/evaluate/metrics/roc_auc"
# # Combine multiple metrics

# # --- Define the metric objects in the global scope of your script ---
# # You can use the local path as before, or directly if the library is installed
# accuracy_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/accuracy/accuracy.py")
# f1_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/f1/f1.py")
# precision_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/precision/precision.py")
# recall_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/recall/recall.py")
# roc_auc_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/roc_auc/roc_auc.py")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred

#     # Step 1: Compute hard predictions for accuracy, f1, precision, recall
#     hard_predictions = np.argmax(predictions, axis=1)

#     # Step 2: Compute probabilities for roc_auc.
#     # We assume binary classification for this example
#     if predictions.shape[1] == 2:
#         probabilities = np.exp(predictions[:, 1]) / np.sum(np.exp(predictions), axis=1)
#     else:
#         # For multi-class, you might need a different approach or skip this metric
#         probabilities = None

#     # Step 3: Compute each metric and store in a dictionary
#     metrics = {
#         "accuracy": accuracy_metric.compute(predictions=hard_predictions, references=labels)["accuracy"],
#         "f1": f1_metric.compute(predictions=hard_predictions, references=labels, average="binary")["f1"],
#         "precision": precision_metric.compute(predictions=hard_predictions, references=labels, average="binary")["precision"],
#         "recall": recall_metric.compute(predictions=hard_predictions, references=labels, average="binary")["recall"],
#     }
    
#     # Only compute ROC AUC if we have probabilities
#     if probabilities is not None:
#         # Corrected line: use 'prediction_scores' instead of 'predictions'
#         metrics["roc_auc"] = roc_auc_metric.compute(prediction_scores=probabilities, references=labels)["roc_auc"]

#     return metrics
accuracy_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/accuracy/accuracy.py")
f1_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/f1/f1.py")
precision_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/precision/precision.py")
recall_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/recall/recall.py")
roc_auc_metric = evaluate.load("/kaggle/input/evaluate-metrics/evaluate/metrics/roc_auc/roc_auc.py")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    hard_predictions = np.argmax(predictions, axis=1)

    if predictions.shape[1] == 2:
        probabilities = np.exp(predictions[:, 1]) / np.sum(np.exp(predictions), axis=1)
    else:
        probabilities = None

    metrics = {
        "accuracy": accuracy_metric.compute(predictions=hard_predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=hard_predictions, references=labels, average="binary")["f1"],
        "precision": precision_metric.compute(predictions=hard_predictions, references=labels, average="binary")["precision"],
        "recall": recall_metric.compute(predictions=hard_predictions, references=labels, average="binary")["recall"],
    }
    
    if probabilities is not None:
        metrics["roc_auc"] = roc_auc_metric.compute(prediction_scores=probabilities, references=labels)["roc_auc"]

    return metrics


checkpoints_dir = 'gemma_prompt_classification'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


training_args = TrainingArguments(
    output_dir=checkpoints_dir,
    learning_rate=1e-5,
    lr_scheduler_type="cosine", # Use a cosine learning rate scheduler
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=EPOCH,
    weight_decay=0.01,
    
    # **CORRECTION**: Changed 'evaluation_strategy' back to 'eval_strategy'
    # for compatibility with the library version.
    eval_strategy='epoch',
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
    gradient_accumulation_steps=4,
    # fp16=True, # This is correct and should be used
    bf16=True, # Explicitly set this to False if it's not already
    warmup_ratio=0.01,
    logging_strategy="steps",
    logging_steps=10,
    # metric_for_best_model='roc_auc',
    # greater_is_better=True, # ROC AUC is a "more is better" metric
    metric_for_best_model='f1',
    greater_is_better=True, # F1 is a "more is better" metric
    # metric_for_best_model='f1',
    # load_best_model_at_end=True, # Load the best model at the end of training
    save_total_limit=2, # Keep the best and the latest checkpoint
)

# **CORRECTION**: Define the data collator once and correctly.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=wrapped_model,
    args=training_args,
    train_dataset=dataset_prompt_tokenized['train'],
    eval_dataset=dataset_prompt_tokenized['valid'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)





torch.cuda.empty_cache()


# --- 8. Train the model ---
print("Starting training...")
trainer.train()


torch.cuda.empty_cache()


# --- 9. Save the fine-tuned model ---
output_dir = 'my_gemma_prompt_classifier_final'
# trainer.model.save_pretrained(output_dir)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nModel saved to '{output_dir}'")


import matplotlib.pyplot as plt

# After the trainer.train() call
logs = trainer.state.log_history

# Extract the metrics from the log history
eval_accuracy = []
eval_f1 = []
eval_roc_auc = [] # New list for ROC AUC

for log in logs:
    if "eval_accuracy" in log:
        eval_accuracy.append(log["eval_accuracy"])
    if "eval_f1" in log:
        eval_f1.append(log["eval_f1"])
    if "eval_roc_auc" in log:
        eval_roc_auc.append(log["eval_roc_auc"])

# Create a plot for the evaluation metrics
plt.figure(figsize=(10, 5))
plt.plot(eval_accuracy, label="Validation Accuracy")
plt.plot(eval_f1, label="Validation F1 Score")
plt.plot(eval_roc_auc, label="Validation ROC AUC") # Plot the ROC AUC
plt.title("Validation Metrics")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()





%%time
# --- Inference Data Preparation (New Section) ---
print("\n--- Preparing Inference Data ---")
test_inference_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# Apply random positive and negative example selection for inference
test_inference_df = select_random_examples(test_inference_df)

# Create Dataset from inference DataFrame
test_inference_dataset = Dataset.from_pandas(test_inference_df)

def inference_preprocess_function(sample):
    """
    Preprocessing function for inference.
    Creates prompt and tokenizes, but does not include 'labels'.
    """
    text = create_prompt(sample)
    tokenized_output = tokenizer(text, truncation=True, padding="max_length", max_length=900)
    # Important: Do NOT add 'labels' for inference
    return tokenized_output

# Apply preprocessing to inference dataset, preserving row_id if needed for submission
test_inference_tokenized = test_inference_dataset.map(
    inference_preprocess_function, 
    batched=False, 
    remove_columns=[col for col in test_inference_dataset.column_names if col != 'row_id'] # Keep row_id
)

print(f"Tokenized inference dataset structure:\n{test_inference_tokenized}")

# You would then use `trainer.predict(test_inference_tokenized)` or a similar method
# to get predictions and generate your submission file, using the `row_id` for mapping.


test_inference_tokenized


predictions = trainer.predict(test_inference_tokenized)

# 2. Extract the logits
# The predictions object contains the raw output logits from your model's classification head.
test_logits = predictions.predictions

# 3. Convert logits to class labels (0 or 1)
# You can get the final predicted class by taking the argmax of the logits.
predicted_labels = np.argmax(test_logits, axis=1)

# 4. Prepare for submission
# Add the predicted labels to your original test DataFrame.
submission_df = test_inference_df[['row_id']].copy()
submission_df['rule_violation'] = predicted_labels
submission_df.to_csv('submission.csv', index=False)


from scipy.special import softmax


%%time
# 1. Get predictions
# The `predict` method returns predictions (logits), labels, and metrics.
predictions = trainer.predict(test_inference_tokenized)

# 2. Extract the raw logits
test_logits = predictions.predictions

# 3. Convert logits to probabilities
# The `softmax` function will convert the logits into a probability distribution.
# We are interested in the probability for the "violation" class, which is at index 1.
test_probabilities = softmax(test_logits, axis=1)

# The 'violation' probability is in the second column (index 1)
violation_probabilities = test_probabilities[:, 1]

# 4. Prepare the submission file
submission_df = test_inference_df[['row_id']].copy()
submission_df['rule_violation'] = violation_probabilities

# 5. Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully with probabilities.")


submission_df.tail(10)




