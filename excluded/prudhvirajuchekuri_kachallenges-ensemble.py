%%capture
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth


import re
import os
import gc
import time
import torch
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from unsloth import FastLanguageModel


def set_random_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_random_seed()


# Function to clear memory when needed

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)


test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")
test_df


id2label = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}
label2id = {v: k for k, v in id2label.items()}


max_seq_length = 2048
dtype = None
load_in_4bit = True
llama_model_dir = "/kaggle/input/kachallenges-math-problem-classification-models/ensemble/llama_1b_model"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = llama_model_dir,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)


FastLanguageModel.for_inference(model)

test_set = test_df.copy()
test_set["instruction"] = "Classify this math problem into one of these eight topics: Algebra, Geometry and Trigonometry, Calculus and Analysis, Probability and Statistics, Number Theory, Combinatorics and Discrete Math, Linear Algebra, Abstract Algebra and Topology."
test_set.rename(columns = {"Question": "input"}, inplace=True)


prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


raw_outputs = []
for i in tqdm(range(len(test_set))):
  inputs = tokenizer(
  [
      prompt.format(
          test_set.iloc[0]["instruction"], # instruction
          test_set.iloc[i]["input"], # input
          "", # output - leave this blank for generation!
      )
  ], return_tensors = "pt").to("cuda")

  outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
  raw_outputs.append(tokenizer.batch_decode(outputs))


test_set["raw_outputs"] = [raw_output[0] for raw_output in raw_outputs]

def parse_output(output):
    re_match = re.search(r'### Response:\n(.*?)<\|end_of_text\|>', output, re.DOTALL)
    if re_match:
        response = re_match.group(1).strip()
        return response
    else:
        return ''

test_set["parsed_outputs"] = test_set["raw_outputs"].apply(parse_output)

llama_labels = test_set["parsed_outputs"].map(label2id).fillna(0).astype(int).tolist()
llama_labels[:10]


del model, tokenizer
clean_memory()


BATCH_SIZE_PER_DEVICE = 128
MAX_TARGET_LENGTH = 32
prefix = "Classify this math problem: "
t5_model_dir = "/kaggle/input/kachallenges-math-problem-classification-models/ensemble/t5-model"


print(f"\nLoading fine-tuned T5 model and tokenizer from {t5_model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(t5_model_dir)

device = 0
model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_dir).to(f"cuda:{device}")
model.eval()

print("Model and tokenizer reloaded successfully.")

classifier_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)


print("\nPredicting on the test set using pipeline...")

test_questions = test_df['Question'].tolist()
prefixed_test_questions = [prefix + q for q in test_questions]

pipeline_batch_size = BATCH_SIZE_PER_DEVICE * 8
raw_predictions = []
for i in tqdm(range(0, len(prefixed_test_questions), pipeline_batch_size)):
    batch = prefixed_test_questions[i:i + pipeline_batch_size]
    raw_predictions.extend(classifier_pipeline(batch, max_length=MAX_TARGET_LENGTH, clean_up_tokenization_spaces=True))

predicted_label_names = [pred['generated_text'].strip() for pred in raw_predictions]

print(f"\nNumber of predictions: {len(predicted_label_names)}")
print(predicted_label_names[:10])


cleaned_preds = predicted_label_names[:]

predicted_labels = []
unknown_count = 0
for pred_name in cleaned_preds:
    if pred_name in label2id:
        predicted_labels.append(label2id[pred_name])
    else:
        predicted_labels.append(0)
        unknown_count += 1
        print(f"Warning: Generated unknown label name '{pred_name}'. Assigned default 0.")

if unknown_count > 0:
     print(f"Total unknown labels generated: {unknown_count}")

t5_labels = predicted_labels[:]
t5_labels[:10]


del model, tokenizer
clean_memory()


MAX_LENGTH = 512
deberta_labels = []
EVAL_BATCH_SIZE = 128
deberta_model_dir = "/kaggle/input/kachallenges-math-problem-classification-models/ensemble/deberta-model"


def clean_math_text_final(text):
    
    text = str(text)
    text = re.sub(r'^\s*\d+\.\s*', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} to load deberta model")


tokenizer = AutoTokenizer.from_pretrained(deberta_model_dir)
print("Tokenizer loaded.")

model = AutoModelForSequenceClassification.from_pretrained(deberta_model_dir)
print("Model loaded.")
model.to(device)
print(f"Model moved to {device}.")

model.eval()


training_args = TrainingArguments(
    output_dir="./",
    push_to_hub=False,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    report_to="none",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer
)


comp_test_df = test_df.copy()

print("Cleaning test data...")
comp_test_df['cleaned_question'] = comp_test_df['Question'].apply(clean_math_text_final)
print("Cleaning complete.")

predict_dataset = Dataset.from_pandas(comp_test_df[['cleaned_question']])
print("Test data converted to Dataset format.")
print(predict_dataset)

def tokenize_for_predict(examples):
    return tokenizer(examples["cleaned_question"],
                     padding="max_length",
                     truncation=True,
                     max_length=MAX_LENGTH)

print("\n--- Tokenizing Competition Test Set ---")
tokenized_predict_dataset = predict_dataset.map(tokenize_for_predict, batched=True)

tokenized_predict_dataset = tokenized_predict_dataset.remove_columns(["cleaned_question"])
tokenized_predict_dataset.set_format("torch")
print("Tokenization complete.")

print("\n--- Making Predictions ---")
predictions_output = trainer.predict(tokenized_predict_dataset)

logits = predictions_output.predictions

predicted_labels = np.argmax(logits, axis=-1)
print("Predictions generated.")

deberta_labels = [i for i in predicted_labels]
deberta_labels[:10]


ensemble_preds = []

# Hard Voting (use deberta label if all the labels are different)
for p1, p2, p3 in zip(llama_labels, t5_labels, deberta_labels):
    if p1 == p2 or p1 == p3:
        ensemble_preds.append(p1)
    elif p2 == p3:
        ensemble_preds.append(p2)
    else:
        ensemble_preds.append(p3)

result = pd.DataFrame({'id': test_df["id"], 'label': ensemble_preds})
result.to_csv('submission.csv', index=False)


result.head()

