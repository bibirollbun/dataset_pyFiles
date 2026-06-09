!pip install evaluate -q


import os
import torch
import evaluate
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


MODEL_CHECKPOINT = "t5-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 32
BATCH_SIZE_PER_DEVICE = 8
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "/kaggle/working/t5-math-classifier"
TRAIN_FILE = "/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv"
TEST_FILE = "/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv"
SUBMISSION_FILE = "/kaggle/working/submission.csv"

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
NUM_LABELS = len(id2label)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using model: {MODEL_CHECKPOINT}")
print(f"Number of labels: {NUM_LABELS}")
print(f"Labels: {id2label}")


train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
display(train_df.head())
print(f"Train data shape: {train_df.shape}")
display(test_df.head())
print(f"Test data shape: {test_df.shape}")


train_df['label_name'] = train_df['label'].map(id2label)
print("\nTrain data with label names:")
print(train_df.head())

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
print(f"\nTrain split shape: {train_df.shape}")
print(f"Validation split shape: {val_df.shape}")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

raw_datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

print("\nDatasetDict created:")
print(raw_datasets)


tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

prefix = "Classify this math problem: "

def preprocess_function(examples):
    """Preprocesses the data for T5: adds prefix, tokenizes inputs and labels."""
    inputs = [prefix + doc for doc in examples["Question"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding=False)

    if "label_name" in examples:
        labels = tokenizer(text_target=examples["label_name"], max_length=MAX_TARGET_LENGTH, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]

    return model_inputs


train_val_cols_to_remove = raw_datasets["train"].column_names
test_cols_to_remove = raw_datasets["test"].column_names

tokenized_train = raw_datasets['train'].map(
    preprocess_function,
    batched=True,
    remove_columns=train_val_cols_to_remove
)
tokenized_val = raw_datasets['validation'].map(
    preprocess_function,
    batched=True,
    remove_columns=train_val_cols_to_remove
)

tokenized_test = raw_datasets['test'].map(
    preprocess_function,
    batched=True,
    remove_columns=test_cols_to_remove
)


tokenized_datasets = DatasetDict({
    'train': tokenized_train,
    'validation': tokenized_val,
    'test': tokenized_test
})


print("\nTokenized datasets:")
print(tokenized_datasets)
print("\nExample tokenized input (train):")
print(tokenized_datasets['train'][0]['input_ids'])
print("\nDecoded example tokenized input (train):")
print(tokenizer.decode(tokenized_datasets['train'][0]['input_ids']))
print("\nExample tokenized label (train):")
print(tokenized_datasets['train'][0]['labels'])
print("\nDecoded example tokenized label (train):")
print(tokenizer.decode(tokenized_datasets['train'][0]['labels']))
print("\nExample tokenized input (test):")
print(tokenized_datasets['test'][0]['input_ids'])
print("\nDecoded example tokenized input (test):")
print(tokenizer.decode(tokenized_datasets['test'][0]['input_ids']))
print("\nColumns in tokenized test set:")
print(tokenized_datasets['test'].column_names)


model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True
)
print("\nData collator initialized.")

accuracy_metric = evaluate.load("accuracy")
print(f"Using label2id mapping in metrics: {label2id}")

def postprocess_text(preds, labels):
    """ Helper function to clean up generated text """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    """Computes accuracy score from model predictions."""
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    pred_ids = []
    label_ids = []
    unknown_preds_count = 0
    for pred_name, label_name in zip(decoded_preds, decoded_labels):
        pred_id = label2id.get(pred_name, -1)
        if pred_id == -1:
            unknown_preds_count += 1

        label_id = label2id.get(label_name, -2)
        if label_id == -2:
             print(f"Error: Could not map true label '{label_name}' to ID!")

        pred_ids.append(pred_id)
        label_ids.append(label_id)

    if unknown_preds_count > 0:
        print(f"Warning: Encountered {unknown_preds_count} predictions during evaluation that did not match known label names.")

    acc_result = accuracy_metric.compute(predictions=pred_ids, references=label_ids)

    return {"accuracy": acc_result["accuracy"]}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
    per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE * 2,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    predict_with_generate=True,
    fp16=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    save_total_limit=3
)

print("\nTraining arguments configured.")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nSeq2SeqTrainer initialized.")

print("\nStarting training...")
train_result = trainer.train()
print("Training finished.")


print("\nPlotting training & validation curves...")

# Access Logs

log_history = trainer.state.log_history
log_df = pd.DataFrame(log_history)
training_logs = log_df[log_df['loss'].notna() & log_df['eval_loss'].isna()].reset_index()
eval_logs = log_df[log_df['eval_loss'].notna()].reset_index()

# --- Plotting ---
# Training Accuracy is not logged by default with this trainer setup

plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(training_logs['step'], training_logs['loss'], label='Training Loss', alpha=0.8)
plt.plot(eval_logs['step'], eval_logs['eval_loss'], label='Validation Loss', marker='o', linestyle='--')
plt.title('Training vs. Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Add annotations
for i, row in eval_logs.iterrows():
   plt.text(row['step'], row['eval_loss'], f"{row['eval_loss']:.2f}", ha='center', va='bottom')

plt.subplot(1, 2, 2)
plt.plot(eval_logs['step'], eval_logs['eval_accuracy'], label='Validation Accuracy', marker='o', linestyle='--', color='green')
plt.title('Validation Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

for i, row in eval_logs.iterrows():
   plt.text(row['step'], row['eval_accuracy'], f"{row['eval_accuracy']:.3f}", ha='center', va='bottom')


plt.suptitle('Training and Validation Metrics')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
print(f"Model saved to {OUTPUT_DIR}")

print("\nEvaluating the best model on the validation set...")
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
print(f"Validation Metrics: {eval_metrics}")


import torch
import gc

print("\nCleaning up training objects...")

del model
del trainer

gc.collect()
torch.cuda.empty_cache()
print("Training objects deleted and CUDA cache cleared.")


print(f"\nLoading fine-tuned model and tokenizer from {OUTPUT_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

device = 0
model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR).to(f"cuda:{device}")
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

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': predicted_labels
})

print("\nSubmission DataFrame head:")
print(submission_df.head())

submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"\nSubmission file saved to {SUBMISSION_FILE}")

