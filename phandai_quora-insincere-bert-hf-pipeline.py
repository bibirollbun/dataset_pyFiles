import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.model_selection import KFold
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import torch

# Set seed for reproducibility
def set_seed(seed=42):
    """
    Sets the seed for reproducibility in NumPy, random, and PyTorch.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seed for reproducibility
set_seed(42)

print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")


# GENERAL CONFIG
eda = False
debug = False
num_labels = 2
id_fold = 0

# TOKENIZE & BATCH CONFIG
max_length = 160

per_device_train_batch_size=128
per_device_eval_batch_size=128


# MODEL CONFIG
model_name = "bert-base-uncased"
learning_rate=2e-5
num_train_epochs=3 if debug == False else 10
weight_decay=0.01
gradient_accumulation_steps=4
fp16=True
metric_for_best_model="f1"

# OTHERS
output_dir= f"./results/fold{id_fold}"  # Directory to save checkpoints and results
logging_dir="./logs"  # Directory to save logs
eval_strategy="epoch"
save_strategy="epoch"
logging_steps= 1000 if debug == False else 10
load_best_model_at_end=True
report_to="none"


train = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")
test_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
if debug:
    train = train[:2000]
    test_df = test_df[:1000]


train.head()


print("There are:",len(train),"samples")


train.nunique()


train["num_word"] = train["question_text"].apply(lambda x: len(str(x).split()))


# Configure plot
plt.figure(figsize=(16, 6))
ax = plt.gca()

# Create histogram bins for each possible word length
counts, bins, patches = plt.hist(train["num_word"], 
                                bins=np.arange(0, 140) - 0.5,  # Center bars on integer values
                                edgecolor='black', 
                                alpha=0.7,
                                color='#1f77b4')

# Formatting
plt.title('Distribution of Word Counts (0-160 words)', fontsize=14, pad=20)
plt.xlabel('Number of Words', fontsize=12)
plt.ylabel('Number of Texts', fontsize=12)
plt.xlim(0, 140)

# Add labels for every 5 words
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for i, (count, bin_center) in enumerate(zip(counts, bin_centers)):
    if bin_center % 5 == 0:  # Label every 5 units
        ax.text(bin_center, count + 500, f'{int(bin_center)}', 
                ha='center', va='bottom', rotation=90, fontsize=8)

# Add statistics box
stats_text = f"""Total Samples: {len(train):,}
Mean: {train['num_word'].mean():.1f}
Median: {train['num_word'].median()}
Max: {train['num_word'].max()}
95th %ile: {np.percentile(train['num_word'], 95)}"""
plt.gcf().text(0.92, 0.6, stats_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


tokenizer = AutoTokenizer.from_pretrained(model_name)


print(f"Tokenizer type: {type(tokenizer).__name__}")


if eda: # take time, so only run for eda
    train["token_length"] = train["question_text"].apply(
        lambda x: len(
            tokenizer.encode_plus(
                x,
                truncation=False,  # Do not truncate to ensure we capture the full length
                add_special_tokens=True  # Include [CLS] and [SEP] tokens
            )["input_ids"]
        )
    )


if eda:
    print("longest token length:",max(train["token_length"]))
    display(train[train["token_length"]>max_length])



# Function to create K-Folds
def create_folds(dataframe, n_splits=5, seed=42):
    set_seed(seed)  # Ensure reproducibility
    dataframe["fold"] = -1  # Initialize fold column with -1
    
    # Shuffle the data
    dataframe = dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Assign folds
    for fold, (_, val_idx) in enumerate(kf.split(X=dataframe)):
        dataframe.loc[val_idx, "fold"] = fold

    return dataframe


train = create_folds(train, n_splits=5, seed=42)

# Select fold 0 as validation set
val_df = train[train["fold"] == id_fold]
train_df = train[train["fold"] != id_fold]


model_name = model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def preprocess_function(examples):
    return tokenizer(examples['question_text'], truncation=True, max_length=max_length)  # No padding here

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

train_dataset = train_dataset.rename_column("target", "labels")
val_dataset = val_dataset.rename_column("target", "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir=output_dir,  # Directory to save checkpoints and results
    eval_strategy=eval_strategy,  # Evaluate at the end of each epoch
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    logging_dir=logging_dir,  # Directory to save logs
    logging_steps=logging_steps,
    save_strategy=save_strategy,
    gradient_accumulation_steps=gradient_accumulation_steps,
    load_best_model_at_end=load_best_model_at_end,  # Load best model at the end of training
    metric_for_best_model=metric_for_best_model,  # Use accuracy to select the best model
    report_to=report_to,  # Disable reporting to external services
    fp16=fp16
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary",zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Step 7: Train the Model
trainer.train()


# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)


predictions = trainer.predict(test_dataset)


predicted_labels = predictions.predictions.argmax(axis=-1)


# Extract 'qid' from the test DataFrame or Dataset
qids = test_df["qid"]  # Replace with the actual source of 'qid'

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    "qid": qids,
    "prediction": predicted_labels
})
submission_df.to_csv("submission.csv", index=False)

