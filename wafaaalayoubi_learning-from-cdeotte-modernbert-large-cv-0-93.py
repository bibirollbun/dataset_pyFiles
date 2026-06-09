import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
model_name = "/kaggle/input/modernbert-large-cv938"
EPOCHS = 3

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


import os
os.listdir(model_name)
# Should contain files like tokenizer.json, vocab.txt, etc.



import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

# Import libraries: pandas & numpy for data handling, LabelEncoder to turn text labels into integers
le = LabelEncoder()

# Load the dataset from the given path
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

# Fill missing values in the Misconception column with the string "NA"
train.Misconception = train.Misconception.fillna('NA')

# Create a new column 'target' by combining Category and Misconception with a colon
train['target'] = train.Category + ":" + train.Misconception

# Encode the combined text labels into integer values for machine learning models
train['label'] = le.fit_transform(train['target'])

# Count how many unique target classes exist
n_classes = len(le.classes_)

# Print dataset shape and number of classes
print(f"Train shape: {train.shape} with {n_classes} target classes")

# Display the first few rows
train.head()


# Create a boolean mask where Category starts with "True"
# (splitting by '_' and checking the first part)
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'

# Select only those rows where the condition is True
correct = train.loc[idx].copy()

# Count how many times each (QuestionId, MC_Answer) pair appears
# Store this count in a new column 'c'
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')

# Sort so the most frequent answers come first
correct = correct.sort_values('c', ascending=False)

# Keep only one row per QuestionId (the most frequent answer survives after sorting)
correct = correct.drop_duplicates(['QuestionId'])

# Keep only QuestionId and MC_Answer columns
correct = correct[['QuestionId', 'MC_Answer']]

# Mark these (QuestionId, MC_Answer) pairs as correct answers
correct['is_correct'] = 1

# Merge this "correct answers" info back into the main train DataFrame
# If a row matches a correct (QuestionId, MC_Answer), it gets is_correct=1
train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')

# Fill all missing is_correct values with 0 (i.e., incorrect answers)
train.is_correct = train.is_correct.fillna(0)



from IPython.display import display, Math, Latex

# Group by QuestionId and MC_Answer to count how many times each answer was selected
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')

# Rank the answers per question by popularity (most selected = rank 0)
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1

# Drop the count column since rank is enough
tmp = tmp.drop('count', axis=1)

# Sort by QuestionId and rank so answers are ordered from most to least popular
tmp = tmp.sort_values(['QuestionId','rank'])

# Get all unique QuestionIds
Q = tmp.QuestionId.unique()

# Loop through each question to display question text and sorted MC answers
for q in Q:
    # Get the question text (taking the first occurrence for each QuestionId)
    question = train.loc[train.QuestionId == q].iloc[0].QuestionText
    
    # Get the MC answer choices in order of popularity
    choices = tmp.loc[tmp.QuestionId == q].MC_Answer.values
    
    # Assign labels A, B, C, D to answers
    labels = "ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    # Display question and answers nicely using LaTeX formatting
    print()
    display(Latex(f"QuestionId {q}: {question}"))
    display(Latex(f"MC Answers: {choice_str}"))


import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

# Load a tokenizer from a pretrained model (e.g., BERT or ModernBERT)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the maximum token length for sequences; longer sequences will be truncated
MAX_LEN = 256


# Define a function to format a single row into a prompt for the LLM
def format_input(row):
    # Default text if the answer is correct
    x = "This answer is correct."
    
    # If the answer is incorrect, update the text
    if not row['is_correct']:
        x = "This answer is incorrect."
    
    # Combine QuestionText, MC_Answer, correctness info, and StudentExplanation into a single string
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

# Apply the function to each row in the train DataFrame to create a 'text' column
train['text'] = train.apply(format_input, axis=1)

# Print an example formatted prompt
print("Example prompt for our LLM:\n")
print(train.text.values[0])



# Compute the token length of each text entry without truncation
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Plot a histogram of the token lengths
plt.hist(lengths, bins=50)  # 50 bins for distribution
plt.title("Token Length Distribution")  # Title of the plot
plt.xlabel("Number of tokens")         # X-axis label
plt.ylabel("Frequency")                 # Y-axis label
plt.grid(True)                          # Add grid lines for readability
plt.show()                              # Display the plot



# Count how many training samples exceed the maximum allowed token length (MAX_LEN)
L = (np.array(lengths) > MAX_LEN).sum()  
# np.array(lengths) > MAX_LEN creates a boolean array: True for samples longer than MAX_LEN
# .sum() counts the number of True values, i.e., the number of samples exceeding MAX_LEN

# Print the number of samples that are longer than MAX_LEN
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")

# Sort the list of token lengths in ascending order
# Useful to see the distribution of lengths or identify extremes
np.sort(lengths)  



# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes,
    reference_compile=False,
)


training_args = TrainingArguments(
    output_dir = f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps", #no for no saving 
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=16*2,
    per_device_eval_batch_size=32*2,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
)


# CUSTOM MAP@3 METRIC

from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

#trainer.train()


trainer.save_model(f"ver_{VER}")      
tokenizer.save_pretrained(f"ver_{VER}")


test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test.shape )
test.head()


test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)

test.head()


ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :3]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
joined_preds = [" ".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()

