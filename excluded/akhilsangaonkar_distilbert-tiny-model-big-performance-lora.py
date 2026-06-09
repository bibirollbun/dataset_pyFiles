# Turn off warnings
import warnings
warnings.filterwarnings("ignore")

# Data Handling and Splitting
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Evaluation metrics
from sklearn.metrics import classification_report, f1_score

# Visualization
import plotly.express as px

# Hugging Face Datasets and Transformers
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# Utilities
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# PEFT (Parameter-Efficient Fine-Tuning)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training



if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")



# Load training and test datasets from CSV files
train_df_org = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")
train_df_aug = pd.read_csv("/kaggle/input/math-problem-classification-data/train_augmented.csv")

# Display dataset shapes for sanity check
print("âœ… Original training data shape:", train_df_org.shape)
print("âœ… Augmented data shape:", train_df_aug.shape)
print("âœ… Test data shape:", test_df.shape)

# Merge training data
train_df = pd.concat([train_df_org, train_df_aug], ignore_index=True)





train_df['Question'] = train_df['Question'].astype(str)
test_df['Question'] = test_df['Question'].astype(str)


num_classes = train_df['label'].nunique()
print(f"Number of unique classes: {num_classes}")



# Mapping of label IDs to topic names
label_map = {
    0: 'Algebra',
    1: 'Geometry and Trigonometry',
    2: 'Calculus and Analysis',
    3: 'Probability and Statistics',
    4: 'Number Theory',
    5: 'Combinatorics and Discrete Math',
    6: 'Linear Algebra',
    7: 'Abstract Algebra and Topology'
}

# Create a new DataFrame for plotting
plot_df = train_df['label'].map(label_map).value_counts().reset_index()
plot_df.columns = ['Topic', 'Count']
plot_df = plot_df.sort_values(by='Count', ascending=False)

custom_colors = [
    "#281dff", "#783afb", "#a158f8", "#bf76f7",
    "#d696f6", "#e8b6f7", "#f6d7fa", "#fff9ff"
]

# Plot with Plotly
fig = px.bar(
    plot_df,
    x='Topic',
    y='Count',
    title='Distribution by Math Topics',
    color='Topic',
    color_discrete_sequence=custom_colors,
    template='plotly_dark',
    width=900, 
    height=500
)

fig.update_layout(
    xaxis_title="Topics",
    yaxis_title="Count",
    xaxis_tickangle=-45
)

fig.show()



# Split the original training data into training and validation sets (80-20 split)
# Stratified by label to preserve class distribution
train_df, eval_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42  # For reproducibility
)



# Reset index after train-validation split to clean up the DataFrames
train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)



# Convert pandas DataFrames into Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Create a DatasetDict for use with Hugging Face Trainer
dataset = DatasetDict({
    "train": train_dataset,
    "eval": eval_dataset
})



# Define the model checkpoint
model_checkpoint = "distilbert-base-uncased"

# Load the tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    


# Load the model with a classification head for 8 classes
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_classes
)



# Apply tokenization to both train and validation sets
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["Question"], 
                        truncation=True,
                         ),
    batched=True,
    remove_columns=["Question"]  # Drop original Question column
)



# Initialize a data collator that dynamically pads the input sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# Function to compute evaluation metrics (Validation F1 micro score)
def compute_metrics(eval_preds):
    
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # Compute F1
    f1score = f1_score(labels, predictions, average='micro')
    
    return {
        "Validation f1_micro": f1score,
    }



# Define LoRA Parameters
LORA_R = 256  # Dimension of the low-rank matrices
LORA_ALPHA = 512  # Scaling factor for the weight matrices
LORA_DROPOUT = 0.05  # Dropout probability for the LoRA layers

# Define LoRA Config
lora_config = LoraConfig(
    r=LORA_R,  # Low-rank dimension
    lora_alpha=LORA_ALPHA,  # Scaling factor for the weight matrices
    lora_dropout=LORA_DROPOUT,  # Dropout for the LoRA layers
    bias="none",  # No bias term for LoRA layers
    task_type="SEQ_CLS",  # Sequence Classification task
    target_modules=["q_lin", "k_lin", "v_lin"],  # Target layers for LoRA in DistilBERT
)

# Prepare model for int-8 quantization training using PEFT
model = prepare_model_for_kbit_training(model)

# Initialize the model with the LoRA framework
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()



# Define the training arguments
training_args = TrainingArguments(
    output_dir="./distilbert",  # Directory to save model checkpoints
    report_to="none",
    per_device_train_batch_size=32,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    fp16=True,           # 16-bit full point precision
    learning_rate=3e-4,  # Learning rate for the optimizer
    num_train_epochs=10,  # Number of epochs for training
    seed=42,              # Random seed for reproducibility
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save checkpoint at the end of each epoch
    save_total_limit=12,  # Maximum number of checkpoints to save
    load_best_model_at_end=False,  # Do not load the best model at the end
    label_names=["labels"]  # Label names for evaluation metrics
)



# Configure the Trainer with the model, training arguments, and datasets
trainer = Trainer(
    model=model,  # LoRA-enhanced model
    args=training_args,  # Hyperparameters for training
    train_dataset=tokenized_dataset['train'],  # The training dataset
    eval_dataset=tokenized_dataset['eval'],  # The validation dataset
    processing_class=tokenizer,  # Tokenizer used for encoding the text data
    data_collator=data_collator,  # Data collator to handle dynamic padding
    compute_metrics=compute_metrics  # Function to compute evaluation metrics F1 micro score
)



# Start the training process
trainer.train()



log_df = pd.DataFrame(trainer.state.log_history)
# Filter rows that have 'epoch' in them 
log_df = log_df[log_df['epoch'].notnull()]

# Keep only relevant columns 
columns_of_interest = ['epoch', 'loss', 'eval_loss', 'eval_Validation f1_micro']
log_df_cleaned = log_df[[col for col in columns_of_interest if col in log_df.columns]]



# Step 1: Forward-fill missing training loss values
log_df_cleaned['loss'] = log_df_cleaned['loss'].fillna(method='ffill')

# Step 2: Drop any remaining rows that contain NaN values in any column
log_df_cleaned = log_df_cleaned.dropna().reset_index(drop=True)


# Rename columns for better legend readability
metrics_plot_df = log_df_cleaned.rename(columns={
    'loss': 'Training Loss',
    'eval_loss': 'Validation Loss',
    'eval_Validation f1_micro': 'Validation F1 (Micro)'
})

# Reshape the DataFrame to long format for Plotly
metrics_plot_df_melted = metrics_plot_df.melt(
    id_vars='epoch',
    var_name='Metric',
    value_name='Value'
)

# Remove any remaining missing values
metrics_plot_df_melted_cleaned = metrics_plot_df_melted.dropna()

# Custom colors for each metric line
color_map = {
    'Training Loss': '#a2a2a2',       
    'Validation Loss': '#a2a2a2',      
    'Validation F1 (Micro)': '#27ff01'
}

# Create a line plot 
fig = px.line(
    metrics_plot_df_melted_cleaned,
    x='epoch',
    y='Value',
    color='Metric',
    color_discrete_map=color_map,
    title='ğŸ“ˆ Epoch-wise Training & Validation Metrics',
    width=900, 
    height=500
)

# Apply dark theme and customize layout
fig.update_layout(
    template='plotly_dark',
    xaxis_title='Epoch',
    yaxis_title='Metric Value',
    legend_title='Metric',
    font=dict(size=14)
)

fig.update_traces(selector=dict(name='Training Loss'), line=dict(dash='solid'))  # Solid line for Training Loss
fig.update_traces(selector=dict(name='Validation Loss'), line=dict(dash='solid'))  # Dashed line for Validation Loss
fig.update_traces(selector=dict(name='Validation F1 (Micro)'), line=dict(dash='dashdot'))  # Dash-dot line for Validation F1 (Micro)

# Show the plot
fig.show()



# Save the trained model
model.save_pretrained("./distilbert_final_model")

# Save the tokenizer
tokenizer.save_pretrained("./distilbert_final_model")



# Specify the paths to the saved model and tokenizer
saved_model_path = "./distilbert_final_model"
saved_tokenizer_path = "./distilbert_final_model"

# Load the saved model
final_model = AutoModelForSequenceClassification.from_pretrained(saved_model_path, num_labels=num_classes)

# Load the saved tokenizer
final_tokenizer = AutoTokenizer.from_pretrained(saved_tokenizer_path)


if final_tokenizer.pad_token is None:
    final_tokenizer.pad_token = final_tokenizer.eos_token



def evaluate_or_predict(model, dataloader, device, is_evaluation=True):
    """
    Function to evaluate the model or make predictions depending on the `is_evaluation` flag.

    Args:
        model: The model to evaluate or make predictions with.
        dataloader: The dataloader (evaluation or test) containing the data.
        device: The device on which the model is loaded (CPU or GPU or mps).
        is_evaluation (bool): Whether to evaluate or just predict (default is True for evaluation).

    Returns:
        If `is_evaluation` is True, returns the classification report.
        If `is_evaluation` is False, returns the predictions for the final test data.
    """
    all_preds = []
    all_labels = []

    model.to(device)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())

            # Only append labels if available (evaluation mode)
            if is_evaluation and 'labels' in batch:
                labels = batch['labels'].to(device)
                all_labels.extend(labels.cpu().tolist())

    if is_evaluation:
        return print(classification_report(all_labels, all_preds, digits=4))
    else:
        return all_preds



# Define the data collator for padding 
eval_data_collator = DataCollatorWithPadding(tokenizer=final_tokenizer)

# Create validation dataloader
eval_dataloader = DataLoader(
    tokenized_dataset['eval'],  
    batch_size=8,
    collate_fn=eval_data_collator
)



evaluate_or_predict(final_model, eval_dataloader, device, is_evaluation=True)



test_df = Dataset.from_pandas(test_df)

# Tokenize the test dataset by applying the tokenizer to the 'Question' column
tokenized_test_dataset = test_df.map(
    lambda x: final_tokenizer(x["Question"], truncation=True),
    batched=True,
    remove_columns=['Question','id']
)



# Define the data collator for padding for test data
test_data_collator = DataCollatorWithPadding(tokenizer=final_tokenizer)

# Create test dataloader
test_dataloader = DataLoader(
    tokenized_test_dataset,  
    batch_size=8,
    collate_fn=test_data_collator
)



test_predictions = evaluate_or_predict(final_model, test_dataloader, device, is_evaluation=False)



def save_predictions(predictions, sample_submission_path, output_path):
    """
    Save model predictions to a CSV file using the sample submission format.
    
    Args:
        predictions (list): List of predicted labels.
        sample_submission_path (str): Path to the sample submission CSV file.
        output_path (str): Path to save the final submission CSV file.
    """
    submission = pd.read_csv(sample_submission_path)
    submission['label'] = predictions
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")



save_predictions(test_predictions, '/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv', 'submission.csv')


