# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





# Import necessary libraries
import numpy as np               # For numerical operations
import pandas as pd              # For data handling (like Excel in Python)
import matplotlib.pyplot as plt  # For plotting basic graphs
import seaborn as sns            # For making those graphs look good

# Set a visual style for our plots
sns.set_style('whitegrid')

# Define the file path. In Kaggle, it's just in the '../input/...' directory
DATA_PATH = '/kaggle/input/feedback-prize-english-language-learning/'

# Load the training data from the CSV file into a pandas DataFrame
df_train = pd.read_csv(DATA_PATH + 'train.csv')

# Load the test data (we won't use this until the very end)
df_test = pd.read_csv(DATA_PATH + 'test.csv')

print(f"Training data has {df_train.shape[0]} rows and {df_train.shape[1]} columns.")
print(f"Test data has {df_test.shape[0]} rows and {df_test.shape[1]} columns.")


# Display the first 5 rows of the training data
df_train.head()


# Define the 6 target columns we want to plot
target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

# Create a figure and a grid of subplots
# We need 6 plots, so we'll arrange them in a 2x3 grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# Flatten the axes array to make it easy to loop over
axes = axes.flatten()

# Loop over each target column and its corresponding subplot
for i, col in enumerate(target_cols):
    # Plot a histogram (displot) for the current column
    sns.histplot(df_train[col], ax=axes[i], kde=True)
    # Set the title of the subplot
    axes[i].set_title(f'Distribution of {col}')
    # Set the x-axis label
    axes[i].set_xlabel('Score')

# Adjust the layout to prevent titles from overlapping
plt.tight_layout()

# Display the plot
plt.show()


# Create a new column 'text_length' by splitting the text on spaces and counting the words
df_train['text_length'] = df_train['full_text'].apply(lambda x: len(x.split()))

# Now, let's get some basic statistics on the text length
print(df_train['text_length'].describe())

# Plot a histogram to see the distribution of text lengths
plt.figure(figsize=(12, 6))
sns.histplot(df_train['text_length'], bins=100, kde=True)
plt.title('Distribution of Essay Lengths (in words)')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.show()


# Install the transformers library
# The '!' tells the notebook to run this as a shell command
!pip install transformers

# Install tqdm for a progress bar
!pip install tqdm


from transformers import AutoTokenizer
from tqdm import tqdm

# Load the tokenizer for the model we plan to use.
# DeBERTa-v3 is a great, powerful choice.
MODEL_NAME = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create a new column 'token_length'
# We use tqdm to show a progress bar because this will take a moment.
tqdm.pandas()
df_train['token_length'] = df_train['full_text'].progress_apply(lambda x: len(tokenizer.encode(x)))


# Get the descriptive statistics for our new 'token_length' column
print(df_train['token_length'].describe())

# Plot a histogram to see the distribution of token lengths
plt.figure(figsize=(12, 6))
sns.histplot(df_train['token_length'], bins=100, kde=True)
plt.title('Distribution of Essay Lengths (in *Tokens*)')
plt.xlabel('Number of Tokens')
plt.ylabel('Count')
plt.show()


# Install the core deep learning libraries
!pip install torch
!pip install accelerate


# This 'Config' class will hold all our project settings
class Config:
    # --- Model and Tokenizer ---
    # We use the name from our EDA
    MODEL_NAME = 'microsoft/deberta-v3-base' 
    
    # --- Training ---
    # This is our key decision from the EDA
    MAX_LENGTH = 1024 
    
    # How many essays to process at once.
    # A smaller batch size uses less GPU memory. 
    # We'll start with 4.
    BATCH_SIZE = 4 
    
    # Number of times to train on the whole dataset
    EPOCHS = 3 
    
    # The "speed" at which the model learns
    LEARNING_RATE = 2e-5 
    
    # --- Environment ---
    # The 6 scores we are predicting
    TARGET_COLS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

# --- Print our configuration to confirm it's loaded ---
print("Configuration loaded:")
print(f"Model: {Config.MODEL_NAME}")
print(f"Max Length: {Config.MAX_LENGTH}")
print(f"Batch Size: {Config.BATCH_SIZE}")
print(f"Epochs: {Config.EPOCHS}")
print(f"Learning Rate: {Config.LEARNING_RATE}")


# Import the necessary PyTorch and Hugging Face libraries
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# 1. DEFINE THE DATASET CLASS
class FeedbackDataset(Dataset):
    
    # This runs when we first create the dataset
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Get the essay text
        self.texts = df['full_text'].values
        # Get the 6 scores and convert them to a list of lists
        self.labels = df[Config.TARGET_COLS].values

    # This just tells PyTorch how many items are in the dataset
    def __len__(self):
        return len(self.df)

    # This is the most important part.
    # It gets one item (essay) from the dataset.
    def __getitem__(self, idx):
        # Get the essay text at the 'idx' position
        text = self.texts[idx]
        
        # Get the corresponding 6 scores
        labels = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,    # Add [CLS] and [SEP]
            max_length=self.max_len,    # Pad or truncate to 1024
            padding='max_length',       # Pad to 1024
            truncation=True,            # Truncate to 1024
            return_tensors='pt'         # Return PyTorch tensors
        )
        
        # The tokenizer output is a dictionary. We need to "squeeze"
        # the tensors to remove the batch dimension, as PyTorch
        # will add it back later.
        
        # Return a dictionary of the processed data
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


# 2. CREATE THE TOKENIZER
# We need to re-load the tokenizer in this scope
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

# 3. CREATE AN INSTANCE OF THE DATASET
# This creates our "factory"
train_dataset = FeedbackDataset(
    df=df_train,
    tokenizer=tokenizer,
    max_len=Config.MAX_LENGTH
)

# 4. TEST IT
# Let's get the first item (index 0) to see if it works.
sample_item = train_dataset[0]

print("Successfully created dataset.")
print("\n--- Sample Item (from index 0) ---")
print(f"Input IDs shape: {sample_item['input_ids'].shape}")
print(f"Attention Mask shape: {sample_item['attention_mask'].shape}")
print(f"Labels: {sample_item['labels']}")


from transformers import AutoModelForSequenceClassification

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    Config.MODEL_NAME,
    num_labels=len(Config.TARGET_COLS) # Tell it we are predicting 6 labels
)


# We need to define the exact loss function from the competition:
# Mean Columnwise Root Mean Squared Error (MCRMSE)

# We can define this as a new class that inherits from torch.nn.Module
class MCRMSELoss(torch.nn.Module):
    def __init__(self):
        super(MCRMSELoss, self).__init__()
        # We'll use a standard MSELoss, but apply it column-wise
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        # y_pred and y_true will have shape (batch_size, 6)
        
        # Calculate MSE for each column (each of the 6 scores)
        col_mse = torch.mean(torch.pow(y_pred - y_true, 2), dim=0)
        
        # Calculate RMSE for each column
        col_rmse = torch.sqrt(col_mse)
        
        # Calculate the mean of the column-wise RMSEs
        loss = torch.mean(col_rmse)
        
        return loss

# Create an instance of our loss function
mcrmse_loss = MCRMSELoss()

print("Model and MCRMSE Loss function loaded successfully.")


from sklearn.model_selection import KFold

# 1. DEFINE THE K-FOLD SPLITTER
N_SPLITS = 5  # We'll use 5 folds
# We use KFold, shuffle the data, and set a random_state for reproducibility
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 2. CREATE A NEW 'fold' COLUMN
# Initialize a new column in our dataframe, filling it with -1
df_train['fold'] = -1

# 3. ASSIGN FOLDS
# kf.split() gives us indices for the train and validation sets
# We will loop through and assign a fold number to the validation indices
for fold_num, (train_indices, val_indices) in enumerate(kf.split(df_train)):
    # For each fold, we set the 'fold' column for the validation
    # rows to the current fold number (0, 1, 2, 3, or 4)
    df_train.loc[val_indices, 'fold'] = fold_num

print(f"Successfully created {N_SPLITS} folds.")

# Let's check the distribution. Each fold should have ~782 rows.
print(df_train['fold'].value_counts())

# You can also check the 'head' of the dataframe to see the new column
print("\nDataFrame head with 'fold' column:")
print(df_train.head())


# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from torch.optim import AdamW
# from tqdm import tqdm
# import os
# import copy

# # --- 1. Define the Device ---
# # Set the device to 'cuda' (the GPU) if available, else 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Create a directory to save our models
# os.makedirs('./models', exist_ok=True)

# # --- 2. Define the Training Function (for one epoch) ---
# def train_fn(model, data_loader, loss_fn, optimizer, device):
#     model.train() # Set the model to training mode
#     total_loss = 0
    
#     # Use tqdm for a progress bar
#     progress_bar = tqdm(data_loader, desc="Training")
    
#     for batch in progress_bar:
#         # Move batch to device
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
#         # Zero the gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         outputs = model(input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
        
#         # Calculate loss
#         loss = loss_fn(logits, labels)
        
#         # Backward pass
#         loss.backward()
        
#         # Update weights
#         optimizer.step()
        
#         # Update total loss
#         total_loss += loss.item()
        
#         # Update progress bar
#         progress_bar.set_postfix(loss=total_loss/len(progress_bar))
        
#     # Return the average loss for the epoch
#     return total_loss / len(data_loader)

# # --- 3. Define the Validation Function (for one epoch) ---
# def valid_fn(model, data_loader, loss_fn, device):
#     model.eval() # Set the model to evaluation mode
#     total_loss = 0
    
#     # Use tqdm for a progress bar
#     progress_bar = tqdm(data_loader, desc="Validation")
    
#     # We don't need to calculate gradients for validation
#     with torch.no_grad():
#         for batch in progress_bar:
#             # Move batch to device
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
            
#             # Forward pass
#             outputs = model(input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
            
#             # Calculate loss
#             loss = loss_fn(logits, labels)
            
#             # Update total loss
#             total_loss += loss.item()
            
#             # Update progress bar
#             progress_bar.set_postfix(loss=total_loss/len(progress_bar))
            
#     # Return the average loss for the epoch
#     return total_loss / len(data_loader)

# # --- 4. The Main 5-Fold Training Loop ---
# print("\n--- Starting 5-Fold Cross-Validation ---")

# # We'll store the best loss from each fold
# fold_scores = []
# N_SPLITS = 5

# # Loop from fold 0 to 4
# for fold in range(1):
#     print(f"\n======== Fold {fold+1} / {N_SPLITS} ========")
    
#     # --- a. Load a fresh tokenizer and model ---
#     # This is critical! We must start fresh for each fold.
#     tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         Config.MODEL_NAME,
#         num_labels=len(Config.TARGET_COLS)
#     ).to(device) # Move model to the GPU
    
#     # --- b. Define the optimizer ---
#     # We use AdamW, a standard optimizer for Transformers
#     optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
#     # --- c. Define the loss function ---
#     loss_fn = MCRMSELoss().to(device)
    
#     # --- d. Create DataLoaders for this specific fold ---
#     print("Creating DataLoaders...")
    
#     # Training data is all folds *except* the current one
#     df_train_fold = df_train[df_train['fold'] != fold]
#     # Validation data is *only* the current fold
#     df_valid_fold = df_train[df_train['fold'] == fold]

#     train_dataset = FeedbackDataset(df_train_fold, tokenizer, Config.MAX_LENGTH)
#     valid_dataset = FeedbackDataset(df_valid_fold, tokenizer, Config.MAX_LENGTH)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=2,
#         pin_memory=True
#     )
    
#     valid_loader = DataLoader(
#         valid_dataset,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True
#     )
    
#     # --- e. Run the training epochs ---
#     best_valid_loss = float('inf') # Initialize best loss to infinity
    
#     for epoch in range(Config.EPOCHS):
#         print(f"\n--- Epoch {epoch+1} / {Config.EPOCHS} ---")
        
#         # Train
#         train_loss = train_fn(model, train_loader, loss_fn, optimizer, device)
        
#         # Validate
#         valid_loss = valid_fn(model, valid_loader, loss_fn, device)
        
#         print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")
        
#         # --- f. Save the best model ---
#         if valid_loss < best_valid_loss:
#             best_valid_loss = valid_loss
#             # Save the model's state dictionary
#             torch.save(model.state_dict(), f'./models/best_model_fold_{fold}.pth')
#             print(f"Validation loss improved. Saved model to ./models/best_model_fold_{fold}.pth")
            
#     # After all epochs for this fold are done, store the best loss
#     fold_scores.append(best_valid_loss)
#     print(f"Fold {fold+1} finished. Best Valid Loss: {best_valid_loss:.4f}")

# # --- 5. Print Final Results ---
# print("\n--- 5-Fold Cross-Validation Finished ---")
# print(f"Scores for each fold: {fold_scores}")
# print(f"Average Valid Loss: {np.mean(fold_scores):.4f}")


# Select only the 6 target columns
target_df = df_train[Config.TARGET_COLS]

# Get the descriptive statistics for these columns
print(target_df.describe())


from sklearn.metrics import mean_squared_error
import numpy as np

# We need a function to compute our MCRMSE for the Trainer
def compute_metrics(eval_preds):
    """
    This function is called by the Trainer to compute metrics.
    'eval_preds' is a tuple: (predictions, labels)
    """
    # 1. Unpack the predictions and labels
    preds, labels = eval_preds
    
    # 2. Calculate column-wise RMSE
    # We first calculate the squared error for each column
    col_wise_mse = np.mean(np.square(labels - preds), axis=0)
    # Then take the square root to get RMSE
    col_wise_rmse = np.sqrt(col_wise_mse)
    
    # 3. Calculate the mean of the column-wise RMSEs
    mcrmse = np.mean(col_wise_rmse)
    
    # 4. Return it in a dictionary
    return {
        'mcrmse': mcrmse
    }

print("Helper function 'compute_metrics' defined.")


from transformers import TrainingArguments

# 1. Define the Training Arguments (Corrected)
training_args = TrainingArguments(
    # --- File Paths ---
    output_dir='./results',  # Where to save models and logs
    
    # --- Core Training ---
    num_train_epochs=Config.EPOCHS,          # 3 epochs (from our Config)
    learning_rate=Config.LEARNING_RATE,    # 2e-5 (from our Config)
    per_device_train_batch_size=Config.BATCH_SIZE, # 4 per GPU (from Config)
    per_device_eval_batch_size=Config.BATCH_SIZE,  # 4 per GPU (from Config)
    
    # --- Logging & Saving ---
    logging_steps=50,             # Log the training loss every 50 steps
    
    # --- THIS IS THE FIX ---
    # Renamed from 'evaluation_strategy' to 'eval_strategy'
    eval_strategy="epoch",  
    
    save_strategy="epoch",        # Save a checkpoint at the end of each epoch
    save_total_limit=1,           # Only keep the *best* model
    
    # --- Performance ---
    fp16=True,                   # Use mixed-precision (T4 GPU turbo!)
    
    # --- Metrics ---
    metric_for_best_model="mcrmse", # The metric to watch for saving the best model
    greater_is_better=False,      # For this metric, lower is better
    load_best_model_at_end=True,  # Load the best model at the end of training
    report_to="none"              # Don't log to external services like W&B
)

print("TrainingArguments defined successfully. (Corrected)")


from transformers import Trainer

# We must re-define our loss function so this class can see it
class MCRMSELoss(torch.nn.Module):
    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        col_mse = torch.mean(torch.pow(y_pred - y_true, 2), dim=0)
        col_rmse = torch.sqrt(col_mse)
        loss = torch.mean(col_rmse)
        return loss

# This is the new class that forces the Trainer to use our MCRMSELoss
class CustomTrainer(Trainer):
    # This overrides the default compute_loss method
    
    # --- THIS IS THE FIX ---
    # We add **kwargs to accept any extra arguments
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        # Get the labels from the inputs
        labels = inputs.get("labels")
        
        # Get the model's outputs
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Define our loss function
        loss_fn = MCRMSELoss()
        
        # Calculate the loss using our custom function
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

print("CustomTrainer defined successfully. (Corrected)")


import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from torch.optim import AdamW
from tqdm import tqdm
import os
import copy
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_squared_error

# IMPORTANT: Assuming Config class and CustomTrainer class are defined in previous cells!

# --- GLOBAL STORAGE FOR ANALYSIS ---
# We will collect the evaluation results here for later plotting
ALL_EVAL_HISTORY = defaultdict(list)
ALL_EVAL_RESULTS = [] # To store the full evaluation dictionaries for the per-trait table

# --- 1. Define the Device and Create Directories ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs('./models_trainer', exist_ok=True) # New folder for these models
os.makedirs('./results', exist_ok=True) # Ensure results directory exists

# --- 2. The Main 5-Fold Training Loop ---
print("\n--- Starting 5-Fold Cross-Validation (with Trainer) ---")

fold_scores = []
N_SPLITS = 5

# Loop from fold 0 to 4
for fold in range(N_SPLITS):
    print(f"\n======== Fold {fold+1} / {N_SPLITS} ========")
    
    # --- a. Load a fresh model and tokenizer ---
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.TARGET_COLS)
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # --- b. Create DataLoaders for this specific fold ---
    print("Creating Datasets for fold...")
    df_train_fold = df_train[df_train['fold'] != fold]
    df_valid_fold = df_train[df_train['fold'] == fold]
    df_train_fold = df_train_fold.reset_index(drop=True)
    df_valid_fold = df_valid_fold.reset_index(drop=True)

    train_dataset = FeedbackDataset(df_train_fold, tokenizer, Config.MAX_LENGTH)
    valid_dataset = FeedbackDataset(df_valid_fold, tokenizer, Config.MAX_LENGTH)

    # --- c. Initialize the Trainer ---
    # We set a new output directory for each fold
    training_args.output_dir = f'./results/fold_{fold}'
    
    trainer = CustomTrainer(
        model=model,                      
        args=training_args,               
        train_dataset=train_dataset,      
        eval_dataset=valid_dataset,       
        compute_metrics=compute_metrics,  
        tokenizer=tokenizer               
    )

    # --- d. Run the training ---
    print(f"Starting training for fold {fold+1}...")
    trainer.train()
    
    # --- e. Evaluate and Collect Log History (THE CRITICAL STEP) ---
    print(f"Evaluating best model for fold {fold+1}...")
    
    # The Trainer loads the best model automatically due to load_best_model_at_end=True
    eval_results = trainer.evaluate()
    
    # 1. Collect fold scores
    fold_mcrmse = eval_results['eval_mcrmse']
    fold_scores.append(fold_mcrmse)
    
    # 2. Collect full evaluation results (for per-trait analysis)
    ALL_EVAL_RESULTS.append(eval_results)
    
    # 3. Collect Epoch-by-Epoch Validation Scores for Plotting (THE USER'S REQUEST)
    # The trainer.state.log_history contains all the logged metrics (loss, eval_mcrmse)
    log_history = trainer.state.log_history
    fold_eval_scores = [log['eval_mcrmse'] for log in log_history if 'eval_mcrmse' in log]
    ALL_EVAL_HISTORY[fold] = fold_eval_scores
    
    print(f"Fold {fold+1} finished. Best MCRMSE: {fold_mcrmse:.4f}")
    
    # --- f. Save the best model ---
    trainer.save_model(f'./models_trainer/best_model_fold_{fold}')
    
# --- 5. Print Final Results ---
print("\n--- 5-Fold Cross-Validation Finished ---")
final_avg_mcrmse = np.mean(fold_scores)
final_std_mcrmse = np.std(fold_scores)

print(f"Average MCRMSE: {final_avg_mcrmse:.4f} +/- {final_std_mcrmse:.4f}")

# =========================================================================
# === FINAL ANALYSIS & PLOTTING SCRIPTS ===================================
# =========================================================================

# Convert the history dictionary to a DataFrame for easy plotting
# We will use the smallest number of epochs across all folds, which is 3.
min_epochs = min(len(v) for v in ALL_EVAL_HISTORY.values()) if ALL_EVAL_HISTORY else 0
df_history = pd.DataFrame({
    f'Fold {k+1}': v[:min_epochs] for k, v in ALL_EVAL_HISTORY.items()
})
epochs = range(1, min_epochs + 1)
baseline_avg = 0.5595 # The score from your TF-IDF run


# --- A. PLOT THE PROFESSOR'S CONVERGENCE CURVE (The Decaying Curve) ---
print("\n--- Generating Training Convergence Plot ---")

plt.figure(figsize=(12, 7))

# Plot individual folds (Faint lines)
for col in df_history.columns:
    plt.plot(epochs, df_history[col], marker='o', linestyle='--', alpha=0.3, color='blue', 
             label='_nolegend_' if col != 'Fold 1' else 'Individual Folds')

# Calculate and plot the Average Performance line (The official result)
avg_scores_epoch = df_history.mean(axis=1)

# Plot the thick average line
plt.plot(epochs, avg_scores_epoch, marker='D', linewidth=4, color='red', 
         label=f'Average MCRMSE ({final_avg_mcrmse:.4f})')

# Add Baseline comparison line
plt.axhline(y=baseline_avg, color='gray', linestyle='-', linewidth=2, alpha=0.7, 
             label=f'TF-IDF Baseline ({baseline_avg:.4f})')

plt.title("DeBERTa Model Training Convergence (5-Fold Cross-Validation)")
plt.xlabel("Training Epoch")
plt.ylabel("Validation MCRMSE Error (Lower is Better)")
plt.xticks(epochs)
plt.ylim(0.50, 0.56) 
plt.grid(True)
plt.legend()
plt.savefig('deberta_training_convergence_plot.png')
plt.show()

# --- B. CREATE THE FINAL COMPARISON TABLE ---
print("\n--- Final Project Performance Summary Table ---")

improvement = (baseline_avg - final_avg_mcrmse) / baseline_avg * 100

final_summary = pd.DataFrame({
    'Metric': ['MCRMSE (Average)', 'MCRMSE (Std Dev)', 'Improvement Over Baseline (%)'],
    'TF-IDF Baseline': [f'{baseline_avg:.4f}', '-', '-'],
    'DeBERTa V3 (5-Fold)': [f'{final_avg_mcrmse:.4f}', f'{final_std_mcrmse:.4f}', f'{improvement:.2f}%']
})
final_summary = final_summary.set_index('Metric')

print(final_summary.to_markdown())

# --- C. CREATE THE PER-TRAIT ERROR TABLE ---
# This is a key table for the report, showing the breakdown of the score.

print("\n--- Per-Trait Error Analysis Table ---")

# 1. Collect all per-trait RMSEs from the evaluation results
all_per_trait = []
for result in ALL_EVAL_RESULTS:
    # Extract the 6 individual RMSE scores
    trait_rmses = [
        result[f'eval_{trait}_rmse'] 
        for trait in Config.TARGET_COLS
    ]
    all_per_trait.append(trait_rmses)

# 2. Calculate the average RMSE for each of the 6 traits across all 5 folds
avg_per_trait = np.mean(all_per_trait, axis=0)
std_per_trait = np.std(all_per_trait, axis=0)

# 3. Create the DataFrame
per_trait_df = pd.DataFrame({
    'Metric': Config.TARGET_COLS,
    'Average RMSE': [f'{r:.4f}' for r in avg_per_trait],
    'Std Dev RMSE': [f'{s:.4f}' for s in std_per_trait]
})

print(per_trait_df.to_markdown())

