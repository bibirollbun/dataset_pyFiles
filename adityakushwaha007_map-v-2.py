# --- Essential Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Display Options (for better viewing in the notebook) ---
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 150)

print("Libraries imported successfully.")

# --- Load Data ---
# All competition data on Kaggle is located in the '/kaggle/input/' directory.
# You might need to change 'map-charting-student-math-misunderstandings'
# if the folder name is different.
try:
    BASE_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/'
    train_df = pd.read_csv(BASE_PATH + 'train.csv')
    test_df = pd.read_csv(BASE_PATH + 'test.csv')
    # We load the sample submission for format reference, as you requested.
    sample_submission_df = pd.read_csv(BASE_PATH + 'sample_submission.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("ERROR: Data files not found. Please check the BASE_PATH variable.")
    # Create empty dataframes to avoid errors in subsequent cells
    train_df, test_df, sample_submission_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# --- Quick Inspection ---
# Let's verify the data has been loaded correctly.

print("\n--- Training Data ---")
print(f"Shape: {train_df.shape}")
print("Columns:", train_df.columns.tolist())
display(train_df.head())

print("\n--- Test Data ---")
print(f"Shape: {test_df.shape}")
print("Columns:", test_df.columns.tolist())
display(test_df.head())


# --- Step 2: Create and Analyze the Target Variable ---

print("--- Analyzing the Target Label ---")

# First, let's handle the NaN values in the 'Misconception' column.
# Based on the submission format, 'NA' is the appropriate placeholder.
train_df['Misconception'] = train_df['Misconception'].fillna('NA')

# Now, create the single 'target' column in the format 'Category:Misconception'
train_df['target'] = train_df['Category'] + ':' + train_df['Misconception']

# 1. Count the number of unique labels
num_unique_labels = train_df['target'].nunique()
print(f"\nSuccessfully created the 'target' column.")
print(f"There are {num_unique_labels} unique labels in the training data.")

# 2. Get the value counts to see the distribution
label_counts = train_df['target'].value_counts()

# 3. Display the Top 20 most common labels
print("\n--- Top 20 Most Common Labels ---")
display(label_counts.head(20))

# 4. Visualize the distribution of the Top 20 labels
print("\n--- Visualizing Label Distribution ---")
plt.figure(figsize=(12, 8))
sns.barplot(y=label_counts.head(20).index, x=label_counts.head(20).values, orient='h')
plt.title('Top 20 Most Frequent Misconceptions')
plt.xlabel('Frequency / Count')
plt.ylabel('Target Label (Category:Misconception)')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# --- Step 3: Create Validation Folds ---
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

print("--- Setting up Validation Strategy ---")

# LabelEncoder converts each unique string label into a number
label_encoder = LabelEncoder()
train_df['target_encoded'] = label_encoder.fit_transform(train_df['target'])

# Initialize StratifiedKFold
# n_splits=5 means we'll have 5 folds
# shuffle=True to randomize the data before splitting
# random_state for reproducibility
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create the 'fold' column and fill it with the fold number for each row
for fold_num, (train_index, val_index) in enumerate(skf.split(train_df, train_df['target_encoded'])):
    train_df.loc[val_index, 'fold'] = fold_num

# Convert fold column to integer
train_df['fold'] = train_df['fold'].astype(int)

# --- Verify the Stratification ---
print("\nVerifying the distribution in Fold 0 vs. Overall:")

# Get the top 5 labels' overall percentage
overall_dist = train_df['target'].value_counts(normalize=True).head(5) * 100

# Get the top 5 labels' percentage within fold 0
fold0_dist = train_df[train_df['fold'] == 0]['target'].value_counts(normalize=True).head(5) * 100

# Create a comparison dataframe
dist_comparison_df = pd.DataFrame({
    'Overall (%)': overall_dist,
    'Fold 0 (%)': fold0_dist
})

print("\nDistribution of Top 5 Labels:")
display(dist_comparison_df)

print("\nDataFrame with the new 'fold' column:")
display(train_df.head())


# --- CONSOLIDATED SETUP CELL ---

# --- Imports ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
import gc
from tqdm.notebook import tqdm
import collections
import os

print("All libraries imported.")

# --- Helper Function: MAP@3 Metric ---
def map_at_3(true_labels, pred_probs):
    avg_precisions = []
    for i in range(len(true_labels)):
        true_label = true_labels[i]
        pred_prob = pred_probs[i]
        top_3_preds = np.argsort(pred_prob)[::-1][:3]
        precision = 0.0
        if true_label in top_3_preds:
            rank = np.where(top_3_preds == true_label)[0][0] + 1
            precision = 1.0 / rank
        avg_precisions.append(precision)
    return np.mean(avg_precisions)

print("MAP@3 function defined.")

# --- Configuration Class ---
class CFG:
    # --- !!! YOU MUST EDIT THIS LINE !!! ---
    # --- Use the "Copy file path" button in the Kaggle sidebar to get the correct path ---
    model_path = '/kaggle/input/roberta-large-for-map/roberta-large'
    
    batch_size = 8      # Reduced for the large model
    num_epochs = 3      # Increased for better training
    max_len = 128
    learning_rate = 1e-5
    num_classes = train_df['target_encoded'].nunique()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {CFG.device}")

# --- Tokenizer ---
# This will now use the correct path you provided above
tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
print("Tokenizer loaded.")

# --- PyTorch Dataset Class ---
class MisconceptionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, with_labels=True):
        self.texts = df['StudentExplanation'].values
        self.questions = df['QuestionText'].values # For combined input
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.with_labels = with_labels
        if self.with_labels:
            self.labels = df['target_encoded'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        question_text = self.questions[idx]
        answer_text = self.texts[idx]
        
        # Combine the question and the answer for full context
        text = f"{question_text} [SEP] {answer_text}"
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        
        if self.with_labels:
            return {
                'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
            }

print("MisconceptionDataset class defined.")

# --- Model Architecture ---
class CustomModel(nn.Module):
    def __init__(self, model_path, num_classes):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token_output = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token_output)
        return logits

print("CustomModel class defined.")
print("\n--- Setup Complete ---")


import os # <-- ADDED FOR FILE OPERATIONS

# --- FINAL SUBMISSION SCRIPT (Robust Version) ---

# All your previous class/function definitions should be in cells above this:
# CFG, MisconceptionDataset, CustomModel, tokenizer, label_encoder, train_df, test_df etc.

print("--- Starting Final Submission Pipeline (Robust Version) ---")

# Create a directory to save predictions
PREDS_PATH = "roberta_preds"
os.makedirs(PREDS_PATH, exist_ok=True) # <-- ADDED

# Create test dataloader
test_dataset = MisconceptionDataset(test_df, tokenizer, CFG.max_len, with_labels=False)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size * 2, shuffle=False, num_workers=2)

# Loop through each fold for training and prediction
for fold in range(5):
    print(f"\n======================")
    print(f"====== FOLD {fold} ======")
    print(f"======================")

    # --- Data Preparation for this fold ---
    train_set_df = train_df[train_df['fold'] != fold].reset_index(drop=True)
    train_dataset = MisconceptionDataset(train_set_df, tokenizer, CFG.max_len, with_labels=True)
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    
    # --- Model, Optimizer, Loss ---
    model = CustomModel(CFG.model_path, CFG.num_classes).to(CFG.device)
    optimizer = AdamW(model.parameters(), lr=CFG.learning_rate)
    num_training_steps = len(train_loader) * CFG.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    loss_fn = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    model.train()
    for epoch in range(CFG.num_epochs):
        for batch in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1} Training"):
            # Move data to GPU and perform one training step
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            labels = batch['labels'].to(CFG.device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    # --- Inference on Test Set for this fold ---
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold {fold} Predicting on Test"):
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            fold_preds.append(probabilities)
            
    # --- Save this fold's predictions to a file ---
    fold_preds_array = np.concatenate(fold_preds)
    np.save(f'{PREDS_PATH}/fold_{fold}_preds.npy', fold_preds_array) # <-- ADDED
    print(f"Fold {fold} predictions saved to file.")
    
    # Clean up memory
    del model, train_loader, optimizer, scheduler, fold_preds_array
    torch.cuda.empty_cache()
    gc.collect()

# --- Averaging and Formatting Submission ---
# This part now loads the saved predictions from files
print("\n--- Creating Submission File from Saved Predictions ---")

all_fold_preds = []
for fold in range(5):
    fold_preds = np.load(f'{PREDS_PATH}/fold_{fold}_preds.npy')
    all_fold_preds.append(fold_preds)

avg_preds = np.mean(all_fold_preds, axis=0)
top_3_preds_indices = np.argsort(avg_preds, axis=1)[:, ::-1][:, :3]
pred_labels = label_encoder.inverse_transform(top_3_preds_indices.flatten())
pred_labels = pred_labels.reshape(top_3_preds_indices.shape)
submission_strings = [' '.join(preds) for preds in pred_labels]

submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': submission_strings
})

submission_df.to_csv('submission.csv', index=False)

print("\nsubmission.csv created successfully!")
display(submission_df.head())

