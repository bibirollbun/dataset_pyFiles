!pip install wandb -q


import wandb 

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=secret_value_0)
    print("✅ W&B login successful")
except:
    print("W&B login failed. Please ensure you have set your WANDB_API_KEY secret.")


import os
import re
import pandas as pd
import numpy as np
from datasets import Dataset,DatasetDict
import warnings
from transformers import AutoTokenizer,AutoModel
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  
from torch.optim import AdamW  
import torch.cuda.amp as amp  
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================================
#                    CONFIGURATION
# =========================================================================
# All key hyperparameters and settings are managed in this class.
class CFG:
    # --- Model and Tokenizer Settings ---
    # The name of the transformer model to use from the Hugging Face Hub.
    # DeBERTa-v3 is a strong, modern choice.
    model_name = "microsoft/deberta-v3-base" 
    # Other options to try:
    # model_name = "distilbert-base-uncased"
    # model_name = "roberta-base"
    
    # Maximum sequence length for the tokenizer.
    max_length = 512

    stride = 256

    # --- Training Settings ---
    # Number of training epochs. A smaller number helps prevent overfitting on small datasets.
    epochs = 4
    
    # Batch size. Can be increased on more powerful GPUs like the T4 x2.
    batch_size = 8
    
    # Learning rate for the AdamW optimizer. 2e-5 is a common, effective starting point.
    lr = 2e-5

    # --- Cross-Validation Settings ---
    # Number of folds for StratifiedKFold. 5 is a standard and robust choice.
    n_splits = 5
    
    # A random seed for reproducibility. Ensures that splits and initializations are the same every time.
    seed = 42

    # llrd decay factor
    llrd = 0.95


df = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')


def train_data_generator(data_dir, csv_path):
    """
    Yield dictionaries of (text1, text2, label) for training.
    Label = 1 if file_1.txt is the real/original text, else 0.
    """
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        folder_id = row["id"]
        real_text_id = row["real_text_id"]

        folder_path = os.path.join(data_dir, f"article_{folder_id:04d}")
        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")

        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read()

        label = 1 if real_text_id == 1 else 0

        yield {
            "id": folder_id,
            "text1": text1,
            "text2": text2,
            "labels": label
        }


def test_data_generator(data_dir):
    """
    Yield dictionaries of (text1, text2) for testing (no labels).
    """
    folders = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f)) and re.match(r'article_\d+', f)
    ])

    for folder in folders:
        folder_id = int(folder.split('_')[1])
        folder_path = os.path.join(data_dir, folder)

        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")

        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read()

        yield {
            "id": folder_id,
            "text1": text1,
            "text2": text2
        }


train_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
train_csv = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

# Create datasets from generators
train_dataset = Dataset.from_generator(lambda: train_data_generator(train_dir, train_csv))
test_dataset = Dataset.from_generator(lambda: test_data_generator(test_dir))

# Combine into a DatasetDict
raw_datasets = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})


def expend_dataset(ds):
    new_rows = []
    for example in ds: 
        label = example['labels']

        # Create two new samples from each pair.
        # Same 'id', different 'text' (text1 and text2), and adjusted labels:
        new_rows.append({
            'id': example['id'],
            'text': example['text1'],
            'text_id': 1,
            'label': 1 if label == 1 else 0  # Positive if original label is 1
        })
        new_rows.append({
            'id': example['id'],
            'text': example['text2'],
            'text_id': 2,
            'label': 1 if label == 0 else 0  # Positive if original label is 0
        })
    return Dataset.from_list(new_rows)

# Apply transformation to the training set
train_expend = expend_dataset(raw_datasets['train'])


# Convert test set: split each text pair into two separate samples
def expand_test_dataset(test_ds):
    rows = []
    for ex in test_ds:
        rows.append({
            'id': ex['id'],
            'text': ex['text1'],
            'text_id': 1
        })
        rows.append({
            'id': ex['id'],
            'text': ex['text2'],
            'text_id': 2
        })
    return Dataset.from_list(rows)

# Apply transformation to the test set
test_expend = expand_test_dataset(raw_datasets['test'])


# Create new dataset dictionary using expanded train and test sets
raw_datasets_expend = DatasetDict({
    'train': train_expend,
    'test': test_expend
})

print(raw_datasets_expend)


        # ✅ Optionally freeze the backbone parameters
        # for param in self.backbone.parameters():
        #     param.requires_grad = False



class SiameseSelfAttentionNetwork(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone model (e.g., BERT/DistilBERT)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone.to(self.device)

        hidden_size = self.backbone.config.hidden_size
        
        # A simple but effective head with a dropout layer for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(hidden_size, num_labels)
        )

    def extract_mean_pooling_vector(self, texts: list[str]):
        """
        Extracts mean-pooled embeddings.
        ✅ This version now supports long texts using a sliding window approach.
        """
        batch_final_embeddings = []

        for text in texts:
            # 1. Tokenize the entire text without truncation to get all input_ids
            all_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # 2. Handle short texts normally for efficiency
            if len(all_input_ids) <= CFG.max_length - 2: # -2 for [CLS] and [SEP]
                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=CFG.max_length,
                    padding='max_length' # Pad to max_length for consistency
                ).to(self.device)
                
                outputs = self.backbone(**encoded)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"]
                
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                masked_hidden = last_hidden_state * mask
                summed = masked_hidden.sum(dim=1)
                count = mask.sum(dim=1)
                mean_vec = summed / count
                batch_final_embeddings.append(mean_vec)
                continue # Go to the next text in the batch

            # 3. Sliding Window for long texts
            chunk_embeddings = []
            
            # Prepare special token IDs
            cls_id = self.tokenizer.cls_token_id
            sep_id = self.tokenizer.sep_token_id
            
            # Iterate through the input_ids with the defined stride
            for i in range(0, len(all_input_ids), CFG.stride):
                # Get a slice of the input_ids for the current chunk
                chunk_ids_slice = all_input_ids[i : i + CFG.max_length - 2]
                
                # Add special tokens [CLS] and [SEP]
                chunk_ids = [cls_id] + chunk_ids_slice + [sep_id]
                
                # Pad the chunk if it's shorter than max_length (for the last chunk)
                padding_len = CFG.max_length - len(chunk_ids)
                chunk_ids += [self.tokenizer.pad_token_id] * padding_len
                
                # Create attention mask for the chunk
                attention_mask = [1] * (len(chunk_ids) - padding_len) + [0] * padding_len
                
                # Convert to tensors and move to the device
                chunk_ids_tensor = torch.tensor([chunk_ids], dtype=torch.long).to(self.device)
                attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)

                # Get model outputs for the chunk
                with torch.no_grad(): # Inference within the loop, grads calculated later
                    outputs = self.backbone(input_ids=chunk_ids_tensor, attention_mask=attention_mask_tensor)
                
                last_hidden_state = outputs.last_hidden_state

                # Mean pooling for this chunk
                mask = attention_mask_tensor.unsqueeze(-1).expand(last_hidden_state.size())
                masked_hidden = last_hidden_state * mask
                summed = masked_hidden.sum(dim=1)
                count = mask.sum(dim=1)
                chunk_mean_vec = summed / count
                chunk_embeddings.append(chunk_mean_vec)
            
            # 4. Aggregate chunk embeddings into one vector for the long text
            # We stack all chunk embeddings and calculate their mean
            final_long_text_vec = torch.mean(torch.cat(chunk_embeddings, dim=0), dim=0, keepdim=True)
            batch_final_embeddings.append(final_long_text_vec)

        # Combine embeddings for all texts in the batch into a single tensor
        final_vecs = torch.cat(batch_final_embeddings, dim=0)
        return final_vecs

    def forward(self, texts, labels=None):
        """
        Forward pass.
        Args:
            texts (List[str]): Batch of input texts.
            labels (Optional[Tensor]): Ground truth labels.
        Returns:
            An object with .loss and .logits attributes.
        """
        # 1. Get the single vector representation for each text (now handles any length)
        final_vecs = self.extract_mean_pooling_vector(texts) # Shape: [B, H]

        # 2. Pass it directly to the simple classifier
        logits = self.classifier(final_vecs) # Shape: [B, num_labels]

        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            labels = labels.to(self.device)
            loss = criterion(logits, labels)

        return type('Outputs', (object,), {'loss': loss, 'logits': logits})()


def train_fn(model, dataloader, optimizer, scheduler, device='cuda'):
    """
    Training loop for one epoch.

    Args:
        model: The neural network model to train.
        dataloader: DataLoader providing training batches.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        device: Device to run the training on (default: 'cuda').

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc='Training'):
        texts = batch['text']
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = F.cross_entropy(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



# =========================================================================
#                    CORRECT AND FINAL eval_fn
# =========================================================================
def eval_fn(model, dataloader, device='cuda', desc='Evaluating'):
    """
    Evaluation function for inference.
    
    This version CORRECTLY returns a dictionary of raw probabilities 
    needed for ensembling across multiple folds.
    Format: {(article_id, text_id): probability}
    """
    model.eval()
    
    all_text_probs = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            texts = batch["text"]
            ids = batch["id"]
            text_ids = batch["text_id"]

            outputs = model(texts)
            probs = F.softmax(outputs.logits, dim=-1)[:, 1]

            for a_id, t_id, prob in zip(ids, text_ids, probs):
                key = (int(a_id), int(t_id))
                all_text_probs[key] = prob.item()

    # The function now returns the correct data structure.
    return all_text_probs


# =========================================================================
#         UPGRADED calculate_accuracy function to return OOF predictions
# =========================================================================
def calculate_accuracy(model, dataloader, device='cuda'):
    """
    Calculates pairwise accuracy on a validation set and returns the raw 
    Out-of-Fold (OOF) probabilities for each pair.
    """
    model.eval()
    
    # --- Part 1: Get individual text probabilities ---
    article_to_probs = defaultdict(dict)
    with torch.no_grad():
        for batch in dataloader:
            texts, ids, text_ids = batch['text'], batch['id'], batch['text_id']
            outputs = model(texts)
            probs = F.softmax(outputs.logits, dim=-1)[:, 1]
            for a_id, t_id, prob in zip(ids, text_ids, probs):
                article_to_probs[int(a_id)][int(t_id)] = prob.item()

    # --- Part 2: Calculate Pairwise Accuracy (for logging) ---
    val_df_subset = df[df['id'].isin(article_to_probs.keys())]
    true_labels = dict(zip(val_df_subset['id'], val_df_subset['real_text_id']))
    
    correct_predictions = 0
    for a_id, prob_dict in article_to_probs.items():
        # Make sure both texts for the pair were found
        if 1 in prob_dict and 2 in prob_dict:
            pred_text_id = 1 if prob_dict[1] > prob_dict[2] else 2
            if pred_text_id == true_labels[a_id]:
                correct_predictions += 1
    
    accuracy = correct_predictions / len(article_to_probs) if article_to_probs else 0

    # --- Part 3: Calculate and Return OOF Probabilities ---
    # The OOF score should be a single probability for each pair.
    # We'll define it as the probability that text1 is the real one.
    oof_probabilities = {}
    for a_id, prob_dict in article_to_probs.items():
        if 1 in prob_dict and 2 in prob_dict:
            prob1 = prob_dict[1]
            prob2 = prob_dict[2]
            # Normalize to get a single probability for the pair
            # P(text1 is real) = prob1 / (prob1 + prob2)
            oof_prob = prob1 / (prob1 + prob2 + 1e-9) # Add epsilon for safety
            oof_probabilities[a_id] = oof_prob
        
    return accuracy, oof_probabilities


# --- 1. PREPARE FOR CROSS-VALIDATION ---
df = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')
X = df['id']
y = df['real_text_id'].apply(lambda x: 1 if x == 1 else 0)
skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

# --- Initialize lists to store results ---
test_preds_list = []
oof_scores = []
# Master dictionary to store OOF predictions from all folds
oof_preds_all = {} 

# --- Optional: W&B Group Name ---
group_name = f"{CFG.model_name.replace('/', '-')}-e{CFG.epochs}-lr{CFG.lr}-llrd{CFG.llrd}"
config_instance = CFG()

# --- 2. START THE MAIN CV LOOP ---
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"========== FOLD {fold+1}/{CFG.n_splits} ==========")
    
    # --- Optional: W&B Initialization ---
    run = wandb.init(project="Kaggle - Impostor Hunt", config=vars(config_instance), group=group_name, name=f"fold-{fold+1}", reinit=True)
    
    # --- Data Preparation ---
    train_data_fold = raw_datasets['train'].select(train_idx)
    val_data_fold = raw_datasets['train'].select(val_idx)
    train_expend = expend_dataset(train_data_fold)
    val_expend = expend_dataset(val_data_fold)
    train_loader = DataLoader(train_expend, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_expend, batch_size=CFG.batch_size, shuffle=False)
    
    # --- Model Initialization ---
    model = SiameseSelfAttentionNetwork(CFG.model_name).to("cuda")

    # --- LLRD Optimizer and Cosine Scheduler ---
    print("Applying Layer-wise Learning Rate Decay...")
    optimizer_parameters = []
    named_parameters = list(model.named_parameters())
    num_layers = model.backbone.config.num_hidden_layers
    
    head_params = [p for n, p in named_parameters if "backbone" not in n]
    optimizer_parameters.append({"params": head_params, "lr": CFG.lr})

    for i in range(num_layers - 1, -1, -1):
        encoder_layer_params = [p for n, p in named_parameters if f"backbone.encoder.layer.{i}." in n]
        layer_lr = CFG.lr * (CFG.llrd ** (num_layers - i))
        optimizer_parameters.append({"params": encoder_layer_params, "lr": layer_lr})

    embedding_params = [p for n, p in named_parameters if "backbone.embeddings" in n]
    embedding_lr = CFG.lr * (CFG.llrd ** (num_layers + 1))
    optimizer_parameters.append({"params": embedding_params, "lr": embedding_lr})

    optimizer = AdamW(optimizer_parameters)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=25,
        num_training_steps=len(train_loader) * CFG.epochs
    )
    
    # --- Early Stopping Setup ---
    best_val_accuracy = 0.0
    best_model_path = f"/kaggle/working/best_model_fold_{fold+1}.pth"

    # --- Inner Training Loop ---
    print(f"--- Starting Training for Fold {fold+1} ---")
    for epoch in range(CFG.epochs):
        train_loss = train_fn(model, train_loader, optimizer, scheduler) 
        val_accuracy, _ = calculate_accuracy(model, val_loader) # We only need accuracy for logging here

        # --- Optional: W&B Logging ---
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_accuracy": val_accuracy, "learning_rate": optimizer.param_groups[0]['lr']})

        print(f"Epoch {epoch+1}/{CFG.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # --- Early Stopping Logic ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved for fold {fold+1} with Val Acc: {best_val_accuracy:.4f}")

    oof_scores.append(best_val_accuracy)

    # --- Store OOF & Test Predictions using the BEST model from this fold ---
    print(f"Loading best model for fold {fold+1} from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    
    # Calculate and store OOF predictions for this fold's validation set
    _, oof_probs_fold = calculate_accuracy(model, val_loader)
    oof_preds_all.update(oof_probs_fold)
    
    # Calculate and store predictions for the test set
    test_loader = DataLoader(raw_datasets_expend['test'], batch_size=CFG.batch_size, shuffle=False)
    test_probs = eval_fn(model, test_loader, desc=f"Predicting with Best Model (Fold {fold+1})")
    test_preds_list.append(test_probs)
    
    # --- Optional: W&B Finish Run ---
    run.finish()

print("\n" + "="*50)
print("          CV TRAINING FINISHED")
print(f"Average Peak Validation Accuracy: {np.mean(oof_scores):.4f}")
print("="*50)


# --- 3. SAVE OOF PREDICTIONS TO CSV ---
print("\nSaving Out-of-Fold (OOF) predictions...")
oof_df = pd.DataFrame(list(oof_preds_all.items()), columns=['id', 'oof_prob'])
oof_df = oof_df.sort_values('id').reset_index(drop=True)
oof_df.to_csv("oof_predictions.csv", index=False)
print("✅ oof_predictions.csv has been saved successfully.")
print("\nOOF Predictions file preview:")
print(oof_df.head())


# --- 4. AGGREGATE PREDICTIONS AND CREATE SUBMISSION ---
print("\nAggregating predictions from all folds...")
aggregated_probs = defaultdict(list)
for fold_preds in test_preds_list:
    for key, prob in fold_preds.items():
        aggregated_probs[key].append(prob)

mean_probs = {key: np.mean(probs) for key, probs in aggregated_probs.items()}

article_to_final_probs = defaultdict(dict)
for (article_id, text_id), final_prob in mean_probs.items():
    article_to_final_probs[article_id][text_id] = final_prob

final_preds = {}
for article_id, prob_dict in article_to_final_probs.items():
    pred_text_id = max(prob_dict.items(), key=lambda item: item[1])[0]
    final_preds[article_id] = pred_text_id

submission = pd.DataFrame({'id': list(final_preds.keys()), 'real_text_id': list(final_preds.values())})
submission = submission.sort_values('id').reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print("\n✅ submission.csv has been saved successfully.")
print("\nSubmission file preview:")
print(submission.head())




