import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
from peft import get_peft_model, LoraConfig, TaskType
from torch.amp import autocast, GradScaler
import random
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import re
max_length = 128  # Maximum length of the input sequences
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set all seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Enable gradient checkpointing for memory efficiency
torch.backends.cuda.matmul.allow_tf32 = True

# Define a custom dataset for our text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=max_length):
        self.texts = texts
        self.labels = labels  # Can be None for test data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item



# Load your CSV files
train_df = pd.read_csv("data/pseudo_train.csv")  # Columns: "Question", "label"
test_df = pd.read_csv("data/test.csv")    # Columns: "id", "Question"

def clean(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df["Question"] = train_df["Question"].apply(clean)
test_df["Question"] = test_df["Question"].apply(clean)

train_texts = train_df["Question"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["Question"].tolist()

# Determine the number of classes
num_classes = len(np.unique(train_labels))

# Initialize the tokenizer 
model_name = "Qwen/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LLM-specific: Optionally set pad_token to eos_token for Qwen
tokenizer.pad_token = tokenizer.eos_token



# Label smoothing loss for classification
import torch.nn.functional as F
def label_smoothed_nll_loss(logits, target, epsilon=0.1):
    n_class = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -log_probs.mean(dim=-1)
    loss = (1 - epsilon) * nll_loss + epsilon * smooth_loss
    return loss.mean()

# Training loop with gradient accumulation and label smoothing
def train_epoch_llm(model, dataloader, optimizer, device, scaler, ema=None, grad_accum_steps=2, label_smoothing=0.1, scheduler=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        for key in batch:
            batch[key] = batch[key].to(device)
        with autocast():
            outputs = model(**batch)
            if label_smoothing > 0 and "labels" in batch:
                loss = label_smoothed_nll_loss(outputs.logits, batch["labels"], epsilon=label_smoothing)
            else:
                print("No label smoothing")
                loss = outputs.loss
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update()
        total_loss += loss.item() * grad_accum_steps
    return total_loss / len(dataloader)

# Training loop with gradient accumulation (no label smoothing)
def train_epoch_llm(model, dataloader, optimizer, device, scaler, ema=None, grad_accum_steps=2, scheduler=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    print("Training with gradient accumulation")
    print("No label smoothing")
    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        for key in batch:
            batch[key] = batch[key].to(device)
        with autocast('cuda'):
            outputs = model(**batch)
            loss = outputs.loss
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update()
        total_loss += loss.detach().item() * grad_accum_steps
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad(): # Disable gradient calculations for evaluation
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(**batch)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.extend(batch_preds)
            if "labels" in batch:
                true_labels.extend(batch["labels"].detach().cpu().numpy())
    
    # Calculate and print classification report
    if true_labels:
        report = classification_report(true_labels, preds, target_names=[f"Class {i}" for i in range(len(np.unique(true_labels)))])
        print(report)
    
    return np.array(preds), np.array(true_labels)

def predict_probas(model, dataloader, device):
    model.eval() 
    all_probs = []
    with torch.no_grad(): # Disable gradient calculations for prediction
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)



def get_llrd_params(model, base_lr=1e-6, lora_lr=1e-4, head_lr=1e-4):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = []
    lora_params_decay = []
    lora_params_no_decay = []
    base_model_params_decay = []
    base_model_params_no_decay = []
    head_params_decay = []
    head_params_no_decay = []

    # Identify LoRA, base model, and classification head parameters
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            if any(nd in n for nd in no_decay):
                lora_params_no_decay.append(p)
            else:
                lora_params_decay.append(p)
        elif "score" in n or "classifier" in n:
            if any(nd in n for nd in no_decay):
                head_params_no_decay.append(p)
            else:
                head_params_decay.append(p)
        else:
            if any(nd in n for nd in no_decay):
                base_model_params_no_decay.append(p)
            else:
                base_model_params_decay.append(p)

    # Group parameters for different learning rates
    if lora_params_decay:
        grouped_params.append({"params": lora_params_decay, "lr": lora_lr, "weight_decay": 0.01})
    if lora_params_no_decay:
        grouped_params.append({"params": lora_params_no_decay, "lr": lora_lr, "weight_decay": 0.0})
    if base_model_params_decay:
        grouped_params.append({"params": base_model_params_decay, "lr": base_lr, "weight_decay": 0.01})
    if base_model_params_no_decay:
        grouped_params.append({"params": base_model_params_no_decay, "lr": base_lr, "weight_decay": 0.0})
    if head_params_decay:
        grouped_params.append({"params": head_params_decay, "lr": head_lr, "weight_decay": 0.01})
    if head_params_no_decay:
        grouped_params.append({"params": head_params_no_decay, "lr": head_lr, "weight_decay": 0.0})

    # Verify all trainable parameters are included
    all_grouped_params = set(p for group in grouped_params for p in group['params'])
    all_trainable_params = set(p for p in model.parameters() if p.requires_grad)
    if all_grouped_params != all_trainable_params:
        print("Warning: Not all trainable parameters are included in the optimizer groups.")
        print("Missing:", all_trainable_params - all_grouped_params)
        print("Extra:", all_grouped_params - all_trainable_params)

    return grouped_params


NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

batch_size = 8
grad_accum_steps = 4  # Effective batch size = batch_size * grad_accum_steps
data_collator = DataCollatorWithPadding(tokenizer)

oof_preds = np.zeros(len(train_texts), dtype=int)
test_probs = np.zeros((len(test_texts), num_classes))
fold_f1_micro = []

# Bagging: use a different seed for each fold
fold_seeds = [42 + i for i in range(NUM_FOLDS)]

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    print(f"===== Fold {fold+1} / {NUM_FOLDS} =====")
    set_seed(fold_seeds[fold])  # Bagging: different seed per fold

    X_trn = [train_texts[i] for i in train_idx]
    y_trn = [train_labels[i] for i in train_idx]
    X_val = [train_texts[i] for i in val_idx]
    y_val = [train_labels[i] for i in val_idx]

    train_dataset = TextDataset(X_trn, y_trn, tokenizer, max_length=max_length)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
    )
    base_model.gradient_checkpointing_enable()

    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # LLRD
    grouped_params = get_llrd_params(model, base_lr=1e-6, lora_lr=2e-4, head_lr=1e-3)
    optimizer = optim.AdamW(grouped_params)
    num_epochs = 3

    scaler = GradScaler('cuda')

    best_f1 = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train_epoch_llm(model, train_loader, optimizer, device, scaler, grad_accum_steps=grad_accum_steps)
        val_preds, val_true = evaluate(model, val_loader, device)
        epoch_f1 = f1_score(val_true, val_preds, average="micro")
        print(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {train_loss:.4f}, F1 (micro): {epoch_f1:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_model_state = model.state_dict()

    # Load best model weights for this fold
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    val_preds, val_true = evaluate(model, val_loader, device)
    oof_preds[val_idx] = val_preds
    fold_f1 = f1_score(val_true, val_preds, average="micro")
    fold_f1_micro.append(fold_f1)
    print(f"Fold {fold+1} F1 (micro): {fold_f1:.4f}")

    test_dataset = TextDataset(test_texts, labels=None, tokenizer=tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    fold_test_probs = predict_probas(model, test_loader, device)
    test_probs += fold_test_probs

    del model, base_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

overall_oof_f1 = f1_score(train_labels, oof_preds, average="micro")
print(f"Overall OOF F1 (micro): {overall_oof_f1:.4f}")

test_probs /= NUM_FOLDS
final_test_preds = np.argmax(test_probs, axis=1)


submission = pd.DataFrame({
    "id": test_df["id"],
    "label": final_test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")



probs_df = pd.DataFrame(test_probs, columns=[f"class_{i}_prob" for i in range(test_probs.shape[1])])
probs_df.insert(0, "id", test_df["id"])
probs_df.to_csv("test_probs.csv", index=False)
print("Test probabilities saved to test_probs.csv")

