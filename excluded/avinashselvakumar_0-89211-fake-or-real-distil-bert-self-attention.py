import os
import re
import pandas as pd
import numpy as np
from datasets import Dataset,DatasetDict
import warnings
from transformers import AutoTokenizer,AutoModel
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  
from torch.optim import AdamW  
import torch.cuda.amp as amp  
import torch.nn.functional as F
from collections import defaultdict
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')
df.head()


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


class SiameseSelfAttentionNetwork(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone model (e.g., BERT/DistilBERT)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.to(self.device)

        # ✅ Optionally freeze the backbone parameters
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        # Multi-head self-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.backbone.config.num_attention_heads,
            batch_first=True
        )

        # Feature interaction head (fully connected layers with dropout)
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Final classification head
        self.classifier = nn.Linear(hidden_size // 4, num_labels)

        # Tokenizer for input text
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_mean_pooling_vector(self, texts: list[str]):
        """Extracts mean-pooled embeddings from transformer output (supports batch processing)."""
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        outputs = self.backbone(**encoded)
        last_hidden_state = outputs.last_hidden_state  # Shape: [B, L, H]
        attention_mask = encoded["attention_mask"]

        # Apply attention mask before pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        masked_hidden = last_hidden_state * mask
        summed = masked_hidden.sum(dim=1)
        count = mask.sum(dim=1)
        mean_vecs = summed / count  # Shape: [B, H]

        return mean_vecs

    def forward(self, texts, labels=None):
        """
        Forward pass.
        Args:
            texts (List[str]): Batch of input texts.
            labels (Optional[Tensor]): Ground truth labels.
        Returns:
            An object with .loss and .logits attributes.
        """
        final_vecs = self.extract_mean_pooling_vector(texts)  # Shape: [B, H]

        # Cross-attention (self-attention on embeddings)
        # Attention input must be of shape [B, T, H], here T=1
        query = key = value = final_vecs.unsqueeze(1)  # Shape: [B, 1, H]
        atten_output, _ = self.cross_attn(query, key, value)  # Shape: [B, 1, H]

        interaction_output = self.interaction_head(atten_output.squeeze(1))  # Shape: [B, H//4]
        logits = self.classifier(interaction_output)  # Shape: [B, num_labels]

        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            labels = labels.to(self.device)
            loss = criterion(logits, labels)

        # Return a simple object with loss and logits attributes
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



def eval_fn(model, dataloader, device='cuda', desc='Evaluating'):
    """
    Evaluation function for inference and prediction reconstruction.

    Args:
        model: The trained model to evaluate.
        dataloader: DataLoader providing evaluation data.
        device: Device to run inference on (default: 'cuda').
        desc: Description string for progress bar.

    Returns:
        A dictionary mapping each article ID to the predicted real text ID.
        Format: {article_id: predicted_text_id}
    """
    model.eval()
    article_to_probs = defaultdict(dict)  # Stores predicted probability per text ID for each article

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            texts = batch["text"]
            ids = batch["id"]
            text_ids = batch["text_id"]

            outputs = model(texts)
            probs = F.softmax(outputs.logits, dim=-1)[:, 1]  # Probability for label=1 (real text)

            # Collect probabilities for each text in each article
            for a_id, t_id, prob in zip(ids, text_ids, probs):
                a_id = int(a_id)
                t_id = int(t_id)
                article_to_probs[a_id][t_id] = prob.item()

    # Select the text ID with the highest probability per article
    final_preds = {}
    for a_id, prob_dict in article_to_probs.items():
        pred_text_id = max(prob_dict.items(), key=lambda x: x[1])[0]
        final_preds[a_id] = pred_text_id

    return final_preds  # {article_id: predicted_real_text_id}



# Load training data
train_loader = DataLoader(raw_datasets_expend['train'], batch_size=4, shuffle=True)

# Initialize the model with DistilBERT backbone
model = SiameseSelfAttentionNetwork("distilbert-base-uncased").to("cuda")

# (Optional) Check which parameters are frozen before training
# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad = {param.requires_grad}")


# Initialize optimizer (only trainable parameters)
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
)

# Linear learning rate scheduler without warm-up
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * 3  # Total steps = num_batches * epochs
)

# Training loop
epochs = 10
for epoch in range(epochs):
    train_loss = train_fn(model, train_loader, optimizer, scheduler)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")


#Load Test data
test_loader = DataLoader(raw_datasets_expend['test'], batch_size=4)
preds = eval_fn(model, test_loader)

#Generate submission file 
submission = pd.DataFrame({
    'id': list(preds.keys()),
    'real_text_id': list(preds.values())
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv has been saved.")

# Preview submission
print(submission.head())

