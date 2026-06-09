!pip install huggingface-hub==0.23.0 --quiet
!pip install transformers==4.41.2 --quiet


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
from torch.cuda.amp import autocast, GradScaler
"""import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))"""

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


train_path = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/train.csv"
test_path = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/test.csv"
submission_file_path = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv"


# Loading the datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(submission_file_path)


display(train_df.head(3))


train_df.columns


train_df["rating"].value_counts()


train_df[train_df["rating"] == "approved"]["target"].describe()


display(test_df.head(1))


# Take only the same fields in the test set
train_df_v1 = train_df[["comment_text", "target"]]

# Drop rows with null comments
train_df_v1.dropna(subset=['comment_text'], inplace=True)

# convert the target to float
train_df_v1['target'] = train_df_v1['target'].astype(float)
train_df_v1.tail(10)


# Summary of the ds
display(train_df_v1.describe())
display(train_df_v1.info())


"""threshold =  0.4
train_df["toxicity_label"] = train_df["target"] >= threshold"""


train_df, val_df = train_test_split(train_df_v1, test_size=0.2, random_state=42)


clean_memory()


# Turn on the internet in the kaggle notebook
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class ToxicityCommentsDs(Dataset):
    def __init__(self, df, tokenizer, max_len=128, is_test=False):
        self.texts = df['comment_text'].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        if not self.is_test :
            # Return target column only with training state
            self.targets = df['target'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        output = {"input_ids": tokens["input_ids"].squeeze(), 
                  "attention_mask": tokens["attention_mask"].squeeze()
                 }
        if not self.is_test:
            output["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return output
            


# Prepare the pytorch dataset
train_ds = ToxicityCommentsDs(train_df, tokenizer)
val_ds   = ToxicityCommentsDs(val_df, tokenizer)

# Prepare the pytorch dataloader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# Prepatre the test set as a dataset and dataloader 
test_ds  = ToxicityCommentsDs(test_df, tokenizer, is_test=True)
test_loader  = DataLoader(test_ds, batch_size=32)


class ToxicCommentsModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1) # return 1 logit 

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask
                           )
        embedding = outputs.last_hidden_state[:, 0, :] 
        output = self.regressor(embedding)
        return output.squeeze()


pip install --upgrade protobuf==3.20.*


device = "cuda" if torch.cuda.is_available() else "cpu"


model = ToxicCommentsModel(MODEL_NAME)
model.to(device)

loss_fun = nn.BCEWithLogitsLoss()  
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


num_epochs = 1
best_val_loss = float("inf")
waiting_epochs = 3
best_model_state = None


scaler = GradScaler()

for epoch in range(num_epochs):
    print("Clean the memory")
    clean_memory()
    print(f"***** Epoch {epoch+1} out of {num_epochs} is started *****")

    # Training using mini-batch approach
    model.train()
    train_losses = []
    # Using tqdm to visiualize the progress of training in each batch
    nan_found = 0
    inf_found = 0
    nan_inf_not_found = 0
    for batch in tqdm(train_loader):
        # clean the previous gradients
        optimizer.zero_grad()
        
        # put the input batch (id, mask, target)in the GPU if avalible
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        # Forward phase using autocast to cast float32 to float16 for fast and light calculation 
        with autocast():
            preds = model(input_ids, attention_mask)
            loss = loss_fun(preds, targets)

        # Scale losses 
        scaler.scale(loss).backward()
        
        # check nan or inf values in the gradients
        # to make sure AMP isn't harming the trainig 
        for parm in model.parameters():
            if parm.grad is not None and (torch.isinf(parm.grad).any()):
                inf_found += 1
                print(f"Found in {epoch+1}, gradients that are inf")
            elif parm.grad is not None and (torch.isnan(parm.grad).any():
                nan_found += 1
                print(f"Found in {epoch+1}, gradients that are nan")
            else:
                nan_inf_not_found += 1

        # Updates parameters using scaled gradients
        scaler.step(optimizer)

        # Reset scaler for next step
        scaler.update()

        train_losses.append(loss.item())
    print(f"In epoch number: {epoch+1}, total number of batchs that aren't nan or inf {nan_inf_not_found}")
    print(f"In epoch number: {epoch+1}, total number of batchs that cintaines nan gradients: {nan_found}")
    print(f"In epoch number: {epoch+1}, total number of batchs that cintaines nan inf gradients: {inf_found}")

    # Validation phase
    model.eval()
    val_losses = []
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            
            # using AMP in validation phase
            with autocast():
                preds = model(input_ids, attention_mask)
                loss = loss_fun(preds, targets)

            val_losses.append(loss.item())
            val_preds.extend(preds.detach().cpu().numpy()) #need to make sure i have the probabilities not the logits 
            val_targets.extend(targets.detach().cpu().numpy())
            val_y_true = np.array(val_targets)
            val_y_pred = np.array(val_preds)
            # Sincr target is considered toxic when ≥ 0.5
            val_y_true_bin = (val_y_true >= 0.5).astype(int)

    # AUC metric 
    val_loss = np.mean(val_losses)
    train_loss = np.mean(train_losses)
    auc = roc_auc_score(val_y_true_bin, val_y_pred)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val AUC:    {auc:.4f}")

    # Early stopping 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        patience = waiting_epochs
        print(">>> New Best Model Saved!")
    else:
        patience -= 1
        print(f"Early Stopping Patience Left: {patience}")

        if patience == 0:
            print(">>> Early Stopping Triggered!")
            break




#torch.save(best_model_state, "best_model.pth")
#print("Model saved to best_model.pth")



# Load Best Model
model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
print("Best Model Loaded!")


model.eval()
test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        preds = model(input_ids, attention_mask)
        test_preds.extend(preds.detach().cpu().numpy())



sub = pd.read_csv("/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv")

sub["prediction"] = test_preds
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission fileis saved")





