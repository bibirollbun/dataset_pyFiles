import sys, os
import pandas as pd
import torch
import transformers
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns


torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na")


columns = ['Source', 'Label', 'Author Label', 'Sentence']
data = pd.read_csv('/kaggle/input/cola-the-corpus-of-linguistic-acceptability/cola_public/raw/in_domain_train.tsv', sep='\t', names=columns)
data = data[['Sentence', 'Label']]
data.head()


tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
    print("Added new pad_token '<|PAD|>' with ID:", tokenizer.pad_token_id)


class Cola_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length=None, test=False):
        if test:
            df = pd.read_csv(data_path, sep='\t', names=['Index', 'Sentence'])
            df = df.drop(index=0) # Drop column headers
            df['Label'] = -1
        else:
            columns = ['Source', 'Label', 'Author Label', 'Sentence']
            df = pd.read_csv(data_path, sep='\t', names=columns)

        encoded_sentences = []
        for sentence in df['Sentence'].tolist():
            try:
                encoded_sentences.append(
                    tokenizer.encode(
                        sentence,
                        add_special_tokens=True
                    )
                )
            except Exception as e:
                raise ValueError(f"Error encoding text: {sentence[:50]}...") from e

        self.sentences = encoded_sentences
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences longer than max_length
            self.sentences = [
                encoded_text[:self.max_length] for encoded_text in self.sentences
            ]

        padded_sentences = []
        attention_masks = []
        for enc in self.sentences:
            enc = enc[:self.max_length]
            attention_mask = [1] * len(enc)
            
            pad_len = self.max_length - len(enc)
            if pad_len > 0:
                enc += [tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
            
            padded_sentences.append(enc)
            attention_masks.append(attention_mask)
        
        self.padded_sentences = padded_sentences
        self.attention_masks = attention_masks
        self.labels = df['Label'].tolist()

    def __len__(self):
        return len(self.sentences)

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.sentences)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.padded_sentences[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_masks[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


max_length = 45 # max of train: 45, iD_Test: 32, ooD_Test: 31, iD_Val: 29, ooD_Val: 34

test_in_domain_dataset = Cola_Dataset("/kaggle/input/cola-in-domain-open-evaluation/cola_in_domain_test.tsv",
                            tokenizer=tokenizer,
                            max_length=45,
                            test=True)
test_out_of_domain_dataset = Cola_Dataset("/kaggle/input/cola-out-of-domain-open-evaluation/cola_out_of_domain_test.tsv",
                            tokenizer=tokenizer,
                            max_length=45,
                            test=True)
train_dataset = Cola_Dataset("/kaggle/input/cola-the-corpus-of-linguistic-acceptability/cola_public/raw/in_domain_train.tsv",
                             max_length=45,
                             tokenizer=tokenizer)
val_in_domain_dataset = Cola_Dataset("/kaggle/input/cola-the-corpus-of-linguistic-acceptability/cola_public/raw/in_domain_dev.tsv",
                                     max_length=45,
                                     tokenizer=tokenizer)
val_out_of_domain_dataset = Cola_Dataset("/kaggle/input/cola-the-corpus-of-linguistic-acceptability/cola_public/raw/out_of_domain_dev.tsv",
                                         max_length=45,
                                         tokenizer=tokenizer)


# Set DataLoader parameters
batch_size = 32
num_workers = 4

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

val_in_domain_loader = torch.utils.data.DataLoader(
    val_in_domain_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False
)
val_out_of_domain_loader = torch.utils.data.DataLoader(
    val_out_of_domain_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False
)

test_in_domain_loader = torch.utils.data.DataLoader(
    test_in_domain_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False
)
test_out_of_domain_loader = torch.utils.data.DataLoader(
    test_out_of_domain_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False
)

dataloaders = [train_loader,
               val_in_domain_loader,
               val_out_of_domain_loader,
               test_in_domain_loader,
               test_out_of_domain_loader
              ]

print(f"""
 Number of training batches:\t\t\t{len(train_loader)}
 Number of in domain validation batches:\t {len(val_in_domain_loader)}
 Number of out of domain validation batches:\t {len(val_out_of_domain_loader)}
 Number of in domain test batches:\t\t {len(test_in_domain_loader)}
 Number of out of domain test batches:\t\t {len(test_out_of_domain_loader)}
 """)


pretrained_gpt2 = transformers.GPT2ForSequenceClassification.from_pretrained( 
    "gpt2", 
    num_labels=2, 
    pad_token_id=tokenizer.pad_token_id 
) 
pretrained_gpt2.resize_token_embeddings(len(tokenizer)) 
pretrained_gpt2.to(device) 
print(pretrained_gpt2)


no_pretraining_gpt2 = copy.deepcopy(pretrained_gpt2)
for name, param in no_pretraining_gpt2.named_parameters():
    if 'bias' in name or param.dim() < 2:
        torch.nn.init.zeros_(param)
    else:
        torch.nn.init.xavier_uniform_(param)


def classify(sentence, model, tokenizer, device, max_length=45, pad_token_id=50256):
    model.to(device)
    model.eval()
    enc = tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=max_length) 
    att_mask = [1] * len(enc) 
    pad_len = max_length - len(enc) 
    if pad_len > 0: 
        enc += [pad_token_id] * pad_len 
        att_mask += [0] * pad_len 

    input_ids = torch.tensor([enc], dtype=torch.long).to(device) 
    attention_mask = torch.tensor([att_mask], dtype=torch.long).to(device)

    with torch.no_grad(): 
        outputs = model(input_ids, attention_mask=attention_mask) 
        logits = outputs.logits 
        print(logits)
        acceptable = torch.argmax(logits, dim=-1).item() 
    return "acceptable" if acceptable else "grammatical error" 


def validate(model, dataloader, device, loss_fn=None, verbose=False):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    if verbose:
        dataloader = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device) 
            attention_mask = batch["attention_mask"].to(device) 
            labels = batch["labels"].to(device) 
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            if not loss_fn:
                loss = outputs.loss
            else:
                loss = loss_fn(logits, labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / total if total > 0 else 0
    return loss, accuracy


print("GPT2 Model without Fine-tuning:")
accuracies = []
for dataloader in dataloaders[:3]:
    _, acc = validate(pretrained_gpt2, dataloader, device)
    accuracies.append(acc)

print(f"Training-set accuracy:\t\t\t{accuracies[0]*100:.2f}%")
print(f"Validation accuracy in Domain:\t\t{accuracies[1]*100:.2f}%") 
print(f"Validation accuracy out of Domain:\t{accuracies[2]*100:.2f}%") 

print("\nGPT2 Model without Fine-tuning or Pretraining:")
accuracies = []
for dataloader in dataloaders[:3]:
    _, acc = validate(no_pretraining_gpt2, dataloader, device)
    accuracies.append(acc)

print(f"Training-set accuracy:\t\t\t{accuracies[0]*100:.2f}%")
print(f"Validation accuracy in Domain:\t\t{accuracies[1]*100:.2f}%") 
print(f"Validation accuracy out of Domain:\t{accuracies[2]*100:.2f}%") 


model = copy.deepcopy(pretrained_gpt2)


for param in model.base_model.parameters(): 
    param.requires_grad = False 
for param in model.score.parameters(): 
    param.requires_grad = True


def weight_decay_by_group(model, decay, exclude=["bias", "LayerNorm.weight"]):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(ex in name for ex in exclude):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def plot_loss(train_losses, val_losses, ood_val_loss=None):
    plt.figure(figsize=(10, 6))
    x = range(1, len(train_losses) + 1)
    plt.plot(x, train_losses, label="Training Loss", color="blue")
    plt.plot(x, val_losses[1:], label="Validation Loss", color="green")
    if ood_val_loss:
        plt.plot(x, ood_val_loss[1:], label="Out of Domain Validation Loss", color="red")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    if len(x) < 11:
        plt.xticks(ticks=x)
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    


class Early_Stopper:
    def __init__(self, patience=3, delta=1e-3, save_path="model_checkpoint.pth"):
        self.max_patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.current_patience = 0

    def __call__(self, loss, model):
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.current_patience = self.max_patience
            torch.save(model.state_dict(), self.save_path)
            return False
        else:
            self.current_patience -= 1
            if self.current_patience <= 0:
                model.load_state_dict(torch.load(self.save_path, weights_only=True, map_location=model.device))
                return True
            return False


def train(model, optimizer, train_loader, val_loader, device, num_epochs, early_stopper, loss_fn=None, scheduler=None, ood_val_loader=None):
    model.to(device)
    train_loss = []
    validation_loss = []
    validation_acc = []
    ood_val_loss = []
    ood_val_acc = []
    print_in = np.linspace(num_epochs // 4, num_epochs - 1, num=4, dtype=int)

    loss, acc = validate(model, val_loader, device)
    validation_loss.append(loss.item())
    validation_acc.append(acc)

    if ood_val_loader:
        loss, acc = validate(model, ood_val_loader, device)
        ood_val_loss.append(loss.item())
        ood_val_acc.append(acc)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
            input_ids = batch["input_ids"].to(device) 
            attention_mask = batch["attention_mask"].to(device) 
            labels = batch["labels"].to(device)   
    
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    
            if not loss_fn:
                loss = outputs.loss # CrossEntropyLoss 
            else:
                loss = loss_fn(outputs.logits[:, 1], labels.float())
            loss.backward()
            optimizer.step()
    
            if scheduler:
                scheduler.step()
               
            optimizer.zero_grad()
            epoch_loss += loss.item()
    
        train_loss.append(epoch_loss / len(train_loader))
    
        loss, acc = validate(model, val_loader, device)
        validation_loss.append(loss.item())
        validation_acc.append(acc)

        if ood_val_loader:
            loss, acc = validate(model, ood_val_loader, device)
            ood_val_loss.append(loss.item())
            ood_val_acc.append(acc)

        if epoch in print_in:
            print(f"Epoch {epoch+1}/{num_epochs}:\n Training Loss:\t\t {train_loss[epoch]:.4f}\n Validation Loss:\t {loss:.4f}\n Validation Accuracy:\t {100*acc:.2f}%")

        if early_stopper(validation_loss[epoch], model):
            print(f"Stopping early at Epoch {epoch+1}/{num_epochs}")
            break

    if ood_val_loader:
        return train_loss, validation_loss, validation_acc, ood_val_loss, ood_val_acc

    return train_loss, validation_loss, validation_acc


num_epochs = 100
learning_rate = 1e-5
# loss_fn is CrossEntropyLoss
optimizer = torch.optim.AdamW(weight_decay_by_group(model, 0.01), lr=learning_rate)

num_training_steps=len(train_loader) * num_epochs
num_warmup_steps=int(0.1 * num_training_steps) # 10% warmup
patience = 5

scheduler = transformers.get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

early_stopper = Early_Stopper(patience=patience)

train_loss, validation_loss, validation_acc, ood_val_loss, ood_val_acc = train(
    model,
    optimizer,
    train_loader,
    val_in_domain_loader,
    device,
    num_epochs,
    early_stopper,
    scheduler=scheduler,
    ood_val_loader=val_out_of_domain_loader
)


plot_loss(train_loss, validation_loss, ood_val_loss)


path = 'Pretrained_Classification_Head_Finetuning.pth'
torch.save(model.state_dict(), path)

model_accuracies = {}
model_accuracies['Pretrained Head FT'] = [validation_acc, ood_val_acc]


model = copy.deepcopy(no_pretraining_gpt2)

for name, param in model.named_parameters():
    if 'bias' in name or param.dim() < 2:
        torch.nn.init.zeros_(param)
    else:
        torch.nn.init.xavier_uniform_(param)

for param in model.base_model.parameters(): 
    param.requires_grad = False 
for param in model.score.parameters(): 
    param.requires_grad = True


num_epochs = 100
learning_rate = 1e-5
# loss_fn is CrossEntropyLoss
parameters = filter(lambda param: param.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(weight_decay_by_group(model, 0.01), lr=learning_rate)

num_training_steps=len(train_loader) * num_epochs
num_warmup_steps=int(0.1 * num_training_steps) # 10% warmup
patience = 5

scheduler = transformers.get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

early_stopper = Early_Stopper(patience=patience)

train_loss, validation_loss, validation_acc, ood_val_loss, ood_val_acc = train(
    model,
    optimizer,
    train_loader,
    val_in_domain_loader,
    device,
    num_epochs,
    early_stopper,
    scheduler=scheduler,
    ood_val_loader=val_out_of_domain_loader
)


plot_loss(train_loss, validation_loss, ood_val_loss)


path = 'Untrained_Classification_Head_Finetuning.pth'
torch.save(model.state_dict(), path)

model_accuracies['Untrained Head FT'] = [validation_acc, ood_val_acc]


model = copy.deepcopy(pretrained_gpt2)


num_epochs = 100
learning_rate = 1e-5
# loss_fn is CrossEntropyLoss
parameters = [
    {'params': model.transformer.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.score.parameters(), 'lr': 1e-3, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

num_training_steps=len(train_loader) * num_epochs
num_warmup_steps=int(0.1 * num_training_steps) # 10% warmup
patience = 5

scheduler = transformers.get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

early_stopper = Early_Stopper(patience=patience)

train_loss, validation_loss, validation_acc, ood_val_loss, ood_val_acc = train(
    model,
    optimizer,
    train_loader,
    val_in_domain_loader,
    device,
    num_epochs,
    early_stopper,
    scheduler=scheduler,
    ood_val_loader=val_out_of_domain_loader
)


plot_loss(train_loss, validation_loss, ood_val_loss)


path = 'Pretrained_Finetuning.pth'
torch.save(model.state_dict(), path)

model_accuracies['Pretrained Full FT'] = [validation_acc, ood_val_acc]


model = copy.deepcopy(no_pretraining_gpt2)

for name, param in model.named_parameters():
    if 'bias' in name or param.dim() < 2:
        torch.nn.init.zeros_(param)
    else:
        torch.nn.init.xavier_uniform_(param)

for param in model.base_model.parameters(): 
    param.requires_grad = True 
for param in model.score.parameters(): 
    param.requires_grad = True


print("GPT2 Model without Fine-tuning or Pretraining (xaviar and zeros):")
accuracies = []
for dataloader in dataloaders[:3]:
    _, acc = validate(model, dataloader, device)
    accuracies.append(acc)

print(f"Training-set accuracy:\t\t\t{accuracies[0]*100:.2f}%")
print(f"Validation accuracy in Domain:\t\t{accuracies[1]*100:.2f}%") 
print(f"Validation accuracy out of Domain:\t{accuracies[2]*100:.2f}%") 


num_epochs = 100
learning_rate = 1e-5
# loss_fn is CrossEntropyLoss
parameters = [
    {'params': model.transformer.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.score.parameters(), 'lr': 1e-3, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(parameters)

num_training_steps=len(train_loader) * num_epochs
num_warmup_steps=int(0.1 * num_training_steps) # 10% warmup
patience = 5

scheduler = transformers.get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

early_stopper = Early_Stopper(patience=patience)

train_loss, validation_loss, validation_acc, ood_val_loss, ood_val_acc = train(
    model,
    optimizer,
    train_loader,
    val_in_domain_loader,
    device,
    num_epochs,
    early_stopper,
    scheduler=scheduler,
    ood_val_loader=val_out_of_domain_loader
)


plot_loss(train_loss, validation_loss, ood_val_loss)


path = 'Untrained_Finetuning.pth'
torch.save(model.state_dict(), path)

model_accuracies['Untrained Full FT'] = [validation_acc, ood_val_acc]


model = copy.deepcopy(no_pretraining_gpt2)

for _, param in model.named_parameters():
    torch.nn.init.zeros_(param)

for param in model.base_model.parameters(): 
    param.requires_grad = True 
for param in model.score.parameters(): 
    param.requires_grad = True


print("GPT2 Model without Fine-tuning or Pretraining (zeros only):")
accuracies = []
for dataloader in dataloaders[:3]:
    _, acc = validate(model, dataloader, device)
    accuracies.append(acc)

print(f"Training-set accuracy:\t\t\t{accuracies[0]*100:.2f}%")
print(f"Validation accuracy in Domain:\t\t{accuracies[1]*100:.2f}%") 
print(f"Validation accuracy out of Domain:\t{accuracies[2]*100:.2f}%") 


num_epochs = 100
learning_rate = 1e-5
# loss_fn is CrossEntropyLoss
parameters = [
    {'params': model.transformer.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.score.parameters(), 'lr': 1e-3, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(parameters)

num_training_steps=len(train_loader) * num_epochs
num_warmup_steps=int(0.1 * num_training_steps) # 10% warmup
patience = 5

scheduler = transformers.get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

early_stopper = Early_Stopper(patience=patience)

train_loss, validation_loss, validation_acc, ood_val_loss, ood_val_acc = train(
    model,
    optimizer,
    train_loader,
    val_in_domain_loader,
    device,
    num_epochs,
    early_stopper,
    scheduler=scheduler,
    ood_val_loader=val_out_of_domain_loader
)


plot_loss(train_loss, validation_loss, ood_val_loss)


path = 'Untrained_Finetuning_zero_init.pth'
torch.save(model.state_dict(), path)

model_accuracies['Untrained Full zero init FT'] = [validation_acc, ood_val_acc]


model = copy.deepcopy(pretrained_gpt2)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


num_epochs = 100
learning_rate = 1e-5
loss_fn = FocalLoss(alpha=0.25, gamma=2.0) # Validation loss is still Cross_Entropy to keep it comparable
parameters = [
    {'params': model.transformer.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.score.parameters(), 'lr': 1e-3, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

num_training_steps=len(train_loader) * num_epochs
num_warmup_steps=int(0.1 * num_training_steps) # 10% warmup
patience = 5

scheduler = transformers.get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

early_stopper = Early_Stopper(patience=patience)

train_loss, validation_loss, validation_acc, ood_val_loss, ood_val_acc = train(
    model,
    optimizer,
    train_loader,
    val_in_domain_loader,
    device,
    num_epochs,
    early_stopper,
    scheduler=scheduler,
    ood_val_loader=val_out_of_domain_loader,
    loss_fn=loss_fn
)


plot_loss(train_loss, validation_loss, ood_val_loss)


path = 'Pretrained_Finetuning_focal_loss.pth'
torch.save(model.state_dict(), path)

model_accuracies['Pretrained Full FT Focal Loss'] = [validation_acc, ood_val_acc]


data = []
max_epochs = num_epochs
for model, (in_acc, out_acc) in model_accuracies.items():

    for epoch, acc in enumerate(in_acc, start=1):
        data.append({'Model': model, 'Epoch': epoch, 'Accuracy': acc, 'Type': 'In-Domain'})
    for epoch, acc in enumerate(out_acc, start=1):
        data.append({'Model': model, 'Epoch': epoch, 'Accuracy': acc, 'Type': 'Out-of-Domain'})

df = pd.DataFrame(data)
csv_path = '/kaggle/working/validation_accuracies.csv'
df.to_csv(csv_path, index=False)


def get_sorted_labels_and_handles(ax, palette):
    handles, labels = ax.get_legend_handles_labels()

    # Extract the last y-values from the DataFrame for each label
    last_values = {}
    for label in labels:
        # Get the corresponding subset of data
        subset = df[df['Combined'] == label]
        last_value = subset['Accuracy'].iloc[-1]  # Get the last accuracy value
        last_values[label] = last_value

    # Sort the labels based on the last y-value
    sorted_labels = sorted(labels, key=lambda label: last_values[label], reverse=True)

    # Format labels to include the last value as a percentage with 2 digits
    formatted_labels = [f"{label} ({last_values[label]*100:.2f}%)" for label in sorted_labels]

    # Reorder the handles according to the sorted labels
    sorted_handles = [handles[labels.index(label)] for label in sorted_labels]

    return sorted_handles, formatted_labels


base_colors = sns.color_palette("tab10", n_colors=6)  # Strong colors for in-domain
pale_colors = [sns.desaturate(color, 0.5) for color in base_colors]  # Pale versions for out-of-domain
palette = {
    f"{model} In-Domain": base_colors[i] for i, model in enumerate(model_accuracies.keys())
}
palette.update({
    f"{model} Out-of-Domain": pale_colors[i] for i, model in enumerate(model_accuracies.keys())
})

# Add a combined key for hue differentiation
df['Combined'] = df['Model'] + " " + df['Type']

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.lineplot(
    data=df, 
    x='Epoch', 
    y='Accuracy', 
    hue='Combined', 
    palette=palette,
    linewidth=2,
    ax=ax1
)

plt.title("Validation Accuracy over Epochs", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Validation Accuracy", fontsize=12)
ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
sns.set_style("whitegrid")
ax2 = ax1.twinx()
ax2.set_ylabel("Validation Accuracy", fontsize=12)
ax2.set_ylim(ax1.get_ylim())
ax2.yaxis.tick_right()
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

sorted_handles, formatted_labels = get_sorted_labels_and_handles(ax1, palette)
ax1.legend(sorted_handles, formatted_labels)

plt.tight_layout()

plot_filename = '/kaggle/working/validation_accuracy_single_plot.pdf'
plt.savefig(plot_filename)

plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Assuming 'model_accuracies' and 'df' are already defined
base_colors = sns.color_palette("tab10", n_colors=6)  # Strong colors for in-domain
pale_colors = [sns.desaturate(color, 0.5) for color in base_colors]  # Pale versions for out-of-domain
palette = {
    f"{model} In-Domain": base_colors[i] for i, model in enumerate(model_accuracies.keys())
}
palette.update({
    f"{model} Out-of-Domain": pale_colors[i] for i, model in enumerate(model_accuracies.keys())
})

# Add a combined key for hue differentiation
df['Combined'] = df['Model'] + " " + df['Type']

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Plot In-Domain on the first subplot (ax1)
sns.lineplot(
    data=df[df['Type'] == 'In-Domain'], 
    x='Epoch', 
    y='Accuracy', 
    hue='Combined', 
    palette=palette,
    linewidth=2, 
    ax=ax1
)
ax1.set_title("In-Domain: Validation Accuracy over Epochs", fontsize=14)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Validation Accuracy", fontsize=12)
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))  # Format as percentage
ax1.yaxis.tick_left()
ax1.grid(True, which='both', axis='both', zorder=0)

sorted_handles, formatted_labels = get_sorted_labels_and_handles(ax1, palette)
ax1.legend(sorted_handles, formatted_labels)

# Plot Out-of-Domain on the second subplot (ax2)
sns.lineplot(
    data=df[df['Type'] == 'Out-of-Domain'], 
    x='Epoch', 
    y='Accuracy', 
    hue='Combined', 
    palette=palette,
    linewidth=2, 
    ax=ax2
)
ax2.set_title("Out-of-Domain: Validation Accuracy over Epochs", fontsize=14)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Validation Accuracy", fontsize=12)
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))  # Format as percentage
ax2.yaxis.tick_left()
sns.set_style("whitegrid")
ax2.grid(True, which='both', axis='both', zorder=0)

sorted_handles, formatted_labels = get_sorted_labels_and_handles(ax2, palette)
ax2.legend(sorted_handles, formatted_labels)

# Adjust the layout so that labels and titles don't overlap
plt.tight_layout()

plot_filename = '/kaggle/working/validation_accuracy_seperate_plots.pdf'
plt.savefig(plot_filename)

# Show the plot
plt.show()



model = copy.deepcopy(pretrained_gpt2)
weights_path = '/kaggle/working/Pretrained_Finetuning.pth'
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=model.device))


model.eval()
labels = []
with torch.no_grad():
    for batch in test_in_domain_loader:
        input_ids = batch["input_ids"].to(device) 
        attention_mask = batch["attention_mask"].to(device) 
            
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        labels.extend(preds.cpu().tolist())


df = pd.DataFrame({
    "id": range(1, len(labels) + 1),  # Generate IDs starting from 0
    "label": labels
})
df.to_csv("submission.csv", index=False)


model = copy.deepcopy(pretrained_gpt2)
weights_path = '/kaggle/working/Pretrained_Finetuning_focal_loss.pth'
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=model.device))


model.eval()
labels = []
with torch.no_grad():
    for batch in test_in_domain_loader:
        input_ids = batch["input_ids"].to(device) 
        attention_mask = batch["attention_mask"].to(device) 
            
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        labels.extend(preds.cpu().tolist())


df = pd.DataFrame({
    "id": range(1, len(labels) + 1),  # Generate IDs starting from 0
    "label": labels
})
df.to_csv("submission2.csv", index=False)

