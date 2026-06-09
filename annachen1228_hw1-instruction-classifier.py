import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os
import warnings
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import logging
import matplotlib.pyplot as plt
from torch import cuda
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn import metrics
import numpy as np
import random

# Setup
device = 'cuda' if cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

# Print CUDA availability
print(torch.cuda.is_available())


def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_seed(42)


# Load data
trainPath = '/kaggle/input/hw-1-instruction-classifier/train.csv'
train_data = pd.read_csv(trainPath)

# Split validation data from training data
valid_data = train_data.iloc[:len(train_data)//10]
train_data = train_data.iloc[len(train_data)//10:]

print(f"Data Lenght:\n Train:{len(train_data)}\n Vaild:{len(valid_data)}")
print(f"Train data: \n{train_data.head(1)}")
print(f"Vaild data: \n{valid_data.head(1)}")


# You can modify it
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
LEARNING_RATE = 0.0005
MODEL_NAME = "bert-base-uncased"
EPOCHS = 4 

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Create dataset class
class TrainData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.command
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.float)
        }


# Print dataset shapes
print("TRAIN Dataset: {}".format(train_data.shape))
print("VALID Dataset: {}".format(valid_data.shape))
# Create dataset objects
training_set = TrainData(train_data, tokenizer, MAX_LEN)
valid_set = TrainData(valid_data, tokenizer, MAX_LEN)


# Setup data loaders
train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

valid_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

training_loader = DataLoader(training_set, **train_params)
valid_loader = DataLoader(valid_set, **valid_params)
print("done")


# Define model architecture
class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = torch.nn.Dropout(0.2)
        self.pre_classifier = torch.nn.Linear(768, 256)       
        self.classifier = torch.nn.Linear(256, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


# Initialize model
model = ModelClass()
model.to(device)

# Define loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
## SGD
optimizer = torch.optim.SGD(
    params=model.parameters(), 
    lr=LEARNING_RATE,  
    momentum=0  
)



def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

# Training function
def train(epoch):
    tr_loss, n_correct, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0
    all_preds = []
    all_targets = []
    model.train()
    progress_bar = tqdm(enumerate(training_loader), total=len(training_loader), desc=f"Epoch {epoch}")
    for _,data in progress_bar:
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)
        
        # 收集預測和目標值用於計算指標
        all_preds.extend(big_idx.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item(), accuracy=(n_correct * 100) / nb_tr_examples)

    # 計算精確率、召回率和F1分數
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples

    return epoch_accu, epoch_loss, precision, recall, f1

# Validation function
def valid(model, valid_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validating")
        for _, data in progress_bar:
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)
            
            all_preds.extend(big_idx.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            progress_bar.set_postfix(loss=loss.item(), accuracy=(n_correct * 100) / nb_tr_examples)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    
    return epoch_accu, epoch_loss, precision, recall, f1, all_preds, all_targets


# Training and validation loop
model_to_save = model
model_acc = 0
model_epoch = 0
model_loss = 100
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(EPOCHS):
    train_acc, train_loss, train_precision, train_recall, train_f1 = train(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    valid_acc, valid_loss, valid_precision, valid_recall, valid_f1, vaild_pred, vaild_target = valid(model, valid_loader)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)
    
    print("Accuracy on valid data = %0.2f%%" % valid_acc)
    
    if train_acc >= model_acc and train_loss <= model_loss:
        model_to_save = model
        model_acc = train_acc
        model_epoch = epoch
        model_loss = train_loss
        print(f"{epoch} model is better")

print(f'All files saved. {model_epoch} model is best')


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, 'b-', marker='o', label='Training Loss')
plt.plot(range(1, EPOCHS+1), valid_losses, 'r-', marker='s', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_accuracies, 'b-', marker='o', label='Training Accuracy')
plt.plot(range(1, EPOCHS+1), valid_accuracies, 'r-', marker='s', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()


valid_acc, valid_loss, valid_precision, valid_recall, valid_f1, vaild_pred, vaild_target = valid(model, valid_loader)
# Generate classification report
report = classification_report(vaild_target, vaild_pred, digits=4)  # digits=4 controls the number of decimal places
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(vaild_target, vaild_pred)
print(conf_matrix)


cm_display = metrics.ConfusionMatrixDisplay(
                confusion_matrix = conf_matrix, 
                display_labels = [0, 1, 2])
cm_display.plot()
plt.show()


testPath = '/kaggle/input/hw-1-instruction-classifier/test_no_label.csv'
test_data = pd.read_csv(testPath)
print(f"Test data: {test_data.head(1)}")
print(f"Test Dataset: {format(test_data.shape)}")


# Create dataset class
class TestData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.command
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }


testing_set = TestData(test_data, tokenizer, MAX_LEN)
test_params = {
    'batch_size': TEST_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 0
}

testing_loader = DataLoader(testing_set, **test_params)


# Prediction function
def predict_my(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    output_list = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            big_val = torch.argmax(outputs.data)
            output_list.append(big_val.item())
    
    return output_list


output_list = predict_my(model_to_save, testing_loader)
print("done")


test_data['label'] = output_list

submission = test_data[['ID', 'label']]

submission.to_csv("/kaggle/working/output.csv", index=False)

print(pd.read_csv("/kaggle/working/output.csv").head())

