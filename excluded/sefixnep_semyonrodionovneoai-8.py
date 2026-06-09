!pip install datasets


import requests
from urllib.parse import urlencode

import pandas as pd
import numpy as np

from torch.utils.data import Dataset as torch_dataset, DataLoader
import torch
from torch.optim import AdamW
import torch.nn as nn

from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForTokenClassification, BertPreTrainedModel, BertModel, AutoConfig, AutoModel
from typing import Dict, List, Union

from accelerate import Accelerator

from tqdm.auto import tqdm

from sklearn.metrics import f1_score

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import numpy as np

import json
import os


DATASET_PATH = "/kaggle/input/neoai-2025-intent-detection-and-slot-filling"


# Read the dataset in conll format
def readConll(path):
    allSents = []
    allComments = []
    curSent = []
    curComments = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if curSent:
                    allSents.append(curSent)
                    allComments.append(curComments)
                    curSent = []
                    curComments = []
            else:
                if line[0] != '#':
                    tok = line.split('\t')
                    if tok[-1] == 'NoLabel':
                        tok[-1] = 'O'
                    curSent.append(tok)
                else:
                    curComments.append(line)

    if curSent:
        allSents.append(curSent)
        allComments.append(curComments)

    return allSents, allComments


data_dict = {
    'train': os.path.join(DATASET_PATH, 'train.conll'),
    'val': os.path.join(DATASET_PATH, 'validation.conll'),
    'test': os.path.join(DATASET_PATH, 'test.conll')
}

parsed_data_dict = {}
for ds_type, filename in data_dict.items():
    # Read dataset
    allSents, allComments = readConll(filename)

    if ds_type == 'train':
        # Get all unique labels
        intents = []
        slots = []
        for sent in allSents:
            for word in sent:
                intents.append(word[2])
                slots.append(word[3])

        intents = list(set(intents))
        slots = list(set(slots))

        # Add slots that are not in the training set, but are present in the test and validation sets.
        slots.append('I-party_size_number')
        slots.append('I-condition_temperature')

        # Create labels dict
        id2label_intents = pd.Series(intents).to_dict()
        label2id_intents = {value: key for key, value in id2label_intents.items()}

        id2label_slots = pd.Series(slots).to_dict()
        label2id_slots = {value: key for key, value in id2label_slots.items()}

    # Save data
    parsed_data_dict[ds_type] = {'sents': allSents, 'comments': allComments}


class JointDataset(torch_dataset):
    def __init__(self, data, label2id_intents, label2id_slots):
        self.data = data
        self.label2id_intents = label2id_intents
        self.label2id_slots = label2id_slots

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Extracting tokens and labels
        tokens = [token[1] for token in example]
        slot_labels = [token[3] for token in example]
        if example[0][2] in self.label2id_intents:
          intent = self.label2id_intents[example[0][2]]
        else:
          intent = 0 # For test dataset

        encoded_slot_labels = []
        for slot in slot_labels:
            encoded_slot_labels.append(self.label2id_slots[slot])

        return {'input_text': tokens, 'intent_label': intent, 'slot_labels': encoded_slot_labels}


class JointDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, slot_pad_token_id=-100):
        super().__init__(tokenizer)
        self.slot_pad_token_id = slot_pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extracting texts for tokenization
        texts = [f['input_text'] for f in features]

        # Tokenizing text splitted on words
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt"
        )

        # Add the res part of fields
        batch['intent_labels'] = torch.tensor([f['intent_label'] for f in features])

        # Supplement slot labels to the maximum length in the batch
        max_length = batch['input_ids'].shape[1]
        slot_labels = []
        for f in features:
            labels = f['slot_labels']
            # Cut if neccessary and supplement
            padded = labels[:max_length] + [self.slot_pad_token_id] * (max_length - len(labels))
            slot_labels.append(padded)

        batch['slot_labels'] = torch.tensor(slot_labels)

        return batch


#DO NOT CHANGE THIS CELL
class JointBertModel(BertPreTrainedModel):
    def __init__(self, model_name_or_path, num_intent_labels, num_slot_labels):
        # Load model config
        config = AutoConfig.from_pretrained(model_name_or_path)
        super().__init__(config)

        # Initialize base model
        self.bert = AutoModel.from_pretrained(model_name_or_path, config=config)

        # Classificator heads
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels + 1)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels + 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, slot_labels=None, intent_labels=None):

        outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        sequence_output = outputs.last_hidden_state

        # Slot classification
        slot_logits = self.slot_classifier(sequence_output)

        # Intent classification (используем [CLS] токен)
        pooled_output = sequence_output[:, 0, :]
        intent_logits = self.intent_classifier(pooled_output)

        loss = None
        if slot_labels is not None and intent_labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            slot_loss = loss_fct(slot_logits.view(-1, self.slot_classifier.out_features),
                               slot_labels.view(-1))
            intent_loss = loss_fct(intent_logits.view(-1, self.intent_classifier.out_features),
                                 intent_labels.view(-1))
            loss = slot_loss + intent_loss

        return {
            "loss": loss,
            "slot_logits": slot_logits,
            "intent_logits": intent_logits
        }


#DO NOT CHANGE THIS CELL

# Model initialization
model_name = 'Ilseyar-kfu/base_model'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = JointBertModel(model_name_or_path=model_name, num_intent_labels=len(intents), num_slot_labels=len(slots))

optimizer = AdamW(model.parameters(), lr=2e-5)


for parameter in model.bert.embeddings.parameters():
    parameter.requieres_grad = False

for parameter in model.bert.encoder.parameters():
    parameter.requieres_grad = False


# DO NOT CHANGE THIS CODE!!!
SEED = 23

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Dataset preparation
collator = JointDataCollator(tokenizer)

train_data = parsed_data_dict['train']['sents']
train_ds = JointDataset(train_data, label2id_intents, label2id_slots)
train_loader = DataLoader(train_ds, batch_size=256, collate_fn=collator, shuffle=True)

val_data = parsed_data_dict['val']['sents']
val_ds = JointDataset(val_data, label2id_intents, label2id_slots)
val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collator, shuffle=False)

test_data = parsed_data_dict['test']['sents']
test_ds = JointDataset(test_data, label2id_intents, label2id_slots)
test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collator, shuffle=False)

accelerator = Accelerator()
model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)


# Train parameters
num_train_epochs = 3


def calculate_metrics(intent_preds, slot_preds, intent_labels, slot_labels):
    """
    Calculate f1-score for intent и slot classification
    """
    metrics = {}
    intent_preds, slot_preds, intent_labels, slot_labels = np.array(intent_preds), np.array(slot_preds), np.array(intent_labels), np.array(slot_labels)

    metrics['intent_weighted_f1'] = f1_score(intent_labels, intent_preds, average='weighted')

    # Filter paddings
    mask = slot_labels != -100
    flat_slot_preds = slot_preds[mask]
    flat_slot_true = slot_labels[mask]

    metrics['slot_weighted_f1'] = f1_score(flat_slot_true, flat_slot_preds, average='weighted')

    metrics['avg_weighted_f1'] = (metrics['intent_weighted_f1'] + metrics['slot_weighted_f1']) / 2

    return metrics


for epoch in range(num_train_epochs):
    # Training
    model.train()
    train_loss = 0
    intent_trues_epoch, intent_preds_epoch = [], []
    slot_trues_epoch, slot_preds_epoch = [], []
    for batch in tqdm(train_loader):
        outputs = model(**batch)
        loss = outputs['loss']
        train_loss += (loss.item() / batch['input_ids'].shape[0])
        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        # Save results
        intent_preds = torch.argmax(outputs['intent_logits'], dim=1).cpu().numpy()
        intent_preds_epoch.extend(intent_preds)
        intent_true = batch['intent_labels'].cpu().numpy()
        intent_trues_epoch.extend(intent_true)
        slot_preds = torch.argmax(outputs['slot_logits'], dim=2).cpu().numpy()
        slot_preds_epoch.extend(slot_preds.reshape(-1).tolist())
        slot_true = batch['slot_labels'].cpu().numpy()
        slot_trues_epoch.extend(slot_true.reshape(-1).tolist())

    # Calculate Metrics
    train_metrics = calculate_metrics(intent_preds_epoch, slot_preds_epoch, intent_trues_epoch, slot_trues_epoch)

    # Evaluation
    train_loss /= len(train_loader)
    val_loss = 0
    intent_trues_epoch, intent_preds_epoch = [], []
    slot_trues_epoch, slot_preds_epoch = [], []
    model.eval()
    for batch in val_loader:
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs['loss']
            val_loss += (loss.item() / batch['input_ids'].shape[0])

            # save results
            intent_preds = torch.argmax(outputs['intent_logits'], dim=1).cpu().numpy()
            intent_preds_epoch.extend(intent_preds)
            intent_true = batch['intent_labels'].cpu().numpy()
            intent_trues_epoch.extend(intent_true)
            slot_preds = torch.argmax(outputs['slot_logits'], dim=2).cpu().numpy()
            slot_preds_epoch.extend(slot_preds.reshape(-1).tolist())
            slot_true = batch['slot_labels'].cpu().numpy()
            slot_trues_epoch.extend(slot_true.reshape(-1).tolist())

    # calculate metrics
    val_metrics = calculate_metrics(intent_preds_epoch, slot_preds_epoch, intent_trues_epoch, slot_trues_epoch)

    val_loss /= len(val_loader)

    print(f'Epoch {epoch}:')
    print(f'Train Loss: {train_loss} Val Loss: {val_loss}')
    print(f'Train metrics: {train_metrics}')
    print(f'Val metrics: {val_metrics}')
    print()



# DO NOT CHANGE THIS CELL

# Testing
intent_preds_epoch = []
slot_preds_epoch = []
slot_trues_epoch = []
model.eval()
for batch in test_loader:
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs['loss']

        intent_preds = torch.argmax(outputs['intent_logits'], dim=1).cpu().numpy()
        intent_preds_epoch.extend(intent_preds)
        slot_preds = torch.argmax(outputs['slot_logits'], dim=2).cpu().numpy()
        slot_preds_epoch.extend(slot_preds.tolist())
        slot_true = batch['slot_labels'].cpu().numpy()
        slot_trues_epoch.extend(slot_true.tolist())


# DO NOT CHANGE THIS CELL

def prepare_submission(intent_preds, slot_preds, slot_trues):
  result_intents = [id2label_intents[intent] for intent in intent_preds]
  result_slots = []
  for slot_pred, slot_true  in zip(slot_preds, slot_trues):
    result_slot = []
    for slot in slot_pred:
      result_slot.append(id2label_slots[slot])
    pad_length = slot_true.count(-100)
    #Remove labels for padding tokens
    result_slots.append(' '.join(result_slot[:len(result_slot) - pad_length]))

  df = pd.DataFrame.from_dict({
      'id': [i+1 for i in range(0, len(result_intents))],
      'intent': result_intents,
      'slots': result_slots
  })
  import hashlib
  hsh = hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
  submit_path = f"submit_{hsh}.csv"
  print(f"SUBMIT_NAME: {submit_path}")
  print(df.head(10))

  df.to_csv(submit_path, index=False)


prepare_submission(intent_preds_epoch, slot_preds_epoch, slot_trues_epoch)

