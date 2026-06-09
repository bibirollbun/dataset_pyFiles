import os
import pandas as pd
from sklearn.model_selection import train_test_split


train_path = r"/kaggle/input/bert-classification-ioai/train.tsv"
test_path = r"/kaggle/input/bert-classification-ioai/test.tsv"
res_dataset_dir = r"corpus_normalized/"
if not os.path.exists(res_dataset_dir):
    os.makedirs(res_dataset_dir)


train_df = pd.read_csv(train_path, sep=",", encoding="utf-8")
test_df = pd.read_csv(test_path, sep=",", encoding="utf-8")
train_df, dev_df, _, _ = \
    train_test_split(train_df, train_df, test_size=0.1, random_state=1488)

train_positive_class_df = train_df[train_df['class'] == 1]
train_negative_class_df = train_df[train_df['class'] == 0]
num_positive_examples = train_positive_class_df.shape[0]
# For training set, we take the same amount of positive and negative examples
train_negative_class_df = train_negative_class_df.sample(num_positive_examples, )
# Concatenating positive and negative examples and shuffling the training set
class_normalized_train_df = pd.concat([train_positive_class_df, train_negative_class_df]).sample(frac=1)


out_train_path = os.path.join(res_dataset_dir, "train.tsv")
out_test_path = os.path.join(res_dataset_dir, "test.tsv")
out_dev_path = os.path.join(res_dataset_dir, "dev.tsv")

# class_normalized_train_df.to_csv(out_train_path, sep="\t", encoding="utf-8", index=False, )
train_df.to_csv(out_train_path, sep="\t", encoding="utf-8", index=False,)
test_df.to_csv(out_test_path, sep="\t", encoding="utf-8", index=False)
dev_df.to_csv(out_dev_path, sep="\t", encoding="utf-8", index=False, )

print(train_df.shape)
print(dev_df.shape)


import re
def list_replace(search, replacement, text):
    """
    Replaces all symbols of text which are present
    in the search string with the replacement string.
    """
    search = [el for el in search if el in text]
    for c in search:
        text = text.replace(c, replacement)
    return text

def clean_text(text):

    text = list_replace \
        ('\u00AB\u00BB\u2039\u203A\u201E\u201A\u201C\u201F\u2018\u201B\u201D\u2019', '\u0022', text)

    text = list_replace \
        ('\u2012\u2013\u2014\u2015\u203E\u0305\u00AF', '\u2003\u002D\u002D\u2003', text)

    text = list_replace('\u2010\u2011', '\u002D', text)

    text = list_replace \
            (
            '\u2000\u2001\u2002\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u202F\u205F\u2060\u3000',
            '\u2002', text)

    text = re.sub('\u2003\u2003', '\u2003', text)
    text = re.sub('\t\t', '\t', text)

    text = list_replace \
            (
            '\u02CC\u0307\u0323\u2022\u2023\u2043\u204C\u204D\u2219\u25E6\u00B7\u00D7\u22C5\u2219\u2062',
            '.', text)

    text = list_replace('\u2217', '\u002A', text)

    text = list_replace('…', '...', text)

    text = list_replace('\u00C4', 'A', text)
    text = list_replace('\u00E4', 'a', text)
    text = list_replace('\u00CB', 'E', text)
    text = list_replace('\u00EB', 'e', text)
    text = list_replace('\u1E26', 'H', text)
    text = list_replace('\u1E27', 'h', text)
    text = list_replace('\u00CF', 'I', text)
    text = list_replace('\u00EF', 'i', text)
    text = list_replace('\u00D6', 'O', text)
    text = list_replace('\u00F6', 'o', text)
    text = list_replace('\u00DC', 'U', text)
    text = list_replace('\u00FC', 'u', text)
    text = list_replace('\u0178', 'Y', text)
    text = list_replace('\u00FF', 'y', text)
    text = list_replace('\u00DF', 's', text)
    text = list_replace('\u1E9E', 'S', text)
    # Removing punctuation
    text = list_replace(',.[]{}()=+-−*&^%$#@!~;:§/\|\?"\n', ' ', text)
    # Replacing all numbers with masks
    text = list_replace('0123456789', 'x', text)

    currencies = list \
            (
            '\u20BD\u0024\u00A3\u20A4\u20AC\u20AA\u2133\u20BE\u00A2\u058F\u0BF9\u20BC\u20A1\u20A0\u20B4\u20A7\u20B0\u20BF\u20A3\u060B\u0E3F\u20A9\u20B4\u20B2\u0192\u20AB\u00A5\u20AD\u20A1\u20BA\u20A6\u20B1\uFDFC\u17DB\u20B9\u20A8\u20B5\u09F3\u20B8\u20AE\u0192'
        )

    alphabet = list \
            (
            '\t\r абвгдеёзжийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')

    allowed = set(currencies + alphabet)

    cleaned_text = [sym for sym in text if sym in allowed]
    cleaned_text = ''.join(cleaned_text)

    return cleaned_text


train_path = r"corpus_normalized/train.tsv"
dev_path = r"corpus_normalized/dev.tsv"
test_path = r"corpus_normalized/test.tsv"

# Loading data
train_df = pd.read_csv(train_path, sep='\t', encoding="utf-8",)
dev_df = pd.read_csv(dev_path, sep='\t', encoding="utf-8",)
test_df = pd.read_csv(test_path, sep='\t', encoding="utf-8",)

# Extracting tweet texts
train_tweet_texts = train_df.tweet.values
test_tweet_texts = test_df.tweet.values
dev_tweet_texts = dev_df.tweet.values

# Preprocessing training tweets
cleaned_train_texts = []
for tweet_text in train_tweet_texts:
    cleaned_text = clean_text(tweet_text).lower()
    split_cleaned_text = cleaned_text.split()
    cleaned_train_texts.append(" ".join(split_cleaned_text))

# Preprocessing test tweets
cleaned_test_texts = []
for tweet_text in test_tweet_texts:
    cleaned_text = clean_text(tweet_text)
    cleaned_test_texts.append(" ".join(cleaned_text.split()))

# Preprocessing validation tweets
cleaned_dev_texts = []
for tweet_text in dev_tweet_texts:
    cleaned_text = clean_text(tweet_text)
    cleaned_dev_texts.append(" ".join(cleaned_text.split()))

train_df["clean_text"] = cleaned_train_texts
dev_df["clean_text"] = cleaned_dev_texts
test_df["clean_text"] = cleaned_test_texts


from collections import Counter

Counter(train_df['class'])


import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW, Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader


PRE_TRAINED_MODEL_NAME = "viktoroo/sberbank-rubert-base-collection3" #'ai-forever/ruBert-base' 
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE=5e-05
NUM_WARMUP_STEPS=0


n_classes = 2


class TwitterClassifier(nn.Module):
  def __init__(self, n_classes):
    super(TwitterClassifier, self).__init__()
    self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.5)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids,
                         attention_mask=attention_mask)
    # last_hidden_state_cls = outputs[0][:, 0, :]
    last_hidden_state_cls = outputs['pooler_output']
    output = self.drop(last_hidden_state_cls)
    return self.out(output)


tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
train_tokenized = [tokenizer.encode(x, add_special_tokens=True) for x in cleaned_train_texts]
dev_tokenized = [tokenizer.encode(x, add_special_tokens=True) for x in cleaned_dev_texts]
test_tokenized = [tokenizer.encode(x, add_special_tokens=True) for x in cleaned_test_texts]


# находим самое длинное предложение
train_max_len = 0
for i in train_tokenized:
    if len(i) > train_max_len:
        train_max_len = len(i)

dev_max_len = 0
for i in dev_tokenized:
    if len(i) > dev_max_len:
        dev_max_len = len(i)

test_max_len = 0
for i in test_tokenized:
    if len(i) > test_max_len:
        test_max_len = len(i)

print(train_max_len)
print(dev_max_len)
print(test_max_len)



class TwitterDataset(Dataset):
  def __init__(self, ids, tweets, targets, tokenizer, max_len):
    self.ids = ids
    self.tweets = tweets
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.tweets)

  def __getitem__(self, item):
    tweet = str(self.tweets[item])
    target = self.targets[item]
    id = self.ids[item]
    encoding = self.tokenizer.encode_plus(
      tweet,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True,
    )
    return {
      'id': id,
      'tweet_text': tweet,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, batch_size, max_len):
  if "label" in df:
    labels = df.label.to_numpy()
  else:
    labels = [0] * len(df)
  ds = TwitterDataset(
    ids = df.id,
    tweets=df.clean_text,
    targets=labels,
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,
  )


print(test_df.columns)
train_df = train_df.rename(columns={'class': 'label'})
dev_df = dev_df.rename(columns={'class': 'label'})
print(train_df.shape)
print(dev_df.shape)
print(test_df.shape)


tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
train_data_loader = create_data_loader(train_df, tokenizer, BATCH_SIZE, train_max_len)
dev_data_loader = create_data_loader(dev_df, tokenizer, BATCH_SIZE, dev_max_len)
test_data_loader = create_data_loader(test_df, tokenizer, BATCH_SIZE, test_max_len)


import torch


n_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwitterClassifier(n_classes)
model = model.to(device)


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
  optimizer,
  num_warmup_steps=NUM_WARMUP_STEPS,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


from tqdm import tqdm

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(input_ids=input_ids,
        attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)


from collections import defaultdict
import torch

history = defaultdict(list)

best_accuracy = 0
for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer,
    device, scheduler, len(train_df))
  print(f'Train loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(model, dev_data_loader, loss_fn, device, len(dev_df))
  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)


history['train_acc'] = [train_acc.cpu() for train_acc in history['train_acc']]
history['val_acc'] = [val_acc.cpu() for val_acc in history['val_acc']]


import matplotlib.pyplot as plt

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);


import torch.nn.functional as F

def get_predictions(model, data_loader):
  model = model.eval()
  tweet_ids = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for d in data_loader:
      ids = d["id"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      outputs = F.softmax(model(
        input_ids=input_ids,
        attention_mask=attention_mask
      ))
      _, preds = torch.max(outputs, dim=1)
      tweet_ids.extend(ids)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  return tweet_ids, predictions, prediction_probs


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

tweet_ids, predicted_dev_labels, prediction_probs = get_predictions(model,dev_data_loader)
dev_labels = dev_df.label
dev_precision = precision_score(dev_labels, predicted_dev_labels)
dev_recall = recall_score(dev_labels, predicted_dev_labels)
dev_f_measure = f1_score(dev_labels, predicted_dev_labels)
dev_roc_auc = roc_auc_score(dev_labels, predicted_dev_labels)
print(f"Dev:\nPrecision: {dev_precision}\n"
        f"Recall: {dev_recall}\n"
        f"F-measure: {dev_f_measure}\n"
        f"ROC_AUC: {dev_roc_auc}")
print(classification_report(dev_labels, predicted_dev_labels))


tweet_ids, predicted_test_labels, prediction_probs = get_predictions(model, test_data_loader)


df_submit = pd.DataFrame(columns=["id", "class"])
df_submit["id"] = test_df['id'].values
df_submit["class"] = [float(x[1]) for x in prediction_probs]
df_submit["class"] = [x.item() for x in predicted_test_labels]
df_submit.to_csv("solution.csv", sep=",", index=False)




