!ls


!pwd


import os


import pandas as pd


train_df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
sub_df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/sample_submission.csv')



print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample submission:", sub_df.shape)



import os
print(os.listdir('/kaggle/input/quora-insincere-questions-classification'))




import pandas as pd


data_dir = '../input/quora-insincere-questions-classification'
train_fname = f'{data_dir}/train.csv'
test_fname = f'{data_dir}/test.csv'
sub_fname = f'{data_dir}/sample_submission.csv'



raw_df = pd.read_csv(train_fname)
test_df = pd.read_csv(test_fname)
sub_df = pd.read_csv(sub_fname)



raw_df


raw_df.sample(10)


test_df


sub_df


SAMPLE_SIZE = 100_000
sample_df = raw_df.sample(100_000)


sample_df


sample_df.target.value_counts(normalize=True)


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = SnowballStemmer(language='english')


english_stopwords = stopwords.words('english')


", ".join(english_stopwords)


def tokenize(text):
  return [stemmer.stem(token) for token in word_tokenize(text)]


vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words=english_stopwords, max_features=1000)


sample_df.question_text


%%time
vectorizer.fit(sample_df.question_text)


vectorizer.get_feature_names_out()[:100]


%%time
inputs = vectorizer.transform(sample_df.question_text)


inputs.shape


inputs


inputs[0].toarray()[:50]


%%time
test_inputs = vectorizer.transform(test_df.question_text)


test_inputs.shape


from sklearn.model_selection import train_test_split


targets = sample_df.target


train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, targets, test_size = 0.3)


train_inputs.shape


val_inputs.shape


train_targets


val_targets


import torch


train_input_tensors = torch.tensor(train_inputs.toarray()).float()
val_input_tensors = torch.tensor(val_inputs.toarray()).float()


train_input_tensors.shape


val_input_tensors.shape


type(train_targets)


train_target_tensors = torch.tensor(train_targets.values).float()
val_target_tensors = torch.tensor(val_targets.values).float()


test_input_tensors = torch.tensor(test_inputs.toarray()).float()


from torch.utils.data import TensorDataset, DataLoader


train_ds = TensorDataset(train_input_tensors, train_target_tensors)
val_ds = TensorDataset(val_input_tensors, val_target_tensors)
test_ds = TensorDataset(test_input_tensors)


BATCH_SIZE = 128


train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size = BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size = BATCH_SIZE)


for batch in train_dl:
  batch_inputs = batch[0]
  batch_targets = batch[1]
  print('batch_inputs.shape', batch_inputs.shape)
  print('batch_targets.shape', batch_targets.shape)
  break


len(train_dl)


547*128


import torch.nn as nn
import torch.nn.functional as F


class QuoraNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(1000, 512)
    self.layer2 = nn.Linear(512, 256)
    self.layer3 = nn.Linear(256, 128)
    self.layer4 = nn.Linear(128, 1)
    pass

  def forward(self, inputs):
    out = self.layer1(inputs)
    out = F.relu(out)
    out = self.layer2(out)
    out = F.relu(out)
    out = self.layer3(out)
    out = F.relu(out)
    out = self.layer4(out)
    return out



model = QuoraNet()


from sklearn.metrics import accuracy_score, f1_score


for batch in train_dl:
  bi, bt = batch
  print('inputs.shape', bi.shape)
  print('targets.shape', bt.shape)

  bo = model(bi)
  print('bo.shape', bo.shape)

  # Convert outputs to probabilities
  probs = torch.sigmoid(bo[:, 0])
  print('probs', probs[:10])

  #Convert probs to predictions
  preds = (probs > 0.5).int()
  print('preds', preds[:10])
  print('targets', bt[:10])

  # Check metrics
  print('accuracy', accuracy_score(bt, preds))
  print('f1_score', f1_score(bt, preds))

  #Loss
  print('loss', F.binary_cross_entropy(preds.float(), bt))

  break


bi[:10]


bt[:10]


bo[:10]


# Evaluate model performance
def evaluate(model, dl):
  losses, accs, f1s = [], [], []
  # Loop over batches
  for batch in dl:
    #Get inputs through model
    inputs, targets = batch

    # Pass inputs through model
    outputs = model(inputs)

    # Convert to probabilities
    probs = torch.sigmoid(outputs[:, 0])

    # Compute loss
    loss = F.binary_cross_entropy(probs, targets, weight=torch.tensor(20))

    # Compute preds
    preds = (probs > 0.5).int()

    # Compute accuracy
    acc = accuracy_score(targets, preds)

    # Compute f1 score
    f1 = f1_score(targets, preds)

    losses.append(loss.item())
    accs.append(acc)
    f1s.append(f1)

  return (torch.mean(torch.tensor(losses)).item(),
          torch.mean(torch.tensor(accs)).item(),
          torch.mean(torch.tensor(f1s)).item())


evaluate(model, train_dl)


evaluate(model, val_dl)


# Train the model batch by batch
def fit(epochs, lr, model, train_dl, val_dl ):
  history = []
  optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)

  for epoch in range(epochs):
    #Training phase
    for batch in train_dl:
      #Get inputs and targets
      inputs, targets = batch

      #GEt model outputs
      outputs = model(inputs)

      #Get probabilities
      probs = torch.sigmoid(outputs[:, 0])

      #Compute loss
      loss = F.binary_cross_entropy(probs, targets, weight=torch.tensor(20))

      #Perform the optimization
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

   # Evaluation phase
    loss, acc, f1 = evaluate(model, val_dl)
    print('Epoch {}; Loss: {:.4f}; Accuracy: {:.4f}; F1 Score: {:.4f}'.format(
         epoch+1, loss, acc, f1))
    history.append([loss, acc, f1])
  return history





model = QuoraNet()


history = []


history.append(evaluate(model, train_dl))


history


history += fit(5, 0.0001, model, train_dl, val_dl)


history


losses = [item[0] for item in history]
accs = [item[1] for item in history]
f1s = [item[2] for item in history]


import matplotlib.pyplot as plt


plt.title('Loss')
plt.plot(losses)


plt.title('Accuracy')
plt.plot(accs)


plt.title('F1 Score')
plt.plot(f1s)


import os
print(os.listdir())



small_df = raw_df.sample(20)


small_df



def predict_df(df):
  inputs = vectorizer.transform(df.question_text)
  inputs_tensors = torch.tensor(inputs.toarray()).float()
  outputs = model(inputs_tensors)
  probs = torch.sigmoid(outputs[:, 0])
  preds = (probs > 0.5).int()
  return preds


small_df.target.values


predict_df(small_df)


small_df.question_text.values


def predict_text(text):
  df = pd.DataFrame({'question_text': [text]})
  inputs = vectorizer.transform(df.question_text)
  inputs_tensors = torch.tensor(inputs.toarray()).float()
  outputs = model(inputs_tensors)
  probs = torch.sigmoid(outputs[:, 0])
  preds = (probs > 0.5).int()
  return preds


predict_text("What is the function of a plasma cell?")


predict_text("Why can't liberals realize that they're stupid?")


test_inputs


import numpy as np
def make_preds(dl):
  all_preds = []
  for batch in dl:
    inputs = batch[0]
    outputs = model(inputs)
    probs = torch.sigmoid(outputs[:, 0])
    preds = (probs > 0.5).int()
    all_preds.append(preds.detach().numpy())

  return np.concatenate(all_preds)


for batch in test_dl:
  print(batch[0][0])
  break


test_preds = make_preds(test_dl)


len(test_preds)


test_preds


sub_df


sub_df['prediction'] = test_preds



pd.Series(test_preds).value_counts()


sub_df.prediction.value_counts()


sub_df.to_csv('submission.csv', index=False)



import pandas as pd

df = pd.read_csv('submission.csv')
print("Submission shape:", df.shape)
print("Columns:", df.columns)
print(df.head())
print(df['prediction'].value_counts())



import pandas as pd

# Load test data
test_df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')

# Create the correct submission file
submission = pd.DataFrame({
    'qid': test_df['qid'],
    'prediction': test_preds  # this must be a list/array of 0s and 1s
})

# Save to CSV
submission.to_csv('submission.csv', index=False)


