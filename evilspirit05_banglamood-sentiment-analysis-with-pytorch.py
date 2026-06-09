! pip install -U bnlp_toolkit

! pip install bangla_stemmer

! pip install torchtext==0.6


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
from bs4 import BeautifulSoup
import re
import string
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    log_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score
)
import re
from bangla_stemmer.stemmer.stemmer import BanglaStemmer

%matplotlib inline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


my_font="/kaggle/input/elegant-and-functional-fonts/kalpurush.ttf"


wordcloud_mask=np.array(Image.open("/kaggle/input/wordcloud-mask-collection/twitter.png"))


df=pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")


df.head()


df.drop(columns=["id"],axis=1,inplace=True)


df.shape


df.isnull().sum()


df.info()


df["sentiment"].value_counts()


plt.figure(figsize=(10,5))
sns.countplot(data=df, y="sentiment",palette=["navy","crimson","darkgreen"])
plt.title("Compare Target")
plt.show()


plt.figure(figsize=(15,5))
positive=df[df["sentiment"]=="positive"]["text"].str.len()
negative=df[df["sentiment"]=="negative"]["text"].str.len()

neutral=df[df["sentiment"]=="neutral"]["text"].str.len()

plt.hist(positive, bins=40,label='Positive Data Length',color="red")
plt.hist(negative, bins=40,label='Negative Data Length',color="green")
plt.hist(neutral, bins=40,label='neutral Data Length',color="blue")
plt.title("Compare Data Length")
plt.legend()
plt.show()


with open('/kaggle/input/slang-dataset/stopwords-bn.txt', 'r', encoding='utf-8-sig') as f:
    stop_word = f.read()
    stop_word = stop_word.replace(" ", "")
    stop_word = stop_word.split('\n')
    print(stop_word)


def clean_text(text):
    row = str(text)
    row = row.replace('\n', ' ')
    row = row.replace('\t', ' ')
    row = row.replace('\\', "")
    row = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', row)
    row = re.sub(r' +', ' ', row)
    row = row.replace("।", "")
    row = re.sub(r'[১২৩৪৫৬৭৮৯০]', '', row)  # Removing Bengali digits
    row = re.sub(r'[1234567890]', '', row)   # Removing English digits
    row = row.replace('’', '')
    row = row.replace('‘', '')

    # Removing stop words
    row = row.split()
    row = [w for w in row if w not in stop_word]  # stop_word is already defined
    row = " ".join(row)

    # Bangla stemming
    stm = BanglaStemmer()
    row = " ".join([stm.stem(w) for w in row.split()])  # Stemming each word

    return row


df['text'] = df['text'].apply(clean_text)


from wordcloud import WordCloud, STOPWORDS
regex = r"[\u0980-\u09FF]+"
plt.figure(figsize=(15,15))
positive_wordcloud=df[df["sentiment"]=="positive"]
positive_text=" ".join(positive_wordcloud['text'].values.tolist())
wordcloud = WordCloud(width=800, height=800,stopwords=STOPWORDS,font_path=my_font, background_color='black',regexp=regex,max_words=800,colormap="hsv",mask=wordcloud_mask).generate(positive_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Positive Data WordCloud")
plt.axis('off')
plt.show()



regex = r"[\u0980-\u09FF]+"
plt.figure(figsize=(15,15))
negative_wordcloud=df[df["sentiment"]=="negative"]
negative_text=" ".join(negative_wordcloud['text'].values.tolist())
wordcloud = WordCloud(width=800, height=800,stopwords=STOPWORDS,font_path=my_font, background_color='black',regexp=regex,max_words=800,colormap="gnuplot2",mask=wordcloud_mask).generate(negative_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Negative Data WordCloud")
plt.axis('off')
plt.show()



regex = r"[\u0980-\u09FF]+"
plt.figure(figsize=(15,15))
neutral_wordcloud=df[df["sentiment"]=="neutral"]
neutral_text=" ".join(neutral_wordcloud['text'].values.tolist())
wordcloud = WordCloud(width=800, height=800,stopwords=STOPWORDS,font_path=my_font, background_color='black',regexp=regex,max_words=800,colormap="brg",mask=wordcloud_mask).generate(neutral_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Neutral Data WordCloud")
plt.axis('off')
plt.show()


import matplotlib.font_manager as fm

my_font = fm.FontProperties(fname="/kaggle/input/elegant-and-functional-fonts/kalpurush.ttf")

cleaned_data = df[df["sentiment"] == "positive"]["text"]

# Split the text into individual words and count their frequency
top_words = cleaned_data.str.split(expand=True).stack().value_counts().head(30)

# Create the plot using the custom font
plt.figure(figsize=(18,10))
sns.barplot(x=top_words.index, y=top_words.values, palette="tab10")
plt.title('Top 30 Most Common Words in Positive Data', fontproperties=my_font, fontsize=20, color="black")
plt.xlabel('Word', fontproperties=my_font, fontsize=15, color="black")
plt.ylabel('Frequency', fontproperties=my_font, fontsize=15, color="black")
plt.xticks(rotation=90, fontproperties=my_font, fontsize=15, color="navy")
plt.yticks(fontproperties=my_font, fontsize=15, color="navy")
plt.show()



cleaned_data = df[df["sentiment"] == "negative"]["text"]

# Split the text into individual words and count their frequency
top_words = cleaned_data.str.split(expand=True).stack().value_counts().head(30)

# Create the plot using the custom font
plt.figure(figsize=(18,10))
sns.barplot(x=top_words.index, y=top_words.values, palette="cool")
plt.title('Top 30 Most Common Words in Negative Data', fontproperties=my_font, fontsize=20, color="black")
plt.xlabel('Word', fontproperties=my_font, fontsize=15, color="black")
plt.ylabel('Frequency', fontproperties=my_font, fontsize=15, color="black")
plt.xticks(rotation=90, fontproperties=my_font, fontsize=15, color="navy")
plt.yticks(fontproperties=my_font, fontsize=15, color="navy")
plt.show()



cleaned_data = df[df["sentiment"] == "neutral"]["text"]

# Split the text into individual words and count their frequency
top_words = cleaned_data.str.split(expand=True).stack().value_counts().head(30)

# Create the plot using the custom font
plt.figure(figsize=(18,10))
sns.barplot(x=top_words.index, y=top_words.values, palette="bwr")
plt.title('Top 30 Most Common Words in Neutral Data', fontproperties=my_font, fontsize=20, color="black")
plt.xlabel('Word', fontproperties=my_font, fontsize=15, color="black")
plt.ylabel('Frequency', fontproperties=my_font, fontsize=15, color="black")
plt.xticks(rotation=45, fontproperties=my_font, fontsize=15, color="navy")
plt.yticks(fontproperties=my_font, fontsize=15, color="navy")
plt.show()



avg_len=df["text"].apply(len)
avg_len=avg_len.mean()
print(f"Average Text Length is : {avg_len:.2f}")


df.head()


import torch
from torchtext import data
import spacy
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
import random


from bnlp import BasicTokenizer
from tabulate import tabulate


def bangla_tokenizer(text):
    tokenizer = BasicTokenizer()
    return tokenizer.tokenize(text)
TEXT = data.Field(tokenize=bangla_tokenizer, batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.long,batch_first=True,sequential=False)


fields =  {'text': ('text', TEXT), 'sentiment': ('label', LABEL)}
df_path="/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv"
training_data = data.TabularDataset(path=df_path,
                                    format="csv",
                                    fields=fields,
                                    skip_header=False)


print(vars(training_data.examples[5]))



seed=42
train_data,test_data = training_data.split(split_ratio=0.70,random_state=random.seed(seed))


TEXT.build_vocab(train_data,min_freq=2)

LABEL.build_vocab(train_data)


TEXT.vocab.freqs.most_common(10)


print("Size of text vocab:", len(TEXT.vocab))
print("Size of label vocab:", len(LABEL.vocab))
print("Label vocab:", LABEL.vocab.itos)
print("Most common tokens:", TEXT.vocab.freqs.most_common(10))


train_data,validation_data = data.BucketIterator.splits((train_data,test_data),batch_size = 64,
                             sort_key = lambda x:len(x.text),
                             sort_within_batch = True,
                             device = device)


# Access the batch object
batch = next(iter(train_data))

# Print the shape of the text in the batch
print("Batch text shape:", batch.text[0].shape)

# Print the lengths of each sentence in the batch
print("Batch text lengths:", batch.text[1])

# Print the sentiment (label) of the batch
print("Batch sentiment (labels):", batch.label)

# Print the unique labels in the batch
print("Unique labels in batch:", batch.label.unique())




print(train_data.dataset.fields['label'].vocab.stoi)



label_name=["Positive","Negative","Neutral"]


class GRUNet(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(GRUNet, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          dropout=dropout,
                          batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # Adjust for bidirectional
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # No need for sigmoid here
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)  # Adjusted here
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        
        packed_output, hidden_state = self.gru(packed_embedded)
        
        hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        
        dense_output = self.fc1(hidden)
        dense_output = torch.relu(dense_output)
        
        dense_output = self.fc2(dense_output)
        dense_output = torch.relu(dense_output)
        
        dense_output = self.fc3(dense_output)
        
        return dense_output  # Raw logits (no sigmoid)



vocab_size=len(TEXT.vocab) 
embedding_dim=100
hidden_dim=128
output_dim=3
n_layers=2
bidirectional=True
dropout=0.2

model=GRUNet(vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout)
model


import torch.optim as optim
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=5e-5)

criterion = nn.CrossEntropyLoss()


def calculate_accuracy(predictions, labels):
    _, predicted_classes = torch.max(predictions, 1)
    correct_matches = (predicted_classes == labels).float()
    accuracy = correct_matches.sum() / len(correct_matches)
    return accuracy


def train_model(model, data_iterator, optim, loss_fn):
    total_loss = 0.0
    total_acc = 0.0
    model.train()

    for data_batch in tqdm(data_iterator, desc="Training", unit="batch"):
        optim.zero_grad()
        inputs, lengths = data_batch.text
        labels = data_batch.label  # Change from `data_batch.type` to `data_batch.label`
        outputs = model(inputs, lengths)  # Outputs will have shape (batch_size, num_classes)
        
        loss = loss_fn(outputs, labels)  # CrossEntropyLoss expects raw logits and target class indices
        loss.backward()
        
        accuracy = calculate_accuracy(outputs, labels)
        optim.step()

        total_loss += loss.item()
        total_acc += accuracy

    # Return average loss and accuracy
    return total_loss / len(data_iterator), total_acc / len(data_iterator)



def validate(net, data_iter, loss_fn):
    total_loss = 0.0
    total_accuracy = 0.0
    net.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient tracking for validation
        for data in tqdm(data_iter, desc="Validation", unit="batch"):
            input_data, input_lengths = data.text
            targets = data.label.long()  # Ensure targets are in the correct format
            
            # Get predictions from the model (raw logits)
            preds = net(input_data, input_lengths)  # Output shape should be (batch_size, num_classes)

            # Calculate loss
            loss = loss_fn(preds, targets)  # CrossEntropyLoss expects raw logits
            acc = calculate_accuracy(preds, targets)  # Compute accuracy
            
            # Accumulate the loss and accuracy for the batch
            total_loss += loss.item()
            total_accuracy += acc

    # Return average loss and accuracy over all batches
    avg_loss = total_loss / len(data_iter)
    avg_accuracy = total_accuracy / len(data_iter)
    
    return avg_loss, avg_accuracy


# Early stopping function
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.best_epoch = 0
        self.wait = 0
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True

# Training loop with early stopping
EPOCH_NUMBER = 500
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

results = []
early_stopping = EarlyStopping(patience=3, delta=0.001)  # Patience of 3 epochs, minimal delta of 0.001

for epoch in range(1, EPOCH_NUMBER + 1):
    train_loss, train_acc = train_model(model, train_data, optimizer, criterion)
    valid_loss, valid_acc = validate(model, validation_data, criterion)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)
    
    results.append([epoch, f"{train_loss:.3f}", f"{train_acc*100:.2f}%", f"{valid_loss:.3f}", f"{valid_acc*100:.2f}%"])

    # Check early stopping condition
    early_stopping(valid_loss, epoch)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Display results as a table
print(tabulate(results, headers=["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"], tablefmt="fancy_grid"))




# Convert tensors to NumPy before plotting
train_accuracies = np.array([x.cpu().numpy() if hasattr(x, 'cpu') else x for x in train_accuracies])
valid_accuracies = np.array([x.cpu().numpy() if hasattr(x, 'cpu') else x for x in valid_accuracies])

# Plot loss curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid()

# Plot accuracy curves
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, len(valid_accuracies) + 1), valid_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



def get_predictions_and_labels(net, data_iter, loss_fn):
    y_true = []
    y_pred = []
    
    net.eval()
    
    with torch.no_grad():
        for data in tqdm(data_iter, desc="Validation", unit="batch"):
            input_data, input_lengths = data.text

            targets=data.label.long()
            preds=net(input_data,input_lengths)

            pred_classes=torch.argmax(preds,dim=1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(pred_classes.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return y_true, y_pred

y_true, y_pred = get_predictions_and_labels(model, validation_data, criterion)



cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="gist_stern", xticklabels=label_name, yticklabels=label_name)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


print(classification_report(y_true,y_pred,target_names=label_name))



accuracy = accuracy_score(y_true, y_pred)
plt.plot([])
plt.text(0, 0, f'Accuracy Score: {accuracy:.4f}', fontsize=16, ha='center', va='center', color="indigo")
plt.axis('off')
plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()


def get_predictions_and_labels(net, data_iter):
    y_true = []
    y_pred_prob = []
    
    net.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        for data in tqdm(data_iter, desc="Validation", unit="batch"):
            input_data, input_lengths = data.text  # Extract inputs
            targets = data.label.long()  # Extract true labels
            
            logits = net(input_data, input_lengths)  # Raw outputs from the model
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # Convert to probability scores
            
            y_true.extend(targets.cpu().numpy())  # Append true labels
            y_pred_prob.extend(probs)  # Append predicted probabilities
    
    return np.array(y_true), np.array(y_pred_prob)

# Get predictions and true labels
y_true, y_pred_prob = get_predictions_and_labels(model, validation_data)



roc_auc = roc_auc_score(y_true, y_pred_prob,multi_class="ovo")

plt.plot([])
plt.text(0, 0, f'ROC AUC Score: {roc_auc:.4f}', fontsize=16, ha='center', va='center', color="indigo")
plt.axis('off')
plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()


classes = [0, 1,2]

logarithm_loss = log_loss(y_true, y_pred_prob, labels=classes)

plt.plot([])
plt.text(0, 0, f'Log Loss: {logarithm_loss:.4f}', fontsize=16, ha='center', va='center', color="black")
plt.axis('off')

plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()


kappa = cohen_kappa_score(y_true,y_pred)
plt.plot([])
plt.text(0,0, f'Cohen Kappa Score: {kappa:.4f}', fontsize=16, ha='center', va='center',color="orangered")
plt.axis('off')

# Set the x-axis limits
plt.xlim(-1, 1)
plt.ylim(-1,1)

plt.show()


mcc = matthews_corrcoef(y_true,y_pred)

# Create a plot and display the MCC value as text
plt.plot([])
plt.text(0,0, f'Matthews Correlation Coefficient: {mcc:.4f}', fontsize=16, ha='center', va='center',color="saddlebrown")
plt.axis('off')

# Set the x-axis limits
plt.xlim(-1, 1)
plt.ylim(-1,1)

plt.show()



from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Define class labels
class_labels = label_name
num_classes = len(class_labels)

# Ensure y_true is an array of labels (0,1,2,3) and y_pred_prob is probability predictions
y_true, y_pred_prob = get_predictions_and_labels(model, validation_data)

# Convert y_true into one-hot encoding
y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

# Plot ROC curve for each class
plt.figure(figsize=(15, 10))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')

# Plot reference diagonal
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

# Show plot
plt.show()




from sklearn.metrics import precision_recall_curve, average_precision_score

# Define class labels
class_labels = label_name
num_classes = len(class_labels)

# Ensure y_true is an array of labels (0,1,2,3) and y_pred_prob is probability predictions
y_true, y_pred_prob = get_predictions_and_labels(model, validation_data)

# Convert y_true into one-hot encoding
y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

# Plot Precision-Recall curve for each class
plt.figure(figsize=(15, 10))

for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
    avg_precision = average_precision_score(y_true_bin[:, i], y_pred_prob[:, i])

    plt.plot(recall, precision, lw=2, label=f'{class_labels[i]} (AP = {avg_precision:.2f})')
    plt.fill_between(recall, precision, alpha=0.2)

# Labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multiclass Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)

# Show plot
plt.show()




from torchtext.data import Field, LabelField, Example, Dataset, Iterator


def bangla_tokenizer(text):
    tokenizer = BasicTokenizer()
    return tokenizer.tokenize(text)

TEXT = Field(
    tokenize=bangla_tokenizer,
    lower=False,
    include_lengths=True,
    batch_first=True
)
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

custom_data = [
    ("আপনি আজকে অনেক ভালো কাজ করেছেন", 1),
    ("আজকের কাজটি মোটেই ভালো হয়নি", 0),
    ("আজকের আবহাওয়া একটু ভালো, তবে মেঘলা রয়েছে", 1)
]

examples = [Example.fromlist([text, label], fields=[('text', TEXT), ('label', LABEL)]) 
            for text, label in custom_data]

custom_dataset = Dataset(examples, fields=[('text', TEXT), ('label', LABEL)])

TEXT.build_vocab(custom_dataset, min_freq=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sort_key(ex):
    return len(ex.text)

custom_iterator = Iterator(
    custom_dataset,
    batch_size=1,
    device=device,
    sort=True,
    sort_key=sort_key,
    sort_within_batch=True
)

def predict_custom_sentences(net, data_iter):
    net.eval()
    predictions = []
    pred_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_iter, desc="Predicting", unit="batch"):
            input_data, input_lengths = batch.text
            
            if input_lengths.dim() > 1:
                input_lengths = input_lengths.squeeze(-1)
            input_lengths = input_lengths.cpu()
            
            try:
                outputs = net(input_data, input_lengths).squeeze(1)
                probs = torch.softmax(outputs, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1)
                
                predictions.extend(pred_classes.cpu().numpy())
                pred_probs.extend(probs.cpu().numpy())
            except RuntimeError as e:
                print(f"Error processing batch. Text shape: {input_data.shape}, Lengths shape: {input_lengths.shape}")
                print(f"Lengths values: {input_lengths}")
                raise e
    
    return predictions, pred_probs

predictions, pred_probs = predict_custom_sentences(model, custom_iterator)

sentiment_map = {0: "negative", 1: "positive", 2: "neutral"}

for (text, _), pred, probs in zip(custom_data, predictions, pred_probs):
    label = sentiment_map[pred]
    confidence = probs[pred]
    print(f"Sentence: {text}")
    print(f"Prediction: {label} (confidence: {confidence:.4f})")
    print("#" * 60)


import torch
from torchtext.data import BucketIterator
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sentiment_map = {0: "positive", 1: "negative", 2: "neutral"}

def get_predictions_for_full_dataset(model, data_iter, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_iter:
            text, text_lengths = batch.text
            text = text.to(device)
            text_lengths = text_lengths.to(device)
            preds = model(text, text_lengths)
            pred_classes = torch.argmax(preds, dim=1)
            pred_labels = [sentiment_map[pred.item()] for pred in pred_classes]
            predictions.extend(pred_labels)

    return predictions

df_path = "/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv"
full_dataset = data.TabularDataset(path=df_path, format="csv", fields=fields, skip_header=False)

full_data_iter = BucketIterator(full_dataset, batch_size=64, sort_key=lambda x: len(x.text), sort_within_batch=True, device=device)

model = model.to(device)

predictions = get_predictions_for_full_dataset(model, full_data_iter, device)

submission_df = pd.DataFrame({
    'id': range(len(predictions)),
    'sentiment': predictions
})

submission_df.to_csv('submission.csv', index=False)
print("Predictions saved to 'submission.csv'")



submission_df.head()




