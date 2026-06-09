#pip install pyspellchecker


pip install wordsegment


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expressions
import matplotlib.pyplot as plt # graph creation
import emoji # checking for emojis
import nltk
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report,  roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import torch
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
#from nltk.stem import WordNetLemmatizer
#from spellchecker import SpellChecker
#import optuna
from nltk.corpus import words
from wordsegment import load, segment


def set_seed(seed=94):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


english_words = set(words.words())

def eda(train_set, val_set, test_set):
    
    # check that all sets have no missing data
    missing = False
    for set in [train_set, val_set, test_set]:
        if(set.isnull().sum().any() != 0):
            missing = True

    print('Missing values detected in one of the sets' if missing else 'No missing values in all sets')

    # count how much data in each set
    print('training set size: ', len(train_set))

    print('validation set size: ', len(val_set))

    print('test set size: ', len(test_set))

    # calculate total amount of data
    total_size = len(train_set) + len(val_set) + len(test_set)
    print('total number of tweets: ', total_size)

    # calculate percentage of each set
    train_perc = (len(train_set)/total_size)*100
    print('percentage of training set: ', train_perc)
    val_perc = (len(val_set)/total_size)*100
    print('percentage of validation set: ', val_perc)
    test_perc = (len(test_set)/total_size)*100
    print('percentage of test set: ', test_perc)
    
    # plot results
    percentages = np.array([train_perc, val_perc, test_perc])
    my_colors = ['#0f6589','#67ac92', '#ffe1ad']
    my_labels = ['Training Set', 'Validation Set', 'Test Set']
    plt.pie(percentages, labels=my_labels, colors=my_colors)
    plt.title('Distribution Of Data In Sets')
    plt.legend(loc='upper right')
    plt.show()
    
    # check sentiment distribution in training set
    positive, negative = train_set['Label'].value_counts()
    pos_perc, neg_perc = train_set['Label'].value_counts(normalize=True)
    print('Percentage of positive tweets in training set: ', pos_perc)
    print('Percentage of negative tweets in training set: ', neg_perc)

    sentiment_labels = np.array(['Positive', 'Negative'])
    values = np.array([positive, negative])

    plt.bar(sentiment_labels, values, width=0.5, color=['#51ac51','#e83f3f'])
    plt.title('Distribution Of Sentiment In Training Set')
    plt.show()

    # check sentiment distribution in validation set
    positive, negative = val_set['Label'].value_counts()
    pos_perc, neg_perc = val_set['Label'].value_counts(normalize=True)
    print('Percentage of positive tweets in validation set: ', pos_perc)
    print('Percentage of negative tweets in validation set: ', neg_perc)

    sentiment_labels = np.array(['Positive', 'Negative'])
    values = np.array([positive, negative])

    plt.bar(sentiment_labels, values, width=0.5, color=['#51ac51','#e83f3f'])
    plt.title('Distribution Of Sentiment In Validation Set')
    plt.show()

    # compare length of positive vs negative tweets in training set
    train_set['Length'] = train_set['Text'].apply(len)

    negative_lengths = np.array(train_set[train_set['Label'] == 0]['Length'])
    positive_lengths = np.array(train_set[train_set['Label'] == 1]['Length'])
    
    plt.hist(negative_lengths, color='#0f6589')
    plt.title('Lengths Of Negative Tweets in Training Set')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(positive_lengths, color='#0f6589')
    plt.title('Lengths Of Positive Tweets in Training Set')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()

    # compare length of positive vs negative tweets in validation set
    val_set['Length'] = val_set['Text'].apply(len)

    negative_lengths = np.array(val_set[val_set['Label'] == 0]['Length'])
    positive_lengths = np.array(val_set[val_set['Label'] == 1]['Length'])
    
    plt.hist(negative_lengths, color='#67ac92')
    plt.title('Lengths Of Negative Tweets in Validation Set')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(positive_lengths, color='#67ac92')
    plt.title('Lengths Of Positive Tweets in Validation Set')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()

    # make wordclouds to show most common words in each set
    text =' '.join(str(x) for x in train_set['Text'])
    train = WordCloud().generate(text)
    plt.imshow(train)
    plt.title('Most Common Words In Training Set')
    plt.axis('off')
    plt.show()

    text =' '.join(str(x) for x in val_set['Text'])
    val = WordCloud().generate(text)
    plt.imshow(val)
    plt.title('Most Common Words In Validation Set')
    plt.axis('off')
    plt.show()

    text =' '.join(str(x) for x in test_set['Text'])
    test = WordCloud().generate(text)
    plt.imshow(test)
    plt.title('Most Common Words In Test Set')
    plt.axis('off')
    plt.show()

    # make wordclouds to show most common words in positive vs negative tweets in training set
    text =' '.join(str(x) for x in train_set[train_set['Label'] == 1]['Text'])
    positive_train = WordCloud().generate(text)
    plt.imshow(positive_train)
    plt.title('Most Common Words In Positive Tweets In Training Set')
    plt.axis('off')
    plt.show()

    text =' '.join(str(x) for x in train_set[train_set['Label'] == 0]['Text'])
    negative_train = WordCloud().generate(text)
    plt.imshow(negative_train)
    plt.title('Most Common Words In Negative In Training Set')
    plt.axis('off')
    plt.show()

    # make wordclouds to show most common words in positive vs negative tweets in validation set
    text =' '.join(str(x) for x in val_set[val_set['Label'] == 1]['Text'])
    positive_val = WordCloud().generate(text)
    plt.imshow(positive_val)
    plt.title('Most Common Words In Positive Tweets In Validation Set')
    plt.axis('off')
    plt.show()

    text =' '.join(str(x) for x in val_set[val_set['Label'] == 0]['Text'])
    negative_val = WordCloud().generate(text)
    plt.imshow(negative_val)
    plt.title('Most Common Words In Negative Tweets In Validation Set')
    plt.axis('off')
    plt.show()

    # regex to find emoticons
    emoticons = r"[:;=xX8]-?[)DdpP/(|oO*']"
    
    for set in [train_set, val_set, test_set]:
        set['Emojis'] = set['Text'].apply(lambda text: sum(1 for char in text if emoji.is_emoji(char)))
        set['Emoticons'] = set['Text'].apply(lambda text: len(re.findall(emoticons, text)))
        set['Links'] = set['Text'].apply(lambda text: len(re.findall(r"http[s]?://\S+", text)))
        set['Mentions'] = set['Text'].apply(lambda text: len(re.findall(r"@\w+", text)))
        set['Has_Emojis'] = set['Emojis'] > 0
        set['Has_Emoticons'] = set['Emoticons'] > 0
        set['Has_Links'] = set['Links'] > 0
        set['Has_Mentions'] = set['Mentions'] > 0

    train_counts = [train_set['Has_Emojis'].sum(), train_set['Has_Emoticons'].sum(), train_set['Has_Links'].sum(), train_set['Has_Mentions'].sum()]
    val_counts = [val_set['Has_Emojis'].sum(), val_set['Has_Emoticons'].sum(), val_set['Has_Links'].sum(), val_set['Has_Mentions'].sum()]
    test_counts = [test_set['Has_Emojis'].sum(), test_set['Has_Emoticons'].sum(), test_set['Has_Links'].sum(), test_set['Has_Mentions'].sum()]
    categories = ['Emojis', 'Emoticons', 'Links', 'Mentions']
    
    # Create graph to show data
    x = np.arange(len(categories))
    width = 0.25  
    
    fig, ax = plt.subplots()
    ax.bar(x - width, train_counts, width, label='Training Set', color='#0f6589')
    ax.bar(x, val_counts, width, label='Validation Set', color='#67ac92')
    ax.bar(x + width, test_counts, width, label='Test Set', color='#ffe1ad')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Tweets')
    ax.set_title('Number Of Tweets With Emojis, Emoticons, Links and Mentions In Each Set')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    plt.show()

     # Check for non-english words
    non_english_train = set()
    non_english_val = set()
    non_english_test = set()
    for name, tweet_set in zip(['train_set', 'val_set', 'test_set'], [train_set, val_set, test_set]):
        for text in tweet_set:
            for word in text.split():
                if (word.lower() not in english_words):
                    if (name == 'train_set'):
                        non_english_train.add(word)
                    elif (name == 'val_set'):
                        non_english_val.add(word)
                    else:
                        non_english_test.add(word)

    if(not non_english_train):
        print('No non-english words in train set.')
    if(not non_english_val):
        print('No non-english words in val set.')
    if(not non_english_test):
        print('No non-english words in test set.')

    # Check if val and test set have words that do not appear in train set
    for name, tweet_set in zip(['train_set', 'val_set', 'test_set'], [train_set, val_set, test_set]):
        vocab = set()
        for text in tweet_set:
            vocab.update(text.split())
        if (name == 'train_set'):
            train_vocab = vocab
        elif (name == 'val_set'):
            val_vocab = vocab
        else:
            test_vocab = vocab
    
    val_oov = val_vocab - train_vocab
    test_oov = test_vocab - train_vocab
    
    print(f"Val out of vocab words: {len(val_oov)}")
    print(f"Test out of vocab words: {len(test_oov)}")


# Define stopwords
# stop_words = set(stopwords.words("english"))
# custom_stopwords = ["im", "ur", "u"]
# stop_words.update(custom_stopwords)

#lemmatizer = WordNetLemmatizer()

#spell = SpellChecker()

load()

def preprocess(dataset):
    text = dataset['Text']
    #lowercase text
    text = text.apply(lambda x: x.lower() if isinstance(x, str) else x)
    #remove html entities (e.g. &quot, &amp)
    text = text.apply(lambda x: re.sub(r'\s*&\w+;\s*', ' ', str(x)))
    #remove mentions and emails
    #text = text.apply(lambda x: re.sub(r'@[\w]+', ' ', str(x)))
    #turn mentions to user
    text = text.apply(lambda x: re.sub(r'@\w+', 'user', str(x)))
    #turn emails to email
    text = text.apply(lambda x: re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', 'email', x))
    #turn phone numbers to phone
    text = text.apply(lambda x: re.sub(r'\b(?:\+?\d{1,3})?[-.\s]?(?:\(?\d{2,4}\)?[-.\s]?){1,3}\d{2,4}\b', 'phone', x))
    #remove emoticons
    text = text.apply(lambda x: re.sub(r"[:;=xX8]-?[)DdpP/(|oO*']", ' ', str(x)))
    #remove links
    text = text.apply(lambda x: re.sub(r'https?://\S+', ' ', str(x)))
    #remove non ascii characters
    text = text.apply(lambda x: re.sub(r"[^\x00-\x7F]+", ' ', str(x)))
    #split hashtags into words (e.g #goodtimes -> good times)
    #text = text.apply(lambda x: ' '.join([' '.join(segment(word[1:])) if word.startswith('#') else word for word in x.split()]))
    #remove punctuation
    text = text.apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
    #tokenize text
    text = text.apply(lambda x: word_tokenize(x))

    ###### UNUSED STOP WORDS ###### 
    
    #remove stopwords
    #tokens = text.apply(lambda x: [word for word in x if word not in stop_words])

    ###### UNUSED LEMMATIZATION ###### 

    #text = text.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    
    ###### UNUSED SPELLCHECKING ######

    #text = text.apply(lambda x: [spell.correction(word) if word not in spell else word for word in x])
    
    return text


###### LOAD WORD2VEC MODEL ###### 

word2vec_model = KeyedVectors.load_word2vec_format('/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)

embedding_dim = word2vec_model.vector_size
print(embedding_dim)



def vectorize_tweet(tokens, word2vec, embedding_dim):
    vectors = [word2vec[word] for word in tokens if word in word2vec]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)


class TweetDataset(Dataset):
    def __init__(self, texts, labels, word2vec, embedding_dim):
        self.vectors = [vectorize_tweet(tokens, word2vec, embedding_dim) for tokens in texts]
        self.labels = labels

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        if self.labels is None:
            # For test set, return only the vector (no label)
            return torch.tensor(self.vectors[idx], dtype=torch.float32)
        else:
            # For train/val set, return both the vector and label
            return torch.tensor(self.vectors[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


###### FIRST MODEL ###### 

# class FFNN(nn.Module):
#     def __init__(self, D_in, H1, H2, H3, D_out):
#         super(FFNN, self).__init__()
#         self.linear1 = nn.Linear(D_in, H1)
#         self.linear2 = nn.Linear(H1, H2)
#         self.linear3 = nn.Linear(H2, H3)
#         self.linear4 = nn.Linear(H3, D_out)

#     def forward(self, x):
#         h1 = F.relu(self.linear1(x))
#         h2 = F.relu(self.linear2(h1))
#         h3 = F.relu(self.linear3(h2))
#         out = self.linear4(h3)
#         return out

###### MODEL DURING OPTUNA TRIAL AND AFTER ###### 

class FFNN(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out, activation_function='relu', dropout_rate=0.3):
        super(FFNN, self).__init__()
        self.activation_function = activation_function
        self.dropout = nn.Dropout(p=dropout_rate) #added after optuna trials for further experimentation
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x):
        # Choose activation function based on trial parameter
        if self.activation_function == 'relu':
            activation = F.relu
        elif self.activation_function == 'leaky_relu':
            activation = F.leaky_relu
        elif self.activation_function == 'elu':
            activation = F.elu
        
        h1 = self.dropout(activation(self.linear1(x)))
        h2 = self.dropout(activation(self.linear2(h1)))
        h3 = self.dropout(activation(self.linear3(h2)))
        out = self.linear4(h3)
        return out


def evaluate_accuracy_and_loss(model, data_loader, loss_func):
    model.eval()
    all_preds = []
    all_labels = []
    batch_losses = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch)
            batch_losses.append(loss.item())

            preds = torch.argmax(y_pred, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Calculate accuracy using sklearn's accuracy_score
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = sum(batch_losses) / len(data_loader)

    return accuracy, avg_loss


def train_model(model, train_loader, val_loader, loss_func, optimizer, epochs):
    train_losses = []
    val_accuracies = []
    train_accuracies = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        all_train_preds = []
        all_train_labels = []

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            preds = torch.argmax(y_pred, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(y_batch.cpu().numpy())

        train_loss = sum(batch_losses) / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)

        val_accuracy, val_loss = evaluate_accuracy_and_loss(model, val_loader, loss_func)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies


def evaluate_model(model, data_loader, test):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    if(test):
        for x_batch in data_loader:
            y_pred = model(x_batch) 
            preds = torch.argmax(y_pred, dim=1)

            all_preds.extend(preds.cpu().numpy())
        return all_labels, all_preds, all_scores 

    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            y_pred = model(x_batch)  
            scores = torch.softmax(y_pred, dim=1)[:, 1] 
            preds = torch.argmax(y_pred, dim=1) 

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            
    print('Accuracy:')
    accuracy = accuracy_score(all_labels, all_preds)
    print(accuracy)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print('Confusion Matrix')
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
    
    return all_labels, all_preds, all_scores


def plot_roc_curve(all_labels, all_scores):
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    auc = roc_auc_score(all_labels, all_scores)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.title("ROC Curve")
    plt.show()


def plot_learning_curve(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))

    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', color='orange')
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='green')
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


###### PREPROCESSING, TRAINING AND EVALUATION OF MODELS ###### 

set_seed(94)

train_set = pd.read_csv('/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/train_dataset.csv')
val_set = pd.read_csv('/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/val_dataset.csv')
test_set = pd.read_csv('/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/test_dataset.csv')

#eda(train_set,val_set,test_set)

train_text = preprocess(train_set)
val_text = preprocess(val_set)
test_text = preprocess(test_set)

train_dataset = TweetDataset(train_text, train_set['Label'], word2vec_model, embedding_dim)
val_dataset = TweetDataset(val_text, val_set['Label'], word2vec_model, embedding_dim)
test_dataset = TweetDataset(test_text, None, word2vec_model, embedding_dim)

g = torch.Generator()
g.manual_seed(94)

###### FIRST MODEL ###### 
# D_in = embedding_dim 
# H1, H2, H3 = 128, 64, 32
# D_out = 2 
# learning_rate = 1e-3
# batch = 64
# epochs = 5
# activation_func = 'relu'

###### BEST MODEL BY FIRST OPTUNA RUN ###### 
# D_in = embedding_dim 
# H1, H2, H3 = 411, 84, 307
# D_out = 2 
# learning_rate = 1.9195860955591215e-05
# batch = 32
# epochs = 20
# activation_func = 'relu'

###### FIRST OPTUNA RUN TRIAL 19 MODEL WITH LESS EPOCHS TO GET MAX VALIDATION ACCURACY BASED ON PREVIOUS PERFORMANCE ###### 
# D_in = embedding_dim 
# H1, H2, H3 = 226, 443, 387
# D_out = 2 
# learning_rate = 0.00038396274693142666
# batch = 16
# epochs = 7
# activation_func = 'relu'

###### FIRST OPTUNA  RUN TRIAL 22 MODEL WITH LESS EPOCHS TO GET MAX VALIDATION ACCURACY BASED ON PREVIOUS PERFORMANCE ###### 
# D_in = embedding_dim 
# H1, H2, H3 = 422, 95, 302
# D_out = 2 
# learning_rate = 2.4083315989971293e-05
# batch = 32
# epochs = 20
# activation_func = 'leaky_relu'

###### FIRST OPTUNA RUN TRIAL 49 MODEL WITH LESS EPOCHS TO GET MAX VALIDATION ACCURACY BASED ON PREVIOUS PERFORMANCE ###### 
# D_in = embedding_dim 
# H1, H2, H3 = 205, 54, 347
# D_out = 2 
# learning_rate = 7.770825409730508e-05
# batch = 32
# epochs = 11
# activation_func = 'relu'

###### FIRST OPTUNA RUN TRIAL 26 MODEL ATTEMPT TO GET BETTER ACCURACY BY ELIMINATING OVERFIT ###### 
# ORIGINAL TRIAL (for showcase purposes)
# D_in = embedding_dim 
# H1, H2, H3 = 413, 146, 127
# D_out = 2 
# learning_rate = 0.00010672102799450752
# batch = 32
# epochs = 26
# activation_func = 'leaky_relu'
# no dropout

# ATTEMPT 1
# D_in = embedding_dim 
# H1, H2, H3 = 413, 146, 127
# D_out = 2 
# learning_rate = 0.00010672102799450752
# batch = 32
# epochs = 26
# activation_func = 'leaky_relu'
# dropout = 0.3

#ATTEMPT 2     ############# KEEP ##############
# D_in = embedding_dim 
# H1, H2, H3 = 413, 146, 127
# D_out = 2 
# learning_rate = 0.00010672102799450752
# batch = 32
# epochs = 13
# activation_func = 'leaky_relu'
# dropout = 0.4

#ATTEMPT 3
# D_in = embedding_dim 
# H1, H2, H3 = 413, 146, 127
# D_out = 2 
# learning_rate = 0.00010672102799450752
# batch = 32
# epochs = 14
# activation_func = 'leaky_relu'
# dropout = 0.5

#ATTEMPT 4 (best model with splitting hashtags to words)
# D_in = embedding_dim 
# H1, H2, H3 = 413, 146, 127
# D_out = 2 
# learning_rate = 0.00010672102799450752
# batch = 32
# epochs = 13
# activation_func = 'leaky_relu'
# dropout = 0.3

#ATTEMPT 5
# D_in = embedding_dim 
# H1, H2, H3 = 413, 146, 127
# D_out = 2 
# learning_rate = 0.00010672102799450752
# batch = 32
# epochs = 15
# activation_func = 'leaky_relu'
# dropout = 0.4

#ATTEMPT 6
# D_in = embedding_dim 
# H1, H2, H3 = 413, 146, 127
# D_out = 2 
# learning_rate = 0.00010672102799450752
# batch = 32
# epochs = 14
# activation_func = 'leaky_relu'
# dropout = 0.4


###### FIRST OPTUNA RUN TRIAL 22 MODEL WITH DROPOUT ###### 
# D_in = embedding_dim 
# H1, H2, H3 = 422, 95, 302
# D_out = 2 
# learning_rate = 2.4083315989971293e-05
# batch = 32
# epochs = 27
# activation_func = 'leaky_relu'
# dropout = 0.3

###### BEST MODEL BY SECOND OPTUNA RUN ###### 
D_in = embedding_dim 
H1, H2, H3 = 487, 381, 345
D_out = 2 
learning_rate = 9.506368766168096e-05
batch = 128
epochs = 25
activation_func = 'leaky_relu'
dropout = 0.3689198960097353
weight_decay =  0.0004652508556074659


train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=batch)
test_loader = DataLoader(test_dataset, batch_size=batch)

model = FFNN(D_in, H1, H2, H3, D_out, activation_func, dropout)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, train_loader, val_loader, loss_func, optimizer, epochs)
print('Evaluation on train set')
train_labels, train_preds, train_scores = evaluate_model(model, train_loader, False)
print('Evaluation on val set')
val_labels, val_preds, val_scores = evaluate_model(model, val_loader, False)
plot_roc_curve(val_labels, val_scores)
plot_learning_curve(train_losses, train_accuracies, val_losses, val_accuracies)

_, test_preds, _ = evaluate_model(model, test_loader, True)
submission_df = pd.DataFrame({
    'ID': test_set['ID'],
    'Label': test_preds
})

submission_df.to_csv('submission.csv', index=False)


###### OPTUNA TRIALS ######

# **NOTE** I am not sure reproducibility is ensured in these functions. To see the runs I mention in my report please refer to versions 1 and 11 of this notebook

###### TRIAL 1 ######

# def objective(trial):
#     set_seed(94)
#     # Hyperparameters to optimize
#     H1 = trial.suggest_int('H1', 32, 512)  # First hidden layer size
#     H2 = trial.suggest_int('H2', 32, 512)  # Second hidden layer size
#     H3 = trial.suggest_int('H3', 32, 512)  # Third hidden layer size
#     epochs = trial.suggest_int('epochs', 5, 50)  # Number of epochs
#     batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])  # Batch size
#     lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)  # Learning rate
#     optimizer_choice = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
#     activation_fn = trial.suggest_categorical('activation_fn', ['relu', 'leaky_relu', 'elu'])
#     loss_func = nn.CrossEntropyLoss()


#     print(f"Trial {trial.number}:")
#     print(f"  H1: {H1}, H2: {H2}, H3: {H3}, epochs: {epochs}, batch_size: {batch_size}")
#     print(f"  Learning rate: {lr}, Optimizer: {optimizer_choice}, Activation function: {activation_fn}")
    
#     # Define the model (pass the chosen activation function)
#     model = FFNN(D_in=300, H1=H1, H2=H2, H3=H3, D_out=2, activation_function=activation_fn)

#     # Set up optimizer
#     if optimizer_choice == 'Adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     elif optimizer_choice == 'RMSprop':
#         optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    
#     g = torch.Generator()
#     g.manual_seed(94)

#     # Prepare DataLoader for current batch size
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # Train the model
#     train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, train_loader, val_loader, loss_func, optimizer, epochs=epochs)
#     print('Evaluation on train set')
#     train_labels, train_preds, train_scores = evaluate_model(model, train_loader, False)
#     print('Evaluation on val set')
#     val_labels, val_preds, val_scores = evaluate_model(model, val_loader, False)
#     # Return the final validation accuracy (or loss)
#     return val_accuracies[-1]  # Maximizing validation accuracy

###### TRIAL 2 ######

# def objective(trial):
#     set_seed(94)

#     # Hyperparameters to optimize
#     H1 = trial.suggest_int('H1', 256, 512)
#     H2 = trial.suggest_int('H2', 64, H1)
#     H3 = trial.suggest_int('H3', 32, H2)
#     dropout = trial.suggest_float('dropout', 0.2, 0.5)
#     epochs = trial.suggest_int('epochs', 10, 30)
#     batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
#     lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
#     weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-3)
#     optimizer_choice = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
#     activation_fn = trial.suggest_categorical('activation_fn', ['relu', 'leaky_relu', 'elu'])

#     print(f"Trial {trial.number}: H1={H1}, H2={H2}, H3={H3}, dropout={dropout}, epochs={epochs}, batch_size={batch_size}, lr={lr}, wd={weight_decay}, opt={optimizer_choice}, act={activation_fn}")

#     model = FFNN(D_in=embedding_dim, H1=H1, H2=H2, H3=H3, D_out=2,
#                  activation_function=activation_fn, dropout_rate=dropout)

#     if optimizer_choice == 'Adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     else:
#         optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

#     g = torch.Generator()
#     g.manual_seed(94)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     loss_func = nn.CrossEntropyLoss()

#     train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, train_loader, val_loader, loss_func, optimizer, epochs=epochs)
#     print('Evaluation on train set')
#     train_labels, train_preds, train_scores = evaluate_model(model, train_loader, False)
#     print('Evaluation on val set')
#     val_labels, val_preds, val_scores = evaluate_model(model, val_loader, False)

#     return val_accuracies[-1]


# # Create and run the study
# study = optuna.create_study(direction='maximize', study_name='Best Model - Validation Accuracy Maximization')
# study.optimize(objective, n_trials=50)

# # Print the best parameters and value
# print(f"Best parameters: {study.best_params}")
# print(f"Best val accuracy: {study.best_value}")

