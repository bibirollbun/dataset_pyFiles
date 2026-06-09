# !pip install palettable


import re
import os
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn


import nltk
from nltk.corpus import stopwords

from tqdm.notebook import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

import transformers
from transformers import BertPreTrainedModel
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split
from transformers import BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

import plotly.io as pio
pio.renderers.default = 'notebook'

import warnings
warnings.filterwarnings("ignore")

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


class CFG:
    epochs = 1
    train_batch_size = 32
    val_batch_size = 16
    max_len = 128
    model_name = 'bert-base-uncased'
    BERT_PATH = '../input/bert-base-uncased/'
    CONFIG_PATH = '../input/bert-base-uncased/config.json'
    MAX_GRAD_NORM = 0.1
    lr = 3e-5

    root_dir = '../input/tweet-sentiment-extraction'
    sample = os.path.join(root_dir, 'sample_submission.csv')
    train_dir = os.path.join(root_dir, 'train.csv')
    test_dir = os.path.join(root_dir, 'test.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(CFG.device)


train_df = pd.read_csv(CFG.train_dir)
train_df.dropna(inplace = True)
print(train_df.info())
train_df.head()


temp = train_df.groupby('sentiment').count()['text'].reset_index().sort_values(by='text', ascending=False)
temp.style.background_gradient(cmap='Greens')


plt.figure(figsize = (10, 7))
sns.countplot(x = 'sentiment', data = train_df)


# fig = go.Figure(go.Funnelarea(
#     text = temp['sentiment'],
#     values = temp.text
# ))
# fig.show()


def jaccard(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0  
    elif len(a) == 0 or len(b) == 0:
        return 0.0  
    else:
        return float(len(c)) / (len(a) + len(b) - len(c))


def Data_Processing(df):
    df['text'] = df['text'].astype(str).str.lower()
    df['selected_text'] = df['selected_text'].astype(str).str.lower()
    jaccard_scores = df.apply(lambda x: jaccard(x['text'], x['selected_text']), axis=1).tolist()
    df['jaccard_scores'] = jaccard_scores
    df['Num_Words_ST'] = df['selected_text'].apply((lambda x: len(str(x).split())))
    df['Num_Words_T'] = df['text'].apply((lambda x: len(str(x).split())))
    df['Num_Words_diff'] = df['Num_Words_T'] - df['Num_Words_ST']
    return df


train_df = Data_Processing(train_df)
train_df.head()


plt.figure(figsize = (12, 6))
p1 = sns.kdeplot(train_df.Num_Words_ST, shade = True, color = 'r').set_title('Distribution of Number Of words')
p1 = sns.kdeplot(train_df.Num_Words_T, shade = True, color = 'b')


plt.figure(figsize = (12, 6))
p1 = sns.kdeplot(train_df[train_df.sentiment == 'positive']['Num_Words_diff'], shade = True, color = 'r')
p1 = sns.kdeplot(train_df[train_df.sentiment == 'negative']['Num_Words_diff'], shade = True, color = 'b')


plt.figure(figsize = (12, 6))
sns.displot(train_df[train_df.sentiment == 'neutral']['Num_Words_diff'], kde =False)


plt.figure(figsize = (12, 6))
p1 = sns.kdeplot(train_df[train_df.sentiment == 'positive']['jaccard_scores'], shade = True, color = 'r')
p1 = sns.kdeplot(train_df[train_df.sentiment == 'negative']['jaccard_scores'], shade = True, color = 'b')


plt.figure(figsize = (12, 6))
sns.displot(train_df[train_df.sentiment == 'neutral']['jaccard_scores'], kde =False)


def clean_text(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    custom_punctuation = string.punctuation.replace('*', '')
    text = re.sub('[%s]' % re.escape(custom_punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


stop_words = set(stopwords.words('english'))
custom_stopwords = ['u','a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"]
normalized_custom_stopwords = [word.replace("'", "") for word in custom_stopwords]
stop_words.update(normalized_custom_stopwords)
def remove_stopword(text):
    return [word for word in text if word not in stop_words]


def add_data_processing(df):
    df['text'] = df['text'].apply(lambda x: clean_text(x))
    df['selected_text'] = df['selected_text'].apply(lambda x: clean_text(x))
    df['temp_list'] = df['selected_text'].apply(lambda x: str(x).split())
    df['temp_list'] = df['temp_list'].apply(lambda x: remove_stopword(x))
    return df


train_df = add_data_processing(train_df)
train_df.head()


all_words = [word for sublist in train_df['temp_list'] for word in sublist]
word_counts = Counter(all_words)
top_20_words = word_counts.most_common(20)
words, counts = zip(*top_20_words)

plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 20 Most Common Words in temp_list')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


Positive_sent = train_df[train_df['sentiment']=='positive']
Negative_sent = train_df[train_df['sentiment']=='negative']
Neutral_sent = train_df[train_df['sentiment']=='neutral']


top = Counter([item for sublist in Positive_sent['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ['Common_words','count']
temp_positive.style.background_gradient(cmap='Greens')


top = Counter([item for sublist in Negative_sent['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ['Common_words','count']
temp_positive.style.background_gradient(cmap='Reds')


top = Counter([item for sublist in Neutral_sent['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ['Common_words','count']
temp_positive.style.background_gradient(cmap='Greys')


# def words_unique(sentiment,numwords,raw_words, train):
#     '''
#     Input:
#         segment - Segment category (ex. 'Neutral');
#         numwords - how many specific words do you want to see in the final result; 
#         raw_words - list  for item in train_data[train_data.segments == segments]['temp_list1']:
#     Output: 
#         dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..

#     '''
#     allother = []
#     for item in train[train.sentiment != sentiment]['temp_list']:
#         for word in item:
#             allother .append(word)
#     allother  = list(set(allother ))
    
#     specificnonly = [x for x in raw_text if x not in allother]
    
#     mycounter = Counter()
    
#     for item in train[train.sentiment == sentiment]['temp_list']:
#         for word in item:
#             mycounter[word] += 1
#     keep = list(specificnonly)
    
#     for word in list(mycounter):
#         if word not in keep:
#             del mycounter[word]
    
#     Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns = ['words','count'])
    
#     return Unique_words


# from palettable.colorbrewer.qualitative import Pastel1_7

# raw_text = [word for word_list in train_df['temp_list'] for word in word_list]

# Unique_Positive= words_unique('positive', 20, raw_text, train_df)
# plt.figure(figsize=(16,10))
# my_circle=plt.Circle((0,0), 0.7, color='white')
# plt.pie(Unique_Positive['count'], labels=Unique_Positive.words, colors=Pastel1_7.hex_colors)
# p=plt.gcf()
# p.gca().add_artist(my_circle)
# plt.title('DoNut Plot Of Unique Positive Words')
# plt.show()


# Unique_Negative= words_unique('negative', 10, raw_text, train_df)
# print("The top 10 unique words in Negative Tweets are:")
# Unique_Negative.style.background_gradient(cmap='Reds')


# plt.figure(figsize=(16,10))
# my_circle=plt.Circle((0,0), 0.7, color='white')
# plt.rcParams['text.color'] = 'black'
# plt.pie(Unique_Negative['count'], labels=Unique_Negative.words, colors=Pastel1_7.hex_colors)
# p=plt.gcf()
# p.gca().add_artist(my_circle)
# plt.title('DoNut Plot Of Unique Negative Words')
# plt.show()


# Unique_Neutral= words_unique('neutral', 10, raw_text, train_df)
# print("The top 10 unique words in Neutral Tweets are:")
# Unique_Neutral.style.background_gradient(cmap='Greys')


# plt.figure(figsize=(16,10))
# my_circle=plt.Circle((0,0), 0.7, color='white')
# plt.pie(Unique_Neutral['count'], labels=Unique_Neutral.words, colors=Pastel1_7.hex_colors)
# p=plt.gcf()
# p.gca().add_artist(my_circle)
# plt.title('DoNut Plot Of Unique Neutral Words')
# plt.show()


def process_data(text, selected_text, sentiment, tokenizer, max_len):
    idx0 = text.find(selected_text)
    if idx0 != -1:
        idx1 = idx0 + len(selected_text) - 1
    else:
        idx0, idx1 = None, None

    mask = [0] * len(text)
    if idx0 is not None and idx1 is not None:
        mask[idx0: idx1 + 1] = [1] * (idx1 - idx0 + 1)

    tok_text = tokenizer.encode(text)
    input_ids_origin = tok_text.ids[1:-1]
    text_offsets = tok_text.offsets[1:-1]
    
    target_idx = []
    for i, (offset1, offset2) in enumerate(text_offsets):
        if sum(mask[offset1:offset2]) > 0:
            target_idx.append(i)

    targets_start = target_idx[0] + 3 if target_idx else 0
    targets_end = target_idx[-1] + 3 if target_idx else 0

    sentiments_id = {
        'positive': 3893, 
        'negative': 4997,
        'neutral': 8699
    }

    input_ids = [101, sentiments_id[sentiment], 102] + input_ids_origin + [102]
    token_type_ids = [0, 0, 0] + [1] * (len(input_ids_origin) + 1)
    mask = [1] * len(input_ids)

    text_offsets = [(0, 0)] * 3 + text_offsets + [(0, 0)]
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [0] * pad_len
        mask += [0] * pad_len
        token_type_ids += [0] * pad_len
        text_offsets += [(0, 0)] * pad_len

    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'target_start': targets_start,
        'target_end': targets_end,
        'text_offsets': text_offsets
    }


class TweetDataset(Dataset):
    def __init__(self, text, selected_text, sentiment, tokenizer, max_len):
        self.text = text  
        self.selected_text = selected_text  
        self.sentiment = sentiment  
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):  
        data = process_data(
            self.text[idx],
            self.selected_text[idx],
            self.sentiment[idx],
            self.tokenizer,
            self.max_len
        )

        return {
            'ids': torch.tensor(data['ids'], dtype=torch.long).to(CFG.device),
            'mask': torch.tensor(data['mask'], dtype=torch.long).to(CFG.device),
            'token_type_ids': torch.tensor(data['token_type_ids'], dtype=torch.long).to(CFG.device),
            'target_start': torch.tensor(data['target_start'], dtype=torch.long).to(CFG.device),
            'target_end': torch.tensor(data['target_end'], dtype=torch.long).to(CFG.device),
            'text_offsets': torch.tensor(data['text_offsets'], dtype=torch.long).to(CFG.device),
            'origin_tweet': self.text[idx],  
            'origin_selected_text': self.selected_text[idx], 
            'sentiment': self.sentiment[idx]  
        }


class TweetModel(BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.bert = transformers.BertModel.from_pretrained(CFG.BERT_PATH, config=conf)
        
        # Increase dropout rate for better regularization
        self.drop_out = nn.Dropout(0.5)  # Increased from 0.3
        
        # Add layer normalization for better stability
        self.layer_norm = nn.LayerNorm(768 * 2)
        
        # Add weight decay through initialization
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.xavier_normal_(self.l0.weight)
        self.l0.bias.data.fill_(0.0)

    def forward(self, ids, token_type_ids, mask):
        outputs = self.bert(
            ids,
            token_type_ids=token_type_ids,
            attention_mask=mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states
        
        # Feature extraction with weighted combination of last 4 layers
        last_four_layers = torch.stack(hidden_states[-4:], dim=0)
        weights = torch.softmax(torch.randn(4, 1, 1, 1, device=ids.device), dim=0)
        weighted_avg = torch.sum(weights * last_four_layers, dim=0)
        
        # Original concatenation of last two layers
        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=2)
        
        # Apply dropout and layer normalization
        out = self.drop_out(out)
        out = self.layer_norm(out)
        
        # Final prediction
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits


def loss_fn(start_logits, end_logits, start_pos, end_pos):
    loss = nn.CrossEntropyLoss()
    start_loss = loss(start_logits, start_pos)
    end_loss = loss(end_logits, end_pos)
    return (end_loss + start_loss)


def trim_predictions(text, start_idx, end_idx, offsets):
    """Convert token indices to character indices and extract prediction"""
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
        
    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]
    
    if char_start < 0:
        char_start = 0
    if char_end > len(text):
        char_end = len(text)
        
    return text[char_start:char_end]


def train_model(model, train_data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    
    progress_bar = tqdm(train_data_loader, desc="Training")
    
    for data in progress_bar:
        ids = data['ids']  
        token_type_ids = data['token_type_ids']
        mask = data['mask']
        target_start = data['target_start']
        target_end = data['target_end']
        
        # print('Start Training')
        
        optimizer.zero_grad()
        start_logits, end_logits = model(ids, token_type_ids, mask)
        
        loss = loss_fn(start_logits, end_logits, target_start, target_end)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        progress_bar.set_postfix({'loss': np.mean(losses[-10:])})
    
    return np.mean(losses)


def evaluate_model(model, valid_data_loader, device):
    """Evaluate the model and calculate Jaccard score"""
    model.eval()
    valid_losses = []
    jaccard_scores = []
    
    progress_bar = tqdm(valid_data_loader, desc="Evaluating")
    
    with torch.no_grad():
        for data in progress_bar:
            ids = data['ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            mask = data['mask'].to(device)
            target_start = data['target_start'].to(device)
            target_end = data['target_end'].to(device)
            
            origin_tweet = data['origin_tweet']
            origin_selected_text = data['origin_selected_text']
            offsets = data['text_offsets']
            
            start_logits, end_logits = model(ids, token_type_ids, mask)

            loss = loss_fn(start_logits, end_logits, target_start, target_end)
            valid_losses.append(loss.item())
            
            start_preds = torch.argmax(start_logits, dim=1).cpu().detach().numpy()
            end_preds = torch.argmax(end_logits, dim=1).cpu().detach().numpy()
            
            for i, (tweet, true_selected) in enumerate(zip(origin_tweet, origin_selected_text)):
                pred_selected = trim_predictions(
                    tweet, 
                    start_preds[i], 
                    end_preds[i], 
                    offsets[i].cpu().numpy()
                )
                jaccard_score = jaccard(true_selected, pred_selected)
                jaccard_scores.append(jaccard_score)
            
            progress_bar.set_postfix({
                'loss': np.mean(valid_losses[-10:]), 
                'jaccard': np.mean(jaccard_scores[-10:])
            })
    
    return np.mean(valid_losses), np.mean(jaccard_scores)


def run():
    tokenizer = BertWordPieceTokenizer(
        '/kaggle/input/bert-base-uncased/vocab.txt',
        lowercase=True
    )
    
    X_train, X_val, y_train, y_val, train_sentiment, val_sentiment = train_test_split(
        train_df['text'], 
        train_df['selected_text'],
        train_df['sentiment'],
        test_size=0.2
    )

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    train_sentiment = train_sentiment.reset_index(drop=True)
    val_sentiment = val_sentiment.reset_index(drop=True)

    print('Init Dataset')

    train_dataset = TweetDataset(
        X_train,
        y_train,
        train_sentiment,
        tokenizer,
        CFG.max_len
    )
    
    val_dataset = TweetDataset(
        X_val,
        y_val,
        val_sentiment,
        tokenizer,
        CFG.max_len
    )
    print('Init DataLoader')

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(  
        val_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=0
    )

    

    config = BertConfig.from_pretrained(CFG.CONFIG_PATH) 
    model = TweetModel(config)
    model.to(CFG.device)
    
    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    
    total_steps = len(train_loader) * CFG.epochs 
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_jaccard = 0
    for epoch in range(CFG.epochs):  
        print(f"\nEpoch {epoch + 1}/{CFG.epochs}")
        
        train_loss = train_model(model, train_loader, optimizer, scheduler, CFG.device)
        
        valid_loss, jaccard_score = evaluate_model(model, valid_loader, CFG.device)  
        
        print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Jaccard Score: {jaccard_score:.4f}")
        
        if jaccard_score > best_jaccard:
            best_jaccard = jaccard_score
            torch.save(model.state_dict(), "best_model.bin")
            print(f"Model saved with Jaccard Score: {best_jaccard:.4f}")
    
    return model

model = run()


def inference(model, tokenizer, test_df, device):
    """
    Run inference on test data and return predictions
    """
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    # Create a DataLoader for test data
    test_dataset = TweetDataset(
        test_df['text'].values,
        test_df['text'].values,  # Using text as placeholder for selected_text
        test_df['sentiment'].values,
        tokenizer,
        CFG.max_len
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Run inference
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Predicting"):
            ids = data['ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            mask = data['mask'].to(device)
            origin_tweet = data['origin_tweet']
            offsets = data['text_offsets']
            
            # Forward pass
            start_logits, end_logits = model(ids, token_type_ids, mask)
            
            # Get predictions
            start_preds = torch.argmax(start_logits, dim=1).cpu().detach().numpy()
            end_preds = torch.argmax(end_logits, dim=1).cpu().detach().numpy()
            
            # Process each prediction
            for i, tweet in enumerate(origin_tweet):
                # Get prediction
                pred_selected_text = trim_predictions(
                    tweet,
                    start_preds[i],
                    end_preds[i],
                    offsets[i].cpu().numpy()
                )
                
                # Handle special case for neutral sentiment
                sentiment = data['sentiment'][i]
                if sentiment == 'neutral' and len(pred_selected_text.strip()) == 0:
                    pred_selected_text = tweet
                
                predictions.append(pred_selected_text)
    
    return predictions

def trim_predictions(text, start_idx, end_idx, offsets):
    """
    Convert token indices to character indices and extract prediction
    """
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
        
    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]
    
    if char_start < 0:
        char_start = 0
    if char_end > len(text):
        char_end = len(text)
        
    trimmed_text = text[char_start:char_end]
    
    # Post-processing to improve predictions
    trimmed_text = trimmed_text.strip()
    
    # If prediction is empty, return the whole text
    if len(trimmed_text) == 0:
        return text
    
    return trimmed_text

def create_submission(model):
    """
    Create a submission file for the Kaggle competition
    """
    # Load test data
    test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
    
    tokenizer = BertWordPieceTokenizer(
        '/kaggle/input/bert-base-uncased/vocab.txt',
        lowercase=True
    )
    
    predictions = inference(model, tokenizer, test_df, CFG.device)
    
    submission_df = pd.DataFrame({
        'textID': test_df['textID'],
        'selected_text': predictions
    })
    
    for i, row in submission_df.iterrows():
        if pd.isna(row['selected_text']) or len(row['selected_text'].strip()) == 0:
            submission_df.at[i, 'selected_text'] = test_df.loc[i, 'text']
    
    submission_df.to_csv('submission.csv', index=False)
    print(f"Submission file created with {len(submission_df)} predictions")
    
    return submission_df

if __name__ == "__main__":
    submission = create_submission(model)
    print("Submission created successfully!")

