# ğŸ§± Built-in
import re
import string
import unicodedata
import numpy as np

# ğŸ“Š Data processing
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# ğŸ“š NLP
import nltk
from nltk.corpus import stopwords

# ğŸ”¥ PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.optim import AdamW  # ğŸ‘ˆ ×�×� ×–×” ×œ×� ×¢×•×‘×“, ×©× ×” ×œ: from transformers.optimization import AdamW

# ğŸ¤— Transformers
from transformers import (
    BertTokenizerFast,
    BertModel,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup
)

# ğŸ“ˆ Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from wordcloud import WordCloud

# ×”×•×¨×“×ª stopwords (×¤×¢×� ×�×—×ª ×‘×œ×‘×“)
nltk.download('stopwords')



# Loading the train & test data
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")


# Our data size
print(train_df.shape)
print(test_df.shape)


train_df.head()


test_df.head()


#Missing values in training set
print("Train's nulls values\n",train_df.isnull().sum())

#Missing values in test set
print("\nTest's nulls values\n",test_df.isnull().sum())


# Dropping rows with null values
train_df = train_df.dropna().reset_index(drop=True)


# checking the sentiment balnace
train_df['sentiment'].value_counts()


train_df['sentiment'].value_counts(normalize=True)


# ×—×™×©×•×‘ ×¢×¨×›×™×� ×�× ×•×¨×�×œ×™×�
sentiment_counts = train_df['sentiment'].value_counts(normalize=True)

# ×”×›×¤×œ×” ×‘-100 ×›×“×™ ×œ×”×¦×™×’ ×‘×�×—×•×–×™×�
sentiment_percentages = sentiment_counts * 100

# ×¦×™×•×¨ ×”×’×¨×£
plt.figure(figsize=(8, 5))
colors = ['#F4D03F', '#58D68D', '#EC7063']

sns.barplot(
    x=sentiment_percentages.index,
    y=sentiment_percentages.values,
    palette=colors
)

# ×›×•×ª×¨×•×ª ×•×¦×™×¨×™×�
plt.title('Normalized Sentiment Distribution (%)', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Percentage of Tweets', fontsize=12)

# ×ª×•×•×™×•×ª ×¢×� ×�×—×•×–×™×�
for i, value in enumerate(sentiment_percentages.values):
    plt.text(i, value + 0.5, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

plt.ylim(0, max(sentiment_percentages.values) + 5)
plt.tight_layout()
plt.show()


# checking the sentiment balnace
test_df['sentiment'].value_counts()


test_df['sentiment'].value_counts(normalize=True)


# ×—×™×©×•×‘ ×¢×¨×›×™×� ×�× ×•×¨×�×œ×™×�
sentiment_counts = test_df['sentiment'].value_counts(normalize=True)

# ×”×›×¤×œ×” ×‘-100 ×›×“×™ ×œ×”×¦×™×’ ×‘×�×—×•×–×™×�
sentiment_percentages = sentiment_counts * 100

# ×¦×™×•×¨ ×”×’×¨×£
plt.figure(figsize=(8, 5))
colors = ['#F4D03F', '#58D68D', '#EC7063']

sns.barplot(
    x=sentiment_percentages.index,
    y=sentiment_percentages.values,
    palette=colors
)

# ×›×•×ª×¨×•×ª ×•×¦×™×¨×™×�
plt.title('Normalized Sentiment Distribution (%)', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Percentage of Tweets', fontsize=12)

# ×ª×•×•×™×•×ª ×¢×� ×�×—×•×–×™×�
for i, value in enumerate(sentiment_percentages.values):
    plt.text(i, value + 0.5, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

plt.ylim(0, max(sentiment_percentages.values) + 5)
plt.tight_layout()
plt.show()


# cleaning the text from sign, etc.
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # ×”×¡×¨×ª ×œ×™× ×§×™×�
    text = re.sub(r'<.*?>+', '', text)  # ×”×¡×¨×ª ×ª×’×™×•×ª HTML
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # × ×™×§×•×™ ×¡×™×�× ×™ ×¤×™×¡×•×§
    text = re.sub('\n', ' ', text)  # ×”×¡×¨×ª ×©×•×¨×•×ª ×—×“×©×•×ª
    text = re.sub(r'\w*\d\w*', '', text)  # ×�×™×œ×™×� ×¢×� ×�×¡×¤×¨×™×�
    return text

def text_preprocessing(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    combined_text = ' '.join(tokenized_text)
    return combined_text


# cleaning train_df
train_text_cleaned = []
for text in train_df['text']:
    if isinstance(text, str):  
        clean = text_preprocessing(text)
        train_text_cleaned.append(clean)
    else:
        train_text_cleaned.append("")

train_df['clean_text'] = train_text_cleaned


# cleaning test_df
test_text_cleaned = []
for text in test_df['text']:
    if isinstance(text, str):
        clean = text_preprocessing(text)
        test_text_cleaned.append(clean)
    else:
        test_text_cleaned.append("")

test_df['clean_text'] = test_text_cleaned


# ×�×•×¨×š ×‘×�×•× ×—×™ ×ª×•×•×™×�
train_df['text_len_chars'] = train_df['clean_text'].astype(str).apply(len)

# ×�×•×¨×š ×‘×�×•× ×—×™ ×�×™×œ×™×�
train_df['text_len_words'] = train_df['clean_text'].astype(str).apply(lambda x: len(x.split()))



train_df


# ×¦×‘×¢×™×� ×œ×›×œ ×¡×•×’ ×¡× ×˜×™×�× ×˜
colors = {
    'positive': 'red',
    'negative': 'green',
    'neutral': 'orange'
}

# ×™×¦×™×¨×ª ×”×’×¨×£
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=train_df, x='sentiment', y='text_len_chars', palette=colors)

# ×›×•×ª×¨×ª
plt.title('Length of the text', fontsize=16)

# ×©×�×•×ª ×§×¨×™×�×™×� ×™×•×ª×¨ ×œ×¦×™×¨ X
sentiment_names = {
    'positive': 'Positive Text',
    'negative': 'Negative Text',
    'neutral': 'Neutral Text'
}

# ×”×•×¡×¤×ª ×”×¢×¨×š ×”×�×§×¡×™×�×œ×™ ×‘×œ×‘×“ ×�×¢×œ ×›×œ ×ª×™×‘×”
for i, sentiment in enumerate(['positive', 'negative', 'neutral']):
    max_val = train_df[train_df['sentiment'] == sentiment]['text_len_chars'].max()
    ax.text(i, max_val + 5,  # ×ª×•×¡×¤×ª ×§×˜× ×” ×œ×�×™×§×•×� ×�×¢×œ ×”×§×•×¤×¡×”
            f"max: {max_val}",
            ha='center', va='bottom',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', lw=1))

# ×¢×™×¦×•×‘ ×¦×™×¨ X
plt.xticks(ticks=[0, 1, 2],
           labels=[sentiment_names[s] for s in ['positive', 'negative', 'neutral']])

# ×¢×™×¦×•×‘ ×›×œ×œ×™
plt.xlabel('')
plt.ylabel('Text Length (Characters)')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# Jaccard function for measuring the accuracy
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    denominator = len(a) + len(b) - len(c)
    if denominator == 0:
        return 0.0
    return float(len(c)) / denominator



sen_1 = "I absolutely love relaxing on the beach during bright sunny days."
sen_2 = "There's nothing better than spending a sunny day by the sea."
sen_3 = "Relaxing at the beach is the perfect way to enjoy warm, sunny weather."



print(jaccard(sen_1, sen_2))
print(jaccard(sen_1, sen_3))
print(jaccard(sen_2, sen_3))


# Adding jaccard value for each row in train data
results_jaccard=[]

for ind,row in train_df.iterrows():
    sentence1 = row.text
    sentence2 = row.selected_text

    jaccard_score = jaccard(sentence1,sentence2)
    results_jaccard.append([sentence1,sentence2,jaccard_score])


jaccard_df = pd.DataFrame(results_jaccard,columns=["text","selected_text","jaccard_score"])
train_df = train_df.merge(jaccard_df ,how='outer')


train_df


stop_words = set(stopwords.words('english'))

# ×¤×•× ×§×¦×™×™×ª ×¦×‘×™×¢×” ×�×•×ª×�×�×ª ×�×™×©×™×ª ×œ×¤×™ ×¦×‘×¢ ×‘×¡×™×¡
def color_func_factory(base_color):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        if base_color == 'green':
            return f"hsl(120, 100%, {30 + font_size % 50}%)"
        elif base_color == 'red':
            return f"hsl(0, 100%, {30 + font_size % 50}%)"
        elif base_color == 'blue':
            return f"hsl(240, 100%, {30 + font_size % 50}%)"
        else:
            return "black"
    return color_func

# ×¤×•× ×§×¦×™×” ×œ×™×¦×™×¨×ª ×¢× ×Ÿ ×�×™×œ×™×� ×¢×� ×¦×‘×¢ ×�×©×ª× ×”
def generate_wordcloud_by_sentiment(df, sentiment_label, base_color):
    text = ' '.join(df[df['sentiment'] == sentiment_label]['selected_text'].dropna())

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stop_words,
        max_words=200
    ).generate(text)

    # ×¦×‘×¢ ×�×•×ª×�×� ×œ×¨×’×©
    color_func = color_func_factory(base_color)
    wordcloud.recolor(color_func=color_func)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud - {sentiment_label.capitalize()} Sentiment", fontsize=16)
    plt.axis('off')
    plt.show()

# ×™×¦×™×¨×ª ×¢× × ×™×� ×œ×¤×™ ×¨×’×©×•×ª
generate_wordcloud_by_sentiment(train_df, 'positive', 'green')
generate_wordcloud_by_sentiment(train_df, 'negative', 'red')
generate_wordcloud_by_sentiment(train_df, 'neutral',  'blue')



# Bert
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Encoding the text to numeric
def encode_text(text, sentiment, tokenizer, max_len=128):
    encoding = tokenizer.encode_plus(
        sentiment,
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_len
    )
    return encoding


# Finding the sub-text  from text
def find_start_end(text, selected_text):
    start = text.find(selected_text)
    end = start + len(selected_text)
    return start, end


class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = row.text
        selected_text = row.selected_text
        sentiment = row.sentiment

        # ×˜×•×§× ×™×–×¦×™×” ×ª×§×™× ×”: ×©×•×�×¨×ª offset ×©×œ text ×‘×œ×‘×“
        encoding = self.tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        offsets = encoding['offset_mapping'].squeeze()

        start_idx, end_idx = find_start_end(text, selected_text)

        target_start, target_end = 0, 0
        for i, (start, end) in enumerate(offsets):
            if start <= start_idx and end > start_idx:
                target_start = i
            if start < end_idx and end >= end_idx:
                target_end = i

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor(target_start, dtype=torch.long),
            'end_positions': torch.tensor(target_end, dtype=torch.long)
        }



class TweetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)  # 2 ×›×™ start + end

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)



def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    return (start_loss + end_loss) / 2


train_df = train_df.dropna().reset_index(drop=True)

train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['sentiment'],
    random_state=42
)

# ×™×¦×™×¨×ª ×“×�×˜×”×¡×˜×™×� ×¢×� ×”×˜×•×§× ×™×™×–×¨
train_dataset = TweetDataset(train_data, tokenizer)
val_dataset = TweetDataset(val_data, tokenizer)

# ×™×¦×™×¨×ª ×“×�×˜×�×œ×•×“×¨×™×� ×¢×� ×¡×�×�×¤×œ×¨×™×� ×�×ª×�×™×�×™×�
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=RandomSampler(train_dataset)  # ×œ×“×’×™×�×” ×�×§×¨×�×™×ª ×‘×�×™×�×•×Ÿ
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    sampler=SequentialSampler(val_dataset)  # ×¡×“×¨ ×§×‘×•×¢ ×‘×•×•×œ×™×“×¦×™×”
)


model = TweetModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)
total_steps = len(train_loader) * 3  # 3 epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def train_epoch(model, data_loader, optimizer, scheduler, max_grad_norm=1.0):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_pos = batch['start_positions'].to(device)
        end_pos = batch['end_positions'].to(device)

        model.zero_grad()
        start_logits, end_logits = model(input_ids, attention_mask)
        loss = loss_fn(start_logits, end_logits, start_pos, end_pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)



def predict(model, tokenizer, test_df, device):
    model.eval()
    selected_texts = []

    for i in tqdm(range(len(test_df))):
        text = test_df.loc[i, 'text']
        sentiment = test_df.loc[i, 'sentiment']

        # ×”×©×ª×�×© ×‘×�×•×ª×• ×§×™×“×•×“ ×›×�×• ×‘×�×™×�×•×Ÿ
        encoding = encode_text(text, sentiment, tokenizer)

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        offset_mapping = encoding["offset_mapping"][0].cpu().numpy()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits, end_logits = outputs  # × × ×™×— ×©×–×” tuple

        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()

        if start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
            selected_texts.append(text)
            continue

        if start_idx > end_idx:
            selected_texts.append(text)
        else:
            start_char = offset_mapping[start_idx][0]
            end_char = offset_mapping[end_idx][1]

            if start_char == 0 and end_char == 0:
                selected_texts.append(text)
            else:
                selected_text = text[start_char:end_char].strip()
                selected_texts.append(selected_text)

    return selected_texts



def evaluate(model, data_loader):
    model.eval()
    jaccard_scores = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_pos = batch['start_positions'].to(device)
            end_pos = batch['end_positions'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)

            for i in range(len(start_preds)):
                input_id = input_ids[i]
                tokens = tokenizer.convert_ids_to_tokens(input_id)
                pred_text = tokenizer.convert_tokens_to_string(tokens[start_preds[i]:end_preds[i]+1])
                true_text = tokenizer.convert_tokens_to_string(tokens[start_pos[i]:end_pos[i]+1])

                score = jaccard(pred_text, true_text)
                jaccard_scores.append(score)

    return sum(jaccard_scores) / len(jaccard_scores)



# ×�×¡×¤×¨ ×§×™×¤×•×œ×™×�
n_splits = 5

# ×—×œ×•×§×” ×œÖ¾Folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ×�×—×¡×•×Ÿ ×ª×•×¦×�×•×ª
all_val_jaccards = []
trained_models = []
val_loaders_per_fold = []

# ×—×“×©: ×©×�×™×¨×ª ×”×™×¡×˜×•×¨×™×™×ª ×œ×•×¡ ×•-Jaccard ×œ×›×œ ×§×™×¤×•×œ
all_train_losses = []
all_val_jaccards_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['sentiment'])):
    print(f'
========== Fold {fold+1} / {n_splits} ==========' )

    # ×—×œ×•×§×ª ×”×“×�×˜×”
    train_data = train_df.iloc[train_idx].reset_index(drop=True)
    val_data = train_df.iloc[val_idx].reset_index(drop=True)

    # ×™×¦×™×¨×ª Dataset ×•Ö¾DataLoader
    train_dataset = TweetDataset(train_data, tokenizer)
    val_dataset = TweetDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    val_loaders_per_fold.append(val_loader)

    # ×�×ª×—×•×œ ×�×•×“×œ ×•×�×•×¤×˜×™×�×™×™×–×¨ ×�×—×“×© ×œ×›×œ Fold
    model = TweetModel()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 5  # 5 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # ×œ×•×œ×�×ª ×�×™×�×•×Ÿ
    train_losses = []
    val_jaccards = []
    best_jaccard = 0
    patience = 2
    patience_counter = 0

    for epoch in range(5):
        print(f'
Epoch {epoch + 1}')
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_jaccard = evaluate(model, val_loader)

        train_losses.append(train_loss)
        val_jaccards.append(val_jaccard)

        print(f'Train Loss: {train_loss:.4f} | Validation Jaccard: {val_jaccard:.4f}')

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

    print(f'
âœ… Fold {fold+1} Finished. Best Jaccard: {best_jaccard:.4f}')

    # ×©×�×™×¨×”
    all_val_jaccards.append(best_jaccard)
    trained_models.append(model)
    all_train_losses.append(train_losses)
    all_val_jaccards_per_fold.append(val_jaccards)



# === ×¡×™×›×•×� ===
print("\n======== Cross Validation Summary ========")
print(f"Jaccard Scores per Fold: {all_val_jaccards}")
print(f"Mean Jaccard: {np.mean(all_val_jaccards):.4f} | Std: {np.std(all_val_jaccards):.4f}")


# === ×�×¦×™×�×ª ×”×§×™×¤×•×œ ×”×›×™ ×˜×•×‘ ===
best_fold = np.argmax([max(j) for j in all_val_jaccards_per_fold])
print(f"\nğŸ“ˆ ×”×§×™×¤×•×œ ×¢×� Jaccard ×”×›×™ ×’×‘×•×”: Fold {best_fold + 1}")


# ×§×¨×™×�×ª ×§×‘×¦×™ ×”×˜×¡×˜ ×•×”×”×’×©×” ×œ×“×•×’×�×”
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sample_submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

# ×™×¦×™×¨×ª ×ª×—×–×™×•×ª
predictions = predict(model, tokenizer, test_df, device)

# ×¢×“×›×•×Ÿ ×”×¢×�×•×“×” ×‘×§×•×‘×¥ ×”×”×’×©×”
sample_submission["selected_text"] = predictions

# ×©×�×™×¨×ª ×”×§×•×‘×¥
sample_submission.to_csv("submission.csv", index=False)
print("File created successfully: submission.csv")



sample_submission.head(10)


val_predictions = predict(model, tokenizer, val_data.reset_index(drop=True), device)



val_data = val_data.reset_index(drop=True)

total_score = 0.0
for i in range(len(val_data)):
    true_text = val_data.loc[i, 'selected_text']
    pred_text = val_predictions[i]
    total_score += jaccard(true_text, pred_text)

average_jaccard = total_score / len(val_data)
print(f'Jaccard Score on validation set: {average_jaccard:.4f}')



val_data['predicted_text'] = val_predictions

val_data['jaccard_score'] = val_data.apply(
    lambda row: jaccard(row['selected_text'], row['predicted_text']),
    axis=1
)

val_data[['textID', 'text', 'sentiment', 'selected_text', 'predicted_text', 'jaccard_score']].head(10)



specific_id = "a088ac278e"
row = val_data[val_data['textID'] == specific_id]
if not row.empty:
    print("text:", row.iloc[0]['text'])
    print("\nsentiment:", row.iloc[0]['sentiment'])
    print("\nselected_text:", row.iloc[0]['selected_text'])
    print("\npredicted_text:", row.iloc[0]['predicted_text'])
    print("\nJaccard:", jaccard(row.iloc[0]['selected_text'], row.iloc[0]['predicted_text']))
else:
    print("id not found")



# === ×¤×•× ×§×¦×™×•×ª ×’×¨×¤×™×� ===
def plot_loss(train_losses):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train Loss', marker='o', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_jaccard(val_jaccards):
    plt.figure(figsize=(6, 4))
    plt.plot(val_jaccards, label='Validation Jaccard', marker='x', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Jaccard Score')
    plt.title('Validation Jaccard over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()


# === ×¦×™×•×¨ ×’×¨×¤×™×� ×œ×›×œ ×”×§×™×¤×•×œ×™×� ===
for i in range(n_splits):
    print(f'
=== Fold {i+1} ===')
    plot_loss(all_train_losses[i])
    plot_jaccard(all_val_jaccards_per_fold[i])


