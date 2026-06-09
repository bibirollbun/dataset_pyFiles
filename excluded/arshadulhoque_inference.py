# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')


def url_to_semantics(text):
    """Extract meaningful keywords from URLs in text and capitalize them."""
    if not isinstance(text, str):
        return ""
    url_pattern = re.compile(r'https?://(?:www\.)?([^/?]+)(?:[^/]*)(/[^/?]*)')
    urls = url_pattern.findall(text)
    if not urls:
        return ""
    keywords = []
    for domain, path in urls:
        domain = domain.split('.')[0]
        if domain and domain not in ['www', 'http', 'https']:
            keywords.append(f"domain:{domain}")
        if path and len(path) > 1:
            path_parts = path.strip('/').split('/')
            for part in path_parts:
                if part and not part.isdigit() and len(part) > 2:
                    if part.lower() not in ['jpg', 'jpeg', 'png', 'gif', 'html', 'php', 'asp', 'aspx']:
                        keywords.append(f"path:{part}")
                        break
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    if unique_keywords:
        capitalized_keywords = [kw.upper() for kw in unique_keywords]
        return "URL KEYWORDS: " + " ".join(capitalized_keywords)
    else:
        return ""


def extract_keywords(text):
    if not isinstance(text, str):
        return []
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    keywords = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
    return keywords

def calculate_jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection, union = set1.intersection(set2), set1.union(set2)
    return len(intersection) / len(union) if union else 0

def get_sentiment_features(text):
    if not isinstance(text, str):
        return 0, 0
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores['compound'], scores['pos'] - scores['neg']


def create_example_features(row):
    pos1_keywords = extract_keywords(row['positive_example_1'])
    pos2_keywords = extract_keywords(row['positive_example_2'])
    neg1_keywords = extract_keywords(row['negative_example_1'])
    neg2_keywords = extract_keywords(row['negative_example_2'])
    all_pos_keywords = pos1_keywords + pos2_keywords
    all_neg_keywords = neg1_keywords + neg2_keywords
    comment_keywords = extract_keywords(row['body'])
    pos_overlap = calculate_jaccard_similarity(comment_keywords, all_pos_keywords)
    neg_overlap = calculate_jaccard_similarity(comment_keywords, all_neg_keywords)
    comment_compound, comment_sent_diff = get_sentiment_features(row['body'])
    pos1_compound, _ = get_sentiment_features(row['positive_example_1'])
    pos2_compound, _ = get_sentiment_features(row['positive_example_2'])
    neg1_compound, _ = get_sentiment_features(row['negative_example_1'])
    neg2_compound, _ = get_sentiment_features(row['negative_example_2'])
    avg_pos_sentiment = (pos1_compound + pos2_compound) / 2
    avg_neg_sentiment = (neg1_compound + neg2_compound) / 2
    pos_sent_diff = abs(comment_compound - avg_pos_sentiment)
    neg_sent_diff = abs(comment_compound - avg_neg_sentiment)
    comment_len = len(row['body'].split())
    pos1_len = len(row['positive_example_1'].split())
    pos2_len = len(row['positive_example_2'].split())
    neg1_len = len(row['negative_example_1'].split())
    neg2_len = len(row['negative_example_2'].split())
    avg_pos_len = (pos1_len + pos2_len) / 2
    avg_neg_len = (neg1_len + neg2_len) / 2
    pos_len_ratio = comment_len / avg_pos_len if avg_pos_len > 0 else 0
    neg_len_ratio = comment_len / avg_neg_len if avg_neg_len > 0 else 0
    features = (
        f"Example Features: "
        f"pos_overlap:{pos_overlap:.3f} "
        f"neg_overlap:{neg_overlap:.3f} "
        f"pos_sent_diff:{pos_sent_diff:.3f} "
        f"neg_sent_diff:{neg_sent_diff:.3f} "
        f"pos_len_ratio:{pos_len_ratio:.3f} "
        f"neg_len_ratio:{neg_len_ratio:.3f} "
        f"comment_sentiment:{comment_sent_diff:.3f}"
    )
    return features

def create_combined_text(row):
    """
    Creates a combined text string using the hybrid approach from training.
    This is the KEY function that must match training exactly.
    """
    subreddit_text = str(row['subreddit']).upper()
    rule_text = str(row['rule']).upper()
    pos_examples = f"{str(row['positive_example_1']).upper()} {str(row['positive_example_2']).upper()}"
    neg_examples = f"{str(row['negative_example_1']).upper()} {str(row['negative_example_2']).upper()}"
    body_text = str(row['body']).upper()
    url_info = url_to_semantics(row['body'])
    features_str = create_example_features(row)
    combined_text = (
        f"SUBREDDIT: {subreddit_text} [SEP] "
        f"RULE: {rule_text} [SEP] "
        f"POSITIVE EXAMPLES: {pos_examples} [SEP] "
        f"NEGATIVE EXAMPLES: {neg_examples} [SEP] "
        f"COMMENT: {body_text} {url_info} [SEP] "
        f"{features_str.upper()}"
    )
    return combined_text


# Load test data
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

print("Creating combined text for inference...")
test_df['combined_text'] = test_df.apply(create_combined_text, axis=1)


# Create test dataset
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
        }


model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/modernbert-base-trained/pytorch/default/4/model")
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/modernbert-base-trained/pytorch/default/4/model")


# Create test dataset
test_dataset = TestDataset(test_df['combined_text'].tolist(), tokenizer)

# Prediction function
def predict_with_model(model, test_dataset, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_predictions.extend(probs)
    return all_predictions

# Get predictions
predictions = predict_with_model(model, test_dataset)


# Create submission file
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': predictions
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file created successfully with updated preprocessing!")

