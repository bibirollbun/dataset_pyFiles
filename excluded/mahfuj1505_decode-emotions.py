
!pip install -q transformers pandas emoji torch

import pandas as pd
import emoji
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tqdm.pandas()


test_df = pd.read_csv('/kaggle/input/binary-biplob-can-you-decode-emotions/bangla/test.csv')
test_df['text'] = test_df['text'].astype(str)


def preprocess(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ").replace(":", "")
    return text.strip()

test_df['text'] = test_df['text'].apply(preprocess)


model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

labels = ['negative', 'neutral', 'positive']


def predict_sentiment(text):
    try:
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            output = model(**encoded_input)
            scores = F.softmax(output.logits, dim=1)
            label_id = torch.argmax(scores).item()
            return labels[label_id]
    except:
        return "neutral"  # safe fallback


test_df['label'] = test_df['text'].progress_apply(predict_sentiment)


submission = test_df[['id', 'label']]
submission.to_csv('submission3.csv', index=False)
print("✅ Final submission saved as submission.csv")


