import random
import numpy as np
import torch
import tensorflow as tf
from sklearn import config_context

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tf.random.set_seed(seed)
    
    with config_context(global_random_seed=seed):
        pass

fix_seed()


! unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip
! unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip


import pandas as pd

df = pd.read_csv('train.csv')
df.head(1)


texts = df['comment_text'].tolist()
labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


import string

def basic_text_prep(texts):
    filtered_texts = []
    for text in tqdm(texts):
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text)
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
        filtered_texts.append(filtered_words)
    joined_filtered_texts = [' '.join(text) for text in filtered_texts]
    return joined_filtered_texts


train_data = basic_text_prep(texts)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
import numpy as np

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(LogisticRegression())) 
])

scoring = {
    'f1_micro': make_scorer(f1_score, average='micro'),
    'precision_micro': make_scorer(precision_score, average='micro'),
    'recall_micro': make_scorer(recall_score, average='micro')
}

results = cross_validate(pipeline, train_data, np.array(labels), cv=4, scoring=scoring)

print("F1 (Micro):", results['test_f1_micro'])
print("Precision (Micro):", results['test_precision_micro'])
print("Recall (Micro):", results['test_recall_micro'])


pipeline.fit(train_data, np.array(labels))


test_df = pd.read_csv('test.csv')
test_texts = test_df['comment_text'].tolist()
test_data = basic_text_prep(test_texts)


predicted_probs = pipeline.predict_proba(test_texts)
predicted_probs = np.array(predicted_probs)[:, :, 1]


def form_submission(test_df, predicted_probs):
    submission = pd.DataFrame({
        'id': test_df['id'],
        'toxic': predicted_probs[0],
        'severe_toxic': predicted_probs[1],
        'obscene': predicted_probs[2],
        'threat': predicted_probs[3],
        'insult': predicted_probs[4],
        'identity_hate': predicted_probs[5]
    })
    submission.to_csv('submission.csv', index=False)

form_submission(test_df, predicted_probs)


from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
test_df = pd.read_csv('test.csv')
test_texts = test_df['comment_text'].tolist()


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')


val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')


import torch
from torch.utils.data import Dataset

class ToxicityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ToxicityDataset(train_encodings, train_labels)
val_dataset = ToxicityDataset(val_encodings, val_labels)


from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)


from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import f1_score
import numpy as np

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # validation
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.sigmoid(logits)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    val_preds = (np.array(val_preds) > 0.5).astype(int)
    f1 = f1_score(val_true, val_preds, average='micro')
    print(f"Validation F1 Score: {f1}")


test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
test_dataset = ToxicityDataset(test_encodings, [[0] * 6] * len(test_texts))


test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
test_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.sigmoid(logits)
        test_preds.extend(preds.cpu().numpy())


model.save_pretrained('./bert_toxicity_model')
tokenizer.save_pretrained('./bert_toxicity_model')


form_submission(test_df, np.array(test_preds).T)


# посмотрим на дисбаланс классов
for i in range(6):
    col = labels[:, i]
    print(f'{i}: {sum(col) / len(col)}')


class_weight = {}
for i in range(6):
    col = labels[:, i]
    coef = len(col) / sum(col)
    class_weight[i] = coef
class_weight


train_data = basic_text_prep(texts)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(LogisticRegression(class_weight=class_weight))) 
])

scoring = {
    'f1_micro': make_scorer(f1_score, average='micro'),
    'precision_micro': make_scorer(precision_score, average='micro'),
    'recall_micro': make_scorer(recall_score, average='micro')
}
results = cross_validate(pipeline, train_data, np.array(labels), cv=4, scoring=scoring)


print("F1 (Micro):", results['test_f1_micro'])
print("Precision (Micro):", results['test_precision_micro'])
print("Recall (Micro):", results['test_recall_micro'])


pipeline.fit(train_data, np.array(labels))

predicted_probs = pipeline.predict_proba(test_texts)
predicted_probs = np.array(predicted_probs)[:, :, 1]
form_submission(test_df, predicted_probs)


ensembled_probs = predicted_probs * 0.2 + np.array(test_preds).T * 0.8 
form_submission(test_df, ensembled_probs)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_data)

log_reg = LogisticRegression()
log_reg_multi = MultiOutputClassifier(log_reg)
log_reg_multi.fit(X, labels)


from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50)
X_reduced = svd.fit_transform(X)


random_forest = RandomForestClassifier(n_jobs=2, n_estimators=10)
random_forest_multi = MultiOutputClassifier(random_forest)
random_forest_multi.fit(X_reduced, labels)


X_test = vectorizer.transform(test_data)
X_test_truncated = svd.transform(X_test)


log_reg_probs = log_reg_multi.predict_proba(X_test)
random_forest_probs = random_forest_multi.predict_proba(X_test_truncated)

combined_probs = (np.array(log_reg_probs)[:, :, 1] + np.array(random_forest_probs)[:, :, 1]) / 2

ensembled_probs = combined_probs * 0.4 + np.array(test_preds).T * 0.6
form_submission(test_df, ensembled_probs)

