!pip install pylatexenc





import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
test = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')

test['label'] = np.nan
train.name = 'Training Set'
test.name = 'Test Set'


print(train.info())
print(train.describe())


print(train.columns)


plt.hist(train['label'])
plt.show()




train['label'].value_counts()


plt.hist(train['label'])




from pylatexenc.latex2text import LatexNodes2Text
def clean_latex(latex_string):
    return LatexNodes2Text().latex_to_text(latex_string)


import re
def remove_question_artifacts(text):
    text = re.sub(r'\b(Question|Example|GS)\.?\s*\d+[:.]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*[\d+\.]*', '', text)
    return text.strip()


!pip install unidecode
!pip install ftfy


import ftfy
from unidecode import unidecode

def normalize_unicode(text):
    return unidecode(ftfy.fix_text(text))


def reduce_formula_noise(text):
    text = re.sub(r'\d+([*/+-]\d+)+', '[MATH]', text)
    text = re.sub(r'\b\d+\b', '[NUM]', text)
    text = re.sub(r'([a-zA-Z]\s*[\+\-\*/=^]\s*)+[a-zA-Z0-9]+', '[MATH]', text)

    return text


import spacy
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    return ' '.join([token.lemma_ for token in nlp(text)])


def full_cleaning_pipeline(text):
    text = clean_latex(text)
    text = normalize_unicode(text)
    text = remove_question_artifacts(text)
    text = reduce_formula_noise(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = lemmatize_text(text)
    return text.strip()

train['Question_cleaned'] = train['Question'].apply(full_cleaning_pipeline)
test['Question_cleaned'] = test['Question'].apply(full_cleaning_pipeline)



!pip install transformers


!pip install torch transformers


import torch
from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2ForSequenceClassification,Trainer, TrainingArguments


tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")


train_texts = list(train['Question_cleaned'])
test_texts = list(test['Question_cleaned'])


train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512,return_tensors="pt")



from torch.utils.data import Dataset

class MathDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MathDataset(train_encodings, train['label'].tolist())


num_labels = len(set(train['label']))
model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=num_labels
)


from sklearn.utils.class_weight import compute_class_weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = np.array(train['label'])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)


import torch.nn as nn

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False , **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


 training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    save_strategy="epoch",
    report_to="none",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=4,
    weight_decay=0.01,
    gradient_accumulation_steps=8,
    fp16=True,
    save_total_limit=2,
    logging_steps=50,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    loss_fn=loss_fn
)

trainer.train()


from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_inputs = tokenizer(
    list(test['Question_cleaned']),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'])
test_loader = DataLoader(test_dataset, batch_size=8)

preds = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1)
        preds.extend(batch_preds.cpu().numpy())

test['label'] = preds



test[['id', 'label']].to_csv('predictions.csv', index=False)


trainer.save_model('/kaggle/working/my_model')

