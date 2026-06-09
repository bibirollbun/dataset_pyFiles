import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

le = LabelEncoder()
train['target'] = train['Category'] + ':' + train['Misconception'].fillna('NA')
le.fit(train['target'])

correct_answers = (train[train['Category'].str.startswith('True')]
                   .groupby(['QuestionId','MC_Answer']).size()
                   .reset_index(name='count')
                   .sort_values('count', ascending=False)
                   .drop_duplicates('QuestionId')[['QuestionId','MC_Answer']]
                   .assign(is_correct=1))

test = test.merge(correct_answers, on=['QuestionId','MC_Answer'], how='left')
test['is_correct'] = test['is_correct'].fillna(0)
test['text'] = test.apply(lambda x: f"Question: {x['QuestionText']}\nAnswer - {x['MC_Answer']}\n{'Correct' if x['is_correct'] else 'Incorrect'}\nExplanation - {x['StudentExplanation']}", axis=1)

m = "/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL"
model = AutoModelForSequenceClassification.from_pretrained(m, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(m)
device = next(model.parameters()).device

logits = []
for text in test['text']:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits.append(model(**inputs).logits.cpu().numpy()[0])

probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
probs /= probs.sum(axis=1, keepdims=True)

pd.DataFrame({
    'row_id': test['row_id'],
    'Category:Misconception': [' '.join(row) for row in le.inverse_transform(np.argsort(-probs, axis=1)[:, :3].flatten()).reshape(-1, 3)]
}).to_csv('submission.csv', index=False)

