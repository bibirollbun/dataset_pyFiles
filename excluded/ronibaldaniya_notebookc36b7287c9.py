import os
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import Dataset

# Path to your uploaded model folder
dataset_path = '/kaggle/input/local-model'

label_names = [
    'Adding_across', 'Adding_terms', 'Additive', 'Base_rate', 'Certainty', 'Definition',
    'Denominator-only_change', 'Division', 'Duplication', 'Firstterm', 'FlipChange',
    'Ignores_zeroes', 'Incomplete', 'Incorrect_equivalent_fraction_addition', 'Interior',
    'Inverse_operation', 'Inversion', 'Irrelevant', 'Longer_is_bigger', 'Mult',
    'Multiplying_by_4', 'Not_variable', 'Positive', 'Scale', 'Shorter_is_bigger',
    'Subtraction', 'SwapDividend', 'Tacking', 'Unknowable', 'WNB', 'Whole_numbers_larger',
    'Wrong_Fraction', 'Wrong_Operation', 'Wrong_fraction', 'Wrong_term'
]

# Load test set
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# Combine input fields
test['text'] = (
    test['StudentExplanation'].fillna('') + ' ' +
    test['MC_Answer'].fillna('') + ' ' +
    test['QuestionText'].fillna('')
)

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join(dataset_path, 'local_tokenizer'))
model = DistilBertForSequenceClassification.from_pretrained(os.path.join(dataset_path, 'local_model'), num_labels=len(label_names))

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

dataset = Dataset.from_dict({'text': test['text'].tolist()})
dataset = dataset.map(tokenize, batched=True)
dataset.set_format('torch', columns=['input_ids', 'attention_mask'])

model.eval()
preds = []
batch_size = 32

for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    with torch.no_grad():
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        pred = torch.argmax(logits, axis=1).cpu().numpy()
        preds.extend(pred)

pred_labels = [label_names[i] for i in preds]

# Prepare submission with 3 predictions per row as required by the competition
# We add placeholders for Category since model only predicts Misconception
# Order: True_Correct:NA False_Neither:NA False_Misconception:<predicted_label>

submission_preds = [
    f"True_Correct:NA False_Neither:NA False_Misconception:{label}"
    for label in pred_labels
]

submission = pd.DataFrame({
    'row_id': test['row_id'],
    'Category:Misconception': submission_preds
})

submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created in correct format!")


