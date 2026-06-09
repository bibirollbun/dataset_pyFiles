import os


%ls '/kaggle/input/deberta-test/transformers/default/1/best'


%ls '/kaggle/input/label_encoder/other/default/1'


%ls '/kaggle/input/map-charting-student-math-misunderstandings'


import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import AutoModel
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import joblib
import pandas as pd


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
model_name = "/kaggle/input/deberta-test/transformers/default/1/best"
EPOCHS = 10

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


import joblib
import pandas as pd


import joblib

le = joblib.load("/kaggle/input/label_encoder/other/default/1/label_encoder.joblib")


os.listdir('/kaggle/input/c/map-charting-student-math-misunderstandings/')


train = pd.read_csv('/kaggle/input/c/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)


n_classes


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


from transformers import DebertaV2ForSequenceClassification
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained(model_name)
model =  DebertaV2ForSequenceClassification.from_pretrained(
#model = AutoModel.from_pretrained(model_name)(
#model = AutoModelForSequenceClassification.from_pretrained(model_name)(    
    model_name,
    num_labels=n_classes 
)
training_args = TrainingArguments(report_to="none")
trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args)



def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)


test = pd.read_csv('/kaggle/input/c/map-charting-student-math-misunderstandings/test.csv')
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)


import torch
from datasets import Dataset
import numpy as np


ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


probs.shape


top3 = np.argsort(-probs, axis=1)[:, :3]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()


max(flat_top3)


# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :3]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
joined_preds = [" ".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()


%ls /kaggle/working

