import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, ModernBertForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import Dataset


train               = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test                = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


le_cat                  = LabelEncoder()
le_misco                = LabelEncoder()
train.Misconception     = train.Misconception.fillna('NA')
train['target_misco']   = train.Misconception
train['target_cat']     = train.Category
train['label_misco']    = le_misco.fit_transform(train['target_misco'])
train['label_cat']      = le_cat.fit_transform(train['target_cat'])
n_classes_cat           = len(le_cat.classes_)
n_classes_misco         = len(le_misco.classes_)
print(f"Train shape: {train.shape} with {n_classes_cat} cat target classes and {n_classes_misco} misco target classes")


def format_input(row):
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

test['text'] = test.apply(format_input,axis=1)

ds_test = Dataset.from_pandas(test[['text']])


model1 = ModernBertForSequenceClassification.from_pretrained("/kaggle/input/mbertmodel-01/ver_1/checkpoint-5505", device_map="cuda:0")


model2 = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/mbertmodel-02/ver_1/checkpoint-5505", device_map="cuda:0")


model2


model1


tokenizer1 = AutoTokenizer.from_pretrained("/kaggle/input/mbertmodel-01/ver_1/checkpoint-5505")
def tokenize1(batch):
    return tokenizer1(batch["text"], padding="max_length", truncation=True, max_length=256)


tokenizer2 = AutoTokenizer.from_pretrained("/kaggle/input/mbertmodel-02/ver_1")
def tokenize2(batch):
    return tokenizer2(batch["text"], padding="max_length", truncation=True, max_length=256)

ds_test1 = ds_test.map(tokenize1, batched=True)
ds_test2 = ds_test.map(tokenize2, batched=True)


test_args = TrainingArguments(
    output_dir="./",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=16, # Adjust as needed
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True
)


# trainer1 = Trainer(
#     model=model1,
#     args=test_args,
#     tokenizer=tokenizer1
# )


# predictions1 = trainer1.predict(ds_test1)


model1.device


import torch

model1.eval() # Set model to evaluation mode

all_predictions = []
with torch.no_grad(): 
    for i in range(len(ds_test1)):
        sample = ds_test1[i]
        # Assuming 'sample' needs to be formatted for your model (e.g., dict with 'input_ids', 'attention_mask')
        # You might need to add a batch dimension if your model expects it
        inputs = {k: torch.tensor(v).unsqueeze(0).to('cuda:0') for k, v in sample.items() if k != 'text'}
        #inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in sample.items() if k != 'text'} # Example for dict-like inputs

        outputs = model1(**inputs)
        # Process outputs (e.g., get logits, probabilities, predicted labels)
        predictions = outputs.logits.cpu().numpy() # Example for classification
        all_predictions.append(predictions)

# Concatenate predictions if needed
final_predictions = np.concatenate(all_predictions, axis=0)
probs = torch.nn.functional.softmax(torch.tensor(final_predictions), dim=1).numpy()


import torch

model2.eval() # Set model to evaluation mode

all_predictions1 = []
with torch.no_grad(): 
    for i in range(len(ds_test1)):
        sample = ds_test1[i]
        # Assuming 'sample' needs to be formatted for your model (e.g., dict with 'input_ids', 'attention_mask')
        # You might need to add a batch dimension if your model expects it
        inputs = {k: torch.tensor(v).unsqueeze(0).to('cuda:0') for k, v in sample.items() if k != 'text'}
        #inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in sample.items() if k != 'text'} # Example for dict-like inputs

        outputs = model2(**inputs)
        # Process outputs (e.g., get logits, probabilities, predicted labels)
        predictions = outputs.logits.cpu().numpy() # Example for classification
        all_predictions1.append(predictions)

# Concatenate predictions if needed
final_predictions1 = np.concatenate(all_predictions1, axis=0)
probs1 = torch.nn.functional.softmax(torch.tensor(final_predictions1), dim=1).numpy()


top3           = np.argsort(-probs, axis=1)[:, :3]
flat_top3      = top3.flatten()
decoded_labels = le_cat.inverse_transform(flat_top3)
top3_labels_cat    = decoded_labels.reshape(top3.shape)
top3_labels_cat


top3_misco           = np.argsort(-probs1, axis=1)[:, :3]
flat_top3            = top3_misco.flatten()
decoded_labels       = le_misco.inverse_transform(flat_top3)
top3_labels_misco          = decoded_labels.reshape(top3.shape)
top3_labels_misco


joined_preds = []

for idx, sample in enumerate(top3_labels_cat):
    miscos = top3_labels_misco[idx]
    temp = ''
    for idx, cat in enumerate(sample):
        if cat in ['True_Misconception','False_Misconception']:
            temp += cat + ':' + miscos[idx]
        else:
            temp += cat + ':' + 'NA'
        if idx<3:
            temp += ' ' 
    joined_preds.append(temp)

print(joined_preds)

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()




