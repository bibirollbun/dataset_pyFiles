import os
os.system("pip install --no-index --find-links /kaggle/input/install-bits-and-bytes-try2 bitsandbytes")


import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['target'] = train.target.apply(lambda x: "_".join(x.split('_')[1:]))
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()


test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
test['label'] = 0
print( test.shape )
test.head()


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
tmp = train.loc[idx].copy()
tmp['c'] = tmp.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
tmp = tmp.sort_values('c',ascending=False)
tmp = tmp.drop_duplicates(['QuestionId'])
tmp = tmp[['QuestionId','MC_Answer']]
tmp['is_correct'] = 1

train = train.merge(tmp, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

test = test.merge(tmp, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)


tmp = train.groupby(['QuestionId','MC_Answer']).target.unique()
tmp.name = 'possible'

tmp = tmp.apply(lambda x: ", ".join(sorted(x)).replace("True_","").replace("False_","").replace(":NA","")
               .replace("Neither","or Not an explanation.").replace("Correct","Correct explanation (regardless of whether reponse is correct)") )

train = train.merge(tmp, on=['QuestionId','MC_Answer'], how='left')
test = test.merge(tmp, on=['QuestionId','MC_Answer'], how='left')

print("Here are examples:\n")
for k in range(8,12):
    i = tmp.index[k]
    c = tmp.iloc[k]
    print(f"QuestionId {i[0]}, MC_Answer {i[1]}\n => has classes = [{c}]")


import pandas as pd, numpy as np
mc = pd.read_csv("/kaggle/input/map-files/MC.csv")
mc = mc.rename({"choice":"cc"},axis=1)
print("Here are examples:\n")
mc.head(4)


train = train.merge(mc, on=['QuestionId','MC_Answer'], how='left')
test = test.merge(mc, on=['QuestionId','MC_Answer'], how='left')


choice_map = {}
tmp = train.groupby(['QuestionId','MC_Answer','cc']).size().reset_index(name='count')
tmp = tmp.sort_values(['QuestionId','cc'])

for q in tmp.QuestionId.unique():
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = ", ".join([f"{labels[i]}) {choice}" for i, choice in enumerate(choices)])
    choice_map[q] = choice_str


import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

model_path = f"/kaggle/input/model-v2422b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  


def format_input(row):
    return (
        f"You are an expert mathematics assessor.\n"
        f"Question: {row['QuestionText']}\n"
        f"Choices: {choice_map[row['QuestionId']]}\n"
        f"Response: {row['MC_Answer']}\n"
        f"Explanation: {row['StudentExplanation']}\n"
        f"Now classify the explanation as {row['possible']}"
    )


train['text'] = train.apply(format_input,axis=1)
test['text'] = test.apply(format_input,axis=1)


print("Here is an example prompt:\n")
print( train.text.values[0] )


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding=False, truncation=True, max_length=320)

COLS = ['text','label']
test_ds = Dataset.from_pandas(test[COLS])
test_ds = test_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
test_ds.set_format(type='torch', columns=columns)


from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,           
    bnb_4bit_compute_dtype="bfloat16",  
    bnb_4bit_use_double_quant=True,     
    bnb_4bit_quant_type="nf4"           
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=n_classes,
    quantization_config=quant_config,
    device_map="auto",  
    torch_dtype=torch.bfloat16,
    ignore_mismatched_sizes=True,
)

model.config.id2label = {i: f"LABEL_{i}" for i in range(n_classes)}
model.config.label2id = {v: k for k, v in model.config.id2label.items()}

model = PeftModel.from_pretrained(model, "/kaggle/input/model-v717d")


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=None,
    do_train=True,
    do_eval=True,
    eval_strategy="no",
    save_strategy="no", 
    num_train_epochs=2,
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=16,
    #gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_dir=None,
    logging_steps=50,
    save_steps=500,
    eval_steps=200,
    save_total_limit=1,
    report_to="none",  
    bf16=False,           
    fp16=True,   
    #gradient_checkpointing=True,
)


from sklearn.metrics import average_precision_score
import numpy as np

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=train_ds,
    #eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
    data_collator=data_collator,
)

#trainer.train()


predictions = trainer.predict(test_ds)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


le2 = LabelEncoder()
train2 = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train2.Misconception = train2.Misconception.fillna('NA')
train2['target'] = train2.Category+":"+train2.Misconception
train2['label'] = le2.fit_transform(train2['target'])

d2 = dict(zip(le2.classes_,le2.transform(le2.classes_)))

mm1 = {}
for i,c in enumerate(le.classes_):
    c2 = f"True_{c}"
    if c2 in d2:
        mm1[int(d2[c2])] = i
mm2 = {}
for i,c in enumerate(le.classes_):
    c2 = f"False_{c}"
    if c2 in d2:
        mm2[int(d2[c2])] = i

i1 = np.array(list(mm1.keys()))
i2 = np.array(list(mm1.values()))
probs3 = np.zeros((len(test),65))
probs3[:,i1] = probs[:,i2]

i1 = np.array(list(mm2.keys()))
i2 = np.array(list(mm2.values()))
probs4 = np.zeros((len(test),65))
probs4[:,i1] = probs[:,i2]

probs5 = np.zeros((len(test),65))
idx = (test.is_correct==1).values
probs5[idx,:] = probs3[idx,:]
probs5[~idx,:] = probs4[~idx,:]


# Get top 3 predicted class indices
top3 = np.argsort(-probs5, axis=1)[:, :3]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le2.inverse_transform(flat_top3)
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

