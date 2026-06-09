import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
#model_name = "google/gemma-2-9b-it"
model_name = "/kaggle/input/gemma2-9b-it-cv945"
EPOCHS = 2

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))


import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256


def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )


lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort( lengths )


# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/input/gemma2-9b-it-bf16",
    num_labels=n_classes,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


from peft import PeftModel
model = PeftModel.from_pretrained(model, model_name)


training_args = TrainingArguments(
    output_dir = f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps", #no for no saving 
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
)


# CUSTOM MAP@3 METRIC

from sklearn.metrics import average_precision_score

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
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

#trainer.train()


#trainer.save_model(f"ver_{VER}")      
#tokenizer.save_pretrained(f"ver_{VER}")


test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test.shape )
test.head()


test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)

test.head()


ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
joined_preds = ["|".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission_gemma.csv", index=False)
sub.head()


sub.iloc[0]['Category:Misconception']


import torch
import gc

del top3_labels, flat_top3, decoded_labels, top3, test, ds_test
del training_args, train_ds, val_ds, model, trainer, predictions, probs
# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, ModernBertForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset



train               = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test                = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')



le                  = LabelEncoder()
train.Misconception     = train.Misconception.fillna('NA')
train['target']   = train.Category + ':' +train.Misconception
train['label']    = le.fit_transform(train['target'])

n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)





test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)




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

test['text'] = test.apply(format_input,axis=1)


ds_test = Dataset.from_pandas(test)




model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL", device_map="cuda:0", torch_dtype=torch.bfloat16)






tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL")
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

ds_test = ds_test.map(tokenize, batched=True)






test_args = TrainingArguments(
    output_dir="./",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=16, # Adjust as needed
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=test_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

predictions = trainer.predict(ds_test)

predictions.predictions






top3           = np.argsort(-predictions.predictions, axis=1)[:, :]
flat_top3      = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels_cat    = decoded_labels.reshape(top3.shape)
top3_labels_cat






joined_preds = []

for preds in top3_labels_cat:
    joined_preds.append("|".join(preds))



# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission_deepseek.csv", index=False)
sub.head()




sub.iloc[0]['Category:Misconception']


import torch
import gc

del top3_labels_cat, flat_top3, decoded_labels, top3, test, ds_test
del test_args, model, trainer, predictions
# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, ModernBertForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset


train               = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test                = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


le                  = LabelEncoder()
train.Misconception     = train.Misconception.fillna('NA')
train['target']   = train.Category + ':' +train.Misconception
train['label']    = le.fit_transform(train['target'])

n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)



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

test['text'] = test.apply(format_input,axis=1)


ds_test = Dataset.from_pandas(test)


model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL", device_map="cuda:1", torch_dtype=torch.bfloat16)


model



tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

ds_test = ds_test.map(tokenize, batched=True)


test_args = TrainingArguments(
    output_dir="./",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=1, # Adjust as needed
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=test_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

predictions = trainer.predict(ds_test)

predictions.predictions


top3           = np.argsort(-predictions.predictions, axis=1)[:, :]
flat_top3      = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels_cat    = decoded_labels.reshape(top3.shape)
top3_labels_cat





joined_preds = []

for preds in top3_labels_cat:
    joined_preds.append("|".join(preds))



# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission_gemma3.csv", index=False)
sub.head()




sub.iloc[0]['Category:Misconception']


from collections import defaultdict

def get_top_k_ensemble(l1, l2, l3, k=3):
    list1, list2, list3 = l1.split('|'), l2.split('|'), l3.split('|')
    weights = [4, 4, 4]  # độ tin cậy: list1 < list2 > list3 
    lists = [list1, list2, list3]
    score = defaultdict(int)

    for i, lst in enumerate(lists):
        weight = weights[i]
        for rank, item in enumerate(lst):
            score[item] += (len(lst) - rank) * weight

    # Sắp xếp theo điểm giảm dần
    sorted_items = sorted(score.items(), key=lambda x: -x[1])
    return ' '.join([item for item, _ in sorted_items[:k]])

list1 = 'a|b|d|f'
list2 = 'b|c|a|e'
list3 = 'c|e|b'

print(get_top_k_ensemble(list1, list2, list3, k=3))


df1 = pd.read_csv('submission_gemma.csv').rename(columns = {'Category:Misconception':'Category:Misconception_gemma'})
df2 = pd.read_csv('submission_deepseek.csv').rename(columns = {'Category:Misconception':'Category:Misconception_deepseek'})
df3 = pd.read_csv('submission_gemma3.csv').rename(columns = {'Category:Misconception':'Category:Misconception_gemma3'})


df = pd.merge(df1, df2, on = 'row_id', how = 'inner')
df = pd.merge(df, df3, on = 'row_id', how = 'inner')


df['Category:Misconception'] = df.apply(lambda x: get_top_k_ensemble(x['Category:Misconception_gemma'], x['Category:Misconception_deepseek'], x['Category:Misconception_gemma3']), axis = 1)
df[['row_id', 'Category:Misconception']].to_csv('submission.csv', index = False)
pd.read_csv('submission.csv')




