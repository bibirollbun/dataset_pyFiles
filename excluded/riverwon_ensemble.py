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


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

VER=1
model_name = "/kaggle/input/gemma2-9b-it-cv945"
EPOCHS = 2

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train['Category'] + ":" + train["Misconception"]
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()


idx = train.apply(lambda row: row.Category.split('_')[0], axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(["QuestionId", 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct']=1

train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset

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


train.head(3)


#토큰별 길이
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train['text']]
lengths[:10]


#split into train and validation sets
train_df, val_df = train_test_split(train, test_size = 0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text', 'label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])


#Tokenization Function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

columns = ['input_ids','attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/input/gemma2-9b-it-bf16",
    num_labels = n_classes,
    torch_dtype = torch.bfloat16,
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


#CUSTOM MAP@3 Metric

from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]
    match = (top3 == labels[:, None])

    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3
)

#trainer.train()
result_gemma = trainer.evaluate()
print("--- Gemma Validation Score ---")
print(result_gemma)


#trainer.save_model(f"ver_{VER}")
#tokenizer.save_pretrained(f"ver_{VER}")


test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
print(test.shape)
test.head()


test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input, axis=1)

test.head()


ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


# get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:,:] #shape: [num_samples, 3]

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


sub.iloc[0]


sub.iloc[0]['Category:Misconception']


import torch
import gc

del top3_labels, flat_top3, decoded_labels, top3, test, ds_test
del training_args, train_ds, val_ds, model, trainer, predictions, probs
# Delete any other lingering references

# 모든 전역 변수를 딕셔너리 형태로
# 딕셔너리 내 값이 torch.nn.Module이거나 torch.Tensor인 경우 삭제
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Pytorch는 메모리 할당 속도를 높이기 위해 더이상 사용되지 않는 텐서가 있더라도 즉시 GPU에 반환하지 않고 Cache형태로 남겨둠
# torch.cuda.empty_cache()는 cache를 비우고 메모리를 GPU에 반환하라는 뜻
torch.cuda.empty_cache()
gc.collect()

torch.cuda.ipc_collect()

# 현재 시점에 tensor들이 실제로 사용하고 있는 메모리 양
print("Memory allocated:", torch.cuda.memory_allocated())
# PyTorch가 GPU로부터 미리 할당받아 확보해 둔 전체 GPU 메모리 양
print("Memory reserved:", torch.cuda.memory_reserved())


"""
GEMINI: 
이 개념을 도서관에서 공부하는 상황에 비유하면 쉽게 이해할 수 있습니다.

memory_allocated() (할당된 메모리)

내가 현재 펼쳐서 읽고 있는 책들이 차지하는 공간입니다.

새 책을 펼치면 이 공간은 늘어나고, 다 읽은 책을 덮으면 줄어듭니다.

memory_reserved() (예약된 메모리)

내가 사용하기 위해 맡아 놓은 책상 전체의 공간입니다.

지금 당장 책 한 권만 읽고 있더라도(allocated), 앞으로 다른 책들을 더 펼쳐 놓을 것을 대비해 책상 전체(reserved)를 점유하고 있는 것입니다.
"""


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

torch.cuda.empty_cache()
gc.collect()

torch.cuda.ipc_collect()

print("Memory allocated: ", torch.cuda.memory_allocated())
print("Memory reserved: ",torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

torch.cuda.empty_cache()
gc.collect()

torch.cuda.ipc_collect()

print("Memory allocated: ", torch.cuda.memory_allocated())
print("Memory reserved: ",torch.cuda.memory_reserved())


import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments, Trainer, ModernBertForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset


train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


le = LabelEncoder()
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category + ":" + train.Misconception
train['label'] = le.fit_transform(train['target'])

n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")


idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)


def format_input(row):
    x = 'This answer is correct.'
    if not row['is_correct']:
        x = 'This answer is incorrect.'
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

test['text'] = test.apply(format_input,axis=1)


ds_test = Dataset.from_pandas(test)


model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL",
                                                  device_map="cuda:0",
                                                  torch_dtype=torch.bfloat16)


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL")
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=256)

ds_test = ds_test.map(tokenize, batched=True)


test_args = TrainingArguments(
    output_dir = "./",
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 16,
    bf16 = False,
    fp16 = True,
    report_to = 'none'
)

trainer = Trainer(
    model = model,
    args = test_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

eval_args = TrainingArguments(
    output_dir="./eval_deepseek",
    do_eval=True,
    per_device_eval_batch_size=16,
    fp16=True,
    report_to='none'
)

eval_trainer = Trainer(
    model=model,
    args=eval_args,
    tokenizer=tokenizer,
    compute_metrics=compute_map3
)
results_deepseek = eval_trainer.evaluate(eval_dataset=val_ds)
print("--- DeepSeek Validation Score ---")
print(results_deepseek)

predictions = trainer.predict(ds_test)

predictions.predictions


top3 = np.argsort(-predictions.predictions, axis=1)[:,:]
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels_cat = decoded_labels.reshape(top3.shape)
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

for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

torch.cuda.empty_cache()
gc.collect()



torch.cuda.ipc_collect()

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


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")


le = LabelEncoder()
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train['Category'] + ":" + train['Misconception']
train['label'] = le.fit_transform(train['target'])

n_class = len(le.classes_)
print(f"Train shape: {train.shape} with {n_class} target calsses")


idx = train.apply(lambda row: row.Category.split('_'), axis=1) =='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)


def format_input(row):
    x = "This answer is correct"
    if not row['is_correct']:
        x = "This answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

test['text'] = test.apply(format_input,axis=1)


ds_test = Dataset.from_pandas(test)


model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL",
                                                          device_map="cuda:1", torch_dtype=torch.bfloat16)


model


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=256)

ds_test = ds_test.map(tokenize, batched=True)


test_args = TrainingArguments(
    output_dir = "/.",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=1,
    bf16=False,
    fp16=True,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=test_args,
    tokenizer=tokenizer,
    data_collator = DataCollatorWithPadding(tokenizer)
)

predictions = trainer.predict(ds_test)

predictions.predictions


# 1. 평가를 위한 TrainingArguments 설정
eval_args_qwen = TrainingArguments(
    output_dir="./eval_qwen",
    do_eval=True,
    per_device_eval_batch_size=16,
    fp16=True,
    report_to='none'
)

# 2. 평가용 Trainer 생성
eval_trainer_qwen = Trainer(
    model=model,
    args=eval_args_qwen,
    tokenizer=tokenizer,
    compute_metrics=compute_map3
)

# 3. val_ds로 평가 실행
results_qwen = eval_trainer_qwen.evaluate(eval_dataset=val_ds)
print("--- Qwen Validation Score ---")
print(results_qwen)


top3 = np.argsort(-predictions.predictions, axis=1)[:,:]
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels_cat = decoded_labels.reshape(top3.shape)
top3_labels_cat


joined_preds = []

for preds in top3_labels_cat:
    joined_preds.append("|".join(preds))

sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})

sub.to_csv("submission_qwen.csv", index=False)
sub.head()


sub.iloc[0]


sub.iloc[0]['Category:Misconception']


from collections import defaultdict

def get_top_k_ensemble(l1, l2, l3, k=3):
    list1, list2, list3 = l1.split('|'), l2.split('|'), l3.split('|')
    weights = [0.2, 0.5, 0.3]
    lists = [list1, list2, list3]
    score = defaultdict(int)

    for i, lst in enumerate(lists):
        weight = weights[i]
        for rank, item in enumerate(lst):
            score[item] += (len(lst) - rank) * weight

    sorted_items = sorted(score.items(), key=lambda x: -x[1])
    return ' '.join([item for item, _ in sorted_items[:k]])

list1 = 'a|b|d|f'
list2 = 'b|c|a|e'
list3 = 'c|e|b'

print(get_top_k_ensemble(list1, list2, list3, k=3))


df1 = pd.read_csv('submission_gemma.csv').rename(columns = {'Category:Misconception' : 'Category:Misconception_gemma'})
df2 = pd.read_csv('submission_deepseek.csv').rename(columns = {'Category:Misconception' : 'Category:Misconception_deepseek'})
df3 = pd.read_csv('submission_qwen.csv').rename(columns = {'Category:Misconception' : 'Category:Misconception_qwen'})

df = pd.merge(df1, df2, on = 'row_id', how = 'inner')
df = pd.merge(df, df3, on = 'row_id', how = 'inner')

df['Category:Misconception'] = df.apply(lambda x: get_top_k_ensemble(x['Category:Misconception_gemma'], x['Category:Misconception_deepseek'], x['Category:Misconception_qwen']), axis=1)
df[['row_id', 'Category:Misconception']].to_csv('submission.csv', index = False)
pd.read_csv('submission.csv')

