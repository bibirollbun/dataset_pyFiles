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


test = pd.read_csv(r"/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
train = pd.read_csv(r"/kaggle/input/map-charting-student-math-misunderstandings/train.csv")


no_nan = train['Misconception'].notna()
print('Misconception column has NaN values:', len(no_nan))


train['Misconception'] = np.where(train['Misconception'].isna(),'no','yes')


train = train.drop(columns=['QuestionId','row_id'], axis=1)
test= test.drop(columns=['QuestionId'], axis=1)


train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)


from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

# Convert pandas DataFrame to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Split the train dataset
train_val_split = train_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

# Create DatasetDict
raw_datasets = DatasetDict({
    'train': train_val_split['train'],
    'validation': train_val_split['test']
})
print(raw_datasets)


import torch


from transformers import AutoModelForSequenceClassification

dataset_path = "/kaggle/input/bert-base-uncased-offline2/bert-base-uncased_offline"
num_labels = 2

# 기존 모델 불러오기 (원래 num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    dataset_path,
    num_labels=num_labels  # 기존 체크포인트와 맞춤
)

# classifier만 새로 생성
import torch.nn as nn
model.classifier = nn.Linear(model.config.hidden_size, num_labels)
model.config.num_labels = num_labels
tokenizer = AutoTokenizer.from_pretrained(
    dataset_path,
    use_fast=True,          # fast tokenizer 사용
    local_files_only=True   # 인터넷 접속 안 하고 로컬 파일만 사용
)
model.gradient_checkpointing_enable()
print(model)



def tokenize_fucntion(math_problem):
    inputs = [q+tokenizer.sep_token +s+tokenizer.sep_token+mc_a for q,s,mc_a in zip(math_problem['QuestionText'],math_problem['StudentExplanation'],math_problem['Misconception'])]
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding='max_length',
        max_length=160,
    )
    all_categories = ['True_Correct', 'False_Neither', 'False_Misconception', 'False_Correct']
    all_misconceptions = ['NA', 'Incomplete', 'Inaccurate']
    combined_raw_labels = all_categories + all_misconceptions
    unique_labels_list = sorted(list(set(combined_raw_labels)))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels_list)}
    print(f"모델의 num_labels는 {len(unique_labels_list)}로 설정되어야 합니다.")
    num_labels = len(unique_labels_list)
    sample_labels_list = []

    for i in range(len(math_problem['QuestionText'])):
        current_category = math_problem['Category'][i]
        current_misconception = math_problem['Misconception'][i]
        one_hot_vector = [0] * num_labels
        if current_category in label_to_id:
            one_hot_vector[label_to_id[current_category]] = 1
        if current_misconception in label_to_id:
            one_hot_vector[label_to_id[current_misconception]] = 1

        sample_labels_list.append(one_hot_vector)
    tokenized['labels'] = torch.tensor(sample_labels_list, dtype=torch.float)
    return tokenized

tokenized_datasets = raw_datasets.map(
    tokenize_fucntion,
    batched=True
)
print('토큰화 적용\n',tokenized_datasets)

remove_columns = ['QuestionText', 'StudentExplanation', 'Misconception','MC_Answer','Category']
for split in tokenized_datasets.keys():
    c_colums = tokenized_datasets[split].column_names
    cols = [col for col in remove_columns if col in c_colums]
    tokenized_datasets[split] = tokenized_datasets[split].remove_columns(cols)
    tokenized_datasets[split].set_format("torch")



print('토큰화 적용 후 컬럼 제거\n',tokenized_datasets)
print('토큰화 적용 후 컬럼 제거 후 데이터셋 형식 변경\n',tokenized_datasets)


test_dict = DatasetDict({
    'test': test_dataset
})


# 이미 model 변수에 로드돼 있다고 가정
print("Model classifier:", model.classifier)
print("num_labels:", model.config.num_labels)



import torch
import torch.nn as nn

# 이미 model 변수가 메모리에 로드되어 있다고 가정
num_labels_new = 7

# hidden size 가져오기
hidden_size = model.config.hidden_size  # 보통 768

# 1) classifier 새로 생성 (초기화)
model.classifier = nn.Linear(hidden_size, num_labels_new)
model.config.num_labels = num_labels_new

# (선택) 가중치 초기화: Xavier / bias 0
nn.init.xavier_normal_(model.classifier.weight)
if model.classifier.bias is not None:
    nn.init.zeros_(model.classifier.bias)

print("교체된 classifier:", model.classifier)
print("model.config.num_labels:", model.config.num_labels)



def test_tokenize_fucntion(math_problem):
    inputs = [q+tokenizer.sep_token +s+tokenizer.sep_token+mc_a+str(z)+tokenizer.sep_token for q,s,mc_a,z in zip(math_problem['QuestionText'],math_problem['StudentExplanation'],math_problem['MC_Answer'],math_problem['row_id'])]
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding='max_length',
        max_length=300,
    )
    return tokenized


test_tokenized_datasets = test_dict.map(
    test_tokenize_fucntion,
    batched=True
)
print('토큰화 적용\n',test_tokenized_datasets)

remove_columns = ['QuestionText', 'StudentExplanation', 'MC_Answer']
for split in test_dict.keys():
    c_colums = test_tokenized_datasets[split].column_names
    cols = [col for col in remove_columns if col in c_colums]
    test_tokenized_datasets[split] = test_tokenized_datasets[split].remove_columns(cols)
    test_tokenized_datasets[split].set_format("torch")

print('토큰화 적용 후 컬럼 제거\n',test_tokenized_datasets)
print('토큰화 적용 후 컬럼 제거 후 테스트용 데이터셋 형식 변경\n',test_tokenized_datasets)


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("데이터 콜레이터")


from transformers import TrainingArguments
output_dir = './results'
num_train_epochs = 10
per_device_train_batch_size = 8
per_device_train_eval_batch_size = 8
learning_rate = 2e-5
save_steps = 5000
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_eval_batch_size,
    learning_rate=learning_rate,
    save_steps=save_steps,
    save_total_limit=2,
    eval_strategy="epoch",
    eval_steps=5000,
    logging_dir='./logs',
    logging_steps=100,
    report_to="none",
    fp16=True,
    load_best_at_end = True,
    metric_for_best_model = "Validation_loss",
    greater_is_better = False
    # Disable reporting to avoid errors in this environment
)
print("훈련 설정 끝")


from transformers import Trainer
trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['validation'],
    data_collator=data_collator,
    callbacks = [EarlyStoppingCallback(early_topping_patience=2)]
)
print("모델 훈련 시작 ")
trainer.train()
print("/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n/n")
print("종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료")
print("종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료")
print("종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료")
print("종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료")
print("종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료")
print("종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료")
print("종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료종료")


pred = trainer.predict(test_tokenized_datasets['test'])


index_to_label_mapping = []
all_categories = ['True_Correct', 'False_Neither', 'False_Misconception', 'False_Correct']
all_misconceptions = ['NA', 'Incomplete', 'Inaccurate']
for categories in all_categories:
    for misconceptions in all_misconceptions:
        index_to_label_mapping.append(categories + ':' + misconceptions)
print(index_to_label_mapping)


logits = pred.predictions
all = []
for r in logits:
    top_3 = np.argsort(r)[-3:][::-1]
    top_3 = set(top_3)

    predicted_labels = [index_to_label_mapping[i] for i in top_3]
    all.append(' '.join(predicted_labels))

row_ids = test_tokenized_datasets['test']['row_id']
submission_df = pd.DataFrame({
    'row_id': row_ids,
    'Category:Misconception': all
})
submission_df.to_csv('submission.csv', index=False)


duplicated = []
count = 0
for index,row in submission_df.iterrows():
    la = row['Category:Misconception'].split(':')
    if len(la) != len(set(la)):
        count +=1

print(count)


logits = pred.predictions
all_predicted_labels = []

for r in logits:
    # 상위 5개 예측의 인덱스를 가져와 후보군을 확보합니다.
    top_5_indices = np.argsort(r)[-5:][::-1]
    
    # 인덱스를 사용하여 라벨로 변환합니다.
    predicted_labels = [index_to_label_mapping[i] for i in top_5_indices]
    
    # 카테고리별 중복 제거
    seen_categories = set()
    unique_labels_by_category = []
    
    for label in predicted_labels:
        category = label.split(':')[0]
        if category not in seen_categories:
            unique_labels_by_category.append(label)
            seen_categories.add(category)
            
    # 최종 결과로 상위 3개 라벨만 선택합니다.
    final_labels = unique_labels_by_category[:3]
    
    # 유일한 라벨들을 공백으로 구분된 문자열로 만듭니다.
    all_predicted_labels.append(' '.join(final_labels))

row_ids = test_tokenized_datasets['test']['row_id']
submission_df = pd.DataFrame({
    'row_id': row_ids,
    'Category:Misconception': all_predicted_labels
})
submission_df.to_csv('submission.csv', index=False)


submission_df.head()





































