!pip install bitsandbytes


import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


!nvidia-smi


#수정불가
train_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv')
valid_df = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/train.csv')
test_df = valid_df.copy()
train_df


train_df.isnull().sum()


NUM_TRAIN_SAMPLES = 10000 # <= 10000
train_df = train_df.fillna('').sample(NUM_TRAIN_SAMPLES, random_state=42)


from datasets import Dataset, DatasetDict, load_dataset

MAX_INPUT = 384

MODEL = "/kaggle/input/llama-3.2/transformers/3b/1"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.cls_token is None:
    tokenizer.cls_token = tokenizer.bos_token or tokenizer.eos_token

tokenizer.padding_side = "right"


texts = (train_df['prompt'] + '\n###\nA: ' + train_df['A'] + '\n###\nB: ' + train_df['B']+ '\n###\nC: ' + train_df['C']+ '\n###\nD: ' + train_df['D']+ '\n###\nE: ' + train_df['E']+ '\n###\nAnswer: ').tolist()
tokenized_texts = [tokenizer.tokenize(text) for text in texts]
token_lengths = [len(tokens) for tokens in tokenized_texts]

def custom_tokenize(tokenizer, text1, text2, max_length):
    #전체 토큰 길이 max_length을 넘지 않도록 trucnate
    #text2의 길이가 max_length을 초과하면, text2의 앞부분을 truncate, 이후 text1의 뒷부분을 truncate

    tokenizer.truncation_side='left'
    text2_encoded = tokenizer.encode(
        text2,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )

    text2_len = len(text2_encoded)
    
    if text2_len < max_length:
        tokenizer.truncation_side='right'
        text1_encoded = tokenizer.encode(
            text1,
            truncation=True,
            max_length=max_length-text2_len,
            add_special_tokens=False
        )
    else:
        text1_encoded = []

    input_ids = text1_encoded + text2_encoded
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def preprocess_function_classification(examples, tokenizer=tokenizer):
    input_ids = []
    labels = []
    attention_masks = []
    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    for q, A, B, C, D, E, answer in zip(examples['prompt'], examples['A'], examples['B'], examples['C'], examples['D'], examples['E'], examples['answer']):
        text1 = tokenizer.cls_token + q #q
        text2 = '\n###\nA: ' + A + '\n###\nB: ' + B + '\n###\nC: ' + C + '\n###\nD: ' + D + '\n###\nE: ' + E + tokenizer.eos_token
        # tokenized = tokenizer(text1, text2, max_length=MAX_LEN, truncation='only_first')
        tokenized = custom_tokenize(tokenizer, text1, text2, max_length=MAX_INPUT)
        input_ids.append(tokenized['input_ids'])
        labels.append(label_mapping[answer])
        attention_masks.append(tokenized['attention_mask'])
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_masks}

train_dataset = Dataset.from_pandas(train_df)
if "__index_level_0__" in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns(["__index_level_0__"])
valid_dataset = Dataset.from_pandas(valid_df)
if "__index_level_0__" in valid_dataset.column_names:
    valid_dataset = valid_dataset.remove_columns(["__index_level_0__"])
dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})

tokenized_dataset = dataset.map(preprocess_function_classification, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])



from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

choice_labels = ["A", "B", "C", "D", "E"]
label2id = {c: i for i, c in enumerate(choice_labels)}
id2label = {i: c for c, i in label2id.items()}  # i: index, c: label

set_seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

num_labels = len(choice_labels)

# 4bit quant 설정 (bitsandbytes)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Qwen base를 4bit로 로드 + 분류 헤드 붙이기
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    quantization_config=quant_config,
    device_map={"": 0},  # 단일 GPU
)

# pad_token 없으면 설정
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# LoRA 학습 준비 (모델 일부만 trainable로)
model = prepare_model_for_kbit_training(model)

# LoRA 설정
lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["score"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir="./llama3-3b",
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    warmup_ratio=0.05,
    learning_rate=1e-4,
    optim='paged_adamw_8bit', #'paged_adamw_32bit', 'paged_adamw_8bit', 'adamw_torch'
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.05,
    save_total_limit=1,
    report_to="none",
    fp16=True,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=8,
    load_best_model_at_end=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest',)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)


# trainer.train()


# save_path = "./saved-llama3-3b-lora"
# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)


# print(trainer.evaluate())


from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

load_path = "/kaggle/input/a-z-llm-fine-tuning-output/saved-llama3-3b-lora"

tokenizer = AutoTokenizer.from_pretrained(load_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=5,
    quantization_config=quant_config,
    device_map="auto",
)
if base_model.config.pad_token_id is None:
    base_model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(base_model, load_path)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


from datasets import load_dataset
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

test_df = test_df.sample(100, random_state=13).reset_index(drop=True)

test_dataset = Dataset.from_pandas(test_df)
if "__index_level_0__" in test_dataset.column_names:
     test_dataset = test_dataset.remove_columns(["__index_level_0__"])
    
tokenized_test = test_dataset.map(preprocess_function_classification, batched=True, remove_columns=test_dataset.column_names,)

test_dataloader = DataLoader(
    tokenized_test,
    batch_size=4,
    shuffle=False,
    collate_fn=data_collator,
)
#TODO: dataloader 정의


model.eval()
test_predictions = []
for batch in test_dataloader:
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    with torch.no_grad():
        outputs = model(**batch)
    test_predictions.append(outputs['logits'].cpu().detach())

test_predictions = torch.cat(test_predictions)
test_predictions = test_predictions.float().numpy()
predictions_as_ids = np.argsort(-test_predictions, 1)
predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
predictions_as_string = test_df['prediction'] = [
    ' '.join(row) for row in predictions_as_answer_letters[:, :3]
]


#수정불가
#https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
import numpy as np
def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U


#수정불가
m = MAP_at_3(test_df.prediction.values, test_df.answer.values)
print( 'CV MAP@3 =',m )




