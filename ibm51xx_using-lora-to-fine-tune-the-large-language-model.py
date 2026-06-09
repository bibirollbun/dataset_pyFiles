!pip install transformers datasets evaluate accelerate peft torch


from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np
import torch
import pandas as pd


train_csv_path='/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'
train_file_path='/kaggle/input/fake-or-real-the-impostor-hunt/data/train'
test_file_path='/kaggle/input/fake-or-real-the-impostor-hunt/data/test'


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


import os
df=pd.read_csv(train_csv_path)
true_texts=[]
false_texts=[]
#转换为列表
train_texts = df['real_text_id'].tolist()
#print(train_texts)
article_list=sorted(os.listdir(train_file_path))
#print(article_list)
for i,article in enumerate(article_list):
    file_dir_path=os.path.join(train_file_path, article)
    files= sorted(os.listdir(file_dir_path))
    text1= read_file(os.path.join(file_dir_path, files[0]))
    text2= read_file(os.path.join(file_dir_path, files[1]))
    if train_texts[i] == 1:
        true_texts.append(text1)
        false_texts.append(text2)
    else:
        true_texts.append(text2)
        false_texts.append(text1)


from sklearn.model_selection import train_test_split
X= true_texts + false_texts
y= [1] * len(true_texts) + [0] * len(false_texts)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y)


from datasets import Dataset
# 3. 加载并预处理数据集（示例使用假新闻数据集，需替换为实际数据）
# 真实数据标注为1，虚假数据标注为0
def load_data():
    # 示例数据集结构（替换为真实路径）
    # 数据集格式：[{"text": "内容", "label": 0/1}]
    dataset = DatasetDict({
        'train': Dataset.from_dict({
            'text': X_train,
            'label': y_train  # 1=真实, 0=虚假
        }),
        'validation': Dataset.from_dict({
            'text': X_val,
            'label': y_val
        })
    })
    return dataset
dataset = load_data()


model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label={0: "Fake", 1: "Real"}
)



def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
tokenized_data = dataset.map(preprocess_function, batched=True)


lora_config = LoraConfig(
    r=4,                  # LoRA秩（矩阵维度）
    lora_alpha=32,         # 缩放因子
    target_modules=["q_lin", "v_lin"],  # 目标模块（DistilBERT的注意力层）
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",   # 序列分类任务
    inference_mode=False
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 显示可训练参数占比（应<1%）


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,    # 微调推荐学习率[7](@ref)
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,    # 推荐2-4个epoch[7](@ref)
    eval_strategy="epoch",  # 每个epoch结束后评估
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# 9. 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

# 10. 开始训练
trainer.train()

# 11. 保存模型
model.save_pretrained("lora_fake_news_detector")




test_file_list=sorted(os.listdir(test_file_path))
test_texts=[]
for i, article in enumerate(test_file_list):
    texts=[]
    file_dir_path=os.path.join(test_file_path, article)
    files= sorted(os.listdir(file_dir_path))
    texts.append(read_file(os.path.join(file_dir_path, files[0])))
    texts.append(read_file(os.path.join(file_dir_path, files[1])))
    test_texts.append(texts)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    return "Real" if prediction == 1 else "Fake", probabilities[0].tolist()


res=pd.DataFrame(columns=['id','read_text_id'])
for i, texts in enumerate(test_texts):
    text1, prob1 = predict(texts[0])
    text2, prob2 = predict(texts[1])
    if prob1[1] > prob2[1]:
        res.loc[i] = [i, 1]
    else:
        res.loc[i] = [i, 2]
#保存为csv文件
res.to_csv('submission.csv', index=False)

