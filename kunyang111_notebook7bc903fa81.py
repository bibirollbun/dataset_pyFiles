test_path = '/kaggle/input/jigsaw-agile-community-rules/test.csv'
model_path = '/kaggle/input/kaggle_reddit_cls/transformers/kaggle_reddit_cls/1/kaggle_reddit_cls'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 的日志（包括 XLA 相关）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 可选，指定 GPU


import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 加载测试数据
def load_test_data(csv_file_path):
    """
    加载测试 CSV 文件并处理数据
    """
    prompt = '''## Role
You are Reddit Manager, the subreddit is {subreddit}, your task is to determine whether a comment violates a given rule.
If the comment violates the rule, return 1; otherwise, return 0.
## Rule
{rule}
## Comment
{body}
## Positive violation example
{positive_example_1}
{positive_example_2}
## Negative violation example
{negative_example_1}
{negative_example_2}
'''
    df = pd.read_csv(csv_file_path)
    
    # 提取需要的列
    row_ids = df['row_id'].tolist()
    bodys = df['body'].tolist()
    rules = df['rule'].tolist()
    subreddits = df['subreddit'].tolist()
    pos1 = df['positive_example_1'].tolist()
    pos2 = df['positive_example_2'].tolist()
    neg1 = df['negative_example_1'].tolist()
    neg2 = df['negative_example_2'].tolist()
    
    texts = []
    for i in range(len(df)):
        texts.append(prompt.format(
            subreddit=subreddits[i], 
            rule=rules[i], 
            body=bodys[i], 
            positive_example_1=pos1[i], 
            positive_example_2=pos2[i], 
            negative_example_1=neg1[i], 
            negative_example_2=neg2[i]
        ))
      
    return texts, row_ids

# 2. 创建测试数据集
def create_test_dataset(csv_file_path):
    texts, row_ids = load_test_data(csv_file_path)
    
    # 创建 Dataset 对象
    dataset_dict = {
        'text': texts,
        'row_id': row_ids
    }
    dataset = Dataset.from_dict(dataset_dict)
    return dataset



# 3. 加载模型和分词器
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 添加 pad_token 如果不存在
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding = 'left'
# 4. Tokenize 函数
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors="pt",  # 返回 PyTorch 张量（CPU）
    )

# 5. 加载测试数据
test_dataset = create_test_dataset(test_path)

# 6. Tokenize 测试数据
test_tokenized = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 7. 加载训练好的模型
model = AutoModelForSequenceClassification.from_pretrained(model_path,device_map=device)

# 8. 设置模型为评估模式
model.eval()


from tqdm import tqdm  # 可选：添加进度条

def predict_batch(model, dataset, batch_size=2, device="cuda"):
    """
    批量预测函数（优化版）
    Args:
        model: 已加载的模型（应在GPU上）
        dataset: 已tokenize的数据集（建议提前设置format为torch）
        batch_size: 批大小
        device: 目标设备（默认cuda）
    """
    predictions = []
    row_ids = []
    
    # 确保数据集返回PyTorch张量
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "row_id"])
    
    total_samples = len(dataset)
    
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size), desc="Predicting"):
            # 获取当前批次
            batch = dataset[i : i + batch_size]
            
            # 移动数据到GPU（避免逐样本处理）
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            
            # 模型预测
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 计算概率并保存结果
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # 取标签1的概率
            predictions.extend(probs)
            row_ids.extend(batch["row_id"].tolist())
    
    return row_ids, predictions

# 示例调用
row_ids, predictions = predict_batch(model, test_tokenized, batch_size=2, device=device)
# 11. 保存结果到CSV文件
results_df = pd.DataFrame({
    'row_id': row_ids,
    'rule_violation': predictions
})



results_df.to_csv("submission.csv",index=False)

