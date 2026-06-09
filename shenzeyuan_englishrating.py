# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. 数据预处理与自定义 Dataset
# -----------------------------
class EssayDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=256):
        """
        texts: 文本列表
        targets: 对应的6个评分指标，shape=(样本数, 6)
        tokenizer: 分词器对象
        max_length: 句子最大长度（超过部分将被截断，不足则padding）
        """
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # 使用 tokenizer 进行编码，同时添加 [CLS] 和 [SEP] 标记
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        # 去除 batch 维度
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        # 如果有目标值，则返回
        if self.targets is not None:
            item['targets'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item



class BERTRegressionModel(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        """
        model_name: 预训练模型名称，如 'bert-base-uncased'
        dropout: dropout 概率
        """
        super(BERTRegressionModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 获取BERT的隐藏层大小
        self.dropout = nn.Dropout(dropout)
        # 全连接层将 BERT 的 [CLS] 向量映射到6个连续输出
        self.regressor = nn.Linear(hidden_size, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # outputs[1] 为 pooler_output，即 [CLS] 对应的向量
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        out = self.regressor(pooled_output)
        return out


def mcrmse(y_true, y_pred):
    """
    计算每个目标列的 RMSE，再取平均
    y_true, y_pred: torch.tensor, shape=(batch_size, 6)
    """
    mse = ((y_true - y_pred) ** 2).mean(dim=0)
    rmse = torch.sqrt(mse)
    return rmse.mean()


def train_epoch(model, data_loader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
    return running_loss / len(data_loader)

def eval_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            all_preds.append(outputs)
            all_targets.append(targets)
    # 拼接所有 batch 的预测结果和真实值
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    score = mcrmse(all_targets, all_preds)
    return running_loss / len(data_loader), score.item()


def create_submission(model, tokenizer, device, max_length, batch_size, test_file='test.csv', submission_file='/kaggle/working/submission.csv'):
    """
    对测试集进行预测，并生成提交文件 submission.csv
    参数:
        model: 训练好的模型
        tokenizer: 对应的分词器
        device: 运行设备（CPU/GPU）
        max_length: 文本最大长度
        batch_size: DataLoader 批次大小
        test_file: 测试集 CSV 文件名（应包含 text_id, full_text 字段）
        submission_file: 生成的提交文件名
    """
    # 读取测试集
    test_df = pd.read_csv(test_file)
    test_texts = test_df['full_text'].astype(str).tolist()
    text_ids = test_df['text_id'].tolist()
    
    # 创建测试集 Dataset（targets 传 None）
    test_dataset = EssayDataset(test_texts, None, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            predictions.append(outputs.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    
    # 生成提交文件 DataFrame，确保列顺序符合要求
    submission_df = pd.DataFrame(predictions, columns=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'])
    submission_df.insert(0, 'text_id', text_ids)
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")


import os

model_dir = "/kaggle/input/bert_base_uncased/pytorch/default/1"
required_files = ["vocab.txt", "tokenizer_config.json", "config.json", "pytorch_model.bin"]

for file in required_files:
    if not os.path.exists(os.path.join(model_dir, file)):
        print(f"❌ 缺失文件: {file}")
    else:
        print(f"✅ 找到文件: {file}")


ls


def main():
    # 超参数设置
    MODEL_NAME = r'/kaggle/input/bert_base_uncased/pytorch/default/1'
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 2e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    
    # 读取训练数据（请确保当前目录下存在 train.csv 文件）
    df = pd.read_csv('/kaggle/input/feedback-prize-english-language-learning/train.csv')
    # 数据集包含的列：text_id, full_text, cohesion, syntax, vocabulary, phraseology, grammar, conventions
    texts = df['full_text'].astype(str).tolist()
    targets = df[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].values
    
    # 划分训练集和验证集（这里采用80%训练，20%验证）
    train_texts, val_texts, train_targets, val_targets = train_test_split(
        texts, targets, test_size=0.2, random_state=42
    )
    
    # 加载预训练模型对应的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, use_fast=False)
    
    # 创建自定义数据集
    train_dataset = EssayDataset(train_texts, train_targets, tokenizer, max_length=MAX_LENGTH)
    val_dataset = EssayDataset(val_texts, val_targets, tokenizer, max_length=MAX_LENGTH)
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 初始化模型并送入设备
    model = BERTRegressionModel(MODEL_NAME, dropout=0.5)
    model.to(device)
    
    # 定义优化器、学习率调度器及损失函数
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.MSELoss()
    
    best_score = float('inf')
    best_epoch = 0

    early_stopping_patience = 5  # 设定连续多少轮无提升后停止
    patience_counter = 0
    
    # 开始训练
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_score = eval_epoch(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MCRMSE: {val_score:.4f}")
        
        # 保存最佳模型
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            print("保存了当前最佳模型！")
            patience_counter = 0
        else:
            patience_counter+=1
            if patience_counter >= early_stopping_patience:
                print(f"连续 {early_stopping_patience} 轮无提升，提前终止训练。")
                break  # 终止训练

    model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))

    create_submission(model, tokenizer, device, MAX_LENGTH, BATCH_SIZE,
                      test_file='/kaggle/input/feedback-prize-english-language-learning/test.csv')
    
    print(f"\n训练结束，最佳验证 MCRMSE: {best_score:.4f} (第 {best_epoch} 轮)")
    
if __name__ == '__main__':
    main()





