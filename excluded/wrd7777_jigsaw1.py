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


!pip install transformers
!pip install scikit-learn


import os
import re
import pandas as pd

# 配置参数
class Config:
    def __init__(self):
        self.raw_test_path = "/kaggle/input/jigsaw-agile-community-rules/test.csv"    # 原始测试数据路径
        self.processed_dir = "./processed_data"       # 处理后数据保存目录

config = Config()
os.makedirs(config.processed_dir, exist_ok=True)


# 1. 数据清洗（适配英文文本）
def clean_data(df, is_test=False):
    # 去重：基于评论+社区+规则的唯一组合
    df = df.drop_duplicates(subset=['body', 'subreddit', 'rule'], keep='first')
    
    # 清洗英文文本（保留常见标点、缩写）
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).strip()
        # 保留：字母、数字、常见标点（.!?,'\"-）和空格
        text = re.sub(r'[^\w\s.!?,\'\"-]', '', text)
        # 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # 对所有文本列进行清洗
    text_columns = ['body', 'subreddit', 'rule', 
                   'positive_example_1', 'positive_example_2',
                   'negative_example_1', 'negative_example_2']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    return df


# 2. 数据处理主函数（仅处理测试集）
def main():
    # 处理外部测试数据（用户提供的测试集）
    print("=== 处理测试数据 ===")
    raw_test = pd.read_csv(config.raw_test_path)
    print(f"原始测试数据量: {len(raw_test)}")
    
    cleaned_test = clean_data(raw_test, is_test=True)
    print(f"清洗后测试数据量: {len(cleaned_test)}")
    
    # 保存处理后的数据
    cleaned_test.to_csv(os.path.join(config.processed_dir, "test.csv"), index=False)
    print(f"\n处理后的数据已保存至: {config.processed_dir}")
    
    # 输出部分数据样例验证处理效果
    print("\n=== 处理后测试集样例（前3行） ===")
    print(cleaned_test[['body', 'subreddit', 'rule']].head(3))


if __name__ == "__main__":
    main()


import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm

# 配置参数
class Config:
    def __init__(self):
        self.processed_dir = "./processed_data"       # 处理后的数据目录
        self.model_dir = "./saved_model"              # 模型保存目录
        self.local_model_path = "/kaggle/input/roberta-base"  # 本地roberta模型文件路径
        self.max_len = 512                            # 文本最大长度
        self.batch_size = 16                           # 推理批次大小
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()


# 1. 模型定义（与训练时一致）
class CustomRobertaModel(torch.nn.Module):
    def __init__(self, model_path, dropout_prob=0.3):
        super(CustomRobertaModel, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=1,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob
        )
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = self.dropout(outputs.logits)
        return logits


# 2. 推理数据集定义
class InferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ids = self.data['row_id']  # 假设数据中存在'id'列作为唯一标识

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # 构建输入文本（与训练时一致的模板）
        text = f"【Subreddit】{item['subreddit']}\n"
        text += f"【Rule】{item['rule']}\n"
        text += f"【Comment】{item['body']}\n"
        positive_examples = self._combine_examples([item['positive_example_1'], item['positive_example_2']])
        negative_examples = self._combine_examples([item['negative_example_1'], item['negative_example_2']])
        text += f"【References】Compliant examples: {positive_examples}; Violation examples: {negative_examples}"
        
        # 编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'row_id': self.ids.iloc[idx]  # 新增id返回
        }
    
    def _combine_examples(self, examples):
        filtered = [str(ex).strip() for ex in examples if pd.notna(ex) and str(ex).strip()]
        if not filtered:
            return "None"
        truncated = [ex[:30] + "..." if len(ex) > 30 else ex for ex in filtered]
        return "; ".join(truncated)


# 3. 推理主函数
def main():
    # 加载测试数据
    test_data_path = os.path.join(config.processed_dir, "test.csv")
    test_df = pd.read_csv(test_data_path)
    print(f"Loaded test data (size: {len(test_df)})")
    
    # 加载本地tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.local_model_path)
    
    # 创建推理数据集
    dataset = InferenceDataset(test_data_path, tokenizer, config.max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 加载模型（使用本地模型路径）
    model = CustomRobertaModel(config.local_model_path)
    model.load_state_dict(torch.load(os.path.join(config.model_dir, "/kaggle/input/final-model/final_model.pth"), map_location=config.device))
    model.to(config.device)
    model.eval()
    print(f"Model loaded from local path, using device: {config.device}")
    
    # 推理
    all_ids = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            ids = batch['row_id']  # 读取批次内的id
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # 处理维度，确保可迭代
            if probs.ndim == 0:
                probs = [probs]
            all_probs.extend(probs)
            # 将张量id转换为Python原生类型
            all_ids.extend(ids.tolist())
    
    # 构建结果DataFrame（仅保留id和预测分数）
    result_df = pd.DataFrame({
        'row_id': all_ids,
        'rule_violation': all_probs
    })
    
    # 直接保存到当前目录的submission.csv
    result_df.to_csv("submission.csv", index=False)
    print("Predictions saved to: submission.csv")


if __name__ == "__main__":
    main()

