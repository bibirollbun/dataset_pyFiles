import json
import csv

def process_json_file(input_file, output_prefix):
    """
    从JSON文件读取数据，按id排序后生成两个CSV文件
    参数:
        input_file (str): JSON文件路径
        output_prefix (str): 输出文件前缀
    """
    # 读取JSON文件并排序
    with open(input_file, 'r', encoding='utf-8') as f:
        records = sorted([json.loads(line) for line in f], key=lambda x: x['id'])
    out_res = []
    # 生成句子CSV
    with open(f'{output_prefix}_sentences.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'sentence', 'label'])
        writer.writeheader()
        for r in records:
            out_res.append({
                'id': r['id'],
                'sentence': r['sentence'],
                'label':r['label']
            })
    return out_res

train_json = '/kaggle/input/lesson-2-text-classify/train_few_all.json'  # 替换为实际JSON文件路径
val_json = '/kaggle/input/lesson-2-text-classify/dev_few_all.json'  # 替换为实际JSON文件路
train_json = process_json_file(train_json, "")
train_json[0:2]


import pandas as pd 
import jieba
from collections import Counter

alllabels = {'100':0,'101':1,'102':2,'103':3,'104':4,'106':5,'107':6,'108':7,'109':8,'110':9,'112':10,'113':11,'114':12,'115':13,'116':14}
lastconvert = {}
for key in alllabels.keys():
    lastconvert[str(alllabels[key])] = key
    
labels = [alllabels[str(label['label'])] for label in train_json]
texts = [text['sentence'] for text in train_json]
# 构建词汇表
words = []
for text in texts:
    words.extend(jieba.cut(text))
word_counts = Counter(words)
vocab = ["<PAD>", "<UNK>"] + [word for word, count in word_counts.items()]
word2idx = {word: idx for idx, word in enumerate(vocab)}
words[:20]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 1. 数据准备
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=50):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词并转换为索引序列
        words = list(jieba.cut(text))[:self.max_len]
        seq = [self.word2idx.get(word, 1) for word in words]  # 1表示UNK
        seq = seq + [0] * (self.max_len - len(seq))  # 填充
        
        return torch.LongTensor(seq), torch.LongTensor([label])
# 数据加载
dataset = TextDataset(texts, labels, word2idx)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True,drop_last=True)
print("dataset[0][0] shape = {} | dataset[0][1] shape = {}".format(dataset[0][0].shape,dataset[0][1].shape))
dataset[0]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 2. 优化后的模型定义
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x, (h_n, _) = self.lstm(x)  # 获取所有时间步的输出
        x = self.dropout(x)
        out = self.fc(x.mean(dim=1))  # 对时间步取平均
        return out


# 3. 辅助函数
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy

def save_model(model, word2idx, config, path='best_model.pth'):
    checkpoint = {
        'model_state': model.state_dict(),
        'word2idx': word2idx,
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"模型已保存到 {path}")

# 模型初始化
config = {
    'vocab_size': len(vocab),
    'embed_dim': 128,
    'hidden_dim': 64,
    'num_classes': len(alllabels)
}
model = TextClassifier(**config)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch学习率乘0.1

# 训练循环
best_val_acc = 0
for epoch in range(50):
    # 训练阶段
    model.train()
    total_loss = 0
    total_acc = 0
    for seq, label in dataloader:
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, label.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += calculate_accuracy(output, label.squeeze())
    
    avg_train_loss = total_loss / len(dataloader)
    avg_train_acc = total_acc / len(dataloader)
    
    print("Epoch epoch+1 = {}, Train Loss: {}, Train Acc: {}".format(epoch+1,avg_train_loss,avg_train_acc))
    
    # 更新学习率
    scheduler.step()
    save_model(model, word2idx, config)



import pandas as pd 
# 示例数据
predict_df = pd.read_csv('/kaggle/input/lesson-2-text-classify/test_idx.csv',encoding='gbk')
predict_df = pd.DataFrame(predict_df)
predict_texts = [text for text in predict_df['sentence']]
predict_ids = [id for id in predict_df['id']]

res = []
with torch.no_grad():
    idx  = 0
    for test_id,test_text in zip(predict_ids,predict_texts):
        test_words = list(jieba.cut(test_text))[:50]
        test_seq = [word2idx.get(word, 1) for word in test_words]
        test_seq = test_seq + [0] * (50 - len(test_seq))

    
        output = model(torch.LongTensor([test_seq]))
        pred = torch.argmax(output).item()
        res.append({'id': test_id, 'label': lastconvert[str(pred)]})
        idx += 1
    print(f"测试文本: '{test_text}'，预测结果: {lastconvert[str(pred)]}")
# 保存到CSV
outdf = pd.DataFrame(res)
outdf.to_csv("out_submit.csv", index=False)
print(res[:10])




