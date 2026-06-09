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


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import Counter
import itertools

# 内置英文停用词列表
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn"
])

# 查看输入文件路径
print("实际数据集文件路径：")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        full_path = os.path.join(dirname, filename)
        print(full_path)

# 1. 定义文件路径并读取数据
train_zip_path = '/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip'
test_zip_path = '/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip'

print("\n读取数据...")
train = pd.read_csv(train_zip_path, header=0, delimiter="\t", quoting=3)
test = pd.read_csv(test_zip_path, header=0, delimiter="\t", quoting=3)

print(f"训练集规模: {train.shape}")
print(f"测试集规模: {test.shape}")


# 2. 数据清洗函数（保留原逻辑）
def review_to_words(raw_review):
    """将原始评论转换为清洗后的单词序列"""
    # 移除HTML标签
    review_text = BeautifulSoup(raw_review).get_text()
    # 移除非字母字符
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # 转换为小写并分词
    words = letters_only.lower().split()
    # 移除停用词
    meaningful_words = [w for w in words if not w in STOPWORDS]
    # 拼接为字符串返回
    return " ".join(meaningful_words)

# 3. 清洗训练集数据
print("\n清洗训练集评论...")
num_train_reviews = train["review"].size
clean_train_reviews = []

for i in range(num_train_reviews):
    if (i + 1) % 1000 == 0:
        print(f"处理训练集评论 {i + 1}/{num_train_reviews}")
    clean_train_reviews.append(review_to_words(train["review"][i]))

# 词频统计（用于辅助设置词汇表参数）
all_words = list(itertools.chain.from_iterable([review.split() for review in clean_train_reviews]))
word_freq = Counter(all_words)
words_min_frequency = 50

print(f"清洗后的训练集中共有 {len(word_freq)} 个不重复的单词。")
print(f"其中共有 {len([word for word, freq in word_freq.items() if freq > words_min_frequency])} 个出现频率大于等于 {words_min_frequency} 的不重复的单词。")
print("\n出现频率最高的10个单词：")
for word, freq in word_freq.most_common(10):
    print(f"'{word}': {freq}")



# # 4. 文本序列转换（替换原TF-IDF特征）
# print("\n构建词汇表并转换为序列...")
# # 计算合适的序列长度（取评论长度的95分位数）
# review_lengths = [len(review.split()) for review in clean_train_reviews]
# max_len = int(np.percentile(review_lengths, 95))  # 覆盖95%的评论长度
# print(f"序列最大长度设置为: {max_len}")

# # 初始化Tokenizer
# tokenizer = Tokenizer(
#     num_words=7000,  # 保留 top 7000 词汇
#     oov_token="<OOV>"  # 未登录词标记
# )
# tokenizer.fit_on_texts(clean_train_reviews)

# # 转换为整数序列
# train_sequences = tokenizer.texts_to_sequences(clean_train_reviews)
# test_sequences = tokenizer.texts_to_sequences([review_to_words(review) for review in test["review"]])

# # 统一序列长度
# train_padded = pad_sequences(
#     train_sequences,
#     maxlen=max_len,
#     padding='post',
#     truncating='post'
# )
# test_padded = pad_sequences(
#     test_sequences,
#     maxlen=max_len,
#     padding='post',
#     truncating='post'
# )

# vocab_size = len(tokenizer.word_index) + 1  # 词汇表大小（+1 预留0索引）
# 4. 文本序列转换（使用纯PyTorch实现，移除TensorFlow依赖）
print("\n构建词汇表并转换为序列...")
# 计算合适的序列长度
review_lengths = [len(review.split()) for review in clean_train_reviews]
max_len = int(np.percentile(review_lengths, 98))
print(f"序列最大长度设置为: {max_len}")

# 构建词汇表（纯Python实现，不依赖TensorFlow）
word_counts = Counter(all_words)
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
top_words = [word for word, _ in sorted_words[:7000]]  # 取top 7000词汇
word_to_idx = {word: i+1 for i, word in enumerate(top_words)}  # 预留0索引
word_to_idx["<OOV>"] = 0  # 未登录词标记

# 转换为整数序列
def text_to_sequence(text, word_to_idx, max_len):
    words = text.split()
    sequence = [word_to_idx.get(word, word_to_idx["<OOV>"]) for word in words]
    # 截断或填充
    if len(sequence) > max_len:
        return sequence[:max_len]
    else:
        return sequence + [0] * (max_len - len(sequence))

train_sequences = [text_to_sequence(review, word_to_idx, max_len) for review in clean_train_reviews]
test_sequences = [text_to_sequence(review_to_words(review), word_to_idx, max_len) for review in test["review"]]

train_padded = np.array(train_sequences)
test_padded = np.array(test_sequences)

vocab_size = len(word_to_idx) + 1  # 词汇表大小


# -----------------------------------------------------------------------------------------
# ---------------------- 新增：划分训练集/验证集（纯numpy，用你已有变量） ----------------------
# 设定验证集比例（20%），生成随机索引（确保可复现）
np.random.seed(42)  # 固定种子，确保每次划分一致（关键！）
total_train_num = len(train_padded)
val_ratio = 0.2
val_num = int(total_train_num * val_ratio)

# 生成“是否为验证集”的掩码（True=验证集，False=训练集）
val_mask = np.zeros(total_train_num, dtype=bool)
val_mask[np.random.choice(total_train_num, val_num, replace=False)] = True  # 随机选val_num个为True

# 基于掩码划分：确保序列和标签的索引完全同步（核心修复点）
X_train = train_padded[~val_mask]  # ~取反：False的是训练集
y_train = train["sentiment"].values[~val_mask]
X_val = train_padded[val_mask]     # True的是验证集
y_val = train["sentiment"].values[val_mask]

print(f"\n划分后数据规模：")
print(f"训练集：{len(X_train)} 样本（正样本占比：{y_train.mean():.2f}）")
print(f"验证集：{len(X_val)} 样本（正样本占比：{y_val.mean():.2f}）")


# 5. 定义数据集类
class SentimentDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return seq, label
        return seq

# # 构建数据加载器
# batch_size = 64
# train_dataset = SentimentDataset(train_padded, train["sentiment"].values)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = SentimentDataset(test_padded)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# 构建数据加载器（修改：新增验证集加载器，用划分好的X_val/y_val）
batch_size = 64
# 训练集加载器（用修正后的X_train/y_train）
train_dataset = SentimentDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 验证集加载器（用修正后的X_val/y_val）
val_dataset = SentimentDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# 测试集加载器（你原代码不变）
test_dataset = SentimentDataset(test_padded)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n数据加载器就绪：")
print(f"训练集批次：{len(train_loader)} | 验证集批次：{len(val_loader)} | 测试集批次：{len(test_loader)}")



# 加载GloVe预训练词向量
def load_glove_embeddings(glove_path, word_to_idx, embedding_dim):
    """生成嵌入矩阵：未在GloVe中的词用随机初始化"""
    embedding_matrix = np.random.uniform(-0.25, 0.25, (len(word_to_idx) + 1, embedding_dim))  # 随机初始化未登录词
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载GloVe词向量"):
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1:
                continue  # 跳过格式异常的行
            word = parts[0]
            if word in word_to_idx:
                idx = word_to_idx[word]
                embedding_matrix[idx] = np.array(parts[1:], dtype=np.float32)
    return embedding_matrix

# 加载GloVe
glove_path = '/kaggle/input/glove6b100dtxt/glove.6B.100d.txt'
embedding_dim = 100  # 必须与GloVe向量维度一致
embedding_matrix = load_glove_embeddings(glove_path, word_to_idx, embedding_dim)


# 6. 定义LSTM模型
class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, embedding_matrix=None, freeze_embedding=False):
        super(SentimentNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 用预训练向量初始化嵌入层
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = not freeze_embedding  # freeze_embedding=False表示微调
        
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size).zero_().to(device),
            weight.new(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size).zero_().to(device)
        )
        return hidden
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步输出
        out = self.sigmoid(out)
        return out, hidden



# 7. 模型训练
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(f"使用设备: {device}")

# 模型参数
embedding_dim = 100
hidden_dim = 128
num_layers = 1
output_dim = 1

model = SentimentNet(
    vocab_size=len(word_to_idx) + 1,  
    embedding_dim=embedding_dim,
    hidden_dim=128,
    num_layers=1,
    output_dim=1,
    embedding_matrix=embedding_matrix,
    freeze_embedding=False  # 允许微调词向量（适配任务）
).to(device)


optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 早停逻辑
best_val_loss = float('inf')
patience = 5  # 连续3轮验证损失不下降则停止
early_stop_counter = 0


epochs = 30
epoch_losses = []
epoch_train_losses = []  # 记录训练损失（新增，可选）
epoch_val_losses = []    # 记录验证损失（新增

# print("\n开始训练LSTM模型...")
# model.train()
# for epoch in range(epochs):
#     train_loss = 0.0
#     # 进度条显示
#     loop = tqdm(train_loader, total=len(train_loader), leave=True)
#     for batch_idx, (seqs, labels) in enumerate(loop):
#         # 初始化隐藏状态
#         batch_size = seqs.size(0)
#         hidden = model.init_hidden(batch_size)
        
#         seqs, labels = seqs.to(device), labels.to(device)
#         optimizer.zero_grad()
        
#         # 前向传播
#         outputs, hidden = model(seqs, hidden)
#         loss = criterion(outputs.view(-1), labels)
        
#         # 反向传播与优化
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 梯度裁剪
#         optimizer.step()
        
#         # 累计损失
#         train_loss += loss.item() * seqs.size(0)
#         loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
#         loop.set_postfix(loss=loss.item())
    
#     # 计算epoch平均损失
#     epoch_loss = train_loss / len(train_loader.dataset)
#     epoch_losses.append(epoch_loss)
#     print(f"Epoch {epoch+1} 平均损失: {epoch_loss:.6f}")
print("\n开始训练LSTM模型（含早停）...")
for epoch in range(epochs):
    # ---------------------- 训练阶段：强制切换train()模式（核心修正） ----------------------
    model.train()  # 确保训练时dropout生效（必须在每轮训练前调用）
    train_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), leave=True)
    for batch_idx, (seqs, labels) in enumerate(loop):
        batch_size = seqs.size(0)
        hidden = model.init_hidden(batch_size)  # 原代码的init_hidden方法（正确）
        
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs, hidden = model(seqs, hidden)
        loss = criterion(outputs.view(-1), labels)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 原代码的梯度裁剪（正确）
        optimizer.step()
        
        train_loss += loss.item() * seqs.size(0)
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(train_loss=loss.item())
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    epoch_train_losses.append(avg_train_loss)

    # ---------------------- 验证阶段：强制切换eval()模式 ----------------------
    model.eval()  # 确保验证时dropout关闭（必须调用）
    val_loss = 0.0
    val_correct = 0  # 新增：验证准确率，判断模型是否在学习
    with torch.no_grad():
        for seqs, labels in val_loader:
            batch_size = seqs.size(0)
            hidden = model.init_hidden(batch_size)
            seqs, labels = seqs.to(device), labels.to(device)
            
            outputs, hidden = model(seqs, hidden)
            loss = criterion(outputs.view(-1), labels)
            val_loss += loss.item() * seqs.size(0)
            
            # 计算验证准确率（新增：判断模型是否学到东西）
            preds = (outputs.view(-1) > 0.5).float()
            val_correct += (preds == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)  # 验证准确率
    epoch_val_losses.append(avg_val_loss)

    # ---------------------- 早停判断+打印关键信息 ----------------------
    print(f"Epoch {epoch+1} | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f} | 验证准确率: {val_acc:.4f}")
    # 只有验证准确率>0.5，才说明模型在学习（否则需检查数据）
    if val_acc < 0.55:
        print(" 警告：验证准确率过低，可能数据对应错误！")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        # 保存最佳模型（必须：后续预测用这个模型）
        torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
        print(f"  → 保存最佳模型（验证损失: {best_val_loss:.6f}, 验证准确率: {val_acc:.4f}）")
    else:
        early_stop_counter += 1
        print(f"  → 早停计数器: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print(f"\n早停触发！共训练 {epoch+1} 轮")
            break

print(f"\n训练结束！最佳验证准确率: {val_acc:.4f}")


# 8. 模型预测
print("\n生成预测结果...")
model = SentimentNet(
    vocab_size=len(word_to_idx) + 1,
    embedding_dim=100,
    hidden_dim=128,
    num_layers=1,
    output_dim=1,
    embedding_matrix=embedding_matrix,
    freeze_embedding=False
).to(device)
# 加载训练过程中保存的最佳模型权重（核心修正）
model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
model.eval()  # 预测时切换为评估模式

predictions = []
with torch.no_grad():
    for seqs in test_loader:
        batch_size = seqs.size(0)
        hidden = model.init_hidden(batch_size)
        seqs = seqs.to(device)
        
        outputs, hidden = model(seqs, hidden)
        # 原代码的阈值判断（正确，但确保输出维度正确）
        preds = (outputs.view(-1) > 0.5).float().cpu().numpy()
        predictions.extend(preds)


model.eval()
predictions = []

with torch.no_grad():
    for seqs in test_loader:
        batch_size = seqs.size(0)
        hidden = model.init_hidden(batch_size)
        seqs = seqs.to(device)
        
        outputs, hidden = model(seqs, hidden)
        preds = (outputs.view(-1) > 0.5).float().cpu().numpy()  # 0.5为阈值
        predictions.extend(preds)

# 9. 生成提交文件
output = pd.DataFrame(data={"id": test["id"], "sentiment": predictions})
output["sentiment"] = output["sentiment"].astype(int)
output.to_csv("/kaggle/working/submission.csv", index=False, quoting=3)
print("提交文件已生成: /kaggle/working/submission.csv")
print("\n提交文件前5行预览：")
print(output.head())

