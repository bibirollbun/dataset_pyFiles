import os
path = "/kaggle/input/notebookc09459c289/bert-base-uncased"
print(f"路径存在: {os.path.exists(path)}")
print(f"目录内容: {os.listdir(path) if os.path.exists(path) else '路径无效'}")


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 加载数据
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# 初始化 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained("/kaggle/input/notebookc09459c289/bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("/kaggle/input/notebookc09459c289/bert-base-uncased", num_labels=1)

# 自定义 Dataset
class CommentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_df["body"], train_df["rule_violation"], test_size=0.2)
train_dataset = CommentDataset(X_train.tolist(), y_train.tolist())
val_dataset = CommentDataset(X_val.tolist(), y_val.tolist())

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
print("ok")


# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

    # 验证（计算 AUC）
    model.eval()
    val_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            
            # 收集真实标签和预测概率
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs.logits).cpu().numpy())

    # 计算 AUC
    val_auc = roc_auc_score(all_labels, all_preds)
    print(f"Epoch {epoch}, Val Loss: {val_loss / len(val_loader)}, Val AUC: {val_auc:.4f}")

# 预测测试集（保持不变）
test_dataset = CommentDataset(test_df["body"].tolist(), [0] * len(test_df))
test_loader = DataLoader(test_dataset, batch_size=32)
model.eval()
print("ok")


predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions.extend(torch.sigmoid(outputs.logits).cpu().numpy().flatten())  # 确保是 1D 列表

print("ok")


# 保存提交文件
print(predictions)
submission = pd.DataFrame({"row_id": test_df["row_id"], "rule_violation": predictions})  # 注意列名是 row_id 还是 id
submission.to_csv("submission.csv", index=False)
submission.head()


# # 训练模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# optimizer = AdamW(model.parameters(), lr=2e-5)  # 使用 torch.optim.AdamW
# criterion = torch.nn.BCEWithLogitsLoss()

# for epoch in range(3):
#     model.train()
#     for batch in train_loader:
#         optimizer.zero_grad()
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["label"].to(device).unsqueeze(1)
#         outputs = model(input_ids, attention_mask=attention_mask)
#         loss = criterion(outputs.logits, labels)
#         loss.backward()
#         optimizer.step()

#     # 验证
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["label"].to(device).unsqueeze(1)
#             outputs = model(input_ids, attention_mask=attention_mask)
#             val_loss += criterion(outputs.logits, labels).item()
#     print(f"Epoch {epoch}, Val Loss: {val_loss / len(val_loader)}")

# # 预测测试集
# test_dataset = CommentDataset(test_df["body"].tolist(), [0] * len(test_df))
# test_loader = DataLoader(test_dataset, batch_size=32)
# model.eval()


# predictions = []
# with torch.no_grad():
#     for batch in test_loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask)
#         #predictions.extend(torch.sigmoid(outputs.logits).cpu().numpy())
#         predictions = torch.sigmoid(outputs.logits).cpu().numpy()  # 形状可能是 (N, 1)

# print("ok")


# import numpy as np
# # 确保 predictions 是 1D 数组
# predictions = np.array(predictions).flatten().tolist()  # 转换为 1D 列表

# # 或者直接使用列表推导式提取标量值
# predictions = [float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x) for x in predictions]

# submission = pd.DataFrame({"row_id": test_df["row_id"], "rule_violation": predictions})
# submission.to_csv("submission.csv", index=False)
# print(predictions)
# # 保存提交文件
# submission = pd.DataFrame({"row_id": test_df["row_id"], "rule_violation": predictions})
# submission.to_csv("submission.csv", index=False)
# submission.head()

