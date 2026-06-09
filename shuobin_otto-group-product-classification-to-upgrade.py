import pandas as pd
train_df = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv")
test_df = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv")



print(train_df.isna().any().unique()) # 说明没有空值


print(train_df["target"].value_counts())


from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder

X = train_df[train_df.columns.difference(["id", "target"])]
y = LabelEncoder().fit_transform(train_df["target"])
selector = SelectKBest(k=65).fit(X, y) # 再往上加特征数，模型的效果提升很小

selected_features = selector.get_feature_names_out()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.loc[:, selected_features], y, test_size=0.2, random_state=0)
print(type(X_train))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(n_estimators=100, random_state=0) # 调整n_estimators没用
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# test_prob_df是X_test的预测概率DataFrame
test_prob = model.predict_proba(X_test)
class_names = model.classes_
test_prob_df = pd.DataFrame(test_prob, columns=class_names, index=X_test.index)
print(test_prob_df)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, columns=model.classes_)
print(cm_df)


# 生成提交文件，包含id和每个类别的概率
X = test_df[test_df.columns.difference(["id"])]
X  = X.loc[:, selected_features]
proba = model.predict_proba(X)
class_names = [f"Class_{i}" for i in range(1, proba.shape[1] + 1)]
submission = pd.DataFrame(proba, columns=class_names)
submission.insert(0, "id", test_df.loc[X.index, "id"].values)
#submission.to_csv("submission.csv", index=False)


from sklearn.ensemble import RandomForestClassifier

# 1. 找到模型A预测为class_1的test样本
pred_class = submission.iloc[:, 1:].idxmax(axis=1)
mask_1 = pred_class == 'Class_1'
test_1_idx = submission[mask_1].index

# 2. 用train.csv中class_1, class_6, class_8, class_9的数据训练模型B（用全部特征）
target_classes = ['Class_1', 'Class_6', 'Class_8', 'Class_9']
train_1_6_8_9 = train_df[train_df['target'].isin(target_classes)]
X_1_6_8_9 = train_1_6_8_9.drop(['id', 'target'], axis=1)
y_1_6_8_9 = train_1_6_8_9['target']
model_B = RandomForestClassifier(n_estimators=100, random_state=0)
model_B.fit(X_1_6_8_9, y_1_6_8_9)

# 3. 用模型B对test.csv中被模型A预测为class_1的样本重新预测概率
X_1_test = test_df.loc[test_1_idx, X_1_6_8_9.columns]
proba_B = model_B.predict_proba(X_1_test)
class_B = model_B.classes_

# 4. 替换submission中对应样本的class_1, class_6, class_8, class_9概率
for i, idx in enumerate(test_1_idx):
    for j, cls in enumerate(class_B):
        submission.at[idx, cls] = proba_B[i, j]


from sklearn.ensemble import RandomForestClassifier

# 1. 找到模型A预测为class_2的test样本
pred_class = submission.iloc[:, 1:].idxmax(axis=1)
mask_2 = pred_class == 'Class_2'
test_2_idx = submission[mask_2].index

# 2. 用train.csv中class_2, class_3, class_4, class_6, class_7的数据训练模型B（用全部特征）
target_classes = ['Class_2', 'Class_3', 'Class_4', 'Class_6', 'Class_7']
train_2_3_4_6_7 = train_df[train_df['target'].isin(target_classes)]
X_2_3_4_6_7 = train_2_3_4_6_7.drop(['id', 'target'], axis=1)
y_2_3_4_6_7 = train_2_3_4_6_7['target']
model_B = RandomForestClassifier(n_estimators=100, random_state=0)
model_B.fit(X_2_3_4_6_7, y_2_3_4_6_7)

# 3. 用模型B对test.csv中被模型A预测为class_2的样本重新预测概率
X_2_test = test_df.loc[test_2_idx, X_2_3_4_6_7.columns]
proba_B = model_B.predict_proba(X_2_test)
class_B = model_B.classes_

# 4. 替换submission中对应样本的class_2, class_3, class_4, class_6, class_7概率
for i, idx in enumerate(test_2_idx):
    for j, cls in enumerate(class_B):
        submission.at[idx, cls] = proba_B[i, j]


from sklearn.ensemble import RandomForestClassifier

# 1. 找到模型A预测为class_3的test样本
pred_class = submission.iloc[:, 1:].idxmax(axis=1)
mask_3 = pred_class == 'Class_3'
test_3_idx = submission[mask_3].index

# 2. 用train.csv中class_2, class_3, class_4, class_7的数据训练模型B（用全部特征）
target_classes = ['Class_2', 'Class_3', 'Class_4', 'Class_7']
train_2_3_4_7 = train_df[train_df['target'].isin(target_classes)]
X_2_3_4_7 = train_2_3_4_7.drop(['id', 'target'], axis=1)
y_2_3_4_7 = train_2_3_4_7['target']
model_B = RandomForestClassifier(n_estimators=100, random_state=0)
model_B.fit(X_2_3_4_7, y_2_3_4_7)

# 3. 用模型B对test.csv中被模型A预测为class_3的样本重新预测概率
X_3_test = test_df.loc[test_3_idx, X_2_3_4_7.columns]
proba_B = model_B.predict_proba(X_3_test)
class_B = model_B.classes_

# 4. 替换submission中对应样本的class_2, class_3, class_4, class_7概率
for i, idx in enumerate(test_3_idx):
    for j, cls in enumerate(class_B):
        submission.at[idx, cls] = proba_B[i, j]


from sklearn.ensemble import RandomForestClassifier

# 1. 找到模型A预测为class_4的test样本
pred_class = submission.iloc[:, 1:].idxmax(axis=1)
mask_4 = pred_class == 'Class_4'
test_4_idx = submission[mask_4].index

# 2. 用train.csv中class_2, class_3, class_4, class_7的数据训练模型B（用全部特征）
target_classes = ['Class_2', 'Class_3', 'Class_4', 'Class_7']
train_2_3_4_7 = train_df[train_df['target'].isin(target_classes)]
X_2_3_4_7 = train_2_3_4_7.drop(['id', 'target'], axis=1)
y_2_3_4_7 = train_2_3_4_7['target']
model_B = RandomForestClassifier(n_estimators=100, random_state=0)
model_B.fit(X_2_3_4_7, y_2_3_4_7)

# 3. 用模型B对test.csv中被模型A预测为class_4的样本重新预测概率
X_4_test = test_df.loc[test_4_idx, X_2_3_4_7.columns]
proba_B = model_B.predict_proba(X_4_test)
class_B = model_B.classes_

# 4. 替换submission中对应样本的class_2, class_3, class_4, class_7概率
for i, idx in enumerate(test_4_idx):
    for j, cls in enumerate(class_B):
        submission.at[idx, cls] = proba_B[i, j]


from sklearn.ensemble import RandomForestClassifier

# 1. 找到模型A预测为class_7的test样本
pred_class = submission.iloc[:, 1:].idxmax(axis=1)
mask_7 = pred_class == 'Class_7'
test_7_idx = submission[mask_7].index

# 2. 用train.csv中class_1, class_2, class_3, class_6, class_7的数据训练模型B（用全部特征）
target_classes = ['Class_1', 'Class_2', 'Class_3', 'Class_6', 'Class_7']
train_1_2_3_6_7 = train_df[train_df['target'].isin(target_classes)]
X_1_2_3_6_7 = train_1_2_3_6_7.drop(['id', 'target'], axis=1)
y_1_2_3_6_7 = train_1_2_3_6_7['target']
model_B = RandomForestClassifier(n_estimators=100, random_state=0)
model_B.fit(X_1_2_3_6_7, y_1_2_3_6_7)

# 3. 用模型B对test.csv中被模型A预测为class_7的样本重新预测概率
X_7_test = test_df.loc[test_7_idx, X_1_2_3_6_7.columns]
proba_B = model_B.predict_proba(X_7_test)
class_B = model_B.classes_

# 4. 替换submission中对应样本的class_1, class_2, class_3, class_6, class_7概率
for i, idx in enumerate(test_7_idx):
    for j, cls in enumerate(class_B):
        submission.at[idx, cls] = proba_B[i, j]


from sklearn.ensemble import RandomForestClassifier

# 1. 找到模型A预测为class_6的test样本
pred_class = submission.iloc[:, 1:].idxmax(axis=1)
mask_6 = pred_class == 'Class_6'
test_6_idx = submission[mask_6].index

# 2. 用train.csv中class_1, class_6, class_7, class_8, class_9的数据训练模型B（用全部特征）
target_classes = ['Class_1', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
train_1_6_7_8_9 = train_df[train_df['target'].isin(target_classes)]
X_1_6_7_8_9 = train_1_6_7_8_9.drop(['id', 'target'], axis=1)
y_1_6_7_8_9 = train_1_6_7_8_9['target']
model_B = RandomForestClassifier(n_estimators=100, random_state=0)
model_B.fit(X_1_6_7_8_9, y_1_6_7_8_9)

# 3. 用模型B对test.csv中被模型A预测为class_6的样本重新预测概率
X_6_test = test_df.loc[test_6_idx, X_1_6_7_8_9.columns]
proba_B = model_B.predict_proba(X_6_test)
class_B = model_B.classes_

# 4. 替换submission中对应样本的class_1, class_6, class_7, class_8, class_9概率
for i, idx in enumerate(test_6_idx):
    for j, cls in enumerate(class_B):
        submission.at[idx, cls] = proba_B[i, j]


# 1. 找到模型A预测为class_8的test样本
pred_class = submission.iloc[:, 1:].idxmax(axis=1)
mask_8 = pred_class == 'Class_8'
test_8_idx = submission[mask_8].index

# 2. 用train.csv中class_1, class_6, class_7, class_8, class_9的数据训练模型B（用全部特征）
# 上面的模型B可以复用

# 3. 用模型B对test.csv中被模型A预测为class_8的样本重新预测概率
X_8_test = test_df.loc[test_8_idx, X_1_6_7_8_9.columns]
proba_B = model_B.predict_proba(X_8_test)
class_B = model_B.classes_

# 4. 替换submission中对应样本的class_1, class_6, class_7, class_8, class_9概率
for i, idx in enumerate(test_8_idx):
    for j, cls in enumerate(class_B):
        submission.at[idx, cls] = proba_B[i, j]


# 1. 找到模型A预测为class_9的test样本
pred_class = submission.iloc[:, 1:].idxmax(axis=1)
mask_9 = pred_class == 'Class_9'
test_9_idx = submission[mask_9].index

# 2. 用train.csv中class_1, class_6, class_7, class_8, class_9的数据训练模型B（用全部特征）
# 上面的模型B可以复用

# 3. 用模型B对test.csv中被模型A预测为class_9的样本重新预测概率
X_9_test = test_df.loc[test_9_idx, X_1_6_7_8_9.columns]
proba_B = model_B.predict_proba(X_9_test)
class_B = model_B.classes_

# 4. 替换submission中对应样本的class_1, class_6, class_7, class_8, class_9概率
for i, idx in enumerate(test_9_idx):
    for j, cls in enumerate(class_B):
        submission.at[idx, cls] = proba_B[i, j]


submission.to_csv("submission.csv", index=False)


import pandas as pd
train_df = pd.read_csv(r"/kaggle/input/otto-group-product-classification-challenge/train.csv")
test_df = pd.read_csv(r"/kaggle/input/otto-group-product-classification-challenge/train.csv")


import torch
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
X = train_df[train_df.columns.difference(["id", "target"])]
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(train_df.iloc[:, 94:])
y = ohe.transform(train_df.iloc[:, 94:])
selector = SelectKBest(k=65).fit(X, LabelEncoder().fit_transform(train_df["target"])) # 再往上加特征数，模型的效果提升很小
selected_features = selector.get_feature_names_out()
X_selected_features_data = X.loc[:, selected_features]

X_tensor = torch.tensor(X_selected_features_data.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)



import torch.nn as nn

class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(len(selected_features), 40)
        self.act = nn.ReLU()
        self.output = nn.Linear(40, 9)
        
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x
    
model = Multiclass()
model.to(device)


import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, train_size=0.7, shuffle=True)


import copy
import numpy as np
# training parameters
best_acc = - np.inf 
best_weights = None
n_epochs = 20
batch_size = 5
batches_per_epoch = len(X_train) # batch_size
print(batches_per_epoch)
for epoch in range(n_epochs): 
    for i in range(0, batches_per_epoch, 5):
        # take a batch
        start = i
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
    y_pred = model(X_test)
    ce = loss_fn(y_pred, y_test)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    if float(acc) > best_acc:
        best_acc = float(acc)
        best_weights = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch} validation: Cross-entropy={float(ce)}, Accuracy={float(acc)}")

model.load_state_dict(best_weights) # 留下最好的模型
print(best_acc)


X = test_df[test_df.columns.difference(["id"])]
X_selected_features_data = X.loc[:, selected_features]
X_tensor = torch.tensor(X_selected_features_data.values, dtype=torch.float32).to(device)

y_pred = model(X_tensor)
max_ele_idx = torch.argmax(y_pred, 1) # 获得每行最大的元素的下标
# 请根据max_ele_idx，生成一个与下面格式相同的csv文件，例如：max_ele_idx第一个元素的值是3，说明id=1时，Class_4=1，其余均为0；max_ele_idx第二个元素的值是7，说明id=2时，Class_8=1，其余均为0。
# id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9
# 1,1,0,0,0,0,0,0,0,0
# 2,1,0,0,0,0,0,0,0,0
# 先构造一个DataFrame，然后toCSV()
sample_df = pd.read_csv(r"/kaggle/input/otto-group-product-classification-challenge/sampleSubmission.csv")
res_df = pd.DataFrame(columns=sample_df.columns)
print(len(y_pred) + 1)
for i in range(1, len(y_pred) + 1):
    tmp_arr = [i, 0,0,0,0,0,0,0,0,0]
    tmp_arr[max_ele_idx[i-1].item() + 1] = 1
    res_df.loc[i-1] = tmp_arr
res_df.to_csv(r"/kaggle/working/result.csv", index= False)

