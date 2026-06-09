# from google.colab import drive
# drive.mount('/content/drive')
path = "/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/"


import pandas as pd
import numpy as np
import torch
from torch import nn
import sklearn
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from collections import defaultdict


train = pd.read_csv(path + "train.csv")


train


for col in train.columns:
    print(len(train[col].unique()), end = " ")


print(train.isna().sum())


id, count = np.unique(train["CustomerId"], return_counts = True)
for i in range(id.size):
    if count[i] > 1:
        print(train[train["CustomerId"] == id[i]])
        break


train = train.drop("id", axis = 1).drop("CustomerId", axis = 1)
train


train.duplicated().sum()


for col in ("Geography", "Gender"):
    val = train[col].unique()
    val_map = {val[i]: i for i in range(len(val))}
    train[col] = train[col].map(lambda val: val_map[val])

train


plt.figure(figsize=(10, 10))
sns.heatmap(train[["CreditScore", "Age", "Gender", "Tenure", "Balance", "EstimatedSalary", "NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"]].corr(), annot=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


fig, axes = plt.subplots(nrows=5, ncols=2, figsize = (5, 10))

for coli, col in enumerate(("Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember")):

    train[col].plot(ax = axes[coli][0], kind = "hist", bins = train[col].unique().size)

    dist = pd.Series()
    for val in sorted(train[col].unique()):
        dist[val] = train["Exited"][train[col] == val].sum() / (train[col] == val).sum()
        print(round(dist[val], 2), end = " ")
    print()
    dist.plot(ax = axes[coli][1], kind = "bar")



target = train["Exited"]
input = train.drop("Surname", axis = 1).drop("Exited", axis = 1)
for col in input.columns:
    input[col] = (input[col] - input[col].min()) / input[col].max()
input


print((np.unique(train["Surname"], return_counts = True)[1] >= 100).sum())


'''
for surname in np.unique(train["Surname"])[np.unique(train["Surname"], return_counts = True)[1] >= 100]:
    input[surname] = (train['Surname'] == surname).astype(np.float32)
input
'''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


train_dataset = torch.utils.data.TensorDataset(torch.Tensor(input.values).to(device), torch.Tensor(target.values).to(device))
print(train_dataset[:5])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_size = 128
        self.input_block = nn.Linear(input.shape[1], self.residual_size)
        self.residual_num = 2
        self.residual_blocks = nn.ModuleList([nn.Sequential(nn.Linear(self.residual_size, self.residual_size), nn.ReLU(), nn.LayerNorm(self.residual_size), nn.Dropout(0.5)) for i in range(self.residual_num)])
        self.output_block = nn.Linear(self.residual_size, 1)


    def forward(self, x):
        x = self.input_block(x)
        for i in range(self.residual_num):
            x = self.residual_blocks[i](x) + x
        x = self.output_block(x)
        return x


# loss function adapted from Erik Drysdale's blog post: https://www.erikdrysdale.com/auc_max/

def cAUROC(pred, target):
    return -((torch.sigmoid(pred - pred.unsqueeze(1)).log() * target.unsqueeze(1) * (1 - target)).sum() / (target.sum() * (1 - target)).sum())


def bin_cross_entropy(pred, target):
    return -(target * torch.log(pred) + (1 - target) * torch.log((1 - pred))).sum()


torch.backends.cuda.matmul.allow_tf32 = True


lr = 1e-3
folds_num = 3
train_batch_size = 250
train_loss_fn = cAUROC
val_batch_size = 250
# val_loss_fn = lambda pred, target: binary_auroc(pred.squeeze(), target)
# val_loss_fn = lambda pred, target: nn.functional.binary_cross_entropy(pred.squeeze(), target) * val_batch_size
# val_loss_fn = lambda pred, target: bin_cross_entropy(pred.squeeze(), target)
val_loss_fn = lambda pred, target: sklearn.metrics.roc_auc_score(target.bool(), torch.sigmoid(pred.squeeze())) * val_batch_size


train_graph = None
val_graph = None
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1, capturable = True, fused = True)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.5)
input_mem = torch.zeros((train_batch_size, input.shape[1]), device = device)
target_mem = torch.zeros((train_batch_size), device = device)


for fold_i, (train_ind, val_ind) in enumerate(KFold(n_splits = folds_num, shuffle = True).split(input.values, target.values)):
    train_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.dataset.Subset(train_dataset, train_ind), batch_size = train_batch_size, shuffle = True, num_workers = 0)
    model.train()

    # warmup train
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for batch_i in range(3):
            input_batch, target_batch = next(iter(train_loader))
            input_mem.copy_(input_batch)
            target_mem.copy_(target_batch)

            optimizer.zero_grad(set_to_none = True)
            train_loss_fn(model(input_mem), target_mem).backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    # capture train
    train_graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none = True)
    with torch.cuda.graph(train_graph):
        train_loss_fn(model(input_mem), target_mem).backward()
        optimizer.step()

    '''
    # warmup val
    val_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.dataset.Subset(train_dataset_cpu, val_ind), batch_size = val_batch_size, num_workers = 0)
    model.eval()

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(warmup_stream):
            for batch_i in range(3):
                input_batch, target_batch = next(iter(val_loader))
                input_mem.copy_(input_batch)
                target_mem.copy_(target_batch)

                val_loss += val_loss_fn(model(input_mem), target_mem)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    # capture val
    val_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(val_graph):
        with torch.no_grad():
            val_loss += val_loss_fn(model(input_mem), target_mem)
    '''

    break


def reset_layer(l):
    if hasattr(l, 'reset_parameters'):
        l.reset_parameters()

for fold_i, (train_ind, val_ind) in enumerate(KFold(n_splits = folds_num, shuffle = True).split(input.values, target.values)):
    train_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.dataset.Subset(train_dataset, train_ind), batch_size = train_batch_size, shuffle = True, num_workers = 0)
    val_loader_train = torch.utils.data.DataLoader(dataset = torch.utils.data.dataset.Subset(train_dataset, train_ind), batch_size = val_batch_size, num_workers = 0)
    val_loader_test = torch.utils.data.DataLoader(dataset = torch.utils.data.dataset.Subset(train_dataset, val_ind), batch_size = val_batch_size, num_workers = 0)

    for item in optimizer.state_dict()['state'].values():
        for tensor in item.values():
            tensor.zero_()

    model.apply(reset_layer)

    print("Fold ", fold_i)


    for epoch_i in range(501):

        for batch_i, (input_batch, target_batch) in enumerate(train_loader):
            if (input_batch.size()[0] != train_batch_size):
                break
            input_mem.copy_(input_batch)
            target_mem.copy_(target_batch)
            train_graph.replay()

        if (epoch_i % 20 == 0):
            val_loss = 0
            with torch.no_grad():
                for batch_i, (input_batch, target_batch) in enumerate(val_loader_test):
                    if (input_batch.size()[0] != val_batch_size):
                        break
                    val_loss += val_loss_fn(model(input_batch).to('cpu'), target_batch.to('cpu'))
            val_loss = val_loss / (len(train_dataset) // folds_num)


            train_loss = 0
            with torch.no_grad():
                for batch_i, (input_batch, target_batch) in enumerate(val_loader_train):
                    if (input_batch.size()[0] != val_batch_size):
                        break
                    train_loss += val_loss_fn(model(input_batch).to('cpu'), target_batch.to('cpu'))

            train_loss = train_loss / ((len(train_dataset) * (folds_num - 1)) // folds_num)

            print(f"epoch {epoch_i:2d}: {train_loss:f} {val_loss:f}")
            print(f"optmizer step: {optimizer.param_groups[0]['lr']}")







train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True, num_workers = 0)
val_loader_train = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = val_batch_size, num_workers = 0)

for item in optimizer.state_dict()['state'].values():
    for tensor in item.values():
        tensor.zero_()

model.apply(reset_layer)

max_train_loss = 0
for epoch_i in range(501):

    for batch_i, (input_batch, target_batch) in enumerate(train_loader):
        if (input_batch.size()[0] != train_batch_size):
            break
        input_mem.copy_(input_batch)
        target_mem.copy_(target_batch)
        train_graph.replay()

    if (epoch_i % 10 == 0):

        train_loss = 0
        with torch.no_grad():
            for batch_i, (input_batch, target_batch) in enumerate(val_loader_train):
                if (input_batch.size()[0] != val_batch_size):
                    break
                train_loss += val_loss_fn(model(input_batch).to('cpu'), target_batch.to('cpu'))

        train_loss = train_loss / len(train_dataset)

        print(f"epoch {epoch_i:2d}: {train_loss:f}")

        if (train_loss > max_train_loss):
            max_train_loss = train_loss
            print("Saving model")
            with torch.serialization.safe_globals([Model]):
                torch.save(model, 'model.pt')


test = pd.read_csv(path + "test.csv")


test = test.drop("id", axis = 1).drop("CustomerId", axis = 1)
test


for col in ("Geography", "Gender"):
    val = test[col].unique()
    val_map = {val[i]: i for i in range(len(val))}
    test[col] = test[col].map(lambda val: val_map[val])

test


input = test.drop("Surname", axis = 1)
for col in input.columns:
    input[col] = (input[col] - input[col].min()) / input[col].max()
input


test_data = torch.Tensor(input.values).to(device)
print(test_data[:5])


test_model = torch.load('model.pt', weights_only = False)
test_model.eval()


result = test_model(test_data)
result


result = torch.sigmoid(result)
result


result_dataframe = pd.DataFrame()
result_dataframe['id'] = [15000 + i for i in range (10000)]
result_dataframe['Exited'] = result.to('cpu').detach()
result_dataframe


result_dataframe.to_csv("result.csv", index = False)

