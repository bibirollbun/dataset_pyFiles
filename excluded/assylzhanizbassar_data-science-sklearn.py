import os
import pandas as pd

root = '/kaggle/input/data-science-london-scikit-learn'

train_path = os.path.join(root, 'train.csv')
train_df = pd.read_csv(train_path, sep=',', header=None)
train_df.head()


train_label_path = os.path.join(root, 'trainLabels.csv')
train_labels_df = pd.read_csv(train_label_path, sep=',', header=None)
train_labels_df.head()


import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SyntheticDataset(Dataset):
    def __init__(self, root, train, transform=None):
        self.features = pd.read_csv(
            os.path.join(root, 'train.csv'),
            sep=',',
            header=None,
            index_col=None,
        )
        self.labels = pd.read_csv(
            os.path.join(root, 'trainLabels.csv'),
            sep=',',
            header=None,
            index_col=None
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.labels,
            test_size=0.33,
            random_state=31
        )
        self.train = train
        self.transform = transform

    def __len__(self):
        if self.train:
            return len(self.X_train)
        return len(self.X_test)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.train:
            sample_features = self.X_train.iloc[idx]
            sample_label = self.y_train.iloc[idx]
        else:
            sample_features = self.X_test.iloc[idx]
            sample_label = self.y_test.iloc[idx]

        sample_features = np.array(sample_features)
        
        if self.transform:
            sample_features = self.transform(sample_features)
            sample_label = self.transform(sample_label)
        return sample_features, sample_label



class ToTensor():
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        if isinstance(sample, np.ndarray):
            features = torch.from_numpy(sample).type(torch.float)
            return features
        else:
            label = torch.tensor(sample, dtype=torch.long)
            return label.squeeze_()

train_ds = SyntheticDataset(
    root=root,
    train=True,
    transform=ToTensor()
)

print(train_ds[33][1])


from torch import nn
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


train_dataloader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True
)

model = NeuralNetwork(train_dataloader.dataset[0][0].shape[0])  # type: ignore
print(model)


X, y = next(iter(train_dataloader))
print(X.shape, y.shape, y[:5])



learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 5
batch_size = 64

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if not batch % 2:
            loss, current = loss.item(), len(X) + batch * batch_size
            print(f'loss: {loss:7>f} [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= batches
    correct /= size

    print(f'Accuracy = {100*correct:>0.1f}%, Avg. loss = {test_loss:>8f}\n')



test_ds = SyntheticDataset(
    root=root,
    train=False,
    transform=ToTensor()
)

test_dataloader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=True
)

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}------------')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print('Done!')


class SubmissionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.validate_data = pd.read_csv(
            os.path.join(root, 'test.csv'),
            sep=',',
            header=None,
            index_col=None
        )
        self.transform = transform

    def __len__(self):
        return len(self.validate_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.validate_data.iloc[idx]
        sample = np.array(sample, dtype=np.float32)

        if self.transform:
            sample = self.transform(sample)
        return sample


validate_ds = SubmissionDataset(
    root=root,
    transform=ToTensor()
)
print(validate_ds[43])
validate_dataloader = DataLoader(validate_ds, batch_size=64)


submission_data = []
model.eval()

with torch.no_grad():
    for X in validate_dataloader:
        logits = model(X)
        submission_data.append(logits.argmax(1).numpy())

result = {
    'Id': [],
    'Solution': []
}
cnt = 1

for item in submission_data:
    for x in item:
        result['Id'].append(int(cnt))
        result['Solution'].append(int(x))
        cnt += 1

print(result)

result = pd.DataFrame(result)
result.to_csv(os.path.join('', 'submission.csv'), index=None)  # type: ignore
print(result.shape)

