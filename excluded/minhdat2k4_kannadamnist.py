import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm


class KannadaMNIST(Dataset):
    def __init__(self, file_path, transform = None, is_train = True):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        self.is_train = is_train
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.is_train:
            image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape(28, 28, 1)
            label = self.data.iloc[index, 0]
        else:
            image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape(28, 28, 1)
        if self.transform:
            image = self.transform(image)
        if self.is_train:
            return image, label
        return image


trainDataset = KannadaMNIST(file_path = '/kaggle/input/Kannada-MNIST/train.csv', transform = transforms.ToTensor(), is_train = True)
testDataset = KannadaMNIST(file_path = '/kaggle/input/Kannada-MNIST/test.csv', transform = transforms.ToTensor(), is_train = False)


trainDataloader = DataLoader(
    dataset = trainDataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 3
)
testDataloader = DataLoader(
    dataset = testDataset,
    batch_size = 1,
    num_workers = 3
)


dataIter = iter(trainDataloader)
images, labels = next(dataIter)
images = images.numpy()

fig = plt.figure(figsize=(30,4))
for idx in np.arange(32):
    ax = fig.add_subplot(2, 16, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(str(labels[idx].item()))


class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x =x.view(-1, 784)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output(x)
        return x


model = MLP(10)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_loss_min = np.Inf
for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()
    for images, labels in tqdm(trainDataloader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)

    train_loss = train_loss / len(trainDataloader.dataset)

    print('Epoch: {} \tTraining Loss: {:.4f}'.format(
        epoch+1, 
        train_loss,
        ))
    
    if train_loss <= train_loss_min:
        print('Training loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(
        train_loss_min,
        train_loss))
        torch.save(model.state_dict(), 'model.pt')
        train_loss_min = train_loss



model.load_state_dict(torch.load('model.pt', weights_only=True))


def predict(model, dataloader):
    prediction_list = []
    for i, batch in enumerate(dataloader):
        outputs = model(batch.to(device))
        _, predicted = torch.max(outputs.data, 1) 
        prediction_list.append(predicted.cpu())
    return prediction_list
    
predictions = predict(model,testDataloader)
predictions = np.array(predictions)
predictions = np.squeeze(predictions)
print(predictions)


test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")


submission = pd.DataFrame(data={
    "id": test_data.index,
    "label": predictions
})
print(submission)
submission.to_csv("submission.csv", index=None)

