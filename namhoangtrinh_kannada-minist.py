import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# DataClass
class TrainMNIST(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28,28,1))
        label = self.data.iloc[index, 0]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TestMNIST(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28,28,1))
        # label = self.data.iloc[index, 0]
        if self.transform is not None:
            image = self.transform(image)
        return image


# Load the data
train_data = TrainMNIST("/kaggle/input/Kannada-MNIST/train.csv", transform=transforms.ToTensor())
test_data = TestMNIST("/kaggle/input/Kannada-MNIST/test.csv", transform=transforms.ToTensor())


# DataLoader
num_workers = 0
batch_size = 20
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


# obtain 1 batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter) # take 1 batch
images = images.numpy()

# plot the images in the batch along with the corresponding labels
fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(str(labels[idx].item()))


# Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer(512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784-->hidden_1)
        self.fc1 = nn.Linear(28*28, hidden_1)
        # linear layer (hidden_1-->hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (hidden_2-->10)
        self.fc3 = nn.Linear(hidden_2, 10)
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28*28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# initialize the NN
device = torch.device("cuda")
model = Net().to(device)
print(model)


# specify loss function(categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer (stochastic gradient descent) and learning rate=0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


from tqdm import tqdm
# number of epochs to train the model
n_epochs = 20
# initialize tracker for minimum validation loss
train_loss_min = np.Inf

for epoch in range(n_epochs):
    train_loss = 0.0
    model.train()
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()
        # move to gpu
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient descent of the loss with respect to model parameter
        loss.backward()
        # perform a single optimization step
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch: {epoch+1} \tTraining Loss:{train_loss:.6f}")

    # save model if validation loss has decreased
    if train_loss <= train_loss_min:
        print(f"Training loss decreased ({train_loss_min}-->{train_loss}). Saving model...")
        torch.save(model.state_dict(), "model.pt")
        train_loss_min = train_loss


model.load_state_dict(torch.load("model.pt", weights_only=True))


# Visualize test results
# Obtain 1 batch of test images
dataiter = iter(test_loader)
images = next(dataiter)

# Move to GPU
images = images.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)
# Move back to CPU for visualization
images = images.cpu().numpy()

# plot the images in the batch along with predicted and true labels
fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(f"{str(preds[idx].item())}")


def predict(model, dataloader):
    prediction_list = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for data in dataloader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            prediction_list.extend(predicted.cpu().numpy())  # Append predictions as a list
    return np.array(prediction_list)  # Convert the list to a 1D NumPy array


predictions = predict(model,test_loader)


test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
submission = pd.DataFrame(data={
    "id": test_data.index + 1,
    "label": predictions
})
submission.to_csv("submission.csv", index=None)

