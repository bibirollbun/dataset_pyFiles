import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm



train_dir = '/kaggle/input/classification-of-butterflies/train_butterflies/train_split'

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)


test_dir = '/kaggle/input/classification-of-butterflies/test_butterflies/valid'

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        img_id = int(img_name.split('.')[0])
        return image, img_id

test_dataset = TestDataset(test_dir, transform=transform_test)


test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)



model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader, criterion, optimizer, epochs=20)



def predict(model, dataloader):
    model.eval()
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for inputs, ids in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(ids.numpy())
    return all_ids, all_preds

test_ids, test_preds = predict(model, test_loader)

submission = pd.DataFrame({'index': test_ids, 'label': test_preds})
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv успешно создан!")





