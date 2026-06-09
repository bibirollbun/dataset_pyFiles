!pip install -U -q ultralytics


# работал в kaggle, некоторые библиотеки в colab, возможно, нужно установить вручную
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import os
from transformers import AutoProcessor, BlipForConditionalGeneration
from huggingface_hub import login
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import copy
from torchvision import transforms
from collections import defaultdict
import timm
from ultralytics import YOLO


# число изображений в трейне
image_folder = "/kaggle/input/lamoda-images-classification/images/train"
images = os.listdir(image_folder)
print(len(images))


# число изображений в тесте
image_folder_test = "/kaggle/input/lamoda-images-classification/images/test"
images_test = os.listdir(image_folder_test)
print(len(images_test))


# разрешения в трейне
res = {}
folder = "/kaggle/input/lamoda-images-classification/images/train"

for root, _, files in os.walk(folder):
    for file in tqdm(files):
        path = os.path.join(root, file)
        with Image.open(path) as img:
            size = img.size
            res[size] = res.get(size, 0) + 1
print(res)


# разрешения в тесте
res_test = {}
folder_test = "/kaggle/input/lamoda-images-classification/images/test"

for root, _, files in os.walk(folder_test):
    for file in tqdm(files):
        path = os.path.join(root, file)
        with Image.open(path) as img:
            size = img.size
            res_test[size] = res_test.get(size, 0) + 1
print(res_test)


# соотношение классов
folder = "/kaggle/input/lamoda-images-classification/images/train"
br_count = 0
bl_count = 0

for root, _, files in os.walk(folder):
    for file in tqdm(files):
        path = os.path.join(root, file)
        if 'bluzy' in path:
            bl_count+=1
        else:
            br_count+=1
print("Брюк:",br_count)
print("Блузок:",bl_count)


model = timm.create_model('resnet50', pretrained=False, num_classes=1000) # модель для получения эмбеддингов изображений
model.fc = nn.Identity() # берем выход предпоследнего слоя (до классификационного полносвязного)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


preprocess = transforms.Compose([ # предобработка изображений для resnet
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze().cpu().numpy()


# протестируем
get_embedding('/kaggle/input/lamoda-images-classification/images/train/0001_bluzy.jpg')


def show_image(image_path):
    image = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(os.path.basename(image_path))
    plt.show()


show_image('/kaggle/input/lamoda-images-classification/images/train/0002_bluzy.jpg')


show_image('/kaggle/input/lamoda-images-classification/images/train/15732_bryuki.jpg')


# попробуем найти мусор через косинусное расстояние от случайных изображений разных классов (заодно посмотрим на качество эмбеддингов)
def get_top_n(main_image_path, folder_path, n = 5):
    main_embedding = get_embedding(main_image_path)
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            if os.path.abspath(img_path) != os.path.abspath(main_image_path):
                image_paths.append(img_path)
    dists = []
    for img_path in tqdm(image_paths):
        emb = get_embedding(img_path)
        cos_dist = 1 - np.dot(main_embedding, emb) / (np.linalg.norm(main_embedding) * np.linalg.norm(emb))
        dists.append((img_path, cos_dist))
    dists.sort(key=lambda x: x[1])
    top_similar = dists[:n]
    top_dissimilar = dists[-n:]
    print('Основное изображение:')
    show_image(main_image_path)
    print('Топ',n,'похожих:')
    for i, (path, dist) in enumerate(top_similar, 1):
        print('Косинусное расстояние =',dist)
        print(path)
        show_image(path)
    print("\n")
    print('Топ',n,'непохожих:')
    for i, (path, dist) in enumerate(top_dissimilar, 1):
        print('Косинусное расстояние =',dist)
        print(path)
        show_image(path)


get_top_n("/kaggle/input/lamoda-images-classification/images/train/0001_bluzy.jpg", '/kaggle/input/lamoda-images-classification/images/train', n = 10)


get_top_n("/kaggle/input/lamoda-images-classification/images/train/15732_bryuki.jpg", '/kaggle/input/lamoda-images-classification/images/train', n = 10)


login() # нужно ввести токен Hugging Face


# загрузка предобученной модели
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# демонстрация генерации описания
image_path = "/kaggle/input/lamoda-images-classification/images/train/0002_bluzy.jpg"
image = Image.open(image_path)

text = "A picture of"
inputs = processor(
    images=image,
    text=text,
    return_tensors="pt",
    truncation=True,      
    padding=True          
)

with torch.no_grad():
    outputs = model.generate(**inputs)
    
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


train_set = pd.DataFrame(columns=['file_name','description','class'])
for img in tqdm(images):
    path = os.path.join(image_folder, img)
    image = Image.open(path)
    path = path.split('/')[-1]
    cls = 2
    if 'bryuki' in path:
        cls = 0
    elif 'bluzy' in path:
        cls = 1
    inputs = processor(
        images=image,
        text="a picture of",
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)    
    caption = processor.decode(outputs[0], skip_special_tokens=True)      
    new_rows = pd.DataFrame({"file_name": [path], "description": [caption], "class": [cls]})
    train_set = pd.concat([train_set, new_rows], ignore_index=True)  


train_set


train_set.to_csv('lamoda_train.csv',index=False)


test_set = pd.DataFrame(columns=['file_name','description','class'])
for img in tqdm(images_test):
    path = os.path.join(image_folder_test, img)
    image = Image.open(path)
    path = path.split('/')[-1]
    cls = 2
    inputs = processor(
        images=image,
        text="a picture of",
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)    
    caption = processor.decode(outputs[0], skip_special_tokens=True)      
    new_rows = pd.DataFrame({"file_name": [path], "description": [caption], "class": [cls]})
    test_set = pd.concat([test_set, new_rows], ignore_index=True)  


test_set


test_set.to_csv('lamoda_test.csv',index=False)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.labels = labels.tolist() if labels is not None and isinstance(labels, pd.Series) else labels
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


texts_train = pd.read_csv('/kaggle/input/lamoda-dd/lamoda_train.csv')
texts_test = pd.read_csv('/kaggle/input/lamoda-dd/lamoda_test.csv')


train_set = texts_train.sample(frac=0.9, random_state=42).reset_index(drop=True)
val_set = texts_train.drop(train_set.index).reset_index(drop=True)


train_dataset = TextDataset(train_set['description'], train_set['class'])
val_dataset = TextDataset(val_set['description'], val_set['class'])

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.tensor([item.get('labels', -1) for item in batch])
    }

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)


# свой класс создавать не нужно, просто инициализируем берт с нужным количеством классов
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


def train(model_base, train_loader, val_loader, num_epochs=100, learning_rate=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = copy.deepcopy(model_base)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_losses, train_accs, val_accs = [], [], []
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        for batch in tqdm(train_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
    
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == inputs['labels']).sum().item()
            total_train += inputs['labels'].size(0)
        
        train_acc = correct_train / total_train
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['labels'].to(device)
                }
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == inputs['labels']).sum().item()
                total_val += inputs['labels'].size(0)
        val_acc = correct_val / total_val
        val_accs.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'lamoda_bert.pth') # функцию можно прерывать, необязательно учить 100 эпох
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


# функция для валидации на всем трейне
def check(model, train_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    correct = 0
    total = 0    
    with torch.no_grad():
        for batch in tqdm(train_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct += (preds == inputs['labels']).sum().item()
            total += inputs['labels'].size(0)
        acc = correct / total
    return acc


def predict(model, tokenizer, sample):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    inputs = tokenizer(
        sample,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(probabilities, dim=1).item()
    return pred_label


torch.cuda.empty_cache()


train(model, train_loader, val_loader)


model.load_state_dict(torch.load('/kaggle/input/lamoda-dd/lamoda_bert.pth'))


# продолжаем обучать после прерывания
train(model, train_loader, val_loader)


train_set = texts_train
train_dataset = TextDataset(train_set['description'], train_set['class'])
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


model.load_state_dict(torch.load('/kaggle/input/lamoda-dd/lamoda_bert.pth'))


check(model, train_loader)


texts_test


for i in tqdm(range(len(texts_test))):
    cls = predict(model, tokenizer, texts_test.loc[i,'description'])
    cls = 'bryuki' if cls==0 else 'bluzy'
    texts_test.loc[i, 'class']=cls


texts_test


texts_test = texts_test.drop(columns=['description'])


texts_test = texts_test.rename(columns={'file_name': 'index', 'class': 'label'})


texts_test


texts_test.to_csv('submission.csv',index=False)


class ResNetDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                img_path = os.path.join(root, file)
                self.image_paths.append(img_path)
                if 'bryuki' in img_path:
                    self.labels.append(0)
                elif 'bluzy' in img_path:
                    self.labels.append(1)
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform(image)
        return image, label


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = ResNetDataset('/kaggle/input/lamoda-images-classification/images/train', data_transforms)
d_size = len(full_dataset)
v_size = int(d_size*0.2)
t_size = d_size - v_size
train_set, val_set = random_split(full_dataset, [t_size, v_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=8, num_workers=4)


class ResNetModel1(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel1, self).__init__()
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=1000)
        for name, param in self.resnet.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        output = self.classifier(features)
        return output


def train_resnet1(model_base, train_loader, val_loader, num_epochs=100, learning_rate=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = copy.deepcopy(model_base)
    model.to(device)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) 
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = train_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)


        print('Epoch',epoch+1)
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'lamoda_resnet.pth')

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


model = ResNetModel1(num_classes=2)


train_resnet1(model, train_loader, val_loader)


def calc_accuracy(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    dataset = ResNetDataset("/kaggle/input/lamoda-images-classification/images/train",data_transforms)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total
    return acc


resnet1 = ResNetModel1(num_classes=2)
resnet1.load_state_dict(torch.load('/kaggle/input/lamoda-dd/lamoda_resnet.pth'))


calc_accuracy(resnet1)


def predict(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    image_names = []
    predictions = []
    with torch.no_grad():
        for image_name in tqdm(os.listdir('/kaggle/input/lamoda-images-classification/images/test')):
            image_path = os.path.join('/kaggle/input/lamoda-images-classification/images/test', image_name)
            image = Image.open(image_path).convert('RGB')
            image = data_transforms(image).unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
            image_names.append(image_name)
            predictions.append(prediction)
    ans = pd.DataFrame({'index': image_names, 'label': predictions})
    return ans


pre_final = predict(resnet1)


pre_final


for i in range(len(pre_final)):
    if pre_final.iloc[i,1]==0:
        pre_final.iloc[i,1]='bryuki'
    else:
        pre_final.iloc[i,1]='bluzy'


pre_final


pre_final.to_csv('submission.csv',index=False)


class YOLODataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_paths = []
        self.labels = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                img_path = os.path.join(root, file)
                self.image_paths.append(img_path)
                if 'bryuki' in img_path:
                    self.labels.append(0)
                elif 'bluzy' in img_path:
                    self.labels.append(1)
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        return image, label


class DatasetWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = self.transform(image)
        return image, label


# введем аугментации
t_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

v_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


full_dataset = YOLODataset('/kaggle/input/lamoda-images-classification/images/train')
d_size = len(full_dataset)
v_size = int(d_size*0.2)
t_size = d_size - v_size
train_set, val_set = random_split(full_dataset, [t_size, v_size])
train_set = DatasetWrapper(train_set, transform=t_transforms)
val_set = DatasetWrapper(val_set, transform=v_transforms)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4) # увеличим размер батча (у резнет было 8)
val_loader = DataLoader(val_set, batch_size=16, num_workers=4)


class YOLO1(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = YOLO('yolov8n-cls.pt').model.model
        for param in list(self.backbone.parameters())[:-5]:
            param.requires_grad = False
        num_features = 256
        self.backbone[-1] = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(), # попробуем вариацию ReLU
            nn.Dropout(0.3), # увеличим дропаут (было у резнет 0.2)
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def train_yolo(model_base, train_loader, val_loader, num_epochs=100, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = copy.deepcopy(model_base)
    model.to(device)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( # на плато accuracy валидации будем снижать скорость обучения
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0 
        total = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = train_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(device) 
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'lamoda_yolo.pth')
        
        scheduler.step(val_epoch_acc)
    
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


model = YOLO1(num_classes=2)


model


train_yolo(model, train_loader, val_loader)


def calc_accuracy(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    dataset = YOLODataset("/kaggle/input/lamoda-images-classification/images/train")
    dataset = DatasetWrapper(dataset, transform=v_transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total
    return acc


yolo1 = YOLO1(num_classes=2)
yolo1.load_state_dict(torch.load('/kaggle/input/lamoda-dd/lamoda_yolo.pth'))


calc_accuracy(yolo1)


def predict(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    image_names = []
    predictions = []
    with torch.no_grad():
        for image_name in tqdm(os.listdir('/kaggle/input/lamoda-images-classification/images/test')):
            image_path = os.path.join('/kaggle/input/lamoda-images-classification/images/test', image_name)
            image = Image.open(image_path).convert('RGB')
            image = v_transforms(image).unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
            image_names.append(image_name)
            predictions.append(prediction)
    ans = pd.DataFrame({'index': image_names, 'label': predictions})
    return ans


pre_final = predict(yolo1)


pre_final


for i in range(len(pre_final)):
    if pre_final.iloc[i,1]==0:
        pre_final.iloc[i,1]='bryuki'
    else:
        pre_final.iloc[i,1]='bluzy'


pre_final


pre_final.to_csv('submission.csv',index=False)

