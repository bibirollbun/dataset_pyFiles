import zipfile
import os

# Path ke file ZIP
zip_path = '/kaggle/input/dogs-vs-cats/train.zip'
extract_to = '/kaggle/working'

# Mengekstrak file ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Selesai ekstraksi.")


import shutil
from tqdm import tqdm

src_dir = os.path.join(extract_to, 'train')
dst_dir = os.path.join(extract_to, 'train_cleaned')

os.makedirs(dst_dir, exist_ok=True)

# Pindahkan semua file .jpg ke dst_dir
for filename in tqdm(os.listdir(src_dir)):
    if filename.endswith('.jpg'):
        shutil.move(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))

print("Selesai merapikan file.")



# 2. Pindahkan semua .jpg ke folder 'train_cleaned'
src_dir = os.path.join(extract_to, 'train')  # hasil ekstrak
dst_dir = os.path.join(extract_to, 'train_cleaned')  # tempat rapi
os.makedirs(dst_dir, exist_ok=True)

for filename in tqdm(os.listdir(src_dir)):
    if filename.endswith('.jpg'):
        shutil.move(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))

print("âœ… Selesai memindahkan file gambar")


# 3. Cek isi folder
all_files = os.listdir(dst_dir)
cat_files = [f for f in all_files if f.startswith('cat')]
dog_files = [f for f in all_files if f.startswith('dog')]

print(f"\nğŸ“¦ Total gambar: {len(all_files)}")
print(f"ğŸ�± Kucing: {len(cat_files)}")
print(f"ğŸ�¶ Anjing: {len(dog_files)}")


import matplotlib.pyplot as plt
from PIL import Image
import random

# Set path direktori yang benar
train_dir = '/kaggle/working/train_cleaned'

# Mendapatkan daftar file
cat_files = [f for f in os.listdir(train_dir) if f.startswith('cat')]
dog_files = [f for f in os.listdir(train_dir) if f.startswith('dog')]

# Memilih 5 gambar secara acak dari masing-masing kelas
sample_cats = random.sample(cat_files, 5)
sample_dogs = random.sample(dog_files, 5)

# Menampilkan gambar kucing
plt.figure(figsize=(15, 3))
for i, file in enumerate(sample_cats):
    img = Image.open(os.path.join(train_dir, file))
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title("Kucing")
    plt.axis('off')
plt.suptitle("Gambar Sampel Kucing", fontsize=16)
plt.show()

# Menampilkan gambar anjing
plt.figure(figsize=(15, 3))
for i, file in enumerate(sample_dogs):
    img = Image.open(os.path.join(train_dir, file))
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title("Anjing")
    plt.axis('off')
plt.suptitle("Gambar Sampel Anjing", fontsize=16)
plt.show()



from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image



IMG_SIZE = 224


#Buat Transformasi Preprocessing
resnet_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])


#Dataset Kustom
from torch.utils.data import random_split

class CatDogDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label = 0 if img_name.startswith('cat') else 1

        if self.transform:
            image = self.transform(image)

        return image, label

#Split Train/Validation
full_dataset = CatDogDataset(train_dir, transform=resnet_transforms)

# Split: 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])



#DataLoader
dataset = CatDogDataset(train_dir, transform=resnet_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


print(f"Sampel Training: {len(train_dataset)}")
print(f"Sampel Validation: {len(val_dataset)}")


from collections import Counter

# Mengambil semua label dari dataset pelatihan
train_labels = [label for _, label in train_dataset]
label_counts = Counter(train_labels)

# Menampilkan distribusi label
print("Distribusi label dalam dataset pelatihan:")
print(f"Kucing (label 0): {label_counts[0]}")
print(f"Anjing (label 1): {label_counts[1]}")


import matplotlib.pyplot as plt

# Fungsi untuk menampilkan gambar hasil transformasi dari dataset
def show_transformed_images(dataset, num_images=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        idx = random.randint(0, len(dataset) - 1)
        img_tensor, label = dataset[idx]
        
        # Mengubah tensor menjadi gambar numpy untuk ditampilkan
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Melakukan unnormalisasi
        img = img.clip(0, 1)
        
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title("Anjing" if label == 1 else "Kucing")
        plt.axis('off')
    plt.suptitle("Gambar Sampel Hasil Transformasi dari Dataset Pelatihan", fontsize=16)
    plt.show()

# Menampilkan gambar
show_transformed_images(train_dataset)


import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

# Memuat model VGG16 dengan bobot pretrained
model = models.vgg16(pretrained=True)

# Membekukan semua layer fitur VGG16
for param in model.features.parameters():
    param.requires_grad = False

# Memodifikasi classifier untuk output biner
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)

# Mengatur device (GPU atau CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Loss, optimizer, scheduler
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

# Early stopping
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 2

# Menyimpan history
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

EPOCHS = 7

# Loop pelatihan
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validasi
    model.eval()
    val_loss_total, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss_total / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_vgg16_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Menyimpan model terbaik yang sudah dilatih
model.load_state_dict(torch.load('best_vgg16_model.pt'))  # Memuat model terbaik
model.eval()  # Mode evaluasi

all_labels = []
all_preds = []

# Evaluasi pada data validasi
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5  # Menentukan prediksi biner (0 atau 1)

        all_labels.extend(labels.cpu().numpy())  # Menyimpan semua label asli
        all_preds.extend(preds.cpu().numpy())  # Menyimpan semua prediksi

# Mengubah list menjadi array untuk evaluasi
all_labels = [int(x[0]) for x in all_labels]
all_preds = [int(x[0]) for x in all_preds]

# Menghitung metrik evaluasi
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

# Menampilkan hasil evaluasi
print(f"\nEvaluasi Model VGG16 pada Validation Set:")
print(f"Akurasi     : {acc:.4f}")
print(f"Presisi     : {prec:.4f}")
print(f"Recall      : {rec:.4f}")
print(f"F1 Score    : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

# Menghitung confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Menampilkan heatmap confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Kucing', 'Anjing'],
            yticklabels=['Kucing', 'Anjing'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - VGG16')
plt.show()





import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm  

# Memuat model ResNet-50 dengan bobot yang telah dilatih sebelumnya (pretrained)
model = models.resnet50(pretrained=True)

# Membekukan semua layer dari model ResNet-50, kecuali layer klasifikasi terakhir
for param in model.parameters():
    param.requires_grad = False  

# Memodifikasi layer klasifikasi akhir untuk klasifikasi biner (Kucing vs Anjing)
model.fc = nn.Linear(model.fc.in_features, 1)

# Memindahkan model ke GPU jika tersedia, jika tidak maka gunakan CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Fungsi loss, optimizer, dan scheduler untuk learning rate
criterion = BCEWithLogitsLoss()  # Digunakan untuk klasifikasi biner (karena output belum melalui sigmoid)
optimizer = Adam(model.parameters(), lr=1e-4)  # Optimizer Adam dengan learning rate 0.0001
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
# Scheduler akan menurunkan learning rate jika loss validasi tidak membaik selama 2 epoch (patience)

# Pengaturan untuk Early Stopping
best_val_loss = float('inf')           # Menyimpan nilai loss validasi terbaik
patience_counter = 0                   # Menghitung jumlah epoch tanpa perbaikan
early_stop_patience = 2               # Hentikan pelatihan jika tidak ada perbaikan selama 2 epoch berturut-turut

# Menyimpan riwayat loss untuk pelatihan dan validasi
train_losses, val_losses = [], []

# Inisialisasi variabel untuk menyimpan metrik pelatihan dan validasi
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Jumlah epoch untuk pelatihan
EPOCHS = 7
best_val_loss = float('inf')  # Menyimpan loss terbaik selama validasi
patience_counter = 0  # Untuk early stopping
early_stop_patience = 2  

# Loop pelatihan
for epoch in range(EPOCHS):
    model.train()  # Mengatur model ke mode pelatihan
    running_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(train_loader):   # Iterasi batch dalam data pelatihan
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
         # .float(): karena kita pakai BCEWithLogitsLoss yang butuh tipe float
        # .unsqueeze(1): memastikan label berbentuk [batch_size, 1] untuk dicocokkan dengan output model
        
        optimizer.zero_grad()  # Mengatur ulang gradien ke nol
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Hitung loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Perbarui bobot

        running_loss += loss.item()  # Menjumlahkan total loss untuk 1 epoch
        preds = torch.sigmoid(outputs) > 0.5  # Konversi output ke prediksi (threshold 0.5)
        correct += (preds == labels).sum().item()  # Hitung prediksi yang benar
        total += labels.size(0)  # Jumlah total sampel

    # Menghitung rata-rata loss dan akurasi pelatihan
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Loop validasi
    model.eval()  # Mengatur model ke mode evaluasi
    val_loss_total, correct, total = 0, 0, 0

    with torch.no_grad():  # Menonaktifkan perhitungan gradien
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Hitung loss

            val_loss_total += loss.item()  # Menambahkan total loss validasi
            preds = torch.sigmoid(outputs) > 0.5  # Konversi ke prediksi biner
            correct += (preds == labels).sum().item()  # Hitung prediksi yang benar
            total += labels.size(0)  # Jumlah total sampel

    # Menghitung rata-rata loss dan akurasi validasi
    val_loss = val_loss_total / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Menampilkan progres untuk setiap epoch
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Meng-update scheduler
    scheduler.step(val_loss)

    # Cek untuk early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_resnet50_model.pt')   # Menyimpan model terbaik sejauh ini
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Menyiapkan model dan data untuk evaluasi
model.load_state_dict(torch.load('best_resnet50_model.pt'))  # Memuat model terbaik
model.eval()  # Mode evaluasi

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Konversi list ke array untuk evaluasi
all_labels = [int(x[0]) for x in all_labels]
all_preds = [int(x[0]) for x in all_preds]

# Menghitung metrik evaluasi
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

# Tampilkan metrik
print(f"\nEvaluasi Model ResNet50 pada Validation Set:")
print(f"Akurasi     : {acc:.4f}")
print(f"Presisi     : {prec:.4f}")
print(f"Recall      : {rec:.4f}")
print(f"F1 Score    : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

# Menghitung confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Menampilkan heatmap confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Kucing', 'Anjing'],
            yticklabels=['Kucing', 'Anjing'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - ResNet50')
plt.show()






import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

# Memuat model AlexNet dengan bobot pretrained
model = models.alexnet(pretrained=True)

# Membekukan semua parameter fitur AlexNet
for param in model.features.parameters():
    param.requires_grad = False

# Memodifikasi classifier terakhir untuk output 1 neuron (klasifikasi biner)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)

# Memindahkan model ke GPU jika tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Loss function, optimizer, dan scheduler
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

# Inisialisasi variabel untuk early stopping
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 2

# History loss dan akurasi
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Jumlah epoch
EPOCHS = 7

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validasi
    model.eval()
    val_loss_total, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss_total / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_alexnet_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Menyimpan model terbaik yang sudah dilatih
model.load_state_dict(torch.load('best_alexnet_model.pt'))  # Memuat model terbaik
model.eval()  # Mode evaluasi

all_labels = []
all_preds = []

# Evaluasi pada data validasi
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5  # Menentukan prediksi biner (0 atau 1)

        all_labels.extend(labels.cpu().numpy())  # Menyimpan semua label asli
        all_preds.extend(preds.cpu().numpy())  # Menyimpan semua prediksi

# Mengubah list menjadi array untuk evaluasi
all_labels = [int(x[0]) for x in all_labels]
all_preds = [int(x[0]) for x in all_preds]

# Menghitung metrik evaluasi
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

# Menampilkan hasil evaluasi
print(f"\nEvaluasi Model AlexNet pada Validation Set:")
print(f"Akurasi     : {acc:.4f}")
print(f"Presisi     : {prec:.4f}")
print(f"Recall      : {rec:.4f}")
print(f"F1 Score    : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

# Menghitung confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Menampilkan heatmap confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Kucing', 'Anjing'],
            yticklabels=['Kucing', 'Anjing'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - AlexNet')
plt.show()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

# Mendefinisikan arsitektur LeNet untuk klasifikasi biner
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # Conv layer 1
        self.pool = nn.MaxPool2d(2, 2)  # Maxpooling
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Conv layer 2
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Fully connected layer 1 (adjusted for image size)
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer 2
        self.fc3 = nn.Linear(84, 1)  # Output layer for binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply conv1 + ReLU + maxpooling
        x = self.pool(torch.relu(self.conv2(x)))  # Apply conv2 + ReLU + maxpooling
        x = x.view(-1, 16 * 53 * 53)  # Flattening output for fully connected layers
        x = torch.relu(self.fc1(x))  # Apply fully connected layer 1 + ReLU
        x = torch.relu(self.fc2(x))  # Apply fully connected layer 2 + ReLU
        x = self.fc3(x)  # Output layer
        return x

# Instansiasi model LeNet
model = LeNet()

# Memindahkan model ke GPU jika tersedia, jika tidak maka gunakan CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Loss function dan optimizer
criterion = BCEWithLogitsLoss()  # Digunakan untuk klasifikasi biner
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Optimizer Adam dengan learning rate 0.0001
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

# Early stopping
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 2

# Menyimpan riwayat loss
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

EPOCHS = 7

# Loop pelatihan
for epoch in range(EPOCHS):
    model.train()  # Set model ke mode pelatihan
    running_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()  # Mengatur ulang gradien
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Hitung loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update bobot

        running_loss += loss.item()  # Menjumlahkan total loss
        preds = torch.sigmoid(outputs) > 0.5  # Mengubah output menjadi prediksi biner
        correct += (preds == labels).sum().item()  # Hitung jumlah prediksi yang benar
        total += labels.size(0)  # Hitung total jumlah sampel

    # Menghitung rata-rata loss dan akurasi untuk pelatihan
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validasi
    model.eval()  # Set model ke mode evaluasi
    val_loss_total, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Hitung loss

            val_loss_total += loss.item()  # Tambahkan loss validasi
            preds = torch.sigmoid(outputs) > 0.5  # Mengubah output menjadi prediksi biner
            correct += (preds == labels).sum().item()  # Hitung jumlah prediksi yang benar
            total += labels.size(0)  # Hitung total jumlah sampel

    # Menghitung rata-rata loss dan akurasi untuk validasi
    val_loss = val_loss_total / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Menampilkan progres untuk setiap epoch
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Meng-update scheduler
    scheduler.step(val_loss)

    # Cek untuk early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_lenet_model.pt')  # Menyimpan model terbaik
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Menyimpan model terbaik yang sudah dilatih
model.load_state_dict(torch.load('best_lenet_model.pt'))  # Memuat model terbaik
model.eval()  # Mode evaluasi

all_labels = []
all_preds = []

# Evaluasi pada data validasi
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5  # Menentukan prediksi biner (0 atau 1)

        all_labels.extend(labels.cpu().numpy())  # Menyimpan semua label asli
        all_preds.extend(preds.cpu().numpy())  # Menyimpan semua prediksi

# Mengubah list menjadi array untuk evaluasi
all_labels = [int(x[0]) for x in all_labels]
all_preds = [int(x[0]) for x in all_preds]

# Menghitung metrik evaluasi
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

# Menampilkan hasil evaluasi
print(f"\nEvaluasi Model LeNet pada Validation Set:")
print(f"Akurasi     : {acc:.4f}")
print(f"Presisi     : {prec:.4f}")
print(f"Recall      : {rec:.4f}")
print(f"F1 Score    : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

# Menghitung confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Menampilkan heatmap confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='seismic',
            xticklabels=['Kucing', 'Anjing'],
            yticklabels=['Kucing', 'Anjing'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - LeNet')
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['VGG16', 'ResNet50', 'AlexNet', 'LeNet']
akurasi = [0.9874, 0.9854, 0.9642, 0.7604]
presisi = [0.9907, 0.9844, 0.9711, 0.7258]
recall = [0.9839, 0.9864, 0.9566, 0.8342]
f1_score = [0.9873, 0.9854, 0.9638, 0.7762]

# Konversi ke persen
akurasi = [x * 100 for x in akurasi]
presisi = [x * 100 for x in presisi]
recall = [x * 100 for x in recall]
f1_score = [x * 100 for x in f1_score]

# Posisi bar
x = np.arange(len(models))
width = 0.2

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - 1.5*width, akurasi, width, label='Akurasi')
bars2 = ax.bar(x - 0.5*width, presisi, width, label='Presisi')
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall')
bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score')

# Menambahkan angka pada tiap bar
def add_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', 
                ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Label dan tampilan
ax.set_ylabel('Nilai (%)')
ax.set_title('Perbandingan Internal Performa Model CNN')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

# Data model dan akurasi
models = [
    "32-layer CNN (Ikan)",
    "DenseNet201 (Pneumonia)",
    "AlexNet (Brain Tumor)",
    "ResNet50 (Daun Kentang)",
    "VGG-16 (Kerbau)",
    "ResNet50 (Kerbau)"
]

accuracy = [96.63, 98.00, 96.10, 97.00, 95.00, 80.00]

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracy, color='cornflowerblue')
plt.ylabel("Akurasi (%)")
plt.title("Perbandingan Eksternal Akurasi Model CNN")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 100)

# Tambahkan label di dalam bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval - 5, f'{yval:.2f}%', 
             ha='center', va='center', color='white', fontsize=9, fontstyle='italic')

plt.tight_layout()
plt.show()


